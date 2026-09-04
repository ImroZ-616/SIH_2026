// SIH 26172 - Genuine TensorFlow Lite Micro (TFLM) Inference Engine Implementation
#include "tflm_engine.h"
#include "../model/kws_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>
#include <math.h>

namespace {
// Tensor Arena for TFLM
constexpr size_t kTensorArenaSize = 64 * 1024; // 64 KB
alignas(16) static uint8_t s_tensor_arena[kTensorArenaSize];

static tflite::MicroErrorReporter s_error_reporter;
static const tflite::Model* s_model = nullptr;
static tflite::MicroMutableOpResolver<5> s_resolver;
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input_tensor = nullptr;
static TfLiteTensor* s_output_tensor = nullptr;
static size_t s_arena_used_bytes = 0;
static bool s_is_initialized = false;
} // namespace

extern "C" {

int tflm_engine_init(size_t* out_arena_size, size_t* out_arena_used) {
    if (s_is_initialized) {
        if (out_arena_size) *out_arena_size = kTensorArenaSize;
        if (out_arena_used) *out_arena_used = s_arena_used_bytes;
        return 0;
    }

    // 1. Load model FlatBuffer
    s_model = tflite::GetModel(g_kws_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(&s_error_reporter,
            "Model provided is schema version %d not equal to supported version %d.",
            s_model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // 2. Register exact 5 operators required by Compact-KWS-CNN
    s_resolver.AddConv2D();
    s_resolver.AddMaxPool2D();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();

    // 3. Instantiate MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        s_model, s_resolver, s_tensor_arena, kTensorArenaSize, &s_error_reporter);
    s_interpreter = &static_interpreter;

    // 4. Allocate Tensors
    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&s_error_reporter, "AllocateTensors() failed with status %d", alloc_status);
        return -2;
    }

    // 5. Query input and output tensors
    s_input_tensor = s_interpreter->input(0);
    s_output_tensor = s_interpreter->output(0);

    if (s_input_tensor == nullptr || s_output_tensor == nullptr) {
        TF_LITE_REPORT_ERROR(&s_error_reporter, "Failed to get input/output tensors");
        return -3;
    }

    // Record measured arena used bytes
    s_arena_used_bytes = s_interpreter->arena_used_bytes();
    s_is_initialized = true;

    if (out_arena_size) *out_arena_size = kTensorArenaSize;
    if (out_arena_used) *out_arena_used = s_arena_used_bytes;

    return 0;
}

int tflm_engine_invoke(const int8_t input_features[KWS_NUM_FRAMES][KWS_NUM_MFCC],
                       int8_t output_tensor[KWS_NUM_CLASSES],
                       float output_probabilities[KWS_NUM_CLASSES],
                       int* predicted_class) {
    if (!s_is_initialized || !s_interpreter) return -1;

    // Copy input features (98x13) to input tensor
    memcpy(s_input_tensor->data.int8, input_features, KWS_NUM_FRAMES * KWS_NUM_MFCC * sizeof(int8_t));

    // Execute genuine TFLM Invoke()
    TfLiteStatus invoke_status = s_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&s_error_reporter, "Invoke() failed with status %d", invoke_status);
        return -2;
    }

    // Copy INT8 output tensor (3,)
    for (int i = 0; i < KWS_NUM_CLASSES; ++i) {
        output_tensor[i] = s_output_tensor->data.int8[i];
    }

    // Dequantize output probabilities using output tensor quantization params
    float scale = s_output_tensor->params.scale;
    int32_t zero_point = s_output_tensor->params.zero_point;
    
    // In our model: scale = 0.00390625 (1/256), zero_point = -128
    float sum_p = 0.0f;
    for (int i = 0; i < KWS_NUM_CLASSES; ++i) {
        float p = (static_cast<float>(output_tensor[i]) - static_cast<float>(zero_point)) * scale;
        if (p < 0.0f) p = 0.0f;
        output_probabilities[i] = p;
        sum_p += p;
    }
    if (sum_p > 0.0f) {
        for (int i = 0; i < KWS_NUM_CLASSES; ++i) {
            output_probabilities[i] /= sum_p;
        }
    }

    int best_cls = 0;
    float best_p = -1.0f;
    for (int i = 0; i < KWS_NUM_CLASSES; ++i) {
        if (output_probabilities[i] > best_p) {
            best_p = output_probabilities[i];
            best_cls = i;
        }
    }
    if (predicted_class) *predicted_class = best_cls;

    return 0;
}

size_t tflm_engine_get_arena_size(void) {
    return kTensorArenaSize;
}

size_t tflm_engine_get_arena_used(void) {
    return s_arena_used_bytes;
}

} // extern "C"
