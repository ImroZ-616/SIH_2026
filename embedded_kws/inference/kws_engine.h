// SIH 26172 - EdgeWake KWS TFLM Inference Engine Header
// Target: Native C++ / TFLite Micro runtime harness

#ifndef KWS_ENGINE_H_
#define KWS_ENGINE_H_

#include <stdint.h>
#include <stddef.h>

#include "../config/kws_config.h"
#include "../model/kws_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inference Engine State Structure
typedef struct {
    int is_initialized;
    size_t arena_size_bytes;
    size_t arena_used_bytes;
    int8_t input_tensor[KWS_NUM_FRAMES][KWS_NUM_MFCC]; // [98][13]
    int8_t output_tensor[KWS_NUM_CLASSES];             // [3]
    float output_probabilities[KWS_NUM_CLASSES];       // [3]
    int predicted_class;
} KwsInferenceEngine;

// Initializes the KWS TFLM Inference Engine (allocates arena, resolves ops, verifies model)
int kws_engine_init(KwsInferenceEngine* engine, size_t arena_size);

// Sets the INT8 input tensor (98x13)
void kws_engine_set_input(KwsInferenceEngine* engine, const int8_t input_features[KWS_NUM_FRAMES][KWS_NUM_MFCC]);

// Executes neural network forward pass (Invoke)
int kws_engine_invoke(KwsInferenceEngine* engine);

// Gets the predicted class index (0=Silence, 1=Unknown, 2=ASTRA)
int kws_engine_get_prediction(const KwsInferenceEngine* engine);

// Gets class probability (0.0 to 1.0)
float kws_engine_get_class_probability(const KwsInferenceEngine* engine, int class_idx);

#ifdef __cplusplus
}
#endif

#endif  // KWS_ENGINE_H_
