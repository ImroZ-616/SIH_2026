// SIH 26172 - Genuine TensorFlow Lite Micro (TFLM) Inference Engine Header
#ifndef TFLM_ENGINE_H_
#define TFLM_ENGINE_H_

#include <stdint.h>
#include <stddef.h>
#include "../config/kws_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the genuine TFLM MicroInterpreter, registers 5 operators, allocates tensor arena
int tflm_engine_init(size_t* out_arena_size, size_t* out_arena_used);

// Executes TFLM forward pass on INT8 input tensor (98x13)
int tflm_engine_invoke(const int8_t input_features[KWS_NUM_FRAMES][KWS_NUM_MFCC],
                       int8_t output_tensor[KWS_NUM_CLASSES],
                       float output_probabilities[KWS_NUM_CLASSES],
                       int* predicted_class);

// Returns configured arena size in bytes
size_t tflm_engine_get_arena_size(void);

// Returns measured arena bytes used after AllocateTensors()
size_t tflm_engine_get_arena_used(void);

#ifdef __cplusplus
}
#endif

#endif // TFLM_ENGINE_H_
