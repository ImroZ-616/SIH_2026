// SIH 26172 - EdgeWake KWS Configuration Header
// Target: Embedded KWS (ESP32-S3 / Native C++)

#ifndef KWS_CONFIG_H_
#define KWS_CONFIG_H_

#include <stdint.h>

// Audio Specifications
#define KWS_SAMPLE_RATE 16000
#define KWS_AUDIO_DURATION_SEC 1.0f
#define KWS_TOTAL_SAMPLES 16000

// Feature Extraction Specifications
#define KWS_FRAME_LENGTH 480    // 30 ms @ 16 kHz
#define KWS_HOP_LENGTH 160      // 10 ms @ 16 kHz
#define KWS_FFT_SIZE 480
#define KWS_FFT_BINS 241        // 480 / 2 + 1
#define KWS_NUM_MELS 40
#define KWS_NUM_MFCC 13
#define KWS_NUM_FRAMES 98

// Compression Epsilon (Strict ground-truth match: log10f(fmaxf(E, 1e-10f)))
#define KWS_LOG_EPSILON 1e-10f

// INT8 Model Parameters
#define KWS_INPUT_SCALE 0.0438336767f
#define KWS_INPUT_ZERO_POINT -4
#define KWS_OUTPUT_SCALE 0.0039062500f
#define KWS_OUTPUT_ZERO_POINT -128

#define KWS_NUM_CLASSES 3
#define KWS_CLASS_SILENCE 0
#define KWS_CLASS_UNKNOWN 1
#define KWS_CLASS_ASTRA 2

#endif  // KWS_CONFIG_H_
