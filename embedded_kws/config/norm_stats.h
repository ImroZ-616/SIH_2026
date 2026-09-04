// SIH 26172 - ML/KWS Normalization & Input Quantization Header
// Generated automatically during Phase 12 Model Export
// Target: R3 Embedded Firmware (ESP32-S3 / ESP32)

#ifndef NORM_STATS_H_
#define NORM_STATS_H_

#include <stdint.h>

#define KWS_NUM_MFCC 13
#define KWS_NUM_FRAMES 98

// INT8 Input Tensor Quantization Parameters
#define KWS_INPUT_SCALE 0.0438336767f
#define KWS_INPUT_ZERO_POINT -4

// Raw Training Set Normalization Vectors (mean and std per MFCC coefficient)
static const float KWS_MFCC_MEAN[KWS_NUM_MFCC] = { -8.83995342f, 2.98094368f, -0.49285635f, 0.27299267f, -0.75096923f, -0.02419784f, -0.65593499f, -0.07901166f, -0.52964151f, -0.06854012f, -0.60945559f, -0.11400218f, -0.42365387f };
static const float KWS_MFCC_STD[KWS_NUM_MFCC]  = { 13.58398056f, 4.20223570f, 2.38524532f, 1.77322018f, 1.58381844f, 1.33143055f, 1.16003978f, 0.96019596f, 0.92623347f, 0.78868353f, 0.83599126f, 0.68238991f, 0.69473630f };

// Precomputed Fused Normalization + Quantization Linear Coefficients:
// q_in[t, i] = clamp(round(X_raw[t, i] * KWS_NORM_A[i] + KWS_NORM_B[i]), -128, 127)
static const float KWS_NORM_A[KWS_NUM_MFCC] = { 1.67944217f, 5.42889833f, 9.56442928f, 12.86558247f, 14.40411949f, 17.13458443f, 19.66614342f, 23.75922394f, 24.63040924f, 28.92606354f, 27.28917313f, 33.43177795f, 32.83765411f };
static const float KWS_NORM_B[KWS_NUM_MFCC] = { 10.84619045f, -20.18323898f, 0.71388960f, -7.51220989f, 6.81704998f, -3.58538008f, 8.89971256f, -2.12274408f, 9.04528713f, -2.01740408f, 12.63153839f, -0.18870425f, 9.91179943f };

#endif  // NORM_STATS_H_
