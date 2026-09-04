// SIH 26172 - EdgeWake C++ MFCC Feature Extraction Header
// Target: Standalone C++ DSP Engine (ESP32-S3 / Native C++)

#ifndef MFCC_H_
#define MFCC_H_

#include <stdint.h>
#include <stddef.h>

#include "../config/kws_config.h"
#include "../config/norm_stats.h"
#include "dsp_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes trigonometric lookup tables (angles for 480-point real FFT)
void kws_dsp_init(void);

// Computes 480-point Real-to-Complex FFT power spectrum (241 positive frequency bins)
void kws_compute_power_spectrum(const float* windowed_frame_480, float* power_spectrum_241);

// Computes 40 Mel band filterbank energies from power spectrum
void kws_compute_mel_energies(const float* power_spectrum_241, float* mel_energies_40);

// Applies log compression with strict epsilon: log10f(fmaxf(E, 1e-10f))
void kws_compute_log_mel(const float* mel_energies_40, float* log_mel_40);

// Applies orthonormal DCT-II to obtain 13 MFCC coefficients
void kws_compute_dct(const float* log_mel_40, float* mfcc_13);

// Extracts 13 MFCC coefficients for a single 480-sample frame
void kws_extract_frame_mfcc(const float* audio_frame_480, float* mfcc_out_13);

// Extracts full 98x13 MFCC matrix from 16,000 float audio samples
void kws_extract_all_mfcc(const float* audio_16000, float mfcc_matrix[KWS_NUM_FRAMES][KWS_NUM_MFCC]);

// Applies fused normalization + INT8 quantization: q = clamp(round(X * A + B), -128, 127)
void kws_quantize_fused(const float mfcc_matrix[KWS_NUM_FRAMES][KWS_NUM_MFCC], int8_t int8_tensor[KWS_NUM_FRAMES][KWS_NUM_MFCC]);

#ifdef __cplusplus
}
#endif

#endif  // MFCC_H_
