// SIH 26172 - EdgeWake C++ MFCC Feature Extraction Implementation
// Ground Truth: audio/features.py, audio/mfcc.py, audio/mel.py

#include "mfcc.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Static trigonometric lookup tables for 480-point real FFT
static float s_cos_table[KWS_FFT_BINS][KWS_FRAME_LENGTH];
static float s_sin_table[KWS_FFT_BINS][KWS_FRAME_LENGTH];
static int s_dsp_initialized = 0;

void kws_dsp_init(void) {
    if (s_dsp_initialized) return;

    for (int k = 0; k < KWS_FFT_BINS; ++k) {
        for (int n = 0; n < KWS_FRAME_LENGTH; ++n) {
            double angle = 2.0 * M_PI * (double)k * (double)n / (double)KWS_FFT_SIZE;
            s_cos_table[k][n] = (float)cos(angle);
            s_sin_table[k][n] = (float)(-sin(angle)); // Negative for e^(-j*angle)
        }
    }
    s_dsp_initialized = 1;
}

void kws_compute_power_spectrum(const float* windowed_frame_480, float* power_spectrum_241) {
    if (!s_dsp_initialized) kws_dsp_init();

    for (int k = 0; k < KWS_FFT_BINS; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        const float* cos_row = s_cos_table[k];
        const float* sin_row = s_sin_table[k];

        for (int n = 0; n < KWS_FRAME_LENGTH; ++n) {
            float x = windowed_frame_480[n];
            re += x * cos_row[n];
            im += x * sin_row[n];
        }
        power_spectrum_241[k] = re * re + im * im;
    }
}

void kws_compute_mel_energies(const float* power_spectrum_241, float* mel_energies_40) {
    for (int m = 0; m < KWS_NUM_MELS; ++m) {
        float sum = 0.0f;
        const float* fb_row = KWS_MEL_FILTERBANK[m];
        for (int k = 0; k < KWS_FFT_BINS; ++k) {
            sum += fb_row[k] * power_spectrum_241[k];
        }
        mel_energies_40[m] = sum;
    }
}

void kws_compute_log_mel(const float* mel_energies_40, float* log_mel_40) {
    // Critical ground truth formula: log10f(fmaxf(mel_energy, 1e-10f))
    for (int m = 0; m < KWS_NUM_MELS; ++m) {
        log_mel_40[m] = log10f(fmaxf(mel_energies_40[m], KWS_LOG_EPSILON));
    }
}

void kws_compute_dct(const float* log_mel_40, float* mfcc_13) {
    for (int n = 0; n < KWS_NUM_MFCC; ++n) {
        float sum = 0.0f;
        const float* dct_row = KWS_DCT_BASIS[n];
        for (int m = 0; m < KWS_NUM_MELS; ++m) {
            sum += dct_row[m] * log_mel_40[m];
        }
        mfcc_13[n] = sum;
    }
}

void kws_extract_frame_mfcc(const float* audio_frame_480, float* mfcc_out_13) {
    float windowed_frame[KWS_FRAME_LENGTH];
    float power_spectrum[KWS_FFT_BINS];
    float mel_energies[KWS_NUM_MELS];
    float log_mel[KWS_NUM_MELS];

    // 1. Hann Windowing
    for (int n = 0; n < KWS_FRAME_LENGTH; ++n) {
        windowed_frame[n] = audio_frame_480[n] * KWS_HANN_WINDOW[n];
    }

    // 2. Power Spectrum
    kws_compute_power_spectrum(windowed_frame, power_spectrum);

    // 3. Mel Energies
    kws_compute_mel_energies(power_spectrum, mel_energies);

    // 4. Log Compression
    kws_compute_log_mel(mel_energies, log_mel);

    // 5. DCT-II (13 MFCCs)
    kws_compute_dct(log_mel, mfcc_out_13);
}

void kws_extract_all_mfcc(const float* audio_16000, float mfcc_matrix[KWS_NUM_FRAMES][KWS_NUM_MFCC]) {
    for (int t = 0; t < KWS_NUM_FRAMES; ++t) {
        int start = t * KWS_HOP_LENGTH;
        kws_extract_frame_mfcc(&audio_16000[start], mfcc_matrix[t]);
    }
}

void kws_quantize_fused(const float mfcc_matrix[KWS_NUM_FRAMES][KWS_NUM_MFCC], int8_t int8_tensor[KWS_NUM_FRAMES][KWS_NUM_MFCC]) {
    for (int t = 0; t < KWS_NUM_FRAMES; ++t) {
        for (int c = 0; c < KWS_NUM_MFCC; ++c) {
            float val = nearbyintf(mfcc_matrix[t][c] * KWS_NORM_A[c] + KWS_NORM_B[c]);
            if (val > 127.0f) val = 127.0f;
            if (val < -128.0f) val = -128.0f;
            int8_tensor[t][c] = (int8_t)val;
        }
    }
}
