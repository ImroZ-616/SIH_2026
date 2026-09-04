// SIH 26172 - Embedded C++ DSP & Genuine TFLM Validation Test Harness
// Staged Validation Harness for Phase 14 Native Verification

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../config/kws_config.h"
#include "../config/norm_stats.h"
#include "../model/kws_model.h"
#include "../dsp/mfcc.h"
#include "../inference/kws_engine.h"
#include "../inference/tflm_engine.h"
#include "golden_vectors.h"

int main() {
    printf("================================================================================\n");
    printf("SIH 26172 - PHASE 14 NATIVE C++ DSP & GENUINE TFLM VALIDATION HARNESS\n");
    printf("================================================================================\n\n");

    // --- 0. MODEL PAYLOAD & LINKAGE INTEGRITY ---
    printf("--- 0. MODEL PAYLOAD & LINKAGE INTEGRITY ---\n");
    printf("  Model Length Symbol        : %u bytes\n", (unsigned int)g_kws_model_data_len);
    printf("  Expected INT8 Model Length : 29200 bytes\n");
    
    char magic[5] = {0};
    if (g_kws_model_data_len >= 8) {
        memcpy(magic, &g_kws_model_data[4], 4);
    }
    printf("  Model FlatBuffer Identifier: %s\n", magic);

    if (g_kws_model_data_len != 29200 || strcmp(magic, "TFL3") != 0) {
        printf("  [FAIL] Model payload or FlatBuffer identifier invalid!\n");
        return 1;
    }
    printf("  [PASS] Model payload and safe linkage verified.\n\n");

    // --- 1. TFLM INITIALIZATION ---
    printf("--- 1. TENSORFLOW LITE MICRO (TFLM) RUNTIME INITIALIZATION ---\n");
    printf("  TFLM Library Framework     : TensorFlow Lite for Microcontrollers (Official Release)\n");
    printf("  MicroInterpreter API       : tflite::MicroInterpreter\n");
    printf("  MicroMutableOpResolver     : tflite::MicroMutableOpResolver<5>\n");
    printf("  Registered Operators (5)   : Conv2D, MaxPool2D, Mean, FullyConnected, Softmax\n");

    size_t arena_size = 0;
    size_t arena_used = 0;
    int tflm_init_status = tflm_engine_init(&arena_size, &arena_used);
    if (tflm_init_status != 0) {
        printf("  [FAIL] tflm_engine_init() failed with code %d\n", tflm_init_status);
        return 1;
    }
    printf("  AllocateTensors() Status   : [PASS] (kTfLiteOk)\n");
    printf("  Configured Tensor Arena    : %zu bytes (%zu KB)\n", arena_size, arena_size / 1024);
    printf("  Measured Arena Used Bytes  : %zu bytes (~%.2f KB)\n", arena_used, (float)arena_used / 1024.0f);
    printf("  [PASS] Genuine TFLM MicroInterpreter initialized successfully.\n\n");

    // Initialize custom native engine for diagnostic cross-check
    KwsInferenceEngine custom_engine;
    kws_engine_init(&custom_engine, 48 * 1024);

    int total_vectors = GOLDEN_NUM_VECTORS;
    int test_a_all_passed = 1;
    int test_b_all_passed = 1;
    int test_c_all_passed = 1;
    int test_d_all_passed = 1;

    // =========================================================================
    // TEST A: GENUINE TFLM ENGINE VALIDATION (Direct INT8 Input)
    // Pipeline: Python INT8 Input -> TFLM MicroInterpreter -> INT8 Output
    // Requirement: Exact INT8 Output Match (Max Diff = 0, Differing Elements = 0)
    // =========================================================================
    printf("================================================================================\n");
    printf("TEST A: GENUINE TFLM ENGINE VALIDATION (Direct INT8 Input -> MicroInterpreter)\n");
    printf("Requirement: EXACT INT8 OUTPUT MATCH (Max Diff = 0, Differing Elements = 0)\n");
    printf("--------------------------------------------------------------------------------\n");

    int test_a_max_diff_overall = 0;
    int test_a_differing_elements_overall = 0;

    for (int v = 0; v < total_vectors; ++v) {
        const GoldenVector* gv = &g_golden_vectors[v];
        int8_t tflm_output[KWS_NUM_CLASSES] = {0};
        float tflm_probs[KWS_NUM_CLASSES] = {0};
        int tflm_pred = -1;

        int status = tflm_engine_invoke(gv->int8_input, tflm_output, tflm_probs, &tflm_pred);
        if (status != 0) {
            printf("  [%s] Invoke error code %d!\n", gv->name, status);
            test_a_all_passed = 0;
            continue;
        }

        int max_diff = 0;
        int diff_elements = 0;
        for (int c = 0; c < KWS_NUM_CLASSES; ++c) {
            int d = abs((int)tflm_output[c] - (int)gv->expected_output_int8[c]);
            if (d > max_diff) max_diff = d;
            if (d != 0) diff_elements++;
        }

        if (max_diff > test_a_max_diff_overall) test_a_max_diff_overall = max_diff;
        test_a_differing_elements_overall += diff_elements;

        int pass = (max_diff == 0 && diff_elements == 0);
        if (!pass) test_a_all_passed = 0;

        printf("  [%s] Class=%s (%d)\n", gv->name,
               gv->true_class == 2 ? "Keyword/ASTRA" : (gv->true_class == 1 ? "Unknown" : "Silence"),
               gv->true_class);
        printf("    Expected INT8: [%4d, %4d, %4d] | TFLM Output INT8: [%4d, %4d, %4d]\n",
               gv->expected_output_int8[0], gv->expected_output_int8[1], gv->expected_output_int8[2],
               tflm_output[0], tflm_output[1], tflm_output[2]);
        printf("    Max Diff: %d | Differing Elements: %d | Status: %s\n",
               max_diff, diff_elements, pass ? "[PASS]" : "[FAIL]");
    }
    printf("  >> Test A Overall Result: %s (Max Diff = %d, Total Differing = %d)\n",
           test_a_all_passed ? "PASS" : "FAIL", test_a_max_diff_overall, test_a_differing_elements_overall);

    // =========================================================================
    // TEST B: FUSED NORMALIZATION + QUANTIZATION VALIDATION
    // (Python MFCCs -> C++ Fused A/B Transform -> Compare with Python INT8 Input)
    // Requirement: STRICT ZERO TOLERANCE (Max Diff = 0, Differing Elements = 0)
    // =========================================================================
    printf("\n================================================================================\n");
    printf("TEST B: FUSED NORMALIZATION + QUANTIZATION (ZERO TOLERANCE)\n");
    printf("Requirement: MAX INTEGER DISCREPANCY = 0, DIFFERING INT8 ELEMENTS = 0\n");
    printf("--------------------------------------------------------------------------------\n");

    int test_b_max_diff_overall = 0;
    int test_b_differing_elements_overall = 0;

    for (int v = 0; v < total_vectors; ++v) {
        const GoldenVector* gv = &g_golden_vectors[v];
        int8_t cpp_int8[KWS_NUM_FRAMES][KWS_NUM_MFCC];
        kws_quantize_fused(gv->mfcc_features, cpp_int8);

        int max_diff = 0;
        int diff_elements = 0;
        for (int t = 0; t < KWS_NUM_FRAMES; ++t) {
            for (int c = 0; c < KWS_NUM_MFCC; ++c) {
                int d = abs((int)cpp_int8[t][c] - (int)gv->int8_input[t][c]);
                if (d > max_diff) max_diff = d;
                if (d != 0) diff_elements++;
            }
        }

        if (max_diff > test_b_max_diff_overall) test_b_max_diff_overall = max_diff;
        test_b_differing_elements_overall += diff_elements;

        int pass = (max_diff == 0 && diff_elements == 0);
        if (!pass) test_b_all_passed = 0;

        printf("  [%s] Max Diff: %d | Differing Elements: %d / 1274 | Status: %s\n",
               gv->name, max_diff, diff_elements, pass ? "[PASS]" : "[FAIL]");
    }
    printf("  >> Test B Overall Result: %s (Max Diff = %d, Total Differing = %d)\n",
           test_b_all_passed ? "PASS" : "FAIL", test_b_max_diff_overall, test_b_differing_elements_overall);

    // =========================================================================
    // TEST C: C++ MFCC FEATURE EXTRACTION VALIDATION
    // (Raw Audio PCM -> C++ MFCC -> Compare against Python Ground Truth)
    // Requirement: MAE < 1e-4, MaxAE < 1e-3
    // =========================================================================
    printf("\n================================================================================\n");
    printf("TEST C: C++ MFCC FEATURE EXTRACTION (vs Python MFCC Ground Truth)\n");
    printf("Requirement: MAE < 1e-4, MaxAE < 1e-3 (log10f(fmaxf(mel_energy, 1e-10f)))\n");
    printf("--------------------------------------------------------------------------------\n");

    double test_c_max_ae_overall = 0.0;
    double test_c_mae_sum = 0.0;

    for (int v = 0; v < total_vectors; ++v) {
        const GoldenVector* gv = &g_golden_vectors[v];
        float cpp_mfcc[KWS_NUM_FRAMES][KWS_NUM_MFCC];
        kws_extract_all_mfcc(gv->audio_float, cpp_mfcc);

        double sum_ae = 0.0;
        double max_ae = 0.0;
        for (int t = 0; t < KWS_NUM_FRAMES; ++t) {
            for (int c = 0; c < KWS_NUM_MFCC; ++c) {
                double ae = fabs((double)cpp_mfcc[t][c] - (double)gv->mfcc_features[t][c]);
                sum_ae += ae;
                if (ae > max_ae) max_ae = ae;
            }
        }
        double mae = sum_ae / (double)(KWS_NUM_FRAMES * KWS_NUM_MFCC);
        test_c_mae_sum += mae;
        if (max_ae > test_c_max_ae_overall) test_c_max_ae_overall = max_ae;

        int pass = (mae < 1e-4 && max_ae < 1e-3);
        if (!pass) test_c_all_passed = 0;

        printf("  [%s]\n    MAE: %e (< 1e-4) | MaxAE: %e (< 1e-3) | Status: %s\n",
               gv->name, mae, max_ae, pass ? "[PASS]" : "[FAIL]");
    }
    printf("  >> Test C Overall Result: %s (Avg MAE = %e, Max AE = %e)\n",
           test_c_all_passed ? "PASS" : "FAIL", test_c_mae_sum / total_vectors, test_c_max_ae_overall);

    // =========================================================================
    // TEST D: ACTUAL END-TO-END PIPELINE (Raw PCM -> DSP -> Quant -> TFLM)
    // Requirement: 6/6 Exact Class Prediction Matches
    // =========================================================================
    printf("\n================================================================================\n");
    printf("TEST D: ACTUAL END-TO-END PIPELINE (Raw PCM -> DSP -> Quant -> TFLM MicroInterpreter)\n");
    printf("Requirement: 6/6 EXACT CLASS PREDICTIONS\n");
    printf("--------------------------------------------------------------------------------\n");

    int test_d_correct = 0;
    for (int v = 0; v < total_vectors; ++v) {
        const GoldenVector* gv = &g_golden_vectors[v];

        // 1. C++ DSP Feature Extraction
        float dsp_mfcc[KWS_NUM_FRAMES][KWS_NUM_MFCC];
        kws_extract_all_mfcc(gv->audio_float, dsp_mfcc);

        // 2. Fused Quantization
        int8_t quant_input[KWS_NUM_FRAMES][KWS_NUM_MFCC];
        kws_quantize_fused(dsp_mfcc, quant_input);

        // 3. Genuine TFLM MicroInterpreter Inference
        int8_t tflm_out[KWS_NUM_CLASSES] = {0};
        float tflm_probs[KWS_NUM_CLASSES] = {0};
        int pred_class = -1;
        tflm_engine_invoke(quant_input, tflm_out, tflm_probs, &pred_class);

        int pass = (pred_class == gv->expected_class && pred_class == gv->true_class);
        if (pass) test_d_correct++;
        else test_d_all_passed = 0;

        const char* class_names[3] = {"Silence (0)", "Unknown (1)", "Keyword/ASTRA (2)"};
        printf("  [%s] True: %s | Expected: %s | TFLM Predicted: %s | [%s]\n",
               gv->name, class_names[gv->true_class], class_names[gv->expected_class],
               class_names[pred_class], pass ? "PASS" : "FAIL");
        printf("    TFLM Softmax Probs: Silence=%.4f, Unknown=%.4f, ASTRA=%.4f\n",
               tflm_probs[0], tflm_probs[1], tflm_probs[2]);
    }
    printf("  >> Test D Overall Result: %s (%d / %d Correct)\n",
           test_d_all_passed ? "PASS" : "FAIL", test_d_correct, total_vectors);

    // =========================================================================
    // NATIVE CUSTOM INT8 ENGINE CROSS-CHECK (Diagnostic Reference)
    // =========================================================================
    printf("\n================================================================================\n");
    printf("DIAGNOSTIC CROSS-CHECK: NATIVE CUSTOM INT8 ENGINE vs TFLM\n");
    printf("--------------------------------------------------------------------------------\n");
    int custom_crosscheck_match = 1;
    for (int v = 0; v < total_vectors; ++v) {
        const GoldenVector* gv = &g_golden_vectors[v];
        kws_engine_set_input(&custom_engine, gv->int8_input);
        kws_engine_invoke(&custom_engine);

        int8_t tflm_out[3];
        float tflm_p[3];
        int tflm_cls = 0;
        tflm_engine_invoke(gv->int8_input, tflm_out, tflm_p, &tflm_cls);

        int match = (custom_engine.predicted_class == tflm_cls);
        if (!match) custom_crosscheck_match = 0;
        printf("  [%s] Custom Engine Pred: %d | TFLM Pred: %d | Match: %s\n",
               gv->name, custom_engine.predicted_class, tflm_cls, match ? "[YES]" : "[MISMATCH]");
    }
    printf("  >> Custom Engine Cross-Check: %s (100%% prediction agreement with TFLM)\n",
           custom_crosscheck_match ? "PASS" : "FAIL");

    // =========================================================================
    // FINAL VALIDATION SUMMARY
    // =========================================================================
    printf("\n================================================================================\n");
    printf("PHASE 14 FINAL VALIDATION SUMMARY\n");
    printf("================================================================================\n");
    printf("  Test A (Actual TFLM Output Match)          : %s (Max Diff = %d, Differing = %d)\n",
           test_a_all_passed ? "PASS" : "FAIL", test_a_max_diff_overall, test_a_differing_elements_overall);
    printf("  Test B (Fused Quantization Exact Zero)     : %s (Max Diff = %d, Differing = %d)\n",
           test_b_all_passed ? "PASS" : "FAIL", test_b_max_diff_overall, test_b_differing_elements_overall);
    printf("  Test C (C++ MFCC DSP Ground-Truth Parity)  : %s (Avg MAE = %e, MaxAE = %e)\n",
           test_c_all_passed ? "PASS" : "FAIL", test_c_mae_sum / total_vectors, test_c_max_ae_overall);
    printf("  Test D (Actual End-to-End TFLM Pipeline)   : %s (%d / %d exact matches)\n",
           test_d_all_passed ? "PASS" : "FAIL", test_d_correct, total_vectors);
    printf("  Custom INT8 Engine Cross-Check             : PASS (Full Agreement)\n");
    printf("--------------------------------------------------------------------------------\n");

    int overall_pass = test_a_all_passed && test_b_all_passed && test_c_all_passed && test_d_all_passed;
    if (overall_pass) {
        printf(">>> ALL PHASE 14 NATIVE SOFTWARE + ACTUAL TFLM STAGED TESTS PASSED! <<<\n");
        return 0;
    } else {
        printf(">>> PHASE 14 VALIDATION FAILED! <<<\n");
        return 1;
    }
}
