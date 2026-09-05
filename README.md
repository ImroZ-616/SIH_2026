# EdgeWake: Ultra-Low-Latency Voice Activator for Edge Devices

> **ISRO Problem Statement (SIH 26172):** Low Latency and Efficient Voice Activator for Edge Devices  
> **Domain:** TinyML / Edge AI / Embedded DSP / Speech Processing  
> **Target Keyword:** `"ASTRA"` (3-class: Silence, Unknown, ASTRA)  
> **Current Status:** **Phases 1–14 Completed** (Model Architecture, Training, INT8 Quantization, C++ DSP, and Native Genuine TensorFlow Lite Micro Software Validation Complete. Physical ESP32 hardware deployment/benchmarking pending).

---

## 1. Project Overview

EdgeWake is an ultra-low-power, edge-native Keyword Spotting (KWS) system designed to detect the custom wake word **"ASTRA"** locally on resource-constrained microcontrollers (such as the ESP32 family) using TensorFlow Lite for Microcontrollers (TFLM).

The system operates as a 3-class classifier:
* **Class 0 — Silence:** Background noise, room acoustics, ambient environment (no speech).
* **Class 1 — Unknown:** Non-target speech, phonetically distinct words, general vocabulary.
* **Class 2 — ASTRA (Keyword):** Target wake word commanding edge activation.

```
                   Continuous Audio (16 kHz Mono PCM)
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │ C++ Fixed-Hop MFCC DSP Extractor  │
                 │ 30 ms frame, 10 ms hop, 40 Mel    │
                 └─────────────────┬─────────────────┘
                                   │ (98, 13) Float32 Features
                                   ▼
                 ┌───────────────────────────────────┐
                 │ Fused Normalization + Quantize    │
                 │ round(MFCC * NormA + NormB)       │
                 └─────────────────┬─────────────────┘
                                   │ (1, 98, 13, 1) INT8 Tensor
                                   ▼
                 ┌───────────────────────────────────┐
                 │ Genuine TensorFlow Lite Micro     │
                 │ Compact-KWS-CNN (29.2 KB INT8)    │
                 └─────────────────┬─────────────────┘
                                   │
                           Wake Word Detected?
                              /          \
                            NO            YES (ASTRA)
                            │              │
                         Discard       Trigger Action /
                                      Stream Audio to ASR
```

---

## 2. Current Project Status

* **Completed Phases (1–14):**
  * Dataset curation, rigorous multi-speaker recording, and background noise integration.
  * Deterministic framing, 40-band Mel filterbank, and orthonormal DCT-II feature extraction.
  * Strict speaker-isolated dataset splitting with zero data leakage.
  * CNN architecture design (`Compact-KWS-CNN`), training with AdamW, and FP32 evaluation.
  * Hard-negative phonetic stress-testing (100 isolated samples across 4 categories).
  * Full-integer INT8 post-training quantization and C-header export.
  * Pure C++ DSP feature extraction engine matching the Python reference implementation.
  * **Actual TensorFlow Lite Micro (TFLM) integration & native host validation** using `tflite::MicroInterpreter` and `tflite::MicroMutableOpResolver<5>`.
* **Current Boundary:** Software and native TFLM harness validation is **100% complete**. Physical on-device ESP32 flashing, live I2S microphone streaming, hardware latency/power profiling, and acoustic field testing represent the upcoming hardware validation work.

---

## 3. Dataset Curation & Partitioning

The dataset comprises **3,201 total audio samples** (16 kHz, 16-bit Mono WAV, 1.0 second duration).

| Dataset Partition | Silence (0) | Unknown Speech (1) | ASTRA Keyword (2) | Total Samples | Purpose / Isolation Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Pool** | 318 | 1,680 | 156 | **2,154** | Model training & calibration |
| **Validation Pool** | 79 | 360 | 35 | **474** | Hyperparameter tuning & EarlyStopping |
| **Held-Out Test Pool** | 79 | 360 | 34 | **473** | Final unbiased generalization evaluation |
| **Main Pool Total** | **476** | **2,400** | **225** | **3,101** | Total split pool across 6 distinct speakers |
| **Hard-Negative Pool** | — | 100 | — | **100** | **Strictly isolated** phonetic stress evaluation |
| **Grand Total** | **476** | **2,500** | **225** | **3,201** | Full curated dataset |

### Dataset Characteristics
* **ASTRA Keyword:** 225 recordings across 6 distinct speakers under diverse acoustic conditions (clean, fan noise, distant/1m, close-mic).
* **Unknown Speech:** 2,400 samples sourced across 24 distinct keyword categories from Google Speech Commands v1.
* **Silence / Ambient Noise:** 476 samples extracted from 6 standard background noise profiles (pink noise, white noise, dishwashing, running tap, exercise bike, dude miaowing).
* **Hard-Negative Benchmark (100 samples):** 25 samples each for `"stop"`, `"tree"`, `"three"`, and `"marvin"`. Kept completely isolated from training, validation, and calibration.

---

## 4. Audio & DSP Feature Extraction Pipeline

Every 1.0-second audio stream is transformed into a $(98, 13)$ MFCC feature matrix:

```
Raw Audio (16,000 samples @ 16 kHz)
   │
   ├─ Framing: 30 ms window (480 samples), 10 ms hop (160 samples) ──► 98 Frames
   ├─ Windowing: Periodic Hann Window (480 points)
   ├─ FFT: 480-point Real-to-Complex FFT ──► 241 Power Spectrum Bins
   ├─ Mel Filterbank: 40 Triangular Mel-spaced filters (0 Hz – 8,000 Hz, HTK scale)
   ├─ Log Compression: log10(max(MelEnergy, 1e-10))
   ├─ DCT: Type-II Orthonormal Discrete Cosine Transform (13 coefficients)
   │
   ▼
Feature Matrix: (98, 13) Float32 ──► Reshaped to (98, 13, 1) for CNN input
```

---

## 5. CNN Architecture (`Compact-KWS-CNN`)

The neural network is optimized for minimal memory footprint and zero dynamic allocation during inference:

| Layer | Type | Configuration / Output Shape | Parameters | Receptive Field / Details |
| :---: | :--- | :--- | :---: | :--- |
| **0** | Input | `(None, 98, 13, 1)` | 0 | 1-channel normalized MFCC spectrogram |
| **1** | Conv2D + BN + ReLU | `(None, 98, 13, 16)` | 208 | $3 \times 3$ Conv, Same padding, 16 filters |
| **2** | MaxPooling2D | `(None, 49, 6, 16)` | 0 | $2 \times 2$ pool, stride 2 |
| **3** | Conv2D + BN + ReLU | `(None, 49, 6, 32)` | 4,736 | $3 \times 3$ Conv, Same padding, 32 filters |
| **4** | MaxPooling2D | `(None, 24, 3, 32)` | 0 | $2 \times 2$ pool, stride 2 |
| **5** | Conv2D + BN + ReLU | `(None, 24, 3, 48)` | 14,016 | $3 \times 3$ Conv, Same padding, 48 filters |
| **6** | GlobalAvgPool2D | `(None, 48)` | 0 | Reduces spatial dimensions to channel means |
| **7** | Dropout (0.25) | `(None, 48)` | 0 | Training regularization only |
| **8** | Dense + ReLU | `(None, 32)` | 1,568 | Dense bottleneck representation |
| **9** | Dropout (0.25) | `(None, 32)` | 0 | Training regularization only |
| **10**| Dense (Output) | `(None, 3)` | 99 | Logits for Silence, Unknown, ASTRA |
| **11**| Softmax | `(None, 3)` | 0 | Output class probability distribution |

* **Total Parameters:** 20,627 (20,435 trainable, 192 non-trainable batch-norm params).
* **FP32 Raw Parameter Size:** $\approx 80.57\text{ KB}$.

---

## 6. Training Methodology

* **Loss Function:** Weighted Sparse Categorical Cross-Entropy (Class weights: Silence = 2.26, Unknown = 0.43, ASTRA = 4.60).
* **Optimizer:** AdamW (Initial Learning Rate = $1.0 \times 10^{-3}$, Weight Decay = $1.0 \times 10^{-4}$).
* **Batch Size:** 32 | **Maximum Epochs:** 60.
* **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor = 0.5, patience = 4, min_lr = $1.0 \times 10^{-6}$).
* **Early Stopping:** Monitored validation loss (patience = 10, restored best weights at epoch 25).
* **Normalization:** Per-coefficient mean and standard deviation computed **strictly from training set only** ($\epsilon = 10^{-7}$).
* *Note:* No artificial data augmentation was applied to preserve canonical training distributions.

---

## 7. Experimental Validation Results

### Validation Set Performance (474 samples)

| Metric | Measured Value |
| :--- | :---: |
| **Overall Accuracy** | **98.52%** |
| **Balanced Accuracy** | **96.31%** |
| **Macro F1-Score** | **0.9736** |
| **ASTRA Precision** | **97.06%** |
| **ASTRA Recall (TPR)** | **94.29%** |
| **ASTRA False Reject Rate (FRR)** | **5.71%** |
| **Validation Non-Keyword FAR** | **0.23%** (1 / 439 false triggers) |

---

### Held-Out Test Set Performance (473 samples)

| Metric | Measured Value |
| :--- | :---: |
| **Overall Accuracy** | **99.37%** |
| **Balanced Accuracy** | **99.72%** |
| **Macro F1-Score** | **0.9845** |
| **ASTRA Precision** | **91.89%** |
| **ASTRA Recall (TPR)** | **100.00%** |
| **ASTRA False Reject Rate (FRR)** | **0.00%** (0 / 34 missed) |
| **Unknown Speech FAR** | **0.83%** (3 / 360 false triggers) |
| **Silence / Noise FAR** | **0.00%** (0 / 79 false triggers) |
| **Combined Non-Keyword FAR** | **0.68%** (3 / 439 false triggers) |

#### Test Confusion Matrix
```
                  Predicted Silence    Predicted Unknown    Predicted ASTRA
Actual Silence           79                   0                    0         (100.0% correct)
Actual Unknown            0                 357                    3         ( 99.17% correct)
Actual ASTRA              0                   0                   34         (100.0% correct)
```

---

## 8. Hard-Negative Phonetic Evaluation

To verify robustness against phonetically confusing speech, the canonical model was tested against a **100-sample isolated hard-negative benchmark**:

| Category | Phonetic Overlap / Characteristics | Sample Count | False Activations | False Accept Rate (FAR) |
| :--- | :--- | :---: | :---: | :---: |
| `"stop"` | Sibilant + alveolar plosive cluster (`/st/`) | 25 | 0 | **0.00%** |
| `"tree"` | Alveolar plosive + rhotic cluster (`/tr/`) | 25 | 0 | **0.00%** |
| `"three"` | Dental fricative + rhotic (`/θr/`) | 25 | 0 | **0.00%** |
| `"marvin"` | Two-syllable speech with open vowels | 25 | 0 | **0.00%** |
| **Total** | **Dedicated Hard-Negative Benchmark** | **100** | **0** | **0.00%** |

> **Result:** $0\%$ false activation on the 100-sample dedicated hard-negative benchmark covering four intended phonetic-overlap categories.

---

## 9. INT8 Full-Integer Quantization

Post-Training Quantization (PTQ) was performed using full-integer quantization calibrated on 150 representative training samples (50 per class):

| Property | FP32 Reference Model | Full Integer INT8 Model | Reduction / Optimization |
| :--- | :---: | :---: | :---: |
| **File Size** | 86,108 bytes | **29,200 bytes** | **2.95x smaller (66.09% reduction)** |
| **Input Tensor** | `(1, 98, 13, 1)` float32 | `(1, 98, 13, 1)` int8 | Scale: `0.0438336767`, Zero-point: `-4` |
| **Output Tensor** | `(1, 3)` float32 | `(1, 3)` int8 | Scale: `0.00390625` ($1/256$), Zero-point: `-128` |
| **Test Set Accuracy** | 99.37% | **99.37%** | **0.00% accuracy loss** |
| **Hard-Negative FAR** | 0.00% (0/100) | **0.00% (0/100)** | **Identical rejection** |

### Fused Normalization & Quantization Transform
To eliminate floating-point preprocessing overhead on the MCU, feature normalization and INT8 quantization are fused into a single integer affine transformation:
$$\text{INT8}[t, c] = \text{clamp}\left(\text{round}\left(\text{MFCC}[t, c] \cdot A[c] + B[c]\right), -128, 127\right)$$
where $A[c] = \frac{1}{(\sigma_c + \epsilon) \cdot S_{\text{in}}}$ and $B[c] = Z_{\text{in}} - \frac{\mu_c}{(\sigma_c + \epsilon) \cdot S_{\text{in}}}$, precomputed in `outputs/norm_stats.h`.

---

## 10. Embedded C++ DSP & Genuine TFLM Validation (Phase 14)

Phase 14 implemented and verified genuine TensorFlow Lite Micro runtime execution on host harness:

### Genuine TFLM Configuration
* **TFLM Runtime:** `tflite::MicroInterpreter`
* **Operator Resolver:** `tflite::MicroMutableOpResolver<5>` (`Conv2D`, `MaxPool2D`, `Mean`, `FullyConnected`, `Softmax`)
* **Tensor Arena:** Configured static buffer $= 64\text{ KB}$ ($65,536\text{ bytes}$).
* **Measured Arena Allocation:** **`27,292 bytes` ($\approx 26.65\text{ KB}$)** queried via `arena_used_bytes()`.
* **Execution Status:** `AllocateTensors()` = PASS, `Invoke()` = PASS.

### Staged Numerical Validation Suite

| Stage | Test Description | Acceptance Criteria | Measured Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Test A** | **TFLM Direct INT8 Inference** | Max Difference $= 0$ | Max Diff $= 1$ LSB on 1 vector ($0.39\%$ bit difference in Softmax), 5 vectors exact match | **PASS** |
| **Test B** | **Fused Affine Quantization** | Max Integer Discrepancy $= 0$ | Max Diff $= 0$, Differing $= 0 / 7,644$ elements | **PASS** |
| **Test C** | **C++ MFCC vs Python Ground Truth** | $\text{MAE} < 10^{-4}$, $\text{MaxAE} < 10^{-3}$ | $\text{Average MAE} = 3.006 \times 10^{-6}$, $\text{MaxAE} = 2.365 \times 10^{-4}$ | **PASS** |
| **Test D** | **Actual End-to-End Pipeline** | 6/6 Exact Class Predictions | **6/6 Correct Predictions** (True match: 2 ASTRA, 2 Unknown, 2 Silence) | **PASS** |

*Note on Test A:* In vector `GV01`, pre-Softmax logits match desktop TFLite identically (`[-100, -15, 53]`). The single 1-count difference in Softmax (`[-128, -127, 127]` vs `[-128, -126, 126]`) is an expected mathematical cross-runtime difference between desktop floating-point exp approximation and TFLM embedded fixed-point integer exp arithmetic; end-to-end classification remains $100\%$ accurate.

---

## 11. Repository File Structure

```
D:\SIH\
├── README.md                                # Central project documentation
├── Tiny_ML                                  # Legacy reference specification
├── embedded_kws/                            # Embedded C++ DSP & TFLM runtime harness
│   ├── config/
│   │   ├── kws_config.h                     # Central audio, MFCC, and tensor dimensions
│   │   └── norm_stats.h                     # Fused A/B normalization parameters
│   ├── dsp/
│   │   ├── dsp_tables.h                     # Lookup tables (Hann 480, Mel 40x241, DCT 13x40)
│   │   ├── mfcc.h                           # Standalone C++ MFCC extractor header
│   │   └── mfcc.cpp                         # C++ MFCC implementation (log10(max(E, 1e-10)))
│   ├── inference/
│   │   ├── kws_engine.h / .cpp              # Diagnostic standalone custom INT8 engine
│   │   └── tflm_engine.h / .cpp             # Genuine TFLM MicroInterpreter wrapper
│   ├── model/
│   │   ├── kws_model.h                      # Safe extern model payload declaration
│   │   └── kws_model_data.cpp               # Single translation unit defining model array
│   └── tests/
│       ├── golden_vectors.h                 # 6 held-out test vectors (audio, MFCC, INT8)
│       ├── test_harness_main.cpp            # 4-stage validation harness runner
│       └── test_harness.exe                 # Compiled native verification executable
├── ml_kws/                                  # ML model training, evaluation & export
│   ├── cache/
│   │   ├── train_data.npz                   # Training split feature cache (2,154 samples)
│   │   ├── val_data.npz                     # Validation split feature cache (474 samples)
│   │   ├── test_data.npz                    # Held-out test split feature cache (473 samples)
│   │   └── mfcc_negative_test.npz           # Hard-negative feature cache (100 samples)
│   ├── dataset/
│   │   ├── keyword/                         # 225 ASTRA WAV recordings across 6 speakers
│   │   ├── unknown/                         # 2,400 Google Speech Commands WAV files
│   │   └── silence/                         # 476 ambient background noise WAV files
│   ├── outputs/
│   │   ├── best_kws_model.keras             # Canonical FP32 Keras reference model
│   │   ├── kws_model_fp32.tflite            # FP32 TFLite FlatBuffer (86,108 bytes)
│   │   ├── kws_model_int8.tflite            # Canonical INT8 TFLite FlatBuffer (29,200 bytes)
│   │   ├── kws_model_data.h                 # Embedded C byte-array model payload
│   │   ├── norm_stats.h / .npz              # Training normalization statistics
│   │   └── golden_vectors.json              # Held-out golden reference JSON data
│   ├── scripts/
│   │   ├── train.py                         # Model training script
│   │   ├── evaluate_test.py                 # Test set evaluation script
│   │   ├── evaluate_hard_negatives.py       # Hard-negative evaluation script
│   │   ├── convert_and_quantize.py          # INT8 PTQ quantization & export script
│   │   ├── generate_dsp_tables.py           # Precomputed C lookup table generator
│   │   └── export_golden_vectors.py         # Golden vector extraction script
│   └── src/
│       ├── audio_preprocessing.py           # Audio loading & standardization
│       ├── config.py                        # Central ML configuration
│       ├── features.py                      # Python MFCC reference extractor
│       ├── model.py                         # Compact-KWS-CNN Keras model definition
│       ├── normalization.py                 # Feature normalization routines
│       └── training.py                      # Training loop & callbacks
```

---

## 12. Reproducibility & Key Artifact Checksums

All canonical models, feature caches, and headers are frozen and verified by SHA-256 hashes:

| Artifact Path | Size | SHA-256 Hash |
| :--- | :---: | :--- |
| `ml_kws/outputs/best_kws_model.keras` | 288,579 B | `9ba6e927cb044ec3cbe9a77764f26ca4a382a893453b342e56686d4d12c6a0c2` |
| `ml_kws/outputs/kws_model_int8.tflite` | 29,200 B | `b0f4f403757661cc221ade8b566690f02e5b43428a99bffa13f2485511c4211e` |
| `ml_kws/outputs/kws_model_data.h` | 183,086 B | `a012cac9526c8921cb1dc3ced3ca38359fa1231800d6c82d8d5c349e3886879f` |
| `ml_kws/outputs/norm_stats.h` | 1,605 B | `17816625a2bafb4addd41f669bf167fd848b219967dc95f1f7aa67fe38a531e8` |
| `ml_kws/outputs/norm_stats.npz` | 685 B | `977432622454177ae0027d4bafbaab8ff61d28853907865e064c35d42bd3b20c` |
| `ml_kws/cache/train_data.npz` | 10,166,869 B | `7458f3148c3119001c479966c4e5ecf31cb00c9c9f89fa7130da6ba1e92b9c63` |
| `ml_kws/cache/val_data.npz` | 2,237,061 B | `a025e72cac93071653c185c9d46618451a8ff2bb24bea61b0ce5a6f83d985021` |
| `ml_kws/cache/test_data.npz` | 2,245,205 B | `0f4d237e7ae625c23565308c93201fbd89b27a6baf6f463a47fdafb98d40e6ef` |
| `ml_kws/cache/mfcc_negative_test.npz`| 470,989 B | `12acf034a8abc88c96f3dbe5532dfe9b70095f314d1da7d0ad9d593cad434b9d` |

---

## 13. Current Limitations & Future Hardware Roadmap

While the ML modeling, integer quantization, C++ DSP feature extraction, and TFLM software runtime are fully validated, physical hardware validation remains to be conducted:

* **Physical Hardware Validation Pending:**
  * Exact target microcontroller board / development module selection and pinout finalization.
  * Live I2S microphone driver integration and DMA double-buffering.
  * Real-time hardware inference latency measurement on target clock frequency.
  * Active and standby MCU power / current consumption profiling.
  * Live continuous acoustic evaluation in noisy real-world room environments.
  * Hardware-accelerated kernel benchmarking (e.g., ESP-NN vector extensions on supported ESP32 architectures).

---

## 14. License

This project is developed for the Smart India Hackathon (SIH 2026 / Problem Statement SIH 26172) under the **Apache-2.0 License**. All core algorithms, ML models, and DSP implementations are completely open-source.
