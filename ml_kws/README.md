# R2 — ML/KWS Model

**Project:** SIH 26172  
**Role:** R2 — ML/KWS Engineer  
**Component:** Keyword Spotting (KWS) Neural Network Model  
**Current Phase:** Phase 1 — Environment & ML Project Setup  

---

## 1. Purpose of the ML/KWS Component

The ML/KWS (Machine Learning / Keyword Spotting) module is a core sub-system of the SIH 26172 voice processing pipeline. Its primary objective is to detect a target wake word / keyword from audio streams in real-time with high accuracy, low latency, and minimal false activations.

Key responsibilities of this component include:
- Curating balanced acoustic datasets (target keyword, unknown speech words, ambient silence/noise, and hard negatives).
- Transforming raw audio waveforms into 2D time-frequency acoustic representations (MFCCs / Mel-Frequency Cepstral Coefficients).
- Designing, training, and evaluating lightweight Convolutional Neural Networks (CNNs).
- Rigorously testing against false positives and acoustic hard negatives.
- Enabling live microphone wake-word inference.
- Quantizing and converting the model to TensorFlow Lite / TinyML format for edge device deployment (R3 handoff).

---

## 2. Storage Location

All ML/KWS files, virtual environments, datasets, and generated artifacts are strictly allocated on drive `D:` to preserve system drive `C:` capacity:
- **Component Root:** `D:\SIH\ml_kws\` (or `<repository_root>\ml_kws\`)
- **Virtual Environment:** `D:\SIH\ml_kws\venv\`

---

## 3. Directory Structure

```
D:\SIH\ml_kws\
│
├── dataset/                    # Audio datasets (ignored by git)
│   ├── keyword/                # Target keyword audio samples
│   ├── unknown/                # Non-keyword speech samples
│   ├── silence/                # Background noise / silence samples
│   └── negative_test/          # Tricky acoustic negative test audio
│
├── src/                        # Core source code modules
│   └── config.py               # Central project path definitions & configuration
│
├── models/                     # Saved model architectures and weights (.h5, .tflite)
├── outputs/                    # Visualizations, confusion matrices, evaluation plots
├── notebooks/                  # Jupyter notebooks for experiments & analysis
├── scripts/                    # Standalone utility & automation scripts
├── logs/                       # Training logs & execution traces
├── cache/                      # Precomputed MFCC arrays & cached features
├── venv/                       # Dedicated Python virtual environment (ignored by git)
├── .gitignore                  # Component-specific Git ignore rules
├── requirements.txt            # Project dependency specifications
└── README.md                   # Component documentation & status tracking
```

---

## 4. Phase Roadmap & Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **PHASE 1** | **Environment & ML Project Setup** | **COMPLETED** |
| **PHASE 2** | Dataset Download & Organization | NOT STARTED |
| **PHASE 3** | Audio Loading & Preprocessing | NOT STARTED |
| **PHASE 4** | MFCC Feature Extraction | NOT STARTED |
| **PHASE 5** | MFCC Visualization | NOT STARTED |
| **PHASE 6** | Dataset Splitting & Preparation | NOT STARTED |
| **PHASE 7** | CNN Construction | NOT STARTED |
| **PHASE 8** | CNN Training | NOT STARTED |
| **PHASE 9** | Model Evaluation | NOT STARTED |
| **PHASE 10** | Hard-Negative & False-Activation Testing | NOT STARTED |
| **PHASE 11** | Live Microphone KWS Inference | NOT STARTED |
| **PHASE 12** | Model Optimization | NOT STARTED |
| **PHASE 13** | TensorFlow Lite Conversion | NOT STARTED |
| **PHASE 14** | TinyML / Edge Deployment Preparation | NOT STARTED |

---

## 5. Phase 1 Status Details

- [x] Main Git repository inspected and preserved without modifying other teammates' files.
- [x] Dedicated project directory structure created on drive `D:`.
- [x] Python virtual environment (`venv`) established and verified.
- [x] `requirements.txt` defined with planned dependencies.
- [x] `src/config.py` configured with path resolution via `pathlib.Path`.
- [x] `.gitignore` established to exclude large datasets, models, caches, and venv.
- [x] Zero ML models trained, zero audio downloaded (reserved strictly for subsequent phases).
