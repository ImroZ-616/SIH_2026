# R2 — ML/KWS Model

**Project:** SIH 26172  
**Role:** R2 — ML/KWS Engineer  
**Component:** Keyword Spotting (KWS) Neural Network Model  
**Current Phase:** Phase 3 — Audio Loading & Preprocessing  

---

## 1. Purpose of the ML/KWS Component

The ML/KWS (Machine Learning / Keyword Spotting) module is a core sub-system of the SIH 26172 voice processing pipeline. Its primary objective is to detect a target wake word / keyword from audio streams in real-time with high accuracy, low latency, and minimal false activations.

Key responsibilities of this component include:
- Curating balanced acoustic datasets (target keyword, unknown speech words, ambient silence/noise, and hard negatives).
- Standardizing raw audio into a consistent, robust representation (16 kHz, Mono, 1.0s, Normalized).
- Transforming waveforms into 2D time-frequency acoustic representations (MFCCs).
- Designing, training, and evaluating lightweight Convolutional Neural Networks (CNNs).
- Rigorously testing against false positives and acoustic hard negatives.
- Enabling live microphone wake-word inference.
- Quantizing and converting the model to TensorFlow Lite / TinyML format for edge device deployment (R3 handoff).

---

## 2. Storage Location

All ML/KWS files, virtual environments, datasets, and generated artifacts are strictly allocated on drive `D:` to preserve system drive `C:` capacity:
- **Component Root:** `D:\SIH\ml_kws\` (or `<repository_root>\ml_kws\`)
- **Dataset Storage:** `D:\SIH\ml_kws\dataset\`
- **Cache Storage:** `D:\SIH\ml_kws\cache\`
- **Virtual Environment:** `D:\SIH\ml_kws\venv\`

---

## 3. Directory Structure

```
D:\SIH\ml_kws\
│
├── dataset/                    # Audio datasets (ignored by git)
│   ├── keyword/                # Target keyword ('ASTRA') audio samples
│   ├── unknown/                # Non-keyword speech samples (Speech Commands v0.01)
│   ├── silence/                # Background noise & ambience slices
│   ├── negative_test/          # Hard acoustic negative test audio
│   └── README.md               # Dataset metadata, licensing & collection docs
│
├── src/                        # Core source code modules
│   ├── config.py               # Central project path definitions & audio standards
│   └── audio.py                # Audio loading, resampling, normalization & padding
│
├── models/                     # Saved model architectures and weights (.h5, .tflite)
├── outputs/                    # Visualizations, confusion matrices, evaluation plots
├── notebooks/                  # Jupyter notebooks for experiments & analysis
├── scripts/                    # Standalone utility & automation scripts
│   ├── download_speech_commands.py  # Dataset downloader & organizer
│   ├── record_keyword.py            # Interactive CLI tool for recording 'ASTRA'
│   ├── validate_dataset.py          # Dataset audio integrity & count validator
│   ├── inspect_audio.py             # Raw audio dataset property inspector
│   └── test_audio.py                # Audio preprocessing validation suite
│
├── logs/                       # Training logs & execution traces
├── cache/                      # Precomputed MFCC arrays & cached archives
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
| **PHASE 2** | **Dataset Download & Organization** | **PARTIALLY COMPLETED** *(Awaiting custom 'ASTRA' recordings)* |
| **PHASE 3** | **Audio Loading & Preprocessing** | **COMPLETED** |
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

## 5. Audio Standardization & Preprocessing Architecture

### Standard Audio Specifications
- **Target Sample Rate:** `16,000 Hz` (16 kHz)
- **Target Channels:** `1` (Mono)
- **Target Duration:** `1.0 second`
- **Target Sample Count:** `16,000 samples`
- **Data Type:** `numpy.float32` (normalized values in $[-1.0, 1.0]$)

### The Preprocessing Pipeline (`src/audio.py`)
```
Raw WAV File / Stream
         ↓
load_audio()          --> Decodes WAV / FLAC / MP3 into float32 array
         ↓
to_mono()             --> Averages stereo / multi-channel audio to 1D mono
         ↓
resample_audio()      --> Polyphase / Soxr HQ resampler to 16,000 Hz
         ↓
normalize_audio()     --> Peak normalization (preserves silence below threshold)
         ↓
pad_or_trim()         --> Centers short audio or center-crops long audio to 16,000 samples
         ↓
Standardized Waveform --> Shape: (16000,), Dtype: float32 (Ready for MFCC Extraction)
```

---

## 6. Preprocessing Verification Summary

- [x] Tested on **9 real audio files** sampled randomly across `dataset/unknown/`, `dataset/silence/`, and `dataset/negative_test/`.
- [x] Tested on **5 synthetic edge cases** (0.5s short, 1.8s long, 2-channel stereo, 44.1 kHz non-standard sample rate, and near-zero silence).
- [x] All test cases resulted in exactly **16,000 float32 samples**, strictly mono, with valid amplitude boundaries $[-1.0, 1.0]$ and zero NaNs/Infs.
- [x] **Overall Validation Result:** `PASS`
