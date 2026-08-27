# R2 — ML/KWS Model

**Project:** SIH 26172  
**Role:** R2 — ML/KWS Engineer  
**Component:** Keyword Spotting (KWS) Neural Network Model  
**Current Phase:** Phase 2 — Dataset Download & Organization  

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
│   └── config.py               # Central project path definitions & configuration
│
├── models/                     # Saved model architectures and weights (.h5, .tflite)
├── outputs/                    # Visualizations, confusion matrices, evaluation plots
├── notebooks/                  # Jupyter notebooks for experiments & analysis
├── scripts/                    # Standalone utility & automation scripts
│   ├── download_speech_commands.py  # Dataset downloader & organizer
│   ├── record_keyword.py            # Interactive CLI tool for recording 'ASTRA'
│   └── validate_dataset.py          # Dataset audio integrity & count validator
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
| **PHASE 2** | **Dataset Download & Organization** | **IN PROGRESS** |
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

## 5. Dataset Architecture & Specifications

### Audio Format Standards
- **Sample Rate:** 16,000 Hz (16 kHz)
- **Bit Depth:** 16-bit Linear PCM
- **Channels:** 1 (Mono)
- **Clip Duration:** 1.0 second (16,000 samples)
- **Container:** Standard `.wav` (RIFF header)

### Dataset Classes & Sources
1. **KEYWORD (`dataset/keyword/`):**
   - **Target Keyword:** "ASTRA"
   - **Source:** Genuine audio samples recorded directly using `scripts/record_keyword.py`.
   - **Collection Diversity:** Multiple speakers, varied microphones (laptop internal, USB, headset), variable speaking speeds (normal, fast, slow), varying volumes (quiet, normal, loud), and different ambient noise conditions.
   - **Baseline Target:** 50–100 samples across 2–4 speakers.
2. **UNKNOWN (`dataset/unknown/`):**
   - **Source:** Google Speech Commands v0.01 ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
   - **Vocabulary:** 30 distinct spoken English words (*yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine, bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow*).
3. **SILENCE (`dataset/silence/`):**
   - **Source:** Google Speech Commands `_background_noise_` tracks (*running tap, pink noise, white noise, dishwashing, exercise bike*) sliced into 1.0s chunks + synthetic near-silence.
4. **NEGATIVE_TEST (`dataset/negative_test/`):**
   - **Source:** Curated Speech Commands words and team-collected acoustic hard negatives (*extra, astro, aster, master, pasta*, coughing, clapping). Used strictly for post-training false-activation benchmarking.

---

## 6. Phase 2 Progress Details

- [x] Download & extraction script [`scripts/download_speech_commands.py`](file:///D:/SIH/ml_kws/scripts/download_speech_commands.py) developed and executed.
- [x] Custom keyword recorder [`scripts/record_keyword.py`](file:///D:/SIH/ml_kws/scripts/record_keyword.py) created.
- [x] Dataset validation suite [`scripts/validate_dataset.py`](file:///D:/SIH/ml_kws/scripts/validate_dataset.py) created.
- [x] Dataset metadata & licensing documented in [`dataset/README.md`](file:///D:/SIH/ml_kws/dataset/README.md).
- [x] Zero ML models built, zero MFCC extraction performed.
