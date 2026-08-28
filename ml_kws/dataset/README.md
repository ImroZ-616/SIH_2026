# SIH 26172 — ML/KWS Dataset Documentation

**Component:** R2 — ML/KWS Dataset Repository  
**Audio Standard:** 16,000 Hz (16 kHz), 16-bit PCM, 1.0 second duration, Mono WAV  
**Storage Location:** `D:\SIH\ml_kws\dataset\`  

---

## 1. Class Taxonomy & Structure

The Keyword Spotting (KWS) system classifies incoming audio windows into 3 primary production classes, alongside a benchmark validation class for false-activation testing:

```
dataset/
├── keyword/           # Class 1: Target wake word ('ASTRA') - To be recorded by team
├── unknown/           # Class 2: Non-keyword English speech words (Speech Commands v0.01)
├── silence/           # Class 3: Ambient background noise and quiet room audio
└── negative_test/     # Benchmark: Phonetically challenging negative examples
```

---

## 2. Dataset Sources & Licensing

| Class | Source | License | Description |
| :--- | :--- | :--- | :--- |
| **KEYWORD** (`ASTRA`) | Proprietary / Team Collection | Project Internal | Genuine voice recordings of the target wake-word 'ASTRA' collected across team members. |
| **UNKNOWN** | Google Speech Commands v0.01 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 30 distinct spoken English words (e.g., *yes, no, up, down, stop, go, marvin, happy, zero-nine*). |
| **SILENCE** | Google Speech Commands `_background_noise_` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Real-world background noise tracks (running tap, pink noise, white noise, dishwashing, bike) sliced into 1.0s clips + digital silence. |
| **NEGATIVE_TEST** | Curated Speech Commands + Team Hard Negatives | Mixed / CC BY 4.0 | Phonetically adjacent words (*extra, astro, aster, master, pasta*) and confusing non-speech acoustics. |

---

## 3. Custom Keyword Collection Plan: 'ASTRA'

To ensure the neural network generalizes reliably in real-world hackathon deployments without overfitting to a single person or environment, keyword collection must adhere to the following acoustic diversity matrix:

### Recommended Collection Matrix
1. **Speaker Diversity:** Minimum 3–5 distinct individuals (diverse pitch, genders, vocal timbres, accents).
2. **Microphone Variety:** Built-in laptop microphones, external USB microphones, wired earbuds, headset mics.
3. **Distance & Angle:** 
   - Near-field (10–20 cm)
   - Mid-field (50–100 cm)
   - Far-field (1.5–2.0 meters)
4. **Speaking Cadence & Volume:**
   - Normal conversational tone
   - Fast utterance
   - Slow / drawn-out utterance
   - Quiet / whisper
   - Loud / projected voice
5. **Acoustic Environments:**
   - Quiet bedroom / office
   - Room with ceiling fan / AC hum
   - Background keyboard typing / mouse clicking
   - Distant speech / living room ambience

### Initial Practical Target
- **Phase 2 Baseline Target:** 50–100 genuine samples across 2–4 speakers.
- **Production Target (Phase 6/8):** 300–500+ samples with extensive data augmentation (speed perturbation, noise injection).

### Collection Procedure
Run the interactive CLI recorder from the project virtual environment:
```powershell
& "D:\SIH\ml_kws\venv\Scripts\python.exe" "D:\SIH\ml_kws\scripts\record_keyword.py"
```

---

## 4. Preventing Data Leakage

- **Speaker-Level Splitting:** In Phase 6 (Dataset Splitting), audio recordings will be partitioned by **Speaker ID**, not randomly shuffled file-by-file. All recordings from a specific speaker will reside strictly within either the Training, Validation, or Test set.
- **Strict Test Isolation:** The final test dataset and `negative_test/` will remain completely unseen during model training and hyperparameter tuning to ensure an honest, real-world evaluation benchmark.

---

## 5. Dataset Validation

Run the automated integrity checker to verify audio formatting, sampling rates, and class counts:
```powershell
& "D:\SIH\ml_kws\venv\Scripts\python.exe" "D:\SIH\ml_kws\scripts\validate_dataset.py"
```
