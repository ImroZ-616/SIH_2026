# 🧠 R2 — ML/KWS Engineer — Week 1 Checklist

**Main responsibility:** Dataset, CNN model, training, evaluation

---

## Day 1 — Shared Foundation (everyone)

- [x] Can explain what KWS (Keyword Spotting) is

- [x] Can explain what TinyML is

- [x] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)

- [x] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific

- [x] Classification basics

- [x] Train / validation / test splits

- [x] Overfitting

- [x] Confusion matrix

- [x] Precision

- [x] Recall

- [x] False positives / false negatives

- [x] Understand: Input → audio features → CNN → keyword probability

- [x] Research the **Speech Commands Dataset** structure

- [x] Create tiny dataset skeleton: `dataset/keyword/`, `dataset/unknown/`, `dataset/silence/`

- [x] **Output:** a clear dataset plan (written)

## Day 2 — Feature Representations

- [x] Compare raw waveform vs spectrogram vs MFCC vs log-mel

- [x] Understand why raw audio isn't always the best model input

## Day 3 — Feature Visualization (team milestone)

- [x] Take R1's MFCC output and visualize keyword / unknown / silence classes

- [ ] Answer: can the keyword visually separate itself from other classes?

## Day 4 — First KWS Model (you lead today)

- [ ] Build a simple CNN: MFCC → Conv2D → ReLU → Pooling → Conv2D → Dense → Softmax

- [ ] Classes: KEYWORD, UNKNOWN, SILENCE

- [ ] Train the model and record accuracy

## Day 5 — Live Microphone Milestone

- [ ] Wire trained model into live pipeline: Mic → MFCC → CNN → probability → wake detection

- [ ] Confirm it reliably prints e.g. `ASTRA 0.95 → WAKE WORD DETECTED`

## Day 6 — False Activation Testing

- [ ] Test model against tricky negatives (similar words, fast/slow speech, whisper, background noise, music, clapping)

- [ ] Collect hard examples into `negative_test/`

## Day 7 — Integration Day

- [ ] Custom KWS deliverable finalized: custom keyword → detection working end-to-end on laptop mic

---

### End-of-day rule

Each day, answer: **"What did I build today that another teammate can actually use?"**