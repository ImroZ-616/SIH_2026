# 🎧 R1 — Audio/DSP Engineer — Week 1 Checklist

**Main responsibility:** Audio capture, sampling, FFT, MFCC/feature extraction

---

## Day 1 — Shared Foundation (everyone)
- [x] Can explain what KWS (Keyword Spotting) is
- [x] Can explain what TinyML is
- [x] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)
- [x] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific
- [x] Sampling & sample rate
- [x] PCM (Pulse Code Modulation)
- [x] Amplitude
- [x] Bit depth
- [x] Mono vs stereo
- [x] Nyquist theorem
- [x] Understand: 16,000 samples/sec → speech waveform
- [x] Install: Python, librosa, numpy, matplotlib, sounddevice
- [ ] Record your own voice
- [ ] Visualize waveform
- [ ] Visualize spectrogram
- [ ] **Output:** `audio_basics.ipynb` (mic recording + waveform + spectrogram)

## Day 2 — Framing & FFT
- [ ] Framing
- [ ] Windowing
- [ ] FFT (Fast Fourier Transform)
- [ ] Spectrogram generation
- [ ] Implement: audio → spectrogram (code)

## Day 3 — MFCC Pipeline (team milestone)
- [ ] Pre-emphasis
- [ ] Framing
- [ ] Windowing
- [ ] FFT
- [ ] Mel filterbank
- [ ] Log
- [ ] DCT
- [ ] Implement full pipeline: Audio → Pre-emphasis → Framing → Window → FFT → Mel filterbank → Log → DCT → MFCC

## Day 4 — Support R2
- [ ] Help optimize feature extraction speed/quality for the CNN pipeline

## Day 5 — Live Microphone Milestone
- [ ] Confirm your feature pipeline feeds correctly into the live mic → MFCC → CNN → probability demo

## Day 6 — False Activation Testing
- [ ] Help collect/label difficult audio samples (fast/slow speech, whisper, background noise, similar-sounding words)

## Day 7 — Integration Day
- [ ] Audio pipeline (Mic → PCM) finalized and committed
- [ ] Feature pipeline (PCM → MFCC/Log-Mel) finalized and committed

---
### End-of-day rule
Each day, answer: **"What did I build today that another teammate can actually use?"**
