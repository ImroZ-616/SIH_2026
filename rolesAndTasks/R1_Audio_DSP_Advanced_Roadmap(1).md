# 🎧 Member 1 — Audio DSP & Speech Processing — Post-Week-1 Roadmap

**Role:** Audio/DSP Engineer
**Final competency target:** *"Exactly what happens to the microphone signal between the microphone and the KWS model?"*

---

## 🟢 Level 1 — Audio Fundamentals
- [ ] Sound waves
- [ ] Frequency
- [ ] Amplitude
- [ ] Phase
- [ ] Sampling
- [ ] Sampling frequency
- [ ] Nyquist theorem
- [ ] Aliasing
- [ ] Quantization
- [ ] Bit depth
- [ ] PCM
- [ ] Mono vs stereo
- [ ] WAV format
- [ ] Dynamic range
- [ ] Signal-to-noise ratio (SNR)
- [ ] Understand: Microphone → Analog signal → ADC → Digital PCM samples

## 🟢 Level 2 — Digital Signal Processing
- [ ] Discrete signals
- [ ] Discrete-time systems
- [ ] Fourier Transform
- [ ] DFT
- [ ] FFT
- [ ] Frequency domain vs time domain
- [ ] STFT (Short-Time Fourier Transform)
- [ ] Windowing
- [ ] Hann / Hamming windows
- [ ] Spectrogram
- [ ] Frequency resolution
- [ ] Time resolution
- [ ] Understand: Audio waveform → FFT → Frequency spectrum

## 🟢 Level 3 — Speech Processing
- [ ] Human speech frequency ranges
- [ ] Voiced vs unvoiced speech
- [ ] Formants
- [ ] Phonemes
- [ ] Speech envelope
- [ ] Voice Activity Detection (VAD)
- [ ] Noise
- [ ] Reverberation
- [ ] Echo
- [ ] Far-field speech

## 🟢 Level 4 — Mel Features (critical)
- [ ] Mel scale
- [ ] Mel filter banks
- [ ] Log-Mel spectrogram
- [ ] MFCC
- [ ] DCT
- [ ] Number of Mel filters
- [ ] Number of MFCC coefficients
- [ ] Frame size
- [ ] Hop length
- [ ] Know the full pipeline: PCM → Pre-emphasis → Framing → Windowing → FFT → Power spectrum → Mel filterbank → Log → DCT → MFCC

## 🟢 Level 5 — Noise Robustness
- [ ] Gaussian noise
- [ ] Background noise
- [ ] SNR handling
- [ ] Noise augmentation
- [ ] Reverberation / room impulse response
- [ ] Pitch shifting
- [ ] Time stretching
- [ ] Volume augmentation

## 🟡 Level 6 — Tools
- [ ] NumPy
- [ ] SciPy
- [ ] librosa
- [ ] sounddevice
- [ ] matplotlib
- [ ] PyAudio
- [ ] WAV manipulation

---
### 🎯 Checkpoint
Can you fully answer: *"Exactly what happens to the microphone signal between the microphone and the KWS model?"*
