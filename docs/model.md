# KWS Model

## Objective

Detect a custom wake word locally on a resource-constrained edge device.

## Input

Audio-derived features such as MFCC or Log-Mel spectrogram.

## Model

Status: TBD

Candidate architectures:

- Small CNN
- Depthwise-separable CNN

## Quantization

Target:

INT8

## Classes

- Target keyword
- Unknown speech
- Silence / background

## Evaluation

- Accuracy
- Precision
- Recall
- F1
- False Accept Rate
- False Reject Rate
- False Activations per Hour

## R1 Audio Feature Configuration

The current reference audio feature pipeline uses:

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Input format | PCM int16 |
| Frame length | 30 ms |
| Frame length | 480 samples |
| Hop length | 10 ms |
| Hop length | 160 samples |
| Window | Hann |
| FFT size | 480 |
| Mel bands | 40 |
| Log transform | log10 |
| DCT | Type-II, orthonormal |
| MFCC coefficients | 13 |
| Normalization | None |
| Output dtype | float32 |

### Output shape

For a 1-second 16 kHz waveform:

- Input shape: `(16000,)`
- Number of frames: `98`
- MFCC coefficients per frame: `13`
- Output shape: `(98, 13)`

For the current 3-second microphone recording:

- Input shape: `(48000,)`
- Output shape: `(298, 13)`

### Feature extraction API

R2 can obtain MFCC features using:

```python
from audio.mfcc import extract_mfcc

features = extract_mfcc(waveform_16k)