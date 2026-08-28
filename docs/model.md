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