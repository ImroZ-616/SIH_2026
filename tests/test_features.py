import numpy as np
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from audio.features import LogMelFeatureExtractor


SAMPLE_RATE = 16000
DURATION = 1
FREQUENCY = 440


# Generate test signal
time = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE

audio = np.sin(
    2 * np.pi * FREQUENCY * time
)


# Create feature extractor
extractor = LogMelFeatureExtractor(
    sample_rate=16000,
    frame_ms=30,
    hop_ms=10,
    n_fft=480,
    n_mels=40
)


# Extract features
features = extractor.extract(audio)


print("Audio shape:", audio.shape)
print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)

print(
    "Finite values:",
    np.all(np.isfinite(features))
)

print(
    "Feature minimum:",
    features.min()
)

print(
    "Feature maximum:",
    features.max()
)