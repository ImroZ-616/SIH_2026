import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from audio.features import LogMelFeatureExtractor


# Load microphone recording
audio = np.load("tests/mic_audio.npy")

# Create feature extractor
extractor = LogMelFeatureExtractor(
    sample_rate=16000,
    frame_ms=30,
    hop_ms=10,
    n_fft=480,
    n_mels=40
)

# Extract Log-Mel features
features = extractor.extract(audio)

print("Audio shape:", audio.shape)
print("Audio dtype:", audio.dtype)

print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)

print("Finite:", np.all(np.isfinite(features)))

# Plot Log-Mel spectrogram
plt.figure(figsize=(12, 5))

plt.imshow(
    features.T,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Log-Mel Energy")

plt.title("Real Microphone Log-Mel Spectrogram")
plt.xlabel("Time Frame")
plt.ylabel("Mel Band")

plt.tight_layout()
plt.show()