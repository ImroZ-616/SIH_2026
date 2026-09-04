import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow importing from the project root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from audio.mel import create_mel_filterbank


# Current R3-compatible audio configuration
SAMPLE_RATE = 16000
DURATION = 1

FRAME_MS = 30
HOP_MS = 10

# Experimental values - NOT FINAL
N_FFT = 480
N_MELS = 40

FREQUENCY = 440


def frame_audio(audio, sample_rate, frame_ms, hop_ms):
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    frames = []

    for start in range(
        0,
        len(audio) - frame_length + 1,
        hop_length
    ):
        frame = audio[start:start + frame_length]
        frames.append(frame)

    return np.array(frames)


# --------------------------------------------------
# 1. Generate test audio
# --------------------------------------------------

time = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE

audio = np.sin(
    2 * np.pi * FREQUENCY * time
)


# --------------------------------------------------
# 2. Framing
# --------------------------------------------------

frames = frame_audio(
    audio,
    SAMPLE_RATE,
    FRAME_MS,
    HOP_MS
)

print("Audio samples:", len(audio))
print("Frames shape:", frames.shape)


# --------------------------------------------------
# 3. Hann window
# --------------------------------------------------

window = np.hanning(frames.shape[1])


# --------------------------------------------------
# 4. FFT → Power Spectrum
# --------------------------------------------------

power_spectra = []

for frame in frames:

    windowed_frame = frame * window

    fft_result = np.fft.rfft(
        windowed_frame,
        n=N_FFT
    )

    magnitude = np.abs(fft_result)

    power = magnitude ** 2

    power_spectra.append(power)


power_spectra = np.array(power_spectra)

print("Power spectrum shape:", power_spectra.shape)


# --------------------------------------------------
# 5. Create Mel filter bank
# --------------------------------------------------

mel_filterbank = create_mel_filterbank(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS
)

print("Mel filter bank shape:", mel_filterbank.shape)


# --------------------------------------------------
# 6. Power Spectrum → Mel Spectrogram
# --------------------------------------------------

mel_spectrogram = []

for power in power_spectra:

    mel_energy = (
        mel_filterbank @ power
    )

    mel_spectrogram.append(
        mel_energy
    )


mel_spectrogram = np.array(
    mel_spectrogram
)

print(
    "Mel spectrogram shape:",
    mel_spectrogram.shape
)


# --------------------------------------------------
# 7. Log compression
# --------------------------------------------------

EPSILON = 1e-10

log_mel_spectrogram = np.log10(
    np.maximum(
        mel_spectrogram,
        EPSILON
    )
)
print("Mel min:", np.min(mel_spectrogram))
print("Mel max:", np.max(mel_spectrogram))
print("Finite Mel values:", np.all(np.isfinite(mel_spectrogram)))

print("Log-Mel min:", np.min(log_mel_spectrogram))
print("Log-Mel max:", np.max(log_mel_spectrogram))
print("Finite Log-Mel values:", np.all(np.isfinite(log_mel_spectrogram)))


print(
    "Log-Mel spectrogram shape:",
    log_mel_spectrogram.shape
)


# --------------------------------------------------
# 8. Visualize
# --------------------------------------------------

frame_times = (
    np.arange(len(frames))
    * HOP_MS
    / 1000
)


plt.figure(figsize=(12, 5))

plt.imshow(
    log_mel_spectrogram.T,
    origin="lower",
    aspect="auto",
    extent=[
        frame_times[0],
        frame_times[-1],
        0,
        N_MELS
    ]
)

plt.colorbar(
    label="Log Mel Energy"
)

plt.title(
    "Log-Mel Spectrogram"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Mel Band"
)

plt.tight_layout()
plt.show()