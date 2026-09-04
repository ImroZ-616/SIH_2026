import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow importing from project root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from audio.mel import create_mel_filterbank


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SAMPLE_RATE = 16000
DURATION = 1

FRAME_MS = 30
HOP_MS = 10

# Experimental values - not final
N_FFT = 480
N_MELS = 40

EPSILON = 1e-10


# --------------------------------------------------
# Framing
# --------------------------------------------------

def frame_audio(audio, sample_rate, frame_ms, hop_ms):

    frame_length = int(
        sample_rate * frame_ms / 1000
    )

    hop_length = int(
        sample_rate * hop_ms / 1000
    )

    frames = []

    for start in range(
        0,
        len(audio) - frame_length + 1,
        hop_length
    ):
        frame = audio[
            start:start + frame_length
        ]

        frames.append(frame)

    return np.array(frames)


# --------------------------------------------------
# Log-Mel extraction
# --------------------------------------------------

def compute_log_mel(audio):

    frames = frame_audio(
        audio,
        SAMPLE_RATE,
        FRAME_MS,
        HOP_MS
    )

    window = np.hanning(
        frames.shape[1]
    )

    mel_filterbank = create_mel_filterbank(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS
    )

    log_mel = []

    for frame in frames:

        # Hann window
        windowed_frame = frame * window

        # FFT
        fft_result = np.fft.rfft(
            windowed_frame,
            n=N_FFT
        )

        # Power spectrum
        power = np.abs(
            fft_result
        ) ** 2

        # Mel filtering
        mel_energy = (
            mel_filterbank @ power
        )

        # Log compression
        log_energy = np.log10(
            np.maximum(
                mel_energy,
                EPSILON
            )
        )

        log_mel.append(
            log_energy
        )

    return np.array(log_mel)


# --------------------------------------------------
# Generate signals
# --------------------------------------------------

time = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE


# 440 Hz tone
signal_440 = np.sin(
    2 * np.pi * 440 * time
)


# 1000 Hz tone
signal_1000 = np.sin(
    2 * np.pi * 1000 * time
)


# White noise
noise = np.random.normal(
    0,
    1,
    len(time)
)


# --------------------------------------------------
# Compute Log-Mel features
# --------------------------------------------------

mel_440 = compute_log_mel(
    signal_440
)

mel_1000 = compute_log_mel(
    signal_1000
)

mel_noise = compute_log_mel(
    noise
)


print("440 Hz Log-Mel shape:", mel_440.shape)
print("1000 Hz Log-Mel shape:", mel_1000.shape)
print("Noise Log-Mel shape:", mel_noise.shape)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

figures = [
    ("440 Hz Tone", mel_440),
    ("1000 Hz Tone", mel_1000),
    ("White Noise", mel_noise)
]


for title, mel in figures:

    plt.figure(figsize=(12, 5))

    plt.imshow(
        mel.T,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label="Log Mel Energy"
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Time Frame"
    )

    plt.ylabel(
        "Mel Band"
    )

    plt.tight_layout()

    plt.show()