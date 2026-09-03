import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 16000
DURATION = 1
FREQUENCY = 440

FRAME_MS = 30
HOP_MS = 10


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
# 1. Generate 440 Hz test signal
# --------------------------------------------------

time = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE

audio = np.sin(
    2 * np.pi * FREQUENCY * time
)


# --------------------------------------------------
# 2. Frame the audio
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
# 3. Window + FFT every frame
# --------------------------------------------------

window = np.hanning(frames.shape[1])

spectrogram = []

for frame in frames:

    # Apply Hann window
    windowed_frame = frame * window

    # FFT
    fft_result = np.fft.rfft(windowed_frame)

    # Magnitude
    magnitude = np.abs(fft_result)

    spectrogram.append(magnitude)


spectrogram = np.array(spectrogram)


print("Spectrogram shape:", spectrogram.shape)


# --------------------------------------------------
# 4. Frequency axis
# --------------------------------------------------

frequencies = np.fft.rfftfreq(
    frames.shape[1],
    d=1 / SAMPLE_RATE
)


# --------------------------------------------------
# 5. Time axis
# --------------------------------------------------

frame_times = (
    np.arange(len(frames))
    * HOP_MS
    / 1000
)


# --------------------------------------------------
# 6. Plot spectrogram
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.imshow(
    spectrogram.T,
    origin="lower",
    aspect="auto",
    extent=[
        frame_times[0],
        frame_times[-1],
        frequencies[0],
        frequencies[-1]
    ]
)

plt.colorbar(label="Magnitude")

plt.title("FFT Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.ylim(0, 1000)

plt.tight_layout()
plt.show()