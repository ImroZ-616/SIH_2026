import numpy as np
import matplotlib.pyplot as plt


def frame_audio(audio, sample_rate, frame_ms=25, hop_ms=10):
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    frames = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)

    return np.array(frames)


def apply_hann_window(frame):
    window = np.hanning(len(frame))
    return frame * window


if __name__ == "__main__":

    sample_rate = 16000

    # Create 1 second of example sine-wave audio
    duration = 1
    frequency = 440

    time = np.arange(sample_rate * duration) / sample_rate
    audio = np.sin(2 * np.pi * frequency * time)

    # Frame the audio
    frames = frame_audio(
        audio,
        sample_rate,
        frame_ms=30,
        hop_ms=10
    )

    # Take the first frame
    frame = frames[0]

    # Apply Hann window
    windowed_frame = apply_hann_window(frame)

    print("Audio samples:", len(audio))
    print("Frames shape:", frames.shape)
    print("Samples per frame:", len(frame))

    print("\nFirst 5 original samples:")
    print(frame[:5])

    print("\nFirst 5 Hann window values:")
    print(np.hanning(len(frame))[:5])

    print("\nFirst 5 windowed samples:")
    print(windowed_frame[:5])

    # Plot original frame
    plt.figure(figsize=(12, 4))
    plt.plot(frame)
    plt.title("Original Audio Frame")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot windowed frame
    plt.figure(figsize=(12, 4))
    plt.plot(windowed_frame)
    plt.title("Audio Frame After Hann Windowing")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()