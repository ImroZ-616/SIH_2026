import numpy as np


def frame_audio(audio, sample_rate, frame_ms=25, hop_ms=10):
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    frames = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)

    return np.array(frames)


if __name__ == "__main__":
    sample_rate = 16000

    # 1 second of example audio
    audio = np.zeros(sample_rate)

    frames = frame_audio(
        audio,
        sample_rate,
        frame_ms=30,
        hop_ms=10
    )

    print("Audio samples:", len(audio))
    print("Frames shape:", frames.shape)
    print("Samples per frame:", frames.shape[1])
    print("Number of frames:", frames.shape[0])