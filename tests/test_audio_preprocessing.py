import numpy as np

from src.audio.audio_preprocessing import (
    SAMPLE_RATE,
    FRAME_LENGTH,
    HOP_LENGTH,
    standardize_audio,
    frame_audio,
    apply_hann_window
)


def test_audio_standardization():

    # Create 1 second of floating-point audio
    audio = np.random.uniform(
        -1,
        1,
        SAMPLE_RATE
    ).astype(np.float32)

    result = standardize_audio(audio)

    assert result.dtype == np.int16
    assert len(result) == SAMPLE_RATE


def test_frame_audio():

    # Create 1 second of silence
    audio = np.zeros(
        SAMPLE_RATE,
        dtype=np.int16
    )

    frames = frame_audio(audio)

    # Each frame should contain 480 samples
    assert frames.shape[1] == FRAME_LENGTH

    assert FRAME_LENGTH == 480
    assert HOP_LENGTH == 160


def test_hann_window():

    audio = np.zeros(
        SAMPLE_RATE,
        dtype=np.int16
    )

    frames = frame_audio(audio)

    windowed = apply_hann_window(frames)

    # Windowing must not change the shape
    assert windowed.shape == frames.shape
