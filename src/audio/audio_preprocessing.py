import numpy as np


# =========================
# Audio Configuration
# =========================

SAMPLE_RATE = 16000
CHANNELS = 1

FRAME_LENGTH_MS = 30
HOP_LENGTH_MS = 10

FRAME_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)


def standardize_audio(audio):
    """
    Convert input audio into mono int16 PCM format.

    Parameters
    ----------
    audio : numpy.ndarray
        Input audio samples.

    Returns
    -------
    numpy.ndarray
        Mono int16 audio.
    """

    audio = np.asarray(audio)

    # Convert stereo/multi-channel audio to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize floating-point audio to int16
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    else:
        audio = audio.astype(np.int16)

    return audio


def frame_audio(audio):
    """
    Divide audio into overlapping frames.

    Current configuration:
        Sample rate  : 16 kHz
        Frame length : 30 ms = 480 samples
        Hop length   : 10 ms = 160 samples
    """

    frames = []

    for start in range(
        0,
        len(audio) - FRAME_LENGTH + 1,
        HOP_LENGTH
    ):
        frame = audio[start:start + FRAME_LENGTH]
        frames.append(frame)

    return np.array(frames, dtype=np.int16)


def apply_hann_window(frames):
    """
    Apply Hann window to every audio frame.
    """

    window = np.hanning(FRAME_LENGTH)

    return frames * window
