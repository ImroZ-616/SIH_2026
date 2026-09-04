import numpy as np

from audio.mfcc import extract_mfcc


def test_mfcc_shape():
    audio = np.zeros(16000, dtype=np.int16)

    mfcc = extract_mfcc(audio)

    assert mfcc.shape == (98, 13)


def test_mfcc_dtype():
    audio = np.zeros(16000, dtype=np.int16)

    mfcc = extract_mfcc(audio)

    assert mfcc.dtype == np.float32


def test_mfcc_finite():
    audio = np.random.default_rng(42).normal(
        0, 1000, 16000
    ).astype(np.int16)

    mfcc = extract_mfcc(audio)

    assert np.all(np.isfinite(mfcc))


def test_mfcc_empty_audio():
    audio = np.array([], dtype=np.int16)

    mfcc = extract_mfcc(audio)

    assert mfcc.shape == (0, 13)
    assert mfcc.dtype == np.float32