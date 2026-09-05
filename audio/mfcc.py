import numpy as np
from scipy.fftpack import dct

from audio.features import LogMelFeatureExtractor


def extract_mfcc(
    waveform_16k,
    sample_rate=16000,
    frame_ms=30,
    hop_ms=10,
    n_fft=480,
    n_mels=40,
    n_mfcc=13
):
    """
    Extract MFCC features from a 16 kHz audio waveform.

    Parameters
    ----------
    waveform_16k : array-like
        Mono audio waveform sampled at 16 kHz.

    sample_rate : int
        Audio sample rate. Default: 16000 Hz.

    frame_ms : int
        Frame length in milliseconds. Default: 30 ms.

    hop_ms : int
        Hop length in milliseconds. Default: 10 ms.

    n_fft : int
        FFT size. Default: 480.

    n_mels : int
        Number of Mel filterbank bands. Default: 40.

    n_mfcc : int
        Number of MFCC coefficients. Default: 13.

    Returns
    -------
    np.ndarray
        MFCC feature matrix with shape:

        (number_of_frames, n_mfcc)

        dtype: float32
    """

    # Step 1: Generate Log-Mel features
    extractor = LogMelFeatureExtractor(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_fft=n_fft,
        n_mels=n_mels
    )

    log_mel = extractor.extract(waveform_16k)

    # Handle empty input
    if log_mel.shape[0] == 0:
        return np.empty((0, n_mfcc), dtype=np.float32)

    # Step 2: Apply DCT along the Mel-band axis
    mfcc = dct(
        log_mel,
        type=2,
        axis=1,
        norm="ortho"
    )

    # Step 3: Keep the first n_mfcc coefficients
    mfcc = mfcc[:, :n_mfcc]

    # Step 4: Ensure output is float32
    return mfcc.astype(np.float32)