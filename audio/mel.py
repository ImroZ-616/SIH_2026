import numpy as np
import matplotlib.pyplot as plt


def hz_to_mel(frequency):
    return 2595 * np.log10(1 + frequency / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels=40,
    f_min=0,
    f_max=None
):
    if f_max is None:
        f_max = sample_rate / 2

    # Convert frequency limits to Mel scale
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Create equally spaced Mel points
    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    # Convert Mel points back to Hz
    hz_points = mel_to_hz(mel_points)

    # Convert Hz points to FFT bin numbers
    bin_points = np.floor(
        (n_fft + 1) * hz_points / sample_rate
    ).astype(int)

    # Number of positive-frequency FFT bins
    n_frequency_bins = n_fft // 2 + 1

    filterbank = np.zeros(
        (n_mels, n_frequency_bins)
    )

    # Create triangular filters
    for m in range(1, n_mels + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        # Rising side of triangle
        if center > left:
            for k in range(left, center):
                filterbank[m - 1, k] = (
                    (k - left) /
                    (center - left)
                )

        # Falling side of triangle
        if right > center:
            for k in range(center, right):
                filterbank[m - 1, k] = (
                    (right - k) /
                    (right - center)
                )

    return filterbank


if __name__ == "__main__":

    SAMPLE_RATE = 16000
    N_FFT = 480
    N_MELS = 40

    # Create Mel filter bank
    filterbank = create_mel_filterbank(
        SAMPLE_RATE,
        N_FFT,
        N_MELS
    )

    print("Sample rate:", SAMPLE_RATE)
    print("FFT size:", N_FFT)
    print("Mel bands:", N_MELS)
    print("Filter bank shape:", filterbank.shape)
    print("Maximum value:", filterbank.max())

    # Frequency corresponding to each FFT bin
    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    # Plot all Mel filters
    plt.figure(figsize=(12, 6))

    for i in range(N_MELS):
        plt.plot(
            frequencies,
            filterbank[i]
        )

    plt.title("Mel Filter Bank")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filter Weight")
    plt.xlim(0, 8000)
    plt.grid(True)

    plt.tight_layout()
    plt.show()