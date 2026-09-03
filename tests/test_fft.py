import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 16000
FREQUENCY = 440
FRAME_MS = 30


# Number of samples in one frame
N = int(SAMPLE_RATE * FRAME_MS / 1000)

# Generate time values
time = np.arange(N) / SAMPLE_RATE

# Generate 440 Hz sine wave
signal = np.sin(2 * np.pi * FREQUENCY * time)

# Apply Hann window
window = np.hanning(N)
windowed_signal = signal * window

# Perform FFT
fft_result = np.fft.rfft(windowed_signal)

# Magnitude of FFT
magnitude = np.abs(fft_result)

# Frequency corresponding to each FFT bin
frequencies = np.fft.rfftfreq(
    N,
    d=1 / SAMPLE_RATE
)

# Find strongest frequency
peak_index = np.argmax(magnitude)
peak_frequency = frequencies[peak_index]

print("Sample rate:", SAMPLE_RATE)
print("Frame samples:", N)
print("Expected frequency:", FREQUENCY, "Hz")
print("Detected peak:", peak_frequency, "Hz")

# Plot frequency spectrum
plt.figure(figsize=(12, 4))

plt.plot(frequencies, magnitude)

plt.title("FFT Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)

plt.grid(True)
plt.tight_layout()

plt.show()