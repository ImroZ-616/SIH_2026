import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DURATION = 3
CHANNELS = 1

print("Recording... Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()

print("Recording finished.")

# Convert from shape (samples, 1) to (samples,)
audio = audio.flatten()

# Create time axis
time = np.arange(len(audio)) / SAMPLE_RATE

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, audio)

plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()