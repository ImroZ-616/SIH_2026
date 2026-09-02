import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3

print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording finished.")
print("Shape:", audio.shape)
print("First 10 samples:")
print(audio[:10])