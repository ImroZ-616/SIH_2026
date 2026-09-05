import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3
CHANNELS = 1

print("Recording...")
print("Speak clearly into the microphone.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()

audio = audio.flatten()

np.save("tests/mic_audio.npy", audio)

print("Recording finished.")
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Saved to: tests/mic_audio.npy")