import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 1
CHANNELS = 1

print("=== PCM Audio Experiment ===")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()

print("\nAudio information:")
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Number of samples:", audio.shape[0])
print("Channels:", audio.shape[1])
print("Minimum:", audio.min())
print("Maximum:", audio.max())

audio_bytes = audio.tobytes()

print("\nRaw PCM information:")
print("Number of bytes:", len(audio_bytes))
print("Expected bytes:", SAMPLE_RATE * DURATION * 2)
print("First 20 bytes:", audio_bytes[:20])
