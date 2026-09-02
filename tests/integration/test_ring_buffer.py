import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.buffer import AudioRingBuffer


SAMPLE_RATE = 16000
BUFFER_SECONDS = 1

MAX_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS


print("=== EdgeWake R6 Ring Buffer Test ===")

buffer = AudioRingBuffer(MAX_SAMPLES)

print("\nBuffer capacity:")
print("Samples:", MAX_SAMPLES)
print("Duration:", MAX_SAMPLES / SAMPLE_RATE, "seconds")


# Simulate 2 seconds of microphone PCM audio
audio = np.arange(SAMPLE_RATE * 2, dtype=np.int16)

print("\nGenerated audio:")
print("Samples:", len(audio))
print("Dtype:", audio.dtype)


# Add audio in chunks
chunk_size = 1600

for i in range(0, len(audio), chunk_size):

    chunk = audio[i:i + chunk_size]

    buffer.add(chunk)


print("\nRing buffer after 2 seconds:")
print("Samples stored:", len(buffer))

stored_audio = buffer.get_audio()

print("First sample:", stored_audio[0])
print("Last sample:", stored_audio[-1])

print("\nExpected:")
print("Samples stored:", MAX_SAMPLES)
print("First sample:", SAMPLE_RATE)
print("Last sample:", SAMPLE_RATE * 2 - 1)
