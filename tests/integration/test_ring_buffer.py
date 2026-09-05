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

# SLIDING_WINDOW
# import numpy as np


# class SlidingWindow:
#     """
#     Generates overlapping audio windows from a continuous
#     stream of PCM audio.

#     Example:
#         Sample rate = 16000 Hz
#         Window = 1 second
#         Hop = 100 ms

#         Window 1: samples 0      - 15999
#         Window 2: samples 1600   - 17599
#         Window 3: samples 3200   - 19199
#     """

#     def __init__(
#         self,
#         window_size: int,
#         hop_size: int
#     ):
#         self.window_size = window_size
#         self.hop_size = hop_size

#         self.total_samples = 0
#         self.next_window_end = window_size

#     def add(self, samples: np.ndarray):
#         """
#         Inform the sliding window that new samples
#         have arrived.

#         Returns True if a new window is ready.
#         """

#         samples = np.asarray(samples)

#         self.total_samples += len(samples)

#         return self.total_samples >= self.next_window_end

#     def window_ready(self):
#         """
#         Check whether a new complete window is ready.
#         """

#         return (
#             self.total_samples >=
#             self.next_window_end
#         )

#     def advance(self):
#         """
#         Move the sliding window forward by hop_size.
#         """

#         if not self.window_ready():
#             return False

#         self.next_window_end += self.hop_size

#         return True
