import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.wake_controller import WakeController


SAMPLE_RATE = 16000

controller = WakeController(
    buffer_seconds=1,
    sample_rate=SAMPLE_RATE
)


print("=== EdgeWake R6 Wake Controller Test ===")


# --------------------------------------------------
# 1. Simulate 1 second of audio BEFORE wake word
# --------------------------------------------------

print("\n[1] Sending pre-wake audio...")

pre_wake_audio = np.arange(
    SAMPLE_RATE,
    dtype=np.int16
)

controller.process_audio(pre_wake_audio)

print("Ring buffer samples:", len(controller.ring_buffer))


# --------------------------------------------------
# 2. Simulate KWS detecting wake word
# --------------------------------------------------

print("\n[2] Simulating KWS wake event...")

buffered_audio = controller.handle_kws_result(True)

print(
    "Buffered audio samples:",
    len(buffered_audio)
)


# --------------------------------------------------
# 3. Simulate audio AFTER wake word
# --------------------------------------------------

print("\n[3] Sending post-wake audio...")

post_wake_audio = np.arange(
    SAMPLE_RATE,
    SAMPLE_RATE * 2,
    dtype=np.int16
)

stream_audio = controller.process_audio(
    post_wake_audio
)

print(
    "Post-wake streaming samples:",
    len(stream_audio)
)


# --------------------------------------------------
# 4. Stop streaming
# --------------------------------------------------

print("\n[4] Stopping stream...")

controller.stop_streaming()

print(
    "Streaming state:",
    controller.streaming
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

print("\n=== VALIDATION ===")

assert len(buffered_audio) == SAMPLE_RATE

assert len(stream_audio) == SAMPLE_RATE

assert controller.streaming is False

print("[PASS] Pre-wake audio buffered")
print("[PASS] Wake event detected")
print("[PASS] Buffered audio released")
print("[PASS] Post-wake audio streamed")
print("[PASS] Streaming stopped")
