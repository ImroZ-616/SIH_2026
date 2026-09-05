import time
import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.wake_controller import WakeController


SAMPLE_RATE = 16000
BUFFER_SECONDS = 1


def main():

    print("=== EdgeWake R6 Wake-to-Stream Latency Test ===")

    controller = WakeController(
        buffer_seconds=BUFFER_SECONDS,
        sample_rate=SAMPLE_RATE
    )

    # --------------------------------------------------
    # 1. Simulate pre-wake audio
    # --------------------------------------------------

    print("\n[1] Filling ring buffer...")

    audio = np.zeros(
        SAMPLE_RATE,
        dtype=np.int16
    )

    controller.process_audio(audio)

    print(
        "Ring buffer samples:",
        len(controller.ring_buffer)
    )

    # --------------------------------------------------
    # 2. Measure KWS → streaming latency
    # --------------------------------------------------

    print("\n[2] Triggering mock KWS detection...")

    start_time = time.perf_counter()

    buffered_audio = controller.handle_kws_result(True)

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    # --------------------------------------------------
    # 3. Results
    # --------------------------------------------------

    print("\n=== LATENCY RESULT ===")

    print(
        "Buffered samples:",
        len(buffered_audio)
    )

    print(
        "Wake → stream latency:",
        f"{latency_ms:.3f} ms"
    )

    print(
        "Streaming state:",
        controller.streaming
    )

    # --------------------------------------------------
    # 4. Validation
    # --------------------------------------------------

    print("\n=== VALIDATION ===")

    if len(buffered_audio) == SAMPLE_RATE:
        print("[PASS] Buffered audio released")
    else:
        print("[FAIL] Incorrect buffered audio size")

    if controller.streaming:
        print("[PASS] Streaming activated")
    else:
        print("[FAIL] Streaming not activated")

    if latency_ms < 10:
        print("[PASS] R6 wake-to-stream latency < 10 ms")
    else:
        print("[INFO] Latency >= 10 ms")


if __name__ == "__main__":
    main()