import asyncio
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
from streaming.client.client import StreamingClient


SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
NETWORK_CHUNK_SIZE = 3200


async def main():

    print("=== EdgeWake R6 Wake → Network Latency Test ===")

    controller = WakeController(
        buffer_seconds=BUFFER_SECONDS,
        sample_rate=SAMPLE_RATE
    )

    client = StreamingClient()

    try:

        # --------------------------------------------------
        # 1. Fill pre-wake buffer
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
        # 2. Connect to WebSocket
        # --------------------------------------------------

        print("\n[2] Connecting to R4 server...")

        await client.connect()

        print("[OK] Connected")

        # --------------------------------------------------
        # 3. Trigger mock KWS
        # --------------------------------------------------

        print("\n[3] Triggering mock KWS detection...")

        wake_time = time.perf_counter()

        buffered_audio = (
            controller.handle_kws_result(True)
        )

        print(
            "[OK] Buffered audio:",
            len(buffered_audio),
            "samples"
        )

        # --------------------------------------------------
        # 4. Start WebSocket stream
        # --------------------------------------------------

        await client.start()

        # --------------------------------------------------
        # 5. Send first audio chunk
        # --------------------------------------------------

        buffered_bytes = buffered_audio.tobytes()

        first_chunk = buffered_bytes[
            :NETWORK_CHUNK_SIZE
        ]

        await client.send_audio(first_chunk)

        first_send_time = time.perf_counter()

        # --------------------------------------------------
        # 6. Calculate latency
        # --------------------------------------------------

        latency_ms = (
            first_send_time - wake_time
        ) * 1000

        print("\n=== LATENCY RESULT ===")

        print(
            "Wake → first network audio:",
            f"{latency_ms:.3f} ms"
        )

        print(
            "Streaming state:",
            controller.streaming
        )

        # --------------------------------------------------
        # 7. Validation
        # --------------------------------------------------

        print("\n=== VALIDATION ===")

        if controller.streaming:
            print("[PASS] Streaming activated")
        else:
            print("[FAIL] Streaming not activated")

        if len(first_chunk) == NETWORK_CHUNK_SIZE:
            print("[PASS] First audio chunk sent")
        else:
            print("[FAIL] Incorrect first chunk size")

        print(
            "\nThis measurement includes:"
        )

        print(
            "KWS event → R6 buffer handoff → "
            "WebSocket send"
        )

        # --------------------------------------------------
        # 8. End stream
        # --------------------------------------------------

        await client.stop()

    finally:

        await client.close()


if __name__ == "__main__":
    asyncio.run(main())