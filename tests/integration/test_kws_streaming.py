import asyncio
import sys
import os
import time

import sounddevice as sd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.wake_controller import WakeController
from streaming.client.client import StreamingClient
from streaming.kws.mock import MockKWS


SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
COMMAND_SECONDS = 3
CHUNK_SIZE = 1600
NETWORK_CHUNK_SIZE = 3200


async def main():

    print("=== EdgeWake R6 KWS → Streaming → ASR Latency Test ===")

    controller = WakeController(
        buffer_seconds=BUFFER_SECONDS,
        sample_rate=SAMPLE_RATE
    )

    kws = MockKWS(
        sample_rate=SAMPLE_RATE,
        detection_after_seconds=1.0
    )

    client = StreamingClient()

    try:

        # --------------------------------------------------
        # 1. Record microphone audio
        # --------------------------------------------------

        print("\n[1] Recording microphone...")
        print("Speak a short command.")

        audio = sd.rec(
            int(COMMAND_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        audio = audio.flatten()

        print("[OK] Recording finished")
        print("Samples:", len(audio))
        print(
            "Duration:",
            len(audio) / SAMPLE_RATE,
            "seconds"
        )

        # --------------------------------------------------
        # 2. Feed audio into R6 + KWS
        # --------------------------------------------------

        print(
            "\n[2] Feeding audio into "
            "WakeController + KWS..."
        )

        wake_index = None
        buffered_audio = None

        # Timing measurements
        kws_event_time = None
        wake_handled_time = None
        first_network_audio_time = None
        transcription_time = None

        for audio_index in range(
            0,
            len(audio),
            CHUNK_SIZE
        ):

            chunk = audio[
                audio_index:
                audio_index + CHUNK_SIZE
            ]

            # R6 continuously buffers incoming audio
            controller.process_audio(chunk)

            # KWS independently processes the same chunk
            detected = kws.process(chunk)

            if detected:

                # ------------------------------------------
                # KWS event timestamp
                # ------------------------------------------

                kws_event_time = time.perf_counter()

                print(
                    "\n[OK] KWS detection event received"
                )

                wake_index = audio_index

                # ------------------------------------------
                # R6 wake handling
                # ------------------------------------------

                buffered_audio = (
                    controller.handle_kws_result(True)
                )

                wake_handled_time = time.perf_counter()

                print(
                    "[OK] Buffered audio:",
                    len(buffered_audio),
                    "samples"
                )

                break

        if wake_index is None:
            raise RuntimeError(
                "KWS did not detect wake event"
            )

        # --------------------------------------------------
        # 3. Connect to ASR server
        # --------------------------------------------------

        print("\n[3] Connecting to ASR server...")

        await client.connect()

        print("[OK] Connected")

        # --------------------------------------------------
        # 4. Start streaming
        # --------------------------------------------------

        print("\n[4] Starting stream...")

        await client.start()

        # --------------------------------------------------
        # 5. Send buffered audio
        # --------------------------------------------------

        print("\n[5] Sending buffered audio...")

        buffered_bytes = buffered_audio.tobytes()

        first_audio_sent = False

        for buffer_index in range(
            0,
            len(buffered_bytes),
            NETWORK_CHUNK_SIZE
        ):

            chunk = buffered_bytes[
                buffer_index:
                buffer_index + NETWORK_CHUNK_SIZE
            ]

            await client.send_audio(chunk)

            # Timestamp the first network audio send
            if not first_audio_sent:

                first_network_audio_time = (
                    time.perf_counter()
                )

                first_audio_sent = True

        print("[OK] Buffered audio sent")

        # --------------------------------------------------
        # 6. Send post-wake audio
        # --------------------------------------------------

        print("\n[6] Sending post-wake audio...")

        post_wake_audio = audio[
            wake_index + CHUNK_SIZE:
        ]

        post_wake_bytes = post_wake_audio.tobytes()

        for post_index in range(
            0,
            len(post_wake_bytes),
            NETWORK_CHUNK_SIZE
        ):

            chunk = post_wake_bytes[
                post_index:
                post_index + NETWORK_CHUNK_SIZE
            ]

            await client.send_audio(chunk)

        print("[OK] Post-wake audio sent")

        # --------------------------------------------------
        # 7. End stream
        # --------------------------------------------------

        print("\n[7] Ending stream...")

        await client.stop()

        # --------------------------------------------------
        # 8. Receive ASR result
        # --------------------------------------------------

        print(
            "\n[8] Waiting for ASR transcription..."
        )

        text = await client.receive_transcription()

        transcription_time = time.perf_counter()

        # --------------------------------------------------
        # 9. Calculate latency measurements
        # --------------------------------------------------

        wake_to_buffer_ms = (
            wake_handled_time - kws_event_time
        ) * 1000

        wake_to_network_ms = (
            first_network_audio_time - kws_event_time
        ) * 1000

        wake_to_transcription_ms = (
            transcription_time - kws_event_time
        ) * 1000

        # --------------------------------------------------
        # 10. Results
        # --------------------------------------------------

        print("\n========================================")
        print("        R6 LATENCY BENCHMARK")
        print("========================================")

        print("\n[RESULT 1] KWS → R6 wake handling")
        print(
            f"{wake_to_buffer_ms:.3f} ms"
        )

        print("\n[RESULT 2] KWS → first network audio")
        print(
            f"{wake_to_network_ms:.3f} ms"
        )

        print("\n[RESULT 3] KWS → final transcription")
        print(
            f"{wake_to_transcription_ms:.3f} ms"
        )

        print("\n========================================")
        print("        FINAL TRANSCRIPTION")
        print("========================================")
        print(text)

        print("\n========================================")
        print("        INTERPRETATION")
        print("========================================")

        print(
            "KWS → R6 handling:"
            " local wake-event processing"
        )

        print(
            "KWS → first network audio:"
            " R6 wake handling + WebSocket transmission"
        )

        print(
            "KWS → transcription:"
            " complete current batch-ASR pipeline"
        )

        print(
            "\nNOTE:"
            " MockKWS does not measure real KWS inference latency."
        )

        print(
            "The current ASR server performs transcription "
            "after END, so the final value is NOT true "
            "continuous streaming-ASR latency."
        )

        # --------------------------------------------------
        # 11. Stop R6 streaming state
        # --------------------------------------------------

        controller.stop_streaming()

        print(
            "\n=== R6 LATENCY TEST COMPLETE ==="
        )

    finally:

        await client.close()


if __name__ == "__main__":
    asyncio.run(main())