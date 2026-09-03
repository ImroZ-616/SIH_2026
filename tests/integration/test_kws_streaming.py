import asyncio
import sys
import os
import sounddevice as sd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.wake_controller import WakeController
from streaming.client.client import StreamingClient


SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
COMMAND_SECONDS = 3
CHUNK_SIZE = 1600
NETWORK_CHUNK_SIZE = 3200


async def main():

    print("=== EdgeWake R6 KWS → Streaming → ASR Test ===")

    controller = WakeController(
        buffer_seconds=BUFFER_SECONDS,
        sample_rate=SAMPLE_RATE
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
        # 2. Feed continuous audio into WakeController
        # --------------------------------------------------

        print("\n[2] Feeding audio into WakeController...")

        wake_index = None
        buffered_audio = None

        for audio_index in range(
            0,
            len(audio),
            CHUNK_SIZE
        ):

            chunk = audio[
                audio_index:
                audio_index + CHUNK_SIZE
            ]

            controller.process_audio(chunk)

            # Simulate KWS detection after 1 second
            if (
                audio_index >= SAMPLE_RATE
                and not controller.streaming
            ):

                print("\n[MOCK KWS] Wake word detected!")

                wake_index = audio_index

                buffered_audio = (
                    controller.wake_detected()
                )

                print(
                    "[OK] Buffered audio:",
                    len(buffered_audio),
                    "samples"
                )

                break

        if wake_index is None:
            raise RuntimeError(
                "Mock KWS did not detect wake event"
            )

        # --------------------------------------------------
        # 3. Connect to ASR server
        # --------------------------------------------------

        print("\n[3] Connecting to ASR server...")

        await client.connect()

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

        print(
            "[OK] Post-wake audio sent"
        )

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

        print("\n=== FINAL TRANSCRIPTION ===")
        print(text)

        # --------------------------------------------------
        # 9. Stop wake controller
        # --------------------------------------------------

        controller.stop_streaming()

        print(
            "\n=== R6 INTEGRATION TEST COMPLETE ==="
        )

    finally:

        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
