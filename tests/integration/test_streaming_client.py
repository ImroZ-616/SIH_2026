import asyncio
import sys
import os
import sounddevice as sd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.client.client import StreamingClient


SAMPLE_RATE = 16000
DURATION = 3


async def main():

    print("=== EdgeWake R6 Streaming Client Test ===")

    # Record microphone audio
    print("\n[1] Recording microphone...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    print("[OK] Recording finished")

    audio_bytes = audio.tobytes()

    # Create client
    client = StreamingClient()

    try:
        # Connect
        print("\n[2] Connecting...")

        await client.connect()

        # Start stream
        print("\n[3] Starting stream...")

        await client.start()

        # Send audio
        print("\n[4] Sending audio...")

        chunk_size = 3200

        for i in range(0, len(audio_bytes), chunk_size):

            chunk = audio_bytes[i:i + chunk_size]

            await client.send_audio(chunk)

        print("[OK] Audio sent")

        # Stop stream
        print("\n[5] Ending stream...")

        await client.stop()

        # Receive ASR result
        print("\n[6] Waiting for transcription...")

        text = await client.receive_transcription()

        print("\n=== FINAL TRANSCRIPTION ===")
        print(text)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
