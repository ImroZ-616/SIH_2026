import asyncio
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
DURATION = 3
CHANNELS = 1
SERVER_URL = "ws://127.0.0.1:5050"


async def main():
    print("=== EdgeWake R6 Streaming → ASR Test ===")

    print("\n[1] Recording microphone...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )
    sd.wait()

    print("[OK] Recording finished")
    print("Audio shape:", audio.shape)
    print("Audio dtype:", audio.dtype)

    audio_bytes = audio.tobytes()

    print("\n[2] Connecting to R4 WebSocket server...")

    async with websockets.connect(SERVER_URL) as websocket:
        print("[OK] Connected")

        print("\n[3] Sending START...")
        await websocket.send("START")

        chunk_size = 3200

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await websocket.send(chunk)

        print("[OK] Audio chunks sent")

        print("\n[4] Sending END...")
        await websocket.send("END")

        print("[OK] Stream finished")


if __name__ == "__main__":
    asyncio.run(main())
