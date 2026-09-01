import asyncio
import wave

import websockets


SERVER_URI = "ws://127.0.0.1:5050"
AUDIO_FILE = "/Users/shaswatnain/Downloads/test.wav"


async def main():

    print("Connecting to R4 server...")

    async with websockets.connect(SERVER_URI) as websocket:

        print("Connected to R4 server")

        await websocket.send("START")
        print("START sent")

        with wave.open(AUDIO_FILE, "rb") as audio:

            print("Sending real audio...")

            while True:

                chunk = audio.readframes(3200)

                if not chunk:
                    break

                await websocket.send(chunk)

                print(
                    f"Sent audio chunk: {len(chunk)} bytes"
                )

        await websocket.send("END")
        print("END sent")

        print("Waiting for ASR result...")

        result = await websocket.recv()

        print("Server response:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
