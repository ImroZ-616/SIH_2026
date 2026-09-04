import asyncio
import json
import wave

import websockets


SERVER_URI = "ws://127.0.0.1:5050"

AUDIO_FILE = "/Users/shaswatnain/Downloads/test.wav"


async def main():

    print(
        "Connecting to R4 server..."
    )

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        print(
            "Connected to R4 server"
        )

        # -------------------------
        # START + AUDIO METADATA
        # -------------------------

        start_message = {

            "type": "START",

            "sample_rate": 16000,

            "channels": 1,

            "sample_width": 2,

            "encoding": "PCM16"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        print(
            "START sent with audio metadata"
        )

        # Wait for server confirmation

        ready = await websocket.recv()

        print(
            "Server response:",
            ready
        )

        if ready != "READY":

            print(
                "Server did not accept "
                "the audio stream"
            )

            return

        # -------------------------
        # SEND AUDIO
        # -------------------------

        with wave.open(
            AUDIO_FILE,
            "rb"
        ) as audio:

            print(
                "Sending real audio..."
            )

            while True:

                # 3200 samples
                # = 200 ms at 16 kHz

                chunk = audio.readframes(
                    3200
                )

                if not chunk:

                    break

                await websocket.send(
                    chunk
                )

                print(
                    f"Sent audio chunk: "
                    f"{len(chunk)} bytes"
                )

        # -------------------------
        # END
        # -------------------------

        await websocket.send(
            "END"
        )

        print(
            "END sent"
        )

        # -------------------------
        # RECEIVE ASR RESULT
        # -------------------------

        print(
            "Waiting for ASR result..."
        )

        result = await websocket.recv()

        print(
            "Server response:"
        )

        print(result)

    print(
        "Connection closed"
    )


if __name__ == "__main__":

    asyncio.run(main())