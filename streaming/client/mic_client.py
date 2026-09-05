import asyncio
import json
import queue

import sounddevice as sd
import websockets


SERVER_URI = "ws://127.0.0.1:5050"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

CHUNK_SAMPLES = 3200  # 200 ms at 16 kHz
RECORDING_SECONDS = 5


async def main():

    print("Connecting to R4 server...")

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        print("Connected to R4 server")

        # -------------------------
        # START + AUDIO METADATA
        # -------------------------

        start_message = {
            "type": "START",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "encoding": "PCM16"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        print("START sent")

        # Wait for server confirmation

        ready = await websocket.recv()

        print("Server response:", ready)

        if ready != "READY":

            print(
                "Server did not accept "
                "the audio stream"
            )

            return

        # -------------------------
        # AUDIO QUEUE
        # -------------------------

        audio_queue = queue.Queue()



        # -------------------------
        # MICROPHONE CALLBACK
        # -------------------------

        def audio_callback(
            indata,
            frames,
            time,
            status
        ):

            if status:

                print(
                    "Audio status:",
                    status
                )

            # Put raw PCM16 audio into queue

            audio_queue.put(
                indata.copy().tobytes()
            )

        # -------------------------
        # SEND AUDIO FUNCTION
        # -------------------------

        async def send_audio():

            while True:

                # Get next microphone chunk
                # without blocking asyncio

                chunk = await asyncio.to_thread(
                    audio_queue.get
                )

                if chunk is None:

                    break

                await websocket.send(
                    chunk
                )

                print(
                    f"Sent live audio chunk: "
                    f"{len(chunk)} bytes"
                )

        # -------------------------
        # START MICROPHONE
        # -------------------------

        print()
        print("Microphone is ready.")
        print("Speak now...")
        print(
            f"Recording for "
            f"{RECORDING_SECONDS} seconds."
        )
        print()

        sender_task = asyncio.create_task(
            send_audio()
        )

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        ):

            await asyncio.sleep(
                RECORDING_SECONDS
            )

        print()
        print("Recording finished.")

        # Tell sender that no more
        # microphone data will arrive

        audio_queue.put(None)

        # Wait until every queued
        # audio chunk has been sent

        await sender_task

        # -------------------------
        # END STREAM
        # -------------------------

        await websocket.send("END")

        print("END sent")

        # -------------------------
        # RECEIVE ASR RESULT
        # -------------------------

        print("Waiting for ASR result...")

        result = await websocket.recv()

        print()
        print("ASR RESULT:")
        print(result)

    print()
    print("Connection closed.")


if __name__ == "__main__":

    asyncio.run(main())