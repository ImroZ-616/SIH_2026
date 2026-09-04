import asyncio
import json
import os
import sys
import wave

import websockets


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from asr.server import ASR


HOST = "0.0.0.0"
PORT = 5050

AUDIO_FILE = "received_audio.wav"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
EXPECTED_ENCODING = "PCM16"

MAX_AUDIO_SECONDS = 30

MAX_AUDIO_BYTES = (
    EXPECTED_SAMPLE_RATE
    * EXPECTED_CHANNELS
    * EXPECTED_SAMPLE_WIDTH
    * MAX_AUDIO_SECONDS
)


# Load ASR model once when server starts
asr = ASR()


def save_wav(
    audio_data,
    sample_rate,
    channels,
    sample_width
):

    with wave.open(AUDIO_FILE, "wb") as wav:

        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)

        wav.writeframes(audio_data)

    print(
        f"Audio saved to {AUDIO_FILE}"
    )


def validate_audio_format(data):

    sample_rate = data.get(
        "sample_rate"
    )

    channels = data.get(
        "channels"
    )

    sample_width = data.get(
        "sample_width"
    )

    encoding = data.get(
        "encoding"
    )

    if sample_rate != EXPECTED_SAMPLE_RATE:

        return False, "Invalid sample rate"

    if channels != EXPECTED_CHANNELS:

        return False, "Invalid channel count"

    if sample_width != EXPECTED_SAMPLE_WIDTH:

        return False, "Invalid sample width"

    if encoding != EXPECTED_ENCODING:

        return False, "Invalid encoding"

    return True, "Audio format valid"


async def handle_client(websocket):

    print("Client connected")

    audio_buffer = bytearray()

    sample_rate = EXPECTED_SAMPLE_RATE
    channels = EXPECTED_CHANNELS
    sample_width = EXPECTED_SAMPLE_WIDTH

    streaming = False

    try:

        async for message in websocket:

            # -------------------------
            # TEXT MESSAGE
            # -------------------------

            if isinstance(message, str):

                print(
                    "Received message:",
                    message
                )

                # END message
                if message == "END":

                    if not streaming:

                        await websocket.send(
                            "ERROR: No active audio stream"
                        )

                        continue

                    print(
                        "Audio stream ended"
                    )

                    streaming = False

                    if len(audio_buffer) == 0:

                        await websocket.send(
                            "ERROR: No audio received"
                        )

                        continue

                    save_wav(
                        audio_buffer,
                        sample_rate,
                        channels,
                        sample_width
                    )

                    print(
                        "Running ASR..."
                    )

                    text = await asyncio.to_thread(
                        asr.transcribe,
                        AUDIO_FILE
                    )

                    print(
                        "Transcription:"
                    )

                    print(text)

                    await websocket.send(
                        "RESULT:" + text
                    )

                    continue

                # Try to parse JSON
                try:

                    data = json.loads(
                        message
                    )

                except json.JSONDecodeError:

                    await websocket.send(
                        "ERROR: Invalid JSON message"
                    )

                    continue

                # -------------------------
                # START message
                # -------------------------

                if data.get("type") == "START":

                    print(
                        "Audio stream started"
                    )

                    valid, error_message = (
                        validate_audio_format(
                            data
                        )
                    )

                    if not valid:

                        print(
                            "Audio format rejected:",
                            error_message
                        )

                        await websocket.send(
                            f"ERROR: {error_message}"
                        )

                        continue

                    sample_rate = data[
                        "sample_rate"
                    ]

                    channels = data[
                        "channels"
                    ]

                    sample_width = data[
                        "sample_width"
                    ]

                    audio_buffer.clear()

                    streaming = True

                    print(
                        f"Audio format accepted: "
                        f"{sample_rate} Hz, "
                        f"{channels} channel(s), "
                        f"{sample_width} byte samples, "
                        f"{data['encoding']}"
                    )

                    await websocket.send(
                        "READY"
                    )

                else:

                    await websocket.send(
                        "ERROR: Unknown message type"
                    )

            # -------------------------
            # BINARY AUDIO
            # -------------------------

            elif isinstance(message, bytes):

                if not streaming:

                    print(
                        "Audio received without "
                        "an active stream"
                    )

                    await websocket.send(
                        "ERROR: Send START first"
                    )

                    continue

                # Check maximum audio size
                if (
                    len(audio_buffer)
                    + len(message)
                    > MAX_AUDIO_BYTES
                ):

                    print(
                        "Audio stream rejected: "
                        "maximum size exceeded"
                    )

                    await websocket.send(
                        "ERROR: Maximum audio "
                        "duration exceeded"
                    )

                    audio_buffer.clear()

                    streaming = False

                    continue

                audio_buffer.extend(
                    message
                )

                print(
                    f"Received audio chunk: "
                    f"{len(message)} bytes "
                    f"(total: "
                    f"{len(audio_buffer)} bytes)"
                )

    except websockets.exceptions.ConnectionClosed:

        print(
            "Client disconnected"
        )


async def main():

    print(
        f"R4 WebSocket server starting "
        f"on port {PORT}..."
    )

    print(
        f"Maximum audio duration: "
        f"{MAX_AUDIO_SECONDS} seconds"
    )

    print(
        f"Maximum audio size: "
        f"{MAX_AUDIO_BYTES} bytes"
    )

    async with websockets.serve(
        handle_client,
        HOST,
        PORT
    ):

        print(
            "R4 WebSocket server running"
        )

        print(
            "Waiting for client..."
        )

        await asyncio.Future()


if __name__ == "__main__":

    asyncio.run(main())