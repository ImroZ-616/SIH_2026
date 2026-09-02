import asyncio
import websockets
import wave
import os
import sys
import json

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from asr.server import ASR

HOST = "0.0.0.0"
PORT = 5050

AUDIO_FILE = "received_audio.wav"

asr = ASR()


def save_wav(audio_data):
    with wave.open(AUDIO_FILE, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio_data)

    print(f"Audio saved to {AUDIO_FILE}")


async def handle_client(websocket):
    print("Client connected")

    audio_buffer = bytearray()

    try:
        async for message in websocket:

            if isinstance(message, str):

                print("Received message:", message)

                if message == "START":
                    print("Audio stream started")
                    audio_buffer.clear()

                elif message == "END":
                    print("Audio stream ended")

                    if len(audio_buffer) > 0:
                        save_wav(audio_buffer)

                        print("Running ASR...")

                        text = asr.transcribe(AUDIO_FILE)

                        print("Transcription:")
                        print(text)

                        response = {
                            "type": "transcription",
                            "text": text
                        }

                        await websocket.send(
                            json.dumps(response)
                        )

            elif isinstance(message, bytes):

                print(
                    f"Received audio chunk: {len(message)} bytes"
                )

                audio_buffer.extend(message)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():

    print(f"R4 WebSocket server starting on port {PORT}...")

    async with websockets.serve(
        handle_client,
        HOST,
        PORT
    ):

        print("R4 WebSocket server running")
        print("Waiting for client...")

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
