import json
import websockets


class StreamingClient:

    def __init__(self, server_url="ws://127.0.0.1:5050"):
        self.server_url = server_url
        self.websocket = None

    async def connect(self):
        print("[STREAM] Connecting to ASR server...")

        self.websocket = await websockets.connect(
            self.server_url
        )

        print("[STREAM] Connected")

    async def start(self):
        if self.websocket is None:
            raise RuntimeError("Client is not connected")

        await self.websocket.send("START")

        print("[STREAM] Stream started")

    async def send_audio(self, audio):
        if self.websocket is None:
            raise RuntimeError("Client is not connected")

        await self.websocket.send(audio)

    async def stop(self):
        if self.websocket is None:
            raise RuntimeError("Client is not connected")

        await self.websocket.send("END")

        print("[STREAM] Stream ended")

    async def receive_transcription(self):
        if self.websocket is None:
            raise RuntimeError("Client is not connected")

        response = await self.websocket.recv()

        data = json.loads(response)

        if data.get("type") == "transcription":
            return data.get("text", "")

        return None

    async def close(self):
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

            print("[STREAM] Connection closed")
