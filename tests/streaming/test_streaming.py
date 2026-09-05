import asyncio
import json

import pytest
import websockets


SERVER_URI = "ws://127.0.0.1:5050"


def run_async(coro):
    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_server_accepts_valid_audio_format():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

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

        response = await websocket.recv()

        assert response == "READY"


@pytest.mark.asyncio
async def test_server_rejects_invalid_sample_rate():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        start_message = {
            "type": "START",
            "sample_rate": 8000,
            "channels": 1,
            "sample_width": 2,
            "encoding": "PCM16"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        response = await websocket.recv()

        assert response == "ERROR: Invalid sample rate"


@pytest.mark.asyncio
async def test_server_rejects_invalid_channels():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        start_message = {
            "type": "START",
            "sample_rate": 16000,
            "channels": 2,
            "sample_width": 2,
            "encoding": "PCM16"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        response = await websocket.recv()

        assert response == "ERROR: Invalid channel count"


@pytest.mark.asyncio
async def test_server_rejects_invalid_sample_width():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        start_message = {
            "type": "START",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 4,
            "encoding": "PCM16"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        response = await websocket.recv()

        assert response == "ERROR: Invalid sample width"


@pytest.mark.asyncio
async def test_server_rejects_invalid_encoding():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        start_message = {
            "type": "START",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "encoding": "PCM8"
        }

        await websocket.send(
            json.dumps(start_message)
        )

        response = await websocket.recv()

        assert response == "ERROR: Invalid encoding"


@pytest.mark.asyncio
async def test_audio_before_start_is_rejected():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        audio = b"\x00\x00" * 100

        await websocket.send(audio)

        response = await websocket.recv()

        assert response == "ERROR: Send START first"


@pytest.mark.asyncio
async def test_end_before_start_is_rejected():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        await websocket.send("END")

        response = await websocket.recv()

        assert response == "ERROR: No active audio stream"


@pytest.mark.asyncio
async def test_invalid_json_is_rejected():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        await websocket.send(
            "{invalid json}"
        )

        response = await websocket.recv()

        assert response == "ERROR: Invalid JSON message"


@pytest.mark.asyncio
async def test_unknown_message_type_is_rejected():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

        await websocket.send(
            json.dumps({
                "type": "UNKNOWN"
            })
        )

        response = await websocket.recv()

        assert response == "ERROR: Unknown message type"
@pytest.mark.asyncio
async def test_audio_stream_reaches_asr():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

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

        response = await websocket.recv()

        assert response == "READY"

        # Send a small valid PCM16 audio chunk
        audio_chunk = b"\x00\x00" * 3200

        await websocket.send(
            audio_chunk
        )

        await websocket.send("END")

        response = await websocket.recv()

        assert response.startswith("RESULT:")
@pytest.mark.asyncio
async def test_audio_size_limit_is_enforced():

    async with websockets.connect(
        SERVER_URI
    ) as websocket:

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

        response = await websocket.recv()

        assert response == "READY"

        # Send slightly more than the 30-second limit.
        oversized_chunk = (
            b"\x00\x00"
            * (16000 * 30 + 1)
        )

        await websocket.send(
            oversized_chunk
        )

        response = await websocket.recv()

        assert response == (
            "ERROR: Maximum audio duration exceeded"
        )