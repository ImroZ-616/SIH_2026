# R4 WebSocket Audio Server

## Overview

R4 provides a Python WebSocket server that receives PCM16 audio
and sends the audio to the ASR engine.

The pipeline is:

ESP32 / Python Client
        ↓
    WebSocket
        ↓
    R4 Server
        ↓
    WAV reconstruction
        ↓
    faster-whisper
        ↓
    Transcribed text

## Server

Start the server with:

```bash
cd ~/SIH_2026
source .venv/bin/activate
python streaming/server/server.py