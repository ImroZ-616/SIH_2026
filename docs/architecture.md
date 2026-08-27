# System Architecture

## High-Level Pipeline

Microphone
    ↓
Audio Capture
    ↓
Preprocessing
    ↓
MFCC / Log-Mel Features
    ↓
TinyML Keyword Spotting
    ↓
Wake Word Detection
    ↓
Ring Buffer
    ↓
Audio Streaming
    ↓
Remote ASR Server
    ↓
Speech-to-Text

## Edge Components

- Microphone
- Audio capture
- Feature extraction
- KWS model
- Ring buffer
- Network client

## Server Components

- Streaming server
- Audio receiver
- ASR engine
- Response handling