# R3 ↔ R4 ESP32 Audio Integration

## Purpose

This document defines the interface between:

R3 — ESP32 / Embedded  
R4 — Networking / ASR

The target pipeline is:

ESP32 Microphone
        ↓
PCM16 Audio
        ↓
WebSocket
        ↓
R4 Python Server
        ↓
faster-whisper
        ↓
Text

---

## R3 Responsibilities

R3 is responsible for:

- Capturing microphone audio on ESP32
- Producing PCM16 audio
- Using 16 kHz sample rate
- Using mono audio
- Buffering audio after a wake event
- Connecting to the R4 WebSocket server
- Sending the required START message
- Sending binary PCM16 audio chunks
- Sending END when the audio segment is complete

R3 does NOT need to implement ASR.

---

## R4 Responsibilities

R4 is responsible for:

- Running the WebSocket server
- Validating audio metadata
- Receiving binary PCM16 audio
- Reconstructing the audio
- Running ASR
- Returning the transcription
- Returning protocol errors when necessary

---

# WebSocket Protocol

## 1. Connect

ESP32 connects to:

ws://<R4_SERVER_IP>:5050

For example:

ws://192.168.1.100:5050

The IP address will depend on the computer running the R4 server.

---

## 2. START Message

After connecting, ESP32 sends:

{
    "type": "START",
    "sample_rate": 16000,
    "channels": 1,
    "sample_width": 2,
    "encoding": "PCM16"
}

This is a WebSocket text message.

---

## 3. READY Response

R4 validates the metadata.

If accepted:

READY

The ESP32 must wait for this response before sending audio.

---

## 4. Audio Streaming

ESP32 sends raw PCM16 audio as WebSocket binary messages.

Required format:

- Sample rate: 16000 Hz
- Channels: 1
- Sample width: 2 bytes
- Encoding: signed PCM16
- Recommended chunk: 3200 samples
- Recommended chunk size: 6400 bytes
- Chunk duration: 200 ms

Example:

Binary PCM16 chunk
        ↓
Binary PCM16 chunk
        ↓
Binary PCM16 chunk
        ↓
...

---

## 5. END Message

After the audio segment is complete, ESP32 sends:

END

This is a WebSocket text message.

R4 then processes the accumulated audio using ASR.

---

## 6. ASR Result

R4 returns:

RESULT:<transcription>

Example:

RESULT:hello this is a test

The ESP32 can then use the returned text for the next stage of the system.

---

# Audio Bandwidth

PCM16 at 16 kHz mono:

16000 samples/sec × 2 bytes/sample

= 32000 bytes/sec

= 32 KB/sec

= 256 kbps

A 5-second recording produces approximately:

16000 × 2 × 5

= 160000 bytes

The actual network bandwidth will be slightly higher because of WebSocket, TCP/IP and Wi-Fi overhead.

---

# Maximum Audio Duration

R4 currently accepts a maximum of:

30 seconds

Maximum raw PCM size:

16000 × 2 × 30

= 960000 bytes

The ESP32 should normally send much shorter audio segments following a wake event.

---

# Important Implementation Notes

1. Audio must be PCM16, not WAV.

2. Do NOT send a WAV header with every audio chunk.

3. START is JSON text.

4. Audio chunks are WebSocket binary messages.

5. END is a text message.

6. ESP32 must wait for READY before sending audio.

7. Audio should be 16 kHz, mono, signed 16-bit PCM.

8. The R4 server handles WAV reconstruction internally.

9. The ESP32 does not need to run Whisper.

10. The ESP32 and computer running R4 must be reachable over the network.

---

# Current R4 Test Setup

The protocol has already been tested using:

- Python audio-file client
- Python live microphone client
- WebSocket server
- faster-whisper ASR

The live microphone pipeline is:

Microphone
    ↓
PCM16
    ↓
WebSocket
    ↓
R4 Server
    ↓
WAV reconstruction
    ↓
faster-whisper
    ↓
Text

---

# R3 Integration Goal

When ESP32 audio capture is ready, replace the Python microphone client with:

ESP32
    ↓
Wi-Fi
    ↓
WebSocket
    ↓
R4 Server

The R4 server should not need to change if the ESP32 follows this protocol.