# 🌐 Member 4 — Networking + ASR — Post-Week-1 Roadmap

**Role:** Audio Streaming / Cloud ASR Engineer
**Final competency target:** *"How quickly does audio reach ASR after the wake word ends, and where exactly is the latency coming from?"*

---

## 🟢 Level 1 — Networking Fundamentals
- [ ] IP
- [ ] TCP
- [ ] UDP
- [ ] HTTP
- [ ] HTTPS
- [ ] WebSocket
- [ ] Client/server architecture
- [ ] Ports
- [ ] DNS
- [ ] Latency
- [ ] Bandwidth
- [ ] Packet loss
- [ ] Jitter

## 🟢 Level 2 — Audio Streaming
- [ ] PCM streaming
- [ ] Chunking
- [ ] Buffering
- [ ] Packetization
- [ ] Streaming audio concepts
- [ ] Audio codecs
- [ ] Compression
- [ ] Bitrate
- [ ] Understand: Audio → chunks → network → server

## 🟢 Level 3 — WebSockets
- [ ] WebSocket client
- [ ] WebSocket server
- [ ] Connection lifecycle
- [ ] Reconnect logic
- [ ] Heartbeat
- [ ] Binary frames
- [ ] Streaming chunks over WebSocket

## 🟢 Level 4 — ASR
- [ ] Automatic Speech Recognition overview
- [ ] Speech-to-text
- [ ] Streaming ASR
- [ ] Batch ASR
- [ ] CTC (Connectionist Temporal Classification)
- [ ] Whisper architecture (high level)
- [ ] Vosk / Whisper / faster-whisper comparison
- [ ] (No need to train an ASR model yourself)

## 🟢 Level 5 — ASR Server
- [ ] Python server basics
- [ ] FastAPI
- [ ] WebSockets in FastAPI
- [ ] Async programming
- [ ] Audio ingestion
- [ ] ASR inference integration
- [ ] Build pipeline: ESP32 → WebSocket → FastAPI → ASR → Text

## 🟢 Level 6 — Latency Engineering
- [ ] End-to-end latency
- [ ] Network latency
- [ ] Processing latency
- [ ] Queue latency
- [ ] Buffering latency
- [ ] Connection setup latency
- [ ] Measure total latency: T = T_KWS + T_buffer + T_network + T_ASR

## 🟡 Level 7 — Advanced
- [ ] UDP/RTP
- [ ] Opus codec
- [ ] WebRTC
- [ ] Adaptive bitrate
- [ ] Packet-loss recovery
- [ ] Streaming ASR optimization

---
### 🎯 Checkpoint
Can you fully answer: *"How quickly does audio reach ASR after the wake word ends, and where exactly is the latency coming from?"*
