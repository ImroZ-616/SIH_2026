# 🌐 R4 — Networking/ASR Engineer — Week 1 Checklist

**Main responsibility:** Streaming + ASR server

---

## Day 1 — Shared Foundation (everyone)
- [ ] Can explain what KWS (Keyword Spotting) is
- [ ] Can explain what TinyML is
- [ ] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)
- [ ] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific
- [ ] HTTP basics
- [ ] WebSocket basics
- [ ] TCP basics
- [ ] Streaming audio concepts
- [ ] ASR (Automatic Speech Recognition) concept
- [ ] Understand target architecture: ESP32 → WebSocket → Python server → ASR → Text
- [ ] Build a simple Python client → WebSocket → Python server (no ESP32 yet)
- [ ] Research open-source ASR options: Whisper / faster-whisper, Vosk, others
- [ ] **Output:** working pipeline — audio file → server → ASR → text

## Day 2 — Audio Formats
- [ ] Study PCM
- [ ] Study WAV
- [ ] Study raw audio streaming
- [ ] Understand why continuous raw audio transmission is expensive (bandwidth)

## Day 3 — MCU-to-Server Research
- [ ] Research how the audio format used by the MCU will reach your server

## Day 4 — ASR Server Build
- [ ] Build the ASR server independently (not yet connected to KWS pipeline)

## Day 5 — Live Microphone Milestone
- [ ] Support the team's laptop KWS demo where networking is relevant

## Day 6 — False Activation Testing
- [ ] Support test data collection if it touches streaming/ASR behavior

## Day 7 — Integration Day
- [ ] Deliver standalone pipeline: Audio → ASR server → Text (separate from KWS, MCU connection not required yet)

---
### End-of-day rule
Each day, answer: **"What did I build today that another teammate can actually use?"**
