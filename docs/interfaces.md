# EdgeWake System Integration Interfaces

**Project:** SIH 26172 — Low Latency and Efficient Voice Activator for Edge Devices
**System:** EdgeWake / ASTRAEDGE
**Owner:** R6 — Integration/System Engineer

---

## 1. Purpose

This document defines the interfaces between the major components of the
EdgeWake system.

The objective is to allow R1, R2, R3, R4 and R5 components to be integrated
without changing their internal implementations.

The interfaces described here define the current integration baseline.
Parameters are considered final only after verification against the actual
implementation, trained model and target hardware.

---

## 2. End-to-End Interface Flow

```text
Microphone
    │
    ▼
Audio Capture
    │
    │ PCM audio
    ▼
Audio Preprocessing
    │
    ▼
KWS Feature Extraction
    │
    ▼
KWS Detector
    │
    │ detected = True
    ▼
WakeController
    │
    ├──────────────► Ring Buffer
    │
    ▼
Streaming Client
    │
    │ WebSocket
    ▼
Streaming Server
    │
    ▼
ASR Engine
    │
    ▼
Transcription
