# EdgeWake System Integration Interfaces

**Project:** SIH 26172 — Low Latency and Efficient Voice Activator for Edge Devices

**Owner:** R6 — Integration/System Engineer

---

## Purpose

This document defines the interfaces between the major components of the
EdgeWake system.

The goal is to ensure that independently developed modules can be connected
without ambiguity.

No parameter should be considered final unless it is verified against the
actual implementation and training configuration.

---

# 1. End-to-End Pipeline

```text
Microphone
    ↓
Audio Capture
    ↓
PCM Audio
    ↓
Audio Preprocessing
    ↓
Feature Extraction
    ↓
KWS Model
    ↓
Wake Decision
    ↓
Ring Buffer
    ↓
Audio Streaming
    ↓
ASR Server
    ↓
Text
