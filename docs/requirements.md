# System Requirements

**Project:** SIH 26172 — Low Latency and Efficient Voice Activator for Edge Devices
**System:** EdgeWake / ASTRAEDGE
**Owner:** R6 — Integration/System Engineer

---

## 1. Problem Statement

Develop a highly accurate, ultra-low-power custom Keyword Spotting (KWS)
system for edge devices that minimizes false activations and detection
latency while efficiently streaming subsequent user speech to a remote ASR
server.

---

## 2. Core Objective

The system shall:

1. continuously listen for a custom wake word locally;
2. perform KWS inference on the edge device;
3. minimize memory, flash, CPU and power usage;
4. minimize false activations and missed activations;
5. transition rapidly from wake detection to audio streaming;
6. stream required speech audio to the remote ASR server; and
7. return the resulting transcription.

---

## 3. Functional Requirements

## REQ-01 — Custom Keyword Detection

**Requirement:**
The system shall detect the team's custom wake keyword.

**Owner:** R2

**Integration responsibility:** R6 shall accept the KWS wake event through the
defined `KWSDetector` interface.

**Current status:** 🟡 Partial

**Evidence:**

- `KWSDetector` interface implemented.
- `MockKWS` implemented for deterministic integration testing.
- Real trained KWS model is not yet connected to the R6 integration path.

**Final validation:**

- Keyword detection accuracy
- False rejection rate
- False accept rate
- False activations per hour

---

## REQ-02 — Local Inference

**Requirement:**
Wake-word detection shall occur locally on the edge device without requiring
continuous cloud inference.

**Owner:** R2/R3

**Integration responsibility:** R6 shall integrate the local KWS decision
with the system state flow.

**Current status:** 🟡 Integration-ready

**Evidence:**

- `KWSDetector` abstraction separates KWS inference from system integration.
- R6 integration does not require cloud communication for the wake decision.

**Final validation:**

- Verify KWS inference executes on target hardware.
- Verify normal listening does not continuously transmit audio to the ASR
  server.

---

## REQ-03 — Low RAM Usage

**Requirement:**
The system shall operate within the available RAM budget of the target edge
device.

**Owner:** R3/R5

**Integration responsibility:** R6 shall ensure that buffering and system
integration do not introduce uncontrolled memory growth.

**Current status:** 🟡 Partially implemented

**Current baseline:**

- Ring buffer uses fixed-size storage.
- Current 1-second buffer at 16 kHz `int16` requires approximately 32 KB for
  raw audio storage.
- The ring buffer overwrites old samples when full.

**Final validation:**

Measure peak RAM usage on the target hardware.

---

## REQ-04 — Low Flash Usage

**Requirement:**
The complete edge deployment shall fit within the available flash memory of
the target device.

**Owner:** R3/R5

**Integration responsibility:** R6 shall track the size of integrated
components and verify the final deployment package.

**Current status:** 🔴 Pending hardware integration

**Final validation:**

Record:

- KWS model size
- application binary size
- filesystem/data size
- total flash usage
- remaining flash capacity

---

## REQ-05 — Low CPU Usage

**Requirement:**
Continuous wake-word detection shall operate within the CPU budget of the
target edge device.

**Owner:** R3/R5

**Integration responsibility:** R6 shall ensure the complete state flow does
not introduce unnecessary processing.

**Current status:** 🔴 Pending hardware integration

**Final validation:**

Measure:

- KWS inference time
- CPU utilization during LISTEN
- CPU utilization during STREAM
- CPU utilization during idle/STOP
- peak CPU utilization

---

## REQ-06 — Low False Activation Rate

**Requirement:**
The system shall minimize false wake activations during normal background
audio.

**Owner:** R2/R5

**Integration responsibility:** R6 shall correctly handle the wake event and
prevent repeated wake events from creating multiple simultaneous streaming
sessions.

**Current status:** 🟡 Integration-ready

**Evidence:**

- `WakeController.handle_kws_result()` ignores additional wake events while
  streaming.
- Integration tests verify the streaming state transition.

**Final validation:**

Measure:

- False Activations per Hour (FAH)
- False Accept Rate (FAR)
- performance with background/noise audio
- performance with negative test data

---

## REQ-07 — Fast ASR Streaming

**Requirement:**
After wake detection, required speech audio shall be transferred to the
remote ASR server with low wake-to-stream latency.

**Owner:** R4

**Integration responsibility:** R6 shall connect the wake event, buffered
audio and subsequent audio chunks to the streaming interface.

**Current status:** 🟡 Partial

**Evidence:**

- WakeController provides buffered audio after detection.
- Post-wake audio is forwarded while streaming.
- WebSocket streaming integration is implemented.
- Preliminary integration tests measure wake-to-network latency.

**Current limitation:**

The current ASR server performs batch transcription after receiving `END`.
True continuous/partial ASR is not yet implemented.

**Final validation:**

Measure:

```text
Wake detection
      ↓
First network audio
      ↓
ASR processing
      ↓
Final transcription
