# R6 Integration Status

**Project:** SIH 26172 — EdgeWake  
**Owner:** R6 — Integration/System Engineer

## Component Status

| Component | Owner | Status |
|---|---|---|
| Audio capture | R1 | 🔴 Pending |
| Audio preprocessing | R1/R2 | 🟢 Implemented |
| MFCC / Log-Mel | R1/R2 | 🔴 Pending |
| KWS model | R2 | 🔴 Pending |
| KWS inference | R2 | 🔴 Pending |
| TinyML / ESP32 | R3 | 🔴 Pending |
| Ring buffer | R3/R6 | 🔴 Pending |
| Streaming | R4 | 🔴 Pending |
| ASR | R4 | 🟡 In progress |
| Benchmarking | R5 | 🔴 Pending |
| Integration tests | R6 | 🔴 Pending |

## Verified R2 Preprocessing

Implementation:

`ml_kws/src/audio.py`

Current standardized output:

- Sample rate: 16 kHz
- Channels: Mono
- Duration: 1 second
- Samples: 16,000
- Datatype: float32
- Output shape: `(16000,)`

## Integration Pipeline

```text
Microphone
    ↓
Audio Capture
    ↓
Preprocessing
    ↓
MFCC / Log-Mel
    ↓
KWS Model
    ↓
Wake Detection
    ↓
Ring Buffer
    ↓
Streaming
    ↓
ASR
    ↓
Text
