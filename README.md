# EdgeWake

### Ultra-Low-Latency & Efficient Voice Activation for Edge Devices

> An open-source TinyML-based keyword spotting system that performs wake-word detection locally on a low-power edge device and streams subsequent audio to a remote ASR server with minimal latency and bandwidth.

**Status:** 🟡 Prototype / Active Development

---

## 📊 Current Performance

| Metric                  | Current |
|--------------------------|--------:|
| KWS accuracy              |     TBD |
| False activations/hour    |     TBD |
| Model size                |     TBD |
| RAM usage                 |     TBD |
| CPU utilization           |     TBD |
| KWS latency                |     TBD |
| Wake → ASR latency        |     TBD |
| Power                     |     TBD |

This table is our live scoreboard — it will be filled in as benchmarks are run and updated continuously as the system improves.

---

## 🛰️ Problem Statement

```text
Problem Statement:
Low Latency and Efficient Voice Activator for Edge Devices

Organization:
Indian Space Research Organisation (ISRO)

Domain:
TinyML / Edge AI / Speech Processing / IoT
```

### The problem

Traditional voice assistants often rely on continuously sending audio to cloud servers. This causes:

- 🌐 Network dependency
- ⏱️ Higher latency
- 🔋 Higher energy consumption
- 💰 Increased bandwidth/cloud costs
- 🔐 Privacy concerns

### Our approach

```text
Continuous audio
       ↓
Local TinyML KWS
       ↓
Wake word detected?
       ↓
      YES
       ↓
Stream subsequent audio
       ↓
Remote ASR
```

---

## 🎯 Project Objectives

- [ ] Build a custom keyword spotting model
- [ ] Run the model locally on a low-power edge device
- [ ] Minimize RAM and Flash usage
- [ ] Minimize idle CPU utilization
- [ ] Achieve high keyword detection accuracy
- [ ] Minimize false activations
- [ ] Detect wake word with low latency
- [ ] Immediately stream subsequent audio to ASR
- [ ] Maintain a short pre/post-trigger audio buffer
- [ ] Benchmark the complete system

---

## 🏗️ System Architecture

```text
                   🎤 MICROPHONE
                         │
                         ▼
                  Audio Capture
                         │
                         ▼
                 ┌──────────────┐
                 │ DSP Pipeline │
                 │              │
                 │ PCM          │
                 │ Framing      │
                 │ MFCC/Log-Mel │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ TinyML KWS   │
                 │ INT8 CNN     │
                 └──────┬───────┘
                        │
                 Wake detected?
                    /        \
                  NO          YES
                  │            │
                  ▼            ▼
               Discard     Ring Buffer
                               │
                               ▼
                         Audio Streaming
                               │
                               ▼
                         WebSocket/TCP
                               │
                               ▼
                          ASR Server
                               │
                               ▼
                           Text Output
```

---

## 🧩 Core Components

| Component        | Responsibility            | Technology       |
|-------------------|----------------------------|-------------------|
| 🎤 Audio Capture  | Capture microphone audio   | I2S / PCM         |
| 🎧 DSP            | Feature extraction         | MFCC / Log-Mel    |
| 🧠 KWS            | Detect custom keyword      | CNN / TinyML      |
| 🤏 Edge Runtime   | Local inference             | TFLite Micro      |
| ⚡ MCU             | Run system                  | ESP32-S3          |
| 🔄 Buffer          | Preserve recent audio       | Ring buffer       |
| 🌐 Streaming       | Transfer audio               | WebSocket         |
| 🗣️ ASR            | Speech → text                | Open-source ASR   |
| 📊 Benchmark        | Measure performance          | Python            |

*Technologies not yet integrated are marked "planned" in `docs/architecture.md`, not implied here as done.*

---

## 🧠 Why TinyML?

### Cloud KWS

```text
Microphone → Network → Cloud → KWS
```

Problems: latency, bandwidth, privacy, network dependency.

### Our architecture

```text
Microphone → TinyML → Wake detected → Network → ASR
```

**Key principle:** Only transmit audio after local wake-word detection.

---

## 🔑 Custom Keyword

```text
Current keyword: TBD
```

To be documented once finalized:

- Why this keyword was chosen
- Number of syllables
- Phonetic characteristics
- Similar/confusing words
- Training samples, speakers, accents, noise conditions

**Status: TBD** — numbers below are placeholders until the dataset is built.

```text
Positive samples: TBD
Negative samples: TBD
Speakers:         TBD
Noise conditions: TBD
```

---

## 📊 Dataset

### Sources

- Google Speech Commands
- Custom recordings
- Synthetic speech
- Background noise datasets

```text
dataset/
├── positive/     # target keyword
├── negative/     # similar-sounding words
├── unknown/      # normal speech
├── silence/      # no speech
└── noise/        # background noise
```

---

## 🧪 Data Augmentation

- Noise injection
- Volume variation
- Time shift
- Time stretch
- Pitch variation
- Reverberation
- Background speech
- Microphone distance variation

> Real-world wake-word detection must work under environmental variation rather than only clean recordings.

---

## 🧠 Model Architecture

```text
Input
 ↓
Log-Mel Spectrogram
 ↓
Conv2D
 ↓
ReLU
 ↓
Depthwise Conv
 ↓
Pooling
 ↓
Dense
 ↓
Softmax
```

| Parameter          | Value |
|---------------------|------:|
| Input shape          |   TBD |
| Number of parameters |   TBD |
| Model size            |   TBD |
| Number of layers      |   TBD |
| Activation functions  |   TBD |
| Quantization           |   TBD |
| Training epochs        |   TBD |
| Learning rate           |   TBD |

---

## 🔢 Quantization

```text
FP32 → INT8 Quantization → Smaller model → Lower RAM → Faster inference
```

| Model | Size | Accuracy | Latency |
|-------|-----:|---------:|--------:|
| FP32  |  TBD |      TBD |     TBD |
| INT8  |  TBD |      TBD |     TBD |

---

## 🤏 Edge Device

```text
Board:            ESP32-S3
Microphone:       TBD
RAM:              TBD
Flash:            TBD
Communication:    Wi-Fi
Audio interface:  I2S
```

Also to document: firmware framework, SDK, compiler, clock frequency, power mode.

---

## 🎧 Audio Pipeline

```text
Microphone
 ↓
I2S
 ↓
16 kHz PCM
 ↓
Frame buffer
 ↓
Windowing
 ↓
FFT
 ↓
Mel Filterbank
 ↓
Log
 ↓
KWS
```

| Parameter    |        Value |
|--------------|-------------:|
| Sample rate   |       16 kHz |
| Channels      |         Mono |
| Bit depth     |       16-bit |
| Frame size    |          TBD |
| Hop size      |          TBD |
| Feature type  | Log-Mel/MFCC |

---

## 🔄 Ring Buffer

> The system maintains a short circular audio buffer so that audio immediately preceding and following wake-word detection is not lost.

```text
Continuous audio
──────────────────────────►

       Ring Buffer
    ┌───────────────┐
    │ recent audio  │
    └───────┬───────┘
            │
      Wake detected
            │
            ▼
    Flush buffer + stream
```

```text
Buffer duration: TBD ms
```

---

## 🌐 Streaming Architecture

```text
ESP32
  │
  │ WebSocket
  ▼
Backend
  │
  ▼
ASR Engine
  │
  ▼
Text
```

To document: protocol, audio format, chunk size, sample rate, server, reconnection strategy, timeout handling.

---

## 🗣️ ASR

**Engine:** TBD (candidates: faster-whisper, Vosk)

### Why this ASR?

- Open source
- Local/server deployment
- Streaming capability
- Supported language
- Resource requirements

*Real-time performance claims will be added only once measured.*

---

## ⚡ Performance Metrics

### KWS metrics
```text
True Positive Rate
False Positive Rate
False Accept Rate
False Reject Rate
Precision
Recall
F1
False activations/hour
```

### Edge metrics
```text
Model size
RAM
Flash
CPU utilization
Inference latency
Power consumption
Energy/inference
```

### System metrics
```text
Keyword detection latency
Wake-to-stream latency
Keyword-end → ASR-receive latency
Network latency
ASR latency
End-to-end latency
```

---

## 🏆 Benchmark Table

| Metric            | Baseline | Optimized | Target |
|--------------------|---------:|----------:|-------:|
| Model size          |        — |         — |      — |
| RAM                  |        — |         — |      — |
| CPU                   |        — |         — |      — |
| TPR                    |        — |         — |      — |
| FAR/hour                |        — |         — |      — |
| KWS latency               |        — |         — |      — |
| Wake→ASR latency          |        — |         — |      — |
| Power                      |        — |         — |      — |

---

## 🧪 Testing

### Environments
- Quiet room
- Fan noise
- Traffic
- Music
- Multiple speakers
- Echo
- Distance
- Different microphones
- Different speakers
- Different accents

### Test cases
```text
TC01 — Exact keyword
TC02 — Similar sounding word
TC03 — Keyword in noise
TC04 — Whispered keyword
TC05 — Far-field keyword
TC06 — Continuous speech
TC07 — Multiple speakers
TC08 — Network disconnected
TC09 — ASR unavailable
TC10 — Buffer overflow
```

---

## 🛡️ Failure Handling

**Network fails**
```text
KWS still works locally → audio isn't transmitted → retry connection
```

**ASR server fails**
```text
Detect failure → stop streaming → return to listening
```

**False / missed activation** — recorded in the benchmark log and analyzed for cause.

---

## 📁 Repository Structure

```text
edge-voice-activator/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── architecture.md
│   ├── requirements.md
│   ├── dataset.md
│   ├── model.md
│   ├── benchmarking.md
│   └── deployment.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── audio/
│   ├── capture.py
│   ├── preprocessing.py
│   ├── mfcc.py
│   └── features.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── augment.py
│
├── models/
│   ├── checkpoints/
│   ├── tflite/
│   └── quantized/
│
├── embedded/
│   ├── firmware/
│   ├── kws/
│   └── audio/
│
├── streaming/
│   ├── client/
│   └── server/
│
├── asr/
│   └── server.py
│
├── benchmarks/
│   ├── accuracy/
│   ├── latency/
│   ├── memory/
│   └── power/
│
└── tests/
    ├── audio/
    ├── kws/
    ├── streaming/
    └── integration/
```

---

## 🛠️ Installation

```bash
git clone <repository>
cd edge-voice-activator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

```bash
python audio/test_microphone.py
```

```bash
python kws/inference.py
```

```bash
python streaming/server.py
```

---

## 🚀 Quick Start

```text
1. Clone repository
2. Install dependencies
3. Connect microphone
4. Run KWS
5. Say custom keyword
6. Start ASR server
7. Connect edge device
8. Speak command
```

---

## 🔧 Hardware Setup

```text
ESP32-S3
│
├── 3.3V → MIC VCC
├── GND  → MIC GND
├── GPIO → I2S CLK
├── GPIO → I2S WS
└── GPIO → I2S DATA
```

*Pin numbers above are placeholders — update once finalized.*

---

## 👥 Team

| Member   | Role           | Responsibilities       |
|----------|----------------|--------------------------|
| Member 1 | Audio DSP       | MFCC, preprocessing      |
| Member 2 | ML/KWS          | Dataset, model            |
| Member 3 | TinyML           | ESP32, TFLite Micro       |
| Member 4 | Networking/ASR    | Streaming, ASR             |
| Member 5 | Benchmarking       | Performance                 |
| Member 6 | Integration          | Architecture, testing        |

---

## 📈 Development Roadmap

- [x] Audio capture
- [x] Feature extraction
- [ ] KWS baseline
- [ ] INT8 quantization
- [ ] ESP32 deployment
- [ ] Ring buffer
- [ ] Streaming
- [ ] ASR integration
- [ ] Benchmarking
- [ ] Final optimization

*Update this checklist continuously as the project progresses.*

---

## 🧪 Experiment Log

Every experiment gets logged under `experiments/`, e.g.:

```text
experiments/
├── exp001_baseline.md
├── exp002_int8.md
├── exp003_noise_aug.md
├── exp004_threshold.md
└── exp005_small_cnn.md
```

Each file should follow this format:

```text
Experiment:   <name>
Hypothesis:   <what you expect to happen and why>
Baseline:     <what you're comparing against>
Change:       <what was changed>
Accuracy:     <result>
RAM:          <result>
Flash:        <result>
Latency:      <result>
Conclusion:   <what you learned>
```

---

## 🧠 Design Decisions (Architecture Decision Records)

**Why ESP32-S3?**
> Chosen because it provides a practical balance between MCU-level resource constraints, audio interfaces, and computational capability.

**Why INT8?**
> Reduces model memory and computation while remaining suitable for MCU inference.

**Why WebSocket?**
> Provides persistent bidirectional communication suitable for low-latency streaming.

**Why a ring buffer?**
> Prevents loss of audio immediately surrounding wake-word detection.

---

## 🔐 Security & Privacy

- No continuous audio transmission
- Local wake-word processing
- Only post-trigger audio transmitted
- Secure transport *(implement before claiming)*
- No unnecessary audio retention
- Authentication for ASR server *(implement before claiming)*
- Data deletion policy *(define before claiming)*

---

## 📚 References

**Official / technical**
- TensorFlow Lite Micro
- ESP32-S3 documentation
- MLPerf Tiny
- Speech Commands Dataset

**Research**
- Keyword spotting papers
- TinyML papers
- Edge AI papers
- Speech processing papers

**Open-source projects**
- openWakeWord
- TFLite Micro examples
- Relevant KWS implementations

*(Add direct links to each resource as they're finalized.)*

---

## 📜 License

```text
License: Apache-2.0
```

Per the ISRO PS requirement (**open source only**), licenses for datasets, pretrained components, libraries, and ASR engines used will also be documented here as they're added.

---

## ⚠️ Known Limitations

```text
- Limited speaker diversity
- Limited far-field testing
- Wi-Fi only
- ASR currently server-based
- Power measurement not yet implemented
```

---

## 🏁 Current Status

**Status: 🟡 Prototype / Active Development**

Will move to **🟢 End-to-End Prototype** once `ESP32 → KWS → Stream → ASR` is working end to end.
