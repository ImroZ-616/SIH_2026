# 🤏 R3 — TinyML/Embedded Engineer — Week 1 Checklist

**Main responsibility:** MCU, TFLite Micro, memory constraints

---

## Day 1 — Shared Foundation (everyone)
- [ ] Can explain what KWS (Keyword Spotting) is
- [ ] Can explain what TinyML is
- [ ] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)
- [ ] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific (don't train models yet)
- [ ] Microcontroller vs Raspberry Pi
- [ ] RAM vs Flash
- [ ] Inference (concept)
- [ ] TensorFlow Lite
- [ ] TensorFlow Lite Micro
- [ ] INT8 quantization
- [ ] Research: ESP32-S3 + TensorFlow Lite Micro
- [ ] Understand: Normal ML (laptop → large model) vs TinyML (MCU → tiny model → limited RAM/CPU)
- [ ] Install Arduino IDE or PlatformIO
- [ ] Install ESP32 board support
- [ ] (Optional) experiment with ESP8266 — but do NOT make it the final target
- [ ] **Output:** working ESP32 → serial monitor program

## Day 2 — MCU Constraints
- [ ] Build a comparison table: ESP8266 vs ESP32 vs ESP32-S3 vs Raspberry Pi (RAM, Flash, suitability)

## Day 3 — Feasibility Research
- [ ] Research whether the chosen MFCC implementation can eventually run efficiently on an MCU

## Day 4 — TFLite Conversion Research
- [ ] Research TensorFlow Lite conversion process
- [ ] Do NOT deploy to hardware yet

## Day 5 — Live Microphone Milestone
- [ ] Observe the laptop-based KWS demo — note what will need to change for MCU deployment later

## Day 6 — False Activation Testing
- [ ] Support the team's negative testing effort where relevant to embedded constraints

## Day 7 — Integration Day
- [ ] Confirm ESP32 → serial monitor pipeline still works
- [ ] Prep notes for Week 2: quantization → TFLite Micro → ESP32-S3 deployment

---
### End-of-day rule
Each day, answer: **"What did I build today that another teammate can actually use?"**
