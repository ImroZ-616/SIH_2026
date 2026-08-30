# 🔧 R6 — Integration/System Engineer — Week 1 Checklist

**Main responsibility:** Architecture, Git, integration, testing

---

## Day 1 — Shared Foundation (everyone)
- [x] Can explain what KWS (Keyword Spotting) is
- [x] Can explain what TinyML is
- [x] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)
- [x] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific
- [x] Git / GitHub workflow
- [x] Python environments
- [x] Project structure best practices
- [x] APIs (concept)
- [x] System architecture (concept)
- [x] requirements.txt / dependency management
- [x] Testing basics
- [x] Create the repository skeleton:
  ```
  edge-voice-activator/
  ├── audio/
  ├── dataset/
  ├── training/
  ├── models/
  ├── kws/
  ├── embedded/
  ├── streaming/
  ├── asr/
  ├── benchmarks/
  ├── tests/
  └── docs/
  ```
- [x] Create `README.md`
- [x] Create `architecture.md`
- [x] Create `requirements.txt`
- [x] **Output:** clean GitHub repo with everyone assigned to their branch/module

## Day 2 — Architecture Diagram
- [x] Create a common team-wide architecture diagram

## Day 3 — Pipeline Integration (team milestone)
- [ ] Integrate R1's audio/MFCC work and R2's feature visualization into one pipeline

## Day 4 — Model + Preprocessing Integration
- [ ] Integrate R2's trained model with R1's preprocessing pipeline

## Day 5 — Live Microphone Milestone
- [ ] Ensure everyone can actually run the live mic → MFCC → CNN → detection demo end-to-end

## Day 6 — False Activation Testing Support
- [ ] Coordinate collection of `negative_test/` samples across the team

## Day 7 — Integration Day (you lead)
- [ ] All modules integrated: Mic → Audio preprocessing → MFCC → KWS → Wake detected
- [ ] Separate pipeline confirmed: Audio → ASR server → Text
- [ ] Confirm all 7 end-of-week deliverables are present in the repo

---
### End-of-day rule
Each day, answer: **"What did I build today that another teammate can actually use?"**
