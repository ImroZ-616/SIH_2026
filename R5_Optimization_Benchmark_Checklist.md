# ⚡ R5 — Optimization/Benchmark Engineer — Week 1 Checklist

**Main responsibility:** Latency, CPU, RAM, false activations

---

## Day 1 — Shared Foundation (everyone)
- [x ] Can explain what KWS (Keyword Spotting) is
- [x ] Can explain what TinyML is
- [x ] Can explain why we don't just send everything to the cloud (latency, bandwidth, privacy, power, cost)
- [x ] Understand the full pipeline: Mic → PCM → Feature extraction → KWS → Wake event → Buffer → Network → ASR

## Day 1 — Role-specific
- [x ] Accuracy
- [x ] Precision
- [x ] Recall
- [ ] F1 score
- [x ] False Acceptance Rate (FAR)
- [x ] False Rejection Rate (FRR)
- [ ] Inference latency
- [x ] CPU utilization
- [x ] RAM
- [x ] Flash
- [x ] Power
- [ ] Research: MLPerf Tiny — Keyword Spotting
- [x ] Understand why accuracy alone is not enough (accuracy + latency + memory + energy)
- [ ] Create initial benchmark spreadsheet (Accuracy, False accepts/hour, Inference latency, RAM, Flash, CPU, Power)
- [ ] **Output:** `benchmark.csv`

## Day 2 — Latency Measurement
- [ ] Learn to measure `start_time` → `end_time` → `latency = end - start`

## Day 3 — Feature Extraction Benchmarking
- [ ] Benchmark MFCC/feature extraction time
- [ ] Understand: KWS latency ≠ model inference time alone

## Day 4 — Evaluation Setup
- [ ] Set up evaluation scripts for the trained model

## Day 5 — Live Microphone Milestone
- [ ] Measure latency/performance of the live laptop KWS demo

## Day 6 — False Activation Testing (you lead today)
- [ ] Measure True Positives, False Positives, False Negatives against tricky negative audio
- [ ] Calculate Precision = TP / (TP + FP)
- [ ] Calculate Recall = TP / (TP + FN)

## Day 7 — Integration Day
- [ ] Baseline metrics deliverable finalized (accuracy, precision, recall, false activation rate)

---
### End-of-day rule
Each day, answer: **"What did I build today that another teammate can actually use?"**
