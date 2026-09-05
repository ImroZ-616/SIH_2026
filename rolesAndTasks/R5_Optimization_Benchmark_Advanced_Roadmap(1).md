# ⚡ Member 5 — Optimization + Benchmarking — Post-Week-1 Roadmap

**Role:** Performance Engineer
**Final competency target:** *"Our model is 15% smaller, uses 30% less RAM, reduces false activations by 40%, and reduces wake-to-ASR latency by 25% compared with our baseline."*

---

## 🟢 Level 1 — ML Benchmarking
- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1 score
- [ ] Confusion matrix
- [ ] ROC curve
- [ ] PR curve
- [ ] False Accept Rate
- [ ] False Reject Rate

## 🟢 Level 2 — KWS-Specific Metrics
- [ ] False Accept Rate — how often the system wakes when it shouldn't
- [ ] False Reject Rate — how often it misses the actual keyword
- [ ] Detection latency — how quickly the keyword is detected
- [ ] False activations per hour (critical metric)

## 🟢 Level 3 — Embedded Benchmarking
- [ ] Measure RAM usage
- [ ] Measure Flash usage
- [ ] Measure CPU usage
- [ ] Measure inference time
- [ ] Measure feature extraction time
- [ ] Measure power
- [ ] Measure energy per inference
- [ ] Measure FPS / inference rate

## 🟢 Level 4 — Profiling
- [ ] CPU profiling
- [ ] Memory profiling
- [ ] Timing instrumentation
- [ ] Bottleneck analysis
- [ ] Logging infrastructure
- [ ] Break down total latency: Audio capture → Feature extraction → KWS → Decision → Buffer flush → Network → ASR reception

## 🟢 Level 5 — Experiment Design
- [ ] Controlled experiments
- [ ] Baselines
- [ ] A/B comparisons
- [ ] Reproducibility
- [ ] Statistical significance
- [ ] Confidence intervals

## 🟢 Level 6 — Optimization Techniques
- [ ] Model compression
- [ ] Quantization (applied)
- [ ] Pruning (applied)
- [ ] Architecture optimization
- [ ] Threshold tuning
- [ ] Temporal smoothing
- [ ] VAD gating

## 🟡 Level 7 — Energy
- [ ] Voltage
- [ ] Current
- [ ] Power (P = VI)
- [ ] Energy (E = Pt)
- [ ] Understand "energy per inference" as a metric, not just CPU %

## 🟡 Level 8 — Benchmarking Frameworks
- [ ] Study MLPerf Tiny
- [ ] MLPerf Tiny — Keyword Spotting track
- [ ] MLPerf Tiny — latency methodology
- [ ] MLPerf Tiny — energy methodology
- [ ] MLPerf Tiny — accuracy methodology

---
### 🎯 Checkpoint
Can you produce a results statement like: *"Our model is X% smaller, uses Y% less RAM, reduces false activations by Z%, and reduces wake-to-ASR latency by N% vs baseline"* — backed by real numbers?
