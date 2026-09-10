# TinyML Memory Measurement

## Tensor Arena Memory — EdgeWake KWS

**Project:** SIH 26172 — Low Latency and Efficient Voice Activator for Edge Devices
**Model:** EdgeWake Keyword Spotting (KWS)
**Target Keyword:** `ASTRA`
**Runtime:** TensorFlow Lite for Microcontrollers (TFLM)
**Validation Platform:** Native C++ host test harness

---

## 1. Objective

Measure the TensorFlow Lite Micro tensor arena memory required by the EdgeWake keyword spotting model during inference.

The measurement helps determine the model's RAM footprint and provides a basis for selecting an appropriate tensor arena size for the final embedded deployment.

---

## 2. Configuration

The current TFLM configuration uses:

| Parameter                |        Value |
| ------------------------ | -----------: |
| Configured Tensor Arena  |        64 KB |
| Configured Arena (bytes) | 65,536 bytes |
| Measured Arena Usage     | 27,292 bytes |
| Measured Usage           |    ~26.65 KB |
| Unused Arena             | 38,244 bytes |
| Unused Memory            |    ~37.35 KB |

---

## 3. Validation Result

The native C++ TFLM validation harness successfully initialized the model using the configured **64 KB tensor arena**.

```text
AllocateTensors() Status   : [PASS] (kTfLiteOk)
Configured Tensor Arena     : 65536 bytes (64 KB)
Measured Arena Used Bytes   : 27292 bytes (~26.65 KB)
```

Therefore, the current model requires approximately **27.3 KB of tensor arena memory** according to the TFLM allocator measurement.

### Memory utilization

The measured utilization of the configured arena is approximately:

**27,292 / 65,536 × 100 ≈ 41.7%**

Thus, approximately **58.3% of the configured tensor arena remains unused**.

---

## 4. Other Validation Results

The memory measurement was obtained during the same genuine TFLM validation run.

| Validation                     | Result         |
| ------------------------------ | -------------- |
| Model payload and linkage      | PASS           |
| TFLM `AllocateTensors()`       | PASS           |
| MFCC C++ parity                | PASS           |
| Fused quantization             | PASS           |
| End-to-end predictions         | 6/6 PASS       |
| Custom INT8 engine cross-check | 100% agreement |

One golden-vector test showed a maximum output difference of only 1 quantization unit in two elements. This did not affect the end-to-end classification results.

---

## 5. Interpretation

The **64 KB tensor arena is currently a safe working configuration**, while the measured TFLM arena usage is approximately **27.3 KB**.

The measured value should be interpreted as the amount of arena memory used by the current TFLM allocation process. It should **not automatically be treated as the exact minimum possible arena size**, because allocator alignment and internal allocation requirements can affect the minimum successful arena size.

A smaller arena can therefore be investigated later if RAM optimization becomes necessary.

For the current implementation, retaining a **64 KB tensor arena** provides a substantial safety margin of approximately **37.35 KB**.

---

## 6. Current Conclusion

> **EdgeWake's TFLM inference requires approximately 27.3 KB of tensor arena memory under the current model and runtime configuration. A 64 KB tensor arena successfully initializes and executes the model, leaving approximately 37.35 KB of unused arena capacity.**

This result demonstrates that the current KWS model has a relatively small tensor-arena RAM footprint and is suitable for further TinyML/edge-device optimization and deployment work.

---

## 7. Future Optimization

If memory becomes a constraint during embedded deployment, the tensor arena can be reduced experimentally to determine the smallest size that still allows:

1. `AllocateTensors()` to succeed.
2. All required operators to initialize correctly.
3. End-to-end inference to produce the expected classifications.
4. Runtime stability across representative inputs.

The current **64 KB configuration should remain the baseline** until such an optimization experiment is performed.
