# 🤏 Member 3 — TinyML + Embedded Systems — Post-Week-1 Roadmap

**Role:** Edge AI / MCU Engineer
**Final competency target:** *"Why does our model fit into this MCU, and exactly how much RAM, Flash and CPU does it consume?"*

---

## 🟢 Level 1 — Embedded Fundamentals
- [ ] Microcontrollers
- [ ] CPUs
- [ ] GPIO
- [ ] ADC
- [ ] DAC
- [ ] I2C
- [ ] SPI
- [ ] UART
- [ ] Interrupts
- [ ] Timers
- [ ] DMA

## 🟢 Level 2 — MCU Architecture
- [ ] CPU
- [ ] SRAM
- [ ] Flash
- [ ] ROM
- [ ] Stack
- [ ] Heap
- [ ] Memory alignment
- [ ] Peripherals
- [ ] Clock frequency
- [ ] Understand: Flash = model + program; RAM = tensors + audio buffers + stack + runtime

## 🟢 Level 3 — ESP32 (if ESP32-S3 is chosen)
- [ ] ESP32-S3 architecture
- [ ] Arduino framework
- [ ] ESP-IDF
- [ ] PlatformIO
- [ ] I2S
- [ ] Digital microphones
- [ ] Wi-Fi
- [ ] FreeRTOS basics
- [ ] Interrupts (on ESP32)
- [ ] DMA (on ESP32)

## 🟢 Level 4 — TinyML
- [ ] Edge AI concepts
- [ ] TinyML overview
- [ ] On-device inference
- [ ] Model optimization
- [ ] TensorFlow Lite
- [ ] TensorFlow Lite Micro
- [ ] CMSIS-NN

## 🟢 Level 5 — Quantization (critical)
- [ ] FP32
- [ ] FP16
- [ ] INT8
- [ ] Quantization overview
- [ ] Quantization-aware training
- [ ] Post-training quantization
- [ ] Dynamic quantization
- [ ] Calibration
- [ ] Understand: FP32 model → Quantization → INT8 model → smaller/faster/lower memory

## 🟢 Level 6 — TinyML Memory Management
- [ ] Tensor arena
- [ ] Static allocation
- [ ] Memory planning
- [ ] Buffer reuse
- [ ] Flash optimization
- [ ] RAM optimization

## 🟡 Level 7 — Model Optimization
- [ ] Pruning
- [ ] Knowledge distillation
- [ ] Depthwise separable CNN
- [ ] Operator fusion
- [ ] SIMD
- [ ] CMSIS-NN (applied)
- [ ] Hardware acceleration

## 🟡 Level 8 — Power Optimization
- [ ] Sleep modes
- [ ] Clock scaling
- [ ] Duty cycling
- [ ] CPU utilization
- [ ] Energy per inference
- [ ] Current measurement

---
### 🎯 Checkpoint
Can you fully answer: *"Why does our model fit into this MCU, and exactly how much RAM, Flash and CPU does it consume?"*
