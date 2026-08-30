import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

# Allow importing ml_kws/src/audio.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ml_kws" / "src"))

from audio import preprocess_audio


SAMPLE_RATE = 16000
DURATION = 3
CHANNELS = 1


print("=== EdgeWake R6 Audio Integration Test ===")

print("\n[1] Recording from microphone...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
)

sd.wait()

print("[OK] Recording finished")
print("Raw shape:", audio.shape)
print("Raw dtype:", audio.dtype)

# Convert int16 PCM to float32 in approximately [-1, 1]
audio_float = audio.flatten().astype(np.float32) / 32768.0

print("\n[2] Sending PCM audio to R2 preprocessing...")

processed = preprocess_audio(
    audio_float,
    orig_sr=SAMPLE_RATE,
)

print("[OK] Preprocessing completed")

print("\n[3] Verifying standardized output...")

print("Processed shape:", processed.shape)
print("Processed dtype:", processed.dtype)
print("Minimum amplitude:", np.min(processed))
print("Maximum amplitude:", np.max(processed))

assert processed.shape == (16000,)
assert processed.dtype == np.float32
assert np.all(np.isfinite(processed))
assert np.max(processed) <= 1.0
assert np.min(processed) >= -1.0

print("\n=== PASS: Microphone → R2 Preprocessing ===")
