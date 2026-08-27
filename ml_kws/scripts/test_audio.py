"""
SIH 26172 - Phase 3 Audio Preprocessing Test & Validation Suite
Validates the complete audio loading, resampling, mono conversion,
normalization, and padding/trimming pipeline on real dataset samples and edge cases.
"""

import sys
import random
from pathlib import Path
import numpy as np

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    DATASET_DIR,
    UNKNOWN_DIR,
    SILENCE_DIR,
    NEGATIVE_TEST_DIR,
    TARGET_SAMPLE_RATE,
    TARGET_DURATION,
    TARGET_SAMPLES,
)
from audio import (
    load_audio,
    to_mono,
    resample_audio,
    normalize_audio,
    pad_or_trim,
    preprocess_audio,
)


def run_tests():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Audio Preprocessing Validation Suite (Phase 3)")
    print(f"Standard Target: {TARGET_SAMPLE_RATE} Hz | {TARGET_DURATION}s ({TARGET_SAMPLES} samples) | Mono float32")
    print("=" * 80)

    # Gather test files from UNKNOWN, SILENCE, and NEGATIVE_TEST
    test_files = []

    unknown_files = list(UNKNOWN_DIR.rglob("*.wav"))
    silence_files = list(SILENCE_DIR.rglob("*.wav"))
    neg_files = list(NEGATIVE_TEST_DIR.rglob("*.wav"))

    random.seed(42)
    if unknown_files:
        test_files.extend(random.sample(unknown_files, min(4, len(unknown_files))))
    if silence_files:
        test_files.extend(random.sample(silence_files, min(3, len(silence_files))))
    if neg_files:
        test_files.extend(random.sample(neg_files, min(2, len(neg_files))))

    if not test_files:
        print("[ERROR] No audio files found in dataset directories!")
        return False

    print(f"\nTesting {len(test_files)} real dataset audio files:")
    print("-" * 80)

    all_passed = True
    results = []

    for idx, f in enumerate(test_files, 1):
        # 1. Load raw audio
        raw_audio, orig_sr = load_audio(f)
        orig_samples = len(raw_audio) if raw_audio.ndim == 1 else raw_audio.shape[0]
        orig_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[1]
        orig_dur = orig_samples / orig_sr
        orig_min = float(np.min(raw_audio))
        orig_max = float(np.max(raw_audio))

        # 2. Run preprocessing pipeline
        proc_audio = preprocess_audio(f)
        proc_sr = TARGET_SAMPLE_RATE
        proc_samples = len(proc_audio)
        proc_shape = proc_audio.shape
        proc_dtype = proc_audio.dtype
        proc_min = float(np.min(proc_audio))
        proc_max = float(np.max(proc_audio))
        proc_peak = float(np.max(np.abs(proc_audio)))

        # Verification checks
        is_exact_samples = (proc_samples == TARGET_SAMPLES)
        is_mono_1d = (proc_shape == (TARGET_SAMPLES,))
        is_float32 = (proc_dtype == np.float32)
        is_valid_range = (-1.01 <= proc_min <= 1.01) and (-1.01 <= proc_max <= 1.01)
        no_nan_inf = not (np.isnan(proc_audio).any() or np.isinf(proc_audio).any())

        file_pass = is_exact_samples and is_mono_1d and is_float32 and is_valid_range and no_nan_inf
        if not file_pass:
            all_passed = False

        status_str = "PASS" if file_pass else "FAIL"
        rel_path = f.relative_to(DATASET_DIR)

        print(f"\n[Test #{idx:02d}] {rel_path} --> {status_str}")
        print(f"  - Original   : SR={orig_sr} Hz, Ch={orig_channels}, Samples={orig_samples} ({orig_dur:.3f}s), Amplitude=[{orig_min:.3f}, {orig_max:.3f}]")
        print(f"  - Processed  : SR={proc_sr} Hz, Shape={proc_shape}, Samples={proc_samples} ({(proc_samples/proc_sr):.3f}s), Dtype={proc_dtype}")
        print(f"  - Amplitude  : Min={proc_min:.4f}, Max={proc_max:.4f}, PeakAbs={proc_peak:.4f}")
        print(f"  - Checks     : ExactSamples={is_exact_samples}, Mono1D={is_mono_1d}, Float32={is_float32}, RangeValid={is_valid_range}, NoNaN/Inf={no_nan_inf}")

        results.append({
            "file": str(rel_path),
            "orig_sr": orig_sr,
            "orig_samples": orig_samples,
            "proc_shape": str(proc_shape),
            "peak": proc_peak,
            "status": status_str
        })

    # Edge Case Tests
    print("\n" + "=" * 80)
    print("Testing Synthetic Edge Cases (Short, Long, Stereo, Non-16k, Silence):")
    print("-" * 80)

    edge_cases = [
        ("Short Audio (0.5s, 8000 samples @ 16kHz)", np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32), 16000),
        ("Long Audio (1.8s, 28800 samples @ 16kHz)", np.sin(2 * np.pi * 440 * np.linspace(0, 1.8, 28800)).astype(np.float32), 16000),
        ("Stereo Audio (2 channels, 16000 samples @ 16kHz)", np.column_stack([
            np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000)),
            np.cos(2 * np.pi * 880 * np.linspace(0, 1.0, 16000))
        ]).astype(np.float32), 16000),
        ("Non-16k Rate (44.1 kHz, 44100 samples)", np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 44100)).astype(np.float32), 44100),
        ("Low-Amplitude Silence (peak 0.00005)", np.random.uniform(-0.00005, 0.00005, 16000).astype(np.float32), 16000),
    ]

    for name, raw_wave, sr in edge_cases:
        proc = preprocess_audio(raw_wave, orig_sr=sr)
        is_pass = (proc.shape == (TARGET_SAMPLES,)) and (proc.dtype == np.float32) and not np.isnan(proc).any()
        if not is_pass:
            all_passed = False
        print(f"  [{'PASS' if is_pass else 'FAIL'}] {name:<45} --> Out Shape: {proc.shape}, Dtype: {proc.dtype}, Peak: {np.max(np.abs(proc)):.4f}")

    print("\n" + "=" * 80)
    print("PHASE 3 AUDIO PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"All Real File Tests Passed  : {all_passed}")
    print(f"Overall Preprocessing Suite : {'PASS (READY FOR PHASE 4 MFCC)' if all_passed else 'FAIL'}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
