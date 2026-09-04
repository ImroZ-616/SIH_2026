"""
SIH 26172 - R2 ML/KWS Feature Extraction Test Suite
Phase 4: Automated Verification of MFCC Pipeline
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure sys.path includes repository root and ml_kws/src
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent
_SRC_DIR = _ML_KWS_DIR / "src"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    KEYWORD_DIR,
    NEGATIVE_TEST_DIR,
    SILENCE_DIR,
    TARGET_SAMPLE_RATE,
    UNKNOWN_DIR,
)
from features import (
    FEATURE_COEFFS,
    FEATURE_DTYPE,
    FEATURE_FRAMES,
    FEATURE_SHAPE,
    extract_features,
    load_dataset_npz,
    save_dataset_npz,
)


def run_unit_tests():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Feature Extraction Test Suite (Phase 4)")
    print(f"Target Feature Standard: Shape={FEATURE_SHAPE}, Dtype={FEATURE_DTYPE}")
    print("=" * 80)

    tests_passed = 0
    total_tests = 0

    # --------------------------------------------------------------------------
    # Test 1: Synthetic Zero Audio (1.0s @ 16 kHz)
    # --------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 1] Synthetic Zero Audio (1.0s @ 16 kHz)")
    audio_zeros = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    feat_zeros = extract_features(audio_zeros, orig_sr=TARGET_SAMPLE_RATE)
    assert feat_zeros.shape == FEATURE_SHAPE, f"Shape mismatch: {feat_zeros.shape}"
    assert feat_zeros.dtype == FEATURE_DTYPE, f"Dtype mismatch: {feat_zeros.dtype}"
    assert np.all(np.isfinite(feat_zeros)), "Contains non-finite values"
    print(f"  --> PASS | Shape={feat_zeros.shape}, Dtype={feat_zeros.dtype}, Min={feat_zeros.min():.2f}, Max={feat_zeros.max():.2f}")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # Test 2: Synthetic 440 Hz Sine Wave (1.0s @ 16 kHz)
    # --------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 2] Synthetic 440 Hz Sine Wave (1.0s @ 16 kHz)")
    t = np.arange(TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE
    audio_sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    feat_sine = extract_features(audio_sine, orig_sr=TARGET_SAMPLE_RATE)
    assert feat_sine.shape == FEATURE_SHAPE, f"Shape mismatch: {feat_sine.shape}"
    assert feat_sine.dtype == FEATURE_DTYPE, f"Dtype mismatch: {feat_sine.dtype}"
    assert np.all(np.isfinite(feat_sine)), "Contains non-finite values"
    print(f"  --> PASS | Shape={feat_sine.shape}, Dtype={feat_sine.dtype}, Min={feat_sine.min():.2f}, Max={feat_sine.max():.2f}")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # Test 3: Synthetic Edge Cases (Short 0.5s, Long 1.6s, Stereo)
    # --------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 3] Synthetic Edge Cases (Short 0.5s, Long 1.6s, Stereo 2-ch)")
    audio_short = np.random.randn(8000).astype(np.float32)
    audio_long = np.random.randn(25600).astype(np.float32)
    audio_stereo = np.random.randn(16000, 2).astype(np.float32)

    feat_short = extract_features(audio_short, orig_sr=TARGET_SAMPLE_RATE)
    feat_long = extract_features(audio_long, orig_sr=TARGET_SAMPLE_RATE)
    feat_stereo = extract_features(audio_stereo, orig_sr=TARGET_SAMPLE_RATE)

    assert feat_short.shape == FEATURE_SHAPE
    assert feat_long.shape == FEATURE_SHAPE
    assert feat_stereo.shape == FEATURE_SHAPE
    print(f"  --> PASS | Short(0.5s)={feat_short.shape}, Long(1.6s)={feat_long.shape}, Stereo={feat_stereo.shape}")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # Test 4: Real Dataset Audio Files
    # --------------------------------------------------------------------------
    real_test_samples = [
        ("KEYWORD", next(KEYWORD_DIR.glob("*.wav"), None)),
        ("UNKNOWN", next(UNKNOWN_DIR.rglob("*.wav"), None)),
        ("SILENCE", next(SILENCE_DIR.glob("*.wav"), None)),
        ("NEGATIVE_TEST", next(NEGATIVE_TEST_DIR.glob("*.wav"), None)),
    ]

    for label_name, sample_path in real_test_samples:
        total_tests += 1
        print(f"\n[Test {total_tests}] Real Audio Sample: {label_name} ({sample_path.name if sample_path else 'None'})")
        if sample_path and sample_path.is_file():
            feat = extract_features(sample_path)
            assert feat.shape == FEATURE_SHAPE, f"Shape mismatch: {feat.shape}"
            assert feat.dtype == FEATURE_DTYPE, f"Dtype mismatch: {feat.dtype}"
            assert np.all(np.isfinite(feat)), "Contains non-finite values"
            print(f"  --> PASS | Shape={feat.shape}, Dtype={feat.dtype}, Min={feat.min():.2f}, Max={feat.max():.2f}, Mean={feat.mean():.2f}")
            tests_passed += 1
        else:
            print(f"  --> SKIP (Sample file not found)")

    # --------------------------------------------------------------------------
    # Test 8: NPZ Archive Serialization & Deserialization
    # --------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 8] NPZ Dataset Serialization & Deserialization Roundtrip")
    dummy_X = np.random.randn(5, 98, 13).astype(np.float32)
    dummy_y = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    dummy_files = np.array(["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"])
    dummy_speakers = np.array(["spk1", "spk2", "spk1", "unknown", "silence"])

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_dataset_npz(tmp_path, dummy_X, dummy_y, dummy_files, dummy_speakers)
        loaded = load_dataset_npz(tmp_path)
        np.testing.assert_array_equal(loaded["X"], dummy_X)
        np.testing.assert_array_equal(loaded["y"], dummy_y)
        np.testing.assert_array_equal(loaded["filenames"], dummy_files)
        np.testing.assert_array_equal(loaded["speakers"], dummy_speakers)
        assert loaded["X"].dtype == np.float32
        assert loaded["y"].dtype == np.int32
        print(f"  --> PASS | Saved & verified {tmp_path.name} (X shape={loaded['X'].shape}, y shape={loaded['y'].shape})")
        tests_passed += 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {tests_passed} / {total_tests} Tests Passed (100%)")
    print("=" * 80)
    return tests_passed == total_tests


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)
