'''
SIH 26172 - Phase 2: Dataset Validation Script
Validates the structural integrity, audio format, and class distribution of the raw dataset.
Does NOT perform MFCC extraction or model training.
'''

import sys
import wave
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    DATASET_DIR,
    KEYWORD_DIR,
    UNKNOWN_DIR,
    SILENCE_DIR,
    NEGATIVE_TEST_DIR,
)


def inspect_directory(name: str, path: Path):
    '''Validates all WAV files in a directory and reports statistics.'''
    print(f"\n--- Class: {name} [{path}] ---")
    if not path.exists():
        print(f"  [ERROR] Directory does not exist: {path}")
        return {"total_files": 0, "valid_wavs": 0, "corrupt_files": 0, "total_size_mb": 0.0}

    wav_files = list(path.rglob("*.wav"))
    all_files = [f for f in path.rglob("*") if f.is_file() and f.name != ".gitkeep"]

    total_files = len(all_files)
    valid_wavs = 0
    corrupt_files = 0
    total_bytes = 0
    sample_rates = set()
    channels_set = set()
    durations = []

    for f in wav_files:
        try:
            total_bytes += f.stat().st_size
            with wave.open(str(f), "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                n_frames = wf.getnframes()
                dur = n_frames / float(sr)

                sample_rates.add(sr)
                channels_set.add(n_ch)
                durations.append(dur)
                valid_wavs += 1
        except Exception as e:
            print(f"  [CORRUPT FILE] {f.name}: {e}")
            corrupt_files += 1

    non_wavs = [f for f in all_files if not f.name.endswith(".wav")]

    total_size_mb = total_bytes / (1024 * 1024)
    print(f"  Total audio files (.wav) : {valid_wavs}")
    if non_wavs:
        print(f"  Non-WAV files            : {len(non_wavs)} ({[f.name for f in non_wavs[:5]]})")
    print(f"  Corrupt/Invalid files    : {corrupt_files}")
    print(f"  Total storage size       : {total_size_mb:.2f} MB")
    if sample_rates:
        print(f"  Detected Sample Rates    : {sorted(list(sample_rates))} Hz")
    if channels_set:
        print(f"  Detected Channels        : {sorted(list(channels_set))} (1=Mono, 2=Stereo)")
    if durations:
        avg_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)
        print(f"  Duration Range           : {min_dur:.2f}s - {max_dur:.2f}s (Avg: {avg_dur:.2f}s)")

    return {
        "total_files": total_files,
        "valid_wavs": valid_wavs,
        "corrupt_files": corrupt_files,
        "total_size_mb": total_size_mb,
    }


def validate_all():
    print("=" * 60)
    print("SIH 26172 - ML/KWS Dataset Validation Suite (Phase 2)")
    print("=" * 60)

    stats = {}
    stats["KEYWORD"] = inspect_directory("KEYWORD (Target: 'ASTRA')", KEYWORD_DIR)
    stats["UNKNOWN"] = inspect_directory("UNKNOWN (Non-keyword Speech)", UNKNOWN_DIR)
    stats["SILENCE"] = inspect_directory("SILENCE (Background / Ambience)", SILENCE_DIR)
    stats["NEGATIVE_TEST"] = inspect_directory("NEGATIVE_TEST (Hard Negatives)", NEGATIVE_TEST_DIR)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Class Name':<20} | {'Valid WAVs':<12} | {'Corrupt':<8} | {'Size (MB)':<10}")
    print("-" * 60)
    total_valid = 0
    total_size = 0.0
    for name, s in stats.items():
        print(f"{name:<20} | {s['valid_wavs']:<12} | {s['corrupt_files']:<8} | {s['total_size_mb']:<10.2f}")
        total_valid += s["valid_wavs"]
        total_size += s["total_size_mb"]
    print("-" * 60)
    print(f"{'TOTAL':<20} | {total_valid:<12} | {0:<8} | {total_size:<10.2f}")
    print("=" * 60)

    if stats["KEYWORD"]["valid_wavs"] == 0:
        print("\n[NOTE] KEYWORD class has 0 files currently. Genuine 'ASTRA' recordings will be added via recording workflow.")


if __name__ == "__main__":
    validate_all()
