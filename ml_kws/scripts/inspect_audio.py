"""
SIH 26172 - Phase 3 Audio Dataset Inspection Utility
Systematically inspects raw audio properties across all classes in the dataset.
"""

import sys
import wave
from pathlib import Path
from collections import Counter
import numpy as np

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DATASET_DIR, UNKNOWN_DIR, SILENCE_DIR, NEGATIVE_TEST_DIR


def inspect_class(name: str, path: Path):
    print(f"\n{'='*70}")
    print(f"Inspecting Dataset Class: {name} [{path}]")
    print(f"{'='*70}")

    wav_files = list(path.rglob("*.wav"))
    if not wav_files:
        print("  No audio files found.")
        return

    sample_rates = Counter()
    channels = Counter()
    sample_widths = Counter()
    durations = []
    sample_counts = []

    for f in wav_files:
        try:
            with wave.open(str(f), "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n_frames = wf.getnframes()
                dur = n_frames / float(sr)

                sample_rates[sr] += 1
                channels[n_ch] += 1
                sample_widths[sw * 8] += 1
                durations.append(dur)
                sample_counts.append(n_frames)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    print(f"  Total audio files analyzed : {len(wav_files)}")
    print(f"  Sample Rates detected       : {dict(sample_rates)}")
    print(f"  Channels detected           : {dict(channels)} (1=Mono, 2=Stereo)")
    print(f"  Bit Depths detected         : {dict(sample_widths)} bits")
    print(f"  Duration Statistics (sec)   : Min={min(durations):.3f}s | Max={max(durations):.3f}s | Avg={np.mean(durations):.3f}s | Std={np.std(durations):.3f}s")
    print(f"  Sample Count Statistics     : Min={min(sample_counts)} | Max={max(sample_counts)} | Avg={np.mean(sample_counts):.1f}")


def main():
    print("=" * 70)
    print("SIH 26172 - ML/KWS Raw Audio Dataset Property Inspection")
    print("=" * 70)
    inspect_class("UNKNOWN (Speech Words)", UNKNOWN_DIR)
    inspect_class("SILENCE (Background / Ambience)", SILENCE_DIR)
    inspect_class("NEGATIVE_TEST (Hard Negatives)", NEGATIVE_TEST_DIR)


if __name__ == "__main__":
    main()
