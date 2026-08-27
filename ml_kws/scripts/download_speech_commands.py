'''
SIH 26172 - Phase 2: Speech Commands Downloader and Organizer
Extracts non-keyword words to dataset/unknown,
extracts and slices background noise into dataset/silence,
and organizes negative test samples into dataset/negative_test.
'''

import os
import sys
import tarfile
import shutil
import wave
import struct
import random
from pathlib import Path

# Add src to path for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    DATASET_DIR,
    UNKNOWN_DIR,
    SILENCE_DIR,
    NEGATIVE_TEST_DIR,
    CACHE_DIR
)

ARCHIVE_PATH = CACHE_DIR / "speech_commands_v0.01.tar.gz"

# Words to populate in UNKNOWN (broad vocabulary excluding custom keyword)
UNKNOWN_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"
]

# Phonetically interesting / tricky words for NEGATIVE_TEST
NEGATIVE_WORDS = ["marvin", "stop", "tree", "three"]

# Max samples per word to maintain class balance (30 words * 80 = 2400 samples)
MAX_SAMPLES_PER_WORD = 80


def extract_and_organize(archive_path: Path):
    '''Iterates through archive and extracts selected audio files.'''
    print(f"[EXTRACT] Reading archive: {archive_path}")

    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    SILENCE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    bg_noise_dir = CACHE_DIR / "background_noise_raw"
    bg_noise_dir.mkdir(parents=True, exist_ok=True)

    word_counts = {w: 0 for w in UNKNOWN_WORDS}
    unknown_total = 0
    bg_count = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar:
            norm_name = m.name.lstrip("./").replace("\\", "/")
            parts = norm_name.split("/")

            # 1. Background noise files
            if len(parts) >= 2 and parts[0] == "_background_noise_" and parts[-1].endswith(".wav"):
                dest_file = bg_noise_dir / parts[-1]
                f = tar.extractfile(m)
                if f is not None:
                    with open(dest_file, "wb") as out:
                        out.write(f.read())
                    bg_count += 1

            # 2. Unknown speech words
            elif len(parts) >= 2 and parts[0] in UNKNOWN_WORDS and parts[-1].endswith(".wav"):
                word = parts[0]
                if word_counts[word] < MAX_SAMPLES_PER_WORD:
                    word_dir = UNKNOWN_DIR / word
                    word_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = word_dir / parts[-1]
                    f = tar.extractfile(m)
                    if f is not None:
                        with open(dest_file, "wb") as out:
                            out.write(f.read())
                        word_counts[word] += 1
                        unknown_total += 1

    print(f"[EXTRACT] Extracted {unknown_total} UNKNOWN audio clips across {len(UNKNOWN_WORDS)} words.")
    print(f"[EXTRACT] Extracted {bg_count} raw background noise audio tracks.")

    # 3. Generate 1.0-second silence / background slices from raw background noise
    generate_silence_samples(bg_noise_dir, SILENCE_DIR, target_samples=500)

    # 4. Populate initial negative test samples
    populate_negative_test(UNKNOWN_DIR, NEGATIVE_TEST_DIR)


def generate_silence_samples(raw_bg_dir: Path, output_dir: Path, target_samples: int = 500):
    '''Slices background noise tracks into 1.0 second (16000 samples) WAV clips.'''
    print(f"[SILENCE] Generating {target_samples} 1-second background/silence audio clips...")
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_files = list(raw_bg_dir.glob("*.wav"))
    if not bg_files:
        print("[SILENCE] Warning: No background noise wav files found.")
        return

    random.seed(42)
    sample_rate = 16000
    slice_samples = sample_rate  # 1 second = 16000 samples

    clip_index = 0
    samples_per_file = max(1, target_samples // (len(bg_files) + 1))

    for bg_file in bg_files:
        try:
            with wave.open(str(bg_file), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                if framerate != sample_rate:
                    continue

                raw_data = wf.readframes(n_frames)

                max_start = max(0, n_frames - slice_samples)
                if max_start == 0:
                    continue

                for i in range(samples_per_file):
                    start_frame = random.randint(0, max_start)
                    start_byte = start_frame * n_channels * sampwidth
                    end_byte = (start_frame + slice_samples) * n_channels * sampwidth
                    chunk_bytes = raw_data[start_byte:end_byte]

                    out_path = output_dir / f"bg_{bg_file.stem}_{i:04d}.wav"
                    with wave.open(str(out_path), "wb") as out_wf:
                        out_wf.setnchannels(n_channels)
                        out_wf.setsampwidth(sampwidth)
                        out_wf.setframerate(framerate)
                        out_wf.writeframes(chunk_bytes)
                    clip_index += 1
        except Exception as e:
            print(f"[SILENCE] Error slicing {bg_file.name}: {e}")

    # Add near-silence / low-amplitude background noise samples
    for i in range(50):
        out_path = output_dir / f"near_silence_{i:04d}.wav"
        with wave.open(str(out_path), "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)  # 16-bit
            out_wf.setframerate(16000)
            samples = [random.randint(-8, 8) for _ in range(16000)]
            raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
            out_wf.writeframes(raw_bytes)
            clip_index += 1

    print(f"[SILENCE] Generated total of {clip_index} silence/background audio clips.")


def populate_negative_test(unknown_dir: Path, negative_dir: Path):
    '''Copies selected challenging negative speech samples to negative_test directory.'''
    print("[NEGATIVE_TEST] Organizing negative test baseline audio...")
    negative_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    random.seed(42)

    for word in NEGATIVE_WORDS:
        word_path = unknown_dir / word
        if word_path.exists():
            files = list(word_path.glob("*.wav"))
            sample_files = random.sample(files, min(25, len(files)))
            for f in sample_files:
                dest = negative_dir / f"hard_neg_{word}_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
                    copied += 1

    print(f"[NEGATIVE_TEST] Populated {copied} initial hard negative benchmark files.")


if __name__ == "__main__":
    print("=" * 60)
    print("SIH 26172 - Speech Commands Organizer")
    print("=" * 60)
    extract_and_organize(ARCHIVE_PATH)
    print("=" * 60)
    print("Dataset organization completed successfully.")
