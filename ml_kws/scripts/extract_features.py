"""
SIH 26172 - R2 ML/KWS Batch Feature Extraction Script
Phase 4: Extracts MFCC features across all dataset classes and caches .npz archives.
"""

import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

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
    CACHE_DIR,
    KEYWORD_DIR,
    NEGATIVE_TEST_DIR,
    SILENCE_DIR,
    UNKNOWN_DIR,
)
from features import (
    CLASS_NAMES,
    FEATURE_DTYPE,
    FEATURE_SHAPE,
    LABEL_MAP,
    extract_features,
    load_dataset_npz,
    save_dataset_npz,
)


def extract_speaker_tag(file_path: Path, class_name: str) -> str:
    """Extracts speaker tag without inventing identities."""
    if class_name == "keyword":
        parts = file_path.stem.split("_")
        if len(parts) >= 4 and parts[0].lower() == "astra":
            return parts[1]  # e.g., 'Imroz', 'Keshav', 'Jay', etc.
        return "keyword_speaker"
    elif class_name == "unknown":
        return "unknown"
    elif class_name == "silence":
        return "silence"
    elif class_name == "negative_test":
        return "negative_test"
    return "other"


def process_class_directory(
    directory: Path,
    class_name: str,
    label_id: int,
    is_recursive: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[str], List[str], List[Tuple[str, str]]]:
    """Processes all WAV files in a given directory."""
    if is_recursive:
        wav_files = sorted(list(directory.rglob("*.wav")))
    else:
        wav_files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])

    features_list = []
    labels_list = []
    filenames_list = []
    speakers_list = []
    failures = []

    print(f"\nProcessing [{class_name.upper()}] from {directory} ({len(wav_files)} files)...")
    start_t = time.time()

    for idx, fpath in enumerate(wav_files, 1):
        try:
            feat = extract_features(fpath)
            features_list.append(feat)
            labels_list.append(label_id)
            filenames_list.append(fpath.name)
            speakers_list.append(extract_speaker_tag(fpath, class_name))
        except Exception as e:
            failures.append((str(fpath), str(e)))

        if idx % 500 == 0 or idx == len(wav_files):
            elapsed = time.time() - start_t
            rate = idx / (elapsed + 1e-6)
            print(f"  Processed {idx}/{len(wav_files)} ({rate:.1f} files/sec)")

    return features_list, labels_list, filenames_list, speakers_list, failures


def run_batch_extraction():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Batch Feature Extraction (Phase 4)")
    print(f"Standard Feature Dimensions: Shape={FEATURE_SHAPE}, Dtype={FEATURE_DTYPE}")
    print(f"Cache Output Directory      : {CACHE_DIR}")
    print("=" * 80)

    overall_start = time.time()
    all_failures = []

    # 1. Extract Keyword Samples ('ASTRA')
    kw_feats, kw_labels, kw_files, kw_spks, kw_fails = process_class_directory(
        KEYWORD_DIR, class_name="keyword", label_id=LABEL_MAP["keyword"], is_recursive=False
    )
    all_failures.extend(kw_fails)

    # 2. Extract Unknown Samples (Speech Commands)
    unk_feats, unk_labels, unk_files, unk_spks, unk_fails = process_class_directory(
        UNKNOWN_DIR, class_name="unknown", label_id=LABEL_MAP["unknown"], is_recursive=True
    )
    all_failures.extend(unk_fails)

    # 3. Extract Silence / Background Noise Samples
    sil_feats, sil_labels, sil_files, sil_spks, sil_fails = process_class_directory(
        SILENCE_DIR, class_name="silence", label_id=LABEL_MAP["silence"], is_recursive=False
    )
    all_failures.extend(sil_fails)

    # Combine Main Training/Val Pool (Keyword + Unknown + Silence)
    X_main = np.array(kw_feats + unk_feats + sil_feats, dtype=FEATURE_DTYPE)
    y_main = np.array(kw_labels + unk_labels + sil_labels, dtype=np.int32)
    files_main = np.array(kw_files + unk_files + sil_files, dtype=str)
    spks_main = np.array(kw_spks + unk_spks + sil_spks, dtype=str)

    # 4. Extract Negative Test Benchmark (Separate Archive)
    neg_feats, neg_labels, neg_files, neg_spks, neg_fails = process_class_directory(
        NEGATIVE_TEST_DIR, class_name="negative_test", label_id=LABEL_MAP["negative_test"], is_recursive=False
    )
    all_failures.extend(neg_fails)

    X_neg = np.array(neg_feats, dtype=FEATURE_DTYPE)
    y_neg = np.array(neg_labels, dtype=np.int32)
    files_neg = np.array(neg_files, dtype=str)
    spks_neg = np.array(neg_spks, dtype=str)

    # Save to Cache
    main_npz_path = CACHE_DIR / "mfcc_dataset.npz"
    neg_npz_path = CACHE_DIR / "mfcc_negative_test.npz"

    print("\n" + "=" * 80)
    print("SAVING COMPRESSED FEATURE ARCHIVES...")
    save_dataset_npz(main_npz_path, X_main, y_main, files_main, spks_main, class_names=CLASS_NAMES)
    save_dataset_npz(neg_npz_path, X_neg, y_neg, files_neg, spks_neg, class_names=np.array(["negative_test"]))

    # Verify Saved Archives
    verified_main = load_dataset_npz(main_npz_path)
    verified_neg = load_dataset_npz(neg_npz_path)

    total_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("PHASE 4 FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time        : {total_time:.2f} seconds ({total_time / 60:.2f} min)")
    print(f"Total WAV Files Processed   : {len(X_main) + len(X_neg)}")
    print(f"Total Failures / Skipped    : {len(all_failures)}")
    if all_failures:
        print(f"  Failures: {all_failures}")

    print("\n--- MAIN FEATURE DATASET (mfcc_dataset.npz) ---")
    print(f"File Path                   : {main_npz_path}")
    print(f"Archive File Size           : {main_npz_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Feature Array X Shape       : {verified_main['X'].shape}")
    print(f"Label Array y Shape         : {verified_main['y'].shape}")
    print(f"Feature Dtype               : {verified_main['X'].dtype}")
    print(f"Label Dtype                 : {verified_main['y'].dtype}")
    print(f"All Values Finite           : {np.all(np.isfinite(verified_main['X']))}")
    print(f"Feature Min Value           : {float(np.min(verified_main['X'])):.4f}")
    print(f"Feature Max Value           : {float(np.max(verified_main['X'])):.4f}")
    print(f"Feature Mean Value          : {float(np.mean(verified_main['X'])):.4f}")
    print(f"Feature Std Deviation       : {float(np.std(verified_main['X'])):.4f}")

    label_counts = Counter(verified_main["y"])
    print("\nClass Distribution:")
    print(f"  [0] SILENCE               : {label_counts[0]:4d} samples ({label_counts[0]/len(y_main)*100:.2f}%)")
    print(f"  [1] UNKNOWN               : {label_counts[1]:4d} samples ({label_counts[1]/len(y_main)*100:.2f}%)")
    print(f"  [2] KEYWORD ('ASTRA')     : {label_counts[2]:4d} samples ({label_counts[2]/len(y_main)*100:.2f}%)")
    print(f"  TOTAL                     : {len(y_main):4d} samples")

    speaker_counts = Counter(verified_main["speakers"])
    print("\nSpeaker Breakdown (Keyword metadata preserved):")
    for spk, cnt in sorted(speaker_counts.items()):
        print(f"  {spk:<20} : {cnt:4d} samples")

    print("\n--- NEGATIVE TEST DATASET (mfcc_negative_test.npz) ---")
    print(f"File Path                   : {neg_npz_path}")
    print(f"Archive File Size           : {neg_npz_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Feature Array X_neg Shape   : {verified_neg['X'].shape}")
    print(f"Label Array y_neg Shape     : {verified_neg['y'].shape}")
    print(f"All Values Finite           : {np.all(np.isfinite(verified_neg['X']))}")
    print(f"Negative Test Samples       : {len(verified_neg['y'])}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_extraction()
