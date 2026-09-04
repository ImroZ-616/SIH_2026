"""
SIH 26172 - R2 ML/KWS Dataset Splitting & Preparation Script
Phase 6: Partitions cached baseline feature dataset (mfcc_dataset.npz) into
train_data.npz, val_data.npz, and test_data.npz with strict anti-leakage boundaries.

Split Strategy:
1. Keyword (ASTRA - 225 samples):
   - Strict Speaker-Disjoint Split:
     * Train (4 speakers = 156 samples): Keshav (45), Sneha (39), Jay (36), Yash (36)
     * Validation (1 speaker = 35 samples): Imroz (35)
     * Test (1 speaker = 34 samples): Shaswat (34)
2. Unknown (Speech Commands - 2,400 samples):
   - Strict Speaker-Hash Disjoint Split:
     * Grouped strictly by speaker hash prefix before '_nohash_' (274 unique hashes)
     * Deterministic allocation with RANDOM_SEED = 42
     * Train: 205 hashes = 1,680 samples
     * Validation: 37 hashes = 360 samples
     * Test: 32 hashes = 360 samples
3. Silence / Background Noise (476 samples):
   - Strict Track-Level Grouping + i.i.d. Near-Silence:
     * Train (318 samples): bg_doing_the_dishes (71), bg_exercise_bike (71),
                           bg_running_tap (71), bg_white_noise (71),
                           near_silence 0000..0033 (34)
     * Validation (79 samples): bg_dude_miaowing (71), near_silence 0034..0041 (8)
     * Test (79 samples): bg_pink_noise (71), near_silence 0042..0049 (8)
4. Negative Test Benchmark (100 samples):
   - mfcc_negative_test.npz remains 100% isolated and untouched.

Target Total Dataset Sizes:
- Train: 2,154 samples (156 Keyword + 1,680 Unknown + 318 Silence)
- Validation: 474 samples (35 Keyword + 360 Unknown + 79 Silence)
- Test: 473 samples (34 Keyword + 360 Unknown + 79 Silence)
- Total: 3,101 samples
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

from config import CACHE_DIR
from features import (
    CLASS_NAMES,
    FEATURE_DTYPE,
    FEATURE_SHAPE,
    LABEL_MAP,
    load_dataset_npz,
    save_dataset_npz,
)

# Deterministic Random Seed
RANDOM_SEED = 42

# 1. Keyword Speaker Allocations
TRAIN_KEYWORD_SPEAKERS: Set[str] = {"Keshav", "Sneha", "Jay", "Yash"}
VAL_KEYWORD_SPEAKERS: Set[str] = {"Imroz"}
TEST_KEYWORD_SPEAKERS: Set[str] = {"Shaswat"}

# 2. Silence Track Allocations
TRAIN_SILENCE_TRACKS: Set[str] = {
    "bg_doing_the_dishes",
    "bg_exercise_bike",
    "bg_running_tap",
    "bg_white_noise",
}
VAL_SILENCE_TRACKS: Set[str] = {"bg_dude_miaowing"}
TEST_SILENCE_TRACKS: Set[str] = {"bg_pink_noise"}

# 3. Unknown Partition Targets
TARGET_UNKNOWN_TRAIN = 1680
TARGET_UNKNOWN_VAL = 360
TARGET_UNKNOWN_TEST = 360

# Expected Split Totals
EXPECTED_SPLIT_COUNTS = {
    "train": {"keyword": 156, "unknown": 1680, "silence": 318, "total": 2154},
    "val": {"keyword": 35, "unknown": 360, "silence": 79, "total": 474},
    "test": {"keyword": 34, "unknown": 360, "silence": 79, "total": 473},
}


def split_keyword_samples(
    kw_indices: np.ndarray,
    speakers: np.ndarray,
) -> Tuple[List[int], List[int], List[int]]:
    """Splits keyword samples based on strict speaker-disjoint assignment."""
    train_idx, val_idx, test_idx = [], [], []

    for idx in kw_indices:
        spk = speakers[idx]
        if spk in TRAIN_KEYWORD_SPEAKERS:
            train_idx.append(idx)
        elif spk in VAL_KEYWORD_SPEAKERS:
            val_idx.append(idx)
        elif spk in TEST_KEYWORD_SPEAKERS:
            test_idx.append(idx)
        else:
            raise ValueError(f"Unrecognized keyword speaker: '{spk}' at index {idx}")

    return train_idx, val_idx, test_idx


def split_unknown_samples(
    unk_indices: np.ndarray,
    filenames: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Tuple[List[int], List[int], List[int], Dict[str, List[str]]]:
    """Splits unknown samples by speaker hash prefix with deterministic seed."""
    # Group file indices by speaker hash
    hash_to_indices = defaultdict(list)
    for idx in unk_indices:
        fname = filenames[idx]
        if "_nohash_" not in fname:
            raise ValueError(f"Unknown filename missing '_nohash_' pattern: {fname}")
        spk_hash = fname.split("_nohash_")[0]
        hash_to_indices[spk_hash].append(idx)

    # Deterministic shuffle of unique hashes
    rng = np.random.RandomState(seed)
    hashes = sorted(list(hash_to_indices.keys()))
    rng.shuffle(hashes)

    train_hashes, val_hashes, test_hashes = [], [], []
    train_idx, val_idx, test_idx = [], [], []

    for h in hashes:
        indices = hash_to_indices[h]
        if len(train_idx) + len(indices) <= TARGET_UNKNOWN_TRAIN:
            train_hashes.append(h)
            train_idx.extend(indices)
        elif len(val_idx) + len(indices) <= TARGET_UNKNOWN_VAL:
            val_hashes.append(h)
            val_idx.extend(indices)
        else:
            test_hashes.append(h)
            test_idx.extend(indices)

    hash_assignment = {
        "train": train_hashes,
        "val": val_hashes,
        "test": test_hashes,
    }

    return train_idx, val_idx, test_idx, hash_assignment


def split_silence_samples(
    sil_indices: np.ndarray,
    filenames: np.ndarray,
) -> Tuple[List[int], List[int], List[int]]:
    """Splits silence samples by background track group and near-silence range."""
    train_idx, val_idx, test_idx = [], [], []

    for idx in sil_indices:
        fname = filenames[idx]
        if fname.startswith("near_silence_"):
            # near_silence_XXXX.wav
            num_str = fname.replace("near_silence_", "").replace(".wav", "")
            num = int(num_str)
            if 0 <= num <= 33:
                train_idx.append(idx)
            elif 34 <= num <= 41:
                val_idx.append(idx)
            elif 42 <= num <= 49:
                test_idx.append(idx)
            else:
                raise ValueError(f"Near silence index out of range: {fname}")
        else:
            # Multi-segment background track (e.g. bg_doing_the_dishes_0000.wav)
            track_name = "_".join(fname.split("_")[:-1])
            if track_name in TRAIN_SILENCE_TRACKS:
                train_idx.append(idx)
            elif track_name in VAL_SILENCE_TRACKS:
                val_idx.append(idx)
            elif track_name in TEST_SILENCE_TRACKS:
                test_idx.append(idx)
            else:
                raise ValueError(f"Unrecognized silence track: '{track_name}' in {fname}")

    return train_idx, val_idx, test_idx


def verify_split_integrity(
    full_dataset: Dict[str, np.ndarray],
    train_dict: Dict[str, np.ndarray],
    val_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    neg_test_dict: Dict[str, np.ndarray],
) -> None:
    """Runs a strict battery of anti-leakage and dimensional assertions."""
    print("\n" + "=" * 80)
    print("RUNNING STRICT ANTI-LEAKAGE AND INTEGRITY ASSERTIONS...")
    print("=" * 80)

    # 1. Total Sample Conservation
    total_split_samples = len(train_dict["y"]) + len(val_dict["y"]) + len(test_dict["y"])
    assert total_split_samples == len(full_dataset["y"]), (
        f"Sample count mismatch: {total_split_samples} != {len(full_dataset['y'])}"
    )
    print("  [PASS] Total sample count conserved (3,101 samples).")

    # 2. Filename Disjointness (Train / Val / Test)
    train_files = set(train_dict["filenames"])
    val_files = set(val_dict["filenames"])
    test_files = set(test_dict["filenames"])
    neg_files = set(neg_test_dict["filenames"])

    assert len(train_files & val_files) == 0, f"Leakage: Train/Val overlap: {train_files & val_files}"
    assert len(train_files & test_files) == 0, f"Leakage: Train/Test overlap: {train_files & test_files}"
    assert len(val_files & test_files) == 0, f"Leakage: Val/Test overlap: {val_files & test_files}"
    print("  [PASS] Zero filename overlap between Train, Validation, and Test partitions.")

    # 3. Negative Test Isolation
    assert len(train_files & neg_files) == 0, "Leakage: Negative test sample found in Train!"
    assert len(val_files & neg_files) == 0, "Leakage: Negative test sample found in Val!"
    assert len(test_files & neg_files) == 0, "Leakage: Negative test sample found in Test!"
    print("  [PASS] Negative test benchmark completely isolated (0 overlap across all partitions).")

    # 4. Keyword Speaker Disjointness
    train_kw_spks = set(train_dict["speakers"][train_dict["y"] == LABEL_MAP["keyword"]])
    val_kw_spks = set(val_dict["speakers"][val_dict["y"] == LABEL_MAP["keyword"]])
    test_kw_spks = set(test_dict["speakers"][test_dict["y"] == LABEL_MAP["keyword"]])

    assert len(train_kw_spks & val_kw_spks) == 0, f"Keyword Speaker Leakage: Train/Val {train_kw_spks & val_kw_spks}"
    assert len(train_kw_spks & test_kw_spks) == 0, f"Keyword Speaker Leakage: Train/Test {train_kw_spks & test_kw_spks}"
    assert len(val_kw_spks & test_kw_spks) == 0, f"Keyword Speaker Leakage: Val/Test {val_kw_spks & test_kw_spks}"
    assert train_kw_spks == TRAIN_KEYWORD_SPEAKERS, f"Train keyword speakers mismatch: {train_kw_spks}"
    assert val_kw_spks == VAL_KEYWORD_SPEAKERS, f"Val keyword speakers mismatch: {val_kw_spks}"
    assert test_kw_spks == TEST_KEYWORD_SPEAKERS, f"Test keyword speakers mismatch: {test_kw_spks}"
    print(f"  [PASS] Keyword speaker-disjointness verified: Train={sorted(train_kw_spks)}, Val={sorted(val_kw_spks)}, Test={sorted(test_kw_spks)}.")

    # 5. Unknown Speaker Hash Disjointness
    train_unk_hashes = {f.split("_nohash_")[0] for f in train_dict["filenames"][train_dict["y"] == LABEL_MAP["unknown"]]}
    val_unk_hashes = {f.split("_nohash_")[0] for f in val_dict["filenames"][val_dict["y"] == LABEL_MAP["unknown"]]}
    test_unk_hashes = {f.split("_nohash_")[0] for f in test_dict["filenames"][test_dict["y"] == LABEL_MAP["unknown"]]}

    assert len(train_unk_hashes & val_unk_hashes) == 0, f"Unknown Hash Leakage: Train/Val {train_unk_hashes & val_unk_hashes}"
    assert len(train_unk_hashes & test_unk_hashes) == 0, f"Unknown Hash Leakage: Train/Test {train_unk_hashes & test_unk_hashes}"
    assert len(val_unk_hashes & test_unk_hashes) == 0, f"Unknown Hash Leakage: Val/Test {val_unk_hashes & test_unk_hashes}"
    print(f"  [PASS] Unknown speaker-hash disjointness verified: Train={len(train_unk_hashes)} hashes, Val={len(val_unk_hashes)} hashes, Test={len(test_unk_hashes)} hashes.")

    # 6. Silence Track Grouping Disjointness
    def extract_bg_track(fname: str) -> str:
        if fname.startswith("near_silence_"):
            return "near_silence"
        return "_".join(fname.split("_")[:-1])

    train_sil_tracks = {extract_bg_track(f) for f in train_dict["filenames"][train_dict["y"] == LABEL_MAP["silence"]] if not f.startswith("near_silence_")}
    val_sil_tracks = {extract_bg_track(f) for f in val_dict["filenames"][val_dict["y"] == LABEL_MAP["silence"]] if not f.startswith("near_silence_")}
    test_sil_tracks = {extract_bg_track(f) for f in test_dict["filenames"][test_dict["y"] == LABEL_MAP["silence"]] if not f.startswith("near_silence_")}

    assert len(train_sil_tracks & val_sil_tracks) == 0, f"Silence Track Leakage: Train/Val {train_sil_tracks & val_sil_tracks}"
    assert len(train_sil_tracks & test_sil_tracks) == 0, f"Silence Track Leakage: Train/Test {train_sil_tracks & test_sil_tracks}"
    assert len(val_sil_tracks & test_sil_tracks) == 0, f"Silence Track Leakage: Val/Test {val_sil_tracks & test_sil_tracks}"
    assert train_sil_tracks == TRAIN_SILENCE_TRACKS, f"Train silence tracks mismatch: {train_sil_tracks}"
    assert val_sil_tracks == VAL_SILENCE_TRACKS, f"Val silence tracks mismatch: {val_sil_tracks}"
    assert test_sil_tracks == TEST_SILENCE_TRACKS, f"Test silence tracks mismatch: {test_sil_tracks}"
    print(f"  [PASS] Silence track-level disjointness verified: Train={sorted(train_sil_tracks)}, Val={sorted(val_sil_tracks)}, Test={sorted(test_sil_tracks)}.")

    # 7. Exact Class Counts and Shapes
    for split_name, split_data in [("train", train_dict), ("val", val_dict), ("test", test_dict)]:
        expected = EXPECTED_SPLIT_COUNTS[split_name]
        counts = Counter(split_data["y"])

        assert len(split_data["X"]) == expected["total"], f"{split_name} total shape {len(split_data['X'])} != {expected['total']}"
        assert counts[LABEL_MAP["keyword"]] == expected["keyword"], f"{split_name} keyword count {counts[LABEL_MAP['keyword']]} != {expected['keyword']}"
        assert counts[LABEL_MAP["unknown"]] == expected["unknown"], f"{split_name} unknown count {counts[LABEL_MAP['unknown']]} != {expected['unknown']}"
        assert counts[LABEL_MAP["silence"]] == expected["silence"], f"{split_name} silence count {counts[LABEL_MAP['silence']]} != {expected['silence']}"

        assert split_data["X"].shape == (expected["total"], FEATURE_SHAPE[0], FEATURE_SHAPE[1]), f"{split_name} X shape invalid: {split_data['X'].shape}"
        assert split_data["X"].dtype == FEATURE_DTYPE, f"{split_name} X dtype {split_data['X'].dtype} != {FEATURE_DTYPE}"
        assert split_data["y"].dtype == np.int32, f"{split_name} y dtype {split_data['y'].dtype} != int32"
        assert np.all(np.isfinite(split_data["X"])), f"{split_name} X contains non-finite values!"
        print(f"  [PASS] {split_name.upper()} counts & shapes verified: Total={expected['total']} (KW={expected['keyword']}, UNK={expected['unknown']}, SIL={expected['silence']}).")

    print("=" * 80)
    print("ALL INTEGRITY AND ANTI-LEAKAGE ASSERTIONS PASSED PERFECTLY!")
    print("=" * 80)


def run_dataset_split() -> None:
    """Main execution function for Phase 6 Dataset Splitting."""
    start_time = time.time()
    print("=" * 80)
    print("SIH 26172 - ML/KWS Dataset Splitting & Preparation (Phase 6)")
    print(f"Random Seed                  : {RANDOM_SEED}")
    print(f"Cache Directory              : {CACHE_DIR}")
    print("=" * 80)

    # 1. Load baseline cached feature dataset
    main_npz_path = CACHE_DIR / "mfcc_dataset.npz"
    neg_npz_path = CACHE_DIR / "mfcc_negative_test.npz"

    print(f"\nLoading baseline feature dataset from: {main_npz_path}")
    dataset = load_dataset_npz(main_npz_path)
    neg_dataset = load_dataset_npz(neg_npz_path)

    X_full = dataset["X"]
    y_full = dataset["y"]
    filenames_full = dataset["filenames"]
    speakers_full = dataset["speakers"]

    print(f"Full Dataset X shape         : {X_full.shape}")
    print(f"Full Dataset y shape         : {y_full.shape}")
    print(f"Negative Test shape          : {neg_dataset['X'].shape}")

    # 2. Extract indices per class
    kw_indices = np.where(y_full == LABEL_MAP["keyword"])[0]
    unk_indices = np.where(y_full == LABEL_MAP["unknown"])[0]
    sil_indices = np.where(y_full == LABEL_MAP["silence"])[0]

    print(f"\nBaseline Class Counts:")
    print(f"  - Keyword (ASTRA)          : {len(kw_indices)} samples")
    print(f"  - Unknown (Speech Commands): {len(unk_indices)} samples")
    print(f"  - Silence / Background     : {len(sil_indices)} samples")

    # 3. Partition Keyword Samples (Speaker-Disjoint)
    kw_train_idx, kw_val_idx, kw_test_idx = split_keyword_samples(kw_indices, speakers_full)
    print(f"\nKeyword Split (Disjoint Speakers):")
    print(f"  - Train ({sorted(TRAIN_KEYWORD_SPEAKERS)}) : {len(kw_train_idx)} samples")
    print(f"  - Val   ({sorted(VAL_KEYWORD_SPEAKERS)})   : {len(kw_val_idx)} samples")
    print(f"  - Test  ({sorted(TEST_KEYWORD_SPEAKERS)}) : {len(kw_test_idx)} samples")

    # 4. Partition Unknown Samples (Speaker-Hash Disjoint)
    unk_train_idx, unk_val_idx, unk_test_idx, hash_assignments = split_unknown_samples(
        unk_indices, filenames_full, seed=RANDOM_SEED
    )
    print(f"\nUnknown Split (Speaker-Hash Disjoint, Seed={RANDOM_SEED}):")
    print(f"  - Train ({len(hash_assignments['train'])} hashes) : {len(unk_train_idx)} samples")
    print(f"  - Val   ({len(hash_assignments['val'])} hashes)   : {len(unk_val_idx)} samples")
    print(f"  - Test  ({len(hash_assignments['test'])} hashes)  : {len(unk_test_idx)} samples")

    # 5. Partition Silence Samples (Track Grouping + Near Silence)
    sil_train_idx, sil_val_idx, sil_test_idx = split_silence_samples(sil_indices, filenames_full)
    print(f"\nSilence Split (Track-Level Disjoint):")
    print(f"  - Train (4 tracks + 34 near-silence) : {len(sil_train_idx)} samples")
    print(f"  - Val   (1 track  +  8 near-silence) : {len(sil_val_idx)} samples")
    print(f"  - Test  (1 track  +  8 near-silence) : {len(sil_test_idx)} samples")

    # 6. Combine and Deterministically Shuffle within each split
    rng_shuffle = np.random.RandomState(RANDOM_SEED)

    train_indices = np.array(kw_train_idx + unk_train_idx + sil_train_idx)
    rng_shuffle.shuffle(train_indices)

    val_indices = np.array(kw_val_idx + unk_val_idx + sil_val_idx)
    rng_shuffle.shuffle(val_indices)

    test_indices = np.array(kw_test_idx + unk_test_idx + sil_test_idx)
    rng_shuffle.shuffle(test_indices)

    # 7. Formulate Split Data Dictionaries
    train_dict = {
        "X": X_full[train_indices],
        "y": y_full[train_indices],
        "filenames": filenames_full[train_indices],
        "speakers": speakers_full[train_indices],
        "class_names": CLASS_NAMES,
    }
    val_dict = {
        "X": X_full[val_indices],
        "y": y_full[val_indices],
        "filenames": filenames_full[val_indices],
        "speakers": speakers_full[val_indices],
        "class_names": CLASS_NAMES,
    }
    test_dict = {
        "X": X_full[test_indices],
        "y": y_full[test_indices],
        "filenames": filenames_full[test_indices],
        "speakers": speakers_full[test_indices],
        "class_names": CLASS_NAMES,
    }

    # 8. Run Assertions BEFORE saving
    verify_split_integrity(dataset, train_dict, val_dict, test_dict, neg_dataset)

    # 9. Save Split Archives
    train_path = CACHE_DIR / "train_data.npz"
    val_path = CACHE_DIR / "val_data.npz"
    test_path = CACHE_DIR / "test_data.npz"

    print("\nSAVING COMPRESSED SPLIT ARCHIVES...")
    save_dataset_npz(train_path, train_dict["X"], train_dict["y"], train_dict["filenames"], train_dict["speakers"])
    save_dataset_npz(val_path, val_dict["X"], val_dict["y"], val_dict["filenames"], val_dict["speakers"])
    save_dataset_npz(test_path, test_dict["X"], test_dict["y"], test_dict["filenames"], test_dict["speakers"])

    # 10. Reload and re-verify from disk
    print("\nRE-VERIFYING SAVED ARCHIVES FROM DISK...")
    reloaded_train = load_dataset_npz(train_path)
    reloaded_val = load_dataset_npz(val_path)
    reloaded_test = load_dataset_npz(test_path)
    verify_split_integrity(dataset, reloaded_train, reloaded_val, reloaded_test, neg_dataset)

    total_time = time.time() - start_time

    # 11. Final Summary Report
    print("\n" + "=" * 80)
    print("PHASE 6 DATASET SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Execution Time         : {total_time:.2f} seconds")
    print(f"\n1. Train Set Archive         : {train_path}")
    print(f"   - File Size               : {train_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   - X shape                 : {reloaded_train['X'].shape} (dtype: {reloaded_train['X'].dtype})")
    print(f"   - y shape                 : {reloaded_train['y'].shape} (dtype: {reloaded_train['y'].dtype})")
    print(f"   - Total Samples           : {len(reloaded_train['y'])}")
    print(f"     * [2] KEYWORD (ASTRA)   : {np.sum(reloaded_train['y'] == 2)} (Speakers: {sorted(TRAIN_KEYWORD_SPEAKERS)})")
    print(f"     * [1] UNKNOWN           : {np.sum(reloaded_train['y'] == 1)} ({len(hash_assignments['train'])} speaker hashes)")
    print(f"     * [0] SILENCE           : {np.sum(reloaded_train['y'] == 0)} ({len(TRAIN_SILENCE_TRACKS)} tracks + 34 near-silence)")

    print(f"\n2. Validation Set Archive    : {val_path}")
    print(f"   - File Size               : {val_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   - X shape                 : {reloaded_val['X'].shape} (dtype: {reloaded_val['X'].dtype})")
    print(f"   - y shape                 : {reloaded_val['y'].shape} (dtype: {reloaded_val['y'].dtype})")
    print(f"   - Total Samples           : {len(reloaded_val['y'])}")
    print(f"     * [2] KEYWORD (ASTRA)   : {np.sum(reloaded_val['y'] == 2)} (Speaker: {sorted(VAL_KEYWORD_SPEAKERS)})")
    print(f"     * [1] UNKNOWN           : {np.sum(reloaded_val['y'] == 1)} ({len(hash_assignments['val'])} speaker hashes)")
    print(f"     * [0] SILENCE           : {np.sum(reloaded_val['y'] == 0)} ({len(VAL_SILENCE_TRACKS)} track + 8 near-silence)")

    print(f"\n3. Test Set Archive          : {test_path}")
    print(f"   - File Size               : {test_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   - X shape                 : {reloaded_test['X'].shape} (dtype: {reloaded_test['X'].dtype})")
    print(f"   - y shape                 : {reloaded_test['y'].shape} (dtype: {reloaded_test['y'].dtype})")
    print(f"   - Total Samples           : {len(reloaded_test['y'])}")
    print(f"     * [2] KEYWORD (ASTRA)   : {np.sum(reloaded_test['y'] == 2)} (Speaker: {sorted(TEST_KEYWORD_SPEAKERS)})")
    print(f"     * [1] UNKNOWN           : {np.sum(reloaded_test['y'] == 1)} ({len(hash_assignments['test'])} speaker hashes)")
    print(f"     * [0] SILENCE           : {np.sum(reloaded_test['y'] == 0)} ({len(TEST_SILENCE_TRACKS)} track + 8 near-silence)")

    print(f"\n4. Negative Test Benchmark   : {neg_npz_path}")
    print(f"   - Isolation Status        : 100% UNTOUCHED (100 samples)")
    print("=" * 80)


if __name__ == "__main__":
    run_dataset_split()
