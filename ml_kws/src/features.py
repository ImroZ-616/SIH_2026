"""
SIH 26172 - R2 ML/KWS Feature Extraction Module
Phase 4: MFCC Feature Extraction Wrapper & Dataset Utilities

This module standardizes 1D audio waveforms using Phase 3 preprocessing
and extracts 2D MFCC feature matrices (98, 13) using R1's DSP feature extractor.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure repository root and ml_kws/src are on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

# Import R1 DSP feature extractor and Phase 3 audio preprocessor
from audio.mfcc import extract_mfcc
try:
    from audio_preprocessing import preprocess_audio
except ImportError:
    from audio import preprocess_audio  # Fallback if audio.py exists

from config import (
    CACHE_DIR,
    TARGET_DURATION,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
)

# Standard Expected Feature Dimensions
FEATURE_FRAMES = 98
FEATURE_COEFFS = 13
FEATURE_SHAPE = (FEATURE_FRAMES, FEATURE_COEFFS)
FEATURE_DTYPE = np.float32

# Class Label Definitions
CLASS_NAMES = np.array(["silence", "unknown", "keyword"])
LABEL_MAP = {
    "silence": 0,
    "unknown": 1,
    "keyword": 2,
    "negative_test": -1,
}


def extract_features(
    file_path_or_audio: Union[str, Path, np.ndarray],
    orig_sr: Optional[int] = None,
) -> np.ndarray:
    """Standardizes input audio to 16 kHz Mono 1.0s and extracts (98, 13) MFCCs.

    Parameters
    ----------
    file_path_or_audio : Union[str, Path, np.ndarray]
        Path to WAV audio file or raw 1D NumPy waveform array.
    orig_sr : Optional[int]
        Original sample rate if passing an array directly.

    Returns
    -------
    np.ndarray
        2D MFCC feature matrix of shape (98, 13) with dtype float32.
    """
    # 1. Standardize waveform to (16000,) float32 Mono @ 16 kHz
    waveform = preprocess_audio(file_path_or_audio, orig_sr=orig_sr)

    if waveform.shape != (TARGET_SAMPLES,):
        raise ValueError(
            f"Preprocessed audio shape {waveform.shape} != ({TARGET_SAMPLES},)"
        )

    # 2. Extract MFCC features using R1 DSP pipeline
    mfcc = extract_mfcc(
        waveform_16k=waveform,
        sample_rate=TARGET_SAMPLE_RATE,
        frame_ms=30,
        hop_ms=10,
        n_fft=480,
        n_mels=40,
        n_mfcc=FEATURE_COEFFS,
    )

    # 3. Validate output shape, dtype, and numerical sanity
    if mfcc.shape != FEATURE_SHAPE:
        raise ValueError(
            f"Extracted MFCC shape {mfcc.shape} != expected {FEATURE_SHAPE}"
        )

    if mfcc.dtype != FEATURE_DTYPE:
        mfcc = mfcc.astype(FEATURE_DTYPE)

    if not np.all(np.isfinite(mfcc)):
        raise ValueError("Extracted MFCC contains non-finite values (NaN or Inf)")

    return mfcc


def save_dataset_npz(
    output_path: Union[str, Path],
    X: np.ndarray,
    y: np.ndarray,
    filenames: np.ndarray,
    speakers: np.ndarray,
    class_names: Optional[np.ndarray] = None,
) -> Path:
    """Saves extracted features and metadata into a compressed .npz archive."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES

    np.savez_compressed(
        out_p,
        X=X.astype(FEATURE_DTYPE),
        y=y.astype(np.int32),
        filenames=filenames.astype(str),
        speakers=speakers.astype(str),
        class_names=class_names.astype(str),
    )
    return out_p


def load_dataset_npz(input_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Loads a feature dataset from a compressed .npz archive."""
    inp_p = Path(input_path)
    if not inp_p.is_file():
        raise FileNotFoundError(f"Feature dataset not found: {inp_p}")

    with np.load(inp_p, allow_pickle=True) as data:
        dataset = {
            "X": data["X"],
            "y": data["y"],
            "filenames": data["filenames"],
            "speakers": data["speakers"],
            "class_names": data["class_names"],
        }
    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("SIH 26172 - ML/KWS Feature Extraction Module (Phase 4)")
    print("=" * 60)
    print(f"Target Feature Shape : {FEATURE_SHAPE}")
    print(f"Target Feature Dtype : {FEATURE_DTYPE}")
    print(f"Class Mappings       : {LABEL_MAP}")
    print("=" * 60)
