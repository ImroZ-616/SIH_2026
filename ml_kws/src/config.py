"""
SIH 26172 - R2 ML/KWS Model Configuration
Central project path definitions and audio standardization settings using pathlib.Path.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Core Component Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENV_DIR = PROJECT_ROOT / "venv"

# Dataset Subdirectories
KEYWORD_DIR = DATASET_DIR / "keyword"
UNKNOWN_DIR = DATASET_DIR / "unknown"
SILENCE_DIR = DATASET_DIR / "silence"
NEGATIVE_TEST_DIR = DATASET_DIR / "negative_test"

# Audio Standardization Constants
TARGET_SAMPLE_RATE = 16000     # 16 kHz sample rate (Standard for KWS & Speech)
TARGET_DURATION = 1.0          # Target audio duration in seconds
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)  # 16000 samples per clip
TARGET_CHANNELS = 1            # Mono (Single channel)
AUDIO_NORM_EPSILON = 1e-8      # Small epsilon to prevent division by zero during normalization

# Project Directories List
ALL_DIRECTORIES = [
    DATASET_DIR,
    KEYWORD_DIR,
    UNKNOWN_DIR,
    SILENCE_DIR,
    NEGATIVE_TEST_DIR,
    SRC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    CACHE_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
]


def ensure_directories():
    """Ensures all required project directories exist on disk."""
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("SIH 26172 - ML/KWS Configuration")
    print("=" * 60)
    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    print(f"DATASET_DIR         : {DATASET_DIR}")
    print(f"  - KEYWORD_DIR     : {KEYWORD_DIR}")
    print(f"  - UNKNOWN_DIR     : {UNKNOWN_DIR}")
    print(f"  - SILENCE_DIR     : {SILENCE_DIR}")
    print(f"  - NEGATIVE_TEST   : {NEGATIVE_TEST_DIR}")
    print(f"TARGET_SAMPLE_RATE  : {TARGET_SAMPLE_RATE} Hz")
    print(f"TARGET_DURATION     : {TARGET_DURATION} s")
    print(f"TARGET_SAMPLES      : {TARGET_SAMPLES} samples")
    print(f"TARGET_CHANNELS     : {TARGET_CHANNELS} (Mono)")
    print("=" * 60)
    print("All paths and audio constants configured properly.")
