'''
SIH 26172 - R2 ML/KWS Model Configuration
Central project path definitions and settings structure using pathlib.Path.
'''

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

# Project Directories List (for automated validation / initialization)
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
    '''Ensures all required project directories exist on disk.'''
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("SIH 26172 - ML/KWS Configuration")
    print("=" * 60)
    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"DATASET_DIR       : {DATASET_DIR}")
    print(f"  - KEYWORD_DIR   : {KEYWORD_DIR}")
    print(f"  - UNKNOWN_DIR   : {UNKNOWN_DIR}")
    print(f"  - SILENCE_DIR   : {SILENCE_DIR}")
    print(f"  - NEGATIVE_TEST : {NEGATIVE_TEST_DIR}")
    print(f"MODELS_DIR        : {MODELS_DIR}")
    print(f"OUTPUTS_DIR       : {OUTPUTS_DIR}")
    print(f"CACHE_DIR         : {CACHE_DIR}")
    print(f"LOGS_DIR          : {LOGS_DIR}")
    print(f"NOTEBOOKS_DIR     : {NOTEBOOKS_DIR}")
    print(f"SCRIPTS_DIR       : {SCRIPTS_DIR}")
    print(f"VENV_DIR          : {VENV_DIR}")
    print("=" * 60)
    print("All paths configured properly.")
