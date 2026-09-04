"""
SIH 26172 - R2 ML/KWS Feature Normalization Module
Phase 7: Per-coefficient feature standardization utilities.

Computes mean and standard deviation vectors across time and samples
STRICTLY from the Training partition (to prevent validation/test leakage),
and standardizes MFCC feature tensors to zero-mean, unit-variance.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Numerical stability epsilon to prevent division by zero
DEFAULT_EPSILON: float = 1e-8


def compute_mfcc_normalization_stats(
    X_train: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-coefficient mean and standard deviation strictly from training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature array of shape (N, 98, 13) or (N, 98, 13, 1).
    epsilon : float
        Small positive constant added for numerical stability.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (mean, std) arrays, each of shape (1, 1, 13) with dtype float32.
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.asarray(X_train)

    if X_train.ndim not in (3, 4):
        raise ValueError(
            f"Expected X_train to have 3 or 4 dimensions (N, T, C) or (N, T, C, 1), got shape {X_train.shape}"
        )

    if X_train.shape[1] != 98 or X_train.shape[2] != 13:
        raise ValueError(
            f"Expected feature shape (*, 98, 13), got {X_train.shape}"
        )

    # Compute mean and standard deviation along sample (0) and time frame (1) axes
    # Resulting shape: (1, 1, 13) or (1, 1, 13, 1)
    if X_train.ndim == 4:
        mean = np.mean(X_train, axis=(0, 1, 3), keepdims=True).astype(np.float32)
        std = np.std(X_train, axis=(0, 1, 3), keepdims=True).astype(np.float32)
        # Squeeze channel dimension to ensure standard (1, 1, 13)
        mean = np.squeeze(mean, axis=3)
        std = np.squeeze(std, axis=3)
    else:
        mean = np.mean(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
        std = np.std(X_train, axis=(0, 1), keepdims=True).astype(np.float32)

    # Ensure numerical stability
    std = np.maximum(std, epsilon)

    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
        raise ValueError("Computed normalization statistics contain non-finite values (NaN/Inf).")

    return mean, std


def apply_mfcc_normalization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Applies precomputed per-coefficient normalization statistics to a feature tensor.

    Formula: X_norm = (X - mean) / (std + epsilon)

    Parameters
    ----------
    X : np.ndarray
        Input feature array of shape (N, 98, 13) or (N, 98, 13, 1) or single sample (98, 13).
    mean : np.ndarray
        Per-coefficient mean vector.
    std : np.ndarray
        Per-coefficient standard deviation vector.
    epsilon : float
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Normalized feature array with dtype float32 and identical shape to X.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32)
    else:
        X = X.astype(np.float32, copy=False)

    # Reshape mean and std to broadcast with X
    mean_b = mean.astype(np.float32)
    std_b = std.astype(np.float32)

    # If X has 4 dimensions (N, 98, 13, 1), ensure broadcasting works
    if X.ndim == 4 and mean_b.ndim == 3:
        mean_b = np.expand_dims(mean_b, axis=-1)
        std_b = np.expand_dims(std_b, axis=-1)

    X_norm = (X - mean_b) / (std_b + epsilon)
    X_norm = X_norm.astype(np.float32)

    if not np.all(np.isfinite(X_norm)):
        raise ValueError("Normalized feature tensor contains non-finite values (NaN/Inf).")

    return X_norm


def save_normalization_stats(
    output_path: Union[str, Path],
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> Path:
    """Saves normalization statistics to a dedicated .npz file."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_p,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        epsilon=np.float32(epsilon),
    )
    return out_p


def load_normalization_stats(
    input_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Loads normalization statistics from a .npz file."""
    inp_p = Path(input_path)
    if not inp_p.is_file():
        raise FileNotFoundError(f"Normalization stats file not found: {inp_p}")

    with np.load(inp_p, allow_pickle=False) as data:
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        epsilon = float(data["epsilon"]) if "epsilon" in data else DEFAULT_EPSILON

    return mean, std, epsilon


if __name__ == "__main__":
    print("=" * 80)
    print("SIH 26172 - ML/KWS Normalization Module (Phase 7)")
    print("=" * 80)
    dummy_train = np.random.randn(100, 98, 13).astype(np.float32) * 10.0 + 5.0
    m, s = compute_mfcc_normalization_stats(dummy_train)
    normed = apply_mfcc_normalization(dummy_train, m, s)
    print(f"Computed Mean Shape: {m.shape}, Std Shape: {s.shape}")
    print(f"Normalized Train Mean (Expected ~0.0): {float(np.mean(normed)):.6f}")
    print(f"Normalized Train Std  (Expected ~1.0): {float(np.std(normed)):.6f}")
    print("=" * 80)
