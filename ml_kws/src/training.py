"""
SIH 26172 - R2 ML/KWS Training Module
Phase 8: Core training pipeline, dataset preparation, loss weighting,
custom validation metrics callback, and model evaluation utilities.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import keras
from keras import callbacks, layers, optimizers

# Ensure sys.path includes repository root and ml_kws/src
_CURRENT_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from config import CACHE_DIR, OUTPUTS_DIR
from features import CLASS_NAMES, LABEL_MAP, load_dataset_npz
from model import build_compact_kws_cnn
from normalization import (
    apply_mfcc_normalization,
    compute_mfcc_normalization_stats,
    save_normalization_stats,
)

# Standard Approved Hyperparameters for Phase 8 Run 1
RANDOM_SEED: int = 42
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_MAX_EPOCHS: int = 60
DEFAULT_INITIAL_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_EARLY_STOPPING_PATIENCE: int = 15
DEFAULT_LR_REDUCE_PATIENCE: int = 5
DEFAULT_LR_REDUCE_FACTOR: float = 0.5
DEFAULT_MIN_LR: float = 1e-5

# Exact Approved Class Weights for 3-Class Inverse Frequency
APPROVED_CLASS_WEIGHTS: Dict[int, float] = {
    0: 2.2578616,  # Silence (318 samples)
    1: 0.4273810,  # Unknown (1680 samples)
    2: 4.6025641,  # Keyword / ASTRA (156 samples)
}


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def load_and_prepare_training_data(
    train_path: Union[str, Path] = CACHE_DIR / "train_data.npz",
    val_path: Union[str, Path] = CACHE_DIR / "val_data.npz",
    save_stats_path: Optional[Union[str, Path]] = OUTPUTS_DIR / "norm_stats.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads train and val partitions, computes normalization stats strictly from train,

    and normalizes both partitions in-memory.

    Parameters
    ----------
    train_path : Union[str, Path]
        Path to cached train_data.npz.
    val_path : Union[str, Path]
        Path to cached val_data.npz.
    save_stats_path : Optional[Union[str, Path]]
        Path to save the computed training normalization statistics.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train_norm (N, 98, 13, 1), y_train (N,),
        X_val_norm (M, 98, 13, 1), y_val (M,),
        mean (1, 1, 13), std (1, 1, 13)
    """
    train_p = Path(train_path)
    val_p = Path(val_path)

    if not train_p.is_file():
        raise FileNotFoundError(f"Training dataset not found: {train_p}")
    if not val_p.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {val_p}")

    # Explicit assertion of file paths to ensure held-out sets are NOT touched
    assert "test_data" not in train_p.name and "negative_test" not in train_p.name, (
        "CRITICAL ERROR: Attempted to load test set in training loader!"
    )
    assert "test_data" not in val_p.name and "negative_test" not in val_p.name, (
        "CRITICAL ERROR: Attempted to load test set in validation loader!"
    )

    train_data = load_dataset_npz(train_p)
    val_data = load_dataset_npz(val_p)

    X_train = train_data["X"]  # (2154, 98, 13)
    y_train = train_data["y"]  # (2154,)
    X_val = val_data["X"]      # (474, 98, 13)
    y_val = val_data["y"]      # (474,)

    # 1. Compute per-coefficient mean and std STRICTLY from training set
    mean_vec, std_vec = compute_mfcc_normalization_stats(X_train)

    # 2. Save training statistics if requested
    if save_stats_path is not None:
        save_normalization_stats(save_stats_path, mean_vec, std_vec)

    # 3. Apply normalization in memory
    X_train_norm = apply_mfcc_normalization(X_train, mean_vec, std_vec)
    X_val_norm = apply_mfcc_normalization(X_val, mean_vec, std_vec)

    # 4. Add channel dimension (N, 98, 13) -> (N, 98, 13, 1)
    X_train_norm = np.expand_dims(X_train_norm, axis=-1).astype(np.float32)
    X_val_norm = np.expand_dims(X_val_norm, axis=-1).astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)

    return X_train_norm, y_train, X_val_norm, y_val, mean_vec, std_vec


def create_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Builds efficient tf.data pipeline for training and validation."""
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def compile_kws_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_INITIAL_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> keras.Model:
    """Compiles the model with AdamW optimizer and sparse categorical crossentropy."""
    optimizer = optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    metrics = ["sparse_categorical_accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    return model


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, int, List, Dict]]:
    """Calculates full multi-class classification and KWS-specific metrics."""
    if class_names is None:
        class_names = ["silence", "unknown", "keyword"]

    y_pred = np.argmax(y_pred_probs, axis=-1)
    num_classes = len(class_names)
    total_samples = len(y_true)

    # 1. Compute 3x3 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # 2. Overall Accuracy
    accuracy = float(np.sum(np.diag(cm)) / total_samples)

    # 3. Per-Class Metrics
    precisions = []
    recalls = []
    f1s = []
    per_class_metrics = {}

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        tn = int(total_samples - (tp + fp + fn))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        c_name = class_names[c]
        per_class_metrics[c_name] = {
            "class_id": c,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(np.sum(cm[c, :])),
        }

    # 4. Macro & Balanced Averages
    balanced_acc = float(np.mean(recalls))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # 5. Keyword Specific Metrics (Class 2: keyword / ASTRA)
    kw_c = LABEL_MAP["keyword"]
    kw_prec = float(precisions[kw_c])
    kw_recall = float(recalls[kw_c])
    kw_f1 = float(f1s[kw_c])
    kw_frr = float(1.0 - kw_recall)

    # Validation Set False Acceptance Rate (Non-keyword samples incorrectly classified as keyword)
    non_kw_samples = int(total_samples - np.sum(cm[kw_c, :]))
    kw_fp = int(np.sum(cm[:, kw_c]) - cm[kw_c, kw_c])
    val_far = float(kw_fp / non_kw_samples) if non_kw_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "keyword_precision": kw_prec,
        "keyword_recall": kw_recall,
        "keyword_f1": kw_f1,
        "keyword_frr": kw_frr,
        "val_far": val_far,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
        "total_samples": total_samples,
    }


class ValidationMetricsCallback(callbacks.Callback):
    """Custom callback to compute and log full validation metrics after each epoch."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.epoch_metrics: List[Dict] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        preds = self.model.predict(self.X_val, verbose=0)
        metrics = compute_evaluation_metrics(self.y_val, preds)
        
        # Inject custom metrics into logs for history tracking
        logs = logs or {}
        logs["val_balanced_accuracy"] = metrics["balanced_accuracy"]
        logs["val_macro_f1"] = metrics["macro_f1"]
        logs["val_keyword_recall"] = metrics["keyword_recall"]
        logs["val_keyword_precision"] = metrics["keyword_precision"]
        logs["val_keyword_f1"] = metrics["keyword_f1"]
        logs["val_keyword_frr"] = metrics["keyword_frr"]
        logs["val_far"] = metrics["val_far"]

        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("sparse_categorical_accuracy", logs.get("accuracy", 0.0))),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_sparse_categorical_accuracy", logs.get("val_accuracy", 0.0))),
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_macro_precision": metrics["macro_precision"],
            "val_macro_recall": metrics["macro_recall"],
            "val_macro_f1": metrics["macro_f1"],
            "val_keyword_precision": metrics["keyword_precision"],
            "val_keyword_recall": metrics["keyword_recall"],
            "val_keyword_f1": metrics["keyword_f1"],
            "val_keyword_frr": metrics["keyword_frr"],
            "val_far": metrics["val_far"],
            "lr": float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer, "learning_rate") else 0.0,
        }
        self.epoch_metrics.append(epoch_record)


if __name__ == "__main__":
    print("=" * 80)
    print("SIH 26172 - ML/KWS Training Module (Phase 8)")
    print("=" * 80)
    print(f"Random Seed           : {RANDOM_SEED}")
    print(f"Class Weights         : {APPROVED_CLASS_WEIGHTS}")
    print(f"Default Batch Size    : {DEFAULT_BATCH_SIZE}")
    print(f"Default Max Epochs    : {DEFAULT_MAX_EPOCHS}")
    print(f"Default Initial LR    : {DEFAULT_INITIAL_LR}")
    print(f"Default Weight Decay  : {DEFAULT_WEIGHT_DECAY}")
    print("=" * 80)
