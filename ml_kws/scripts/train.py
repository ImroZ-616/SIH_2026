"""
SIH 26172 - R2 ML/KWS Training Execution Script
Phase 8: Executes baseline training of Compact-KWS-CNN with class weighting,
monitors validation metrics, saves checkpoints, and exports all training artifacts.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

# Ensure sys.path includes repository root and ml_kws/src
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent
_SRC_DIR = _ML_KWS_DIR / "src"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CACHE_DIR, OUTPUTS_DIR
from model import build_compact_kws_cnn
from training import (
    APPROVED_CLASS_WEIGHTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_INITIAL_LR,
    DEFAULT_LR_REDUCE_FACTOR,
    DEFAULT_LR_REDUCE_PATIENCE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_LR,
    DEFAULT_WEIGHT_DECAY,
    RANDOM_SEED,
    ValidationMetricsCallback,
    compile_kws_model,
    compute_evaluation_metrics,
    create_tf_datasets,
    load_and_prepare_training_data,
    set_all_seeds,
)


def plot_training_curves(
    history_records: list,
    best_epoch: int,
    output_path: Path,
) -> None:
    """Generates clean publication-quality training curves."""
    epochs = [r["epoch"] for r in history_records]
    train_loss = [r["loss"] for r in history_records]
    val_loss = [r["val_loss"] for r in history_records]
    train_acc = [r["accuracy"] for r in history_records]
    val_acc = [r["val_accuracy"] for r in history_records]
    val_bal_acc = [r["val_balanced_accuracy"] for r in history_records]
    val_macro_f1 = [r["val_macro_f1"] for r in history_records]
    val_kw_recall = [r["val_keyword_recall"] for r in history_records]
    val_kw_f1 = [r["val_keyword_f1"] for r in history_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss Curve
    axes[0, 0].plot(epochs, train_loss, label="Train Loss (Weighted)", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, val_loss, label="Validation Loss", color="#d62728", lw=2)
    axes[0, 0].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    axes[0, 0].set_title("Training vs. Validation Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right")

    # 2. Accuracy Curve
    axes[0, 1].plot(epochs, train_acc, label="Train Accuracy", color="#1f77b4", lw=2)
    axes[0, 1].plot(epochs, val_acc, label="Validation Accuracy", color="#2ca02c", lw=2)
    axes[0, 1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    axes[0, 1].set_title("Training vs. Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="lower right")

    # 3. Validation Balanced Metrics
    axes[1, 0].plot(epochs, val_bal_acc, label="Val Balanced Accuracy", color="#ff7f0e", lw=2)
    axes[1, 0].plot(epochs, val_macro_f1, label="Val Macro F1", color="#9467bd", lw=2)
    axes[1, 0].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    axes[1, 0].set_title("Validation Balanced Performance", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="lower right")

    # 4. Keyword Specific Metrics (ASTRA)
    axes[1, 1].plot(epochs, val_kw_recall, label="Val ASTRA Recall (TPR)", color="#8c564b", lw=2)
    axes[1, 1].plot(epochs, val_kw_f1, label="Val ASTRA F1", color="#e377c2", lw=2)
    axes[1, 1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    axes[1, 1].set_title("Validation Keyword ('ASTRA') Metrics", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="lower right")

    plt.suptitle("SIH 26172 — Phase 8 Compact-KWS-CNN Training Performance", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
) -> None:
    """Renders and saves a 3x3 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix (Phase 8 Baseline)",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:d}\n({cm[i, j]/np.sum(cm[i, :])*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_training():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Baseline Model Training (Phase 8)")
    print("=" * 80)

    # 1. Ensure Reproducibility
    set_all_seeds(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Environment Info
    env_info = {
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "platform": platform.platform(),
        "seed": RANDOM_SEED,
    }
    print(f"Environment: Python {env_info['python_version']} | TF {env_info['tensorflow_version']} | Keras {env_info['keras_version']}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")

    # 3. Load Datasets & Compute Normalization (Train ONLY)
    train_npz = CACHE_DIR / "train_data.npz"
    val_npz = CACHE_DIR / "val_data.npz"
    norm_stats_file = OUTPUTS_DIR / "norm_stats.npz"

    print("\nLoading datasets and computing training normalization statistics...")
    X_train, y_train, X_val, y_val, mean_vec, std_vec = load_and_prepare_training_data(
        train_path=train_npz,
        val_path=val_npz,
        save_stats_path=norm_stats_file,
    )

    print(f"  - Training Set   : X={X_train.shape}, y={y_train.shape} (dtype: {X_train.dtype})")
    print(f"    * Keyword (2)  : {np.sum(y_train == 2)} samples")
    print(f"    * Unknown (1)  : {np.sum(y_train == 1)} samples")
    print(f"    * Silence (0)  : {np.sum(y_train == 0)} samples")
    print(f"  - Validation Set : X={X_val.shape}, y={y_val.shape} (dtype: {X_val.dtype})")
    print(f"    * Keyword (2)  : {np.sum(y_val == 2)} samples")
    print(f"    * Unknown (1)  : {np.sum(y_val == 1)} samples")
    print(f"    * Silence (0)  : {np.sum(y_val == 0)} samples")
    print(f"  - Normalization Stats Saved to: {norm_stats_file}")

    # Explicitly confirm test/negative_test isolation
    print("\nData Isolation Confirmation:")
    print("  [CONFIRMED] test_data.npz is 100% ISOLATED (Not loaded).")
    print("  [CONFIRMED] mfcc_negative_test.npz is 100% ISOLATED (Not loaded).")
    print("  [CONFIRMED] Normalization computed strictly from X_train.")

    # 4. Build tf.data Datasets
    train_ds, val_ds = create_tf_datasets(
        X_train, y_train, X_val, y_val,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    # 5. Build and Compile Model
    print("\nInstantiating Compact-KWS-CNN...")
    model = build_compact_kws_cnn(input_shape=(98, 13, 1), num_classes=3)
    model = compile_kws_model(
        model,
        learning_rate=DEFAULT_INITIAL_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    total_params = model.count_params()
    trainable_params = sum(p.numpy().size for p in model.trainable_variables)
    non_trainable_params = sum(p.numpy().size for p in model.non_trainable_variables)
    print(f"  - Model Name         : {model.name}")
    print(f"  - Total Parameters   : {total_params:,}")
    print(f"  - Trainable Params   : {trainable_params:,}")
    print(f"  - Non-Trainable      : {non_trainable_params:,}")

    # 6. Configure Callbacks
    best_model_path = OUTPUTS_DIR / "best_kws_model.keras"
    final_model_path = OUTPUTS_DIR / "final_kws_model.keras"

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1,
    )

    lr_scheduler_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=DEFAULT_LR_REDUCE_FACTOR,
        patience=DEFAULT_LR_REDUCE_PATIENCE,
        min_lr=DEFAULT_MIN_LR,
        verbose=1,
    )

    val_metrics_cb = ValidationMetricsCallback(X_val=X_val, y_val=y_val)

    callbacks_list = [ckpt_cb, early_stopping_cb, lr_scheduler_cb, val_metrics_cb]

    # 7. Execute Training Loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING LOOP (Class-Weighted Cross-Entropy)...")
    print(f"Class Weights: {APPROVED_CLASS_WEIGHTS}")
    print(f"Batch Size   : {DEFAULT_BATCH_SIZE}")
    print(f"Max Epochs   : {DEFAULT_MAX_EPOCHS}")
    print("=" * 80 + "\n")

    start_train_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=DEFAULT_MAX_EPOCHS,
        class_weight=APPROVED_CLASS_WEIGHTS,
        callbacks=callbacks_list,
        verbose=1,
    )

    training_duration = time.time() - start_train_time

    # 8. Save Final Model
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # 9. Determine Best Epoch
    val_losses = [r["val_loss"] for r in val_metrics_cb.epoch_metrics]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(val_losses[best_epoch_idx])

    print(f"\nBest Epoch: {best_epoch} (Validation Loss: {best_val_loss:.6f})")

    # 10. Load and Evaluate Best Model
    print(f"\nLoading best checkpoint model from: {best_model_path}")
    best_model = keras.models.load_model(str(best_model_path))
    val_preds = best_model.predict(X_val, verbose=0)
    final_val_metrics = compute_evaluation_metrics(y_val, val_preds)

    # 11. Generate and Save Visualizations
    curves_plot_path = OUTPUTS_DIR / "training_curves.png"
    cm_plot_path = OUTPUTS_DIR / "val_confusion_matrix.png"

    plot_training_curves(val_metrics_cb.epoch_metrics, best_epoch, curves_plot_path)
    plot_confusion_matrix(np.array(final_val_metrics["confusion_matrix"]), ["Silence", "Unknown", "Keyword"], cm_plot_path)
    print(f"Training curves plot saved to: {curves_plot_path}")
    print(f"Validation confusion matrix plot saved to: {cm_plot_path}")

    # 12. Save History & Config JSONs
    history_json_path = OUTPUTS_DIR / "training_history.json"
    config_json_path = OUTPUTS_DIR / "train_config.json"

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics_cb.epoch_metrics, f, indent=2)

    config_record = {
        "experiment_name": "Phase 8 Baseline Compact-KWS-CNN",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_info,
        "model_name": model.name,
        "model_params": {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        },
        "input_shape": [98, 13, 1],
        "class_mapping": {"silence": 0, "unknown": 1, "keyword": 2},
        "dataset_sample_counts": {
            "train": {"keyword": int(np.sum(y_train == 2)), "unknown": int(np.sum(y_train == 1)), "silence": int(np.sum(y_train == 0)), "total": len(y_train)},
            "val": {"keyword": int(np.sum(y_val == 2)), "unknown": int(np.sum(y_val == 1)), "silence": int(np.sum(y_val == 0)), "total": len(y_val)},
        },
        "optimizer": {
            "name": "AdamW",
            "initial_lr": DEFAULT_INITIAL_LR,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
        },
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "total_epochs_trained": len(val_losses),
        "early_stopping": {
            "monitor": "val_loss",
            "patience": DEFAULT_EARLY_STOPPING_PATIENCE,
            "min_delta": 1e-4,
            "restore_best_weights": True,
        },
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "val_loss",
            "factor": DEFAULT_LR_REDUCE_FACTOR,
            "patience": DEFAULT_LR_REDUCE_PATIENCE,
            "min_lr": DEFAULT_MIN_LR,
        },
        "class_weights": APPROVED_CLASS_WEIGHTS,
        "augmentation": "none",
        "normalization": {
            "method": "per-coefficient mean/std",
            "source": "X_train only",
            "artifact_path": str(norm_stats_file),
        },
        "results": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_metrics": final_val_metrics,
            "training_duration_sec": training_duration,
        },
    }

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    print(f"Training history saved to: {history_json_path}")
    print(f"Train configuration saved to: {config_json_path}")

    # 13. Print Comprehensive Phase 8 Final Report
    print("\n" + "=" * 80)
    print("PHASE 8 BASELINE TRAINING REPORT")
    print("=" * 80)
    print(f"Training Duration          : {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")
    print(f"Total Epochs Run           : {len(val_losses)} / {DEFAULT_MAX_EPOCHS}")
    print(f"Best Epoch                 : {best_epoch}")
    print(f"Best Validation Loss       : {best_val_loss:.6f}")
    print(f"Final Train Loss           : {history.history['loss'][-1]:.6f}")
    print(f"Final Train Accuracy       : {history.history['sparse_categorical_accuracy'][-1]*100:.2f}%")
    print("-" * 80)
    print("VALIDATION METRICS (AT BEST CHECKPOINT):")
    print(f"  - Overall Accuracy       : {final_val_metrics['accuracy']*100:.2f}% (Target >= 96%)")
    print(f"  - Balanced Accuracy      : {final_val_metrics['balanced_accuracy']*100:.2f}% (Target >= 95%)")
    print(f"  - Macro Precision        : {final_val_metrics['macro_precision']*100:.2f}%")
    print(f"  - Macro Recall           : {final_val_metrics['macro_recall']*100:.2f}%")
    print(f"  - Macro F1-Score         : {final_val_metrics['macro_f1']:.4f} (Target >= 0.95)")
    print("-" * 80)
    print("KEYWORD / 'ASTRA' METRICS (UNSEEN SPEAKER 'Imroz' - 35 samples):")
    print(f"  - ASTRA Precision        : {final_val_metrics['keyword_precision']*100:.2f}%")
    print(f"  - ASTRA Recall (TPR)     : {final_val_metrics['keyword_recall']*100:.2f}% (Target >= 95%)")
    print(f"  - ASTRA F1-Score         : {final_val_metrics['keyword_f1']:.4f}")
    print(f"  - ASTRA FRR              : {final_val_metrics['keyword_frr']*100:.2f}% (Target <= 5%)")
    print(f"  - Validation FAR         : {final_val_metrics['val_far']*100:.2f}% (439 non-keyword validation samples)")
    print("-" * 80)
    print("CONFUSION MATRIX (Validation Set):")
    print(f"  {'':<12} | {'Pred Silence':<14} | {'Pred Unknown':<14} | {'Pred Keyword':<14}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    cm_arr = np.array(final_val_metrics["confusion_matrix"])
    for i, name in enumerate(["True Silence", "True Unknown", "True Keyword"]):
        print(f"  {name:<12} | {cm_arr[i, 0]:<14} | {cm_arr[i, 1]:<14} | {cm_arr[i, 2]:<14}")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
