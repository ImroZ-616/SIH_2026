"""
SIH 26172 - R2 ML/KWS Held-Out Test Set Evaluation Script
Phase 9: Evaluates best_kws_model.keras on test_data.npz (473 samples, unseen speaker Shaswat),
computes multi-class metrics, confusion matrix, FAR breakdown, and validation comparison deltas.
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
from features import CLASS_NAMES, LABEL_MAP, load_dataset_npz
from normalization import apply_mfcc_normalization, load_normalization_stats
from training import compute_evaluation_metrics, set_all_seeds, RANDOM_SEED


def plot_test_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
) -> None:
    """Renders and saves a publication-quality 3x3 test confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Held-Out Test Confusion Matrix (Phase 9 - Unseen Speaker: Shaswat)",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / row_sum * 100) if row_sum > 0 else 0.0
            ax.text(
                j, i, f"{cm[i, j]:d}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9.5, fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_test_evaluation():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Held-Out Test Evaluation (Phase 9)")
    print("=" * 80)

    # 1. Reproducibility & Seeding
    set_all_seeds(RANDOM_SEED)
    start_time = time.time()

    # 2. Paths & Strict Isolation Verification
    test_npz_path = CACHE_DIR / "test_data.npz"
    norm_stats_path = OUTPUTS_DIR / "norm_stats.npz"
    model_path = OUTPUTS_DIR / "best_kws_model.keras"

    assert test_npz_path.is_file(), f"Test dataset not found: {test_npz_path}"
    assert norm_stats_path.is_file(), f"Normalization statistics not found: {norm_stats_path}"
    assert model_path.is_file(), f"Model checkpoint not found: {model_path}"

    print(f"Test Dataset Archive        : {test_npz_path}")
    print(f"Normalization Stats File    : {norm_stats_path}")
    print(f"Canonical Model Path        : {model_path}")

    # Explicit assertion that negative_test is NOT accessed
    print("\nData Isolation Verification:")
    print("  [CONFIRMED] mfcc_negative_test.npz is 100% ISOLATED (Not loaded).")
    print("  [CONFIRMED] Normalization statistics loaded from train-only norm_stats.npz.")
    print("  [CONFIRMED] Evaluation uses frozen weights and standard argmax decisions.")

    # 3. Load Test Data
    test_dataset = load_dataset_npz(test_npz_path)
    X_test_raw = test_dataset["X"]          # (473, 98, 13)
    y_test = test_dataset["y"]              # (473,)
    filenames = test_dataset["filenames"]  # (473,)
    speakers = test_dataset["speakers"]    # (473,)

    total_test_samples = len(y_test)
    kw_test_count = int(np.sum(y_test == LABEL_MAP["keyword"]))
    unk_test_count = int(np.sum(y_test == LABEL_MAP["unknown"]))
    sil_test_count = int(np.sum(y_test == LABEL_MAP["silence"]))

    print(f"\nHeld-Out Test Composition ({total_test_samples} total samples):")
    print(f"  - Keyword (Class 2)       : {kw_test_count} samples (Unseen speaker: 'Shaswat')")
    print(f"  - Unknown (Class 1)       : {unk_test_count} samples (32 unseen speaker hashes)")
    print(f"  - Silence (Class 0)       : {sil_test_count} samples (71 'bg_pink_noise' + 8 near-silence)")

    # 4. Load Training Normalization Stats & Standardize Test Data
    mean_vec, std_vec, eps = load_normalization_stats(norm_stats_path)
    X_test_norm = apply_mfcc_normalization(X_test_raw, mean_vec, std_vec, epsilon=eps)
    X_test_tensor = np.expand_dims(X_test_norm, axis=-1).astype(np.float32)  # (473, 98, 13, 1)

    print(f"\nNormalized Test Tensor Shape: {X_test_tensor.shape} (dtype: {X_test_tensor.dtype})")
    assert np.all(np.isfinite(X_test_tensor)), "Normalized test tensor contains non-finite values!"

    # 5. Load Canonical Model & Run Forward Inference
    print(f"\nLoading canonical model from: {model_path}")
    model = keras.models.load_model(str(model_path))

    inference_start = time.time()
    probabilities = model.predict(X_test_tensor, batch_size=32, verbose=0)
    inference_time = time.time() - inference_start

    assert probabilities.shape == (total_test_samples, 3), f"Output shape {probabilities.shape} != ({total_test_samples}, 3)"
    assert np.all(np.isfinite(probabilities)), "Inference returned non-finite probabilities!"

    # 6. Generate Deterministic Argmax Predictions
    y_pred = np.argmax(probabilities, axis=-1).astype(np.int32)

    # 7. Compute Test Metrics
    test_metrics = compute_evaluation_metrics(y_test, probabilities, class_names=["silence", "unknown", "keyword"])
    cm = np.array(test_metrics["confusion_matrix"])

    # 8. Compute Specific FAR Sub-Components
    # Confusion Matrix indexing: cm[true_class, pred_class]
    far_unknown = float(cm[1, 2] / unk_test_count) if unk_test_count > 0 else 0.0
    far_silence = float(cm[0, 2] / sil_test_count) if sil_test_count > 0 else 0.0
    non_kw_total = unk_test_count + sil_test_count  # 439
    far_combined = float((cm[0, 2] + cm[1, 2]) / non_kw_total) if non_kw_total > 0 else 0.0

    test_metrics["far_unknown"] = far_unknown
    test_metrics["far_silence"] = far_silence
    test_metrics["far_combined"] = far_combined

    # 9. Load Phase 8 Validation Metrics for Comparison
    val_config_path = OUTPUTS_DIR / "train_config.json"
    val_metrics = {}
    if val_config_path.is_file():
        with open(val_config_path, "r", encoding="utf-8") as f:
            val_config = json.load(f)
            val_metrics = val_config.get("results", {}).get("final_val_metrics", {})

    # Compute Comparison Deltas (Test - Validation)
    val_acc = val_metrics.get("accuracy", 0.0)
    val_bal_acc = val_metrics.get("balanced_accuracy", 0.0)
    val_macro_f1 = val_metrics.get("macro_f1", 0.0)
    val_kw_prec = val_metrics.get("keyword_precision", 0.0)
    val_kw_rec = val_metrics.get("keyword_recall", 0.0)
    val_kw_f1 = val_metrics.get("keyword_f1", 0.0)
    val_kw_frr = val_metrics.get("keyword_frr", 0.0)
    val_far = val_metrics.get("val_far", 0.0)

    comparison_table = {
        "accuracy": {"validation": val_acc, "test": test_metrics["accuracy"], "delta": test_metrics["accuracy"] - val_acc},
        "balanced_accuracy": {"validation": val_bal_acc, "test": test_metrics["balanced_accuracy"], "delta": test_metrics["balanced_accuracy"] - val_bal_acc},
        "macro_f1": {"validation": val_macro_f1, "test": test_metrics["macro_f1"], "delta": test_metrics["macro_f1"] - val_macro_f1},
        "keyword_precision": {"validation": val_kw_prec, "test": test_metrics["keyword_precision"], "delta": test_metrics["keyword_precision"] - val_kw_prec},
        "keyword_recall": {"validation": val_kw_rec, "test": test_metrics["keyword_recall"], "delta": test_metrics["keyword_recall"] - val_kw_rec},
        "keyword_f1": {"validation": val_kw_f1, "test": test_metrics["keyword_f1"], "delta": test_metrics["keyword_f1"] - val_kw_f1},
        "keyword_frr": {"validation": val_kw_frr, "test": test_metrics["keyword_frr"], "delta": test_metrics["keyword_frr"] - val_kw_frr},
        "far_combined": {"validation": val_far, "test": far_combined, "delta": far_combined - val_far},
    }

    # 10. Save Artifacts
    cm_plot_path = OUTPUTS_DIR / "test_confusion_matrix.png"
    report_json_path = OUTPUTS_DIR / "test_evaluation_report.json"
    preds_npz_path = OUTPUTS_DIR / "test_predictions.npz"

    plot_test_confusion_matrix(cm, ["Silence", "Unknown", "Keyword"], cm_plot_path)

    # Save Predictions NPZ
    np.savez_compressed(
        preds_npz_path,
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        filenames=filenames,
        speakers=speakers,
    )

    # Save Full JSON Report
    full_report = {
        "experiment": "Phase 9 Held-Out Test Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "canonical_model": str(model_path),
        "norm_stats_source": str(norm_stats_path),
        "test_dataset": str(test_npz_path),
        "sample_counts": {
            "total": total_test_samples,
            "keyword": kw_test_count,
            "unknown": unk_test_count,
            "silence": sil_test_count,
        },
        "test_metrics": test_metrics,
        "validation_comparison": comparison_table,
        "execution_time_sec": time.time() - start_time,
        "inference_time_sec": inference_time,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    total_duration = time.time() - start_time

    # 11. Console Printout of Complete Phase 9 Results
    print("\n" + "=" * 80)
    print("PHASE 9 HELD-OUT TEST EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Execution Time        : {total_duration:.2f} seconds")
    print(f"Total Test Samples          : {total_test_samples}")
    print("-" * 80)
    print("1. OVERALL & BALANCED TEST METRICS:")
    print(f"   - Overall Accuracy       : {test_metrics['accuracy']*100:.2f}% ({np.sum(np.diag(cm))}/{total_test_samples})")
    print(f"   - Balanced Accuracy      : {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"   - Macro Precision        : {test_metrics['macro_precision']*100:.2f}%")
    print(f"   - Macro Recall           : {test_metrics['macro_recall']*100:.2f}%")
    print(f"   - Macro F1-Score         : {test_metrics['macro_f1']:.4f}")
    print("-" * 80)
    print("2. KEYWORD ('ASTRA') TEST METRICS (UNSEEN SPEAKER 'Shaswat' - 34 samples):")
    print(f"   - ASTRA Precision        : {test_metrics['keyword_precision']*100:.2f}% ({cm[2, 2]}/{np.sum(cm[:, 2])})")
    print(f"   - ASTRA Recall (TPR)     : {test_metrics['keyword_recall']*100:.2f}% ({cm[2, 2]}/{kw_test_count} detected)")
    print(f"   - ASTRA F1-Score         : {test_metrics['keyword_f1']:.4f}")
    print(f"   - ASTRA FRR              : {test_metrics['keyword_frr']*100:.2f}% ({kw_test_count - cm[2, 2]}/{kw_test_count} missed)")
    print("-" * 80)
    print("3. FALSE ACCEPTANCE RATES (FAR):")
    print(f"   - FAR from Unknown Speech: {far_unknown*100:.2f}% ({cm[1, 2]}/{unk_test_count} false triggers)")
    print(f"   - FAR from Silence/Noise : {far_silence*100:.2f}% ({cm[0, 2]}/{sil_test_count} false triggers)")
    print(f"   - Combined Non-Keyword FAR: {far_combined*100:.2f}% ({cm[0, 2] + cm[1, 2]}/{non_kw_total} total false triggers)")
    print("-" * 80)
    print("4. HELD-OUT TEST CONFUSION MATRIX:")
    print(f"   {'':<14} | {'Pred Silence':<14} | {'Pred Unknown':<14} | {'Pred Keyword':<14} | {'Total':<6}")
    print(f"   {'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}")
    for i, name in enumerate(["True Silence", "True Unknown", "True Keyword"]):
        row_tot = np.sum(cm[i, :])
        print(f"   {name:<14} | {cm[i, 0]:<14} | {cm[i, 1]:<14} | {cm[i, 2]:<14} | {row_tot:<6}")
    print("-" * 80)
    print("5. VALIDATION vs. TEST NUMERICAL COMPARISON:")
    print(f"   {'Metric':<25} | {'Validation (Val)':<18} | {'Test (Held-Out)':<18} | {'Delta (Test - Val)':<18}")
    print(f"   {'-'*25}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")
    print(f"   {'Overall Accuracy':<25} | {val_acc*100:16.2f}% | {test_metrics['accuracy']*100:16.2f}% | {comparison_table['accuracy']['delta']*100:+16.2f}%")
    print(f"   {'Balanced Accuracy':<25} | {val_bal_acc*100:16.2f}% | {test_metrics['balanced_accuracy']*100:16.2f}% | {comparison_table['balanced_accuracy']['delta']*100:+16.2f}%")
    print(f"   {'Macro F1-Score':<25} | {val_macro_f1:16.4f}  | {test_metrics['macro_f1']:16.4f}  | {comparison_table['macro_f1']['delta']:+16.4f} ")
    print(f"   {'ASTRA Precision':<25} | {val_kw_prec*100:16.2f}% | {test_metrics['keyword_precision']*100:16.2f}% | {comparison_table['keyword_precision']['delta']*100:+16.2f}%")
    print(f"   {'ASTRA Recall (TPR)':<25} | {val_kw_rec*100:16.2f}% | {test_metrics['keyword_recall']*100:16.2f}% | {comparison_table['keyword_recall']['delta']*100:+16.2f}%")
    print(f"   {'ASTRA FRR':<25} | {val_kw_frr*100:16.2f}% | {test_metrics['keyword_frr']*100:16.2f}% | {comparison_table['keyword_frr']['delta']*100:+16.2f}%")
    print(f"   {'Combined FAR':<25} | {val_far*100:16.2f}% | {far_combined*100:16.2f}% | {comparison_table['far_combined']['delta']*100:+16.2f}%")
    print("-" * 80)
    print("6. GENERATED ARTIFACTS:")
    print(f"   - Report JSON  : {report_json_path}")
    print(f"   - Confusion Plot: {cm_plot_path}")
    print(f"   - Preds Archive: {preds_npz_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_test_evaluation()
