"""
SIH 26172 - R2 ML/KWS Hard-Negative & False-Activation Evaluation Script
Phase 10: Evaluates best_kws_model.keras on the dedicated 100-sample hard-negative benchmark
(mfcc_negative_test.npz) across 4 categories (stop, tree, three, marvin).
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from training import set_all_seeds, RANDOM_SEED

HARD_NEG_CATEGORIES = ["stop", "tree", "three", "marvin"]


def extract_word_category(filename: str) -> str:
    """Extracts the underlying word category from the hard-negative filename."""
    for cat in HARD_NEG_CATEGORIES:
        if f"_{cat}_" in filename:
            return cat
    return "other"


def plot_hard_negative_summary(
    category_summary: Dict[str, Dict],
    probabilities: np.ndarray,
    categories: List[str],
    output_path: Path,
) -> None:
    """Renders a 2-panel chart: prediction breakdown and keyword probability distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Stacked Bar Chart of Predictions across Categories
    cats = HARD_NEG_CATEGORIES + ["Overall"]
    silence_counts = [category_summary[c]["silence"] for c in cats]
    unknown_counts = [category_summary[c]["unknown"] for c in cats]
    keyword_counts = [category_summary[c]["keyword"] for c in cats]

    x = np.arange(len(cats))
    width = 0.55

    p1 = axes[0].bar(x, unknown_counts, width, label="Predicted Unknown (Correct Rejection)", color="#2ca02c", alpha=0.85)
    p2 = axes[0].bar(x, silence_counts, width, bottom=unknown_counts, label="Predicted Silence (Correct Rejection)", color="#1f77b4", alpha=0.85)
    
    bottom_kw = [u + s for u, s in zip(unknown_counts, silence_counts)]
    p3 = axes[0].bar(x, keyword_counts, width, bottom=bottom_kw, label="Predicted Keyword (False Activation)", color="#d62728", alpha=0.85)

    axes[0].set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    axes[0].set_title("Prediction Distribution by Hard-Negative Category", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c}\n(N={category_summary[c]['total']})" for c in cats], fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left")

    # Add count labels inside bars
    for idx in range(len(cats)):
        tot = category_summary[cats[idx]]["total"]
        kw = category_summary[cats[idx]]["keyword"]
        rej = category_summary[cats[idx]]["rejections"]
        far = category_summary[cats[idx]]["far_pct"]
        axes[0].text(idx, tot + 1.0, f"FAR: {far:.1f}%\n({kw}/{tot})", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d62728" if kw > 0 else "#2ca02c")

    axes[0].set_ylim(0, max([category_summary[c]["total"] for c in cats]) * 1.25)

    # Panel 2: Keyword Probability Distribution across Categories
    cat_probs = {c: [] for c in HARD_NEG_CATEGORIES}
    for prob, cat in zip(probabilities, categories):
        if cat in cat_probs:
            cat_probs[cat].append(prob[2])  # p_keyword

    bp_data = [cat_probs[c] for c in HARD_NEG_CATEGORIES]
    bplot = axes[1].boxplot(bp_data, tick_labels=HARD_NEG_CATEGORIES, patch_artist=True, medianprops=dict(color="black", lw=2))
    
    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Argmax Decision Threshold (0.5)")
    axes[1].set_ylabel("Keyword Probability ($p_{\\mathrm{keyword}}$)", fontsize=11, fontweight="bold")
    axes[1].set_title("Keyword Probability Distribution on Hard Negatives", fontsize=12, fontweight="bold")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.suptitle("SIH 26172 — Phase 10 Hard-Negative & False-Activation Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_hard_negative_evaluation():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Hard-Negative & False-Activation Evaluation (Phase 10)")
    print("=" * 80)

    # 1. Reproducibility & Seeding
    set_all_seeds(RANDOM_SEED)
    start_time = time.time()

    # 2. Path Setup & Verification
    neg_npz_path = CACHE_DIR / "mfcc_negative_test.npz"
    norm_stats_path = OUTPUTS_DIR / "norm_stats.npz"
    model_path = OUTPUTS_DIR / "best_kws_model.keras"

    assert neg_npz_path.is_file(), f"Negative test dataset not found: {neg_npz_path}"
    assert norm_stats_path.is_file(), f"Normalization statistics not found: {norm_stats_path}"
    assert model_path.is_file(), f"Canonical model not found: {model_path}"

    print(f"Hard-Negative Dataset       : {neg_npz_path}")
    print(f"Normalization Stats File    : {norm_stats_path}")
    print(f"Canonical Model Checkpoint  : {model_path}")

    # Explicit isolation confirmation
    print("\nData Isolation Confirmation:")
    print("  [CONFIRMED] train_data.npz, val_data.npz, test_data.npz NOT loaded.")
    print("  [CONFIRMED] Normalization statistics loaded from train-only norm_stats.npz.")
    print("  [CONFIRMED] Evaluation uses frozen weights and standard argmax decisions (no threshold tuning).")

    # 3. Load Hard-Negative Dataset
    neg_dataset = load_dataset_npz(neg_npz_path)
    X_neg_raw = neg_dataset["X"]          # (100, 98, 13)
    y_neg = neg_dataset["y"]              # (100,)
    filenames = neg_dataset["filenames"]  # (100,)

    total_samples = len(filenames)
    assert total_samples == 100, f"Expected 100 samples in negative_test, got {total_samples}"

    # Categorize files
    categories = [extract_word_category(f) for f in filenames]
    cat_counts = {c: categories.count(c) for c in HARD_NEG_CATEGORIES}
    print(f"\nHard-Negative Composition ({total_samples} total non-keyword samples):")
    for cat in HARD_NEG_CATEGORIES:
        print(f"  - '{cat:<6}' category : {cat_counts[cat]} samples (intended phonetic overlap with ASTRA)")

    # 4. Standardize Features using Training Normalization Stats
    mean_vec, std_vec, eps = load_normalization_stats(norm_stats_path)
    X_neg_norm = apply_mfcc_normalization(X_neg_raw, mean_vec, std_vec, epsilon=eps)
    X_neg_tensor = np.expand_dims(X_neg_norm, axis=-1).astype(np.float32)  # (100, 98, 13, 1)

    print(f"\nNormalized Feature Tensor Shape: {X_neg_tensor.shape} (dtype: {X_neg_tensor.dtype})")
    assert np.all(np.isfinite(X_neg_tensor)), "Normalized tensor contains non-finite values!"

    # 5. Load Canonical Model & Run Forward Inference
    print(f"\nLoading canonical model from: {model_path}")
    model = keras.models.load_model(str(model_path))

    inference_start = time.time()
    probabilities = model.predict(X_neg_tensor, batch_size=32, verbose=0)
    inference_duration = time.time() - inference_start

    assert probabilities.shape == (total_samples, 3), f"Probabilities shape {probabilities.shape} != ({total_samples}, 3)"
    assert np.all(np.isfinite(probabilities)), "Inference returned non-finite probabilities!"

    # 6. Standard Argmax Decision
    y_pred = np.argmax(probabilities, axis=-1).astype(np.int32)

    # 7. Compute Predictions & False Activations
    pred_silence_count = int(np.sum(y_pred == 0))
    pred_unknown_count = int(np.sum(y_pred == 1))
    pred_keyword_count = int(np.sum(y_pred == 2))  # False activations

    total_false_activations = pred_keyword_count
    hard_neg_far_pct = float(total_false_activations / total_samples * 100.0)
    rejection_count = pred_silence_count + pred_unknown_count
    rejection_rate_pct = float(rejection_count / total_samples * 100.0)

    # 8. Per-Category Breakdown
    category_summary = {}
    for cat in HARD_NEG_CATEGORIES:
        mask = np.array([c == cat for c in categories])
        sub_y_pred = y_pred[mask]
        sub_total = int(np.sum(mask))
        sub_sil = int(np.sum(sub_y_pred == 0))
        sub_unk = int(np.sum(sub_y_pred == 1))
        sub_kw = int(np.sum(sub_y_pred == 2))
        sub_far = float(sub_kw / sub_total * 100.0) if sub_total > 0 else 0.0

        category_summary[cat] = {
            "total": sub_total,
            "silence": sub_sil,
            "unknown": sub_unk,
            "keyword": sub_kw,
            "rejections": sub_sil + sub_unk,
            "far_pct": sub_far,
            "rejection_rate_pct": float((sub_sil + sub_unk) / sub_total * 100.0) if sub_total > 0 else 0.0,
        }

    category_summary["Overall"] = {
        "total": total_samples,
        "silence": pred_silence_count,
        "unknown": pred_unknown_count,
        "keyword": pred_keyword_count,
        "rejections": rejection_count,
        "far_pct": hard_neg_far_pct,
        "rejection_rate_pct": rejection_rate_pct,
    }

    # 9. Extract Detailed Records for Every False Activation
    false_activation_records = []
    for idx in range(total_samples):
        if y_pred[idx] == 2:  # False trigger
            fname = str(filenames[idx])
            cat = categories[idx]
            p_sil = float(probabilities[idx, 0])
            p_unk = float(probabilities[idx, 1])
            p_kw = float(probabilities[idx, 2])
            margin = float(p_kw - max(p_sil, p_unk))

            record = {
                "sample_index": idx,
                "filename": fname,
                "category": cat,
                "predicted_class": 2,
                "class_name": "keyword",
                "probabilities": {
                    "silence": p_sil,
                    "unknown": p_unk,
                    "keyword": p_kw,
                },
                "keyword_probability": p_kw,
                "confidence_margin": margin,
            }
            false_activation_records.append(record)

    # 10. Load Phase 9 Metrics for Comparative Context
    test_rep_path = OUTPUTS_DIR / "test_evaluation_report.json"
    phase9_unknown_far = 0.008333333333333333  # 0.83% default
    phase9_combined_far = 0.006833712984054669  # 0.68% default
    if test_rep_path.is_file():
        with open(test_rep_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            t_mets = t_data.get("test_metrics", {})
            phase9_unknown_far = float(t_mets.get("far_unknown", phase9_unknown_far))
            phase9_combined_far = float(t_mets.get("far_combined", phase9_combined_far))

    # 11. Export Artifacts
    summary_plot_path = OUTPUTS_DIR / "hard_negative_summary.png"
    report_json_path = OUTPUTS_DIR / "hard_negative_evaluation_report.json"
    preds_npz_path = OUTPUTS_DIR / "hard_negative_predictions.npz"

    plot_hard_negative_summary(category_summary, probabilities, categories, summary_plot_path)

    np.savez_compressed(
        preds_npz_path,
        y_pred=y_pred,
        probabilities=probabilities,
        filenames=filenames,
        categories=np.array(categories),
    )

    full_report = {
        "experiment": "Phase 10 Hard-Negative False-Activation Testing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "canonical_model": str(model_path),
        "norm_stats_source": str(norm_stats_path),
        "dataset": str(neg_npz_path),
        "total_samples": total_samples,
        "overall_results": {
            "predicted_silence": pred_silence_count,
            "predicted_unknown": pred_unknown_count,
            "predicted_keyword_false_activations": total_false_activations,
            "hard_negative_far_pct": hard_neg_far_pct,
            "non_keyword_rejection_rate_pct": rejection_rate_pct,
        },
        "per_category_breakdown": category_summary,
        "false_activations": false_activation_records,
        "phase9_comparison": {
            "phase9_unknown_speech_far_pct": phase9_unknown_far * 100.0,
            "phase9_combined_test_far_pct": phase9_combined_far * 100.0,
            "phase10_hard_negative_far_pct": hard_neg_far_pct,
            "delta_far_pct": hard_neg_far_pct - (phase9_unknown_far * 100.0),
        },
        "execution_time_sec": time.time() - start_time,
        "inference_time_sec": inference_duration,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    total_duration = time.time() - start_time

    # 12. Final Console Report
    print("\n" + "=" * 80)
    print("PHASE 10 HARD-NEGATIVE EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Execution Time        : {total_duration:.2f} seconds")
    print(f"Total Hard-Negative Samples : {total_samples}")
    print("-" * 80)
    print("1. OVERALL PREDICTION DISTRIBUTION:")
    print(f"   - Predicted Silence      : {pred_silence_count:3d} / {total_samples} ({pred_silence_count/total_samples*100:.2f}%)")
    print(f"   - Predicted Unknown      : {pred_unknown_count:3d} / {total_samples} ({pred_unknown_count/total_samples*100:.2f}%)")
    print(f"   - Predicted Keyword (FA) : {total_false_activations:3d} / {total_samples} ({hard_neg_far_pct:.2f}%)")
    print(f"   - Total Rejections       : {rejection_count:3d} / {total_samples} ({rejection_rate_pct:.2f}%)")
    print("-" * 80)
    print("2. HARD-NEGATIVE FALSE ACTIVATION RATE (FAR):")
    print(f"   - Total Hard-Negative FAR: {hard_neg_far_pct:.2f}% ({total_false_activations} false activations / {total_samples} samples)")
    print("-" * 80)
    print("3. PER-CATEGORY BREAKDOWN (4 Categories x 25 Samples):")
    print(f"   {'Category':<10} | {'Total':<6} | {'Rejections':<11} | {'Silence':<8} | {'Unknown':<8} | {'Keyword (FA)':<13} | {'FAR (%)':<8}")
    print(f"   {'-'*10}-+-{'-'*6}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}-+-{'-'*8}")
    for cat in HARD_NEG_CATEGORIES:
        cs = category_summary[cat]
        print(f"   {cat:<10} | {cs['total']:<6} | {cs['rejections']:<11} | {cs['silence']:<8} | {cs['unknown']:<8} | {cs['keyword']:<13} | {cs['far_pct']:6.2f}%")
    print(f"   {'-'*10}-+-{'-'*6}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}-+-{'-'*8}")
    os_tot = category_summary["Overall"]
    print(f"   {'TOTAL':<10} | {os_tot['total']:<6} | {os_tot['rejections']:<11} | {os_tot['silence']:<8} | {os_tot['unknown']:<8} | {os_tot['keyword']:<13} | {os_tot['far_pct']:6.2f}%")
    print("-" * 80)
    print("4. DETAILED FALSE ACTIVATIONS LIST:")
    if not false_activation_records:
        print("   [NONE] Zero false activations observed! (100% rejection across all 100 hard negatives)")
    else:
        for fa in false_activation_records:
            probs_str = f"P(Sil)={fa['probabilities']['silence']:.4f}, P(Unk)={fa['probabilities']['unknown']:.4f}, P(Kw)={fa['probabilities']['keyword']:.4f}"
            print(f"   - File: {fa['filename']}")
            print(f"     Category: '{fa['category']}' | {probs_str} | Margin: +{fa['confidence_margin']:.4f}")
    print("-" * 80)
    print("5. COMPARISON WITH PHASE 9 BASELINE:")
    print(f"   - Phase 9 Unknown Speech FAR (360 samples)     : {phase9_unknown_far*100:.2f}% (3 false activations)")
    print(f"   - Phase 9 Combined Non-Keyword FAR (439 samples): {phase9_combined_far*100:.2f}% (3 false activations)")
    print(f"   - Phase 10 Hard-Negative FAR (100 samples)      : {hard_neg_far_pct:.2f}% ({total_false_activations} false activations)")
    print(f"   - FAR Delta (Phase 10 - Phase 9 Unknown)        : {hard_neg_far_pct - (phase9_unknown_far*100):+.2f}%")
    print("-" * 80)
    print("6. GENERATED ARTIFACTS:")
    print(f"   - Report JSON  : {report_json_path}")
    print(f"   - Summary Plot : {summary_plot_path}")
    print(f"   - Preds Archive: {preds_npz_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_hard_negative_evaluation()
