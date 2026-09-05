"""
SIH 26172 - R2 ML/KWS MFCC Visualization Script
Phase 5: Visual Inspection of 2D MFCC Feature Matrices (98, 13) from Cached Dataset
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
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

from config import CACHE_DIR, OUTPUTS_DIR


def find_sample_by_criteria(
    filenames: np.ndarray,
    speakers: np.ndarray,
    y: np.ndarray,
    class_id: int,
    speaker: Optional[str] = None,
    keyword_in_name: Optional[str] = None,
    fallback_index: int = 0,
) -> int:
    """Finds the index of a sample matching given criteria."""
    matches = np.where(y == class_id)[0]
    if speaker:
        matches = [idx for idx in matches if speakers[idx] == speaker]
    if keyword_in_name:
        matches = [idx for idx in matches if keyword_in_name.lower() in filenames[idx].lower()]

    if len(matches) > 0:
        return int(matches[0])

    # Fallback to general class index
    class_indices = np.where(y == class_id)[0]
    return int(class_indices[fallback_index % len(class_indices)])


def plot_single_mfcc(
    mfcc: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "magma",
    vmin: float = None,
    vmax: float = None,
):
    """Plots and saves a single 2D MFCC heatmap (98 frames x 13 coefficients)."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    
    # Transpose so X-axis is time frames (0..97) and Y-axis is MFCC coefficient (0..12)
    im = ax.imshow(
        mfcc.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frame Index (0 to 97 | ~10 ms hop)", fontsize=10)
    ax.set_ylabel("MFCC Coefficient Index (0 to 12)", fontsize=10)
    ax.set_yticks(np.arange(13))
    ax.set_xticks(np.arange(0, 98, 10))
    ax.grid(False)
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("MFCC Amplitude (DCT energy)", fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {output_path.name}")


def generate_visualizations():
    print("=" * 80)
    print("SIH 26172 - ML/KWS Phase 5: MFCC Feature Visualization")
    print(f"Loading cached dataset from: {CACHE_DIR / 'mfcc_dataset.npz'}")
    print(f"Outputs destination        : {OUTPUTS_DIR}")
    print("=" * 80)

    cache_file = CACHE_DIR / "mfcc_dataset.npz"
    if not cache_file.is_file():
        raise FileNotFoundError(f"Cached dataset not found: {cache_file}")

    data = np.load(cache_file, allow_pickle=True)
    X = data["X"]                    # Shape: (3101, 98, 13), float32
    y = data["y"]                    # Shape: (3101,), int32
    filenames = data["filenames"]    # Shape: (3101,)
    speakers = data["speakers"]      # Shape: (3101,)
    class_names = data["class_names"]# ['silence', 'unknown', 'keyword']

    print(f"Dataset Loaded: X shape = {X.shape}, y shape = {y.shape}")

    # Class Indices
    kw_indices = np.where(y == 2)[0]
    unk_indices = np.where(y == 1)[0]
    sil_indices = np.where(y == 0)[0]

    # Global color scale range for consistent cross-comparison (clip extreme outliers)
    vmin = -35.0
    vmax = 15.0

    # --------------------------------------------------------------------------
    # 1. Representative ASTRA MFCC Heatmap
    # --------------------------------------------------------------------------
    print("\n[1/5] Generating Representative ASTRA MFCC Heatmap...")
    kw_idx = find_sample_by_criteria(filenames, speakers, y, class_id=2, speaker="Keshav", keyword_in_name="normal")
    kw_sample_name = filenames[kw_idx]
    plot_single_mfcc(
        mfcc=X[kw_idx],
        title=f"Representative KEYWORD ('ASTRA') MFCC Heatmap\nSample: {kw_sample_name} | Speaker: Keshav | Shape: {X[kw_idx].shape}",
        output_path=OUTPUTS_DIR / "mfcc_single_keyword.png",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    # --------------------------------------------------------------------------
    # 2. ASTRA vs UNKNOWN vs SILENCE Comparison (1x3 Panel)
    # --------------------------------------------------------------------------
    print("\n[2/5] Generating Core Classes Comparison (Keyword vs Unknown vs Silence)...")
    comp_kw_idx = find_sample_by_criteria(filenames, speakers, y, class_id=2, speaker="Imroz", keyword_in_name="normal")
    comp_unk_idx = unk_indices[0]
    comp_sil_idx = sil_indices[0]

    comp_kw_name = filenames[comp_kw_idx]
    comp_unk_name = filenames[comp_unk_idx]
    comp_sil_name = filenames[comp_sil_idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200, sharey=True)
    samples_to_plot = [
        (axes[0], X[comp_kw_idx], f"KEYWORD ('ASTRA')\n{comp_kw_name}", "magma"),
        (axes[1], X[comp_unk_idx], f"UNKNOWN (Speech)\n{comp_unk_name}", "viridis"),
        (axes[2], X[comp_sil_idx], f"SILENCE / NOISE\n{comp_sil_name}", "cividis"),
    ]

    for ax, mfcc, title, cmap in samples_to_plot:
        im = ax.imshow(
            mfcc.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Time Frame Index (0 to 97)", fontsize=9)
        ax.set_xticks(np.arange(0, 98, 20))
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Amplitude", fontsize=8)

    axes[0].set_ylabel("MFCC Coefficient Index (0 to 12)", fontsize=10)
    axes[0].set_yticks(np.arange(13))

    fig.suptitle("Core Class Acoustic Comparison: Keyword vs. Unknown vs. Silence", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_core = OUTPUTS_DIR / "mfcc_core_classes_comparison.png"
    plt.savefig(out_core, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_core.name}")

    # --------------------------------------------------------------------------
    # 3. ASTRA MFCC Comparison Across All 6 Speakers (2x3 Grid)
    # --------------------------------------------------------------------------
    print("\n[3/5] Generating Speaker Diversity Comparison across all 6 Speakers...")
    speaker_names = ["Imroz", "Jay", "Keshav", "Shaswat", "Sneha", "Yash"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), dpi=200, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, spk in enumerate(speaker_names):
        ax = axes_flat[idx]
        spk_idx = find_sample_by_criteria(filenames, speakers, y, class_id=2, speaker=spk, keyword_in_name="normal")
        fname = filenames[spk_idx]
        mfcc = X[spk_idx]
        
        im = ax.imshow(
            mfcc.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Speaker: {spk}\n{fname}", fontsize=10, fontweight="bold", pad=6)
        ax.grid(False)
        ax.set_xticks(np.arange(0, 98, 20))
        ax.set_yticks(np.arange(13))
        
        if idx in [0, 3]:
            ax.set_ylabel("MFCC Coefficient", fontsize=9)
        if idx in [3, 4, 5]:
            ax.set_xlabel("Time Frame Index (0 to 97)", fontsize=9)

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MFCC Amplitude (DCT Energy)", fontsize=9)

    fig.suptitle("Speaker Diversity: 'ASTRA' MFCC Footprint Across 6 Distinct Speakers", fontsize=13, fontweight="bold", y=0.98)
    out_spk = OUTPUTS_DIR / "mfcc_speaker_diversity.png"
    plt.savefig(out_spk, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_spk.name}")

    # --------------------------------------------------------------------------
    # 4. Multi-Sample Comparison Across 3 Classes (3x3 Grid)
    # --------------------------------------------------------------------------
    print("\n[4/5] Generating Multi-Sample Comparison Across Classes (3x3 Grid)...")
    grid_samples = [
        # Row 0: KEYWORD (3 conditions)
        ("KEYWORD (Normal)", find_sample_by_criteria(filenames, speakers, y, class_id=2, keyword_in_name="normal"), "magma"),
        ("KEYWORD (Close-Mic)", find_sample_by_criteria(filenames, speakers, y, class_id=2, keyword_in_name="close_mic"), "magma"),
        ("KEYWORD (Noise/Fan)", find_sample_by_criteria(filenames, speakers, y, class_id=2, keyword_in_name="fan"), "magma"),
        # Row 1: UNKNOWN (3 distinct speech samples)
        ("UNKNOWN (Speech Sample 1)", unk_indices[0], "viridis"),
        ("UNKNOWN (Speech Sample 2)", unk_indices[200], "viridis"),
        ("UNKNOWN (Speech Sample 3)", unk_indices[600], "viridis"),
        # Row 2: SILENCE (3 distinct background noise slices)
        ("SILENCE (Noise Sample 1)", sil_indices[0], "cividis"),
        ("SILENCE (Noise Sample 2)", sil_indices[50], "cividis"),
        ("SILENCE (Noise Sample 3)", sil_indices[150], "cividis"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), dpi=200, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (label_title, s_idx, cmap) in enumerate(grid_samples):
        ax = axes_flat[idx]
        fname = filenames[s_idx]
        mfcc = X[s_idx]

        im = ax.imshow(
            mfcc.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{label_title}\n{fname}", fontsize=9, fontweight="bold", pad=5)
        ax.grid(False)
        ax.set_xticks(np.arange(0, 98, 20))
        ax.set_yticks(np.arange(13))

        if idx % 3 == 0:
            ax.set_ylabel("MFCC Coeff", fontsize=9)
        if idx >= 6:
            ax.set_xlabel("Time Frame (0 to 97)", fontsize=9)

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MFCC Amplitude", fontsize=9)

    fig.suptitle("Multi-Sample Acoustic Diversity Grid (3 Classes x 3 Samples)", fontsize=13, fontweight="bold", y=0.98)
    out_multi = OUTPUTS_DIR / "mfcc_multisample_grid.png"
    plt.savefig(out_multi, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_multi.name}")

    # --------------------------------------------------------------------------
    # 5. Useful MFCC Statistical Visualizations (2-Panel)
    # --------------------------------------------------------------------------
    print("\n[5/5] Generating MFCC Statistical Profiles & Distributions...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200)

    # Panel A: Mean MFCC profile per coefficient across classes
    coeffs = np.arange(13)
    for class_id, class_name, color in [(2, "Keyword (ASTRA)", "#d95f02"), (1, "Unknown (Speech)", "#7570b3"), (0, "Silence (Noise)", "#1b9e77")]:
        class_mask = (y == class_id)
        class_X = X[class_mask]  # Shape: (N, 98, 13)
        # Average across time frames and samples for each coefficient
        mean_profile = np.mean(class_X, axis=(0, 1))  # Shape: (13,)
        std_profile = np.std(class_X, axis=(0, 1))   # Shape: (13,)

        ax1.plot(coeffs, mean_profile, marker="o", label=f"{class_name} (N={np.sum(class_mask)})", color=color, linewidth=2)
        ax1.fill_between(coeffs, mean_profile - std_profile, mean_profile + std_profile, alpha=0.15, color=color)

    ax1.set_title("Mean MFCC Energy Profile per Coefficient", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("MFCC Coefficient Index (0 to 12)", fontsize=10)
    ax1.set_ylabel("Mean Amplitude (DCT Energy)", fontsize=10)
    ax1.set_xticks(coeffs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel B: Histogram / KDE of feature values across classes
    for class_id, class_name, color in [(2, "Keyword", "#d95f02"), (1, "Unknown", "#7570b3"), (0, "Silence", "#1b9e77")]:
        class_mask = (y == class_id)
        class_vals = X[class_mask].flatten()
        # Subsample to 50,000 points for fast, clean histogram
        sampled_vals = np.random.choice(class_vals, size=min(50000, len(class_vals)), replace=False)
        ax2.hist(sampled_vals, bins=60, density=True, alpha=0.5, label=f"{class_name}", color=color)

    ax2.set_title("Overall MFCC Value Distribution Across Classes", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("MFCC Value", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("MFCC Feature Statistics & Energy Distributions (Cached Dataset: 3,101 Samples)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_stats = OUTPUTS_DIR / "mfcc_statistics_distribution.png"
    plt.savefig(out_stats, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_stats.name}")

    print("\n" + "=" * 80)
    print("PHASE 5 VISUALIZATION SUMMARY")
    print("=" * 80)
    print("1. Single Keyword Heatmap       : mfcc_single_keyword.png")
    print("2. Core Classes Comparison      : mfcc_core_classes_comparison.png")
    print("3. Speaker Diversity Grid       : mfcc_speaker_diversity.png")
    print("4. Multi-Sample Classes Grid    : mfcc_multisample_grid.png")
    print("5. Statistical Distribution     : mfcc_statistics_distribution.png")
    print(f"All 5 visualization figures saved successfully in: {OUTPUTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    generate_visualizations()
