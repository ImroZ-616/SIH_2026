"""
SIH 26172 - R2 ML/KWS Golden Vector Exporter
Phase 14: Extracts 6 held-out test samples (2 Keyword, 2 Unknown, 2 Silence),
runs the Python reference pipeline with the approved fused A/B quantization contract,
and exports golden vectors for C++ verification.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent
_SRC_DIR = _ML_KWS_DIR / "src"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CACHE_DIR, OUTPUTS_DIR, DATASET_DIR
from features import CLASS_NAMES, extract_features, load_dataset_npz
from normalization import apply_mfcc_normalization, load_normalization_stats
from audio_preprocessing import preprocess_audio, load_audio


def find_test_sample_audio_path(filename: str, class_id: int) -> Path:
    """Finds the absolute WAV path for a given test set filename."""
    if class_id == 2:  # Keyword
        p = DATASET_DIR / "keyword" / filename
        if p.is_file():
            return p
    elif class_id == 0:  # Silence
        p = DATASET_DIR / "silence" / filename
        if p.is_file():
            return p
    elif class_id == 1:  # Unknown
        matches = list((DATASET_DIR / "unknown").rglob(filename))
        if matches:
            return matches[0]

    all_matches = list(DATASET_DIR.rglob(filename))
    if all_matches:
        return all_matches[0]

    raise FileNotFoundError(f"Could not locate audio file for test sample: {filename}")


def export_golden_vectors():
    print("=" * 80)
    print("SIH 26172 - Phase 14 Golden Reference Vector Exporter")
    print("=" * 80)

    test_npz_path = CACHE_DIR / "test_data.npz"
    norm_stats_path = OUTPUTS_DIR / "norm_stats.npz"
    int8_tflite_path = OUTPUTS_DIR / "kws_model_int8.tflite"

    assert test_npz_path.is_file(), f"Missing {test_npz_path}"
    assert norm_stats_path.is_file(), f"Missing {norm_stats_path}"
    assert int8_tflite_path.is_file(), f"Missing {int8_tflite_path}"

    test_data = load_dataset_npz(test_npz_path)
    X_test = test_data["X"]  # (473, 98, 13)
    y_test = test_data["y"]  # (473,)
    filenames = test_data["filenames"]  # (473,)

    mean_vec, std_vec, eps = load_normalization_stats(norm_stats_path)
    mean_arr = mean_vec.flatten()
    std_arr = std_vec.flatten()

    # Load TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=str(int8_tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    s_in, z_in = input_details["quantization"]
    s_out, z_out = output_details["quantization"]
    norm_a = 1.0 / ((std_arr + eps) * s_in)
    norm_b = z_in - (mean_arr / ((std_arr + eps) * s_in))

    print(f"INT8 Input: Scale={s_in:.10f}, ZeroPoint={z_in}")
    print(f"INT8 Output: Scale={s_out:.10f}, ZeroPoint={z_out}")

    # Select 6 representative samples: 2 Keyword (2), 2 Unknown (1), 2 Silence (0)
    kw_indices = np.where(y_test == 2)[0]
    unk_indices = np.where(y_test == 1)[0]
    sil_indices = np.where(y_test == 0)[0]

    selected_indices = [
        kw_indices[0],  kw_indices[1],    # GV-01, GV-02 (Keyword)
        unk_indices[0], unk_indices[1],   # GV-03, GV-04 (Unknown)
        sil_indices[0], sil_indices[1],   # GV-05, GV-06 (Silence)
    ]

    golden_vectors = []
    vector_names = [
        "GV01_Keyword_ASTRA_1",
        "GV02_Keyword_ASTRA_2",
        "GV03_Unknown_Speech_1",
        "GV04_Unknown_Speech_2",
        "GV05_Silence_Background_1",
        "GV06_Silence_Background_2",
    ]

    for v_idx, test_idx in enumerate(selected_indices):
        fn = str(filenames[test_idx])
        true_cls = int(y_test[test_idx])
        v_name = vector_names[v_idx]

        audio_path = find_test_sample_audio_path(fn, true_cls)
        if true_cls == 1:
            candidates = list((DATASET_DIR / "unknown").rglob(fn))
            for cand in candidates:
                mfcc_cand = extract_features(cand)
                if np.max(np.abs(mfcc_cand - X_test[test_idx])) < 1e-5:
                    audio_path = cand
                    break

        # Standardize audio to 16 kHz Mono float32 (16000,)
        std_audio_float = preprocess_audio(audio_path, target_sr=16000, target_samples=16000)
        assert len(std_audio_float) == 16000, f"Expected 16000 samples, got {len(std_audio_float)}"
        pcm_int16 = np.clip(std_audio_float * 32767.0, -32768, 32767).astype(np.int16)

        # 1. Python MFCC Extraction
        mfcc_features = extract_features(audio_path)
        assert mfcc_features.shape == (98, 13), f"Bad shape {mfcc_features.shape}"

        # 2. Approved Fused A/B Quantization Contract
        quant_fused = np.clip(np.round(mfcc_features * norm_a + norm_b), -128, 127).astype(np.int8)

        # 3. TFLite INT8 Inference
        input_tensor = quant_fused.reshape((1, 98, 13, 1)).astype(np.int8)
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(output_details["index"])[0]  # (3,) int8

        # Output dequantization
        output_dequant = (output_tensor.astype(np.float32) - z_out) * s_out
        sum_p = np.sum(output_dequant)
        output_probs = output_dequant / sum_p if sum_p > 0 else output_dequant
        pred_class = int(np.argmax(output_probs))

        print(f"  [{v_name}] Class={CLASS_NAMES[true_cls]} ({true_cls}) | File={fn}")
        print(f"    - Pred Class: {CLASS_NAMES[pred_class]} ({pred_class}) | Output INT8: {output_tensor.tolist()} | Probs: {output_probs.round(4).tolist()}")

        golden_vectors.append({
            "vector_id": v_name,
            "filename": fn,
            "audio_path": str(audio_path),
            "true_class_id": true_cls,
            "true_class_name": str(CLASS_NAMES[true_cls]),
            "pcm_samples_count": len(pcm_int16),
            "pcm_samples": pcm_int16.tolist(),
            "audio_float": std_audio_float.tolist(),
            "mfcc_shape": list(mfcc_features.shape),
            "mfcc_features": mfcc_features.tolist(),
            "int8_input": quant_fused.tolist(),
            "expected_output_int8": output_tensor.tolist(),
            "expected_probabilities": output_probs.tolist(),
            "expected_class": pred_class,
        })

    embedded_dir = _REPO_ROOT / "embedded_kws"
    embedded_tests_dir = embedded_dir / "tests"

    # Export JSON
    json_path = OUTPUTS_DIR / "golden_vectors.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(golden_vectors, f, indent=2)
    print(f"\n[SAVED] Golden Vectors JSON : {json_path}")

    # Export Header
    header_path = embedded_tests_dir / "golden_vectors.h"
    generate_cpp_golden_header(golden_vectors, header_path)
    print(f"[SAVED] Golden Vectors C Header: {header_path}")


def generate_cpp_golden_header(golden_vectors: List[Dict], output_path: Path):
    num_vectors = len(golden_vectors)
    num_pcm = 16000
    num_frames = 98
    num_mfcc = 13

    lines = [
        "// SIH 26172 - Golden Reference Test Vectors for Phase 14 Validation",
        "// Generated automatically by ml_kws/scripts/export_golden_vectors.py",
        "// Source: ml_kws/cache/test_data.npz (Held-out Test Set)",
        "",
        "#ifndef GOLDEN_VECTORS_H_",
        "#define GOLDEN_VECTORS_H_",
        "",
        "#include <stdint.h>",
        "",
        f"#define GOLDEN_NUM_VECTORS {num_vectors}",
        f"#define GOLDEN_PCM_SAMPLES {num_pcm}",
        f"#define GOLDEN_NUM_FRAMES {num_frames}",
        f"#define GOLDEN_NUM_MFCC {num_mfcc}",
        "",
        "struct GoldenVector {",
        "    const char* name;",
        "    const char* filename;",
        "    int true_class;",
        "    int expected_class;",
        "    const float* audio_float;                     // [16000] float32 normalized [-1.0, 1.0]",
        "    const int16_t* pcm_samples;                   // [16000] int16 PCM",
        "    const float (*mfcc_features)[GOLDEN_NUM_MFCC]; // [98][13] float32",
        "    const int8_t (*int8_input)[GOLDEN_NUM_MFCC];   // [98][13] int8",
        "    const int8_t* expected_output_int8;          // [3] int8",
        "    const float* expected_probabilities;         // [3] float32",
        "};",
        "",
    ]

    for v_idx, gv in enumerate(golden_vectors):
        v_name = gv["vector_id"]
        float_chunks = [", ".join([f"{x:.8f}f" for x in gv["audio_float"][i:i+8]]) for i in range(0, num_pcm, 8)]
        lines.append(f"static const float g_audio_float_{v_name}[{num_pcm}] = {{")
        lines.append("    " + ",\n    ".join(float_chunks))
        lines.append("};")
        lines.append("")

        pcm_chunks = [", ".join(map(str, gv["pcm_samples"][i:i+16])) for i in range(0, num_pcm, 16)]
        lines.append(f"static const int16_t g_pcm_{v_name}[{num_pcm}] = {{")
        lines.append("    " + ",\n    ".join(pcm_chunks))
        lines.append("};")
        lines.append("")

        mfcc_flat = [f"{x:.8f}f" for row in gv["mfcc_features"] for x in row]
        mfcc_chunks = [", ".join(mfcc_flat[i:i+13]) for i in range(0, len(mfcc_flat), 13)]
        lines.append(f"static const float g_mfcc_{v_name}[{num_frames}][{num_mfcc}] = {{")
        lines.append("    {" + "},\n    {".join(mfcc_chunks) + "}")
        lines.append("};")
        lines.append("")

        int8_flat = [str(x) for row in gv["int8_input"] for x in row]
        int8_chunks = [", ".join(int8_flat[i:i+13]) for i in range(0, len(int8_flat), 13)]
        lines.append(f"static const int8_t g_int8_{v_name}[{num_frames}][{num_mfcc}] = {{")
        lines.append("    {" + "},\n    {".join(int8_chunks) + "}")
        lines.append("};")
        lines.append("")

        out_int8_str = ", ".join(map(str, gv["expected_output_int8"]))
        probs_str = ", ".join([f"{p:.8f}f" for p in gv["expected_probabilities"]])
        lines.append(f"static const int8_t g_out_int8_{v_name}[3] = {{ {out_int8_str} }};")
        lines.append(f"static const float g_probs_{v_name}[3] = {{ {probs_str} }};")
        lines.append("")

    lines.append(f"static const GoldenVector g_golden_vectors[{num_vectors}] = {{")
    for v_idx, gv in enumerate(golden_vectors):
        v_name = gv["vector_id"]
        lines.append("    {")
        lines.append(f'        "{v_name}",')
        lines.append(f'        "{gv["filename"]}",')
        lines.append(f'        {gv["true_class_id"]},')
        lines.append(f'        {gv["expected_class"]},')
        lines.append(f'        g_audio_float_{v_name},')
        lines.append(f'        g_pcm_{v_name},')
        lines.append(f'        g_mfcc_{v_name},')
        lines.append(f'        g_int8_{v_name},')
        lines.append(f'        g_out_int8_{v_name},')
        lines.append(f'        g_probs_{v_name}')
        lines.append("    }" + ("," if v_idx < num_vectors - 1 else ""))
    lines.append("};")
    lines.append("")
    lines.append("#endif  // GOLDEN_VECTORS_H_")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    export_golden_vectors()
