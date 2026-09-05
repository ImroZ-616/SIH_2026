"""
SIH 26172 - DSP Tables Generator
Generates precomputed C++ header with Hann window, Mel Filterbank, and DCT-II basis matrices
directly matching the Python reference implementation in audio/.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.fftpack import dct

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.mel import create_mel_filterbank

def generate_dsp_tables():
    output_path = _REPO_ROOT / "embedded_kws" / "dsp" / "dsp_tables.h"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Hann Window (480 samples)
    hann = np.hanning(480).astype(np.float32)

    # 2. Mel Filterbank (40 filters x 241 bins)
    mel_fb = create_mel_filterbank(sample_rate=16000, n_fft=480, n_mels=40).astype(np.float32)
    assert mel_fb.shape == (40, 241)

    # 3. DCT-II Orthonormal Basis Matrix (13 coeffs x 40 mel bands)
    # In Python scipy: dct(x, type=2, axis=1, norm='ortho')[:, :13]
    # The orthonormal DCT-II formula for basis matrix:
    # DCT[n, m] = sqrt(2/M) * cos(pi * n * (2m + 1) / (2M)) for n > 0
    # DCT[0, m] = sqrt(1/M)
    M = 40
    N = 13
    dct_basis = np.zeros((N, M), dtype=np.float32)
    for n in range(N):
        for m in range(M):
            if n == 0:
                dct_basis[n, m] = np.sqrt(1.0 / M)
            else:
                dct_basis[n, m] = np.sqrt(2.0 / M) * np.cos(np.pi * n * (2 * m + 1) / (2.0 * M))

    # Verify DCT basis against scipy.fftpack.dct
    test_x = np.random.RandomState(42).randn(1, 40).astype(np.float32)
    scipy_dct = dct(test_x, type=2, axis=1, norm='ortho')[:, :13]
    manual_dct = test_x @ dct_basis.T
    diff_dct = np.max(np.abs(scipy_dct - manual_dct))
    print(f"DCT Basis Verification vs scipy.fftpack.dct: Max diff = {diff_dct:.10f}")
    assert diff_dct < 1e-6, "DCT basis mismatch!"

    # Format C++ Header
    lines = [
        "// SIH 26172 - Precomputed DSP Lookup Tables",
        "// Generated automatically from audio/mel.py and scipy.fftpack.dct",
        "",
        "#ifndef DSP_TABLES_H_",
        "#define DSP_TABLES_H_",
        "",
        "#include <stdint.h>",
        "",
        "#define DSP_FRAME_LENGTH 480",
        "#define DSP_HOP_LENGTH 160",
        "#define DSP_FFT_SIZE 480",
        "#define DSP_FFT_BINS 241",
        "#define DSP_NUM_MELS 40",
        "#define DSP_NUM_MFCC 13",
        "#define DSP_NUM_FRAMES 98",
        "",
    ]

    # Hann Table
    hann_str = ", ".join([f"{x:.8f}f" for x in hann])
    lines.append(f"static const float KWS_HANN_WINDOW[DSP_FRAME_LENGTH] = {{ {hann_str} }};")
    lines.append("")

    # Sparse / Full Mel Filterbank Table (40 x 241)
    lines.append(f"static const float KWS_MEL_FILTERBANK[DSP_NUM_MELS][DSP_FFT_BINS] = {{")
    for m in range(40):
        row_str = ", ".join([f"{x:.8f}f" for x in mel_fb[m]])
        lines.append(f"    {{ {row_str} }}{',' if m < 39 else ''}")
    lines.append("};")
    lines.append("")

    # DCT Basis Matrix (13 x 40)
    lines.append(f"static const float KWS_DCT_BASIS[DSP_NUM_MFCC][DSP_NUM_MELS] = {{")
    for n in range(13):
        row_str = ", ".join([f"{x:.8f}f" for x in dct_basis[n]])
        lines.append(f"    {{ {row_str} }}{',' if n < 12 else ''}")
    lines.append("};")
    lines.append("")

    lines.append("#endif  // DSP_TABLES_H_")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] DSP Tables Header: {output_path}")

if __name__ == "__main__":
    generate_dsp_tables()
