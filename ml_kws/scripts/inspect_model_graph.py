"""
Inspects all layer weights, biases, scales, zero points, and quantization parameters
from kws_model_int8.tflite.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ML_KWS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _ML_KWS_DIR.parent

def inspect_model():
    model_path = _ML_KWS_DIR / "outputs" / "kws_model_int8.tflite"
    interp = tf.lite.Interpreter(str(model_path))
    interp.allocate_tensors()

    print("=" * 80)
    print("TFLite Model Layers & Quantization Details:")
    print("=" * 80)

    for i, op in enumerate(interp._get_ops_details()):
        print(f"\nOp {i}: {op['op_name']}")
        print(f"  Inputs: {op['inputs']}")
        print(f"  Outputs: {op['outputs']}")
        for in_idx in op['inputs']:
            t = interp.get_tensor_details()[in_idx]
            data = interp.get_tensor(in_idx)
            print(f"    in [{in_idx}] {t['name']:<50} shape={str(t['shape']):<15} dtype={str(t['dtype']):<25} quant={t['quantization_parameters']}")
        for out_idx in op['outputs']:
            t = interp.get_tensor_details()[out_idx]
            print(f"    out [{out_idx}] {t['name']:<50} shape={str(t['shape']):<15} dtype={str(t['dtype']):<25} quant={t['quantization_parameters']}")

if __name__ == "__main__":
    inspect_model()
