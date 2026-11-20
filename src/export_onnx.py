"""
export_onnx.py
==============
Q1.7 / Q2.8 / Deliverable 10  -  Export the trained models to ONNX (dynamic batch, opset 17) and verify
every exported file with ONNX Runtime on the CPU (max |PyTorch - ONNX Runtime| output difference).

Usage:  python src/export_onnx.py                       # every model whose checkpoint exists
        python src/export_onnx.py --models mobilenetv2_kd nas_best
Output: onnx_models/<model_name>.onnx
"""
import argparse
import os
import warnings

import numpy as np
import torch

from common import ALL_MODELS, ckpt_path, file_size_mb, load_model_by_name, onnx_path

warnings.filterwarnings("ignore")
OPSET = 17


def export_model(name):
    model = load_model_by_name(name).cpu().eval()
    path = onnx_path(name)
    dummy = torch.randn(1, 3, 32, 32)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=OPSET, do_constant_folding=True)
    try:                                   # newer PyTorch: force the stable TorchScript-based exporter
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:                      # older PyTorch has no `dynamo` argument
        torch.onnx.export(model, dummy, path, **kwargs)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    err = float(np.abs(ref - out).max())
    print(f"[onnx] {name:26s} -> {path}  ({file_size_mb(path):.1f} MB) | max|torch - onnxruntime| = {err:.2e}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: all with a checkpoint)")
    a = ap.parse_args()
    names = a.models or [n for n in ALL_MODELS if os.path.exists(ckpt_path(n))]
    for n in names:
        try:
            export_model(n)
        except Exception as e:
            print(f"[onnx] {n}: export FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
