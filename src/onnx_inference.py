"""
onnx_inference.py
=================
Q1.7 / Q2.8 / Deliverable 10  -  CPU inference + benchmark of the exported ONNX models with ONNX Runtime
(CPUExecutionProvider).  Students are benchmarked on CIFAR-100, the NAS model on CIFAR-10.

Output: results/onnx_students_benchmark.csv (+ _predictions) and results/onnx_nas_benchmark.csv (+ _predictions)
Usage:  python src/onnx_inference.py
        python src/inference.py --model nas_best --backend onnx --num-images 8     # single-model demo
"""
import argparse
import os

from common import ALL_MODELS, DATA_DIR, RES_DIR, onnx_path
from benchmark_lib import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--skip-teacher", action="store_true")
    a = ap.parse_args()
    have = [n for n in ALL_MODELS if os.path.exists(onnx_path(n))]
    if a.skip_teacher:
        have = [n for n in have if n != "teacher"]
    students = [n for n in have if n != "nas_best"]
    if students:
        run_benchmark(students, "onnx", "onnx_students", RES_DIR, a.data_dir)
    if "nas_best" in have:
        run_benchmark(["nas_best"], "onnx", "onnx_nas", RES_DIR, a.data_dir)
