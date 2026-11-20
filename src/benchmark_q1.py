"""
benchmark_q1.py
===============
Deliverable 11 (Q1)  -  Automatically evaluates the teacher, the three BASELINE students and the three
DISTILLED students on the CIFAR-100 test set (CPU).

Output (results/):
    q1_benchmark.csv      accuracy, size, params, MACs, latency, throughput, peak memory per model
    q1_predictions.csv    per-image predictions / confidences / correctness of all models
    q1_kd_gain.csv        baseline vs KD accuracy per student, compression & speed-up w.r.t. the teacher

Usage:  python src/benchmark_q1.py [--backend torch|onnx] [--skip-teacher]
"""
import argparse
import os

import pandas as pd

from common import DATA_DIR, RES_DIR, STUDENTS, ckpt_path, onnx_path
from benchmark_lib import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--skip-teacher", action="store_true", help="skip ResNet-50 (slow on CPU)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    path_of = ckpt_path if a.backend == "torch" else onnx_path
    names = ([] if a.skip_teacher else ["teacher"]) + [f"{s}_{k}" for s in STUDENTS for k in ("baseline", "kd")]
    names = [n for n in names if os.path.exists(path_of(n))]
    prefix = "q1" if a.backend == "torch" else "q1_onnx"
    df = run_benchmark(names, a.backend, prefix, RES_DIR, a.data_dir).set_index("model")

    rows = []
    for s in STUDENTS:
        b, k = f"{s}_baseline", f"{s}_kd"
        if b in df.index and k in df.index:
            r = {"student": s, "baseline_top1": df.loc[b, "top1_acc"], "kd_top1": df.loc[k, "top1_acc"],
                 "kd_gain_pts": round(df.loc[k, "top1_acc"] - df.loc[b, "top1_acc"], 2),
                 "baseline_top5": df.loc[b, "top5_acc"], "kd_top5": df.loc[k, "top5_acc"],
                 "size_mb": df.loc[k, "size_mb"], "latency_ms": df.loc[k, "latency_bs1_mean_ms"]}
            if "teacher" in df.index:
                t = df.loc["teacher"]
                r.update(teacher_top1=t["top1_acc"], gap_to_teacher_pts=round(t["top1_acc"] - df.loc[k, "top1_acc"], 2),
                         size_compression_x=round(t["size_mb"] / df.loc[k, "size_mb"], 2),
                         latency_speedup_x=round(t["latency_bs1_mean_ms"] / df.loc[k, "latency_bs1_mean_ms"], 2))
            rows.append(r)
    if rows:
        out = os.path.join(RES_DIR, f"{prefix}_kd_gain.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote {out}")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
