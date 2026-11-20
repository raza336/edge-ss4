"""
benchmark_q2.py
===============
Deliverable 11 (Q2)  -  Benchmarks the NAS-discovered model (PyTorch and ONNX Runtime) on the CIFAR-10 test
set and checks it against the deployment constraints used during the search.  It also summarises the search.

Output (results/):
    q2_benchmark.csv / q2_predictions.csv          PyTorch CPU
    q2_onnx_benchmark.csv / q2_onnx_predictions.csv  ONNX Runtime CPU
    q2_constraint_check.csv                        measured values vs. constraints
    q2_search_summary.csv                          candidates tried / rejected / trained, accuracy range

Usage:  python src/benchmark_q2.py
"""
import argparse
import json
import os

import pandas as pd

from common import DATA_DIR, RES_DIR, ckpt_path, onnx_path
from benchmark_lib import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    dfs = {}
    dfs["torch"] = run_benchmark(["nas_best"], "torch", "q2", RES_DIR, a.data_dir)
    if os.path.exists(onnx_path("nas_best")):
        dfs["onnx"] = run_benchmark(["nas_best"], "onnx", "q2_onnx", RES_DIR, a.data_dir)

    arch_file = os.path.join(RES_DIR, "nas_best_architecture.json")
    if os.path.exists(arch_file):
        info = json.load(open(arch_file))
        cons, search = info["constraints"], info["metrics_during_search"]
        t = dfs["torch"].iloc[0]
        rows = []
        for key, measured in (("max_latency_ms", t["latency_bs1_mean_ms"]), ("max_size_mb", t["size_mb"]),
                              ("max_peak_mem_mb", t.get("peak_mem_mb")), ("max_macs_m", t.get("macs_m"))):
            limit = cons.get(key)
            rows.append({"constraint": key, "limit": limit, "measured_final_model": measured,
                         "satisfied": None if limit is None or pd.isna(measured) else bool(measured <= limit)})
        pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, "q2_constraint_check.csv"), index=False)
        print("\nconstraint check:\n", pd.DataFrame(rows).to_string(index=False))

    hist_file = os.path.join(RES_DIR, "nas_history.csv")
    if os.path.exists(hist_file):
        h = pd.read_csv(hist_file)
        acc = h[h.status == "accepted"]
        summary = {"candidates_considered": len(h), "trained": len(acc), "rejected_by_constraints": int((h.status == "rejected").sum()),
                   "failed": int((h.status == "failed").sum()),
                   "val_acc_min": acc.val_acc.min(), "val_acc_median": acc.val_acc.median(), "val_acc_max": acc.val_acc.max(),
                   "latency_ms_min": acc.latency_ms.min(), "latency_ms_max": acc.latency_ms.max(),
                   "total_train_time_min": round(acc.train_time_s.sum() / 60, 1)}
        pd.DataFrame([summary]).to_csv(os.path.join(RES_DIR, "q2_search_summary.csv"), index=False)
        print("\nsearch summary:", summary)


if __name__ == "__main__":
    main()
