"""
benchmark_lib.py
================
Engine shared by benchmark_q1.py, benchmark_q2.py and onnx_inference.py.

For every model (always CPU, full test set of the model's dataset) it measures
  top-1 / top-5 accuracy, per-image predictions + confidence, model file size,
  parameters, MACs, single-image latency (batch 1, 1 thread), batch throughput (batch 256, all threads)
  and (PyTorch back-end) approximate peak inference memory.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from common import (DATA_DIR, count_macs, count_params, file_size_mb, load_model_by_name, load_test_arrays,
                    measure_latency)
from hardware import measure_peak_memory_mb
from inference import Predictor, softmax


def benchmark_one(name, backend, x, y, lat_iters=200, batch_size=256):
    threads = max(1, os.cpu_count() or 1)
    p = Predictor(name, backend, threads=threads)
    p(x[:batch_size])                                              # warm-up
    t0 = time.perf_counter()
    logits = p.predict_batches(x, batch_size)
    wall = time.perf_counter() - t0
    prob = softmax(logits)
    pred = prob.argmax(1)
    top5 = np.argsort(-prob, axis=1)[:, :min(5, prob.shape[1])]
    yn = y.numpy()
    p1 = Predictor(name, backend, threads=1)
    lat = measure_latency(p1, x[:1], warmup=30, iters=lat_iters)
    row = {"model": name, "backend": backend, "dataset": p.dataset,
           "top1_acc": round(100 * (pred == yn).mean(), 2),
           "top5_acc": round(100 * (top5 == yn[:, None]).any(1).mean(), 2),
           "model_file": os.path.basename(p.path), "size_mb": round(file_size_mb(p.path), 2),
           "latency_bs1_mean_ms": round(lat["mean"], 3), "latency_bs1_p50_ms": round(lat["p50"], 3),
           "latency_bs1_p95_ms": round(lat["p95"], 3),
           "throughput_bs256_img_per_s": round(len(x) / wall, 1), "throughput_threads": threads}
    if backend == "torch":
        m = load_model_by_name(name)
        torch.set_num_threads(1)
        peak = measure_peak_memory_mb(m, torch.randn(1, 3, 32, 32))
        row.update(params=count_params(m), macs_m=round(count_macs(m) / 1e6, 2),
                   peak_mem_mb=None if peak is None else round(peak, 2))
        torch.set_num_threads(threads)
    return row, pred, prob.max(1)


def run_benchmark(names, backend, prefix, res_dir, data_dir=DATA_DIR):
    """Benchmark `names` (all must use the same dataset) -> <prefix>_benchmark.csv + <prefix>_predictions.csv."""
    from common import dataset_of
    ds = {dataset_of(n) for n in names}
    assert len(ds) == 1, "benchmark one dataset at a time"
    x, y, classes = load_test_arrays(ds.pop(), data_dir)
    rows, preds = [], {}
    for n in names:
        print(f"benchmarking {n} ({backend}) ...", flush=True)
        row, pr, cf = benchmark_one(n, backend, x, y)
        rows.append(row)
        preds[n] = (pr, cf)
        print(f"   top1 {row['top1_acc']:.2f}% | size {row['size_mb']} MB | lat {row['latency_bs1_mean_ms']} ms | "
              f"thr {row['throughput_bs256_img_per_s']} img/s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res_dir, f"{prefix}_benchmark.csv"), index=False)

    pdf = pd.DataFrame({"image_index": np.arange(len(y)), "true_label": y.numpy(),
                        "true_class": [classes[i] for i in y.numpy()]})
    for n in names:
        pr, cf = preds[n]
        pdf[f"{n}_pred"] = pr
        pdf[f"{n}_class"] = [classes[i] for i in pr]
        pdf[f"{n}_confidence"] = np.round(cf, 4)
        pdf[f"{n}_correct"] = (pr == y.numpy()).astype(int)
    pdf.to_csv(os.path.join(res_dir, f"{prefix}_predictions.csv"), index=False)
    print(f"\nWrote {prefix}_benchmark.csv and {prefix}_predictions.csv in {res_dir}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["model_file", "throughput_threads", "dataset"]).to_string(index=False))
    return df
