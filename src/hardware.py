"""
hardware.py
===========
Q2.5 / Q2.6  -  Hardware-aware measurements and deployment constraints for candidate architectures.

For every candidate the search measures (on the CPU, with real forward passes):
  * latency_ms      median single-image (batch 1) PyTorch CPU inference latency
  * size_mb         storage size of the serialised FP32 weights (state_dict)
  * peak_mem_mb     peak inference memory: process-RSS increase sampled while the model runs
                    (needs `psutil`; None if unavailable)
  * act_mem_mb      size of the largest intermediate feature map (analytical, always available)
  * params, macs
and `Constraints` rejects candidates that violate the predefined deployment budget.
"""
import copy
import gc
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from common import count_macs, count_params, state_dict_size_mb


@dataclass
class Constraints:
    """Deployment constraints (edge device budget). None = unconstrained."""
    max_latency_ms: Optional[float] = 6.0
    max_size_mb: Optional[float] = 2.0
    max_peak_mem_mb: Optional[float] = None
    max_macs_m: Optional[float] = None

    def violations(self, hw):
        """List of human-readable violations (empty list = architecture is feasible)."""
        v = []
        if self.max_latency_ms is not None and hw["latency_ms"] > self.max_latency_ms:
            v.append(f"latency {hw['latency_ms']:.2f}ms > {self.max_latency_ms}ms")
        if self.max_size_mb is not None and hw["size_mb"] > self.max_size_mb:
            v.append(f"size {hw['size_mb']:.2f}MB > {self.max_size_mb}MB")
        if self.max_macs_m is not None and hw["macs_m"] > self.max_macs_m:
            v.append(f"MACs {hw['macs_m']:.1f}M > {self.max_macs_m}M")
        if self.max_peak_mem_mb is not None:
            mem = hw["peak_mem_mb"] if hw["peak_mem_mb"] is not None else hw["act_mem_mb"]
            if mem > self.max_peak_mem_mb:
                v.append(f"peak memory {mem:.2f}MB > {self.max_peak_mem_mb}MB")
        return v

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def measure_peak_memory_mb(model, x, runs=20):
    """Peak RSS increase (MB) while running `runs` inferences. Approximate; None without psutil."""
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process()
    gc.collect()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    for _ in range(runs):
        model(x)
    stop.set()
    th.join()
    return max(0.0, (peak[0] - base) / 1e6)


@torch.no_grad()
def largest_activation_mb(model, x):
    """Analytical memory estimate: size (MB, FP32) of the largest feature map produced by any layer."""
    biggest = [x.numel()]
    hooks = []

    def hook(m, i, o):
        if isinstance(o, torch.Tensor):
            biggest[0] = max(biggest[0], o.numel())

    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))
    model(x)
    for h in hooks:
        h.remove()
    return biggest[0] * 4 / 1e6


@torch.no_grad()
def measure_hardware(model, threads=1, warmup=10, iters=50, input_shape=(1, 3, 32, 32)):
    """Measure latency / size / memory of `model` on the CPU. The model passed in is not modified."""
    m = copy.deepcopy(model).cpu().eval()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randn(*input_shape)
    try:
        for _ in range(warmup):
            m(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t) * 1000)
        ts = np.array(ts)
        hw = {"latency_ms": float(np.median(ts)), "latency_p95_ms": float(np.percentile(ts, 95)),
              "size_mb": float(state_dict_size_mb(m)), "params": int(count_params(m)),
              "macs_m": float(count_macs(m) / 1e6), "act_mem_mb": float(largest_activation_mb(m, x)),
              "peak_mem_mb": measure_peak_memory_mb(m, x)}
    finally:
        torch.set_num_threads(prev_threads)
    return hw
