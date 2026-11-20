"""
nas_search.py
=============
Q2.3 - Q2.7 / Deliverables 6-7  -  Hardware-aware evolutionary Neural Architecture Search on CIFAR-10.

Algorithm: REGULARIZED (aging) EVOLUTION (Real et al., 2019)
    1. Fill an initial population with random feasible architectures.
    2. Repeat until the training budget is used up:
         - draw a random tournament of `--tournament` individuals, take the fittest as parent
         - mutate the parent -> child (resolution / channels / depth / kernel / block / activation ...)
         - HARDWARE CHECK of the child (before any training!): measure CPU latency, model size,
           peak memory; if a deployment constraint is violated the child is REJECTED and a new one is drawn
         - otherwise train it briefly on a CIFAR-10 subset, evaluate on a held-out validation split
         - add the child to the population and discard the OLDEST individual
Fitness (accuracy-vs-cost trade-off, only for feasible architectures):
         fitness = val_acc(%) - lat_weight * latency/max_latency - size_weight * size/max_size
Outputs (results/):
    nas_history.csv / nas_history.json   every candidate ever considered (accepted AND rejected) + all measurements
    nas_best_architecture.json           best architecture config + its measured metrics + constraints used
    nas_pareto.csv                       non-dominated candidates (val acc vs latency vs size)

Usage:  python src/nas_search.py --budget 60 --population 12 --max-latency-ms 6 --max-size-mb 2
"""
import argparse
import json
import os
import random
import time

import torch

from common import (DATA_DIR, LOG_DIR, RES_DIR, Logger, evaluate, fit, get_device, get_loaders, set_seed,
                    write_csv)
from hardware import Constraints, measure_hardware
from nas_space import (build_network, config_hash, describe, flatten_config, mutate, sample_config)


def proxy_train(model, loaders, epochs, lr, device):
    """Short training of a candidate; returns validation top-1 (%)."""
    quiet = lambda *a: None
    fit(model, loaders["train"], loaders["val"], epochs, lr, 5e-4, device, quiet, "nas", warmup_epochs=0)
    return evaluate(model, loaders["val"], device)["top1"]


def pareto_front(records):
    """Non-dominated records: higher val_acc, lower latency, lower size."""
    front = []
    for r in records:
        dominated = any(o["val_acc"] >= r["val_acc"] and o["latency_ms"] <= r["latency_ms"]
                        and o["size_mb"] <= r["size_mb"]
                        and (o["val_acc"] > r["val_acc"] or o["latency_ms"] < r["latency_ms"]
                             or o["size_mb"] < r["size_mb"]) for o in records)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["latency_ms"])


def main():
    ap = argparse.ArgumentParser()
    # search budget
    ap.add_argument("--budget", type=int, default=60, help="number of candidates that get TRAINED")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--tournament", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=60, help="max rejected draws before giving up one step")
    # proxy training
    ap.add_argument("--proxy-epochs", type=int, default=4)
    ap.add_argument("--proxy-train-size", type=int, default=15000)
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--proxy-lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    # deployment constraints (hardware-aware)
    ap.add_argument("--max-latency-ms", type=float, default=6.0)
    ap.add_argument("--max-size-mb", type=float, default=2.0)
    ap.add_argument("--max-peak-mem-mb", type=float, default=None)
    ap.add_argument("--max-macs-m", type=float, default=None)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads used for latency measurement")
    # selection
    ap.add_argument("--lat-weight", type=float, default=3.0, help="accuracy points lost at the latency limit")
    ap.add_argument("--size-weight", type=float, default=2.0, help="accuracy points lost at the size limit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    set_seed(a.seed)
    rng = random.Random(a.seed)
    dev = get_device()
    log = Logger(os.path.join(LOG_DIR, "q2_nas_search.log"))
    cons = Constraints(a.max_latency_ms, a.max_size_mb, a.max_peak_mem_mb, a.max_macs_m)
    log(f"=== Hardware-aware evolutionary NAS | train device={dev} | latency measured on CPU ({a.threads} thread)")
    log(f"args: {vars(a)}")
    log(f"constraints: {cons.as_dict()}")
    torch.backends.cudnn.benchmark = True

    loaders = get_loaders("cifar10", a.batch_size, a.workers, val_size=a.val_size,
                          train_subset=a.proxy_train_size, seed=a.seed, data_dir=a.data_dir)
    history, cache = [], {}
    n_trained = [0]
    t_start = time.time()

    def evaluate_candidate(cfg, generation, origin):
        """Hardware check -> (optional) proxy training. Returns the record (also stored in history)."""
        h = config_hash(cfg)
        if h in cache:
            return cache[h], True
        rec = {"id": len(history) + 1, "generation": generation, "origin": origin, "hash": h,
               **flatten_config(cfg), "config_json": json.dumps(cfg)}
        model = build_network(cfg, 10)
        hw = measure_hardware(model, threads=a.threads)
        rec.update(hw)
        viol = cons.violations(hw)
        if viol:
            rec.update(status="rejected", reason="; ".join(viol), val_acc=None, fitness=None, train_time_s=0.0)
            log(f"[#{rec['id']:03d}] REJECTED  {describe(cfg)} :: {rec['reason']}")
        else:
            t0 = time.time()
            try:
                model.to(dev)
                acc = proxy_train(model, loaders, a.proxy_epochs, a.proxy_lr, dev)
                fit_score = (acc - a.lat_weight * hw["latency_ms"] / a.max_latency_ms
                             - a.size_weight * hw["size_mb"] / a.max_size_mb)
                rec.update(status="accepted", reason="", val_acc=round(acc, 3), fitness=round(fit_score, 3),
                           train_time_s=round(time.time() - t0, 1))
                n_trained[0] += 1
                log(f"[#{rec['id']:03d}] ACCEPTED  ({n_trained[0]}/{a.budget}) val_acc {acc:.2f} fitness {fit_score:.2f} | "
                    f"lat {hw['latency_ms']:.2f}ms size {hw['size_mb']:.2f}MB params {hw['params']:,} | {describe(cfg)}")
            except Exception as e:                                   # diverged / OOM etc.
                rec.update(status="failed", reason=f"{type(e).__name__}: {e}", val_acc=None, fitness=None,
                           train_time_s=round(time.time() - t0, 1))
                log(f"[#{rec['id']:03d}] FAILED    {describe(cfg)} :: {rec['reason']}")
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        history.append(rec)
        cache[h] = rec
        write_csv(os.path.join(RES_DIR, "nas_history.csv"), history)     # saved after every candidate
        return rec, False

    # ---- 1. initial population -------------------------------------------------------------------
    population = []                                                     # list of (config, fitness), oldest first
    attempts = 0
    while len(population) < a.population and attempts < a.max_attempts * a.population:
        attempts += 1
        cfg = sample_config(rng)
        rec, dup = evaluate_candidate(cfg, 0, "random")
        if rec["status"] == "accepted" and not dup:
            population.append((cfg, rec["fitness"]))
    if not population:
        raise SystemExit("No feasible architecture found - relax --max-latency-ms / --max-size-mb.")
    log(f"initial population ready: {len(population)} feasible architectures ({attempts} draws)")

    # ---- 2. evolution -----------------------------------------------------------------------------
    while n_trained[0] < a.budget:
        pool = rng.sample(population, min(a.tournament, len(population)))
        parent = max(pool, key=lambda p: p[1])[0]
        generation = n_trained[0] // a.population
        child_rec, child_cfg = None, None
        for _ in range(a.max_attempts):
            cfg = mutate(parent, rng)
            rec, dup = evaluate_candidate(cfg, generation, "mutation")
            if rec["status"] == "accepted" and not dup:
                child_rec, child_cfg = rec, cfg
                break
            if n_trained[0] >= a.budget:
                break
        if child_rec is None:                                           # mutation keeps failing -> random restart
            cfg = sample_config(rng)
            rec, dup = evaluate_candidate(cfg, generation, "random")
            if rec["status"] == "accepted" and not dup:
                child_rec, child_cfg = rec, cfg
        if child_rec is not None:
            population.append((child_cfg, child_rec["fitness"]))
            population.pop(0)                                           # aging: remove the oldest

    # ---- 3. results ---------------------------------------------------------------------------------
    accepted = [r for r in history if r["status"] == "accepted"]
    best = max(accepted, key=lambda r: r["fitness"])
    best_cfg = json.loads(best["config_json"])
    metrics = {k: best[k] for k in ("val_acc", "fitness", "latency_ms", "latency_p95_ms", "size_mb", "params",
                                    "macs_m", "peak_mem_mb", "act_mem_mb")}
    with open(os.path.join(RES_DIR, "nas_best_architecture.json"), "w") as f:
        json.dump({"config": best_cfg, "description": describe(best_cfg), "metrics_during_search": metrics,
                   "constraints": cons.as_dict(), "search_args": vars(a), "candidate_id": best["id"]}, f, indent=2)
    with open(os.path.join(RES_DIR, "nas_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    write_csv(os.path.join(RES_DIR, "nas_history.csv"), history)
    write_csv(os.path.join(RES_DIR, "nas_pareto.csv"), pareto_front(accepted))

    n_rej = sum(r["status"] == "rejected" for r in history)
    log(f"\nsearch finished in {(time.time() - t_start) / 60:.1f} min: {len(history)} candidates "
        f"({len(accepted)} trained, {n_rej} rejected by constraints, "
        f"{sum(r['status'] == 'failed' for r in history)} failed)")
    log(f"BEST: {describe(best_cfg)}")
    log(f"      val_acc {best['val_acc']:.2f}% | latency {best['latency_ms']:.2f} ms | size {best['size_mb']:.2f} MB | "
        f"params {best['params']:,} | fitness {best['fitness']:.2f}")
    log("saved -> results/nas_best_architecture.json, nas_history.csv/json, nas_pareto.csv")


if __name__ == "__main__":
    main()
