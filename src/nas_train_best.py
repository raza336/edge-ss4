"""
nas_train_best.py
=================
Q2.7 / Deliverable 8  -  Load the best architecture found by nas_search.py and TRAIN IT FROM SCRATCH
(random initialisation, full CIFAR-10 training set, long schedule).

Usage:  python src/nas_train_best.py --epochs 100
Output: checkpoints/nas_best.pth, results/nas_best_training_history.csv
"""
import argparse
import json
import os

from common import (DATA_DIR, LOG_DIR, RES_DIR, Logger, ckpt_path, count_params, evaluate, fit, get_device,
                    get_loaders, save_checkpoint, set_seed, write_csv)
from nas_space import build_network, describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=os.path.join(RES_DIR, "nas_best_architecture.json"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    log = Logger(os.path.join(LOG_DIR, "q2_nas_train_best.log"))
    cfg = json.load(open(a.arch))["config"]
    log(f"=== Training best NAS architecture from scratch | device={dev}")
    log(f"architecture: {describe(cfg)}")
    log(f"args: {vars(a)}")

    data = get_loaders("cifar10", a.batch_size, a.workers, data_dir=a.data_dir)
    model = build_network(cfg, 10)                       # fresh random initialisation (no weights from the search)
    log(f"parameters: {count_params(model):,}")
    hist = fit(model, data["train"], data["test"], a.epochs, a.lr, a.wd, dev, log, "nas-best",
               label_smoothing=a.label_smoothing)
    final = evaluate(model, data["test"], dev)
    log(f"FINAL NAS model test top-1 {final['top1']:.2f}%")
    save_checkpoint(model.cpu(), ckpt_path("nas_best"), {"name": "nas", "config": cfg, "num_classes": 10},
                    dataset="cifar10", test_top1=final["top1"], epochs=a.epochs)
    write_csv(os.path.join(RES_DIR, "nas_best_training_history.csv"), hist)
    log(f"saved -> {ckpt_path('nas_best')}")


if __name__ == "__main__":
    main()
