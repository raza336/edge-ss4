"""
train_baseline.py
=================
Q1.5 / Deliverable 3  -  Baseline training of the three students WITHOUT distillation
(plain cross-entropy on the hard labels).  Same recipe / epochs as distill.py so that the
comparison baseline-vs-KD is fair.

Usage:  python src/train_baseline.py --student all --epochs 100
        python src/train_baseline.py --student mobilenetv2
"""
import argparse
import os

from common import (DATA_DIR, LOG_DIR, RES_DIR, STUDENTS, Logger, ckpt_path, count_params, evaluate,
                    fit, get_device, get_loaders, save_checkpoint, set_seed, write_csv)
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="all", help="mobilenetv2 | squeezenet | efficientnet_lite0 | all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    dev = get_device()
    log = Logger(os.path.join(LOG_DIR, "q1_train_baseline.log"))
    names = STUDENTS if a.student == "all" else [a.student]
    data = get_loaders("cifar100", a.batch_size, a.workers, data_dir=a.data_dir)

    for name in names:
        set_seed(a.seed)
        log(f"=== Baseline (no KD): {name} | device={dev} | {vars(a)}")
        model = build_model(name, 100)
        log(f"parameters: {count_params(model):,}")
        hist = fit(model, data["train"], data["test"], a.epochs, a.lr, a.wd, dev, log, f"baseline-{name}")
        final = evaluate(model, data["test"], dev)
        log(f"FINAL {name} baseline test top-1 {final['top1']:.2f}%  top-5 {final['top5']:.2f}%")
        save_checkpoint(model.cpu(), ckpt_path(f"{name}_baseline"), {"name": name, "num_classes": 100},
                        dataset="cifar100", test_top1=final["top1"], test_top5=final["top5"],
                        epochs=a.epochs, method="baseline")
        write_csv(os.path.join(RES_DIR, f"q1_baseline_{name}_history.csv"), hist)
        log(f"saved -> {ckpt_path(f'{name}_baseline')}")


if __name__ == "__main__":
    main()
