"""
train_teacher.py
================
Q1.1 / Deliverable 1  -  Train the ResNet-50 teacher on CIFAR-100 and save the checkpoint.

Usage:  python src/train_teacher.py --epochs 80
"""
import argparse
import os

from common import (DATA_DIR, LOG_DIR, RES_DIR, Logger, ckpt_path, count_params, evaluate, fit,
                    get_device, get_loaders, save_checkpoint, set_seed, write_csv)
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    log = Logger(os.path.join(LOG_DIR, "q1_train_teacher.log"))
    log(f"=== Teacher ResNet-50 / CIFAR-100 | device={dev} | {vars(a)}")

    data = get_loaders("cifar100", a.batch_size, a.workers, data_dir=a.data_dir)
    model = build_model("resnet50", 100)
    log(f"parameters: {count_params(model):,}")

    hist = fit(model, data["train"], data["test"], a.epochs, a.lr, a.wd, dev, log, "teacher",
               label_smoothing=a.label_smoothing)
    final = evaluate(model, data["test"], dev)
    log(f"FINAL teacher test top-1 {final['top1']:.2f}%  top-5 {final['top5']:.2f}%")
    save_checkpoint(model.cpu(), ckpt_path("teacher"), {"name": "resnet50", "num_classes": 100},
                    dataset="cifar100", test_top1=final["top1"], test_top5=final["top5"], epochs=a.epochs)
    write_csv(os.path.join(RES_DIR, "q1_teacher_history.csv"), hist)
    log(f"saved -> {ckpt_path('teacher')}")


if __name__ == "__main__":
    main()
