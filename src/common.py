"""
common.py
=========
Shared utilities for Assignment 4 (Knowledge Distillation + Hardware-aware NAS).

  * paths, logging, seeding
  * CIFAR-10 / CIFAR-100 data loaders (+ in-memory test arrays for inference/benchmarks)
  * generic training loop (`fit`) with AMP, cosine LR and warm-up, used by EVERY training script
  * evaluation (top-1 / top-5), parameter / MAC counting, model size, CPU latency
  * checkpoint save / load and the model-name registry used by inference + benchmarks
"""
import copy
import csv
import io
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
RES_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RES_DIR, "logs")
for _d in (CKPT_DIR, ONNX_DIR, RES_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DATASETS = {
    "cifar10": dict(cls=torchvision.datasets.CIFAR10, num_classes=10,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(cls=torchvision.datasets.CIFAR100, num_classes=100,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
}

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
# every deployable model name understood by inference.py / export_onnx.py / benchmarks
ALL_MODELS = (["teacher"] + [f"{s}_baseline" for s in STUDENTS] + [f"{s}_kd" for s in STUDENTS]
              + ["nas_best"])


def ckpt_path(name):
    """Registry:  model name -> checkpoint file."""
    if name == "teacher":
        return os.path.join(CKPT_DIR, "teacher_resnet50.pth")
    if name == "nas_best":
        return os.path.join(CKPT_DIR, "nas_best.pth")
    student, kind = name.rsplit("_", 1)                   # e.g. 'mobilenetv2_kd'
    return os.path.join(CKPT_DIR, f"{'kd' if kind == 'kd' else 'baseline'}_{student}.pth")


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"{name}.onnx")


def dataset_of(name):
    """The NAS model works on CIFAR-10, everything else on CIFAR-100."""
    return "cifar10" if name == "nas_best" else "cifar100"


def get_loaders(name, batch_size=128, workers=4, val_size=0, train_subset=None, seed=0,
                data_dir=DATA_DIR):
    """Return dict(train, val, test, classes).  `val_size` images are held out of the training
    set (no augmentation) for validation; `train_subset` limits the training set (NAS proxy)."""
    cfg = DATASETS[name]
    norm = T.Normalize(cfg["mean"], cfg["std"])
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_eval = T.Compose([T.ToTensor(), norm])
    train_full = cfg["cls"](data_dir, train=True, transform=tf_train, download=True)
    eval_full = cfg["cls"](data_dir, train=True, transform=tf_eval, download=True)
    test = cfg["cls"](data_dir, train=False, transform=tf_eval, download=True)

    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    if train_subset:
        train_idx = train_idx[:train_subset]
    pin = torch.cuda.is_available()
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                        pin_memory=pin, drop_last=shuffle,
                                        persistent_workers=workers > 0)
    return {"train": mk(Subset(train_full, train_idx), True),
            "val": mk(Subset(eval_full, val_idx), False) if val_size else None,
            "test": mk(test, False), "classes": test.classes}


def load_test_arrays(name, data_dir=DATA_DIR):
    """Full test set as uint8 tensor [N,3,32,32] + labels (used by inference / benchmarks)."""
    ds = DATASETS[name]["cls"](data_dir, train=False, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(ds.targets, dtype=torch.long), ds.classes


def normalize_u8(x_u8, name):
    cfg = DATASETS[name]
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    return (x_u8.float() / 255.0 - mean) / std


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    """print() + append to a log file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def build_from_arch(arch):
    """arch = {'name': 'mobilenetv2'|..., 'num_classes': n}  or  {'name': 'nas', 'config': {...}}"""
    if arch["name"] == "nas":
        from nas_space import SearchNet
        return SearchNet(arch["config"], arch.get("num_classes", 10))
    from models import build_model
    return build_model(arch["name"], arch.get("num_classes", 100))


def save_checkpoint(model, path, arch, **extra):
    torch.save({"state_dict": model.state_dict(), "arch": arch, **extra}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model = build_from_arch(ck["arch"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def load_model_by_name(name):
    return load_checkpoint(ckpt_path(name))[0]


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def make_scheduler(opt, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay (stepped once per iteration)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return dict(top1, top5, loss) in percent / nats."""
    model.eval()
    tot = c1 = c5 = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x).float()
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        k = min(5, out.shape[1])
        top = out.topk(k, dim=1).indices
        c1 += (top[:, 0] == y).sum().item()
        c5 += (top == y.unsqueeze(1)).any(1).sum().item()
        tot += y.numel()
    return {"top1": 100 * c1 / tot, "top5": 100 * c5 / tot, "loss": loss_sum / tot}


def train_epoch(model, loader, opt, sched, step_fn, device, scaler, amp, extra_modules=()):
    model.train()
    for m in extra_modules:
        m.train()
    loss_sum = correct = n = 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            loss, logits = step_fn(x, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sched.step()
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, 100 * correct / n


def fit(model, train_loader, eval_loader, epochs, lr, wd, device, log, tag, step_fn=None,
        extra_modules=(), label_smoothing=0.0, warmup_epochs=1, on_epoch_end=None):
    """Generic SGD training loop shared by teacher / baseline / KD / NAS scripts.

    step_fn(x, y) -> (loss, logits) lets the caller customise the objective (knowledge
    distillation).  extra_modules are additional trainable modules (e.g. feature regressors)
    whose parameters are optimised together with the model.  Returns a list of per-epoch dicts.
    """
    model.to(device)
    for m in extra_modules:
        m.to(device)
    params = list(model.parameters())
    for m in extra_modules:
        params += list(m.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    total = epochs * len(train_loader)
    sched = make_scheduler(opt, total, max(1, warmup_epochs * len(train_loader)))
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp else None
    if step_fn is None:
        crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        def step_fn(x, y):
            logits = model(x)
            return crit(logits.float(), y), logits

    history, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, sched, step_fn, device, scaler, amp,
                                      extra_modules)
        ev = evaluate(model, eval_loader, device)
        row = {"epoch": ep, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 2),
               "eval_top1": round(ev["top1"], 2), "eval_top5": round(ev["top5"], 2),
               "eval_loss": round(ev["loss"], 4), "lr": round(sched.get_last_lr()[0], 6),
               "elapsed_min": round((time.time() - t0) / 60, 2)}
        history.append(row)
        log(f"[{tag}] ep {ep:3d}/{epochs} train_loss {row['train_loss']:.4f} train_acc {row['train_acc']:.2f} "
            f"| eval top1 {row['eval_top1']:.2f} top5 {row['eval_top5']:.2f} | {row['elapsed_min']:.1f} min")
        if on_epoch_end:
            on_epoch_end(ep, row)
    return history


# --------------------------------------------------------------------------- #
# Model statistics
# --------------------------------------------------------------------------- #
def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model, input_shape=(1, 3, 32, 32)):
    """Multiply-accumulates of all Conv2d / Linear layers (groups-aware, so depthwise convs are right)."""
    total = [0]
    hooks = []

    def conv_hook(m, i, o):
        total[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        total[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(*input_shape, device=p.device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total[0]


def state_dict_size_mb(model):
    """Storage size of the FP32 weights (serialised state_dict) in MB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def measure_latency(fn, sample, warmup=20, iters=100):
    """Latency in ms of fn(sample): returns dict(mean, p50, p95)."""
    for _ in range(warmup):
        fn(sample)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn(sample)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean": float(ts.mean()), "p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95))}
