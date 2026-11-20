"""
distill.py
==========
Q1.3 / Q1.4 / Deliverable 4  -  Knowledge-distillation framework + training of the 3 students.

Implemented components
  1. Soft-target distillation   : KL( softmax(t/T) || softmax(s/T) )  on the teacher's predictions
  2. Temperature softmax        : both distributions are softened with temperature T, and the
                                  soft term is multiplied by T^2 so its gradient scale stays constant
  3. Hard + soft combination    : L = alpha * T^2 * KL_soft + (1 - alpha) * CE(hard labels)
  4. Feature-based distillation : FitNets-style hints.  Student feature maps are mapped with a
                                  learnable 1x1 conv + BN to the teacher's channel count and compared
                                  with the (spatially pooled) teacher maps (cosine or MSE loss):
                                      L_total = L_kd + beta * sum_i L_feat_i
Methods (--method)
  soft      alpha = 1, beta = 0         (pure soft-target distillation)
  kd        alpha = --alpha, beta = 0   (hard + soft, Hinton et al.)
  kd_feat   alpha = --alpha, beta = --beta (hard + soft + feature)        [default, checkpoint kd_<student>.pth]

Usage:  python src/distill.py --student all --method kd_feat --epochs 100
        python src/distill.py --student mobilenetv2 --method kd        # ablation
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (DATA_DIR, LOG_DIR, RES_DIR, STUDENTS, CKPT_DIR, Logger, ckpt_path, count_params,
                    evaluate, fit, get_device, get_loaders, load_checkpoint, save_checkpoint,
                    set_seed, write_csv)
from models import build_model


# --------------------------------------------------------------------------- #
# Loss components
# --------------------------------------------------------------------------- #
def soft_target_loss(student_logits, teacher_logits, T):
    """Temperature-softmax KL divergence, scaled by T^2 (Hinton et al., 2015)."""
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = F.softmax(teacher_logits.float() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def feature_loss(s_feat, t_feat, kind="cosine"):
    """Compare a (regressed) student map with the teacher map; teacher is pooled to the student's HxW."""
    if t_feat.shape[-2:] != s_feat.shape[-2:]:
        t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[-2:])
    s_feat, t_feat = s_feat.float(), t_feat.float()
    if kind == "cosine":                                      # scale-invariant: 1 - cos over channels
        return (1.0 - F.cosine_similarity(s_feat, t_feat, dim=1)).mean()
    return F.mse_loss(s_feat, t_feat)                         # classic FitNets hint loss


class FeatureHint(nn.Module):
    """Regressor: student channels -> teacher channels (1x1 conv + BN)."""

    def __init__(self, c_student, c_teacher):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_student, c_teacher, 1, bias=False), nn.BatchNorm2d(c_teacher))

    def forward(self, x):
        return self.net(x)


class DistillationLoss(nn.Module):
    """Combined hard + soft + feature objective.  The hint regressors are part of this module, so
    they must be trained together with the student (fit(..., extra_modules=[criterion]))."""

    def __init__(self, student, teacher, device, T=4.0, alpha=0.7, beta=0.0, feat_loss="cosine"):
        super().__init__()
        self.T, self.alpha, self.beta, self.feat_kind = T, alpha, beta, feat_loss
        self.hints = nn.ModuleList()
        if beta > 0:                                          # discover channel counts with a dummy pass
            with torch.no_grad():
                dummy = torch.zeros(2, 3, 32, 32, device=device)
                _, sf = student.eval()(dummy, return_feats=True)
                _, tf = teacher.eval()(dummy, return_feats=True)
            self.hints = nn.ModuleList([FeatureHint(s.shape[1], t.shape[1]) for s, t in zip(sf, tf)])

    def forward(self, s_logits, t_logits, y, s_feats=None, t_feats=None):
        ce = F.cross_entropy(s_logits.float(), y)
        kd = soft_target_loss(s_logits, t_logits, self.T)
        loss = self.alpha * kd + (1.0 - self.alpha) * ce
        feat = torch.zeros((), device=s_logits.device)
        if self.beta > 0:
            for hint, sf, tf in zip(self.hints, s_feats, t_feats):
                feat = feat + feature_loss(hint(sf), tf.detach(), self.feat_kind)
            loss = loss + self.beta * feat
        return loss, {"ce": ce.item(), "kd": kd.item(), "feat": float(feat)}


METHODS = {"soft": lambda a: (1.0, 0.0),
           "kd": lambda a: (a.alpha, 0.0),
           "kd_feat": lambda a: (a.alpha, a.beta)}


def out_path(student, method):
    return ckpt_path(f"{student}_kd") if method == "kd_feat" else \
        os.path.join(CKPT_DIR, f"kd_{student}_{method}.pth")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="all", help="mobilenetv2 | squeezenet | efficientnet_lite0 | all")
    ap.add_argument("--method", default="kd_feat", choices=list(METHODS))
    ap.add_argument("--teacher", default=ckpt_path("teacher"))
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of the soft-target term")
    ap.add_argument("--beta", type=float, default=2.0, help="weight of the feature-distillation term")
    ap.add_argument("--feat-loss", default="cosine", choices=["cosine", "mse"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    dev = get_device()
    log = Logger(os.path.join(LOG_DIR, "q1_distill.log"))
    alpha, beta = METHODS[a.method](a)
    names = STUDENTS if a.student == "all" else [a.student]
    data = get_loaders("cifar100", a.batch_size, a.workers, data_dir=a.data_dir)

    teacher, tck = load_checkpoint(a.teacher)
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_acc = evaluate(teacher, data["test"], dev)
    log(f"teacher test top-1 {t_acc['top1']:.2f}%  top-5 {t_acc['top5']:.2f}%")

    for name in names:
        set_seed(a.seed)
        log(f"=== KD [{a.method}] teacher=ResNet-50 -> student={name} | T={a.T} alpha={alpha} beta={beta} | {vars(a)}")
        student = build_model(name, 100).to(dev)
        log(f"student parameters: {count_params(student):,}")
        crit = DistillationLoss(student, teacher, dev, a.T, alpha, beta, a.feat_loss).to(dev)

        def step_fn(x, y):
            with torch.no_grad():
                t_logits, t_feats = teacher(x, return_feats=True)
            s_logits, s_feats = student(x, return_feats=True)
            loss, _ = crit(s_logits, t_logits, y, s_feats, t_feats)
            return loss, s_logits

        hist = fit(student, data["train"], data["test"], a.epochs, a.lr, a.wd, dev, log,
                   f"kd-{name}", step_fn=step_fn, extra_modules=[crit])
        final = evaluate(student, data["test"], dev)
        log(f"FINAL {name} [{a.method}] test top-1 {final['top1']:.2f}%  top-5 {final['top5']:.2f}%")
        save_checkpoint(student.cpu(), out_path(name, a.method), {"name": name, "num_classes": 100},
                        dataset="cifar100", test_top1=final["top1"], test_top5=final["top5"],
                        epochs=a.epochs, method=a.method, T=a.T, alpha=alpha, beta=beta,
                        teacher_top1=t_acc["top1"])
        write_csv(os.path.join(RES_DIR, f"q1_kd_{name}_{a.method}_history.csv"), hist)
        log(f"saved -> {out_path(name, a.method)}")


if __name__ == "__main__":
    main()
