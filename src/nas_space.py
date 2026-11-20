"""
nas_space.py
============
Q2.1 / Q2.2 / Deliverable 5  -  Searchable CNN architecture space + automatic architecture generator.

Search space (one architecture = one `config` dict, JSON serialisable)
  resolution      : input resolution the network runs at {16, 20, 24, 28, 32}
                    (CIFAR images are resized INSIDE the model, so the deployed model always takes 3x32x32)
  stem_channels   : {8, 16, 24, 32}
  num_stages      : {3, 4}                                   -> variable depth (number of stages)
  per stage:
    depth         : number of blocks {1, 2, 3, 4}            -> variable convolutional depth
    width         : output channels {16, 24, 32, 48, 64, 96, 128, 192}   -> variable filters/channels
    kernel        : {3, 5, 7}                                -> different kernel sizes
    block         : 'conv' (standard Conv-BN-Act) or 'dws' (depthwise-separable)
    activation    : relu | relu6 | hardswish | silu | gelu | leaky_relu
Stage strides are fixed to [1, 2, 2, 2]; blocks with stride 1 and equal channels get a residual connection.

`build_network(config)` is the automated generator: config -> torch.nn.Module (`SearchNet`).
`sample_config`, `mutate` and `config_hash` are used by the search algorithm.
"""
import copy
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

RESOLUTIONS = [16, 20, 24, 28, 32]
STEM_CHANNELS = [8, 16, 24, 32]
NUM_STAGES = [3, 4]
DEPTHS = [1, 2, 3, 4]
WIDTHS = [16, 24, 32, 48, 64, 96, 128, 192]
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dws"]
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu", "gelu", "leaky_relu"]
STRIDES = [1, 2, 2, 2]
MAX_STAGES = 4

ACT_FACTORY = {"relu": lambda: nn.ReLU(inplace=False), "relu6": lambda: nn.ReLU6(inplace=False),
               "hardswish": lambda: nn.Hardswish(), "silu": lambda: nn.SiLU(),
               "gelu": lambda: nn.GELU(), "leaky_relu": lambda: nn.LeakyReLU(0.1)}


# --------------------------------------------------------------------------- #
# Sampling / mutation / identity
# --------------------------------------------------------------------------- #
def sample_config(rng):
    """Uniform random architecture. `rng` is a random.Random instance."""
    return {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
            "num_stages": rng.choice(NUM_STAGES),
            "stages": [{"depth": rng.choice(DEPTHS), "width": rng.choice(WIDTHS),
                        "kernel": rng.choice(KERNELS), "block": rng.choice(BLOCKS),
                        "activation": rng.choice(ACTIVATIONS)} for _ in range(MAX_STAGES)]}


def canonical(cfg):
    """Config restricted to the active stages (unused stage entries do not change the network)."""
    c = copy.deepcopy(cfg)
    c["stages"] = c["stages"][:c["num_stages"]]
    return c


def config_hash(cfg):
    return hashlib.md5(json.dumps(canonical(cfg), sort_keys=True).encode()).hexdigest()[:10]


def _neighbour(choices, cur, rng):
    """Ordered choice lists: move to an adjacent value (70%) or jump anywhere else (30%)."""
    i = choices.index(cur)
    if rng.random() < 0.7:
        j = i + rng.choice([-1, 1])
        if 0 <= j < len(choices):
            return choices[j]
    others = [c for c in choices if c != cur]
    return rng.choice(others)


def mutate(cfg, rng):
    """Return a mutated copy (1-2 random changes) - the variation operator of the evolutionary search."""
    child = copy.deepcopy(cfg)
    for _ in range(rng.choice([1, 2])):
        field = rng.choice(["resolution", "stem_channels", "num_stages",
                            "depth", "width", "kernel", "block", "activation", "width", "depth"])
        if field == "resolution":
            child["resolution"] = _neighbour(RESOLUTIONS, child["resolution"], rng)
        elif field == "stem_channels":
            child["stem_channels"] = _neighbour(STEM_CHANNELS, child["stem_channels"], rng)
        elif field == "num_stages":
            child["num_stages"] = 7 - child["num_stages"]        # toggles 3 <-> 4
        else:
            st = child["stages"][rng.randrange(child["num_stages"])]
            if field == "depth":
                st["depth"] = _neighbour(DEPTHS, st["depth"], rng)
            elif field == "width":
                st["width"] = _neighbour(WIDTHS, st["width"], rng)
            elif field == "kernel":
                st["kernel"] = _neighbour(KERNELS, st["kernel"], rng)
            elif field == "block":
                st["block"] = "dws" if st["block"] == "conv" else "conv"
            else:
                st["activation"] = rng.choice([a for a in ACTIVATIONS if a != st["activation"]])
    return child


def describe(cfg):
    """One-line human readable summary."""
    c = canonical(cfg)
    st = " | ".join(f"{s['block']}x{s['depth']} w{s['width']} k{s['kernel']} {s['activation']}" for s in c["stages"])
    return f"res{c['resolution']} stem{c['stem_channels']} [{st}]"


def flatten_config(cfg):
    """Flat dict of the config (for CSV history)."""
    c = canonical(cfg)
    s = c["stages"]
    return {"resolution": c["resolution"], "stem_channels": c["stem_channels"], "num_stages": c["num_stages"],
            "depths": "-".join(str(x["depth"]) for x in s), "widths": "-".join(str(x["width"]) for x in s),
            "kernels": "-".join(str(x["kernel"]) for x in s), "blocks": "-".join(x["block"] for x in s),
            "activations": "-".join(x["activation"] for x in s)}


# --------------------------------------------------------------------------- #
# Architecture generator: config -> network
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """k x k Conv-BN-Act (standard convolution), residual if stride 1 and cin == cout."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = ACT_FACTORY[act]()
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.bn(self.conv(x))
        return self.act(y + x if self.res else y)


class DWSBlock(nn.Module):
    """Depthwise-separable block: depthwise k x k -> BN -> Act -> pointwise 1x1 -> BN (+res) -> Act."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = ACT_FACTORY[act]()
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = ACT_FACTORY[act]()
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.act1(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return self.act2(y + x if self.res else y)


class SearchNet(nn.Module):
    """Network generated from an architecture config. Input is always 3x32x32; the image is resized
    to `config['resolution']` inside the model, so latency/FLOPs reflect the chosen resolution."""

    def __init__(self, cfg, num_classes=10):
        super().__init__()
        cfg = canonical(cfg)
        self.cfg = cfg
        self.resolution = cfg["resolution"]
        first_act = cfg["stages"][0]["activation"]
        self.stem = nn.Sequential(nn.Conv2d(3, cfg["stem_channels"], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(cfg["stem_channels"]), ACT_FACTORY[first_act]())
        layers, cin = [], cfg["stem_channels"]
        for si, st in enumerate(cfg["stages"]):
            block = ConvBlock if st["block"] == "conv" else DWSBlock
            for b in range(st["depth"]):
                layers.append(block(cin, st["width"], st["kernel"], STRIDES[si] if b == 0 else 1, st["activation"]))
                cin = st["width"]
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        if self.resolution != x.shape[-1]:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return self.head(self.blocks(self.stem(x)))


def build_network(cfg, num_classes=10):
    """Automated architecture generator (Q2.2)."""
    return SearchNet(cfg, num_classes)
