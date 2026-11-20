"""
models.py
=========
Q1.1 / Q1.2 / Deliverables 1-2  -  Teacher and student architectures for 32x32 inputs.

  teacher : ResNet-50   (torchvision bottleneck stack, CIFAR stem = 3x3 conv, no max-pool)
  student : MobileNetV2 (torchvision implementation, stem stride 1 for CIFAR)
  student : SqueezeNet  (own implementation of SqueezeNet-v1.1 style Fire modules + BatchNorm)
  student : EfficientNet-Lite0 (own implementation: MBConv, ReLU6, no squeeze-excite, fixed stem/head)

All networks share one interface so the distillation code is architecture-agnostic:

    logits                   = model(x)
    logits, [f_mid, f_last]  = model(x, return_feats=True)

f_mid / f_last are two intermediate feature maps used by feature-based distillation
(FitNets-style hints).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


def _out(logits, feats, return_feats):
    return (logits, feats) if return_feats else logits


# --------------------------------------------------------------------------- #
# Teacher: ResNet-50
# --------------------------------------------------------------------------- #
class ResNet50CIFAR(nn.Module):
    """ResNet-50 for 32x32 images. Feature taps: layer3 (1024 ch, 8x8) and layer4 (2048 ch, 4x4)."""

    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)    # CIFAR stem (replaces 7x7/2 + max-pool)
        self.bn1, self.relu = r.bn1, r.relu
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.avgpool, self.fc = r.avgpool, r.fc

    def forward(self, x, return_feats=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer2(self.layer1(x))
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        logits = self.fc(torch.flatten(self.avgpool(f4), 1))
        return _out(logits, [f3, f4], return_feats)


# --------------------------------------------------------------------------- #
# Student 1: MobileNetV2
# --------------------------------------------------------------------------- #
class MobileNetV2CIFAR(nn.Module):
    """torchvision MobileNetV2 with stride-1 stem.  Taps: features[13] (96 ch, 4x4), features[18] (1280 ch, 2x2)."""

    def __init__(self, num_classes=100):
        super().__init__()
        m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
        m.features[0][0].stride = (1, 1)                       # 32x32 inputs: keep resolution in the stem
        self.features, self.classifier = m.features, m.classifier

    def forward(self, x, return_feats=False):
        mid = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 13:
                mid = x
        logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        return _out(logits, [mid, x], return_feats)


# --------------------------------------------------------------------------- #
# Student 2: SqueezeNet (v1.1-style, with BatchNorm for stable from-scratch training)
# --------------------------------------------------------------------------- #
class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.expand1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1, bias=False), nn.BatchNorm2d(e1))
        self.expand3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1, bias=False), nn.BatchNorm2d(e3))

    def forward(self, x):
        s = self.squeeze(x)
        return F.relu(torch.cat([self.expand1(s), self.expand3(s)], dim=1), inplace=True)


class SqueezeNetCIFAR(nn.Module):
    """Taps: output of fire6 (384 ch, 8x8) and fire8 (512 ch, 4x4)."""
    CFG = [(64, 16, 64, 64), (128, 16, 64, 64), (128, 32, 128, 128), (256, 32, 128, 128),
           (256, 48, 192, 192), (384, 48, 192, 192), (384, 64, 256, 256), (512, 64, 256, 256)]

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fires = nn.ModuleList([Fire(*c) for c in self.CFG])
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(512, num_classes, 1),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x, return_feats=False):
        f = self.fires
        x = f[1](f[0](self.stem(x)))
        x = F.max_pool2d(x, 2)                                 # 32 -> 16
        x = f[3](f[2](x))
        x = F.max_pool2d(x, 2)                                 # 16 -> 8
        mid = f[5](f[4](x))
        x = F.max_pool2d(mid, 2)                               # 8 -> 4
        last = f[7](f[6](x))
        return _out(self.head(last), [mid, last], return_feats)


# --------------------------------------------------------------------------- #
# Student 3: EfficientNet-Lite0
# --------------------------------------------------------------------------- #
def _cbr(cin, cout, k, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    """Inverted residual block (expand 1x1 -> depthwise kxk -> project 1x1). No squeeze-excite (Lite)."""

    def __init__(self, cin, cout, k, stride, expand):
        super().__init__()
        mid = cin * expand
        layers = []
        if expand != 1:
            layers.append(_cbr(cin, mid, 1))
        layers += [_cbr(mid, mid, k, stride, groups=mid), _cbr(mid, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class EfficientNetLite0(nn.Module):
    """EfficientNet-B0 layout with the 'Lite' changes: ReLU6, no squeeze-excite, stem/head not scaled.
    CIFAR adaptation: stem stride 1.  Taps: stage 5 (112 ch, 4x4) and head conv (1280 ch, 2x2)."""
    # (expand, out_channels, repeats, stride, kernel)
    STAGES = [(1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
              (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = _cbr(3, 32, 3, 1)
        stages, cin = [], 32
        for t, c, n, s, k in self.STAGES:
            blocks = []
            for i in range(n):
                blocks.append(MBConv(cin, c, k, s if i == 0 else 1, t))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.head_conv = _cbr(cin, 1280, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        init_weights(self)

    def forward(self, x, return_feats=False):
        x = self.stem(x)
        mid = None
        for i, st in enumerate(self.stages):
            x = st(x)
            if i == 4:
                mid = x
        x = self.head_conv(x)
        logits = self.fc(self.dropout(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))
        return _out(logits, [mid, x], return_feats)


# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {"resnet50": ResNet50CIFAR, "mobilenetv2": MobileNetV2CIFAR,
                  "squeezenet": SqueezeNetCIFAR, "efficientnet_lite0": EfficientNetLite0}


def build_model(name, num_classes=100):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}', choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes)
