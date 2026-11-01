"""ResNet-18 with an independent FATReLU module at every activation site.

torchvision's BasicBlock re-uses a single nn.ReLU for both activations of the
block, which makes per-site thresholds / statistics impossible.  This module
re-implements ResNet-18 with the *same parameter names* as torchvision/timm so
that pretrained checkpoints load directly, but with 17 separately addressable
activation sites:

    act_stem, layer{1..4}.{0,1}.act1, layer{1..4}.{0,1}.act2

FATReLU (Kurtz et al., ICML 2020):  FATReLU_T(x) = x if x >= T else 0.
With T = 0 this is exactly ReLU.
"""
import torch
import torch.nn as nn


class FATReLU(nn.Module):
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        # buffer -> saved in state_dict, moved with .to(), not trained
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.record = False      # when True, keep last output for Hoyer / stats
        self.last = None

    def forward(self, x):
        # x * [x >= T]; for T == 0 this equals ReLU (same value and gradient)
        y = x * (x >= self.threshold).to(x.dtype)
        if self.record:
            self.last = y
        return y

    def extra_repr(self):
        return f"threshold={self.threshold.item():.4f}"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = FATReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = FATReLU()
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(y + idt)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act_stem = FATReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, stride):
        ds = None
        if stride != 1 or self.inplanes != planes:
            ds = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                               nn.BatchNorm2d(planes))
        blocks = [BasicBlock(self.inplanes, planes, stride, ds), BasicBlock(planes, planes)]
        self.inplanes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.act_stem(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))

    # ---- helpers -----------------------------------------------------------
    def act_sites(self):
        """Ordered list of (name, FATReLU) for all 17 activation sites."""
        return [(n, m) for n, m in self.named_modules() if isinstance(m, FATReLU)]

    def set_record(self, flag: bool):
        for _, m in self.act_sites():
            m.record = flag
            if not flag:
                m.last = None

    def thresholds(self):
        return {n: m.threshold.item() for n, m in self.act_sites()}

    def set_thresholds(self, th: dict):
        for n, m in self.act_sites():
            if n in th:
                m.threshold.fill_(float(th[n]))


def conv_input_sites():
    """For each of the 20 convs: (conv name, activation site producing its input).
    The stem conv input is the image (None).  Order = execution order."""
    out = [("conv1", None)]
    prev = "act_stem"          # maxpool(act_stem) feeds layer1.0
    for li in range(1, 5):
        for bi in range(2):
            b = f"layer{li}.{bi}"
            out.append((f"{b}.conv1", prev))
            if li > 1 and bi == 0:
                out.append((f"{b}.downsample.0", prev))
            out.append((f"{b}.conv2", f"{b}.act1"))
            prev = f"{b}.act2"
    return out


def hoyer_square(a: torch.Tensor, eps=1e-12):
    """Square Hoyer regularizer H(v) = (sum|v|)^2 / sum v^2 over the whole tensor."""
    a = a.float()
    return a.abs().sum() ** 2 / (a.pow(2).sum() + eps)


def load_pretrained(path):
    """Load a torchvision/timm ResNet-18 state_dict (safetensors or .pt)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
    m = ResNet18()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("threshold")]
    assert not missing and not unexpected, (missing, unexpected)
    return m
