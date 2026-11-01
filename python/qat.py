"""Quantization-aware training (fake quantization, straight-through estimator)
matching the integer scheme of quantize.py / the bare-metal C code:

  * conv weights: int8 symmetric per output channel.  Per-channel symmetric
    quantisation is invariant to the per-channel BN scale, so fake-quantising
    the pre-fold weight gives the same grid as quantising the folded weight.
  * FC weight: int8 symmetric per tensor.
  * activations at every quantisation point (image, FATReLU outputs, pre-add
    conv2/bn2 outputs, downsample outputs, avgpool output) with fixed per-tensor
    scales from calibration (saved next to the checkpoint and reused by
    quantize.py --scales so PTQ uses exactly the trained grid).
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import BLOCKS


def fq(x, s, lo, hi):
    """Fake-quantise with STE (gradient passes inside the clamp range)."""
    xf = x.float()
    xc = torch.clamp(xf, lo * s, hi * s)
    return (xc + (torch.round(xc / s) * s - xc).detach())


def fq_weight_per_channel(w):
    s = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True)
    s = (s.clamp_min(s.max() * 1e-3) / 127.0)
    return w + (torch.clamp(torch.round(w / s), -127, 127) * s - w).detach()


def fq_weight_per_tensor(w):
    s = w.detach().abs().max() / 127.0
    return w + (torch.clamp(torch.round(w / s), -127, 127) * s - w).detach()


def _conv_fq_forward(self, x):
    return self._conv_forward(x, fq_weight_per_channel(self.weight), self.bias)


def _linear_fq_forward(self, x):
    return F.linear(x, fq_weight_per_tensor(self.weight), self.bias)


def attach_qat(model, scales):
    """Patch `model` in place for QAT. Returns hook handles."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.forward = types.MethodType(_conv_fq_forward, m)
        elif isinstance(m, nn.Linear):
            m.forward = types.MethodType(_linear_fq_forward, m)
    hs = []

    def out_hook(s, lo, hi):
        return lambda mod, inp, out: fq(out, s, lo, hi)

    hs.append(model.conv1.register_forward_pre_hook(
        lambda mod, inp: (fq(inp[0], scales["in"], -128, 127),)))
    hs.append(model.act_stem.register_forward_hook(out_hook(scales["stem"], 0, 255)))
    for b in BLOCKS:
        blk = model.get_submodule(b)
        hs.append(blk.act1.register_forward_hook(out_hook(scales[f"{b}.act1"], 0, 255)))
        hs.append(blk.bn2.register_forward_hook(out_hook(scales[f"{b}.pre"], -128, 127)))
        if blk.downsample is not None:
            hs.append(blk.downsample.register_forward_hook(out_hook(scales[f"{b}.ds"], -128, 127)))
        hs.append(blk.act2.register_forward_hook(out_hook(scales[f"{b}.act2"], 0, 255)))
    hs.append(model.avgpool.register_forward_hook(out_hook(scales["pool"], 0, 255)))
    return hs
