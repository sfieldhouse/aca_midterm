"""W8A8 post-training quantization of (FAT)ReLU ResNet-18 + bit-exact integer reference.

Integer scheme (identical to the bare-metal C implementation):
  * BN folded into conv; conv weights int8 symmetric per output channel,
    bias int32 (scale s_in*s_w[c]).  FC weights int8 per-tensor (argmax-safe).
  * Activations: zero point 0 so real 0 == integer 0 (needed for sparsity).
      - image input           int8   (symmetric)
      - post-(FAT)ReLU tensors uint8  [0,255]
      - pre-add conv2 / downsample outputs int8
  * Requantisation  y = sat(((acc * M + 2^(sh-1)) >> sh) + zp), zp = 0,
    M < 2^31, 1 <= sh <= 63; ReLU outputs: y < T -> 0  (FATReLU threshold).
  * Residual add: v = y2*A + x*B (A,B integers, k = 16 fractional bits),
    out = requant(v, M=1, sh=k) with ReLU/FATReLU.
  * Avgpool: sum of 7x7 -> requant(M,sh) to uint8.   FC: int32/64 logits.
"""
import math
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_fat import ResNet18, BasicBlock  # noqa: E402

ADD_K = 16
BLOCKS = [f"layer{l}.{b}" for l in range(1, 5) for b in range(2)]


# --------------------------------------------------------------------------- helpers
def fold_bn(conv, bn):
    w = conv.weight.detach().double()
    g = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    return w * g[:, None, None, None], bn.bias.detach().double() - bn.running_mean.double() * g


def mult_shift(r):
    """Real multiplier r>0 -> (M, sh) with M in [2^30, 2^31), r ~= M * 2^-sh."""
    r = np.asarray(r, dtype=np.float64)
    e = np.floor(np.log2(r)).astype(np.int64)
    sh = 30 - e
    M = np.round(r * np.exp2(sh.astype(np.float64))).astype(np.int64)
    over = M >= (1 << 31)
    M[over] //= 2
    sh[over] -= 1
    big = sh > 63                      # negligible multiplier (dead channel): keep sh = 63
    M[big] = np.round(r[big] * 2.0 ** 63).astype(np.int64)
    sh[big] = 63
    assert np.all(sh >= 1) and np.all(sh <= 63), sh
    return M, sh


def requant_t(acc, M, sh, mode, T=0):
    """Integer requantisation on int64 tensors; M, sh broadcast over dim 1 (channels)."""
    v = acc * M
    v = (v + (torch.ones_like(sh) << (sh - 1))) >> sh
    if mode == 1:
        v = v.clamp(0, 255)
        if T > 0:
            v = torch.where(v < T, torch.zeros_like(v), v)
    else:
        v = v.clamp(-128, 127)
    return v


def conv_i(x, w, stride, pad):
    """Exact integer convolution (float64 is exact for these magnitudes)."""
    return F.conv2d(x.double(), w.double(), stride=stride, padding=pad).round().long()


def best_scale(v, qmax, signed):
    """MSE-optimal clipping for symmetric/unsigned quantisation of sample v."""
    v = v.float()
    amax = v.abs().max().item()
    best = (float("inf"), amax / qmax)
    for frac in torch.linspace(0.3, 1.0, 71):
        c = amax * frac.item()
        s = c / qmax
        lo = -qmax - 1 if signed else 0
        qv = torch.clamp(torch.round(v / s), lo, qmax) * s
        err = (qv - v).pow(2).mean().item()
        if err < best[0]:
            best = (err, s)
    return best[1]


# --------------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(model: ResNet18, batches, device="cuda", sample=2_000_000):
    """Collect samples at every quantisation point and return per-point scales."""
    model = model.to(device).eval()
    pts = {}

    def keep(name):
        def h(mod, inp, out):
            o = out.flatten()
            idx = torch.randint(0, o.numel(), (min(sample // len(batches), o.numel()),), device=o.device)
            pts.setdefault(name, []).append(o[idx].float())
        return h

    hs = [model.act_stem.register_forward_hook(keep("stem"))]
    for b in BLOCKS:
        blk = model.get_submodule(b)
        hs.append(blk.act1.register_forward_hook(keep(f"{b}.act1")))
        hs.append(blk.bn2.register_forward_hook(keep(f"{b}.pre")))
        if blk.downsample is not None:
            hs.append(blk.downsample.register_forward_hook(keep(f"{b}.ds")))
        hs.append(blk.act2.register_forward_hook(keep(f"{b}.act2")))
    hs.append(model.avgpool.register_forward_hook(keep("pool")))
    xs = []
    for x in batches:
        x = x.to(device)
        xs.append(x.flatten()[torch.randint(0, x.numel(), (sample // len(batches),), device=device)])
        model(x)
    for h in hs:
        h.remove()
    scales = {"in": best_scale(torch.cat(xs), 127, True)}
    for k, v in pts.items():
        v = torch.cat(v)
        signed = k.endswith(".pre") or k.endswith(".ds")
        scales[k] = best_scale(v, 127 if signed else 255, signed)
    return scales


# --------------------------------------------------------------------------- build
def build_qmodel(model: ResNet18, scales):
    th = model.thresholds()
    q = {"scales": scales, "convs": {}, "adds": {}, "thresholds": th}

    def tq(site, s_out):
        t = th.get(site, 0.0)
        return int(round(t / s_out)) if t > 0 else 0

    def qconv(name, conv, bn, s_in, s_out, mode, T):
        w, b = fold_bn(conv, bn)
        s_w = w.abs().amax(dim=(1, 2, 3)) / 127.0
        s_w = s_w.clamp_min(s_w.max() * 1e-3)          # dead channels (BN gamma ~ 0)
        wq = torch.round(w / s_w[:, None, None, None]).clamp(-127, 127).to(torch.int8)
        bq = torch.round(b / (s_in * s_w)).to(torch.int64)
        M, sh = mult_shift((s_in * s_w / s_out).numpy())
        q["convs"][name] = dict(w=wq, bias=bq, M=torch.from_numpy(M), sh=torch.from_numpy(sh),
                               mode=mode, T=T, stride=conv.stride[0], pad=conv.padding[0],
                               k=conv.kernel_size[0], s_in=s_in, s_out=s_out)

    qconv("conv1", model.conv1, model.bn1, scales["in"], scales["stem"], 1, tq("act_stem", scales["stem"]))
    x_s = scales["stem"]
    for b in BLOCKS:
        blk = model.get_submodule(b)
        qconv(f"{b}.conv1", blk.conv1, blk.bn1, x_s, scales[f"{b}.act1"], 1, tq(f"{b}.act1", scales[f"{b}.act1"]))
        qconv(f"{b}.conv2", blk.conv2, blk.bn2, scales[f"{b}.act1"], scales[f"{b}.pre"], 0, 0)
        if blk.downsample is not None:
            qconv(f"{b}.downsample.0", blk.downsample[0], blk.downsample[1], x_s, scales[f"{b}.ds"], 0, 0)
            r_s = scales[f"{b}.ds"]
        else:
            r_s = x_s
        s_out = scales[f"{b}.act2"]
        A = int(round(scales[f"{b}.pre"] / s_out * (1 << ADD_K)))
        B = int(round(r_s / s_out * (1 << ADD_K)))
        assert 0 < A < (1 << 24) and 0 < B < (1 << 24)
        q["adds"][b] = dict(A=A, B=B, k=ADD_K, T=tq(f"{b}.act2", s_out), s_out=s_out)
        x_s = s_out
    M, sh = mult_shift(np.array([x_s / (49.0 * scales["pool"])]))
    q["pool"] = dict(M=int(M[0]), sh=int(sh[0]))
    w = model.fc.weight.detach().double()
    s_w = w.abs().max() / 127.0
    q["fc"] = dict(w=torch.round(w / s_w).clamp(-127, 127).to(torch.int8),
                   bias=torch.round(model.fc.bias.detach().double() / (scales["pool"] * s_w)).to(torch.int64))
    return q


def quantize_input(x_norm, q):
    return torch.round(x_norm / q["scales"]["in"]).clamp(-128, 127).long()


# --------------------------------------------------------------------------- integer forward
class QRunner:
    """Bit-exact integer forward pass on GPU (or CPU)."""

    def __init__(self, q, device="cuda"):
        self.q, self.dev = q, device
        self.c = {}
        for n, c in q["convs"].items():
            self.c[n] = dict(c, w=c["w"].to(device), bias=c["bias"].to(device)[None, :, None, None],
                             M=c["M"].to(device)[None, :, None, None], sh=c["sh"].to(device)[None, :, None, None])
        self.fc_w = q["fc"]["w"].to(device).double()
        self.fc_b = q["fc"]["bias"].to(device)

    def conv(self, name, x):
        c = self.c[name]
        acc = conv_i(x, c["w"], c["stride"], c["pad"]) + c["bias"]
        return requant_t(acc, c["M"], c["sh"], c["mode"], c["T"])

    @torch.no_grad()
    def __call__(self, xq, stats=None):
        """xq: int64 NCHW image tensor.  stats: optional dict collecting zero counts of conv inputs."""
        def rec(name, t):
            if stats is not None:
                z, n = stats.get(name, (0, 0))
                stats[name] = (z + (t == 0).sum().item(), n + t.numel())

        x = xq.to(self.dev)
        rec("conv1", x)
        x = self.conv("conv1", x)
        x = F.max_pool2d(x.double(), 3, 2, 1).long()
        for b in BLOCKS:
            rec(f"{b}.conv1", x)
            y = self.conv(f"{b}.conv1", x)
            rec(f"{b}.conv2", y)
            y = self.conv(f"{b}.conv2", y)
            if f"{b}.downsample.0" in self.c:
                rec(f"{b}.downsample.0", x)
                r = self.conv(f"{b}.downsample.0", x)
            else:
                r = x
            a = self.q["adds"][b]
            v = y * a["A"] + r * a["B"]
            x = requant_t(v, torch.tensor(1, device=self.dev), torch.tensor(a["k"], device=self.dev), 1, a["T"])
        s = x.sum(dim=(2, 3))
        p = self.q["pool"]
        s = requant_t(s, torch.tensor(p["M"], device=self.dev), torch.tensor(p["sh"], device=self.dev), 1)
        rec("fc", s)
        return (s.double() @ self.fc_w.t()).round().long() + self.fc_b


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    import json
    from data import eval_tf, imagenet_train, imagenet_val, loader  # noqa: E402
    from resnet_fat import load_pretrained  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True, help="output .pt with integer model")
    ap.add_argument("--calib_batches", type=int, default=8)
    ap.add_argument("--eval", choices=["none", "val"], default="val")
    ap.add_argument("--json", default=None)
    ap.add_argument("--scales", default=None, help="use these activation scales (e.g. from QAT) instead of calibrating")
    a = ap.parse_args()

    torch.manual_seed(0)
    model = load_pretrained(a.ckpt)
    g = torch.Generator().manual_seed(0)
    cal_ds = imagenet_train(eval_tf)
    idx = torch.randperm(len(cal_ds), generator=g)[: a.calib_batches * 128]
    cal = loader(torch.utils.data.Subset(cal_ds, idx.tolist()), 128, workers=16)
    scales = json.load(open(a.scales)) if a.scales else calibrate(model, [x for x, _ in cal])
    q = build_qmodel(model.cpu(), scales)
    torch.save(q, a.out)
    print("scales:", {k: round(v, 5) for k, v in scales.items()})
    print("int thresholds:", {n: c["T"] for n, c in q["convs"].items() if c["T"]}, {b: v["T"] for b, v in q["adds"].items() if v["T"]})
    if a.eval == "val":
        run = QRunner(q)
        c1 = n = 0
        stats = {}
        for x, y in loader(imagenet_val(), 250, workers=32):
            lg = run(quantize_input(x, q), stats)
            c1 += (lg.argmax(1).cpu() == y).sum().item()
            n += y.numel()
        sp = {k: z / t for k, (z, t) in stats.items()}
        r = {"ckpt": a.ckpt, "int8_top1": c1 / n, "n": n, "conv_input_sparsity_int8": sp,
             "mean_sparsity_19": float(np.mean([v for k, v in sp.items() if k not in ("conv1", "fc")]))}
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}))
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)
