"""FATReLU activation-sensitivity analysis (Kurtz et al. 2020, Sec. 3).

For each of the 17 activation sites independently, raise the FATReLU threshold
so that an extra fraction f of that site's positive activations is zeroed, and
record the minival loss / top-1.  The per-site threshold is the largest one
whose loss increase stays within `--tol` (absolute CE).  Results + chosen
thresholds are written as JSON (thresholds json is consumed by
finetune_sparse.py --thresholds).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import imagenet_minival, loader  # noqa: E402
from resnet_fat import load_pretrained  # noqa: E402

FRACS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


@torch.no_grad()
def run(model, xs, ys, bs=500):
    loss = c1 = 0.0
    for i in range(0, len(xs), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(xs[i:i + bs]).float()
        loss += F.cross_entropy(o, ys[i:i + bs], reduction="sum").item()
        c1 += (o.argmax(1) == ys[i:i + bs]).sum().item()
    return loss / len(xs), c1 / len(xs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.005, help="max per-site CE increase")
    ap.add_argument("--exclude", nargs="*", default=["layer4.1.act2"],
                    help="sites left at T=0 (layer4.1.act2 feeds only avgpool/fc: no sparse-conv benefit)")
    a = ap.parse_args()
    model = load_pretrained(a.ckpt).cuda().eval()
    xs, ys = [], []
    for x, y in loader(imagenet_minival(), 250, workers=16):
        xs.append(x)
        ys.append(y)
        if sum(len(t) for t in xs) >= a.n:
            break
    xs = torch.cat(xs)[: a.n].cuda().to(torch.bfloat16)
    ys = torch.cat(ys)[: a.n].cuda()
    base = model.thresholds()
    L0, A0 = run(model, xs, ys)
    print(f"base loss {L0:.4f} top1 {A0:.4f}")

    # positive-activation quantiles per site (from a subset)
    model.set_record(True)
    qs = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(xs[:500])
    for n, m in model.act_sites():
        v = m.last.float().flatten()
        v = v[v > 0]
        v = v[torch.randint(0, v.numel(), (min(v.numel(), 4_000_000),), device=v.device)]
        qs[n] = {f: torch.quantile(v, f).item() for f in FRACS}
    model.set_record(False)

    res = {"base": {"loss": L0, "top1": A0, "thresholds": base}, "sites": {}}
    chosen = dict(base)
    for n, m in model.act_sites():
        rows = []
        for f in FRACS:
            t = max(qs[n][f], base[n])
            m.threshold.fill_(t)
            L, A = run(model, xs, ys)
            rows.append({"frac": f, "T": t, "loss": L, "top1": A})
            if L - L0 <= a.tol:
                chosen[n] = t
        m.threshold.fill_(base[n])
        res["sites"][n] = rows
        print(n, " ".join(f"{r['frac']:.2f}:{r['loss'] - L0:+.4f}" for r in rows), "->", f"{chosen[n]:.4f}", flush=True)
    for n in a.exclude:
        chosen[n] = base[n]
    model.set_thresholds(chosen)
    Lc, Ac = run(model, xs, ys)
    res["chosen"] = chosen
    res["combined"] = {"loss": Lc, "top1": Ac, "tol": a.tol}
    print(f"combined loss {Lc:.4f} top1 {Ac:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(chosen, open(a.out.replace(".json", "_thresholds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
