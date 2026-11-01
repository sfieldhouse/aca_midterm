"""FP32 activation sparsity per conv input / activation site on ImageNet val
(a fixed subset), plus an activation histogram of one layer (paper Fig. 1).

  python python/measure_sparsity.py --name baseline --ckpt models/resnet18_tv_in1k.safetensors
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import imagenet_val, loader  # noqa: E402
from resnet_fat import load_pretrained  # noqa: E402
from sparsity import measure  # noqa: E402

HIST_SITE = "layer2.0.act1"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nimg", type=int, default=5000)
    ap.add_argument("--out", default="results/sparsity")
    a = ap.parse_args()
    m = load_pretrained(a.ckpt).cuda().eval()
    ds = imagenet_val()
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(1))[: a.nimg].tolist()
    dl = loader(torch.utils.data.Subset(ds, idx), 250, workers=16)
    conv, site = measure(m, dl, max_batches=10 ** 9)
    # histogram of one site's activations (first batch)
    x, _ = next(iter(dl))
    m.set_record(True)
    m(x.cuda())
    v = dict(m.act_sites())[HIST_SITE].last.float().flatten().cpu().numpy()
    m.set_record(False)
    hist, edges = np.histogram(v, bins=120, range=(0, max(float(v.max()), 1e-6)))
    os.makedirs(a.out, exist_ok=True)
    r = {"name": a.name, "ckpt": a.ckpt, "nimg": a.nimg, "conv_input_sparsity": conv, "site_sparsity": site,
         "thresholds": m.thresholds(), "hist_site": HIST_SITE, "hist": hist.tolist(), "hist_edges": edges.tolist()}
    json.dump(r, open(os.path.join(a.out, f"fp32_{a.name}.json"), "w"), indent=1)
    s19 = [v for k, v in conv.items() if k not in ("conv1", "fc")]
    print(f"{a.name}: mean conv-input sparsity (19 convs) {np.mean(s19):.4f}")


if __name__ == "__main__":
    main()
