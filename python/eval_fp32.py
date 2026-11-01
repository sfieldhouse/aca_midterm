"""FP32 top-1/top-5 of a ResNet-18 checkpoint on ImageNet val (or minival)."""
import argparse
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import evaluate, imagenet_minival, imagenet_val, loader  # noqa: E402
from resnet_fat import load_pretrained  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="val", choices=["val", "minival"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = load_pretrained(a.ckpt).cuda()
    ds = imagenet_val() if a.split == "val" else imagenet_minival()
    t1, t5, loss = evaluate(m, loader(ds, 250, workers=32))
    r = {"ckpt": a.ckpt, "split": a.split, "top1": t1, "top5": t5, "loss": loss}
    print(json.dumps(r))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
