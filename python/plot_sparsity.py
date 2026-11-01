"""Per-layer activation-sparsity figures (paper Fig. 4(a) / Fig. 1 style).

Inputs (results/sparsity/fp32_<name>.json from measure_sparsity.py and
results/accuracy/int8_<model>_val.json from quantize.py):
  results/figures/sparsity_fp32_per_layer.png
  results/figures/sparsity_int8_per_layer.png
  results/figures/activation_hist.png
  results/sparsity/summary.csv
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_fat import conv_input_sites  # noqa: E402

# reference categorical palette (validated slots 1-3), light surface
C = ["#2a78d6", "#eb6834", "#1baf7a"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CONVS = [c for c, _ in conv_input_sites()][1:]          # 19 sparse-able convs
SHORT = [c.replace("layer", "L").replace(".downsample.0", ".ds").replace(".conv", ".c") for c in CONVS]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(series, title, path, ylabel="conv-input sparsity (% zeros)"):
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURF)
    style(ax)
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(CONVS))
    for i, (lab, vals) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, [100 * v for v in vals], w * 0.9, color=C[i],
               label=f"{lab} (mean {100 * np.mean(vals):.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=60, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=n, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp, fig = os.path.join(root, "results/sparsity"), os.path.join(root, "results/figures")
    os.makedirs(fig, exist_ok=True)
    names = [("baseline", "Baseline (pretrained)"), ("hoyer", "Hoyer regularized"),
             ("sparse", "Hoyer + FATReLU (final)")]
    fp = [(lab, json.load(open(os.path.join(sp, f"fp32_{n}.json")))) for n, lab in names
          if os.path.exists(os.path.join(sp, f"fp32_{n}.json"))]
    bars([(lab, [d["conv_input_sparsity"][c] for c in CONVS]) for lab, d in fp],
         "ResNet-18 FP32 activation sparsity at each conv input (ImageNet val, 5k images)",
         os.path.join(fig, "sparsity_fp32_per_layer.png"))
    i8 = []
    for n, lab in (("dense", "Normal model, INT8"), ("sparse", "High-sparsity model, INT8")):
        p = os.path.join(root, f"results/accuracy/int8_{n}_val.json")
        if os.path.exists(p):
            d = json.load(open(p))
            i8.append((f"{lab} (top-1 {100 * d['int8_top1']:.2f}%)", [d["conv_input_sparsity_int8"][c] for c in CONVS]))
    if i8:
        bars(i8, "W8A8 ResNet-18: zero activations at each conv input (ImageNet val, 50k images)",
             os.path.join(fig, "sparsity_int8_per_layer.png"))

    # activation histograms
    fig_, axs = plt.subplots(1, len(fp), figsize=(4 * len(fp), 3.2), facecolor=SURF, sharey=True)
    for i, (ax, (lab, d)) in enumerate(zip(np.atleast_1d(axs), fp)):
        style(ax)
        e = np.array(d["hist_edges"])
        h = np.array(d["hist"], dtype=float)
        h[h == 0] = np.nan                                  # empty bins: no bar
        ax.bar(e[:-1], h, width=np.diff(e), align="edge", color=C[i], log=True)
        ax.set_title(f"{lab}\n{d['hist_site']} output", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("activation value", color=INK2, fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel("count (log)", color=INK2)
    fig_.tight_layout()
    p = os.path.join(fig, "activation_hist.png")
    fig_.savefig(p, dpi=160)
    plt.close(fig_)
    print("wrote", p)

    # CSV summary
    with open(os.path.join(sp, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv"] + [f"fp32_{n}" for n, _ in names if os.path.exists(os.path.join(sp, f"fp32_{n}.json"))] +
                   [lab.split(",")[0] for lab, _ in i8])
        for j, c in enumerate(CONVS):
            w.writerow([c] + [f"{d['conv_input_sparsity'][c]:.4f}" for _, d in fp] + [f"{v[j]:.4f}" for _, v in i8])
        w.writerow(["MEAN"] + [f"{np.mean([d['conv_input_sparsity'][c] for c in CONVS]):.4f}" for _, d in fp] +
                   [f"{np.mean(v):.4f}" for _, v in i8])


if __name__ == "__main__":
    main()
