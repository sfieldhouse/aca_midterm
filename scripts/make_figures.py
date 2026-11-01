#!/usr/bin/env python3
"""ISA-level result figures + markdown tables from results/*.csv
(produced by scripts/run_benchmarks.py).

  results/figures/isa_total_instructions.png   dynamic instructions per inference
  results/figures/isa_speedup.png              reduction vs B0 / B1
  results/figures/isa_op_breakdown.png         per-operator instructions (stacked)
  results/figures/isa_instruction_mix.png      dynamic instruction mix
  results/figures/isa_layer_s2_vs_x2.png       per-conv speedup of sparse kernel vs dense xqmac8
  results/figures/isa_ngrp_sweep.png           accumulator-file size sweep
  results/summary.md                           tables used in README / EXPERIMENTS
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
CFGS = ["B0", "B1", "X1", "X2", "S0", "S1", "S2"]
CFG_LABEL = {"B0": "B0\nnaive", "B1": "B1\nSW-opt", "X1": "X1\n+xqmac8", "X2": "X2\n+xqmac8\n+xrequant",
             "S0": "S0\nsparse SW", "S1": "S1\nsparse\n+xsqmac8", "S2": "S2\nall custom"}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [("dense", "Normal model"), ("sparse", "High-sparsity model")]


def style(ax, ygrid=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=160, facecolor=SURF)
    plt.close(fig)
    print("wrote", p)


def read(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def op_group(op):
    if op == "conv1":
        return "stem conv (7x7, dense)"
    if op.startswith("csr:"):
        return "CSR compression"
    if op.startswith("add:"):
        return "residual add + requant"
    if op == "maxpool":
        return "maxpool"
    if op in ("avgpool", "fc"):
        return "avgpool + fc"
    return "3x3 / 1x1 convs"


GROUPS = ["3x3 / 1x1 convs", "stem conv (7x7, dense)", "CSR compression", "maxpool",
          "residual add + requant", "avgpool + fc"]


def main():
    os.makedirs(FIG, exist_ok=True)
    bench = [r for r in read("benchmarks.csv") if int(r["ngrp"]) == 8 and int(r["dispatch"]) == 1001]
    tot = {(r["model"], r["cfg"]): float(r["instret_mean"]) for r in bench}
    md = []

    # ---------------------------------------------------------------- totals
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    style(ax)
    x = np.arange(len(CFGS))
    for i, (m, lab) in enumerate(MODELS):
        v = [tot.get((m, c), np.nan) / 1e6 for c in CFGS]
        b = ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=C[i], label=lab)
        for rect, val in zip(b, v):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2, val * 1.08, f"{val:,.0f}", ha="center",
                        va="bottom", fontsize=7, color=INK2, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(100, 60000)
    ax.set_xticks(x)
    ax.set_xticklabels([CFG_LABEL[c] for c in CFGS], fontsize=8)
    ax.set_ylabel("dynamic instructions / inference (M, log)", color=INK2)
    ax.set_title("ResNet-18 W8A8 on RV64IMC (Spike): retired instructions per image", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "isa_total_instructions.png")

    # ---------------------------------------------------------------- speedups
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURF)
    for k, (ax, base) in enumerate(zip(axs, ("B0", "B1"))):
        style(ax)
        for i, (m, lab) in enumerate(MODELS):
            v = [tot.get((m, base), np.nan) / tot.get((m, c), np.nan) for c in CFGS]
            b = ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=C[i], label=lab)
            for rect, val in zip(b, v):
                if not np.isnan(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.1f}x", ha="center", va="bottom",
                            fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([c for c in CFGS], fontsize=8)
        ax.set_title(f"instruction reduction vs {base}", loc="left", color=INK, fontsize=10)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, "isa_speedup.png")

    # ---------------------------------------------------------------- table
    md.append("### Dynamic instructions per inference (mean over images)\n")
    md.append("| Config | Description | Normal model | High-sparsity model | vs B0 (normal / sparse) | vs X2 (normal / sparse) | .text bytes |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    desc = {r["cfg"]: r["desc"] for r in bench}
    size = {r["cfg"]: r["text_bytes"] for r in bench}
    for c in CFGS:
        d, s = tot.get(("dense", c)), tot.get(("sparse", c))
        f = lambda v: f"{v / 1e6:,.1f} M" if v else "-"
        r0 = f"{tot[('dense', 'B0')] / d:.1f}x / {tot[('sparse', 'B0')] / s:.1f}x" if d and s and ("dense", "B0") in tot and ("sparse", "B0") in tot else "-"
        rx = f"{tot[('dense', 'X2')] / d:.2f}x / {tot[('sparse', 'X2')] / s:.2f}x" if d and s and ("dense", "X2") in tot and ("sparse", "X2") in tot else "-"
        md.append(f"| {c} | {desc.get(c, '')} | {f(d)} | {f(s)} | {r0} | {rx} | {size.get(c, '')} |")
    md.append("")
    mpi = {(r["model"], r["cfg"]): r["mac_per_instr"] for r in bench}
    md.append("MAC / instruction (real MACs = %s per image): " % (bench[0]["macs"] if bench else "?") +
              ", ".join(f"{c} {mpi.get(('dense', c), '-')} / {mpi.get(('sparse', c), '-')}" for c in CFGS) + "\n")

    # ---------------------------------------------------------------- op breakdown
    ops = read("ops.csv")
    if ops:
        agg = defaultdict(float)
        for r in ops:
            agg[(r["model"], r["cfg"], op_group(r["op"]))] += float(r["instret_per_img"])
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF, sharey=True)
        for ax, (m, lab) in zip(axs, MODELS):
            style(ax)
            bottom = np.zeros(len(CFGS))
            for gi, g in enumerate(GROUPS):
                v = np.array([agg.get((m, c, g), 0) / 1e6 for c in CFGS])
                ax.bar(x, v, 0.6, bottom=bottom, color=C[gi], label=g, edgecolor=SURF, linewidth=1)
                bottom += v
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(CFGS, fontsize=8)
            ax.set_title(lab, loc="left", color=INK, fontsize=10)
        axs[0].set_ylabel("instructions / image (M, log)", color=INK2)
        axs[1].legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper right")
        save(fig, "isa_op_breakdown.png")
        md.append("### Per-operator instructions per image (M), high-sparsity model\n")
        md.append("| Operator group | " + " | ".join(CFGS) + " |")
        md.append("|---|" + "---:|" * len(CFGS))
        for g in GROUPS:
            md.append(f"| {g} | " + " | ".join(f"{agg.get(('sparse', c, g), 0) / 1e6:,.1f}" for c in CFGS) + " |")
        md.append("")

        # per-layer sparse-vs-dense speedup
        per = {(r["model"], r["cfg"], r["op"]): float(r["instret_per_img"]) for r in ops}
        convs = [r["op"] for r in ops if r["model"] == "dense" and r["cfg"] == "X2" and r["op"].startswith("layer")]
        if convs and any(k[1] == "S2" for k in per):
            fig, ax = plt.subplots(figsize=(12, 3.8), facecolor=SURF)
            style(ax)
            xx = np.arange(len(convs))
            for i, (m, lab) in enumerate(MODELS):
                v = [per.get((m, "X2", o), np.nan) / (per.get((m, "S2", o), np.nan) + per.get((m, "S2", "csr:" + o), 0))
                     for o in convs]
                ax.bar(xx + (i - 0.5) * 0.38, v, 0.36, color=C[i], label=lab)
            ax.axhline(1, color=INK2, lw=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels([o.replace("layer", "L").replace(".downsample.0", ".ds").replace(".conv", ".c") for o in convs],
                               rotation=60, ha="right")
            ax.set_ylabel("X2 instr / S2 instr (incl. CSR)", color=INK2)
            ax.set_title("Per-layer gain of the sparse custom kernel over dense xqmac8 (both with xrequant)",
                         loc="left", color=INK, fontsize=10)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
            save(fig, "isa_layer_s2_vs_x2.png")

    # ---------------------------------------------------------------- mix
    mix = read("mix.csv")
    if mix:
        cls = ["load", "store", "alu", "mul", "branch", "custom"]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURF, sharey=True)
        mx = {(r["model"], r["cfg"]): r for r in mix if int(r["ngrp"]) == 8 and int(r["dispatch"]) == 1001}
        for ax, (m, lab) in zip(axs, MODELS):
            style(ax)
            bottom = np.zeros(len(CFGS))
            for ci, k in enumerate(cls):
                v = np.array([100 * float(mx[(m, c)][k]) / float(mx[(m, c)]["total"]) if (m, c) in mx else 0 for c in CFGS])
                ax.bar(x, v, 0.6, bottom=bottom, color=C[ci], label=k, edgecolor=SURF, linewidth=1)
                bottom += v
            ax.set_xticks(x)
            ax.set_xticklabels(CFGS, fontsize=8)
            ax.set_title(lab, loc="left", color=INK, fontsize=10)
        axs[0].set_ylabel("% of dynamic instructions", color=INK2)
        axs[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", bbox_to_anchor=(1.18, 1))
        save(fig, "isa_instruction_mix.png")
        md.append("### Dynamic instruction mix (1 image, spike -g PC histogram)\n")
        md.append("| Model | Config | total | load | store | ALU | mul | branch | custom | loads+stores / MAC | custom breakdown |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for m, _ in MODELS:
            for c in CFGS:
                r = mx.get((m, c))
                if not r:
                    continue
                t = float(r["total"])
                pc = lambda k: f"{100 * float(r[k]) / t:.1f}%"
                ls = (float(r["load"]) + float(r["store"])) / float(r["macs"])
                cb = ", ".join(f"{k[5:]} {float(r[k]) / 1e6:.1f}M" for k in r if k.startswith("insn_") and float(r[k]) > 0)
                md.append(f"| {m} | {c} | {t / 1e6:,.0f}M | {pc('load')} | {pc('store')} | {pc('alu')} | {pc('mul')} | "
                          f"{pc('branch')} | {pc('custom')} | {ls:.3f} | {cb} |")
        md.append("")

    # ---------------------------------------------------------------- sweep
    sw = [r for r in read("benchmarks.csv") if r["cfg"] in ("S1", "S2") and int(r["dispatch"]) == 1001]
    if len({r["ngrp"] for r in sw}) > 1:
        fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURF)
        style(ax)
        md.append("### Accumulator-file size sweep (S2, instructions per image, M)\n")
        md.append("| accumulators | " + " | ".join(lab for _, lab in MODELS) + " |")
        md.append("|---:|" + "---:|" * len(MODELS))
        rows = defaultdict(dict)
        for i, (m, lab) in enumerate(MODELS):
            pts = sorted((int(r["ngrp"]) * 8, float(r["instret_mean"]) / 1e6) for r in sw if r["model"] == m and r["cfg"] == "S2")
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C[i], lw=2, ms=6, label=lab)
                for a_, v in pts:
                    rows[a_][m] = v
        for a_ in sorted(rows):
            md.append(f"| {a_} | " + " | ".join(f"{rows[a_].get(m, float('nan')):,.1f}" for m, _ in MODELS) + " |")
        md.append("")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128])
        ax.set_xticklabels(["16", "32", "64", "128"])
        ax.set_xlabel("internal accumulators (8 x groups)", color=INK2)
        ax.set_ylabel("instructions / image (M)", color=INK2)
        ax.set_title("S2: accumulator-file size vs instructions", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        save(fig, "isa_ngrp_sweep.png")
    disp = [r for r in read("benchmarks.csv") if r["cfg"] == "S2" and int(r["ngrp"]) == 8]
    if len({r["dispatch"] for r in disp}) > 1:
        md.append("### S2 dense-dispatch threshold sweep (layers with input density > threshold use dense xqmac8)\n")
        md.append("| threshold (permille) | " + " | ".join(lab for _, lab in MODELS) + " |")
        md.append("|---:|" + "---:|" * len(MODELS))
        for d in sorted({int(r["dispatch"]) for r in disp}):
            vals = {r["model"]: float(r["instret_mean"]) / 1e6 for r in disp if int(r["dispatch"]) == d}
            md.append(f"| {d if d <= 1000 else 'off'} | " + " | ".join(f"{vals.get(m, float('nan')):,.1f}" for m, _ in MODELS) + " |")
        md.append("")

    # ---------------------------------------------------------------- runtime density
    sp = read("sparsity_runtime.csv")
    if sp:
        md.append("### Conv-input density observed at runtime in Spike (S2, mean over images)\n")
        md.append("| conv | normal | high-sparsity |")
        md.append("|---|---:|---:|")
        dd = {(r["model"], r["conv"]): float(r["density"]) for r in sp if r["cfg"] == "S2"}
        for cv in dict.fromkeys(r["conv"] for r in sp):
            md.append(f"| {cv} | {dd.get(('dense', cv), float('nan')):.3f} | {dd.get(('sparse', cv), float('nan')):.3f} |")
        md.append("")
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(RES, "summary.md"))


if __name__ == "__main__":
    main()
