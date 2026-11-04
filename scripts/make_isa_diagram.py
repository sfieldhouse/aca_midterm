#!/usr/bin/env python3
"""One-page diagram of xqmac8(u) vs xsqmac8 -> docs/xqmac8_xsqmac8.pdf.

All values are the ones from the Spike traces of the two small demos:
xqmac8u a0, a5, a4 (a5 = 0x0164210007ff000c, a4 = 0x0904ff7f800205fd) and the
second xsqmac8 acc0 (rs1 = 0x6002, rs2 = 0xf9fafbfcfdfeff00).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_T, ORANGE_T, GREEN_T = "#e3eefb", "#fdeae1", "#def5ec"
INK, INK2, LINE, FAINT = "#0b0b0b", "#52514e", "#8a8984", "#f1f0ec"
MONO = "DejaVu Sans Mono"

CW, CH = 0.92, 0.62                 # cell size
LX = 1.75                           # x of the first cell, relative to the panel


def cell(ax, x, y, txt, fc, ec, w=CW, h=CH, color=INK, size=9.5, font=MONO):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x + w / 2, y, str(txt).replace("-", "−"), ha="center", va="center", color=color, fontsize=size, family=font, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=LINE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=1,
                                 color=color, shrinkA=0, shrinkB=0, zorder=2))


def mul(ax, x, y):
    ax.add_patch(Circle((x, y), 0.2, fc="white", ec=INK2, lw=1, zorder=3))
    ax.text(x, y - 0.01, "×", ha="center", va="center", fontsize=11, color=INK, zorder=4)


def label(ax, x, y, main, sub=None):
    ax.text(x, y + (0.13 if sub else 0), main, ha="right", va="center", fontsize=9.5, family=MONO, color=INK)
    if sub:
        ax.text(x, y - 0.17, sub, ha="right", va="center", fontsize=7.5, color=INK2)


def head(ax, x0, title, syntax, formula):
    ax.text(x0, 10.05, title, fontsize=13, weight="bold", color=INK)
    ax.text(x0, 9.6, syntax, fontsize=10, family=MONO, color=INK)
    ax.text(x0, 9.2, formula, fontsize=9, family=MONO, color=INK2)


def xqmac8(ax, x0):
    a = [12, 0, 255, 7, 0, 33, 100, 1]
    w = [-3, 5, 2, -128, 127, -1, 4, 9]
    p = [i * j for i, j in zip(a, w)]
    assert sum(p) == -46
    head(ax, x0, "xqmac8 / xqmac8u: dense dot product", "xqmac8u rd, rs1, rs2", "rd += Σ(i=0..7) u8(rs1.b[i]) × s8(rs2.b[i])")
    xs = [x0 + LX + i * CW for i in range(8)]
    label(ax, x0 + LX - 0.12, 8.35, "rs1", "8 activations")
    label(ax, x0 + LX - 0.12, 7.73, "rs2", "8 weights")
    for i, x in enumerate(xs):
        cx = x + CW / 2
        z = a[i] == 0
        cell(ax, x, 8.35, a[i], FAINT if z else BLUE_T, BLUE, color=INK2 if z else INK)
        cell(ax, x, 7.73, w[i], ORANGE_T, ORANGE)
        ax.text(cx, 8.78, f"b{i}", ha="center", fontsize=7.5, color=INK2)
        arrow(ax, cx, 7.42, cx, 6.72)
        mul(ax, cx, 6.5)
        arrow(ax, cx, 6.3, cx, 5.86)
        cell(ax, x, 5.55, p[i], FAINT if z else "white", LINE, color=INK2 if z else INK)
        arrow(ax, cx, 5.24, cx, 4.5)
    label(ax, x0 + LX - 0.12, 5.55, "products")
    ax.add_patch(Rectangle((xs[0], 3.9), 8 * CW, 0.6, fc="white", ec=INK2, lw=1, zorder=3))
    ax.text(xs[0] + 4 * CW, 4.2, "Σ of all 8 products = −46", ha="center", va="center", fontsize=9.5, family=MONO, color=INK, zorder=4)
    label(ax, x0 + LX - 0.12, 4.2, "adder")
    mid = xs[0] + 4 * CW
    arrow(ax, mid, 3.9, mid, 3.3)
    cell(ax, xs[0], 3.0, "rd: 0 + (−46) = −46", GREEN_T, GREEN, w=8 * CW, h=0.6)
    label(ax, x0 + LX - 0.12, 3.0, "rd", "normal register")
    ax.text(xs[0] + 4 * CW, 2.0, "8 inputs → 1 output", ha="center", fontsize=11, weight="bold", color=INK)
    ax.text(xs[0] + 4 * CW, 1.58, "Every activation is multiplied, including zeros (grey).",
            ha="center", fontsize=9, color=INK2)
    ax.text(xs[0] + 4 * CW, 1.22, "Uses only x0–x31. xqmac8 is the same with signed activations.",
            ha="center", fontsize=9, color=INK2)


def xsqmac8(ax, x0):
    a = 2
    w = [0, -1, -2, -3, -4, -5, -6, -7]
    old = [5, 10, 15, 20, 25, 30, 35, 40]
    p = [a * j for j in w]
    new = [o + q for o, q in zip(old, p)]
    assert new == [5, 8, 11, 14, 17, 20, 23, 26]
    head(ax, x0, "xsqmac8: sparse outer product", "xsqmac8 g, rs1, rs2",
         "ACC[g][i] += u8(rs1[7:0]) × s8(rs2.b[i]),  i = 0..7")
    xs = [x0 + LX + i * CW for i in range(8)]
    label(ax, x0 + LX - 0.12, 7.73, "rs2", "8 weights")
    # one nonzero activation, broadcast along the bus through every multiplier
    bx = x0 + 0.1
    cell(ax, bx, 6.5, a, BLUE_T, BLUE, w=0.8, h=0.6)
    ax.text(bx + 0.4, 7.0, "rs1[7:0]", ha="center", fontsize=8.5, family=MONO, color=INK)
    ax.text(bx + 0.4, 6.0, "1 nonzero", ha="center", fontsize=7.5, color=INK2)
    ax.text(bx + 0.4, 5.8, "activation", ha="center", fontsize=7.5, color=INK2)
    ax.plot([bx + 0.8, xs[-1] + CW / 2], [6.5, 6.5], color=BLUE, lw=1.6, zorder=1)
    for i, x in enumerate(xs):
        cx = x + CW / 2
        cell(ax, x, 7.73, w[i], ORANGE_T, ORANGE)
        ax.text(cx, 8.16, f"b{i}", ha="center", fontsize=7.5, color=INK2)
        arrow(ax, cx, 7.42, cx, 6.72)
        mul(ax, cx, 6.5)
        arrow(ax, cx, 6.3, cx, 5.86)
        cell(ax, x, 5.55, p[i], "white", LINE)
        arrow(ax, cx, 5.24, cx, 4.84)
        ax.text(cx + 0.09, 5.03, "+", ha="left", va="center", fontsize=9, color=INK2)
        cell(ax, x, 4.53, old[i], "white", LINE)
        arrow(ax, cx, 4.22, cx, 3.82)
        cell(ax, x, 3.51, new[i], GREEN_T, GREEN)
    label(ax, x0 + LX - 0.12, 5.55, "products")
    label(ax, x0 + LX - 0.22, 4.53, "ACC[g]", "before")
    label(ax, x0 + LX - 0.22, 3.51, "ACC[g]", "after")
    # the rest of the accumulator file
    cell(ax, xs[0], 2.88, "other groups: unchanged", FAINT, LINE, w=8 * CW, h=0.34, color=INK2, size=8, font="DejaVu Sans")
    ax.add_patch(Rectangle((xs[0] - 0.09, 2.62), 8 * CW + 0.18, 1.3, fill=False, ec=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2))
    ax.text(xs[0] + 8 * CW + 0.09, 2.47, "internal accumulator file: 8 groups × 8 int32", ha="right", va="center",
            fontsize=7.5, color=INK2)
    ax.text(xs[0] + 4 * CW, 2.0, "1 input → 8 outputs", ha="center", fontsize=11, weight="bold", color=INK)
    ax.text(xs[0] + 4 * CW, 1.58, "Zero activations are not in the list, so they cost nothing.",
            ha="center", fontsize=9, color=INK2)
    ax.text(xs[0] + 4 * CW, 1.22, "No register is written. xsaccrd reads each result back.",
            ha="center", fontsize=9, color=INK2)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig, ax = plt.subplots(figsize=(11.69, 6.0), facecolor="white")
    ax.set_xlim(-0.1, 20.3)
    ax.set_ylim(0.85, 10.35)
    ax.axis("off")
    xqmac8(ax, 0)
    ax.plot([9.85, 9.85], [1.1, 10.2], color="#e4e3df", lw=1)
    xsqmac8(ax, 10.2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = os.path.join(root, "docs", "xqmac8_xsqmac8.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
