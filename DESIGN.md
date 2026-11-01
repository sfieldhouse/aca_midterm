# Design: Xqnn — packed-INT8 and activation-sparse ISA extensions for CNN inference on RV64IMC

This document defines the custom instructions, explains why they look the way
they do, and describes the bare-metal kernels and the quantization scheme that
use them. Measured results are in [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Baseline machine and workload

* **ISA**: RV64IMC (+Zicsr for `minstret`), bare-metal, M-mode, no FP, no V.
* **Simulator**: Spike (riscv-isa-sim, pinned submodule) with the Xqnn extension
  compiled in (`--extension=xqnn`). We report **retired instructions**
  (`minstret`, and Spike's PC histogram for instruction mix) — an ISA-level
  metric, not cycles.
* **Workload**: ResNet-18 (ImageNet, 1.81 GMAC/inference), W8A8, two models:
  * *normal* — the pretrained torchvision weights (from Hugging Face
    `timm/resnet18.tv_in1k`), post-training quantized;
  * *high-sparsity* — the same network fine-tuned with square-Hoyer activation
    regularization + FATReLU thresholds (Kurtz et al., ICML 2020), then
    quantized identically.

## 2. Quantization scheme (shared by Python reference and C)

| Tensor | Type | Zero point | Scale |
|---|---|---|---|
| conv / FC weights | int8, symmetric, per-output-channel (FC: per-tensor) | 0 | `max|w|/127` |
| bias | int32 | 0 | `s_in · s_w[c]` |
| image input | int8 | 0 | MSE-calibrated |
| post-(FAT)ReLU activations | uint8 | 0 | MSE-calibrated, per tensor |
| pre-add conv2 / downsample outputs | int8 | 0 | MSE-calibrated, per tensor |

* BatchNorm is folded into the convolutions.
* **Zero point 0 is deliberate.** A real 0 must be an integer 0, otherwise
  activation sparsity cannot be exploited. Post-ReLU tensors are non-negative,
  so they are stored as uint8 with the full 8-bit range (as in x86 VNNI).
* Requantization: `y = sat(((acc·M + 2^(sh−1)) >> sh) + zp)`, with
  `M ∈ [2^30, 2^31)` and `1 ≤ sh ≤ 63`. Rounding is round-half-up; the shift is
  arithmetic.
* FATReLU in the integer domain: `y < T_q → 0`, where `T_q = round(T / s_out)`.
  This is fused into the requantization of every ReLU site.
* Residual add: `v = y2·A + x·B` (A and B are integers with 16 fractional bits),
  then requantized with `M = 1, sh = 16`, ReLU/FATReLU, to uint8.
* Avgpool: the 7×7 sum is requantized (the 1/49 is folded into M). FC produces
  int32 logits, and argmax is taken directly.
* Activation scales are calibrated with MSE-optimal clipping on 1024 train
  images. For the high-sparsity model they are the fixed scales used during its
  quantization-aware fine-tuning (`python/qat.py`, `quantize.py --scales`).
* The Python integer reference (`python/quantize.py:QRunner`) is **bit-exact**
  with the C code. Every Spike run checks all 1000 logits of every image against
  it (`golden.bin`).

## 3. The Xqnn extension

All instructions use the **custom-0** major opcode (`0001011`), R-type,
`funct7 = 0000000`, and are selected by `funct3`. They are emitted from C with
`.insn r CUSTOM_0, f3, 0, rd, rs1, rs2`, so no binutils patch is needed.

```
 31      25 24  20 19  15 14  12 11   7 6      0
+----------+------+------+------+------+--------+
| 0000000  | rs2  | rs1  |funct3|  rd  | 0001011|
+----------+------+------+------+------+--------+
```

| funct3 | Mnemonic | Semantics |
|---|---|---|
| 000 | `xqmac8 rd, rs1, rs2` | `rd ← rd + Σ_{i=0..7} s8(rs1[8i+7:8i]) · s8(rs2[8i+7:8i])` (64-bit wrap) |
| 001 | `xqmac8u rd, rs1, rs2` | as above, with **unsigned** activations: `u8(rs1.b[i]) · s8(rs2.b[i])` |
| 010 | `xrequant rd, rs1, rs2` | `rd ← requant(acc = rs1 (int64), params = rs2)`, see below |
| 011 | `xsqmac8 g, rs1, rs2` | `ACC[g][i] ← ACC[g][i] + u8(rs1[7:0]) · s8(rs2.b[i])`, i = 0..7 (32-bit wrap); `g` = the rd field; no GPR is written |
| 100 | `xsaccrd rd, rs1` | `rd ← sext(ACC[x[rs1]]); ACC[x[rs1]] ← 0` (read-and-clear, flat index `8g+i`) |
| 101 | `xsaccclr` | `ACC[*] ← 0` |

**`xrequant` parameter word (rs2):**

| bits | field | meaning |
|---|---|---|
| 31:0 | M | unsigned multiplier |
| 37:32 | sh | right shift 0..63 (rounding term `2^(sh−1)` when `sh > 0`) |
| 47:40 | zp | signed output zero point |
| 55:48 | T | unsigned FATReLU threshold (mode 1 only) |
| 57:56 | mode | 0: saturate to int8; 1: saturate to uint8 (ReLU), then `y < T → 0` |

The product `acc·M` is computed exactly (up to 96 bits). Our kernels keep
`|acc| < 2^31` and `M < 2^31`, so the software fallback's 64-bit arithmetic
matches it exactly.

**Accumulator file `ACC`:** NGRP × 8 signed 32-bit accumulators. NGRP = 8
(64 accumulators, 2 Kbit) by default. NGRP is configurable through the
`XQNN_NGRP` environment variable (Spike) and the `NGRP` make variable (C), for
the design-space sweep (16 to 128 accumulators). A group or index outside the
file raises illegal-instruction. 32 bits are enough:
`max |acc| ≤ 4608 · 255 · 128 < 2^28`.

### 3.1 Why these instructions

**`xqmac8` / `xqmac8u` (the required packed MAC).**
* Profiling B1 (optimized scalar) shows the conv inner loop needs about 3
  instructions per MAC (`lbu`, `lb`, `mul`, `add` after register blocking).
  Multiplies and byte loads dominate the instruction mix.
* A 64-bit register holds 8 INT8 values, so one instruction can do 8 MACs and
  replace about 8 loads, 8 multiplies and 8 adds.
* The unsigned-activation variant exists because every conv input after the stem
  is post-ReLU uint8.
* `rd` is both source and destination, as in the P-extension's
  `smaqa`/`maddr32`. This costs a third register-file read port, or a 2-cycle
  issue, in hardware.

**`xrequant` (the required requantization instruction).**
* In software, requantization is a 64-bit `mul`, rounding `add`, `sra`, `add zp`,
  two clamps, the FATReLU compare, and the loads of M and sh: about 10–14
  instructions with 3 branches per output.
* Once xqmac8 shrinks the MAC loop, this becomes visible, especially in the
  sparse kernel, where there are far fewer MAC instructions per output.
* One instruction plus one load of a precomputed per-channel parameter word
  replaces all of it.
* Fusing the FATReLU threshold costs one 8-bit comparator.

**`xsqmac8` + `ACC` (the sparse instruction).**
* Kurtz et al. exploit activation sparsity with a sparse-input convolution
  (paper Alg. 1/2). Runtime-compressed nonzero activations are each broadcast
  and multiply-accumulated against a vector of output-channel weights
  (`vbroadcast` + `vfmadd` on AVX-512, with the output vectors held in zmm
  registers).
* A dot-product instruction like `xqmac8` cannot exploit unstructured zeros.
  - At 67% sparsity, only 0.67⁸ ≈ 4% of 8-element groups are all zero.
  - Compacting the nonzeros needs an 8-way gather of the matching weights.
* The **outer-product** form needs no gather: one nonzero activation × 8
  contiguous weights `W[tap][ic][oc..oc+7]` (one `ld`) gives 8 MACs into 8
  output-channel accumulators. Zero activations are simply never issued.
* The 8 × 32-bit results do not fit in a 64-bit GPR, so `xsqmac8` accumulates
  into a small **internal accumulator file**, which plays the role of AVX-512's
  zmm accumulators (cf. Arm SME's ZA array, Intel AMX tiles).
* The 8 byte-multipliers can be shared with xqmac8, with 8 accumulator adders
  instead of an adder tree.
* The CSR entry format (`(ic·OC) << 8 | value`, see §4.3) lets `rs1` be the raw
  entry: `xsqmac8` uses only bits 7:0.

**The register-only alternative was rejected.** With accumulators kept in GPRs
as 2 × 32-bit, each instruction does only 2 MACs.

* Per nonzero for 32 output channels: 5 + 4 × (1 `ld` + 4 MAC ops) = 25
  instructions per 32 MACs. That is about 1.3 useful MACs per instruction,
  roughly 3.9 dense-equivalent MACs per instruction at 33% density.
* That is no better than dense xqmac8 (about 3.5–4).
* With the accumulator file: 5 + 8 × 2 = 21 instructions per 64 MACs, which is
  3.05 useful MACs per instruction, or about 9 dense-equivalent at 33% density.
* Conclusion: sparse MACs only pay off with internal accumulators.

## 4. Kernels (src/nn/kernels.c)

All activations are NHWC with a **1-pixel zero border**, so 3×3/pad-1 convs need
no bounds checks. Producers zero the border of their output buffer, which costs
<0.1% of instructions. The stem input is a pre-padded `[230][232][4]` int8
image: pad 3, 4th channel zero, and the kernel width padded 7→8, so each kernel
row is 32 contiguous bytes.

### 4.1 B0 — naive dense

`for oy, ox, oc: for kh: for j < KW·IC: acc += in·w`, followed by software
requantization. This is plain C, compiled with `-O2`.

### 4.2 B1 / X1 / X2 — blocked dense

* Weights are laid out `[oc/4][kh][(kw·ic)/8][4][8]`, so a single pointer walks
  4 output channels in 8-byte chunks.
* For each kernel row, the receptive field is one contiguous byte run, so the
  reduction is a 1-D dot product of length `KW·IC`.
* B1 (scalar) uses a 2-pixel × 4-channel register block (8 accumulators), byte
  loads and `mul`.
* X1/X2 use a 4-pixel × 4-channel block (16 accumulators). Per 8-channel chunk
  that is 4 `ld` for activations, 4 `ld` for weights and 16 `xqmac8u`.
* X2 replaces software requantization with `ld` param word + `xrequant`.

### 4.3 S0 / S1 / S2 — sparse-input conv (Kurtz Alg. 1/2)

1. **Runtime CSR compression of the conv input**
   (`csr_compress`):
   * one row per (bordered) pixel;
   * entry = `(ic·OC) << 8 | value`, i.e. the byte offset of the weight row
     `W[tap][ic][·]` plus the activation;
   * 8 channels are tested at a time (`ld`, skip if zero), with an early exit on
     the remaining bytes;
   * the downsample conv reuses its block's CSR, since it has the same input and
     the same OC.
2. **Compute.** For every output pixel and every block of `8·NGRP` output
   channels, loop over the k×k taps and the nonzeros of each tap's input pixel:

   ```
   lwu e, (ep)             # entry
   srli t, e, 8
   add  p, wtap, t         # &W[tap][ic][ob]
   ld w0,0(p);  xsqmac8 acc0, e, w0
   ld w1,8(p);  xsqmac8 acc1, e, w1      ... ×NGRP
   ```

   Then read out with `xsaccrd`, add the bias, and requantize (`xrequant` in
   S2).
3. **S0** is the same algorithm with 16 GPR accumulators (`lb`, `mul`, `add`
   per output channel), which is the software-only version of the paper's
   kernel.
4. **S1** uses `xsqmac8` with software requantization. Its stem/FC are scalar,
   because S1 adds only the sparse instruction.
5. **S2** uses everything: `xsqmac8` sparse convs, `xqmac8` for the dense stem
   and FC, and `xrequant` everywhere. A per-layer **density dispatch**
   (`DISPATCH`, in permille) can route very dense layers to the dense xqmac8
   kernel. The density comes for free from the compression pass. The sweep in
   EXPERIMENTS.md §2.4 shows dispatch gains < 0.1%, so it is off by default.

### 4.4 Other operators

* **maxpool 3×3/s2:** scalar, identical in all configs.
* **residual add:** 2 multiplies plus requantization (`xrequant` in X2/S2).
* **avgpool:** sum plus requantization.
* **FC:** 1×4 output blocking, scalar or `xqmac8u`.

## 5. Verification

* `src/tests/test_xqnn.c`: about 105k directed and random checks of every
  instruction (saturation, rounding, shift 0/1/63, threshold edges, every
  group and lane, 32-bit wrap-around, read-clear, clear-all) against an
  independent base-ISA reference. Runs on Spike and natively.
* `src/tests/test_kernels.c`: every kernel configuration (and NGRP 2/4/8/16)
  against a reference convolution on random tensors with different shapes,
  strides, densities (0%–100%) and requant modes. Also checks CSR nnz.
* Full model: every config on Spike reproduces all 1000 logits of the Python
  integer reference for every embedded image (`RESULT PASS`). The native build
  checks ≥1000 val images.
