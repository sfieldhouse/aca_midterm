# Experiments

Measurements were taken on Spike (pinned riscv-isa-sim + Xqnn) for the ISA-level
results, and on ImageNet-1k val (50,000 images) for accuracy. All numbers in
this file come from the logs and CSVs under `results/` and the training logs in
`models/*/train.out`. The training logs are not committed because `models/` is
large; regenerate them with `scripts/train_sparse.sh`.

## 1. Inducing activation sparsity (Kurtz et al. 2020)

### 1.1 Method as run

| Phase | What | Settings |
|---|---|---|
| 0 | Pretrained ResNet-18 from Hugging Face `timm/resnet18.tv_in1k` (torchvision weights) | — |
| 1 | Fine-tune with **square-Hoyer** regularization on all 17 activation sites: `L = CE + λ Σ H(a)`, `H(v) = ‖v‖₁² / ‖v‖₂²` over each site's per-GPU batch tensor | λ = 3e-9 (batch 256/GPU, so ≈ 2–3e-8 per site at the paper's scale), SGD + Nesterov, LR 0.01 cosine, 10 epochs, 3 GPUs |
| 2 | **FATReLU sensitivity analysis** (paper §3): for each site, sweep the threshold that zeroes an extra 5…70% of positive activations; keep the largest whose minival CE rise ≤ τ | τ = 0.005 CE per site on 5k held-out train images; `layer4.1.act2` (feeds only avgpool/FC) is excluded |
| 3 | **Recovery fine-tune with FATReLU thresholds fixed** | thresholds = 0.85 × the τ-thresholds, LR 0.003 cosine, 6 epochs. Self-distillation from the phase-1 model (same architecture, no thresholds; α = 0.7, T = 2) and quantization-aware training on exactly the W8A8 integer grid of the bare-metal code |

**"minival"** is 10 images per class held out of ImageNet *train*. It is used
for every tuning decision, so the official val set is only used for reporting.

### 1.2 What did not work (kept for honesty and reproducibility)

| Attempt | Result on val | Why abandoned |
|---|---|---|
| τ = 0.02, Hoyer λ = 3e-9 kept in phase 3, LR 0.01 | 82.7% sparsity after 3 epochs, minival −9.5 points | far too aggressive; stopped after 3 epochs |
| τ = 0.005, Hoyer λ = 2e-9 kept in phase 3, LR 0.005, 10 epochs | FP32 69.06%, INT8 (PTQ) 68.28%, 72.2% INT8 sparsity | below both gates; keeping Hoyer pressure on during recovery costs about 1 point |
| + 8 epochs of self-distillation (Hoyer off) | FP32 69.70% | 0.06 below the FP32 gate |
| + 5 epochs of QAT + distillation | FP32 69.22%, INT8 69.29%, 69.9% INT8 sparsity | 0.29 below the INT8 gate: the high-sparsity network is noticeably more quantization-sensitive (its PTQ loss is 0.78 vs 0.18 for the dense model) |

We drew two lessons from these attempts.
1. Hoyer pressure should be switched off once the thresholds are applied.
2. For a model this sparse, the final recovery has to be quantization-aware.
   Lowering the thresholds on an already-adapted model does not restore
   accuracy (measured: 69.0–69.6% for 0×–1× threshold scale).

The final recipe (phase 3 above) therefore restarts from the phase-1 Hoyer
model.

### 1.3 Final models (ImageNet val, 50,000 images)

| Model | FP32 top-1 | INT8 (W8A8) top-1 | conv-input sparsity, FP32 (19 convs, 5k val) | conv-input sparsity, INT8 (19 convs, 50k val) | MAC-weighted INT8 density seen in Spike |
|---|---:|---:|---:|---:|---:|
| Normal (pretrained, PTQ) | **69.76%** | **69.58%** | 53.4% | 54.1% | 0.427 |
| Phase-1 Hoyer only (intermediate) | 70.12% | — | 58.6% | — | — |
| **High-sparsity (Hoyer + FATReLU, QAT)** | 69.55% ¹ | **69.67%** | **67.6%** | **66.5%** | **0.314** |

The paper reports 53% baseline and 67% "boosted Hoyer" average sparsity for
ResNet-18 (its Table 1). We reproduce both: 53.4% and 67.6%.

**Accuracy gate.**
* **INT8 gate: passes.** The deployed W8A8 high-sparsity model (69.67%) is
  *more* accurate than the W8A8 normal model (69.58%), and within 0.09 points of
  the original FP32 network (69.76%).
* ¹ **FP32 gate: misses by 0.21 points.** The final weights were trained with
  quantization in the loop (QAT), so running them in FP32 is not the mode they
  were optimized for: 69.55% vs 69.76%.
  - The best pure-FP32 checkpoint we produced reached 69.70% at 72% sparsity
    (§1.2), which is also just below the gate.
  - A strict FP32 ≥ 69.76% sparse checkpoint therefore needs one more
    non-QAT recovery run at these thresholds, about 1 GPU-hour. See *Open
    items*.

Per-layer figures:
* `results/figures/sparsity_fp32_per_layer.png`: baseline / Hoyer / final,
  in the style of paper Fig. 4(a).
* `results/figures/sparsity_int8_per_layer.png`: normal vs high-sparsity INT8.
* `results/figures/activation_hist.png`: paper Fig. 1 style histogram.

Numbers are in `results/sparsity/summary.csv` and `results/accuracy/*.json`.
The largest gains are at the block inputs (post-residual-add sites), where the
pretrained network is least sparse. For example, `layer1.1.conv1` goes from 19%
to 41% and `layer2.0.conv1` from 23% to 46%.

## 2. ISA-level evaluation on Spike

Setup:
* Every number is **retired instructions (`minstret`) per inference**, averaged
  over 8 ImageNet val images per configuration.
* Instruction counts of the dense configurations are input-independent to
  within 0.01%. The sparse configurations vary with the image: the std over
  8 images is 0.3–1.7% (`results/benchmarks.csv`).
* Every run checked all 1000 logits of every image against the Python integer
  reference (**all PASS**).
* Real MACs per image: 1.814 G.

### 2.1 Headline: instructions per inference

| Config | Kernels / instructions | Normal model | High-sparsity model | vs B0 (normal / sparse) | vs X2 (normal / sparse) |
|---|---|---:|---:|---:|---:|
| B0 | naive dense, RV64IMC | 13,350.7 M | 13,352.3 M | 1.0× / 1.0× | — |
| B1 | SW-optimized dense, RV64IMC | 5,428.6 M | 5,430.2 M | 2.5× / 2.5× | — |
| X1 | + `xqmac8` | 586.2 M | 587.8 M | 22.8× / 22.7× | — |
| X2 | + `xqmac8` + `xrequant` | 553.8 M | 553.6 M | 24.1× / 24.1× | 1.00× / 1.00× |
| S0 | sparse-input CSR conv, RV64IMC only | 2,907.0 M | 2,310.1 M | 4.6× / 5.8× | — |
| S1 | sparse conv + `xsqmac8` (stem/FC scalar) | 844.2 M | 785.2 M | 15.8× / 17.0× | — |
| **S2** | **`xsqmac8` sparse convs + `xqmac8` stem/FC + `xrequant`** | **340.6 M** | **279.9 M** | **39.2× / 47.7×** | **1.63× / 1.98×** |

MAC per instruction:

| Config | Normal model | High-sparsity model |
|---|---:|---:|
| B0 | 0.136 | 0.136 |
| B1 | 0.334 | 0.334 |
| X1 | 3.10 | 3.09 |
| X2 | 3.28 | 3.28 |
| S0 | 0.62 | 0.79 |
| S1 | 2.15 | 2.31 |
| S2 | 5.33 | **6.48** (dense-equivalent) |

Figures: `results/figures/isa_total_instructions.png`, `isa_speedup.png`.

The incremental story asked for by the course, one instruction on top of the
other:

| Step | Instructions | Gain |
|---|---|---|
| B1 → X1 (+`xqmac8`) | 5,429 M → 586 M | **9.3×** |
| X1 → X2 (+`xrequant`) | 586 M → 554 M | 1.06× (the remaining requantization overhead, see 2.3) |
| X2 → S2 (+`xsqmac8`, normal model) | 554 M → 341 M | 1.63× |
| X2 → S2 (+`xsqmac8`, high-sparsity model) | 554 M → 280 M | **1.98×** |

**Sparse instruction vs. no sparse instruction.**
* With the same CSR algorithm, S0 (software) → S1 (`xsqmac8`) is **2.9×** on
  the high-sparsity model and 3.4× on the normal model. S1 is limited by its
  scalar 7×7 stem: 539 M of its 785 M instructions.
* Sparsity alone, in software (B1 → S0), is worth 1.9× (normal model) or 2.4×
  (high-sparsity model).
* Kurtz-style sparsification plus `xsqmac8` turns into a further **1.22×** over
  the same instructions on the normal model (S2: 340.6 M → 279.9 M).

### 2.2 Per-layer and per-operator view

* `results/figures/isa_layer_s2_vs_x2.png`: the gain of the sparse kernel over
  dense xqmac8 per conv. It tracks 1/density:
  - about 0.9–1.3× on the early, dense layers (`layer1.*.conv1`,
    `layer2.0.conv1`/`ds`, density 0.57–0.80). On the normal model the 1×1
    downsample and `layer1.*.conv1` are slightly *slower* sparse (0.87–0.95×),
    which is what the density-dispatch option targets;
  - up to **8.6×** on `layer4.1.conv2` (density 0.11).
* Per-operator breakdown, in M instructions per image for the high-sparsity
  model (`results/figures/isa_op_breakdown.png`):

| Operator group | B0 | B1 | X1 | X2 | S0 | S1 | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3×3 / 1×1 convs | 11,980.1 | 4,864.1 | 495.4 | 476.7 | 1,731.1 | 206.2 | **189.9** |
| stem conv (7×7, dense, 3 ch) | 1,342.9 | 538.6 | 66.5 | 57.5 | 538.6 | 538.6 | **57.5** |
| CSR compression | — | — | — | — | 13.0 | 13.0 | 13.0 |
| maxpool | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 |
| residual add + requant | 16.9 | 16.9 | 16.9 | 10.5 | 16.9 | 16.9 | 10.5 |
| avgpool + FC | 3.8 | 1.9 | 0.4 | 0.4 | 1.9 | 1.9 | 0.4 |

### 2.3 Instruction mix, memory instructions, custom-instruction utilization

From Spike's `-g` PC histogram plus objdump, 1 image
(`results/mix.csv`, `results/figures/isa_instruction_mix.png`):

| Config (high-sparsity model) | load | store | ALU | mul | branch | custom | (loads+stores)/MAC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 28.4% | 0.1% | 43.1% | 14.1% | 14.3% | — | 2.10 |
| B1 | 26.5% | 0.2% | 37.7% | 34.7% | 0.9% | — | 0.80 |
| X2 | 29.6% | 1.4% | 21.9% | 0.3% | 3.8% | 43.0% | 0.096 |
| S0 | 30.0% | 0.6% | 37.6% | 29.4% | 2.4% | — | 0.40 |
| S2 | 34.2% | 2.8% | 24.3% | 0.6% | 7.3% | 30.8% | **0.060** |

Custom instructions in S2 (high-sparsity model), per image:
* `xsqmac8` 63.2 M (21.5% of all retired instructions);
* `xqmac8` 22.5 M (stem);
* `xrequant` 3.2 M;
* `xsaccrd` 1.7 M.

With the normal model there are 83.4 M `xsqmac8`: 25% fewer sparse MACs issue
on the high-sparsity model.

**Requantization.**
* X1 → X2 saves 32.4 M instructions over 3.2 M requantized outputs, i.e. about
  **12 → 2 instructions per output element** (load parameter word +
  `xrequant`).
* Its share of runtime grows as the MACs shrink: it is about 5.5% of X1, and
  without `xrequant` S2 would need about 32 M more instructions (+11%).

**Code size (`.text`, bytes):**

| Config | B0 | B1 | X1 | X2 | S0 | S1 | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `.text` | 9,372 | 9,770 | 11,282 | 8,140 | 10,244 | 10,046 | 8,978 |

`xrequant` shrinks the code because the inlined software requantization
sequences disappear. X1 grows because of the 4×4-blocked xqmac8 kernel and the
stem specialization.

### 2.4 Design-space sweeps (S2)

**Accumulator-file size** (`results/figures/isa_ngrp_sweep.png`):

| Accumulators | Normal model | High-sparsity model |
|---:|---:|---:|
| 16 (2 groups) | 511.4 M | 409.3 M |
| 32 (4 groups) | 397.5 M | 323.1 M |
| **64 (8 groups, chosen)** | **340.6 M** | **279.9 M** |
| 128 (16 groups) | 324.7 M | 268.2 M |

* The per-nonzero overhead (load the entry, shift, add address, loop) is
  amortized over 8·NGRP output channels.
* 64 → 128 accumulators buys only 4%, while doubling the 2 Kbit state. 64 is
  the knee.
* Even 16 accumulators (409 M) beat dense xqmac8 + requantization (554 M) on
  the high-sparsity model.

**Density dispatch** (route dense layers to the xqmac8 kernel):

| Threshold (density) | Normal model | High-sparsity model |
|---:|---:|---:|
| 0.5 | 361.1 M | 302.9 M |
| 0.7 | 343.2 M | 279.9 M |
| 0.8 | 340.4 M | 279.9 M |
| off | 340.6 M | 279.9 M |

The sparse kernel is at least as good as dense xqmac8 on almost every layer,
even at densities around 0.8, so dispatch brings < 0.1%. It is left off by
default.

**Register-only alternative (analytical, see DESIGN.md §3.1).**
* With 2 × 32-bit accumulators per GPR, the sparse MAC delivers about 1.3
  useful MACs per instruction, about 3.9–4.2 dense-equivalent at our 0.31
  MAC-weighted density.
* That is about the same as X2's 3.3. The internal accumulator file is what
  makes sparsity pay: S2 reaches 6.5.

## 3. Bottleneck analysis

**B1 (optimized RV64IMC).** About 3 instructions per MAC.
1. **Multiply + byte-load per MAC:** `mul` is 34.7% and loads are 26.5% of
   instructions. Each MAC needs `lb`/`lbu` + `mul` + `add`.
   → `xqmac8`: 9.3× fewer instructions.
2. **Loop / address overhead:** ALU is 37.7%. Blocking already reduced it.
   → Amortized by the 8-wide instruction and 4×4 blocking.
3. **Requantization:** about 12 instructions and 3 branches per output.
   → `xrequant`.

**X2 (after xqmac8 + xrequant).** 3.3 MAC per instruction.
1. **Memory instructions:** 29.6% loads, about 0.7 `ld` per `xqmac8`. Weights
   are re-streamed for every 4-pixel block.
2. **Zero activations still issue MACs:** 57% (normal) / 69% (high-sparsity)
   of the MAC-weighted xqmac8 work multiplies zeros.
   → `xsqmac8` + runtime CSR: 1.63× (normal) / 1.98× (high-sparsity).
3. **Loop/address arithmetic:** 22% ALU.

**S2 (all extensions, high-sparsity model, 280 M).** The next bottlenecks:
1. **The dense 7×7 stem** is now **20.5%** of all instructions (57.5 M). Its 3
   input channels map onto 8-lane xqmac8 with only 66% lane utilization (3/4
   channels × 7/8 taps), and the input image has no zeros to skip.
   Candidates:
   - a dedicated im2col/space-to-depth layout;
   - a 3-channel MAC variant.
2. **Weight loads in the sparse loop:** one `ld` per `xsqmac8`. Loads are 34%
   of S2. A **fused load + MAC** (`xsqmac8` with a memory operand, or a
   post-increment weight stream) would remove about 63 M instructions (about
   23%), which is exactly the "Fused Load+MAC" / "Post-increment" extension
   from the course list.
3. **Per-nonzero overhead + non-MAC operators:**
   - the CSR entry decode (`lwu`, `srli`, `add`) plus loop: 5 instructions per
     nonzero, about 40 M;
   - CSR compression: 13 M, 4.6%;
   - residual add: 10.5 M;
   - maxpool: 8.4 M.

   A hardware loop over the CSR row, and fusing compression into the producer's
   requantization epilogue (the producer already knows which outputs are zero),
   are the natural next steps.

## 4. Course quantitative targets

| Target | Result |
|---|---|
| ≥ 2× dynamic-instruction reduction for a major GEMM/conv kernel | conv kernels B1 → X2: **10.2×** (4,864 M → 477 M); 3×3/1×1 convs X2 → S2: **2.5×** on the high-sparsity model (2.4× including CSR compression) |
| Xqmac passes directed + randomized verification | 105,305 checks, 0 fails, Spike + native |
| Quantized kernel output matches reference | every config, every embedded image, all 1000 logits bit-exact on Spike; 1000 val images bit-exact natively for B1/X2/S0/S1/S2 on both models |
| Microbenchmark / two operators / full model | instruction tests + kernel tests / conv 3×3, conv 1×1, stem, add, FC per-operator counts / full ResNet-18 |
| No floating point | `-march=rv64imc_zicsr`, no F/D instructions in any ELF |

## 5. Open items

* **FP32 gate (−0.21).** For a strict FP32 ≥ baseline sparse checkpoint, run
  one more non-QAT distillation recovery from `models/hoyer` with
  `final_thresholds.json` (about 1 GPU-hour). The INT8 model used in all ISA
  results would not change.
* **Cycle-level evaluation** belongs to the final project (CV-Wally RTL). Spike
  reports retired instructions only.

## 6. Reproducibility notes

* GPU work was done on 3× RTX A5000:
  - phase 1: 10 epochs, about 58 min;
  - phase 3: 6 epochs with QAT + teacher, about 56 min;
  - the abandoned attempts in §1.2 took about 3 more GPU-hours.
* `results/sensitivity/hoyer.json` holds the full per-site sweep. Its
  `chosen`/`combined` fields are from the τ = 0.02 run. The τ = 0.005
  thresholds actually used are `hoyer_thresholds.json`, derived from the same
  sweep rows. `final_thresholds.json` is those × 0.85.
* Spike: `third_party/riscv-isa-sim` at commit `535112531c7e`. Toolchain:
  riscv-collab GCC 16.1 (`2026.08.27`).

