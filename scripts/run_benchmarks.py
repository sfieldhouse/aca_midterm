#!/usr/bin/env python3
"""Build and run the ResNet-18 workload on Spike for every (model, config) and
collect ISA-level statistics.

  scripts/run_benchmarks.py                       # full matrix, 8 images
  scripts/run_benchmarks.py --models sparse --cfgs S2 --nimg 2
  scripts/run_benchmarks.py --mix                 # + instruction mix (spike -g, 1 image)
  scripts/run_benchmarks.py --sweep               # accumulator-count / dispatch sweep

Outputs (results/):
  raw/<model>/<tag>.log        Spike stdout
  benchmarks.csv               one row per run (total instret, PASS, code size, MAC/instr)
  ops.csv                      per-operator instret per image
  sparsity_runtime.csv         conv-input density observed at runtime (sparse configs)
  mix.csv                      dynamic instruction mix (with --mix)
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = os.path.join(ROOT, "tools/spike/bin/spike")
SPIKE_FLAGS = ["--isa=rv64imc_zicsr", "--extension=xqnn"]
CFGS = ["B0", "B1", "X1", "X2", "S0", "S1", "S2"]
CFG_DESC = {
    "B0": "naive dense, RV64IMC",
    "B1": "optimized dense, RV64IMC",
    "X1": "dense + xqmac8",
    "X2": "dense + xqmac8 + xrequant",
    "S0": "sparse-input (CSR), RV64IMC",
    "S1": "sparse + xsqmac8",
    "S2": "sparse + xsqmac8 + xqmac8 + xrequant (all)",
}


def tag(cfg, nimg, ngrp, disp):
    return f"{cfg}_n{nimg}_g{ngrp}_d{disp}"


def build(model, cfg, nimg, ngrp, disp):
    subprocess.run(["make", "-s", "elf", f"MODEL={model}", f"CFG={cfg}", f"NIMG={nimg}",
                    f"NGRP={ngrp}", f"DISPATCH={disp}"], cwd=ROOT, check=True)
    return os.path.join(ROOT, "build", model, tag(cfg, nimg, ngrp, disp))


def run_spike(outdir, log, hist=False, ngrp=8):
    env = dict(os.environ, XQNN_NGRP=str(ngrp))
    cmd = [SPIKE] + (["-g"] if hist else []) + SPIKE_FLAGS + [os.path.join(outdir, "resnet18.elf")]
    with open(log, "w") as fo, open(log + (".hist" if hist else ".err"), "w") as fe:
        r = subprocess.run(cmd, stdout=fo, stderr=fe, env=env)
    return r.returncode


def parse(log):
    imgs, ops, nnz, res = [], {}, {}, None
    for line in open(log):
        p = line.split()
        if not p:
            continue
        if p[0] == "IMG":
            imgs.append(dict(i=int(p[1]), instret=int(p[3]), pred=int(p[5]), label=int(p[7]), ok=p[9] == "OK"))
        elif p[0] == "OP":
            ops[p[1]] = int(p[2])
        elif p[0] == "NNZ":
            nnz[int(p[1])] = (int(p[2]), int(p[3]))
        elif p[0] == "RESULT":
            res = p[1]
    return imgs, ops, nnz, res


def text_size(outdir):
    for line in open(os.path.join(outdir, "size.txt")):
        p = line.split()
        if p and p[0] == ".text":
            return int(p[1])
    return 0


# ------------------------------------------------------------------ instruction mix
LOADS = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.ld", "c.lw", "c.ldsp", "c.lwsp"}
STORES = {"sb", "sh", "sw", "sd", "c.sd", "c.sw", "c.sdsp", "c.swsp"}
MULS = {"mul", "mulh", "mulhu", "mulhsu", "mulw", "div", "divu", "rem", "remu", "divw", "divuw", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble",
          "bgtu", "bleu", "c.beqz", "c.bnez", "j", "jal", "jalr", "jr", "ret", "c.j", "c.jr", "c.jalr", "c.jal", "call", "tail"}
XQNN = {0: "xqmac8", 1: "xqmac8u", 2: "xrequant", 3: "xsqmac8", 4: "xsaccrd", 5: "xsaccclr"}


def load_dis(path):
    """pc -> (mnemonic, function) from objdump -d output."""
    m, fn = {}, "?"
    for line in open(path):
        h = re.match(r"^([0-9a-f]+) <(.+)>:", line)
        if h:
            fn = h.group(2)
            continue
        r = re.match(r"^\s+([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)", line)
        if r:
            pc, word, mn = int(r.group(1), 16), int(r.group(2), 16), r.group(3)
            if mn.startswith(".insn") or (len(r.group(2)) == 8 and (word & 0x7F) == 0x0B):
                mn = XQNN.get((word >> 12) & 7, "custom")
            m[pc] = (mn, fn)
    return m


def classify(mn):
    if mn in XQNN.values():
        return "custom"
    if mn in LOADS:
        return "load"
    if mn in STORES:
        return "store"
    if mn in MULS:
        return "mul"
    if mn in BRANCH:
        return "branch"
    if mn.startswith("csr"):
        return "csr"
    return "alu"


def parse_hist(hist, dis):
    counts, per_mn, per_fn = {}, {}, {}
    for line in open(hist):
        p = line.split()
        if len(p) != 2 or not re.fullmatch(r"[0-9a-f]+", p[0]):
            continue
        pc, n = int(p[0], 16), int(p[1])
        mn, fn = dis.get(pc, ("?", "?"))
        c = classify(mn)
        counts[c] = counts.get(c, 0) + n
        per_mn[mn] = per_mn.get(mn, 0) + n
        per_fn[fn] = per_fn.get(fn, 0) + n
    return counts, per_mn, per_fn


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--cfgs", nargs="+", default=CFGS)
    ap.add_argument("--nimg", type=int, default=8)
    ap.add_argument("--ngrp", type=int, default=8)
    ap.add_argument("--dispatch", type=int, default=1001)
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count()))
    ap.add_argument("--mix", action="store_true", help="instruction mix via spike -g (1 image)")
    ap.add_argument("--sweep", action="store_true", help="S2 accumulator-group and dispatch sweep")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    runs = []
    for m in a.models:
        for c in a.cfgs:
            runs.append((m, c, a.nimg, a.ngrp, a.dispatch, False))
        if a.mix:
            for c in a.cfgs:
                runs.append((m, c, 1, a.ngrp, a.dispatch, True))
        if a.sweep:
            for g in (2, 4, 8, 16):
                runs.append((m, "S1", a.nimg, g, 1001, False))
                runs.append((m, "S2", a.nimg, g, 1001, False))
            for d in (500, 600, 700, 800, 900):
                runs.append((m, "S2", a.nimg, a.ngrp, d, False))
    runs = list(dict.fromkeys(runs))

    print(f"building {len(runs)} configurations ...", flush=True)
    dirs = {r: build(*r[:5]) for r in runs}

    def job(r):
        m, c, n, g, d, hist = r
        os.makedirs(os.path.join(a.results, "raw", m), exist_ok=True)
        log = os.path.join(a.results, "raw", m, tag(c, n, g, d) + ("_mix" if hist else "") + ".log")
        rc = run_spike(dirs[r], log, hist, g)
        print(f"  done {m:6s} {tag(c, n, g, d):22s} {'mix ' if hist else ''}rc={rc}", flush=True)
        return r, log

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, runs))

    bench_rows, op_rows, sp_rows, mix_rows = [], [], [], []
    for r, log in results:
        m, c, n, g, d, hist = r
        meta = json.load(open(os.path.join(ROOT, "models", m, "export", "meta.json")))
        macs = meta["total_macs"]
        imgs, ops, nnz, res = parse(log)
        if hist:
            dis = load_dis(os.path.join(dirs[r], "resnet18.dis"))
            counts, per_mn, per_fn = parse_hist(log + ".hist", dis)
            tot = sum(counts.values())
            row = dict(model=m, cfg=c, ngrp=g, dispatch=d, total=tot, macs=macs)
            row.update({k: counts.get(k, 0) for k in ("load", "store", "alu", "mul", "branch", "custom", "csr")})
            row.update({f"insn_{k}": per_mn.get(k, 0) for k in XQNN.values()})
            row["top_functions"] = json.dumps(dict(sorted(per_fn.items(), key=lambda x: -x[1])[:8]))
            mix_rows.append(row)
            continue
        ins = [x["instret"] for x in imgs]
        if not ins:
            print(f"WARNING: no output for {log}")
            continue
        mean = statistics.mean(ins)
        bench_rows.append(dict(model=m, cfg=c, desc=CFG_DESC[c], ngrp=g, dispatch=d, nimg=len(ins),
                               instret_mean=round(mean), instret_std=round(statistics.pstdev(ins)),
                               instret_min=min(ins), instret_max=max(ins), result=res,
                               golden_ok=sum(x["ok"] for x in imgs), top1_correct=sum(x["pred"] == x["label"] for x in imgs),
                               text_bytes=text_size(dirs[r]), macs=macs, mac_per_instr=round(macs / mean, 3)))
        if g == a.ngrp and d == a.dispatch:
            for k, v in ops.items():
                op_rows.append(dict(model=m, cfg=c, op=k, instret_per_img=v / len(ins)))
            for ci, (nz, tot) in sorted(nnz.items()):
                sp_rows.append(dict(model=m, cfg=c, conv=meta["layers"][ci]["name"], density=nz / tot))

    os.makedirs(a.results, exist_ok=True)

    def write(name, rows, key):
        if not rows:
            return
        path = os.path.join(a.results, name)
        old = []
        if os.path.exists(path):   # merge with previous results (replace same keys)
            new_keys = {tuple(str(r[k]) for k in key) for r in rows}
            old = [r for r in csv.DictReader(open(path)) if tuple(r[k] for k in key) not in new_keys]
        fields = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for r in old + rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print("wrote", path)

    write("benchmarks.csv", bench_rows, ["model", "cfg", "ngrp", "dispatch"])
    write("ops.csv", op_rows, ["model", "cfg", "op"])
    write("sparsity_runtime.csv", sp_rows, ["model", "cfg", "conv"])
    write("mix.csv", mix_rows, ["model", "cfg", "ngrp", "dispatch"])
    for r in sorted(bench_rows, key=lambda r: (r["model"], CFGS.index(r["cfg"]), r["ngrp"], r["dispatch"])):
        print(f"{r['model']:6s} {r['cfg']} g{r['ngrp']:<2d} d{r['dispatch']:<4d} {r['instret_mean']:>14,d} instr/img "
              f"MAC/instr {r['mac_per_instr']:6.3f}  {r['result']}")
    if any(r["result"] != "PASS" for r in bench_rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
