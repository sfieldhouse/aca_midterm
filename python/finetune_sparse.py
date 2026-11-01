"""Fine-tune ResNet-18 with square-Hoyer activation regularisation and FATReLU
thresholds (Kurtz et al., ICML 2020).  Multi-GPU via torchrun (DDP).

    torchrun --nproc_per_node=3 python/finetune_sparse.py \
        --init models/resnet18_tv_in1k.safetensors --hoyer 2e-9 --epochs 8 \
        --out models/hoyer

loss = CE + lambda * sum_sites H(a_site),  H(v) = (sum|v|)^2 / sum v^2
computed over each activation site's per-GPU batch tensor (batch 256/GPU).
FATReLU thresholds (if --thresholds json given) are fixed during training.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import evaluate, imagenet_minival, imagenet_train, loader  # noqa: E402
from resnet_fat import load_pretrained  # noqa: E402
from sparsity import measure  # noqa: E402


def hoyer_term(a):
    l1 = torch.linalg.vector_norm(a, 1, dtype=torch.float32)
    l2 = torch.linalg.vector_norm(a, 2, dtype=torch.float32)
    return (l1 / (l2 + 1e-6)) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256, help="per-GPU batch")
    ap.add_argument("--hoyer", type=float, default=0.0)
    ap.add_argument("--thresholds", default=None, help="json {site: T}")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps/epoch")
    ap.add_argument("--kd_teacher", default=None, help="optional teacher checkpoint for distillation")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--qat", action="store_true", help="quantization-aware training (python/qat.py)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_pretrained(a.init)
    if a.thresholds:
        model.set_thresholds(json.load(open(a.thresholds)))
    model = model.cuda()
    raw = model
    if a.qat:
        from qat import attach_qat
        from quantize import calibrate
        from data import eval_tf
        sp = os.path.join(a.out, "qat_scales.json")
        if rank == 0:
            os.makedirs(a.out, exist_ok=True)
            cds = imagenet_train(eval_tf)
            ci = torch.randperm(len(cds), generator=torch.Generator().manual_seed(0))[:1024].tolist()
            cal = [x for x, _ in loader(torch.utils.data.Subset(cds, ci), 128, workers=16)]
            json.dump(calibrate(model, cal), open(sp, "w"), indent=1)
            model.train()
        dist.barrier()
        attach_qat(model, json.load(open(sp)))
    model.set_record(a.hoyer > 0)
    ddp = DDP(model, device_ids=[dev])
    teacher = load_pretrained(a.kd_teacher).cuda().eval() if a.kd_teacher else None

    tr = imagenet_train()
    sampler = torch.utils.data.distributed.DistributedSampler(tr, shuffle=True)
    dl = loader(tr, a.bs, sampler=sampler, workers=a.workers, drop_last=True)
    mv = loader(imagenet_minival(), 250, workers=8) if rank == 0 else None

    decay, no_decay = [], []
    for n, p in raw.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(dl) if not a.max_steps else min(a.max_steps, len(dl))
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))

    os.makedirs(a.out, exist_ok=True)
    log = []
    if rank == 0:
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
        if a.thresholds:
            json.dump(raw.thresholds(), open(os.path.join(a.out, "thresholds.json"), "w"), indent=1)

    for ep in range(a.epochs):
        sampler.set_epoch(ep)
        ddp.train()
        t0 = time.time()
        sce = sh = 0.0
        for it, (x, y) in enumerate(dl):
            if a.max_steps and it >= a.max_steps:
                break
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ddp(x)
            ce = F.cross_entropy(out.float(), y)
            loss = ce
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tl = teacher(x).float()
                kd = F.kl_div(F.log_softmax(out.float() / a.kd_T, 1), F.softmax(tl / a.kd_T, 1),
                              reduction="batchmean") * a.kd_T ** 2
                loss = (1 - a.kd_alpha) * ce + a.kd_alpha * kd
            if a.hoyer > 0:
                h = sum(hoyer_term(m.last) for _, m in raw.act_sites())
                loss = loss + a.hoyer * h
                sh += h.item() if it % 50 == 0 else 0
                for _, m in raw.act_sites():
                    m.last = None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % 50 == 0:
                sce += ce.item()
                if rank == 0:
                    print(f"ep {ep} it {it}/{steps_per_epoch} ce {ce.item():.3f} "
                          f"hoyer {a.hoyer * (h.item() if a.hoyer > 0 else 0):.3f} "
                          f"lr {sched.get_last_lr()[0]:.5f} {(it + 1) * a.bs * world / (time.time() - t0):.0f} img/s",
                          flush=True)
        if rank == 0:
            raw.set_record(False)
            t1, t5, vl = evaluate(raw, mv)
            conv_sp, site_sp = measure(raw, mv, max_batches=8)
            raw.set_record(a.hoyer > 0)
            rec = {"epoch": ep, "minival_top1": t1, "minival_top5": t5, "minival_loss": vl,
                   "mean_conv_input_sparsity": sum(v for k, v in conv_sp.items() if k not in ("conv1", "fc")) / 19,
                   "conv_input_sparsity": conv_sp, "site_sparsity": site_sp,
                   "epoch_time_s": time.time() - t0}
            log.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, dict)}), flush=True)
            json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
            torch.save(raw.state_dict(), os.path.join(a.out, "model.pt"))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
