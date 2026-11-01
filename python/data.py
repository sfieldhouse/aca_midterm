"""ImageNet data loading.

IMAGENET_ROOT (env) defaults to ~/imagenet_local with train/ and val/ in
ImageFolder layout.  A fixed 10 000-image subset of train/ ("minival", 10 per
class) is held out for threshold / gating decisions so the official val set is
used only for final reporting.
"""
import os
import random

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

ROOT = os.path.expanduser(os.environ.get("IMAGENET_ROOT", "~/imagenet_local"))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# un-normalised 224x224 uint8 crop (used to export test images for bare-metal)
raw_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.PILToTensor()])


def _split_indices(ds, per_class=10, seed=0):
    by_cls = {}
    for i, (_, y) in enumerate(ds.samples):
        by_cls.setdefault(y, []).append(i)
    rng = random.Random(seed)
    hold = []
    for y in sorted(by_cls):
        hold += rng.sample(by_cls[y], per_class)
    hs = set(hold)
    train = [i for i in range(len(ds.samples)) if i not in hs]
    return train, sorted(hold)


def imagenet_train(tf=train_tf):
    """Training set minus the minival hold-out."""
    ds = datasets.ImageFolder(os.path.join(ROOT, "train"), tf)
    tr, _ = _split_indices(ds)
    return Subset(ds, tr)


def imagenet_minival(tf=eval_tf):
    ds = datasets.ImageFolder(os.path.join(ROOT, "train"), tf)
    _, ho = _split_indices(ds)
    return Subset(ds, ho)


def imagenet_val(tf=eval_tf):
    return datasets.ImageFolder(os.path.join(ROOT, "val"), tf)


def loader(ds, batch_size=256, shuffle=False, workers=16, sampler=None, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, drop_last=drop_last)


@torch.no_grad()
def evaluate(model, dl, device="cuda", max_batches=None, amp=False):
    """Returns (top1, top5, mean CE loss)."""
    model.eval()
    c1 = c5 = n = 0
    loss = 0.0
    for b, (x, y) in enumerate(dl):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(x)
        o = o.float()
        loss += torch.nn.functional.cross_entropy(o, y, reduction="sum").item()
        t5 = o.topk(5, 1).indices
        c1 += (t5[:, 0] == y).sum().item()
        c5 += (t5 == y[:, None]).any(1).sum().item()
        n += y.numel()
    return c1 / n, c5 / n, loss / n
