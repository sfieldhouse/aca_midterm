"""Activation-sparsity measurement helpers (fraction of exact zeros)."""
import torch
import torch.nn as nn


class ConvInputRecorder:
    """Forward-pre-hooks on every Conv2d/Linear: accumulate #zeros and #elements of its input."""

    def __init__(self, model):
        self.stats = {}
        self.handles = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.handles.append(m.register_forward_pre_hook(self._hook(n)))

    def _hook(self, name):
        def f(mod, inp):
            x = inp[0]
            z, t = self.stats.get(name, (0, 0))
            self.stats[name] = (z + (x == 0).sum().item(), t + x.numel())
        return f

    def sparsity(self):
        return {k: z / t for k, (z, t) in self.stats.items()}

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def measure(model, dl, max_batches=8, device="cuda"):
    """Returns ({conv_name: input sparsity}, {site_name: output sparsity})."""
    model.eval()
    rec = ConvInputRecorder(model)
    site = {}
    model.set_record(True)
    for b, (x, _) in enumerate(dl):
        if b >= max_batches:
            break
        model(x.to(device))
        for n, m in model.act_sites():
            z, t = site.get(n, (0, 0))
            site[n] = (z + (m.last == 0).sum().item(), t + m.last.numel())
    model.set_record(False)
    rec.close()
    return rec.sparsity(), {k: z / t for k, (z, t) in site.items()}
