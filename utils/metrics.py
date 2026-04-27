"""
utils/metrics.py — Helper functions để tính và track metrics trong training.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


class AverageMeter:
    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        fmtstr = f"{{name}} {{val{self.fmt}}} (avg: {{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple = (1, 5)) -> List[float]:

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Lấy top-k predicted classes
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                               # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N)

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            results.append(acc.item())

        return results


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"