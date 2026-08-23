"""MVT-Flow: an affine-coupling normalizing flow over multivariate time series.

Self-contained PyTorch reimplementation (no FrEIA). The flow maps a cycle
x:(C,L) to a latent z:(C,L); density is evaluated under a standard-normal base
plus the log-det Jacobian. Trained on normal cycles by minimizing NLL; the
anomaly score is the per-cycle NLL. Subnets are 1D convolutions along the time
axis so the flow captures temporal structure (per the paper's design).

To match the paper byte-for-byte, swap this module for the official FrEIA-based
flow at github.com/vorausrobotik/voraus-ad-dataset; the train/score/attribution
interface below stays the same.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class Subnet(nn.Module):
    """Conv1d subnet predicting (scale, translate) for the coupling block."""
    def __init__(self, c_in: int, c_out: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, 2 * c_out, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)   # identity init for stable start
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        s, t = self.net(x).chunk(2, dim=1)
        return s, t


class AffineCoupling(nn.Module):
    """RealNVP-style affine coupling; `swap` alternates which half is transformed."""
    def __init__(self, c: int, hidden: int, swap: bool, clamp: float = 2.0):
        super().__init__()
        self.swap, self.clamp = swap, clamp
        self.c1 = c // 2
        self.c2 = c - self.c1
        cond_c, tgt_c = (self.c2, self.c1) if swap else (self.c1, self.c2)
        self.subnet = Subnet(cond_c, tgt_c, hidden)

    def forward(self, x):
        x1, x2 = x[:, :self.c1], x[:, self.c1:]
        cond, target = (x2, x1) if self.swap else (x1, x2)
        s, t = self.subnet(cond)
        s = self.clamp * torch.tanh(s)
        y = target * torch.exp(s) + t
        logdet = s.flatten(1).sum(1)
        out = torch.cat([y, x2], 1) if self.swap else torch.cat([x1, y], 1)
        return out, logdet


class MVTFlow(nn.Module):
    def __init__(self, c: int, hidden: int = 128, n_blocks: int = 4, clamp: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AffineCoupling(c, hidden, swap=bool(i % 2), clamp=clamp)
             for i in range(n_blocks)])

    def forward(self, x):
        logdet = x.new_zeros(x.shape[0])
        z = x
        for b in self.blocks:
            z, ld = b(z)
            logdet = logdet + ld
        return z, logdet

    def nll(self, x):
        """Per-cycle negative log-likelihood = the anomaly score."""
        z, logdet = self.forward(x)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(1)
        return -(log_pz + logdet)
