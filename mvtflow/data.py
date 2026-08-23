"""voraus-AD data loading: parquet -> fixed-length, per-channel-normalized tensors.

Each robot cycle ('sample') becomes one (C, L) example. We train on normal
cycles only (category == normal_category) and standardize per channel using
TRAIN statistics. For full reproduction, replace the random split with the
official voraus-AD train/test split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resample(arr: np.ndarray, L: int) -> np.ndarray:
    """Resample a (T,) signal to length L on a normalized time grid."""
    t = np.linspace(0, 1, len(arr))
    g = np.linspace(0, 1, L)
    return np.interp(g, t, arr)


def cycles_to_tensor(df: pd.DataFrame, channels: list[str], L: int) -> tuple[torch.Tensor, list]:
    """Return X of shape (N, C, L) and the list of sample ids."""
    xs, ids = [], []
    for sid, run in df.sort_values("time").groupby("sample"):
        chans = [_resample(run[c].to_numpy(), L) for c in channels]
        xs.append(np.stack(chans, axis=0))           # (C, L)
        ids.append(int(sid))
    return torch.tensor(np.stack(xs), dtype=torch.float32), ids


class Normalizer:
    """Per-channel standardization fitted on training data."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, X: torch.Tensor):                    # X: (N, C, L)
        mean = X.mean(dim=(0, 2), keepdim=True)
        std = X.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        return cls(mean, std)

    def __call__(self, X):
        return (X - self.mean) / self.std

    def state(self):
        return {"mean": self.mean, "std": self.std}


class TensorCycles(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i]


def build_datasets(cfg):
    """Official voraus-AD split when 'setting' is present:
      train = anomaly-free runs with setting == cfg.train_setting (Variant.PRE_A),
      test  = everything else (held-out normal runs + ALL anomalies).
    Falls back to (all-normal / all-anomaly) on the demo .pkl, which has no
    'setting' column. The test set is labelled (1 = anomaly) for AUROC / AU-PR.
    """
    if cfg.parquet_path.endswith(".pkl"):
        df = pd.read_pickle(cfg.parquet_path)
    else:
        import pyarrow.parquet as pq
        meta = ["time", "sample", "category", "setting"]
        avail = set(pq.ParquetFile(cfg.parquet_path).schema_arrow.names)
        df = pd.read_parquet(cfg.parquet_path,
                             columns=[c for c in list(cfg.channels) + meta if c in avail])

    has_setting = "setting" in df.columns
    if has_setting:
        train_df = df[(df["category"] == cfg.normal_category) & (df["setting"] == cfg.train_setting)]
        test_df = df[df["setting"] != cfg.train_setting]
    else:                                    # demo .pkl fallback
        train_df = df[df["category"] == cfg.normal_category]
        test_df = df[df["category"] != cfg.normal_category]

    Xn, ids = cycles_to_tensor(train_df, cfg.channels, cfg.seq_len)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(Xn), generator=g)
    n_val = int(len(Xn) * cfg.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    norm = Normalizer.fit(Xn[tr_idx])
    Xtr, Xval = norm(Xn[tr_idx]), norm(Xn[val_idx])

    # labelled test set for detector evaluation (AUROC / AU-PR)
    Xte, test_ids = cycles_to_tensor(test_df, cfg.channels, cfg.seq_len)
    Xte = norm(Xte)
    test_cat = (test_df.groupby("sample")["category"].first()
                .reindex(test_ids).to_numpy())
    test_label = (test_cat != cfg.normal_category).astype(int)   # 1 == anomaly

    return {
        "train": TensorCycles(Xtr), "val": TensorCycles(Xval),
        "test_X": Xte, "test_ids": test_ids, "test_cat": test_cat, "test_label": test_label,
        "normalizer": norm, "channels": cfg.channels,
    }
