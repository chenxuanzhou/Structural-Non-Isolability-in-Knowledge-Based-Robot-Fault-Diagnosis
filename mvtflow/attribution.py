"""Scoring, per-signal gradient attribution, and bridge integration.

The paper's temporal attribution = L1-norm of d(NLL)/d(input) summed over signals
(localizes WHEN). The extension here keeps the gradient PER SIGNAL (localizes
WHICH joint/signal) -- the model-internal attribution that cross-checks the
envelope-based attribution in attribution_bridge.py.

`detector_fields_for_bridge()` returns the {anomaly_score, threshold} that fill
the two `null` fields of the bridge record, plus a per-signal attribution that
can SEED or VALIDATE the bridge's localization.
"""
from __future__ import annotations
import torch

from model import MVTFlow


def load_checkpoint(path: str, device: str = "cpu"):
    # our own checkpoint (trusted) carries cfg/normalizer objects, not just weights
    ckpt = torch.load(path, map_location=device, weights_only=False)
    C = len(ckpt["channels"])
    cfg = ckpt["cfg"]
    model = MVTFlow(C, cfg["hidden"], cfg["n_blocks"], cfg["clamp"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def attribution(model: MVTFlow, x: torch.Tensor):
    """x: (1, C, L). Returns per-signal (C,), per-time (L,), and full (C,L) maps."""
    x = x.clone().requires_grad_(True)
    nll = model.nll(x).sum()
    grad = torch.autograd.grad(nll, x)[0].abs()[0]   # (C, L)
    per_signal = grad.sum(dim=1)                      # WHICH signal/joint  <-- extension
    per_time = grad.sum(dim=0)                        # WHEN (paper's attribution)
    return per_signal.detach(), per_time.detach(), grad.detach()


def detector_fields_for_bridge(model, ckpt, x: torch.Tensor) -> dict:
    """Produce the detector half of the bridge record for one cycle x:(1,C,L)."""
    with torch.no_grad():
        score = float(model.nll(x).item())
    per_signal, per_time, _ = attribution(model, x)
    channels = ckpt["channels"]
    ranked = sorted(
        ({"signal": channels[i],
          "joint": (int(channels[i].split("_")[-1]) if channels[i][-1].isdigit() else None),
          "attribution": float(per_signal[i])} for i in range(len(channels))),
        key=lambda d: d["attribution"], reverse=True)
    return {
        "anomaly_score": score,
        "threshold": ckpt["threshold"],
        "verdict": "anomaly" if score > ckpt["threshold"] else "normal",
        "peak_time_index": int(torch.argmax(per_time)),
        "model_attribution_ranked": ranked,   # cross-check vs envelope attribution
    }


if __name__ == "__main__":
    # Example wiring (needs a trained checkpoint):
    import numpy as np, pandas as pd
    from config import Config
    from data import cycles_to_tensor, Normalizer

    cfg = Config()
    model, ckpt = load_checkpoint(f"{cfg.out_dir}/mvtflow.pt", device="cpu")
    df = (pd.read_pickle(cfg.parquet_path) if cfg.parquet_path.endswith(".pkl")
          else pd.read_parquet(cfg.parquet_path, columns=list(cfg.channels) + ["sample", "category", "time"]))
    run = df[df["category"] == 5]
    X, _ = cycles_to_tensor(run, ckpt["channels"], cfg.seq_len)
    norm = Normalizer(ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"])
    x = norm(X)[:1]
    import json
    print(json.dumps(detector_fields_for_bridge(model, ckpt, x), indent=2)[:1200])
