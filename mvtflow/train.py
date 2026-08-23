"""Train MVT-Flow on normal cycles, calibrate a threshold, save a checkpoint,
and evaluate the detector (AUROC + AU-PR) on the official test split."""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import build_datasets
from model import MVTFlow


def _nll_batched(model, X, dev, bs=64):
    """Per-cycle NLL over a big tensor, in batches (keeps CPU memory bounded)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model.nll(X[i:i + bs].to(dev)).cpu())
    return torch.cat(out) if out else torch.empty(0)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    ds = build_datasets(cfg)
    C = len(cfg.channels)
    model = MVTFlow(C, cfg.hidden, cfg.n_blocks, cfg.clamp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    dl = DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=True)

    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for xb in dl:
            xb = xb.to(dev)
            loss = model.nll(xb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        print(f"epoch {ep+1:3d}/{cfg.epochs}  train NLL/cycle = {tot/len(ds['train']):.2f}")

    # Calibrate threshold on normal-val (e.g. 95th percentile of NLL).
    # On the demo slice there may be too few normal cycles to form a val split;
    # fall back to train scores so the scaffold still produces a checkpoint
    # (on the full 948-run dataset the val split is well populated).
    model.eval()
    calib = ds["val"].X if len(ds["val"]) > 0 else ds["train"].X
    if len(ds["val"]) == 0:
        print("WARNING: empty val split (too few normal cycles) -> "
              "calibrating threshold on TRAIN scores; use full dataset for a real threshold.")
    val_scores = _nll_batched(model, calib, dev)
    threshold = float(torch.quantile(val_scores, 0.95))

    # ---- detector evaluation on the official test split (normal vs anomaly) ----
    from sklearn.metrics import roc_auc_score, average_precision_score
    test_scores = _nll_batched(model, ds["test_X"], dev).numpy()
    y = ds["test_label"]
    auroc = float(roc_auc_score(y, test_scores)) if len(set(y)) > 1 else float("nan")
    aupr = float(average_precision_score(y, test_scores)) if len(set(y)) > 1 else float("nan")
    pred = (test_scores > threshold).astype(int)
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    fpr = float((pred[y == 0] == 1).mean()) if (y == 0).any() else float("nan")
    print(f"\nTEST detector eval (n={len(y)}: {int((y==1).sum())} anomaly / "
          f"{int((y==0).sum())} normal)")
    print(f"  AUROC = {auroc:.4f}   AU-PR = {aupr:.4f}")
    print(f"  @PRE_A-val threshold:   TPR(recall) = {tpr:.3f}  FPR = {fpr:.3f}")

    # b1: recalibrate the operating threshold on HELD-OUT normal (settings != PRE_A).
    # The PRE_A-val threshold over-flags because test-normal is a different setting
    # distribution; split held-out normal into cal/eval so FPR is reported honestly.
    rng = np.random.default_rng(cfg.seed)
    norm_idx = np.where(y == 0)[0]
    rng.shuffle(norm_idx)
    cut = len(norm_idx) // 2
    cal_idx, evaln_idx = norm_idx[:cut], norm_idx[cut:]
    threshold_recal = float(np.quantile(test_scores[cal_idx], 0.95))
    fpr_recal = float((test_scores[evaln_idx] > threshold_recal).mean())
    tpr_recal = float((test_scores[y == 1] > threshold_recal).mean())
    print(f"  @recalibrated threshold (95th pct of held-out normal cal-split): "
          f"TPR = {tpr_recal:.3f}  FPR = {fpr_recal:.3f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "normalizer": ds["normalizer"].state(),
        "channels": cfg.channels,
        "threshold": threshold,
        "cfg": cfg.__dict__,
        "val_nll_mean": float(val_scores.mean()),
        "test_auroc": auroc, "test_aupr": aupr, "test_tpr": tpr, "test_fpr": fpr,
        "threshold_recal": threshold_recal,
        "test_tpr_recal": tpr_recal, "test_fpr_recal": fpr_recal,
    }, os.path.join(cfg.out_dir, "mvtflow.pt"))
    print(f"saved checkpoint. threshold(95th val NLL) = {threshold:.2f}")
    return model, ds, threshold


if __name__ == "__main__":
    cfg = Config()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    train(cfg)
