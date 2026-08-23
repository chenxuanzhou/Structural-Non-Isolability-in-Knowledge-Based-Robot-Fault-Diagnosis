"""
Integration test / root-cause-accuracy harness
===============================================
Runs the full pipeline over every anomalous cycle and scores whether the line
KG retrieval surfaces the TRUE cause in top-k:

    detector(optional) -> attribution_bridge -> line_kg.retrieve -> score vs truth

TWO MODES
---------
* SCORED  : if CATEGORY_TO_FM is filled, computes top-1/3/5 accuracy + MRR,
            overall and per anomaly category. This is your first real number.
* PLUMBING: if CATEGORY_TO_FM is empty, it still runs end-to-end and reports
            that records are well-formed and retrieval returns candidates --
            a smoke test, NOT a result.

WHAT YOU MUST PROVIDE FOR A REAL NUMBER
---------------------------------------
1. DATA_PATH -> the FULL voraus-AD data (so the normal envelope uses all 948
   normal runs, not 1). A .pkl of the decoded DataFrame or the parquet.
2. CATEGORY_TO_FM -> map each voraus-AD category integer to the FailureMode id
   in line_kg.FMEA. Recover the integer<->name mapping from the dataset docs.
3. (optional) a trained MVT-Flow checkpoint to fill anomaly_score/threshold.
"""
from __future__ import annotations
import csv
import numpy as np
import pandas as pd

from attribution_bridge import (build_reference, attribute, to_bridge_record,
                                 JOINT_SIGNALS, ROBOT_SIGNALS, JOINTS, GRID_N)
from line_kg import retrieve

# ----------------------------------------------------------------------------
# CONFIG -- edit these
# ----------------------------------------------------------------------------
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_PATH = _os.path.join(_ROOT, "data", "voraus-ad-dataset-100hz.parquet")
NORMAL_CATEGORY = 12             # voraus-AD: 12 == anomaly-free
TRAIN_SETTING = 72               # Variant.PRE_A: the 948 normal-train runs (official split)
TOP_K = 5

# voraus-AD category-integer <-> FailureMode id mapping. Categories are the
# official `Category` IntEnum from vorausrobotik/voraus-ad-dataset (voraus_ad.py);
# 12 == NORMAL_OPERATION (the anomaly-free class, handled as NORMAL_CATEGORY).
CATEGORY_TO_FM: dict[int, str] = {
    0:  "FM_friction",        # AXIS_FRICTION
    1:  "FM_axisweight",      # AXIS_WEIGHT
    2:  "FM_collision_foam",  # COLLISION_FOAM
    3:  "FM_collision_cable", # COLLISION_CABLE
    4:  "FM_collision_card",  # COLLISION_CARTON
    5:  "FM_misgrip",         # MISS_CAN
    6:  "FM_losecan",         # LOSE_CAN
    7:  "FM_varweight",       # CAN_WEIGHT
    8:  "FM_cable_routed",    # ENTANGLED (entangled in the dress-pack/cables)
    9:  "FM_invalidgrip",     # INVALID_POSITION
    10: "FM_miscommutation",  # MOTOR_COMMUTATION
    11: "FM_platform",        # WOBBLING_STATION (unstable/wobbling base)
}

# Optional MVT-Flow checkpoint (set to a path to enable real detector scores).
# Set to None to skip the detector (faster; detector fields stay null).
MVTFLOW_CKPT: str | None = "mvtflow/checkpoints/mvtflow.pt"

_DETECTOR: dict = {}   # lazy, load-once cache of the trained flow


def _load_detector():
    """Load the MVT-Flow checkpoint once and cache it (the mvtflow modules use
    flat imports, so we add that folder to sys.path rather than import as a pkg)."""
    if "model" not in _DETECTOR:
        import os, sys
        mvt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mvtflow")
        if mvt not in sys.path:
            sys.path.insert(0, mvt)
        from attribution import load_checkpoint, detector_fields_for_bridge
        from data import cycles_to_tensor, Normalizer
        # resolve a relative checkpoint path against THIS file's dir (cwd-robust)
        ckpt_path = MVTFLOW_CKPT if os.path.isabs(MVTFLOW_CKPT) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), MVTFLOW_CKPT)
        model, ckpt = load_checkpoint(ckpt_path, device="cpu")
        _DETECTOR.update(
            model=model, ckpt=ckpt, fields_fn=detector_fields_for_bridge,
            to_tensor=cycles_to_tensor,
            norm=Normalizer(ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]))
    return _DETECTOR


# ----------------------------------------------------------------------------
def maybe_detector(record, run, channels):
    """If a checkpoint is configured, fill detector fields; else leave as-is."""
    if not MVTFLOW_CKPT:
        return record
    try:
        d = _load_detector()
        X, _ = d["to_tensor"](run, d["ckpt"]["channels"], d["ckpt"]["cfg"]["seq_len"])
        fields = d["fields_fn"](d["model"], d["ckpt"], d["norm"](X)[:1])
        record["detector"]["anomaly_score"] = fields["anomaly_score"]
        record["detector"]["threshold"] = fields["threshold"]
        record["detector"]["verdict"] = ("anomaly" if fields["anomaly_score"] > fields["threshold"]
                                         else "normal")
        # store the MVT-Flow per-signal gradient attribution (model-internal
        # cross-check vs the envelope attribution) for the UI / future ranking use.
        # NOTE: with the current light CPU flow this gradient is dominated by
        # channel-sensitivity (joint_velocity) and is NOT fault-discriminative, so
        # it is recorded for display but deliberately NOT fed into retrieve().
        record["detector"]["model_attribution_ranked"] = fields["model_attribution_ranked"][:8]
    except Exception as e:  # keep the harness running even if detector fails
        record["detector"]["_detector_error"] = str(e)
    return record


def run_harness():
    from attribution_bridge import measured_read_families, add_residual_columns
    read_chan = ([f"{f}_{j}" for f in measured_read_families() for j in JOINTS] + list(ROBOT_SIGNALS))
    chan = ([f"{f}_{j}" for f in JOINT_SIGNALS for j in JOINTS] + list(ROBOT_SIGNALS))  # attribution cols
    meta = ["time", "sample", "category", "action", "setting"]
    # the detector may need MORE channels than the envelope; include them in the read.
    det_chan: list[str] = []
    if MVTFLOW_CKPT and not DATA_PATH.endswith(".pkl"):
        try:
            det_chan = list(_load_detector()["ckpt"]["channels"])
        except Exception:
            det_chan = []
    if DATA_PATH.endswith(".pkl"):
        df = pd.read_pickle(DATA_PATH)
    else:
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
        df = pd.read_parquet(DATA_PATH,
                             columns=[c for c in dict.fromkeys(read_chan + det_chan + meta) if c in avail])
    add_residual_columns(df)                         # derive torque_residual / pos_error / vel_error
    cols = [c for c in chan if c in df.columns]
    grid = np.linspace(0, 1, GRID_N)

    # Reference (normal envelope): use the official train-normal split when the
    # 'setting' column is present (Variant.PRE_A == TRAIN_SETTING == the 948
    # normal-train runs) -- no test leakage; else fall back to all normal cycles
    # (the demo .pkl has no 'setting').
    normal = df[df["setting"] == TRAIN_SETTING] if "setting" in df.columns \
        else df[df.category == NORMAL_CATEGORY]
    n_normal = normal["sample"].nunique()
    ref = build_reference(normal, cols, grid)

    anom = df[df.category != NORMAL_CATEGORY]
    scored = bool(CATEGORY_TO_FM)
    rows, n_valid = [], 0

    for sid, run in anom.sort_values("time").groupby("sample"):
        cat = int(run["category"].iloc[0])
        setting = int(run["setting"].iloc[0]) if "setting" in run.columns else None
        rec = to_bridge_record(attribute(run, ref, grid), sample_id=sid,
                               category=cat, setting=setting)
        rec = maybe_detector(rec, run, cols)
        cands = retrieve(rec, top_k=TOP_K)
        cand_ids = [c["id"] for c in cands]
        valid = rec["localization"]["dominant_joint"] is not None and len(cands) > 0
        n_valid += int(valid)

        row = {
            "sample": sid, "category": cat,
            "dominant_joint": rec["localization"]["dominant_joint"],
            "dominant_phase": rec["localization"]["dominant_cycle_phase"],
            "top1": cand_ids[0] if cand_ids else None,
            "topk": "|".join(cand_ids),
        }
        if scored:
            true_fm = CATEGORY_TO_FM.get(cat)
            rank = (cand_ids.index(true_fm) + 1) if true_fm in cand_ids else 0
            row.update(true_fm=true_fm,
                       hit_top1=int(rank == 1),
                       hit_topk=int(rank >= 1),
                       rr=(1.0 / rank if rank else 0.0))
        rows.append(row)

    # ---- write per-cycle results ----
    out = _os.path.join(_ROOT, "results", "integration_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summary ----
    N = len(rows)
    print(f"normal runs used for envelope : {n_normal}"
          + ("   (WARNING: <5 -> attribution z is unreliable)" if n_normal < 5 else ""))
    print(f"anomalous cycles processed    : {N}")
    print(f"well-formed records + retrieval: {n_valid}/{N}")
    print(f"per-cycle results written to  : {out}\n")

    if not scored:
        print("MODE = PLUMBING (CATEGORY_TO_FM empty) -> this is a smoke test, NOT a result.")
        rdf = pd.DataFrame(rows)
        print("\ntop-1 retrieved cause per category (eyeball check):")
        print(rdf[["category", "dominant_joint", "top1"]].to_string(index=False))
        return

    rdf = pd.DataFrame(rows)
    print("MODE = SCORED")
    print(f"  top-1 accuracy : {rdf['hit_top1'].mean():.3f}")
    print(f"  top-{TOP_K} accuracy : {rdf['hit_topk'].mean():.3f}")
    print(f"  MRR            : {rdf['rr'].mean():.3f}\n")
    print("per-category top-k accuracy:")
    per = rdf.groupby("category").agg(n=("sample", "size"),
                                      top1=("hit_top1", "mean"),
                                      topk=("hit_topk", "mean"))
    print(per.to_string())


if __name__ == "__main__":
    run_harness()
