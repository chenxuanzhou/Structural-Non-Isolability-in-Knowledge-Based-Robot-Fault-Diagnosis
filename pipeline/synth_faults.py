# -*- coding: utf-8 -*-
"""Semi-synthetic fault INSTANTIATIONS on real normal cycles.

WHY THIS EXISTS
---------------
The leave-setting-out protocol (repair.py) needs independent fault
*instantiations*, not more cycles. voraus-AD has 755 anomalous cycles but only
72 instantiations, and five categories have three or fewer:

    coll_card 3 | platform 3 | misgrip 1 | entangled 1 | invalidgrip 1

That is why the gripping cluster is reported UNDETERMINED and the collision
cluster INCONCLUSIVE. No amount of resampling fixes it: a second cycle from the
same `setting` is the very leakage the protocol exists to exclude.

This module manufactures new instantiations by injecting physically motivated
perturbations into the 948 REAL train-normal cycles, so the trajectory, the
sensor noise floor and the controller behaviour all stay real; only the fault is
modelled. Each instantiation draws its own parameter vector, mirroring what a
real `setting` is: one physical configuration, run many times.

WHAT IT CAN AND CANNOT SHOW -- read before quoting any number
-------------------------------------------------------------
CIRCULARITY IS THE CENTRAL RISK. An injected fault carries the signature we gave
it, so "the repair search recovered the dimension we injected" proves nothing
about real faults. Worse, `misgrip` is injected by removing the payload torque
over the carry phase, which is *definitionally* the `payload` grammar dimension
(carry minus empty). Any claim of the form "representation repair separates
these failure modes" is therefore INADMISSIBLE from this module.

What IS admissible, because the answer is not built in:

  1. HOW MANY INSTANTIATIONS the leave-setting-out certificate needs before it
     stops returning "inconclusive". We control the count, so we can sweep it.
     This turns a dead end ("the dataset cannot decide") into a quantified data
     requirement ("this fault needs >= N instantiations; the dataset has 3").
  2. Whether the DEPLOYED representation (max|z|) already separates a pair, so
     that no repair is needed at all. Not knowable a priori.
  3. Whether stage-1 knowledge retrieval routes an injected fault to the cluster
     the FMEA says it belongs to -- a test of the knowledge base, not of us.

Each injector is transcribed from its failure mode's own FMEA `effect` text,
which predates the repair grammar, and none is written in terms of a grammar
statistic except where flagged above.

WHERE THE INJECTION IS DELIBERATELY CONSERVATIVE
------------------------------------------------
The perturbation is applied to `motor_torque` only. A real load change would
also move the joint-side torque sensor and the robot-level currents, but the
motor-to-sensor transfer is only estimable on three of six joints (correlation
0.68/0.81/0.62 on J1/J2/J6 against 0.22-0.27 on J3/J4/J5), and fitting a gain
where the data does not support one would be fabrication. Leaving the other
channels unperturbed UNDER-injects: a synthetic fault is less visible than the
real thing, so any "N instantiations needed" figure derived here is if anything
an over-estimate -- the safe direction for a data-requirement claim.

PAYLOAD TORQUE IS MEASURED, NOT MODELLED
----------------------------------------
No URDF is used. The payload's contribution to each joint torque is estimated
from the real normal cycles by POSE MATCHING: for each carry-phase sample, find
the nearest empty-phase sample of the same cycle in 6-D joint space (median
nearest-neighbour distance 0.023 rad) and difference the torques. Across the
normal cycles this gives a stable, physically coherent profile -- J2 -18.0 N.m
against J3 +14.2 N.m, the opposing pair gravity compensation requires when a
mass is held at the tool, with 100% sign consistency on J1-J4 and J6.

Run:
    ..\\.venv\\Scripts\\python.exe synth_faults.py --payload   # estimate + cache
    ..\\.venv\\Scripts\\python.exe synth_faults.py --power     # instantiation sweep
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PAYLOAD_CACHE = os.path.join(_ROOT, "data", "payload_torque.json")
RESULTS_OUT = os.path.join(_ROOT, "results", "synth_instantiation_power.json")

JOINTS = (1, 2, 3, 4, 5, 6)
GRID_N = 500

from attribution_bridge import CARRY_ACTIONS   # noqa: E402

POS = [f"joint_position_{j}" for j in JOINTS]
TQ = [f"motor_torque_{j}" for j in JOINTS]
TS = [f"torque_sensor_a_{j}" for j in JOINTS]
VEL = [f"joint_velocity_{j}" for j in JOINTS]


# ---------------------------------------------------------------------------
# 1. Payload torque, estimated from real normal cycles by pose matching
# ---------------------------------------------------------------------------
def estimate_payload_torque(n_cycles=200, pose_tol=0.05, verbose=True):
    """tau_payload_j(s) over normalized carry-phase progress s, per joint.

    Estimated ONLY from anomaly-free cycles, so it carries no label information.
    """
    from scipy.spatial import cKDTree
    from integration_test import DATA_PATH, TRAIN_SETTING

    df = pd.read_parquet(DATA_PATH,
                         columns=POS + TQ + ["time", "sample", "setting", "action"])
    nrm = df[df["setting"] == TRAIN_SETTING]
    sids = sorted(nrm["sample"].unique())[:n_cycles]
    grid = np.linspace(0, 1, 64)
    prof, used = [], 0
    for sid in sids:
        r = nrm[nrm["sample"] == sid].sort_values("time")
        P, T = r[POS].to_numpy(), r[TQ].to_numpy()
        carry = np.isin(r["action"].to_numpy(), list(CARRY_ACTIONS))
        if carry.sum() < 20 or (~carry).sum() < 20:
            continue
        d, idx = cKDTree(P[~carry]).query(P[carry])
        ok = d < pose_tol
        if ok.sum() < 10:
            continue
        dt = T[carry][ok] - T[~carry][idx[ok]]
        s = np.linspace(0, 1, int(carry.sum()))[ok]
        # BIN, do not interpolate. Only ~60% of carry samples find a pose match,
        # and interpolating across the gaps drags the profile toward whichever
        # region happens to be matched -- it flipped the sign of J2 when we did.
        # Bins without a match fall back to this cycle's own per-joint mean.
        b = np.clip(np.digitize(s, grid) - 1, 0, len(grid) - 1)
        prof_c = np.empty((6, len(grid)))
        for k in range(6):
            overall = dt[:, k].mean()
            sums = np.bincount(b, weights=dt[:, k], minlength=len(grid))
            cnts = np.bincount(b, minlength=len(grid))
            prof_c[k] = np.where(cnts > 0, sums / np.maximum(cnts, 1), overall)
        prof.append(prof_c)
        used += 1
    if not prof:
        raise SystemExit("pose matching failed -- no usable normal cycles")
    stack = np.stack(prof)                                   # (cycles, 6, 64)
    out = {"grid": grid.tolist(), "n_cycles": used, "pose_tol": pose_tol,
           "mean": stack.mean(axis=0).tolist(),
           "per_joint_mean": stack.mean(axis=(0, 2)).tolist(),
           "per_joint_sd": stack.mean(axis=2).std(axis=0).tolist()}
    with open(PAYLOAD_CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f)
    if verbose:
        print(f"payload torque from {used} normal cycles (pose tol {pose_tol} rad):")
        for k, j in enumerate(JOINTS):
            print(f"  J{j}: {out['per_joint_mean'][k]:+7.3f} N.m"
                  f"   (sd across cycles {out['per_joint_sd'][k]:.3f})")
        print(f"cached -> {PAYLOAD_CACHE}")
    return out


def load_payload():
    if not os.path.exists(PAYLOAD_CACHE):
        return estimate_payload_torque(verbose=False)
    with open(PAYLOAD_CACHE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Injectors -- one per failure mode, transcribed from its FMEA `effect` text
# ---------------------------------------------------------------------------
def _carry_mask(run):
    return np.isin(run["action"].to_numpy(), list(CARRY_ACTIONS))


def _payload_on_run(run, pay):
    """tau_payload evaluated on this run's carry samples -> (n_carry, 6)."""
    carry = _carry_mask(run)
    n = int(carry.sum())
    if n == 0:
        return carry, np.zeros((0, 6))
    g = np.asarray(pay["grid"])
    mean = np.asarray(pay["mean"])                           # (6, 64)
    s = np.linspace(0, 1, n)
    return carry, np.vstack([np.interp(s, g, mean[k]) for k in range(6)]).T


def inject_misgrip(run, pay, rng, p):
    """FM_misgrip -- 'Lower-than-expected lift load; failed place.'
    The part is never picked up, so the payload torque is absent for the whole
    carry phase. CIRCULARITY: this is the `payload` grammar dimension by
    construction (see the module docstring)."""
    carry, tp = _payload_on_run(run, pay)
    d = np.zeros((len(run), 6))
    d[carry] -= p["strength"] * tp
    return d, None


def inject_losecan(run, pay, rng, p):
    """FM_losecan -- 'Sudden torque drop on load joints late in cycle.'"""
    carry, tp = _payload_on_run(run, pay)
    d = np.zeros((len(run), 6))
    n = tp.shape[0]
    if n:
        k = int(p["drop_at"] * n)
        d[np.where(carry)[0][k:]] -= p["strength"] * tp[k:]
    return d, None


def inject_varweight(run, pay, rng, p):
    """FM_varweight -- 'Torque offset proportional to weight.'"""
    carry, tp = _payload_on_run(run, pay)
    d = np.zeros((len(run), 6))
    d[carry] += (p["mass_ratio"] - 1.0) * tp
    return d, None


def inject_entangled(run, pay, rng, p):
    """FM_cable_routed -- 'Elevated torque across motion; intermittent
    resistance.' Coulomb friction opposing motion on the affected joints: a
    standard friction term, not a grammar statistic."""
    d = np.zeros((len(run), 6))
    V = run[VEL].to_numpy()
    for j in p["joints"]:
        k = j - 1
        d[:, k] += -p["coulomb"] * np.tanh(V[:, k] / 0.05)
    return d, None


def inject_platform(run, pay, rng, p):
    """FM_platform -- 'Low-frequency oscillation; broad torque deviation on
    proximal joints.' A compliant base: a low-frequency perturbation on J1/J2
    whose amplitude follows the joint's own motion."""
    d = np.zeros((len(run), 6))
    t = run["time"].to_numpy()
    t = t - t.min()
    V = run[VEL].to_numpy()
    for j in (1, 2):
        k = j - 1
        env = np.abs(V[:, k]) / (np.abs(V[:, k]).max() + 1e-9)
        d[:, k] += p["amp"] * env * np.sin(2 * np.pi * p["freq"] * t + p["phase"])
    return d, None


def inject_invalidgrip(run, pay, rng, p):
    """FM_invalidgrip -- 'Position deviation + torque anomaly at grip.' The
    taught pose is off, so the wrist sits at an offset while the part is
    grasped and the payload acts about a shifted moment arm."""
    carry, tp = _payload_on_run(run, pay)
    d = np.zeros((len(run), 6))
    if carry.sum():
        d[carry] += p["lever"] * tp
    P = run[POS].to_numpy().copy()
    grip = np.isin(run["action"].to_numpy(), [5, 6, 7])       # DOWN_PICK / UP_PICK
    for j in p["joints"]:
        P[grip, j - 1] += p["dq"]
    return d, P


INJECTORS = {
    "FM_misgrip": inject_misgrip,
    "FM_losecan": inject_losecan,
    "FM_varweight": inject_varweight,
    "FM_cable_routed": inject_entangled,
    "FM_platform": inject_platform,
    "FM_invalidgrip": inject_invalidgrip,
}


def draw_params(fm, rng):
    """One draw = one physical configuration = one `setting`. Ranges straddle a
    plausible severity band and were not tuned against any result."""
    if fm == "FM_misgrip":
        return {"strength": float(rng.uniform(0.85, 1.0))}
    if fm == "FM_losecan":
        return {"strength": float(rng.uniform(0.85, 1.0)),
                "drop_at": float(rng.uniform(0.15, 0.75))}
    if fm == "FM_varweight":
        lo_hi = rng.uniform(0.55, 0.85) if rng.random() < 0.5 else rng.uniform(1.15, 1.5)
        return {"mass_ratio": float(lo_hi)}
    if fm == "FM_cable_routed":
        js = sorted(int(x) for x in rng.choice([1, 2, 3, 4, 5, 6],
                                               size=int(rng.integers(1, 4)), replace=False))
        return {"joints": js, "coulomb": float(rng.uniform(1.5, 6.0))}
    if fm == "FM_platform":
        return {"amp": float(rng.uniform(2.0, 8.0)),
                "freq": float(rng.uniform(1.5, 5.0)),
                "phase": float(rng.uniform(0.0, 6.283))}
    if fm == "FM_invalidgrip":
        js = sorted(int(x) for x in rng.choice([4, 5, 6],
                                               size=int(rng.integers(1, 3)), replace=False))
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        sgn2 = 1.0 if rng.random() < 0.5 else -1.0
        return {"joints": js, "dq": float(rng.uniform(0.03, 0.12)) * sgn,
                "lever": float(rng.uniform(0.1, 0.4)) * sgn2}
    raise KeyError(fm)


# ---------------------------------------------------------------------------
# 3. Generate bridge records through the UNCHANGED pipeline
# ---------------------------------------------------------------------------
def _load_normals():
    """Real train-normal cycles plus the exact column set the deployed
    attribution reads, so a synthetic record has the SAME 26 signature slots as
    a real one (24 per-joint + robot_current + system_current)."""
    from attribution_bridge import JOINT_SIGNALS, ROBOT_SIGNALS
    from integration_test import DATA_PATH, TRAIN_SETTING
    cols = ([f"{f}_{j}" for f in JOINT_SIGNALS for j in JOINTS] + list(ROBOT_SIGNALS))
    want = sorted(set(cols + POS + TQ + TS + VEL))
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
    want = [c for c in want if c in avail]
    df = pd.read_parquet(DATA_PATH, columns=want + ["time", "sample", "setting", "action"])
    return df[df["setting"] == TRAIN_SETTING], [c for c in cols if c in avail]


def generate(fm, n_inst, cycles_per_inst=8, seed=0, normals=None, ref=None,
             id_base=900000, verbose=False):
    """Bridge records for `n_inst` independent instantiations of `fm`.

    The envelope, the attribution and the record schema are the deployed ones;
    only the torque (and, for invalidgrip, the position) of an otherwise real
    normal cycle is perturbed. `setting` is the instantiation id, so the
    leave-setting-out machinery groups on it unchanged.
    """
    from attribution_bridge import build_reference, attribute, to_bridge_record

    pay = load_payload()
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, GRID_N)
    if normals is None:
        normals, cols = _load_normals()
        ref = build_reference(normals, cols, grid)
    # pre-group once: filtering the full frame per cycle is O(n) each time and
    # dominates the runtime of a sweep
    if not hasattr(generate, "_groups") or generate._src is not id(normals):
        generate._groups = {k: v.sort_values("time")
                            for k, v in normals.groupby("sample", sort=False)}
        generate._src = id(normals)
    groups = generate._groups
    sids = sorted(groups)

    recs = []
    for inst in range(n_inst):
        p = draw_params(fm, rng)
        pick = rng.choice(sids, size=cycles_per_inst, replace=False)
        for ci, sid in enumerate(pick):
            run = groups[sid].copy()
            delta, P = INJECTORS[fm](run, pay, rng, p)
            for k, j in enumerate(JOINTS):
                run[f"motor_torque_{j}"] = run[f"motor_torque_{j}"].to_numpy() + delta[:, k]
                if P is not None:
                    run[f"joint_position_{j}"] = P[:, k]
            # id_base keeps ids unique when several modes are generated into
            # one pool -- without it the second mode silently overwrites the first
            rec = to_bridge_record(attribute(run, ref, grid),
                                   sample_id=id_base + inst * 1000 + ci,
                                   category=-1, setting=id_base + inst)
            rec["synthetic"] = {"fm": fm, "instantiation": inst, "params": p,
                                "base_cycle": int(sid)}
            recs.append(rec)
        if verbose and (inst + 1) % 5 == 0:
            print(f"    {fm}: {inst + 1}/{n_inst} instantiations")
    return recs


if __name__ == "__main__":
    if "--payload" in sys.argv:
        estimate_payload_torque()
    elif "--power" in sys.argv:
        import synth_power
        synth_power.main()
    else:
        print(__doc__)
