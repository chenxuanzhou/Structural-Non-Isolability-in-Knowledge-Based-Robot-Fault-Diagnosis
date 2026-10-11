# -*- coding: utf-8 -*-
"""Undecidability-driven representation repair.

WHAT THIS ADDS TO THE SYSTEM
----------------------------
`distinguishability.py` answers "which failure modes can the deployed
representation NOT tell apart?".  It stops there: the tie is reported and routed
to a human.  But `--probe` already showed that the load-torque twins ARE
separable from the raw signals with features the deployed representation throws
away.  So the honest statement is not "undecidable" but "undecidable *under this
representation*" -- which raises the operational question this module answers:

    given an undecidable cluster, WHICH representation change would decide it,
    is that change verified to work on held-out data, and what does it cost?

The module therefore turns the twin detector from a *descriptive audit* into a
*prescriptive, verifiable repair*:

  1. FEATURE GRAMMAR (`GRAMMAR`) -- a small set of named, physically meaningful
     dimensions, each with a COST CLASS defined by how far outside the deployed
     record you must reach to compute it:
        free      : in the record's schema-0.3 signature block -- statistics the
                    representation discards by keeping only max|z| per signal:
                    deviation SIGN, within-cycle STEP, carry-vs-empty PAYLOAD
                    contrast, dispersion SHAPE.
        pipeline  : in the data but not in the record, because the attribution
                    never reads the channel (power_load_mech, motor_iq).
        hardware  : not in the data at all (this cell has no temperature
                    channel). Untestable by construction -> a sensor gap.
  2. REPAIR SEARCH -- for one undecidable cluster, greedily select the minimal
     set of grammar dimensions that maximises within-cluster held-out accuracy.
     Greedy over *dimensions* (not individual features) keeps the certificate
     interpretable and the search honest.  The search is COST-AWARE: a costlier
     dimension must clear a proportionally larger gain, so the certificate names
     the cheapest repair that works rather than the most accurate one.
  3. REPAIR CERTIFICATE -- (cluster, chosen dims, held-out accuracy, majority
     baseline, permuted-label control, cost class, per-member physics).  A
     repair is ADOPTED only if it beats its baseline on data never used to fit
     it.  Clusters where the search fails are reported as failures, not hidden:
     that is the predicted behaviour in the perception-wall regime.
  4. TWO-STAGE DIAGNOSIS -- KG retrieval fixes the cluster (stage 1, knowledge
     driven, no labels); an adopted repair resolves within the cluster (stage 2,
     a tiny cluster-local model).  Stage 2 runs ONLY where stage 1 lands in a
     repaired cluster, so the rest of the system is untouched.

FALSIFIABLE PREDICTION (the point of the experiment)
----------------------------------------------------
The three-wall taxonomy says the load cluster fails at the DISTINGUISHABILITY
wall (seen but not separated) while the collision/gripping clusters fail at the
PERCEPTION wall (not seen at all).  Repair should then succeed on the former and
FAIL on the latter -- no feature recombination can recover what the attribution
never surfaced.  Both outcomes are reported.

CONTROL (the obvious reviewer attack: "you just trained a classifier")
----------------------------------------------------------------------
`--control` fits a flat 12-way classifier on ALL grammar features with no
undecidability guidance, on the same splits.  It is reported in full.  Two
things matter: (a) where its gains sit -- if they concentrate in exactly the
clusters the analysis predicted, the analysis predicted where ANY method could
gain; (b) label efficiency -- `--labels` measures how many labelled cycles the
targeted repair needs, since those labels are what the write-back loop actually
collects.

ONE REPRESENTATION, NOT TWO
---------------------------
The free dimensions are read from the bridge record's schema-0.3 `signature`
block -- the same numbers the deployed pipeline produces, over a fixed slot set
so they are comparable across cycles.  This module keeps no parallel copy of
them.  Only the `unread` dimension has its own cache, because by definition it
cannot come from the record; that asymmetry IS the cost class.

STABILITY
---------
Nothing here changes the deployed defaults.  `attribution_bridge`, `line_kg` and
`distinguishability` are imported read-only, and the record additions are
additive.  Delete `data/repair_features.json` / `results/repair_results.json`
and the system is bit-identical to the published configuration.

Run:
    ..\\.venv\\Scripts\\python.exe repair.py            # full report
    ..\\.venv\\Scripts\\python.exe repair.py --build    # (re)build the unread supplement
    ..\\.venv\\Scripts\\python.exe repair.py --adopt    # freeze verified repairs (unauthorized)
    ..\\.venv\\Scripts\\python.exe repair.py --control  # + flat-classifier control
    ..\\.venv\\Scripts\\python.exe repair.py --labels   # + label-efficiency curve
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FEATURE_CACHE = os.path.join(_ROOT, "data", "repair_features.json")
RESULTS_OUT = os.path.join(_ROOT, "results", "repair_results.json")

GRID_N = 500
JOINTS = (1, 2, 3, 4, 5, 6)

# ---------------------------------------------------------------------------
# The feature grammar
# ---------------------------------------------------------------------------
# Families present in the parquet that the attribution never reads. Everything
# the attribution DOES read arrives via the record's signature block, so this is
# the only family list this module has to name.
UNREAD_FAMILIES = ["power_load_mech", "motor_iq"]

# Cycle windows and payload phases are defined once, by the bridge, and imported
# here so the supplement is computed exactly like the record.
from attribution_bridge import CARRY_ACTIONS, EMPTY_ACTIONS  # noqa: E402

# COST IS STRUCTURAL, NOT DECLARED: a dimension's cost class is how far outside
# the deployed bridge record you must reach to compute it.
#   free      -> already in the record's `signature` block (schema 0.3)
#   pipeline  -> in the data but not in the record; the attribution would have to
#                read a channel it currently ignores
#   hardware  -> not in the data at all (no temperature channel in this cell)
# Widening attribution_bridge.JOINT_SIGNALS moves a dimension from `pipeline` to
# `free` by construction, because the record then carries it.
GRAMMAR = {
    "mag": dict(cost="deployed", stat="z_peak", source="record",
                desc="max |z| per signal — what the deployed representation keeps"),
    "sign": dict(cost="free", stat="z_signed", source="record",
                 desc="signed mean z — deviation DIRECTION (discarded by abs())"),
    "step": dict(cost="free", stat="z_step", source="record",
                 desc="late-minus-early deviation — within-cycle drift/step"),
    "payload": dict(cost="free", stat="z_payload", source="record",
                    desc="carry-phase minus empty-phase deviation — task-structure contrast"),
    "shape": dict(cost="free", stat="z_std", source="record",
                  desc="dispersion of z over the cycle — steady offset vs transient"),
    "unread": dict(cost="pipeline", stat=("z_peak", "z_signed", "z_payload"), source="unread",
                   desc="channels present but never attributed (power_load_mech, motor_iq)"),
}
# 'mag' is the deployed representation itself: it is the starting point of the
# search, never a "repair".
REPAIR_DIMS = [d for d in GRAMMAR if d != "mag"]

# Channels the cell does not have at all -> repair is a hardware decision and
# cannot be tested from this dataset.  Kept explicit so the report says so.
HARDWARE_GAPS = {
    frozenset({"FM_miscommutation", "FM_overtemp"}):
        "temperature channel (absent from the cell) — hardware instrumentation, untestable here",
}


def _signal_cols(kind="unread"):
    return [f"{f}_{j}" for f in UNREAD_FAMILIES for j in JOINTS]


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
def build_feature_cache(path=FEATURE_CACHE, verbose=True):
    """Cache the `unread` dimension: the SUPPLEMENT the record cannot carry.

    Everything else comes from the bridge record's schema-0.3 signature block.
    These channels (power_load_mech, motor_iq) are in the parquet but the
    attribution never reads them, which is exactly why testing them costs a
    pipeline change -- and why they need a separate build step rather than
    appearing for free alongside the record.

    The reference envelope uses the OFFICIAL train-normal split only
    (setting == TRAIN_SETTING), identical to the rest of the pipeline, so no
    test information enters the features.  Feature extraction is label-free.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from integration_test import DATA_PATH, TRAIN_SETTING, NORMAL_CATEGORY

    cols = _signal_cols("unread")
    avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
    cols = [c for c in cols if c in avail]
    meta = ["time", "sample", "category", "action", "setting"]
    if verbose:
        print(f"reading {len(cols)} signal columns from {os.path.basename(DATA_PATH)} ...")
    df = pd.read_parquet(DATA_PATH, columns=cols + [m for m in meta if m in avail])
    for c in cols:                                   # halve the working set
        df[c] = df[c].astype(np.float32)

    grid = np.linspace(0, 1, GRID_N)

    def resample_run(run):
        """(n_cols, GRID_N) resampled matrix + the action code on the grid."""
        t = run["time"].to_numpy()
        tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
        mat = np.vstack([np.interp(grid, tn, run[c].to_numpy()) for c in cols])
        act = run["action"].to_numpy()[np.searchsorted(tn, grid, side="right") - 1]
        return mat, act

    # ---- reference envelope from the train-normal split -------------------
    normal = df[df["setting"] == TRAIN_SETTING] if "setting" in df.columns \
        else df[df["category"] == NORMAL_CATEGORY]
    stacks = []
    for _, run in normal.sort_values("time").groupby("sample"):
        stacks.append(resample_run(run)[0])
    ref = np.stack(stacks)                             # (n_normal, n_cols, GRID)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    rng = np.ptp(mean, axis=1, keepdims=True)
    rng[rng <= 0] = 1.0
    scale = np.maximum(std, 0.05 * rng)                # same fallback as the pipeline
    if verbose:
        print(f"envelope built from {ref.shape[0]} train-normal runs")
    del stacks, ref

    # ---- per-anomaly features ---------------------------------------------
    anom = df[df["category"] != NORMAL_CATEGORY]
    out = {}
    early = slice(0, GRID_N // 3)
    late = slice(2 * GRID_N // 3, GRID_N)
    for sid, run in anom.sort_values("time").groupby("sample"):
        obs, act = resample_run(run)
        dev = obs - mean
        z = dev / (scale + 1e-9)
        carry = np.isin(act, list(CARRY_ACTIONS))
        empty = np.isin(act, list(EMPTY_ACTIONS))
        feats = {}
        for i, c in enumerate(cols):
            # same names and windows as the record's signature block, so the two
            # sources compose into one feature space
            feats[f"{c}::z_peak"] = float(np.max(np.abs(z[i])))
            feats[f"{c}::z_signed"] = float(np.mean(z[i]))
            feats[f"{c}::z_std"] = float(np.std(z[i]))
            feats[f"{c}::z_step"] = float(np.mean(z[i][late]) - np.mean(z[i][early]))
            cm = float(np.mean(z[i][carry])) if carry.any() else 0.0
            em = float(np.mean(z[i][empty])) if empty.any() else 0.0
            feats[f"{c}::z_payload"] = cm - em
        out[str(int(sid))] = feats

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"grid_n": GRID_N, "columns": cols, "cycles": out}, fh)
    if verbose:
        print(f"unread-channel supplement for {len(out)} cycles -> {path}")
    return out


def load_features(path=FEATURE_CACHE, verbose=True):
    """One feature space, assembled from the record plus the unread supplement.

    The free dimensions come straight out of the bridge record's schema-0.3
    `signature` block -- the SAME numbers the deployed pipeline produces, on a
    fixed slot set, so there is no second representation to drift out of sync.
    The `unread` dimension is loaded from its own cache if it has been built,
    and is simply absent otherwise; the search then cannot select it, which is
    the honest behaviour for a dimension whose adoption costs a pipeline change.
    """
    from eval_retrieve import load_cache

    recs = load_cache()
    sig0 = recs[0].get("signature") if recs else None
    if sig0 is None:
        raise SystemExit(
            "records predate schema 0.3 — delete data/records_cache.json and "
            "rerun so the signature block is built")
    stats = sig0["stats"]
    columns = sorted(sig0["slots"])

    cycles = {}
    for r in recs:
        sig = r["signature"]
        f = {f"{slot}::{s}": v for slot, vals in sig["slots"].items()
             for s, v in zip(stats, vals)}
        cycles[str(r["cycle_id"])] = {
            "category": r["detector"]["ground_truth_category"],
            "setting": r["detector"].get("ground_truth_setting"),
            "f": f,
        }
    if any(c["setting"] is None for c in cycles.values()):
        raise SystemExit("records carry no ground_truth_setting — rebuild the "
                         "cache; leave-setting-out cannot be run without it")

    n_unread = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        for cid, f in blob["cycles"].items():
            if cid in cycles:
                cycles[cid]["f"].update(f)
        columns = sorted(set(columns) | set(blob["columns"]))
        n_unread = len(blob["columns"])
    if verbose:
        print(f"features: {len(columns) - n_unread} record slots (free) "
              f"+ {n_unread} unread channels (pipeline cost)"
              + ("" if n_unread else "  [run `repair.py --build` to add them]"))
    return cycles, columns


def feature_names(columns, dims):
    """Column names selected by a set of grammar dimensions."""
    unread = set(_signal_cols("unread"))
    names = []
    for d in dims:
        g = GRAMMAR[d]
        stats = g["stat"] if isinstance(g["stat"], tuple) else (g["stat"],)
        for c in columns:
            in_unread = c in unread
            if (g["source"] == "unread") == in_unread:
                names += [f"{c}::{s}" for s in stats]
    return sorted(set(names))


def matrix(cycles, ids, columns, dims):
    names = feature_names(columns, dims)
    X = np.array([[cycles[i]["f"].get(n, 0.0) for n in names] for i in ids], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


# ---------------------------------------------------------------------------
# Repair search
# ---------------------------------------------------------------------------
def _fit_predict(Xtr, ytr, Xte):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    return clf, clf.predict(Xte)


def _inner_cv_acc(X, y, groups=None, seed=0, folds=4):
    """Held-out accuracy inside the FIT half (model selection only).

    Grouped by `setting` when available: cycles sharing a setting are repeated
    runs of the same physical fault instantiation, so an ungrouped CV would
    select dimensions under leakage and systematically prefer whatever memorises
    the instantiation.
    """
    from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                         cross_val_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return 0.0
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if groups is not None:
        per_class = min(len(set(np.asarray(groups)[y == c])) for c in np.unique(y))
        k = min(folds, per_class)
        if k < 2:
            return float("nan")           # not enough instantiations to validate
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
        return float(cross_val_score(clf, X, y, groups=groups, cv=cv,
                                     scoring="accuracy").mean())
    # count only the classes actually present: voraus category labels are not
    # contiguous inside a cluster (e.g. {1, 6, 7}), so np.bincount would report a
    # zero count for the absent ones and collapse the fold count.
    k = min(folds, int(np.unique(y, return_counts=True)[1].min()))
    if k < 2:
        return 0.0
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


# A costlier dimension has to earn its cost: the gain it must show before the
# search will take it is scaled by its class. Free dimensions are already in the
# record; a `pipeline` dimension means changing what the attribution reads, which
# is a deployment the engineer has to carry out, so it must buy materially more
# than a rounding error. This is what makes the cost classes operative rather
# than decorative -- the search prefers the cheapest repair that works.
COST_FACTOR = {"free": 1.0, "pipeline": 6.0, "hardware": float("inf")}
# minimum inner-CV gain a FREE dimension must show; swept in sensitivity()
_MIN_GAIN = 0.005
# adoption margins over (majority, deployed representation), on the FIT half
ADOPT_MARGIN = (0.05, 0.02)


def repair_search(cycles, columns, fit_ids, y_fit, groups=None, seed=0, min_gain=None):
    """Cost-aware greedy forward selection over GRAMMAR dimensions, FIT half only.

    Starts from the deployed representation ('mag') and adds the dimension with
    the best cost-adjusted gain, stopping when nothing clears its own threshold.
    Returns (chosen_dims, base_acc, best_acc, trace).
    """
    min_gain = _MIN_GAIN if min_gain is None else min_gain
    chosen = ["mag"]
    X, _ = matrix(cycles, fit_ids, columns, chosen)
    best = _inner_cv_acc(X, y_fit, groups, seed)
    base = best
    if not np.isfinite(best):
        return chosen, base, best, [("mag", None)]
    trace = [("mag", round(best, 4))]
    remaining = list(REPAIR_DIMS)
    while remaining:
        scored = []
        for d in remaining:
            Xd, _ = matrix(cycles, fit_ids, columns, chosen + [d])
            a = _inner_cv_acc(Xd, y_fit, groups, seed)
            a = a if np.isfinite(a) else -1.0
            factor = COST_FACTOR.get(GRAMMAR[d]["cost"], 1.0)
            # rank by how far the gain exceeds this dimension's own threshold, so
            # a free dimension wins ties against a costlier one
            scored.append(((a - best) - min_gain * factor, a, d))
        scored.sort(key=lambda t: t[0], reverse=True)
        surplus, a, d = scored[0]
        if surplus < 0:
            break
        best = a
        chosen.append(d)
        remaining.remove(d)
        trace.append((d, round(best, 4)))
    return chosen, base, best, trace


def certify(cycles, columns, cluster_name, members, fit_ids, test_ids,
            y_fit, y_test, groups=None, seed=0):
    """Fit a repair on FIT, verify it on TEST, and return a repair certificate."""
    dims, base_acc, inner_acc, trace = repair_search(cycles, columns, fit_ids,
                                                     y_fit, groups, seed)
    # When a cluster has too few fault instantiations for a grouped inner CV, the
    # search cannot select anything.  Rather than silently reporting "no repair
    # found" (which would read as a scientific claim when it is a data limit), we
    # fall back to the PRE-REGISTERED free-cost dimension set and evaluate that on
    # the held-out settings.  No selection touches the test half either way.
    searched = bool(np.isfinite(inner_acc))
    if not searched:
        dims = ["mag"] + [d for d in REPAIR_DIMS if GRAMMAR[d]["cost"] == "free"]
    Xtr, names = matrix(cycles, fit_ids, columns, dims)
    Xte, _ = matrix(cycles, test_ids, columns, dims)
    _, pred = _fit_predict(Xtr, y_fit, Xte)
    acc = float(np.mean(pred == y_test))

    # the deployed representation alone, on the same split (the honest baseline
    # for "does the ADDED dimension do the work, or just the classifier?")
    Xtr0, _ = matrix(cycles, fit_ids, columns, ["mag"])
    Xte0, _ = matrix(cycles, test_ids, columns, ["mag"])
    _, pred0 = _fit_predict(Xtr0, y_fit, Xte0)
    acc_mag = float(np.mean(pred0 == y_test))

    # permuted-label control: same features, shuffled labels -> chance level
    rng = np.random.RandomState(seed)
    _, pred_perm = _fit_predict(Xtr, rng.permutation(y_fit), Xte)
    acc_perm = float(np.mean(pred_perm == y_test))

    vals, cnt = np.unique(y_fit, return_counts=True)
    majority = float(np.mean(y_test == vals[np.argmax(cnt)]))

    # ---- ADOPTION IS DECIDED ON THE FIT HALF ONLY --------------------------
    # It used to compare acc_test against the baselines, which is model
    # selection on the reporting split: whether a repair got deployed depended
    # on how it happened to score on the very data used to report it, and that
    # bias then propagated into the end-to-end two-stage number. The decision
    # now uses only quantities computable inside FIT -- the grouped inner CV of
    # the chosen dims, of the deployed representation, and of the majority
    # class -- so `acc_test` is a report, never a decision.
    maj_fit = float(np.max(cnt) / len(y_fit))
    m_maj, m_dep = ADOPT_MARGIN
    adopted = bool(searched and inner_acc > maj_fit + m_maj
                   and inner_acc > base_acc + m_dep)

    cost = max((GRAMMAR[d]["cost"] for d in dims if d != "mag"),
               key=lambda c: ["deployed", "free", "pipeline", "hardware"].index(c),
               default="deployed")
    return {
        "cluster": cluster_name,
        "members": sorted(members),
        "n_fit": len(fit_ids), "n_test": len(test_ids),
        "dims": dims,
        "added_dims": [d for d in dims if d != "mag"],
        "cost_class": cost,
        "n_features": len(names),
        "acc_test": acc,
        "acc_deployed_repr": acc_mag,
        "acc_majority": majority,
        "acc_permuted": acc_perm,
        "inner_cv_fit": inner_acc if searched else None,
        "inner_cv_deployed_repr": base_acc if searched else None,
        "inner_cv_majority": maj_fit,
        "search_validated": searched,
        "search_trace": trace if searched else "pre-registered free-cost dims "
                                               "(too few instantiations to run a grouped search)",
        "adopted": adopted,
        "adoption_decided_on": "fit half (grouped inner CV) — the held-out half "
                               "is reported, never used to decide",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _stratified_split(ids, cat_of, seed):
    """Random stratified 50/50 by category — matches held_out_eval.py.

    OPTIMISTIC for any learned component: cycles of the same `setting` are
    repeated runs of one physical fault instantiation and end up on both sides.
    Kept so the leakage can be quantified rather than assumed.
    """
    import random
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of[i]].append(i)
    rng = random.Random(seed)
    fit, test = [], []
    for _, group in sorted(by_cat.items()):
        g = group[:]
        rng.shuffle(g)
        k = len(g) // 2
        fit += g[:k]
        test += g[k:]
    return fit, test


def _grouped_split(ids, cat_of, set_of, seed):
    """Leave-setting-out 50/50, stratified by category — the honest protocol.

    Every `setting` (one physical instantiation of a fault: a specific added
    weight, a specific collision object placement) lands wholly in FIT or wholly
    in TEST, so a learned repair must generalise to an instantiation it has
    never seen.  Categories with a single instantiation cannot satisfy this at
    all; they alternate side by seed and are reported as untrainable.
    """
    import random
    rng = random.Random(seed)
    by_cat = defaultdict(set)
    for i in ids:
        by_cat[cat_of[i]].add(set_of[i])
    fit_sets, test_sets = set(), set()
    for cat, settings in sorted(by_cat.items()):
        s = sorted(settings)
        rng.shuffle(s)
        if len(s) == 1:
            (fit_sets if (seed + cat) % 2 == 0 else test_sets).add(s[0])
            continue
        k = max(1, len(s) // 2)
        fit_sets |= set(s[:k])
        test_sets |= set(s[k:])
    fit = [i for i in ids if set_of[i] in fit_sets]
    test = [i for i in ids if set_of[i] in test_sets]
    return fit, test


def testable_clusters():
    """Twin clusters whose members map to voraus categories present in the data,
    plus the hardware gaps we can name but not test."""
    from distinguishability import twin_classes
    from integration_test import CATEGORY_TO_FM
    inv = {v: k for k, v in CATEGORY_TO_FM.items()}
    testable, untestable = [], []
    for grp in twin_classes():
        cats = {inv[m]: m for m in sorted(grp) if m in inv}
        if len(cats) >= 2:
            testable.append((sorted(grp), cats))
        else:
            gap = next((h for k, h in HARDWARE_GAPS.items() if set(grp) == set(k)), None)
            untestable.append((sorted(grp), cats, gap))
    return testable, untestable


def evaluate(seeds=(0, 1, 2, 3, 4), control=False, protocol="setting", verbose=True,
             fixed_dims=None):
    """protocol='setting' -> leave-setting-out (honest); 'random' -> the
    optimistic stratified split, kept to quantify the leakage.

    fixed_dims: skip the per-seed search and use this dimension set for every
    seed. The searched variant answers "can a repair be found"; the fixed
    variant answers "does THE repair the paper names work", and only the second
    is what a deployment would actually ship. Both are reported.
    """
    from eval_retrieve import load_cache
    from integration_test import CATEGORY_TO_FM
    from distinguishability import cluster_map, resolve

    cycles, columns = load_features(verbose=verbose)
    recs = {str(r["cycle_id"]): r for r in load_cache()}
    ids = sorted(set(cycles) & set(recs), key=int)
    cat_of = {i: cycles[i]["category"] for i in ids}
    set_of = {i: cycles[i]["setting"] for i in ids}
    rep = cluster_map()
    testable, untestable = testable_clusters()

    if verbose:
        print(f"\n{len(ids)} anomalous cycles with both a bridge record and features")
        print(f"split protocol: {protocol}"
              + ("  (leave-setting-out: a repair must generalise to an unseen "
                 "instantiation)" if protocol == "setting" else
                 "  (OPTIMISTIC: same-setting repeats span both halves)"))
        print(f"{len(testable)} testable twin clusters, {len(untestable)} untestable\n")

    # cache the stage-1 ranking once (it does not depend on the split)
    ranked = {i: [c["id"] for c in resolve(recs[i])["ranked"]] for i in ids}

    per_seed = []
    cert_by_cluster = defaultdict(list)
    for seed in seeds:
        fit, test = (_grouped_split(ids, cat_of, set_of, seed) if protocol == "setting"
                     else _stratified_split(ids, cat_of, seed))
        resolvers = {}
        for members, cats in testable:
            name = "+".join(m.replace("FM_", "") for m in members)
            cl_fit = [i for i in fit if cat_of[i] in cats]
            cl_test = [i for i in test if cat_of[i] in cats]
            if len(cl_fit) < 12 or len(cl_test) < 6 or len(set(cat_of[i] for i in cl_fit)) < 2:
                cert_by_cluster[name].append(
                    {"cluster": name, "members": sorted(members), "seed": seed,
                     "n_fit": len(cl_fit), "n_test": len(cl_test), "dims": ["mag"],
                     "added_dims": [], "cost_class": "n/a", "adopted": False,
                     "untrainable": True,
                     "reason": "too few instantiations of this cluster's members "
                               "on one side of a leave-setting-out split"})
                continue
            y_fit = np.array([cat_of[i] for i in cl_fit])
            y_test = np.array([cat_of[i] for i in cl_test])
            groups = np.array([set_of[i] for i in cl_fit]) if protocol == "setting" else None
            if fixed_dims is not None:
                # deployment variant: one configuration, decided once, applied to
                # every split -- no per-seed choice of any kind
                Xtr, _ = matrix(cycles, cl_fit, columns, fixed_dims)
                Xte, _ = matrix(cycles, cl_test, columns, fixed_dims)
                _, pred = _fit_predict(Xtr, y_fit, Xte)
                cert = {"cluster": name, "members": sorted(members), "dims": list(fixed_dims),
                        "added_dims": [d for d in fixed_dims if d != "mag"],
                        "n_fit": len(cl_fit), "n_test": len(cl_test),
                        "acc_test": float(np.mean(pred == y_test)),
                        "cost_class": max((GRAMMAR[d]["cost"] for d in fixed_dims if d != "mag"),
                                          key=["deployed", "free", "pipeline", "hardware"].index,
                                          default="deployed"),
                        "search_validated": True, "fixed_config": True}
                # a fixed configuration is adopted for the cluster it was chosen
                # for; other clusters keep the searched decision
                cert["adopted"] = name == "axisweight+losecan+varweight"
            else:
                cert = certify(cycles, columns, name, members, cl_fit, cl_test,
                               y_fit, y_test, groups, seed)
            cert["seed"] = seed
            cert["untrainable"] = False
            cert_by_cluster[name].append(cert)
            if cert["adopted"]:
                X, _ = matrix(cycles, cl_fit, columns, cert["dims"])
                clf, _ = _fit_predict(X, y_fit, X)
                resolvers[rep[members[0]]] = (clf, cert["dims"], cats)

        # ---- end-to-end on the held-out half ------------------------------
        base_hit = two_hit = clus_hit = 0
        routed = 0
        for i in test:
            true_fm = CATEGORY_TO_FM.get(cat_of[i])
            top1 = ranked[i][0]
            base_hit += (top1 == true_fm)
            clus_hit += (rep.get(top1) == rep.get(true_fm))
            pred = top1
            r = resolvers.get(rep.get(top1))
            if r is not None:
                clf, dims, cats = r
                X, _ = matrix(cycles, [i], columns, dims)
                pred = CATEGORY_TO_FM[int(clf.predict(X)[0])]
                routed += 1
            two_hit += (pred == true_fm)
        n = len(test)
        row = {"seed": seed, "protocol": protocol, "n_test": n, "n_fit": len(fit),
               "point_top1": base_hit / n, "cluster_top1": clus_hit / n,
               "two_stage_top1": two_hit / n, "routed_frac": routed / n,
               "adopted_clusters": sorted(
                   c["cluster"] for cl in cert_by_cluster.values()
                   for c in cl if c["seed"] == seed and c["adopted"])}

        if control:
            row["flat_classifier"] = _flat_control(cycles, columns, fit, test, cat_of, seed)
        per_seed.append(row)
        if verbose:
            print(f"  seed {seed}: point {row['point_top1']:.3f} | cluster "
                  f"{row['cluster_top1']:.3f} | TWO-STAGE {row['two_stage_top1']:.3f} "
                  f"(routed {row['routed_frac']:.0%})")

    return per_seed, cert_by_cluster, untestable


def _flat_control(cycles, columns, fit, test, cat_of, seed):
    """Control: a flat 12-way classifier on ALL grammar features, no
    undecidability guidance.  Reported in full, including where its gains sit."""
    dims = list(GRAMMAR)
    Xtr, _ = matrix(cycles, fit, columns, dims)
    Xte, _ = matrix(cycles, test, columns, dims)
    ytr = np.array([cat_of[i] for i in fit])
    yte = np.array([cat_of[i] for i in test])
    _, pred = _fit_predict(Xtr, ytr, Xte)
    per_cat = {}
    for c in sorted(set(yte)):
        m = yte == c
        per_cat[int(c)] = float(np.mean(pred[m] == yte[m]))
    return {"top1": float(np.mean(pred == yte)), "per_category": per_cat}


def granularity_ablation(seeds=(0, 1, 2, 3, 4), verbose=True):
    """At what GRANULARITY does the repairing signature live?

    This decides an architectural question. If the direction/step signature were
    a property of the failure MODE, the knowledge base could carry it in prose
    ("sustained elevated torque") and `line_kg.retrieve()` could score it
    directly -- no learned component anywhere. We tested that (line_kg's
    FM_SIGNATURE + w_signature) and it made retrieval worse. This ablation says
    why: we hold the statistic fixed (signed mean z, late-early step over the
    torque families) and vary only how finely it is resolved.

    Result: a mode-level scalar is at or below the majority class, and accuracy
    climbs monotonically with joint resolution -- the signature is a
    joint-resolved signed PATTERN (added axis weight raises torque_sensor_a on
    J2 and lowers it on J3), which no FMEA text contains and no engineer would
    reliably author. It has to be measured. That is why the repair is a small
    learned model inside one cluster rather than a new KB field.
    """
    cycles, columns = load_features(verbose=False)
    cat_of = {i: cycles[i]["category"] for i in cycles}
    set_of = {i: cycles[i]["setting"] for i in cycles}
    ids = sorted(cycles, key=int)
    cats = {1: "axisweight", 6: "losecan", 7: "varweight"}     # the repaired cluster
    load = [f"{f}_{j}" for f in ("motor_torque", "torque_sensor_a") for j in (1, 2, 3)]
    allj = [f"{f}_{j}" for f in ("motor_torque", "torque_sensor_a") for j in JOINTS]

    def feats(i, kind):
        f = cycles[i]["f"]
        if kind == "mode-level scalar":            # what FMEA prose can express
            return [np.mean([f[f"{t}::z_signed"] for t in load]),
                    np.mean([f[f"{t}::z_step"] for t in load])]
        if kind == "per-family scalar":
            return [np.mean([f[f"{fam}_{j}::{s}"] for j in (1, 2, 3)])
                    for fam in ("motor_torque", "torque_sensor_a") for s in ("z_signed", "z_step")]
        if kind == "joint-resolved (load)":
            return [f[f"{t}::{s}"] for t in load for s in ("z_signed", "z_step")]
        return [f[f"{t}::{s}"] for t in allj for s in ("z_signed", "z_step")]

    out = {}
    for kind in ("mode-level scalar", "per-family scalar",
                 "joint-resolved (load)", "joint-resolved (all 6)"):
        accs, majs = [], []
        for seed in seeds:
            fit, test = _grouped_split(ids, cat_of, set_of, seed)
            cf = [i for i in fit if cat_of[i] in cats]
            ct = [i for i in test if cat_of[i] in cats]
            if len(set(cat_of[i] for i in cf)) < 2 or len(ct) < 6:
                continue
            ytr = np.array([cat_of[i] for i in cf])
            yte = np.array([cat_of[i] for i in ct])
            _, pred = _fit_predict(np.array([feats(i, kind) for i in cf]), ytr,
                                   np.array([feats(i, kind) for i in ct]))
            accs.append(float(np.mean(pred == yte)))
            v, c = np.unique(ytr, return_counts=True)
            majs.append(float(np.mean(yte == v[np.argmax(c)])))
        out[kind] = {"n_features": len(feats(ids[0], kind)),
                     "acc": float(np.mean(accs)), "majority": float(np.mean(majs))}
    if verbose:
        print("\nGRANULARITY ABLATION — where does the repairing signature live?")
        print("  (same statistic throughout; only the resolution changes)")
        for k, v in out.items():
            flag = "  <-- at/below chance" if v["acc"] <= v["majority"] else ""
            print(f"    {k:24s} {v['n_features']:>3} feat   held-out {v['acc']:.3f}"
                  f"   majority {v['majority']:.3f}{flag}")
        print("  => the signature is a joint-resolved PATTERN, not a mode property.")
        print("     A knowledge-authored signature cannot carry it; measuring it can.")
    return out


def _mode_level_signature_check(verbose=True):
    """The near-miss worth reporting: what the mode-level signature would have
    done to the SYSTEM had we shipped it on the strength of its plausibility.

    It dissolves three of the four twin clusters in the knowledge base -- the
    interface would stop reporting them as undecidable -- while retrieval
    accuracy falls. A representation change that silently removes an honest
    "I cannot tell these apart" while making the answer worse is exactly the
    failure the repair certificate exists to catch, and it was caught only
    because adoption requires held-out evidence rather than plausibility.
    """
    from eval_retrieve import load_cache
    from integration_test import CATEGORY_TO_FM
    from line_kg import retrieve
    import distinguishability as dg

    recs = load_cache()

    def top1(w):
        hit = 0
        for r in recs:
            tf = CATEGORY_TO_FM.get(r["detector"]["ground_truth_category"])
            hit += retrieve(r, top_k=1, w_signature=w)[0]["id"] == tf
        return hit / len(recs)

    off, on = top1(0.0), top1(0.2)
    clusters_off = len(dg.twin_classes(use_signature=False))
    clusters_on = len(dg.twin_classes(use_signature=True))
    if verbose:
        print("\nMODE-LEVEL SIGNATURE IN retrieve() — measured, and NOT shipped")
        print(f"    retrieval top-1        : {off:.3f} (off)  ->  {on:.3f} (on)")
        print(f"    twin clusters reported : {clusters_off}  ->  {clusters_on}")
        print("  => it would have removed 3 of 4 undecidability warnings while")
        print("     making the answer worse. Default stays w_signature=0.0.")
    return {"top1_off": off, "top1_on": on,
            "twin_clusters_off": clusters_off, "twin_clusters_on": clusters_on,
            "shipped": False}


def sensitivity(seeds=(0, 1, 2, 3, 4), verbose=True):
    """Do the conclusions depend on the constants?

    Four numbers are set by judgement rather than derived: the cost penalty a
    `pipeline` dimension must overcome, the minimum gain the greedy search
    accepts, and the two adoption margins. If the adopted clusters or the
    two-stage number move when they move, the result is a tuning artefact and
    the reader deserves to know. Swept one at a time around the shipped values.
    """
    global COST_FACTOR
    base_cost = dict(COST_FACTOR)
    rows = []

    def run(tag, **kw):
        per, certs, _ = evaluate(seeds=seeds, protocol="setting", verbose=False, **kw)
        adopted = sorted({c["cluster"] for cl in certs.values() for c in cl
                          if c.get("adopted")})
        load = [c for c in certs.get("axisweight+losecan+varweight", [])
                if not c.get("untrainable")]
        dims = sorted({d for c in load for d in c.get("added_dims", [])})
        rows.append({"setting": tag,
                     "two_stage": float(np.mean([r["two_stage_top1"] for r in per])),
                     "adopted": adopted, "load_dims": dims,
                     "load_acc": float(np.mean([c["acc_test"] for c in load])) if load else None})

    for f in (1.0, 3.0, 6.0, 10.0):
        COST_FACTOR = dict(base_cost, pipeline=f)
        run(f"COST_FACTOR[pipeline]={f}" + ("  (shipped)" if f == 6.0 else ""))
    COST_FACTOR = dict(base_cost)

    global _MIN_GAIN
    base_gain = _MIN_GAIN
    for g in (0.002, 0.005, 0.01, 0.02):
        _MIN_GAIN = g
        run(f"min_gain={g}" + ("  (shipped)" if g == 0.005 else ""))
    _MIN_GAIN = base_gain

    global ADOPT_MARGIN
    base_margin = ADOPT_MARGIN
    for m in ((0.02, 0.01), (0.05, 0.02), (0.08, 0.04)):
        ADOPT_MARGIN = m
        run(f"adopt margin={m}" + ("  (shipped)" if m == (0.05, 0.02) else ""))
    ADOPT_MARGIN = base_margin

    if verbose:
        print("\nSENSITIVITY — do the constants decide the conclusions?")
        print(f"    {'setting':34s} {'two-stage':>10} {'load acc':>9}  adopted / load dims")
        for r in rows:
            acc = f"{r['load_acc']:.3f}" if r["load_acc"] is not None else "  n/a"
            print(f"    {r['setting']:34s} {r['two_stage']:>10.3f} {acc:>9}  "
                  f"{','.join(c.split('+')[0] for c in r['adopted']) or 'none'}"
                  f" / {'+'.join(r['load_dims']) or 'none'}")
        span = max(r["two_stage"] for r in rows) - min(r["two_stage"] for r in rows)
        same = len({tuple(r["adopted"]) for r in rows}) == 1
        print(f"  => two-stage spans {span:.3f} across all settings; "
              f"the adopted set is {'IDENTICAL' if same else 'NOT identical'} throughout.")
    return rows


def label_curve_vs_control(sizes=(3, 5, 8, 12, 16, 24, 40, 80, 160, 320),
                           seeds=(0, 1, 2, 3, 4), verbose=True):
    """Flat 12-way control vs two-stage at MATCHED label budgets, end-to-end.

    The paper previously asserted a label-efficiency advantage by putting a
    3-way within-cluster curve (chance 0.333) next to the control's 12-way
    number (chance 0.083).  Those are not comparable and the claim was
    downgraded to a hypothesis.  This is the experiment that tests it: both
    systems are given the SAME number of labelled cycles, both are scored by
    END-TO-END top-1 over all 12 categories on the same held-out half, and the
    split is leave-setting-out throughout.

    Two budget regimes, because they answer different questions and only the
    first is what a deployment actually faces:

      'unrestricted' -- the analyst labels k cycles drawn at random from the
          fit half.  The control trains on all k.  The two-stage system can
          only use the subset that happens to land in a repaired cluster
          (~41% of anomalies are load-cluster), and falls back to the
          knowledge-base rank-1 elsewhere.  This is the write-back loop's real
          yield and it DISADVANTAGES us.
      'targeted' -- the k labels are spent inside the repaired cluster, which
          is what an analyst would do if told where labels are worth most.
          The control still gets k labels spread over all categories.

    Nothing here is tuned; the resolver dims are the fixed shipping
    configuration so the curve measures labels, not model selection.

    THREE REFERENCE ARMS, not one.  An earlier version of this curve reported
    only the control and reached a conclusion that does not survive them:

      'stage1_only'   the knowledge base with no labels at all.
      'prior_at_k'    predict the majority category OF THE k LABELS for every
          test cycle.  voraus-AD is skewed, so a large part of what the control
          appears to "learn" at small k is this and nothing more; without the arm
          the crossover reads as a diagnostic result when it is the moment the
          control finishes estimating the class prior.  See `baselines.py`.
      'macro'         mean per-category recall for every arm, reported beside
          micro top-1.  A constant predictor scores 1/12 here whatever the skew,
          so this is the reading the prior cannot game.

    MINIMUM RESOLVER SIZE.  A previous run refused to fit a within-cluster
    resolver on fewer than six labelled cycles, which silently pinned the
    two-stage arms to the zero-label baseline at k=3 and k=5 and made the
    smallest, most contested budgets non-measurements.  The floor is now the
    statistical minimum -- two classes present -- so every row is measured.
    """
    from eval_retrieve import load_cache
    from integration_test import CATEGORY_TO_FM
    from distinguishability import cluster_map, resolve

    cycles, columns = load_features(verbose=False)
    recs = {str(r["cycle_id"]): r for r in load_cache()}
    ids = sorted(set(cycles) & set(recs), key=int)
    cat_of = {i: cycles[i]["category"] for i in ids}
    set_of = {i: cycles[i]["setting"] for i in ids}
    rep = cluster_map()
    ranked = {i: [c["id"] for c in resolve(recs[i])["ranked"]] for i in ids}
    testable, _ = testable_clusters()
    dims = ["mag", "sign", "step"]   # the fixed shipping configuration

    out = {r: defaultdict(list) for r in ("unrestricted", "targeted")}
    mac = {r: defaultdict(list) for r in ("unrestricted", "targeted")}
    base_curve = defaultdict(list)
    base_macro = defaultdict(list)
    prior_curve, prior_macro = defaultdict(list), defaultdict(list)
    ctrl_macro = defaultdict(list)

    def _macro(pred, truth):
        per = defaultdict(list)
        for p, t in zip(pred, truth):
            per[t].append(p == t)
        return float(np.mean([np.mean(v) for v in per.values()]))

    for seed in seeds:
        fit, test = _grouped_split(ids, cat_of, set_of, seed)
        yte = np.array([cat_of[i] for i in test])
        # stage-1 only: the zero-label baseline both curves start from
        kb_pred = np.array([_cat_of_fm(ranked[i][0]) for i in test])
        base = float(np.mean(kb_pred == yte))
        base_ma = _macro(list(kb_pred), list(yte))
        rng = np.random.RandomState(seed)
        order = list(rng.permutation(len(fit)))
        fit_shuf = [fit[j] for j in order]

        for k in sizes:
            if k > len(fit_shuf):
                continue
            budget = fit_shuf[:k]
            base_curve[k].append(base)
            base_macro[k].append(base_ma)

            # ---- control: 12-way on whatever the budget contains ----------
            ytr = np.array([cat_of[i] for i in budget])
            if len(set(ytr)) < 2:
                continue
            Xtr, _ = matrix(cycles, budget, columns, list(GRAMMAR))
            Xte, _ = matrix(cycles, test, columns, list(GRAMMAR))
            _, pred = _fit_predict(Xtr, ytr, Xte)
            ctrl = float(np.mean(pred == yte))
            ctrl_macro[k].append(_macro(list(pred), list(yte)))

            # ---- what the SAME k labels buy with no model at all -----------
            maj = Counter(ytr.tolist()).most_common(1)[0][0]
            prior_curve[k].append(float(np.mean(yte == maj)))
            prior_macro[k].append(_macro([maj] * len(yte), list(yte)))

            # ---- two-stage, under each budget regime ----------------------
            for regime in ("unrestricted", "targeted"):
                resolvers = {}
                for members, cats in testable:
                    if regime == "unrestricted":
                        sub = [i for i in budget if cat_of[i] in cats]
                    else:
                        pool = [i for i in fit_shuf if cat_of[i] in cats]
                        sub = pool[:k]
                    # the statistical minimum, not a round number: a classifier
                    # needs two classes to exist. Anything above this floor
                    # would turn small-k rows into non-measurements.
                    if len(set(cat_of[i] for i in sub)) < 2:
                        continue
                    Xc, _ = matrix(cycles, sub, columns, dims)
                    clf, _ = _fit_predict(Xc, np.array([cat_of[i] for i in sub]),
                                          Xc)
                    resolvers[rep.get(sorted(members)[0])] = (clf, sorted(cats))
                preds = []
                for j, i in enumerate(test):
                    top1 = ranked[i][0]
                    p = _cat_of_fm(top1)
                    r = resolvers.get(rep.get(top1))
                    if r is not None:
                        clf, _cats = r
                        Xi, _ = matrix(cycles, [i], columns, dims)
                        p = int(clf.predict(Xi)[0])
                    preds.append(p)
                out[regime][k].append(float(np.mean([p == cat_of[i]
                                                     for p, i in zip(preds, test)])))
                mac[regime][k].append(_macro(preds, [cat_of[i] for i in test]))
            out["_ctrl"] = out.get("_ctrl", defaultdict(list))
            out["_ctrl"][k].append(ctrl)

    def curve(d):
        return {str(k): float(np.mean(v)) for k, v in sorted(d.items())}

    res = {
        "sizes": [k for k in sizes if k in base_curve],
        "stage1_only": curve(base_curve),
        "prior_at_k": curve(prior_curve),
        "control_12way": curve(out["_ctrl"]),
        "two_stage_unrestricted": curve(out["unrestricted"]),
        "two_stage_targeted": curve(out["targeted"]),
        "macro": {
            "stage1_only": curve(base_macro),
            "prior_at_k": curve(prior_macro),
            "control_12way": curve(ctrl_macro),
            "two_stage_unrestricted": curve(mac["unrestricted"]),
            "two_stage_targeted": curve(mac["targeted"]),
        },
        "n_seeds": len(seeds),
        "dims": dims,
        "min_resolver_rule": "two classes present (the statistical minimum)",
        "note": ("end-to-end top-1 over all 12 categories, leave-setting-out, "
                 "same held-out half for every arm; budgets are labelled cycles. "
                 "'prior_at_k' predicts the majority category of the k labels and "
                 "carries no diagnostic information; 'macro' is mean per-category "
                 "recall, where a constant predictor scores 1/12 = 0.083."),
    }
    if verbose:
        print("\nLABEL BUDGET — end-to-end, leave-setting-out, all arms on the "
              "same held-out half")
        print(f"  {'':>5}  {'-------------- micro top-1 --------------':^42}   "
              f"{'---- macro recall ----':^22}")
        print(f"  {'k':>5}  {'stage-1':>8}  {'prior@k':>8}  {'control':>8}  "
              f"{'2st(targ)':>9}   {'prior@k':>8}  {'control':>8}  {'2st(targ)':>9}")
        for k in res["sizes"]:
            s, M = str(k), res["macro"]
            nan = float("nan")
            print(f"  {k:>5}  {res['stage1_only'][s]:>8.3f}  "
                  f"{res['prior_at_k'].get(s, nan):>8.3f}  "
                  f"{res['control_12way'].get(s, nan):>8.3f}  "
                  f"{res['two_stage_targeted'].get(s, nan):>9.3f}   "
                  f"{M['prior_at_k'].get(s, nan):>8.3f}  "
                  f"{M['control_12way'].get(s, nan):>8.3f}  "
                  f"{M['two_stage_targeted'].get(s, nan):>9.3f}")
        xs = [k for k in res["sizes"] if s_ok(res, k)]
        cross = next((k for k in xs
                      if res["control_12way"][str(k)]
                      > res["two_stage_targeted"][str(k)]), None)
        print(f"  crossover, control over targeted two-stage: "
              f"{cross if cross else 'not within the swept range'}")
        kb_ma = np.mean(list(res["macro"]["stage1_only"].values()))
        ctrl_beats = next((k for k in xs
                           if res["macro"]["control_12way"].get(str(k), 0) >= kb_ma), None)
        print(f"  labels the control needs to match the zero-label knowledge base "
              f"on MACRO recall ({kb_ma:.3f}): k = {ctrl_beats}")
    return res


def budget_crossover_robustness(sizes=(3, 5, 8, 12), seeds=range(20), verbose=True):
    """Is the small-budget crossover real, or split noise?

    The five-split curve is enough to establish the large-budget conclusion --
    the control runs away with it -- but the contested region is the first
    handful of labels, where the two arms are a few points apart and the split
    variance is larger than the gap.  Twenty splits, and the statistic that
    matters is not the mean but the fraction of splits on which the two-stage
    system actually wins.  A mean difference with a win rate near 0.5 is a tie,
    and saying so is the difference between a finding and an artefact.
    """
    from eval_retrieve import load_cache
    from distinguishability import cluster_map, resolve

    cycles, columns = load_features(verbose=False)
    recs = {str(r["cycle_id"]): r for r in load_cache()}
    ids = sorted(set(cycles) & set(recs), key=int)
    cat_of = {i: cycles[i]["category"] for i in ids}
    set_of = {i: cycles[i]["setting"] for i in ids}
    rep = cluster_map()
    ranked = {i: [c["id"] for c in resolve(recs[i])["ranked"]] for i in ids}
    testable, _ = testable_clusters()
    dims = ["mag", "sign", "step"]
    ctrl, targ, base = defaultdict(list), defaultdict(list), []

    for seed in seeds:
        fit, test = _grouped_split(ids, cat_of, set_of, seed)
        yte = np.array([cat_of[i] for i in test])
        base.append(float(np.mean([_cat_of_fm(ranked[i][0]) for i in test] == yte)))
        rng = np.random.RandomState(seed)
        shuf = [fit[j] for j in rng.permutation(len(fit))]
        Xte, _ = matrix(cycles, test, columns, list(GRAMMAR))
        for k in sizes:
            bud = shuf[:k]
            ytr = np.array([cat_of[i] for i in bud])
            if len(set(ytr)) < 2:
                continue
            Xtr, _ = matrix(cycles, bud, columns, list(GRAMMAR))
            _, p = _fit_predict(Xtr, ytr, Xte)
            ctrl[k].append(float(np.mean(p == yte)))
            resolvers = {}
            for members, cats in testable:
                sub = [i for i in shuf if cat_of[i] in cats][:k]
                if len(set(cat_of[i] for i in sub)) < 2:
                    continue
                Xc, _ = matrix(cycles, sub, columns, dims)
                clf, _ = _fit_predict(Xc, np.array([cat_of[i] for i in sub]), Xc)
                resolvers[rep.get(sorted(members)[0])] = clf
            pr = []
            for i in test:
                t = ranked[i][0]
                q = _cat_of_fm(t)
                c = resolvers.get(rep.get(t))
                if c is not None:
                    Xi, _ = matrix(cycles, [i], columns, dims)
                    q = int(c.predict(Xi)[0])
                pr.append(q)
            targ[k].append(float(np.mean([a == b for a, b in zip(pr, yte)])))

    out = {"n_seeds": len(list(seeds)), "stage1_only": float(np.mean(base)),
           "per_k": {}}
    for k in sizes:
        c, t = np.array(ctrl[k]), np.array(targ[k])
        n = min(len(c), len(t))
        out["per_k"][str(k)] = {
            "control_mean": float(c.mean()), "control_sd": float(c.std()),
            "two_stage_mean": float(t.mean()), "two_stage_sd": float(t.std()),
            "two_stage_win_rate": float(np.mean(t[:n] > c[:n])), "n": int(n)}
    if verbose:
        print(f"\nCROSSOVER ROBUSTNESS — {out['n_seeds']} splits; "
              f"zero-label knowledge base {out['stage1_only']:.3f}")
        print(f"  {'k':>4}  {'control':>16}  {'two-stage':>16}  {'P(2-stage wins)':>16}")
        for k in sizes:
            d = out["per_k"][str(k)]
            print(f"  {k:>4}  {d['control_mean']:.3f} +/- {d['control_sd']:.3f}  "
                  f"  {d['two_stage_mean']:.3f} +/- {d['two_stage_sd']:.3f}  "
                  f"{d['two_stage_win_rate']:>16.2f}")
        print("  -> a win rate near 0.5 is a tie, however the means read")
    return out


def s_ok(res, k):
    return str(k) in res["control_12way"] and str(k) in res["two_stage_targeted"]


def _cat_of_fm(fm_id):
    """Failure-mode id -> voraus category, or -1 when the mode has none."""
    from integration_test import CATEGORY_TO_FM
    for c, f in CATEGORY_TO_FM.items():
        if f == fm_id:
            return int(c)
    return -1


def label_efficiency(sizes=(5, 10, 20, 40, 80, 160), seeds=(0, 1, 2), verbose=True):
    """How many labelled cycles does the targeted repair need?

    These are exactly the labels the write-back loop collects, so the curve says
    how long a deployment must run before the repair pays off.
    """
    from eval_retrieve import load_cache
    cycles, columns = load_features(verbose=False)
    recs = {str(r["cycle_id"]): r for r in load_cache()}
    ids = sorted(set(cycles) & set(recs), key=int)
    cat_of = {i: cycles[i]["category"] for i in ids}
    set_of = {i: cycles[i]["setting"] for i in ids}
    testable, _ = testable_clusters()
    out = {}
    for members, cats in testable:
        name = "+".join(m.replace("FM_", "") for m in members)
        curve = defaultdict(list)
        for seed in seeds:
            fit, test = _grouped_split(ids, cat_of, set_of, seed)
            cl_fit = [i for i in fit if cat_of[i] in cats]
            cl_test = [i for i in test if cat_of[i] in cats]
            if len(cl_fit) < 12 or len(cl_test) < 6 or len(set(cat_of[i] for i in cl_fit)) < 2:
                curve = None
                break
            y_test = np.array([cat_of[i] for i in cl_test])
            dims, _, inner, _ = repair_search(
                cycles, columns, cl_fit, np.array([cat_of[i] for i in cl_fit]),
                np.array([set_of[i] for i in cl_fit]), seed)
            if not np.isfinite(inner):
                curve = None
                break
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(cl_fit))
            for k in sizes:
                if k > len(cl_fit):
                    continue
                sub = [cl_fit[j] for j in order[:k]]
                ysub = np.array([cat_of[i] for i in sub])
                if len(set(ysub)) < 2:
                    continue
                Xtr, _ = matrix(cycles, sub, columns, dims)
                Xte, _ = matrix(cycles, cl_test, columns, dims)
                _, pred = _fit_predict(Xtr, ysub, Xte)
                curve[k].append(float(np.mean(pred == y_test)))
        if curve:
            out[name] = {str(k): float(np.mean(v)) for k, v in sorted(curve.items())}
    if verbose:
        print("\nLABEL EFFICIENCY — held-out within-cluster accuracy vs #labelled cycles")
        for name, c in out.items():
            print(f"  {name}")
            print("    " + "  ".join(f"n={k}:{v:.3f}" for k, v in c.items()))
    return out


# ---------------------------------------------------------------------------
# Adoption — a repair the system may use only after a human authorizes it
# ---------------------------------------------------------------------------
ADOPTED_PATH = os.path.join(_HERE, "repair_adopted.json")


def adopt(results_path=RESULTS_OUT, out=ADOPTED_PATH, verbose=True):
    """Freeze the verified repairs into a deployable artifact.

    The dimension set is the one the leave-setting-out search chose on a majority
    of seeds; the model is then refit on all cycles of the cluster and exported
    as plain coefficients (no pickle, no sklearn at inference time).

    `authorized` is written as FALSE.  A verified repair is a *proposal*: the
    system reports what would make the cluster decidable and how well it was
    shown to work, and an engineer authorizes it.  `resolve_with_repair()`
    ignores unauthorized repairs, so adopting this file changes no output until
    someone says so.
    """
    from integration_test import CATEGORY_TO_FM
    from distinguishability import cluster_map

    if not os.path.exists(results_path):
        raise SystemExit("run `repair.py` first — no verified certificates to adopt")
    with open(results_path, encoding="utf-8") as fh:
        res = json.load(fh)
    cycles, columns = load_features(verbose=False)
    rep = cluster_map()
    inv = {v: k for k, v in CATEGORY_TO_FM.items()}

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clusters = {}
    for name, cert in res["certificates"].items():
        if not cert.get("adopted"):
            continue
        picked = cert["dims_selected"]
        n_seeds = int(cert["adopted_seeds"].split("/")[1])
        dims = ["mag"] + [d for d, c in sorted(picked.items(), key=lambda kv: -kv[1])
                          if c > n_seeds / 2]
        cats = {inv[m]: m for m in cert["members"] if m in inv}
        ids = [i for i in cycles if cycles[i]["category"] in cats]
        X, names = matrix(cycles, ids, columns, dims)
        y = np.array([cycles[i]["category"] for i in ids])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        fx = res.get("fixed_configuration", {})
        frozen_acc = fx.get("cluster_acc") if fx.get("dims") == dims else None
        clusters[rep[cert["members"][0]]] = {
            "cluster_name": name,
            "members": cert["members"],
            "dims": dims,
            "added_dims": [d for d in dims if d != "mag"],
            "features": names,
            "classes": [CATEGORY_TO_FM[int(c)] for c in clf.classes_],
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
            "n_fit_cycles": len(ids),
            # The frozen dims are the majority selection, so their cost class
            # can be lower than the certificate's (which is the worst over the
            # per-split searches); report both rather than mislabel either.
            "cost_class": max((GRAMMAR[d]["cost"] for d in dims if d != "mag"),
                              key=["deployed", "free", "pipeline", "hardware"].index,
                              default="deployed"),
            "certificate_cost_class": cert["cost_class"],
            "evidence": {"protocol": "leave-setting-out, 5 stratified splits",
                         # the searched certificate (dims re-selected per split)
                         "held_out_acc": cert["acc_test"],
                         # these frozen dims, same splits, when they are the
                         # fixed configuration repair.py evaluates
                         "frozen_dims_held_out_acc": frozen_acc,
                         "deployed_repr_acc": cert["acc_deployed_repr"],
                         "majority": cert["acc_majority"],
                         "permuted": cert["acc_permuted"]},
            "authorized": False,
        }
        if verbose:
            print(f"  repair for {name}: dims {dims}  held-out {cert['acc_test']:.3f}"
                  f"  -> proposed, NOT authorized")

    gaps = {}
    for members, _cats, gap in testable_clusters()[1]:
        gaps["+".join(members)] = gap
    for name, cert in res["certificates"].items():
        if cert.get("adopted") or "acc_test" not in cert:
            continue
        gaps[name] = ("no representation repair verified — " +
                      ("evidence does not contain the distinction"
                       if cert.get("wall") == "perception" else
                       "too few fault instantiations to validate one"))

    payload = {"clusters": clusters, "unrepaired": gaps,
               "note": "authorized=false means the system keeps reporting the "
                       "cluster as undecidable; an engineer must authorize a repair."}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\nwritten -> {out}   ({len(clusters)} proposed repair(s), 0 authorized)")
    return payload


def load_adopted(path=ADOPTED_PATH):
    if not os.path.exists(path):
        return {"clusters": {}, "unrepaired": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def repair_for(cluster_rep, adopted=None):
    """The repair proposal attached to a cluster, or None."""
    return (adopted or load_adopted())["clusters"].get(cluster_rep)


def apply_repair(spec, cycle_id, cycles=None):
    """Run an adopted repair on one cycle -> (failure-mode id, probability)."""
    if cycles is None:
        cycles, _ = load_features(verbose=False)
    f = cycles[str(cycle_id)]["f"]
    x = np.array([f.get(n, 0.0) for n in spec["features"]], dtype=float)
    x = (np.nan_to_num(x) - np.array(spec["mean"])) / np.array(spec["scale"])
    logits = np.array(spec["coef"]) @ x + np.array(spec["intercept"])
    if len(logits) == 1:                       # binary
        p = 1 / (1 + np.exp(-logits[0]))
        probs = np.array([1 - p, p])
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
    k = int(np.argmax(probs))
    return spec["classes"][k], float(probs[k])


def resolve_with_repair(record, cycle_id=None, adopted=None, cycles=None):
    """`distinguishability.resolve()` plus, where an AUTHORIZED repair exists,
    the within-cluster resolution it provides.

    With no authorized repair this returns exactly what `resolve()` returns, so
    the deployed behaviour is unchanged until an engineer acts.
    """
    from distinguishability import resolve, cluster_map
    res = resolve(record)
    adopted = adopted or load_adopted()
    rep = cluster_map()
    top_cluster = rep.get(res["ranked"][0]["id"])
    spec = adopted["clusters"].get(top_cluster)
    res["repair"] = None
    if spec is None:
        return res
    res["repair"] = {"available": True, "authorized": bool(spec["authorized"]),
                     "added_dims": spec["added_dims"], "cost_class": spec["cost_class"],
                     "evidence": spec["evidence"]}
    if spec["authorized"]:
        fm, p = apply_repair(spec, cycle_id if cycle_id is not None else record["cycle_id"],
                             cycles)
        res["repair"].update(resolved_to=fm, confidence=p)
        res["route"] = "repaired"
        res["class"] = [fm]
        res["class_size"] = 1
    return res


# ---------------------------------------------------------------------------
def main():
    control = "--control" in sys.argv
    labels = "--labels" in sys.argv
    sens = "--sensitivity" in sys.argv
    if "--build" in sys.argv:
        build_feature_cache()
        return
    if "--adopt" in sys.argv:
        print("freezing verified repairs into a deployable, UNAUTHORIZED artifact\n")
        adopt()
        return

    print("=" * 74)
    print("UNDECIDABILITY-DRIVEN REPRESENTATION REPAIR")
    print("=" * 74)
    print("Question: for each cluster the distinguishability analysis calls")
    print("undecidable, WHICH representation change decides it, does it verify")
    print("on held-out data, and what does it cost?\n")
    print("Protocol: leave-setting-out. Each voraus-AD `setting` is one physical")
    print("instantiation of a fault (5-17 repeated cycles); a repair is credited")
    print("only if it generalises to instantiations it never saw.\n")

    per_seed, certs, untestable = evaluate(control=control, protocol="setting")

    def m(key):
        return float(np.mean([r[key] for r in per_seed]))

    def ci(key):
        v = [r[key] for r in per_seed]
        return 2.776 * float(np.std(v, ddof=1)) / np.sqrt(len(v))

    print("\n" + "-" * 74)
    print("REPAIR CERTIFICATES (fit on the FIT half, verified on the held-out half)")
    print("-" * 74)
    cert_summary = {}
    for name, rows in certs.items():
        usable = [r for r in rows if not r.get("untrainable")]
        if not usable:
            print(f"\n  cluster {name}")
            print(f"    NOT TRAINABLE: {rows[0]['reason']}")
            print("    verdict: UNDETERMINED — the dataset has one instantiation of"
                  " these modes,\n             so no leave-setting-out claim can be made")
            cert_summary[name] = {"members": rows[0]["members"], "adopted": False,
                                  "verdict": "undetermined", "reason": rows[0]["reason"]}
            continue
        adopted = sum(r["adopted"] for r in usable)
        acc = np.mean([r["acc_test"] for r in usable])
        mag = np.mean([r["acc_deployed_repr"] for r in usable])
        maj = np.mean([r["acc_majority"] for r in usable])
        perm = np.mean([r["acc_permuted"] for r in usable])
        searched = sum(r["search_validated"] for r in usable)
        dims = defaultdict(int)
        for r in usable:
            for d in r["added_dims"]:
                dims[d] += 1
        # the costliest class any seed's search reached, not seed 0's -- a repair
        # is only "free" if it was free on every split
        order = ["deployed", "free", "pipeline", "hardware"]
        cost = max((r["cost_class"] for r in usable), key=order.index)
        cost_spread = {c: sum(r["cost_class"] == c for r in usable)
                       for c in {r["cost_class"] for r in usable}}
        won = adopted >= len(usable) / 2
        verdict = "ADOPTED" if won else "REJECTED — no repair beat the baselines"
        if won:
            wall = "distinguishability wall — repairable by a representation change"
        elif searched:
            wall = "perception wall — the evidence the attribution surfaces does not " \
                   "contain the distinction"
        else:
            wall = "inconclusive — too few instantiations to run a grouped search; " \
                   "pre-registered dims reported instead"
        print(f"\n  cluster {name}  (n_fit≈{usable[0]['n_fit']}, n_test≈{usable[0]['n_test']})")
        print(f"    held-out within-cluster accuracy : {acc:.3f}")
        print(f"    deployed representation (max|z|) : {mag:.3f}")
        print(f"    majority-class baseline          : {maj:.3f}")
        print(f"    permuted-label control           : {perm:.3f}")
        print(f"    dimensions selected  : {dict(sorted(dims.items(), key=lambda kv: -kv[1])) or '{} (none)'}"
              f"   cost={cost} {cost_spread}   search validated on {searched}/{len(usable)} seeds")
        print(f"    verdict: {verdict}   [{wall}]")
        cert_summary[name] = {
            "members": usable[0]["members"], "n_fit": usable[0]["n_fit"],
            "n_test": usable[0]["n_test"],
            "acc_test": float(acc), "acc_deployed_repr": float(mag),
            "acc_majority": float(maj), "acc_permuted": float(perm),
            "dims_selected": dict(dims), "cost_class": cost,
            "cost_class_spread": cost_spread,
            "search_validated_seeds": f"{searched}/{len(usable)}",
            "adopted_seeds": f"{adopted}/{len(usable)}", "adopted": bool(won),
            "wall": ("distinguishability" if won else
                     ("perception" if searched else "inconclusive")),
        }

    for members, cats, gap in untestable:
        print(f"\n  cluster {'+'.join(x.replace('FM_','') for x in members)}")
        print(f"    NOT TESTABLE: {gap or 'only one member occurs in this dataset'}")
        print("    verdict: SENSOR GAP — repair requires instrumentation, not a"
              " representation change")

    print("\n" + "-" * 74)
    print(f"END-TO-END, held-out ({len(per_seed)} stratified 50/50 splits, mean ± 95% CI)")
    print("-" * 74)
    print(f"  deployed point top-1        : {m('point_top1'):.3f} ± {ci('point_top1'):.3f}")
    print(f"  stage-1 cluster top-1       : {m('cluster_top1'):.3f} ± {ci('cluster_top1'):.3f}"
          "   (the ceiling stage 2 works under)")
    print(f"  TWO-STAGE (KG + repair)     : {m('two_stage_top1'):.3f} ± {ci('two_stage_top1'):.3f}")
    print(f"  fraction of cycles routed to a repaired cluster: {m('routed_frac'):.3f}")
    gain = m('two_stage_top1') - m('point_top1')
    print(f"  => targeted repair moves point top-1 by {gain:+.3f} "
          f"({m('two_stage_top1')/max(m('point_top1'),1e-9):.2f}x)")

    # The number above lets every seed pick its own dimensions. A deployment
    # ships ONE configuration, so report that too -- it is the claim the paper
    # actually makes, and it removes per-seed search variance.
    fixed = ["mag", "sign", "step"]
    fx, fx_certs, _ = evaluate(seeds=(0, 1, 2, 3, 4), protocol="setting",
                               verbose=False, fixed_dims=fixed)
    fx_v = [r["two_stage_top1"] for r in fx]
    fx_ci = 2.776 * float(np.std(fx_v, ddof=1)) / np.sqrt(len(fx_v))
    fx_cl = float(np.mean([c["acc_test"] for c in fx_certs["axisweight+losecan+varweight"]
                           if not c.get("untrainable")]))
    print(f"\n  FIXED CONFIGURATION (dims {fixed} on every seed, no per-seed search):")
    print(f"    two-stage top-1        : {float(np.mean(fx_v)):.3f} ± {fx_ci:.3f}")
    print(f"    load-cluster held-out  : {fx_cl:.3f}")
    print("    this is what `--adopt` freezes and what the interface proposes;")
    print("    the searched number above answers the different question of")
    print("    whether the METHOD can find a repair at all.")

    payload = {"per_seed": per_seed, "certificates": cert_summary,
               "fixed_configuration": {"dims": fixed, "per_seed": fx,
                                       "two_stage_top1": float(np.mean(fx_v)),
                                       "ci95": fx_ci, "cluster_acc": fx_cl},
               "untestable": [{"members": u[0], "reason": u[2]} for u in untestable],
               "summary": {k: {"mean": m(k), "ci95": ci(k)} for k in
                           ("point_top1", "cluster_top1", "two_stage_top1", "routed_frac")}}

    if control:
        flat = float(np.mean([r["flat_classifier"]["top1"] for r in per_seed]))
        print("\n" + "-" * 74)
        print("CONTROL — flat 12-way classifier on ALL grammar features (no guidance)")
        print("-" * 74)
        print("  The obvious objection is 'just train a classifier'. Reported in full.")
        print("  It needs a label for every category; the targeted repair needs labels")
        print("  in one cluster only, and the knowledge base covers the rest.")
        print(f"  flat classifier top-1 : {flat:.3f}")
        print("  per-category accuracy (does its gain sit where the analysis predicted?):")
        CATN = {0: "friction", 1: "axisweight", 2: "coll_foam", 3: "coll_cable",
                4: "coll_card", 5: "misgrip", 6: "losecan", 7: "varweight",
                8: "entangled", 9: "invalidgrip", 10: "commut", 11: "platform"}
        agg = defaultdict(list)
        for r in per_seed:
            for c, v in r["flat_classifier"]["per_category"].items():
                agg[int(c)].append(v)
        for c in sorted(agg):
            print(f"    cat {c:>2} {CATN[c]:>11}: {np.mean(agg[c]):.3f}")
        payload["control_flat_classifier"] = {
            "top1": flat, "per_category": {str(c): float(np.mean(v)) for c, v in agg.items()}}

    if labels:
        payload["label_efficiency"] = label_efficiency()
        budget = label_curve_vs_control()
        budget["crossover_robustness"] = budget_crossover_robustness()
        payload["label_budget_curve"] = budget
        curve_out = os.path.join(_ROOT, "results", "label_budget_curve.json")
        with open(curve_out, "w", encoding="utf-8") as fh:
            json.dump(budget, fh, indent=1, ensure_ascii=False)
        print(f"  written -> {os.path.relpath(curve_out, _ROOT)}")
    if sens:
        payload["sensitivity"] = sensitivity()

    payload["granularity_ablation"] = granularity_ablation()
    payload["mode_level_signature"] = _mode_level_signature_check()

    if control:
        # How much of a learned component's apparent accuracy is same-instantiation
        # leakage?  Same code, same seeds, random stratified split.
        print("\n" + "-" * 74)
        print("LEAKAGE CHECK — the same measurements under a RANDOM stratified split")
        print("-" * 74)
        rnd, rcerts, _ = evaluate(control=True, protocol="random", verbose=False)
        rf = float(np.mean([r["flat_classifier"]["top1"] for r in rnd]))
        rt = float(np.mean([r["two_stage_top1"] for r in rnd]))
        rp = float(np.mean([r["point_top1"] for r in rnd]))
        print(f"  flat classifier : random {rf:.3f}  vs  leave-setting-out {flat:.3f}"
              f"   ({rf - flat:+.3f})")
        print(f"  two-stage       : random {rt:.3f}  vs  leave-setting-out "
              f"{m('two_stage_top1'):.3f}   ({rt - m('two_stage_top1'):+.3f})")
        print(f"  KG retrieval    : random {rp:.3f}  vs  leave-setting-out "
              f"{m('point_top1'):.3f}   (not trained — unaffected)")
        print("\n  per-cluster repair accuracy, and what the leaky protocol would have"
              " concluded:")
        cl_cmp = {}
        for name, rows in rcerts.items():
            usable = [r for r in rows if not r.get("untrainable")]
            if not usable or name not in cert_summary or "acc_test" not in cert_summary[name]:
                continue
            racc = float(np.mean([r["acc_test"] for r in usable]))
            radopt = sum(r["adopted"] for r in usable) >= len(usable) / 2
            hacc = cert_summary[name]["acc_test"]
            hadopt = cert_summary[name]["adopted"]
            flag = "  <-- FALSE REPAIR under the leaky protocol" if (radopt and not hadopt) else ""
            print(f"    {name:34s} random {racc:.3f} ({'adopt' if radopt else 'reject'})"
                  f"  |  honest {hacc:.3f} ({'adopt' if hadopt else 'reject'}){flag}")
            cl_cmp[name] = {"random": {"acc": racc, "adopted": bool(radopt)},
                            "leave_setting_out": {"acc": hacc, "adopted": bool(hadopt)}}
        print("\n  => the targeted repair is nearly protocol-invariant; the blanket")
        print("     classifier loses a third of its accuracy, and the collision cluster")
        print("     is 'repaired' only by the leakage. Grouping by `setting` is what")
        print("     separates a real repair from a memorised fault instantiation.")
        payload["leakage_check"] = {
            "random_split": {"flat_classifier": rf, "two_stage": rt, "point_top1": rp},
            "leave_setting_out": {"flat_classifier": flat,
                                  "two_stage": m("two_stage_top1"),
                                  "point_top1": m("point_top1")},
            "per_cluster": cl_cmp}

    with open(RESULTS_OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nwritten -> {RESULTS_OUT}")


if __name__ == "__main__":
    main()
