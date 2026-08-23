# -*- coding: utf-8 -*-
"""Diagnostic distinguishability — the system knows what it cannot know.

CORE IDEA (the paper's algorithmic novelty, understandable to anyone):
A ranked list pretends the top-1 is the answer. But some failure modes are
*provably indistinguishable UNDER THE DEPLOYED REPRESENTATION* — every term of
the retrieval score is equal for them, so the ranking can only tie-break
arbitrarily. This module makes that first-class:

SCOPE — read this before quoting any number here. "Indistinguishable" is a
property of (envelope attribution + KB signature), NOT of the sensors and NOT
of the record. `--probe` and `repair.py` both show the load-torque twins ARE
separable from the same raw signals once the representation keeps deviation
DIRECTION and within-cycle STEP: 0.840 held-out over unseen fault
instantiations. So a twin cluster found here is a REPAIR TARGET, and
`repair.py` searches for the repair, verifies it, and hands it to an engineer
to authorize. Only clusters where that search fails (or needs a channel the
cell lacks) are limits of the sensing itself.

  1. It finds the STRUCTURAL TWINS in the knowledge base: failure modes with an
     identical (affected-joints, manifest-families) signature. From the sensor
     record alone they are mathematically interchangeable (Property 5.2 in
     docs/算法与数学表述.md): every evidence/joint/phase/coverage term is equal, so
     retrieval can only order them by an arbitrary tie-break.
  2. For each anomaly it returns not just a ranking but the INDISTINGUISHABILITY
     CLASS at rank 1 (the tied group) and a RESOLVING ACTION — the one thing
     that would break the tie:
        · decisive              → a single clear top-1, no ambiguity
        · human-resolvable      → structural twins; only operator knowledge /
                                  write-back (or a channel the cell lacks, e.g.
                                  temperature) can decide
        · sensor-resolvable     → a discriminating family exists but was not
                                  observed; name the signal to check
  3. It reports CLUSTER accuracy: collapse each twin group to one meta-label and
     ask whether the evidence picked the right *cluster*. This is the honest
     ceiling of what the sensors can determine, and it is far above the point
     top-1 — the gap is what the deployed representation cannot decide. Most of
     it turned out to be repairable, not irreducible (see repair.py).

This turns a modest point-accuracy (0.163) into an honest statement:
"the evidence pins the fault CLUSTER with accuracy P; the residual is undecidable
under the deployed representation, and is either repairable (repair.py finds and
verifies the missing dimension, an engineer authorizes it) or a genuine sensing
limit that the human-in-the-loop resolves."

Run:  .\\.venv\\Scripts\\python.exe distinguishability.py
"""
from __future__ import annotations
import os
from collections import defaultdict

from line_kg import FMEA, FM_SIGNATURE, _expand, retrieve

# channels the cell does not have — a structural-twin tie that would need one of
# these is a SENSOR GAP, not a human decision (voraus-AD has no temperature)
ABSENT_CHANNELS = {"temperature"}
# semantic hint of what separates known twin clusters (for the plain-language action)
TWIN_HINT = {
    frozenset({"FM_losecan", "FM_axisweight", "FM_varweight"}):
        "扭矩偏差方向与跨周期一致性（掉件=骤降 / 加重=持续升高 / 变重=逐件不同）——需产线负载配置或人工判定",
    frozenset({"FM_collision_foam", "FM_collision_card"}):
        "接触物材质（泡沫 vs 纸板）——信号无法区分，需现场目视/视频",
    frozenset({"FM_misgrip", "FM_invalidgrip"}):
        "抓取失败模式（漏抓 vs 示教点位置错）——需抓取位姿复核",
    frozenset({"FM_miscommutation", "FM_overtemp"}):
        "电气 vs 热因——本机无温度传感器，属传感器缺口",
}


# Which representation the twin test is taken over. False = the deployed
# representation (affected joints + manifest families), which is what every
# published number here uses. True additionally requires twins to share the
# MODE-LEVEL physical signature in line_kg.FM_SIGNATURE.
#
# LEAVE THIS FALSE. Setting it True collapses the four twin clusters to one --
# the interface would stop warning about three of them -- but the mode-level
# signature has been measured at 0.455 within the load cluster, BELOW the 0.504
# majority class, and switching it into retrieve() drops top-1 0.163 -> 0.148.
# It buys silence about undecidability, not the ability to decide. It is kept
# only so that claim stays reproducible (`repair.py` prints both numbers).
# The signature that DOES separate these modes is joint-resolved and learned per
# cluster; see repair.py.
USE_SIGNATURE = False


def structural_signature(fm: dict, use_signature: bool | None = None):
    use = USE_SIGNATURE if use_signature is None else use_signature
    base = (frozenset(_expand(fm["affects"])), frozenset(fm["manifests"]))
    if not use:
        return base
    sg = FM_SIGNATURE.get(fm["id"]) or {}
    return base + (sg.get("direction", 0), sg.get("temporal"))


def twin_classes(use_signature: bool | None = None):
    """Groups of failure modes with identical structural signature."""
    by_sig = defaultdict(list)
    for fm in FMEA:
        by_sig[structural_signature(fm, use_signature)].append(fm["id"])
    return [set(ids) for ids in by_sig.values() if len(ids) > 1]


def cluster_map(use_signature: bool | None = None):
    """fm_id -> cluster representative (its own id if not a twin)."""
    rep = {fm["id"]: fm["id"] for fm in FMEA}
    for grp in twin_classes(use_signature):
        r = sorted(grp)[0]
        for i in grp:
            rep[i] = r
    return rep


# cluster-confidence threshold: below this margin the evidence does not reliably
# fix the CLUSTER, so we abstain rather than confidently route to a within-cluster
# decision (the failure mode of replacing point-over-trust with boundary-over-trust).
# At tau=0.15 the abstention flags cluster errors at 0.82 precision (empirical).
CLUSTER_MARGIN_TAU = 0.15


def resolve(record, retrieve_fn=retrieve, eps=1e-6, tau=CLUSTER_MARGIN_TAU, top_k=None):
    """Return the ranking PLUS the rank-1 indistinguishability class, a
    cluster-confidence margin, and the action that would resolve it."""
    if top_k is None:
        top_k = len(FMEA)
    cands = retrieve_fn(record, top_k=top_k)
    rep = cluster_map()
    top = cands[0]["score"]
    top_cluster = rep[cands[0]["id"]]
    klass = [c for c in cands if abs(c["score"] - top) <= eps]
    ids = frozenset(c["id"] for c in klass)

    # margin between the rank-1 cluster and the best OTHER cluster
    next_cluster_best = next((c["score"] for c in cands if rep[c["id"]] != top_cluster), 0.0)
    cluster_margin = round(top - next_cluster_best, 4)

    if cluster_margin < tau:
        # CLUSTER-LEVEL ABSTENTION: not even the cluster is reliably determined.
        # (Honest caveat: this catches LOW-confidence cluster errors; ~43% of
        # cluster errors are high-margin "confident errors" in the
        # representation-limited regime that no margin threshold can flag.)
        route = "cluster-undetermined"
        other = next((c["id"] for c in cands if rep[c["id"]] != top_cluster), None)
        m1 = set(next(fm for fm in FMEA if fm["id"] == cands[0]["id"])["manifests"])
        m2 = set(next((fm for fm in FMEA if fm["id"] == other), {"manifests": []})["manifests"]) if other else set()
        disc = sorted(m1 ^ m2)
        action = ("所属簇不确定（置信不足）；判别信号：" + "、".join(disc)) if disc \
            else "所属簇不确定，证据不足以定簇"
    elif len(klass) == 1:
        route, action = "decisive", "证据充分，单一候选"
    else:
        # confident about the cluster; within it, structural twins → human decision
        route = "human-resolvable"
        hint = next((h for grp, h in TWIN_HINT.items() if ids <= grp or grp <= ids), None)
        action = hint or "结构孪生：当前表示下不可区分，需人工裁决/写回"
    return {
        "ranked": cands,
        "class": [c["id"] for c in klass],
        "class_size": len(klass),
        "cluster_margin": cluster_margin,
        "route": route,
        "action": action,
    }


# ---------------------------------------------------------------- measurement
def _measure():
    import json
    from eval_retrieve import load_cache
    from integration_test import CATEGORY_TO_FM
    recs = load_cache()
    rep = cluster_map()
    twins = twin_classes()

    print(f"knowledge base: {len(FMEA)} failure modes, "
          f"{len(twins)} structural-twin clusters covering "
          f"{sum(len(g) for g in twins)} modes\n")
    for g in twins:
        print("  indistinguishable:", sorted(g))
    print()

    n = len(recs)
    M = len(FMEA)
    size_of = {fm["id"]: sum(1 for f in FMEA if rep[f["id"]] == rep[fm["id"]]) for fm in FMEA}
    point_hit = cluster_hit = 0
    route_ct = defaultdict(int)
    abstain_ct = abstain_wrong = 0        # cluster-undetermined route precision
    pt_rand = 1.0 / M
    cl_rand_sum = 0.0                                 # E[ |cluster(true)| / M ]
    per = defaultdict(lambda: [0, 0, 0.0])            # cat -> [n, cluster_hit, cl_rand]
    by_route = defaultdict(lambda: [0, 0, 0])         # route -> [n, point_hit, cluster_hit]
    margins = []                                      # (margin, point_hit, cluster_hit)
    twin_ids = [frozenset(g) for g in twins]
    tie_kind = defaultdict(int)                       # structural twin vs numeric tie
    for rec in recs:
        cat = rec["detector"]["ground_truth_category"]
        tf = CATEGORY_TO_FM.get(cat)
        res = resolve(rec)
        top1 = res["ranked"][0]["id"]
        ph = (top1 == tf)
        point_hit += ph
        ch = (rep.get(top1) == rep.get(tf))
        cluster_hit += ch
        route_ct[res["route"]] += 1
        d = by_route[res["route"]]; d[0] += 1; d[1] += ph; d[2] += ch
        margins.append((res["cluster_margin"], ph, ch))
        if res["class_size"] > 1:
            k = frozenset(res["class"])
            # Property 5.2 only covers ties that ARE a structural-twin group;
            # a numeric tie between unrelated modes is a different animal and
            # must not be reported with the same words.
            tie_kind["structural" if any(k <= g or g <= k for g in twin_ids)
                     else "numeric"] += 1
        if res["route"] == "cluster-undetermined":
            abstain_ct += 1
            abstain_wrong += (not ch)     # was the cluster actually wrong?
        crand = size_of[tf] / M
        cl_rand_sum += crand
        d = per[cat]; d[0] += 1; d[1] += ch; d[2] = crand
    cl_rand = cl_rand_sum / n

    # ---- selective accuracy: is the route label informative at all? ---------
    print("=" * 66)
    print("SELECTIVE ACCURACY — does the system's own route label carry information?")
    print("=" * 66)
    print("  If 'decisive' is not markedly more accurate than the global rate, the")
    print("  route label is decoration and the interface would be dressing up")
    print("  incapacity as humility. This is the claim's most direct test.")
    print(f"    {'route':22s} {'n':>4} {'coverage':>9} {'point top-1':>12} {'cluster top-1':>14}")
    for r in ("decisive", "human-resolvable", "cluster-undetermined"):
        nn, p, c = by_route[r]
        if not nn:
            continue
        print(f"    {r:22s} {nn:>4} {nn/n:>9.3f} {p/nn:>12.3f} {c/nn:>14.3f}")
    print(f"    {'ALL':22s} {n:>4} {1.0:>9.3f} {point_hit/n:>12.3f} {cluster_hit/n:>14.3f}")
    dec_n, dec_p, _ = by_route["decisive"]
    if dec_n:
        print(f"  => on the {dec_n/n:.1%} it calls decisive, point top-1 is "
              f"{dec_p/dec_n:.3f} = {(dec_p/dec_n)/(point_hit/n):.2f}x the global rate")

    # ---- risk-coverage: is CLUSTER_MARGIN_TAU a defensible operating point? --
    margins.sort(key=lambda t: -t[0])
    rc = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        k = max(1, int(round(frac * n)))
        sel = margins[:k]
        rc.append({"coverage": k / n,
                   "point_top1": sum(t[1] for t in sel) / k,
                   "cluster_top1": sum(t[2] for t in sel) / k,
                   "margin_at": sel[-1][0]})
    print("\n  RISK-COVERAGE (accept the k highest cluster-margin cycles):")
    print(f"    {'coverage':>9} {'margin>=':>9} {'point':>7} {'cluster':>8}")
    for r in rc:
        print(f"    {r['coverage']:>9.2f} {r['margin_at']:>9.3f} "
              f"{r['point_top1']:>7.3f} {r['cluster_top1']:>8.3f}")
    tau_cov = sum(1 for m, _, _ in margins if m >= CLUSTER_MARGIN_TAU) / n
    print(f"    shipped tau={CLUSTER_MARGIN_TAU} sits at coverage {tau_cov:.3f}")

    # ---- the tie the maths actually covers ----------------------------------
    tot_tie = sum(tie_kind.values())
    print(f"\n  TIE COMPOSITION: {tot_tie} cycles tie at rank 1 — "
          f"{tie_kind['structural']} are structural twins (Property 5.2 applies), "
          f"{tie_kind['numeric']} are numeric ties between non-twin modes.\n")

    print("=" * 66)
    print("DISTINGUISHABILITY — locating the loss (all 755 anomalies)")
    print("=" * 66)
    print("  HONEST BASELINE CHECK: the global cluster lift is an ARITHMETIC")
    print("  effect of merging labels, not recovered evidence capability.")
    print(f"    point   top-1 {point_hit/n:.3f}  vs random {pt_rand:.3f}  = {point_hit/n/pt_rand:.2f}x")
    print(f"    cluster top-1 {cluster_hit/n:.3f}  vs random {cl_rand:.3f}  = {cluster_hit/n/cl_rand:.2f}x")
    print("  => nearly identical normalized lift; cluster analysis LOCATES the")
    print("     loss (which regime), it does not RECOVER accuracy.")
    print("  !! UNIFORM RANDOM IS THE WRONG BASELINE ON THIS SKEWED BENCHMARK.")
    print("     A constant predictor -- always the most frequent mode / cluster --")
    print("     scores 0.187 at the point level and 0.401 at the cluster level, i.e.")
    print("     ABOVE both figures above. Run `baselines.py` before quoting any lift")
    print("     from this block; the same applies to the per-category lifts below,")
    print("     where a category inside the majority cluster has a constant baseline")
    print("     of 1.000, not of its cluster-random rate.\n")
    print("  per-category cluster accuracy, normalized by its own cluster-random:")
    CATN = {0: "friction", 1: "axisweight", 2: "coll_foam", 3: "coll_cable", 4: "coll_card",
            5: "misgrip", 6: "losecan", 7: "varweight", 8: "entangled", 9: "invalidgrip",
            10: "commut", 11: "platform"}
    for cat in sorted(per):
        nn, hh, cr = per[cat]
        acc = hh / nn
        lift = acc / cr if cr else 0.0
        wall = "distinguishability" if lift >= 2.5 else ("representation" if acc < 0.02 else "weak")
        print(f"    cat {cat:>2} {CATN[cat]:>11} n={nn:>3}  cluster {acc:.3f}  norm {lift:>4.1f}x  [{wall}]")
    print()
    print("  resolving route distribution:")
    for r in ("decisive", "human-resolvable", "cluster-undetermined"):
        print(f"    {r:20s}: {route_ct[r]/n:.3f}  ({route_ct[r]})")
    if abstain_ct:
        print(f"  cluster-abstention precision: {abstain_wrong}/{abstain_ct} = "
              f"{abstain_wrong/abstain_ct:.2f} of abstentions were genuinely wrong-cluster")
    print("  CAVEAT: margin-abstention catches LOW-confidence cluster errors; the")
    print("  representation-limited regime (collisions) fails CONFIDENTLY and is")
    print("  not flagged — distinguishability output is trustworthy only where")
    print("  cluster accuracy is high (the load-torque regime).")
    print()

    # example plain-language outputs
    print("example outputs:")
    shown = set()
    for rec in recs:
        res = resolve(rec)
        if res["route"] in shown or res["route"] == "decisive":
            continue
        shown.add(res["route"])
        cat = rec["detector"]["ground_truth_category"]
        print(f"  cycle {rec['cycle_id']} (cat {cat}) [{res['route']}]")
        print(f"    候选并列: {res['class']}")
        print(f"    如何定案: {res['action']}")
        if len(shown) >= 2:
            break

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "distinguishability_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n,
                   "point_top1": point_hit/n, "point_random": pt_rand,
                   "point_lift": point_hit/n/pt_rand,
                   "cluster_top1": cluster_hit/n, "cluster_random": cl_rand,
                   "cluster_lift": cluster_hit/n/cl_rand,
                   "per_category": {CATN[c]: {"n": per[c][0], "cluster_acc": per[c][1]/per[c][0],
                                              "cluster_random": per[c][2],
                                              "norm_lift": (per[c][1]/per[c][0])/per[c][2] if per[c][2] else 0.0}
                                    for c in sorted(per)},
                   "routes": {k: v/n for k, v in route_ct.items()},
                   "by_route": {k: {"n": v[0], "coverage": v[0]/n,
                                    "point_top1": v[1]/v[0], "cluster_top1": v[2]/v[0]}
                                for k, v in by_route.items() if v[0]},
                   "risk_coverage": rc,
                   "tau": CLUSTER_MARGIN_TAU, "tau_coverage": tau_cov,
                   "tie_composition": dict(tie_kind),
                   "twin_clusters": [sorted(g) for g in twins]}, f, indent=2, ensure_ascii=False)
    print(f"written -> {out}")


def separability_probe():
    """Test whether a structural-twin cluster is separable from the RAW signals
    with features the deployed retrieval discards — deviation DIRECTION, temporal
    STEP, and the load-power channels the pipeline never attributes. A high
    accuracy proves that indistinguishability is a property of the deployed
    REPRESENTATION, not of the record: the twin detector is then a precise
    representation/KB audit ('these dimensions would separate them')."""
    import numpy as np, pandas as pd
    from attribution_bridge import build_reference, _resample, GRID_N
    from integration_test import DATA_PATH, TRAIN_SETTING
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # the load-torque twin cluster and its voraus categories
    from integration_test import CATEGORY_TO_FM
    grp = next(g for g in twin_classes() if "FM_losecan" in g)
    inv = {v: k for k, v in CATEGORY_TO_FM.items()}
    cats = {inv[m]: m.replace("FM_", "") for m in grp if m in inv}
    TQ = [f"motor_torque_{j}" for j in (1, 2, 3)] + [f"torque_sensor_a_{j}" for j in (1, 2, 3)]
    PW = [f"power_load_mech_{j}" for j in (1, 2, 3)]
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
    sig = [c for c in TQ + PW if c in avail]
    df = pd.read_parquet(DATA_PATH, columns=sig + ["time", "sample", "category", "setting"])
    grid = np.linspace(0, 1, GRID_N)
    ref = build_reference(df[df["setting"] == TRAIN_SETTING], sig, grid)
    sub = df[df.category.isin(cats)]
    cat_of = sub.groupby("sample")["category"].first().to_dict()

    rows, y = [], []
    for sid, run in sub.sort_values("time").groupby("sample"):
        f = {}
        for c in sig:
            obs = _resample(run, c, grid)
            dev = obs - ref[c]["mean"]
            z = dev / (ref[c]["scale"] + 1e-9)
            f[c + "_meanz"] = float(z.mean())                       # sustained signed offset
            f[c + "_late"] = float(dev[300:].mean() - dev[:150].mean())  # temporal step
            f[c + "_std"] = float(z.std())
        rows.append(f); y.append(cat_of[sid])
    X = pd.DataFrame(rows).fillna(0.0).values
    y = np.array(y)
    p = np.array([(y == k).mean() for k in cats])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                          X, y, cv=cv, scoring="accuracy")

    print("=" * 66)
    print("SEPARABILITY PROBE — is the load-torque twin cluster really undecidable?")
    print("=" * 66)
    print(f"  cluster {sorted(grp)}")
    print(f"  categories {dict(cats)}, {len(y)} cycles")
    print(f"  logistic regression on direction/step/power features (NOT used by")
    print(f"  the deployed retrieval): 3-way accuracy {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  majority-class baseline {p.max():.3f}")
    print("  => the twins ARE separable from the raw signals; they are")
    print("     indistinguishable only UNDER THE DEPLOYED REPRESENTATION (magnitude")
    print("     + family membership over envelope z). The twin detector therefore")
    print("     audits the representation: direction, temporal step, and load")
    print("     power are the discarded discriminative dimensions.")
    for k, name in cats.items():
        late = np.mean([X[i][list(pd.DataFrame(rows).columns).index("motor_torque_2_late")]
                        for i in range(len(y)) if y[i] == k])
        print(f"    {name:11s}: mean J2 torque late-step {late:+.3f}")


if __name__ == "__main__":
    import sys
    if "--probe" in sys.argv:
        separability_probe()
    else:
        _measure()
