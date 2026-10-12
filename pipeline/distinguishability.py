# -*- coding: utf-8 -*-
"""Diagnostic distinguishability: which failure modes the deployed retrieval
cannot tell apart, and how each anomalous cycle is routed as a result.

A ranked list implies that the top-1 entry is the answer. For some failure
modes the deployed representation gives the ranking nothing to separate them
with, and the order among them is set by a tie-break. This module finds those
modes, reports the tie for every cycle, and routes the cycle accordingly.

SCOPE. "Indistinguishable" here is a property of the deployed representation
(envelope attribution plus the knowledge-base signature). It is not a property
of the sensors or of the raw record. `--probe` and `repair.py` show that the
load-torque twins can be separated from the same raw signals once the
representation keeps the sign of the deviation and the within-cycle step
(0.840 held-out over unseen fault instantiations). A twin group found here is
therefore a repair target. Only a group for which the repair search fails, or
for which the repair needs a channel the cell does not have, is a limit of the
sensing itself.

What the module does:

  1. It finds the structural twins: failure modes with the same affected
     joints and the same manifest families. For two such modes the joint,
     family-overlap, evidence and coverage terms of retrieve() are equal on
     every record. The phase term is not, because it compares the record's
     phase class with each mode's own phase hint. Twins that also share the
     hint (the two collisions: transfer; mis-grip and invalid grip: grip;
     miscommutation and over-temperature: any) get the same score on every
     record for any setting of the joint, family, evidence, phase and coverage
     weights, so only the tie-break orders them. (The feedback prior and the
     signature term are per-mode; both are off by default.) Inside the
     load-torque group the hints differ: axis weight "any", lost payload
     "transfer", variable weight "grip". "any" agrees with every phase class,
     so axis weight scores at least as high as the other two on every record,
     and the phase class decides which of them ties with it: lost payload in
     transfer phases, variable weight in pick/place phases, both when the
     phase class is undefined. With w_phase = 0 all three tie. The results
     file reports this under kb_columns, tie_composition_sets and
     phase_sensitivity.
  2. For each record it returns the ranking, the rank-1 tied set, the score
     margin between the rank-1 cluster and the best other cluster, and a route:
        decisive              a single rank-1 mode and a margin >= tau
        human-resolvable      the rank-1 modes are twins and the margin is
                              >= tau; operator knowledge, write-back, or a
                              channel the cell lacks (e.g. temperature) has
                              to decide between them
        cluster-undetermined  margin < tau; the action names the signal
                              families that differ between the rank-1 mode
                              and the best mode of another cluster
  3. It reports cluster accuracy: each twin group is merged into one label and
     the question is whether the rank-1 mode is in the right group. Merging
     labels raises the chance rate by about as much as the accuracy, so the
     cluster view shows where the errors are; it does not recover accuracy
     (see _measure and baselines.py).

tau = 0.15 (CLUSTER_MARGIN_TAU) was chosen on the same 755 cycles it is
evaluated on. route_margin_tau() re-chooses it on stratified fit halves and
scores the choice on the held-out halves.

Run (from pipeline/):  .\\.venv\\Scripts\\python.exe distinguishability.py
`--probe` additionally needs the voraus-AD parquet.
"""
from __future__ import annotations
import inspect
import os
from collections import Counter, defaultdict

from line_kg import (FMEA, FM_SIGNATURE, HINT_PHASE_CLASSES, PHASE_OF_ACTION,
                     _expand, retrieve)

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


# Cluster-margin threshold. Below it the evidence does not fix the cluster
# reliably, so the cycle is routed cluster-undetermined instead of being passed
# to a within-cluster decision. tau = 0.15 was chosen on the same 755 cycles it
# is evaluated on. There 0.82 of the abstained cycles have a wrong rank-1
# cluster, and the abstentions cover 100 of the 498 cluster errors. 151 cycles
# have a margin of exactly 1/6 (load group vs unstable platform, separated by
# family overlap alone), so any tau above 1/6 abstains on at least 310 cycles
# instead of 122.
# Choosing tau on 200 stratified fit halves gives held-out precision 0.80
# (route_margin_tau in the results file).
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
# voraus-AD category -> short name used as a key in the results file
CATN = {0: "friction", 1: "axisweight", 2: "coll_foam", 3: "coll_cable", 4: "coll_card",
        5: "misgrip", 6: "losecan", 7: "varweight", 8: "entangled", 9: "invalidgrip",
        10: "commut", 11: "platform"}


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

    extra = revision_checks(recs)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "distinguishability_results.json")
    with open(out, "w", encoding="utf-8", newline="\r\n") as f:
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
                   "twin_clusters": [sorted(g) for g in twins],
                   **extra}, f, indent=2, ensure_ascii=False)
    print(f"written -> {out}")


# ---------------------------------------------------------------- revision checks
# Offline checks on the cached records (data/records_cache.json; no parquet).
# _measure() writes each result under its own key in the results file.
TIE_EPS = 1e-6                                # same tolerance as resolve()
W_PHASE_DEPLOYED = inspect.signature(retrieve).parameters["w_phase"].default
PHASE_WEIGHTS = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0)
SPLIT_SEEDS = (0, 1, 2, 3, 4)                 # leave-setting-out splits of Table 1
TAU_GRID = (0.05, 0.10, 0.15, 0.16, 0.17, 0.20, 0.25)
TAU_SPLITS = 200                              # stratified 50/50 splits, seeds 0..199
TAU_MIN_COVERAGE = 0.80                       # selection rule, see route_margin_tau()
COLLISION_CATS = (2, 3, 4)
GRIPPING_CATS = (5, 9)                        # mis-grip and invalid grip (a twin pair)


def _phase_class(rec):
    """The phase class retrieve() compares with each mode's phase hint
    (None for VOID_* segments; no mode gets the phase term then)."""
    return PHASE_OF_ACTION.get(rec["localization"].get("dominant_cycle_phase"))


def _tied_ids(cands, eps=TIE_EPS):
    top = cands[0]["score"]
    return [c["id"] for c in cands if abs(c["score"] - top) <= eps]


def _set_key(ids):
    return "+".join(sorted(ids))


def _counts(counter):
    return {str(k): int(v) for k, v in counter.most_common()}


def _context(recs):
    """Records keyed by cycle id (sorted as in repair.py), labels, and the
    deployed ranking of every record."""
    from integration_test import CATEGORY_TO_FM
    by_id = {str(r["cycle_id"]): r for r in recs}
    ids = sorted(by_id, key=int)
    cat_of = {i: by_id[i]["detector"]["ground_truth_category"] for i in ids}
    return dict(recs=by_id, ids=ids, cat_of=cat_of,
                set_of={i: by_id[i]["detector"]["ground_truth_setting"] for i in ids},
                true_fm={i: CATEGORY_TO_FM[cat_of[i]] for i in ids},
                rep=cluster_map(),
                ranked={i: retrieve(by_id[i], top_k=len(FMEA)) for i in ids})


def tie_sets(ctx):
    """Rank-1 tied sets by composition at the deployed weights, with the
    record's phase class and the true category behind each set."""
    rows = defaultdict(lambda: {"n": 0, "phase": Counter(), "cat": Counter(), "hit": 0})
    for i in ctx["ids"]:
        t = _tied_ids(ctx["ranked"][i])
        if len(t) < 2:
            continue
        d = rows[_set_key(t)]
        d["n"] += 1
        d["phase"][_phase_class(ctx["recs"][i]) or "undefined"] += 1
        d["cat"][CATN[ctx["cat_of"][i]]] += 1
        d["hit"] += ctx["true_fm"][i] in t
    sets = {k: {"n": d["n"], "phase_class": _counts(d["phase"]),
                "true_category": _counts(d["cat"]), "true_mode_in_set": d["hit"]}
            for k, d in sorted(rows.items(), key=lambda kv: -kv[1]["n"])}
    return {"n_cycles": len(ctx["ids"]),
            "n_tied": sum(d["n"] for d in sets.values()),
            "phase_class_rule": "line_kg.PHASE_OF_ACTION[dominant_cycle_phase]; "
                                "'undefined' is a VOID_* segment, where no mode "
                                "gets the phase term",
            "sets": sets}


def phase_sensitivity(ctx, weights=PHASE_WEIGHTS, seeds=SPLIT_SEEDS):
    """Rank-1 ties, their composition and the tied-set oracle as w_phase varies
    (all other weights at their defaults). The oracle is the share of cycles
    whose true mode is in the rank-1 tied set."""
    import numpy as np
    import repair as R
    ids, rep, true_fm = ctx["ids"], ctx["rep"], ctx["true_fm"]
    tests = [R._grouped_split(ids, ctx["cat_of"], ctx["set_of"], s)[1] for s in seeds]
    by_w = {}
    for w in weights:
        comp, lost, hit = Counter(), Counter(), {}
        point = clus = 0
        for i in ids:
            c = retrieve(ctx["recs"][i], top_k=len(FMEA), w_phase=w)
            t = _tied_ids(c)
            if len(t) > 1:
                comp[_set_key(t)] += 1
            hit[i] = true_fm[i] in t
            point += c[0]["id"] == true_fm[i]
            in_group = rep[c[0]["id"]] == rep[true_fm[i]]
            clus += in_group
            if in_group and not hit[i]:
                lost[CATN[ctx["cat_of"][i]]] += 1
        per_split = [float(np.mean([hit[i] for i in test])) for test in tests]
        by_w[str(w)] = {
            "n_tied": sum(comp.values()), "composition": _counts(comp),
            "point_top1": point / len(ids), "cluster_top1": clus / len(ids),
            "oracle_all": sum(hit.values()) / len(ids),
            "oracle_grouped_mean": float(np.mean(per_split)),
            "oracle_grouped_per_split": per_split,
            "true_in_group_not_in_tied": sum(lost.values()),
            "true_in_group_not_in_tied_by_category": _counts(lost)}
    on, off = by_w[str(W_PHASE_DEPLOYED)], by_w[str(0.0)]
    return {"w_phase_deployed": W_PHASE_DEPLOYED, "weights": list(weights),
            "grouped_split": "repair._grouped_split (leave-setting-out), held-out half, "
                             "mean over seeds",
            "split_seeds": list(seeds), "n_test": [len(t) for t in tests],
            "oracle_all_with_phase": on["oracle_all"],
            "oracle_all_without_phase": off["oracle_all"],
            "oracle_grouped_with_phase": on["oracle_grouped_mean"],
            "oracle_grouped_without_phase": off["oracle_grouped_mean"],
            "true_in_group_not_in_tied": on["true_in_group_not_in_tied"],
            "group_definition": "the twin group (cluster_map) of the rank-1 mode",
            "by_w_phase": by_w}


def kb_columns(ctx):
    """Each mode's column (scope, manifest families, phase hint), the pairs with
    the same (joints, families), and the strict inclusions between columns."""
    order = [fm["id"] for fm in FMEA]
    fm_of = {fm["id"]: fm for fm in FMEA}
    col = {f: (frozenset(_expand(fm_of[f]["affects"])), frozenset(fm_of[f]["manifests"]))
           for f in order}
    hint = {f: fm_of[f]["phase_hint"] for f in order}
    # phase classes that earn the phase term (None never does in retrieve())
    agree = {f: HINT_PHASE_CLASSES.get(hint[f], set()) - {None} for f in order}
    scores = {i: {c["id"]: c["score"] for c in ctx["ranked"][i]} for i in ctx["ids"]}

    def relation(a, b):
        if agree[a] == agree[b]:
            return "same"
        if agree[a] > agree[b]:
            return f"{a} covers {b}"
        if agree[b] > agree[a]:
            return f"{b} covers {a}"
        return "overlap" if agree[a] & agree[b] else "disjoint"

    equal, inclusions = [], []
    for x, a in enumerate(order):
        for b in order[x + 1:]:
            if col[a] != col[b]:
                continue
            d = [scores[i][a] - scores[i][b] for i in ctx["ids"]]
            equal.append({"modes": [a, b], "phase_hints": [hint[a], hint[b]],
                          "same_phase_hint": hint[a] == hint[b],
                          "phase_relation": relation(a, b),
                          "n_first_higher": sum(v > TIE_EPS for v in d),
                          "n_equal": sum(abs(v) <= TIE_EPS for v in d),
                          "n_second_higher": sum(v < -TIE_EPS for v in d)})
    for a in order:
        for b in order:
            if a != b and col[a] != col[b] and col[a][0] <= col[b][0] and col[a][1] <= col[b][1]:
                inclusions.append({"inner": a, "outer": b,
                                   "phase_hints": [hint[a], hint[b]],
                                   "outer_phase_covers_inner": agree[b] >= agree[a]})
    return {"column": "scope expanded to a joint set, manifest families, phase hint",
            "modes": {f: {"scope": fm_of[f]["affects"], "joints": sorted(col[f][0]),
                          "families": sorted(col[f][1]), "phase_hint": hint[f]}
                      for f in order},
            "n_equal_pairs": len(equal),
            "n_equal_pairs_same_phase_hint": sum(p["same_phase_hint"] for p in equal),
            "equal_pairs": equal,
            "n_strict_inclusions": len(inclusions),
            "strict_inclusions": inclusions,
            "pair_scores_over": f"{len(ctx['ids'])} records, deployed weights"}


def localization(ctx):
    """Where the record localizes, whether a distal-scoped mode ever earns the
    joint term, and the rank of the true mode."""
    import numpy as np
    from line_kg import JOINTS, DISTAL_JOINTS
    recs, ids = ctx["recs"], ctx["ids"]
    dom = Counter(recs[i]["localization"]["dominant_joint"] for i in ids)
    top_z = Counter(recs[i]["localization"]["joint_ranking_by_z"][0]["joint"]
                    for i in ids if recs[i]["localization"]["joint_ranking_by_z"])
    distal = [fm["id"] for fm in FMEA if fm["affects"] == "distal"]
    jm = {f: sum(next(c for c in ctx["ranked"][i] if c["id"] == f)["provenance"]["joint_match"]
                 for i in ids) for f in distal}
    rank, best = defaultdict(list), defaultdict(list)
    for i in ids:
        c, tf, cat = ctx["ranked"][i], ctx["true_fm"][i], ctx["cat_of"][i]
        s = next(x["score"] for x in c if x["id"] == tf)
        rank[cat].append([x["id"] for x in c].index(tf) + 1)
        best[cat].append(1 + sum(x["score"] > s + TIE_EPS for x in c))

    def summary(cats):
        r = [v for k in cats for v in rank[k]]
        b = [v for k in cats for v in best[k]]
        return {"n": len(r), "median_rank": float(np.median(r)),
                "median_best_rank": float(np.median(b))}

    return {"dominant_joint": {str(j): dom.get(j, 0) for j in JOINTS},
            "dominant_joint_rule": "joint with the largest absolute peak torque deviation "
                                   "(attribution_bridge.to_bridge_record)",
            "dominant_joint_distal": sum(dom.get(j, 0) for j in DISTAL_JOINTS),
            "top_joint_by_z": {str(j): top_z.get(j, 0) for j in JOINTS},
            "distal_modes": distal,
            "distal_joint_match": jm,
            "distal_joint_match_total": sum(jm.values()),
            "rank_rule": "median_rank: position in retrieve()'s order (ties in declaration "
                         "order); median_best_rank: 1 + modes scoring strictly higher",
            "collision": {"categories": [CATN[k] for k in COLLISION_CATS],
                          **summary(COLLISION_CATS)},
            "gripping": {"categories": [CATN[k] for k in GRIPPING_CATS],
                         **summary(GRIPPING_CATS)},
            "per_category": {CATN[k]: summary((k,)) for k in sorted(rank)}}


def route_margin_tau(ctx, grid=TAU_GRID, n_splits=TAU_SPLITS, min_cov=TAU_MIN_COVERAGE):
    """The cluster margin of resolve(): a tau scan, the mass point at 1/6, the
    margin's AUROC for cluster correctness, and tau re-chosen on fit halves.

    Selection rule, applied on the fit half only: keep the scanned taus that
    still route at least `min_cov` of the fit cycles (abstention stays a reject
    option for a minority of cycles, as the comment at CLUSTER_MARGIN_TAU
    describes; the shipped 0.15 routes 0.838 of all cycles), then take the one
    with the highest abstention precision, the share of abstained cycles whose
    rank-1 cluster is wrong. Ties go to the smaller tau. The chosen tau is
    scored on the test half of the same split.
    """
    import numpy as np
    import repair as R
    from sklearn.metrics import roc_auc_score
    ids, rep = ctx["ids"], ctx["rep"]
    margin, ok, res_of = {}, {}, {}
    for i in ids:
        res = resolve(ctx["recs"][i])
        res_of[i] = res
        margin[i] = res["cluster_margin"]
        ok[i] = rep[res["ranked"][0]["id"]] == rep[ctx["true_fm"][i]]

    def stats(sub, tau):
        ab = [i for i in sub if margin[i] < tau]
        ac = [i for i in sub if margin[i] >= tau]
        wrong = sum(not ok[i] for i in ab)
        n_wrong = sum(not ok[i] for i in sub)
        return {"tau": tau, "n": len(sub), "n_abstained": len(ab),
                "coverage": len(ac) / len(sub),
                "cluster_acc_accepted": float(np.mean([ok[i] for i in ac])) if ac else None,
                "cluster_acc_abstained": float(np.mean([ok[i] for i in ab])) if ab else None,
                "abstention_precision": wrong / len(ab) if ab else None,
                "n_wrong_abstained": wrong,
                "share_of_errors_abstained": wrong / n_wrong if n_wrong else None}

    # the mass point: margin exactly 1/6 (scores are rounded to 4 decimals)
    sixth = [i for i in ids if margin[i] == round(1 / 6, 4)]
    pair, fams, no_ev = Counter(), Counter(), 0
    for i in sixth:
        c = res_of[i]["ranked"]
        other = next(x for x in c if rep[x["id"]] != rep[c[0]["id"]])
        pair[f"{rep[c[0]['id']]} cluster vs {other['id']}"] += 1
        fams[_set_key({s["family"] for s in ctx["recs"][i]["attributed_signals"]})] += 1
        no_ev += not (c[0]["provenance"]["evidence_signals"]
                      or other["provenance"]["evidence_signals"])

    chosen, te_prec, te_cov, fit_prec, fixed = Counter(), [], [], [], []
    for seed in range(n_splits):
        fit, test = R._stratified_split(ids, ctx["cat_of"], seed)
        cand = [stats(fit, t) for t in grid]
        feas = [s for s in cand if s["coverage"] >= min_cov and s["n_abstained"]] or cand[:1]
        pick = max(feas, key=lambda s: (s["abstention_precision"], -s["tau"]))
        te = stats(test, pick["tau"])
        chosen[pick["tau"]] += 1
        fit_prec.append(pick["abstention_precision"])
        te_prec.append(te["abstention_precision"])
        te_cov.append(te["coverage"])
        fixed.append(stats(test, CLUSTER_MARGIN_TAU)["abstention_precision"])

    def pct(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    shipped = stats(ids, CLUSTER_MARGIN_TAU)
    return {
        "tau": CLUSTER_MARGIN_TAU,
        "tau_chosen_on": f"the same {len(ids)} cycles it is evaluated on (in-sample)",
        "margin_definition": "rank-1 score minus the best score of a mode outside "
                             "the rank-1 cluster (resolve(), rounded to 4 decimals)",
        "in_sample": shipped,
        "scan": [stats(ids, t) for t in grid],
        "n_margin_one_sixth": len(sixth),
        "one_sixth_rule": "cluster_margin == round(1/6, 4) == 0.1667",
        "n_margin_within_5e-4_of_one_sixth": sum(abs(margin[i] - 1 / 6) <= 5e-4 for i in ids),
        "one_sixth_pairs": _counts(pair),
        "one_sixth_record_families": _counts(fams),
        "one_sixth_no_joint_gated_evidence": no_ev,
        "one_sixth_cluster_acc": float(np.mean([ok[i] for i in sixth])) if sixth else None,
        "auroc_margin_vs_cluster_correct": float(roc_auc_score(
            [int(ok[i]) for i in ids], [margin[i] for i in ids])),
        "fit_half_selection": {
            "splits": f"repair._stratified_split, seeds 0-{n_splits - 1}",
            "candidates": list(grid),
            "rule": f"on the fit half keep taus with coverage >= {min_cov} and at "
                    "least one abstention; take the highest abstention precision; "
                    "ties to the smaller tau; score it on the test half",
            "min_coverage": min_cov,
            "chosen_tau_counts": {str(k): v for k, v in sorted(chosen.items())},
            "fit_precision_mean": float(np.mean(fit_prec)),
            "heldout_precision_mean": float(np.mean(te_prec)),
            "heldout_precision_p2_5": pct(te_prec)[0],
            "heldout_precision_p97_5": pct(te_prec)[1],
            "heldout_coverage_mean": float(np.mean(te_cov)),
            "shipped_tau_heldout_precision_mean": float(np.mean(fixed)),
            "shipped_tau_heldout_precision_p2_5": pct(fixed)[0],
            "shipped_tau_heldout_precision_p97_5": pct(fixed)[1],
            "in_sample_precision_shipped_tau": shipped["abstention_precision"]},
    }


def authoring_sensitivity(ctx, hints=("any", "transfer", "grip")):
    """Rank-1 ties when axis weight's phase hint is declared differently.

    The tie count belongs to this knowledge base, not to the robot: one
    plausible change to one declaration moves it. The hint is restored after.
    """
    fm = next(f for f in FMEA if f["id"] == "FM_axisweight")
    shipped = fm["phase_hint"]
    out = {"mode": "FM_axisweight", "shipped_hint": shipped, "by_hint": {}}
    try:
        for h in hints:
            fm["phase_hint"] = h
            comp = Counter()
            for i in ctx["ids"]:
                t = _tied_ids(retrieve(ctx["recs"][i], top_k=len(FMEA)))
                if len(t) > 1:
                    comp[_set_key(t)] += 1
            out["by_hint"][h] = {"n_tied": int(sum(comp.values())),
                                 "composition": _counts(comp)}
    finally:
        fm["phase_hint"] = shipped
    return out


def revision_checks(recs, verbose=True):
    """Run the checks above; keys are the ones written to the results file."""
    ctx = _context(recs)
    out = {"tie_composition_sets": tie_sets(ctx),
           "phase_sensitivity": phase_sensitivity(ctx),
           "kb_columns": kb_columns(ctx),
           "localization": localization(ctx),
           "route_margin_tau": route_margin_tau(ctx),
           "authoring_sensitivity": authoring_sensitivity(ctx)}
    # abstaining at random would already be right this often
    wrong = sum(ctx["rep"].get(ctx["ranked"][i][0]["id"]) != ctx["rep"].get(ctx["true_fm"][i])
                for i in ctx["ids"])
    out["route_margin_tau"]["wrong_cluster_base_rate"] = wrong / len(ctx["ids"])
    if not verbose:
        return out
    ts, ps, kc = out["tie_composition_sets"], out["phase_sensitivity"], out["kb_columns"]
    lo, rt = out["localization"], out["route_margin_tau"]
    print("=" * 66)
    print("REVISION CHECKS (records cache only)")
    print("=" * 66)
    print(f"  rank-1 tied sets ({ts['n_tied']} of {ts['n_cycles']}):")
    for k, d in ts["sets"].items():
        print(f"    {d['n']:>4}  {k}  phase {d['phase_class']}  true {d['true_category']}")
    print("  phase weight sweep (n tied / oracle all / oracle grouped / cluster top-1):")
    for w, d in ps["by_w_phase"].items():
        print(f"    w_phase {w:>4}: {d['n_tied']:>4}  {d['oracle_all']:.3f}  "
              f"{d['oracle_grouped_mean']:.3f}  {d['cluster_top1']:.3f}  {d['composition']}")
    print(f"  true mode in the rank-1 group but not in the tied set "
          f"(w_phase {ps['w_phase_deployed']}): {ps['true_in_group_not_in_tied']}")
    print(f"  KB columns: {kc['n_equal_pairs']} equal (joints, families) pairs, "
          f"{kc['n_equal_pairs_same_phase_hint']} of them with the same phase hint; "
          f"{kc['n_strict_inclusions']} strict inclusions")
    for p in kc["equal_pairs"]:
        print(f"    {p['modes']}  hints {p['phase_hints']}  {p['phase_relation']}")
    print(f"  dominant joint {lo['dominant_joint']}; distal-scoped joint matches "
          f"{lo['distal_joint_match_total']}")
    print(f"  median true-mode rank: collision {lo['collision']['median_rank']}, "
          f"gripping {lo['gripping']['median_rank']}")
    print(f"  cluster margin: {rt['n_margin_one_sixth']} cycles at exactly 1/6, "
          f"AUROC {rt['auroc_margin_vs_cluster_correct']:.3f}")
    for s in rt["scan"]:
        print(f"    tau {s['tau']:.2f}: abstain {s['n_abstained']:>3}  "
              f"acc accepted {s['cluster_acc_accepted']:.3f}  "
              f"acc abstained {s['cluster_acc_abstained']:.3f}  "
              f"precision {s['abstention_precision']:.3f}")
    print("  axis weight phase hint -> rank-1 ties: " + ", ".join(
        f"{h} {d['n_tied']}" for h, d in out["authoring_sensitivity"]["by_hint"].items()))
    print(f"  wrong-cluster base rate {rt['wrong_cluster_base_rate']:.3f}")
    fs = rt["fit_half_selection"]
    print(f"  tau chosen on fit halves {fs['chosen_tau_counts']}: held-out precision "
          f"{fs['heldout_precision_mean']:.3f} [{fs['heldout_precision_p2_5']:.3f}, "
          f"{fs['heldout_precision_p97_5']:.3f}] vs in-sample "
          f"{fs['in_sample_precision_shipped_tau']:.3f}\n")
    return out


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
