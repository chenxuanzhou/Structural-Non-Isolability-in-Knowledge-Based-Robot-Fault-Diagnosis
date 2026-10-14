# -*- coding: utf-8 -*-
"""Checks on the load-torque certificate, the fixed configuration and the
label budget.

The load-torque group is axisweight + losecan + varweight (voraus categories
1, 6, 7). The fixed configuration is the dimension set mag + sign + step that
repair.py freezes and the paper reports (0.309 two-stage top-1, 0.213 balanced
recall, 0.869 within the group). Sections, in the order they are written to
results/certificate_checks.json:

  searched_dims      dimensions the certificate's search selects for the group,
                     per split (seeds 0-4 and 0-19)
  nested_fixed       on each paper split, the dimensions chosen on a majority
                     of the other four splits, scored like the fixed one
  routing_fixed      held-out cycles whose rank-1 mode is in the group: how
                     many are group faults and how the resolver does on them
  group_pairs_fixed  per-mode recall and pairwise accuracy inside the group
  dim_subsets        all 32 subsets of the repair dimensions on top of mag
  mode_signature     line_kg.retrieve with w_signature 0.2 against 0 on the
                     paper's splits
  label_budget       flat control against the two two-stage arms at matched
                     label budgets, paired over twenty splits

Splits, features, the classifier and the ranking are imported from repair.py,
baselines.py and line_kg.py. Protocol: leave-setting-out
(repair._grouped_split); the paper's splits are seeds 0-4.

Run:
    python certificate_checks.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from collections import Counter

import numpy as np

import baselines as B
import repair as R

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_OUT = os.path.join(_ROOT, "results", "certificate_checks.json")
GROUP = "axisweight+losecan+varweight"
FIXED = ["mag", "sign", "step"]
PAPER_SEEDS = (0, 1, 2, 3, 4)
WIDE_SEEDS = tuple(range(20))
BUDGETS = (3, 5, 8, 12, 16, 320)
W_SIGNATURE = 0.2
COST_ORDER = ["deployed", "free", "pipeline", "hardware"]


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def _context():
    ctx = B._setup()
    testable, _ = R.testable_clusters()
    ctx["testable"] = testable
    members, cats = next((m, c) for m, c in testable
                         if "+".join(x.replace("FM_", "") for x in m) == GROUP)
    ctx["g_members"], ctx["g_cats"] = members, cats
    ctx["g_rep"] = ctx["rep"][members[0]]
    ctx["g_name"] = {c: f.replace("FM_", "") for c, f in sorted(cats.items())}
    # rank-1 mode, declaration order among ties (= resolve()["ranked"][0])
    ctx["top1"] = {i: B._tied(ctx, i)[0] for i in ctx["ids"]}
    return ctx


def _split(ctx, seed):
    """Leave-setting-out split, plus its group cycles on each side."""
    fit, test = R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], seed)
    g = ctx["g_cats"]
    return (fit, test, [i for i in fit if ctx["cat_of"][i] in g],
            [i for i in test if ctx["cat_of"][i] in g])


def _cost(dims):
    return max((R.GRAMMAR[d]["cost"] for d in dims if d != "mag"),
               key=COST_ORDER.index, default="deployed")


def _key(added):
    """Canonical name of a set of repair dimensions (REPAIR_DIMS order)."""
    return "+".join(d for d in R.REPAIR_DIMS if d in added) or "none"


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def score_fixed(ctx, dims, seed):
    """Score one dimension set on one split as evaluate(fixed_dims=...) does.

    A resolver is fitted on the group's fit-half cycles. A held-out cycle whose
    rank-1 mode lies in the group is relabelled by the resolver; every other
    cycle keeps its rank-1 mode. Only the load-torque group gets a resolver,
    as in evaluate's fixed configuration.
    """
    cyc, col, cat_of = ctx["cycles"], ctx["columns"], ctx["cat_of"]
    _, test, cf, ct = _split(ctx, seed)
    yf = np.array([cat_of[i] for i in cf])
    yt = np.array([cat_of[i] for i in ct])
    Xf, _ = R.matrix(cyc, cf, col, dims)
    Xt, _ = R.matrix(cyc, ct, col, dims)
    clf, pg = R._fit_predict(Xf, yf, Xt)
    routed = [i for i in test if ctx["rep"][ctx["top1"][i]] == ctx["g_rep"]]
    pred = {i: ctx["cat_of_fm"][ctx["top1"][i]] for i in test}
    if routed:
        Xr, _ = R.matrix(cyc, routed, col, dims)
        pred.update(zip(routed, (int(p) for p in clf.predict(Xr))))
    p = [pred[i] for i in test]
    y = [cat_of[i] for i in test]
    return {"seed": seed, "test": test, "routed": routed, "pred": pred,
            "group_true": yt.tolist(), "group_pred": [int(v) for v in pg],
            "group_proba": clf.predict_proba(Xt).tolist(),
            "group_classes": [int(c) for c in clf.classes_],
            "two_stage_top1": float(np.mean([a == b for a, b in zip(p, y)])),
            "balanced_recall": B._macro(p, y)[0],
            "group_acc": float(np.mean(pg == yt))}


def _stored_fixed():
    """The fixed-configuration numbers already in results/, for comparison."""
    out = {}
    try:
        with open(os.path.join(_ROOT, "results", "repair_results.json"), encoding="utf-8") as fh:
            fx = json.load(fh)["fixed_configuration"]
        out["two_stage_top1"] = fx["two_stage_top1"]
        out["group_acc"] = fx["cluster_acc"]
    except (OSError, KeyError):
        pass
    try:
        with open(os.path.join(_ROOT, "results", "baselines.json"), encoding="utf-8") as fh:
            out["balanced_recall"] = json.load(fh)["balanced_recall"]["two-stage"]["macro"]
    except (OSError, KeyError):
        pass
    return out


# ---------------------------------------------------------------------------
# 1. Searched dimensions per split
# ---------------------------------------------------------------------------
def _dim_summary(rows):
    use = [r for r in rows if not r.get("untrainable")]
    n = len(use)
    counts = {d: sum(d in r["added_dims"] for r in use) for d in R.REPAIR_DIMS}
    sets = Counter(_key(r["added_dims"]) for r in use)
    cost = Counter(r["cost_class"] for r in use)
    return {
        "seeds": [r["seed"] for r in use],
        "n_splits": n,
        "dim_counts": counts,
        "dim_freq": {d: c / n for d, c in counts.items()},
        "dim_set_counts": dict(sorted(sets.items(), key=lambda kv: (-kv[1], kv[0]))),
        "cost_class_counts": {c: cost[c] for c in COST_ORDER if cost[c]},
        "pipeline_freq": cost["pipeline"] / n,
        "n_adopted": sum(r["adopted"] for r in use),
        "acc_test_mean": _mean([r["acc_test"] for r in use]),
    }


def searched_dims(ctx, seeds=WIDE_SEEDS, verbose=True):
    """Run repair.certify on the group for each split, as evaluate() does."""
    cat_of = ctx["cat_of"]
    rows = []
    for seed in seeds:
        _, _, cf, ct = _split(ctx, seed)
        yf = np.array([cat_of[i] for i in cf])
        yt = np.array([cat_of[i] for i in ct])
        if len(cf) < 12 or len(ct) < 6 or len(set(yf)) < 2:   # evaluate()'s rule
            rows.append({"seed": seed, "untrainable": True})
            continue
        groups = np.array([ctx["set_of"][i] for i in cf])
        c = R.certify(ctx["cycles"], ctx["columns"], GROUP, ctx["g_members"],
                      cf, ct, yf, yt, groups, seed)
        trace = c["search_trace"]
        rows.append({
            "seed": seed, "added_dims": c["added_dims"], "dim_set": _key(c["added_dims"]),
            "cost_class": c["cost_class"], "adopted": c["adopted"],
            "acc_test": c["acc_test"], "acc_deployed_repr": c["acc_deployed_repr"],
            "inner_cv_fit": c["inner_cv_fit"],
            "inner_cv_deployed_repr": c["inner_cv_deployed_repr"],
            "search_trace": trace if isinstance(trace, str) else [list(t) for t in trace],
        })
    out = {"per_seed": rows,
           "paper_splits": _dim_summary([r for r in rows if r["seed"] in PAPER_SEEDS]),
           "all_splits": _dim_summary(rows),
           "note": "added_dims in greedy selection order; dim_set in REPAIR_DIMS order. "
                   "cost_class is the costliest class among the added dims."}
    if verbose:
        print("\n1. SEARCHED DIMENSIONS for the load-torque group (repair.certify)")
        for r in rows:
            if r.get("untrainable"):
                print(f"   seed {r['seed']:>2}: untrainable")
                continue
            print(f"   seed {r['seed']:>2}: {r['dim_set']:<22} cost {r['cost_class']:<9}"
                  f" held-out {r['acc_test']:.3f}  adopted {r['adopted']}")
        for tag in ("paper_splits", "all_splits"):
            s = out[tag]
            print(f"   {tag} (n={s['n_splits']}): counts {s['dim_counts']}  "
                  f"cost {s['cost_class_counts']}  pipeline {s['pipeline_freq']:.2f}")
    return out


# ---------------------------------------------------------------------------
# 2. Nested fixed configuration
# ---------------------------------------------------------------------------
def nested_fixed(ctx, searched, fixed_runs, verbose=True):
    """Choose the dimensions for split s from the other four paper splits only."""
    per_seed = {r["seed"]: r for r in searched["per_seed"]}
    rows = []
    for s in PAPER_SEEDS:
        others = [per_seed[t] for t in PAPER_SEEDS
                  if t != s and not per_seed[t].get("untrainable")]
        votes = {d: sum(d in r["added_dims"] for r in others) for d in R.REPAIR_DIMS}
        dims = ["mag"] + [d for d in R.REPAIR_DIMS if votes[d] > len(others) / 2]
        sc = score_fixed(ctx, dims, s)
        fx = fixed_runs[s]
        ret = _mean([ctx["cat_of_fm"][ctx["top1"][i]] == ctx["cat_of"][i]
                     for i in fx["test"]])
        rows.append({"seed": s, "votes": votes, "dims": dims, "cost_class": _cost(dims),
                     "two_stage_top1": sc["two_stage_top1"],
                     "balanced_recall": sc["balanced_recall"],
                     "group_acc": sc["group_acc"],
                     "fixed_two_stage_top1": fx["two_stage_top1"],
                     "fixed_balanced_recall": fx["balanced_recall"],
                     "fixed_group_acc": fx["group_acc"],
                     "retrieval_top1": ret,
                     "fixed_gain_over_retrieval": fx["two_stage_top1"] - ret})
    keys = ("two_stage_top1", "balanced_recall", "group_acc")
    out = {
        "rule": "for split s, a dimension is kept if the certificate's search selected it "
                "on more than half of the other four paper splits (3 or 4 of 4), the "
                "rule repair.adopt uses; mag is always kept. Scored like the fixed "
                "configuration: resolver for the load-torque group only, fitted on the "
                "group's fit-half cycles of split s.",
        "per_split": rows,
        "mean": {k: _mean([r[k] for r in rows]) for k in keys},
        "fixed_mean": {k: _mean([r["fixed_" + k] for r in rows]) for k in keys},
        "fixed_stored": _stored_fixed(),
        "caveat": "the other four splits' fit halves share settings with split s's "
                  "test half, so the vote is not fully independent of split s",
    }
    out["nested_minus_fixed"] = {k: out["mean"][k] - out["fixed_mean"][k] for k in keys}
    gain = [r["fixed_gain_over_retrieval"] for r in rows]
    out["fixed_gain_over_retrieval"] = {"per_split": gain, "min": float(min(gain)),
                                        "max": float(max(gain)), "mean": _mean(gain)}
    if verbose:
        print("\n2. NESTED FIXED CONFIGURATION (dims from a majority of the other four splits)")
        for r in rows:
            print(f"   seed {r['seed']}: {'+'.join(r['dims']):<18} two-stage "
                  f"{r['two_stage_top1']:.3f}  balanced {r['balanced_recall']:.3f}  "
                  f"group {r['group_acc']:.3f}")
        m, f = out["mean"], out["fixed_mean"]
        print(f"   nested mean : {m['two_stage_top1']:.3f} / {m['balanced_recall']:.3f} / "
              f"{m['group_acc']:.3f}")
        print(f"   fixed mean  : {f['two_stage_top1']:.3f} / {f['balanced_recall']:.3f} / "
              f"{f['group_acc']:.3f}   stored {out['fixed_stored']}")
    return out


# ---------------------------------------------------------------------------
# 3. Routing of the fixed configuration
# ---------------------------------------------------------------------------
def routing(ctx, fixed_runs, verbose=True):
    """What stage 1 sends to the group's resolver, and what comes out."""
    cat_of, g = ctx["cat_of"], ctx["g_cats"]
    rows, tot = [], Counter()
    for s, run in fixed_runs.items():
        routed = run["routed"]
        members = [i for i in routed if cat_of[i] in g]
        c = {"n_test": len(run["test"]), "n_routed": len(routed),
             "n_routed_members": len(members),
             "n_members_test": len(run["group_true"]),
             "n_route_correct": sum(run["pred"][i] == cat_of[i] for i in routed),
             "n_members_correct": sum(run["pred"][i] == cat_of[i] for i in members),
             "n_retrieval_correct": sum(ctx["cat_of_fm"][ctx["top1"][i]] == cat_of[i]
                                        for i in routed)}
        tot.update(c)
        rows.append(dict(seed=s, **c, **_route_rates(c)))
    keys = list(_route_rates(tot))
    out = {
        "definitions": {
            "routed": "held-out cycle whose rank-1 mode (declaration order) is in the "
                      "load-torque group",
            "member_frac": "routed cycles whose true category is in the group / routed",
            "route_acc": "routed cycles the resolver labels correctly / routed",
            "resolver_acc_members": "correct / routed cycles that are true group members",
            "retrieval_top1_routed": "rank-1 mode correct / routed (no resolver)",
            "member_recall": "routed true members / true members in the test half",
        },
        "per_split": rows,
        "mean": {k: _mean([r[k] for r in rows]) for k in keys},
        "pooled": dict(tot, **_route_rates(tot)),
    }
    if verbose:
        m = out["mean"]
        print("\n3. ROUTING of the fixed configuration (mean over the five splits)")
        print(f"   routed frac {m['routed_frac']:.3f}  member frac {m['member_frac']:.3f}  "
              f"route acc {m['route_acc']:.3f}  resolver on members "
              f"{m['resolver_acc_members']:.3f}  retrieval top-1 on routed "
              f"{m['retrieval_top1_routed']:.3f}  member recall {m['member_recall']:.3f}")
    return out


def _route_rates(c):
    return {"routed_frac": c["n_routed"] / c["n_test"],
            "member_frac": c["n_routed_members"] / c["n_routed"],
            "route_acc": c["n_route_correct"] / c["n_routed"],
            "resolver_acc_members": c["n_members_correct"] / c["n_routed_members"],
            "retrieval_top1_routed": c["n_retrieval_correct"] / c["n_routed"],
            "member_recall": c["n_routed_members"] / c["n_members_test"]}


# ---------------------------------------------------------------------------
# 4. Per-mode recall and pairwise accuracy inside the group
# ---------------------------------------------------------------------------
def _pair_scores(y, p, pair):
    """Accuracy and per-mode-balanced accuracy over cycles with y and p in `pair`."""
    m = np.isin(y, pair) & np.isin(p, pair)
    if not m.any():
        return {"n": 0, "n_correct": 0, "acc": None, "balanced": None}
    ym, pm = y[m], p[m]
    return {"n": int(m.sum()), "n_correct": int((pm == ym).sum()),
            "acc": float(np.mean(pm == ym)),
            "balanced": float(np.mean([np.mean(pm[ym == c] == c)
                                       for c in pair if (ym == c).any()]))}


def group_pairs(ctx, fixed_runs, verbose=True):
    """Within-group predictions of the fixed configuration, broken down."""
    cyc, col, cat_of = ctx["cycles"], ctx["columns"], ctx["cat_of"]
    cats = sorted(ctx["g_cats"])
    name = ctx["g_name"]
    pairs = list(itertools.combinations(cats, 2))
    pkey = {p: f"{name[p[0]]}_vs_{name[p[1]]}" for p in pairs}
    rows = []
    conf = np.zeros((len(cats), len(cats)), dtype=int)
    for s, run in fixed_runs.items():
        y, p = np.array(run["group_true"]), np.array(run["group_pred"])
        row = {"seed": s, "recall": {}, "pairs": {}, "pairs_binary": {},
               "pairs_restricted": {}}
        proba, classes = np.array(run["group_proba"]), run["group_classes"]
        for c in cats:
            m = y == c
            row["recall"][name[c]] = float(np.mean(p[m] == c)) if m.any() else None
        _, _, cf, ct = _split(ctx, s)
        for pr in pairs:
            row["pairs"][pkey[pr]] = dict(
                _pair_scores(y, p, pr),
                n_other_mode=int((np.isin(y, pr) & ~np.isin(p, pr)).sum()))
            # every held-out cycle of the pair, the 3-way resolver choosing
            # between the pair's two modes only
            m = np.isin(y, pr)
            cols = [classes.index(c) for c in pr]
            pr_pred = np.array(pr)[np.argmax(proba[m][:, cols], axis=1)]
            row["pairs_restricted"][pkey[pr]] = _pair_scores(y[m], pr_pred, pr)
            # a separate two-class resolver fitted on this pair's fit cycles only
            f = [i for i in cf if cat_of[i] in pr]
            t = [i for i in ct if cat_of[i] in pr]
            Xf, _ = R.matrix(cyc, f, col, FIXED)
            Xt, _ = R.matrix(cyc, t, col, FIXED)
            _, pb = R._fit_predict(Xf, np.array([cat_of[i] for i in f]), Xt)
            row["pairs_binary"][pkey[pr]] = _pair_scores(
                np.array([cat_of[i] for i in t]), pb, pr)
        for a, b in zip(y, p):
            conf[cats.index(a), cats.index(b)] += 1
        rows.append(row)

    def pair_mean(field, stat):
        return {pkey[pr]: _mean([r[field][pkey[pr]][stat] for r in rows
                                 if r[field][pkey[pr]][stat] is not None]) for pr in pairs}

    out = {
        "definition": "pairwise accuracy for modes (a, b): among held-out group cycles whose "
                      "true mode is a or b AND whose predicted mode is a or b, the fraction "
                      "predicted correctly. The 3-way resolver is used as is; cycles "
                      "predicted as the third mode are counted in n_other_mode, not scored. "
                      "'balanced' is the mean of the two modes' recalls on the same cycles.",
        "definition_binary": "pairs_binary: a separate two-class resolver (same dims) fitted "
                             "on the pair's fit-half cycles and scored on all of the pair's "
                             "held-out cycles",
        "definition_restricted": "pairs_restricted: all held-out cycles whose true mode is a "
                                 "or b, the 3-way resolver choosing between a and b only "
                                 "(argmax of its probabilities over the two). Unlike "
                                 "pair_acc it does not drop the cycles the resolver sends to "
                                 "the third mode, so it is the figure to quote.",
        "modes": {str(c): name[c] for c in cats},
        "per_split": rows,
        "recall_mean": {name[c]: _mean([r["recall"][name[c]] for r in rows
                                        if r["recall"][name[c]] is not None]) for c in cats},
        "pair_acc_mean": pair_mean("pairs", "acc"),
        "pair_balanced_mean": pair_mean("pairs", "balanced"),
        "binary_acc_mean": pair_mean("pairs_binary", "acc"),
        "restricted_acc_mean": pair_mean("pairs_restricted", "acc"),
        "binary_balanced_mean": pair_mean("pairs_binary", "balanced"),
        "confusion_pooled": {"rows_true_cols_pred": [name[c] for c in cats],
                             "counts": conf.tolist()},
    }
    out["recall_pooled"] = {name[c]: float(conf[k, k] / conf[k].sum())
                            for k, c in enumerate(cats)}
    out["pair_acc_pooled"] = {}
    for a, b in pairs:
        ia, ib = cats.index(a), cats.index(b)
        sub = conf[np.ix_([ia, ib], [ia, ib])]
        out["pair_acc_pooled"][pkey[(a, b)]] = float(np.trace(sub) / sub.sum())
    if verbose:
        print("\n4. INSIDE THE GROUP (fixed configuration, five splits)")
        for label, key in (("recall, mean", "recall_mean"),
                           ("pairs, mean", "pair_acc_mean"),
                           ("pairs, pooled", "pair_acc_pooled"),
                           ("pairs, balanced", "pair_balanced_mean"),
                           ("binary, mean", "binary_acc_mean"),
                           ("restricted, mean", "restricted_acc_mean"),
                           ("binary, balanced", "binary_balanced_mean")):
            print(f"   {label:<17}" + "  ".join(f"{k} {v:.3f}" for k, v in out[key].items()))
    return out


# ---------------------------------------------------------------------------
# 5. All subsets of the repair dimensions
# ---------------------------------------------------------------------------
def dim_subsets(ctx, verbose=True):
    """Every subset of REPAIR_DIMS on top of mag, scored as a fixed configuration."""
    rows = []
    for r in range(len(R.REPAIR_DIMS) + 1):
        for combo in itertools.combinations(R.REPAIR_DIMS, r):
            dims = ["mag"] + list(combo)
            runs = [score_fixed(ctx, dims, s) for s in PAPER_SEEDS]
            rows.append({
                "dim_set": _key(combo), "added_dims": list(combo),
                "cost_class": _cost(dims),
                "n_features": len(R.feature_names(ctx["columns"], dims)),
                "group_acc": _mean([x["group_acc"] for x in runs]),
                "group_acc_per_split": [x["group_acc"] for x in runs],
                "two_stage_top1": _mean([x["two_stage_top1"] for x in runs]),
                "balanced_recall": _mean([x["balanced_recall"] for x in runs]),
            })
    rows.sort(key=lambda x: (-x["group_acc"], len(x["added_dims"]), x["dim_set"]))
    for k, x in enumerate(rows, 1):
        x["rank"] = k
    by = {x["dim_set"]: x for x in rows}
    best_unread = next(x for x in rows if "unread" in x["added_dims"])
    out = {
        "ranking": "by mean within-group held-out accuracy over seeds 0-4, descending; "
                   "ties broken by fewer dims, then name",
        "ranked": rows,
        "best": {"dim_set": rows[0]["dim_set"], "group_acc": rows[0]["group_acc"]},
        "sign_step": {"dim_set": "sign+step", "rank": by["sign+step"]["rank"],
                      "group_acc": by["sign+step"]["group_acc"]},
        "best_with_unread": {"dim_set": best_unread["dim_set"], "rank": best_unread["rank"],
                             "group_acc": best_unread["group_acc"]},
        "mag_only": {"rank": by["none"]["rank"], "group_acc": by["none"]["group_acc"]},
        "n_subsets": len(rows),
    }
    if verbose:
        print("\n5. ALL 32 SUBSETS on top of mag (within-group held-out accuracy, top 10)")
        for x in rows[:10]:
            flag = "  <- fixed" if x["dim_set"] == "sign+step" else ""
            print(f"   {x['rank']:>2}. {x['dim_set']:<30} {x['group_acc']:.3f}  "
                  f"({x['cost_class']}){flag}")
        print(f"   sign+step rank {out['sign_step']['rank']}  "
              f"{out['sign_step']['group_acc']:.3f}; best with unread "
              f"{out['best_with_unread']['dim_set']} rank {out['best_with_unread']['rank']} "
              f"{out['best_with_unread']['group_acc']:.3f}; mag only "
              f"{out['mag_only']['group_acc']:.3f}")
    return out


# ---------------------------------------------------------------------------
# 6. Mode-level signature term on the paper's protocol
# ---------------------------------------------------------------------------
def _macro_credit(credit, y):
    per = {}
    for c, t in zip(credit, y):
        per.setdefault(t, []).append(c)
    return float(np.mean([np.mean(v) for v in per.values()]))


def mode_signature(ctx, w_on=W_SIGNATURE, verbose=True):
    """retrieve() with w_signature = w_on against 0: ties, top-1, balanced recall."""
    from line_kg import retrieve, FMEA
    ids, cat_of, cat_of_fm = ctx["ids"], ctx["cat_of"], ctx["cat_of_fm"]
    out = {}
    for tag, w in (("off", 0.0), ("on", w_on)):
        scores = ctx["scores"] if w == 0.0 else {
            i: {c["id"]: c["score"]
                for c in retrieve(ctx["recs"][i], top_k=len(FMEA), w_signature=w)}
            for i in ids}
        sub = dict(ctx, scores=scores)
        tied = {i: B._tied(sub, i) for i in ids}
        dec = {i: cat_of_fm[tied[i][0]] for i in ids}
        # expected accuracy under a uniform random tie-break
        credit = {i: sum(cat_of_fm[f] == cat_of[i] for f in tied[i]) / len(tied[i])
                  for i in ids}
        splits = []
        for s in PAPER_SEEDS:
            _, test = R._grouped_split(ids, cat_of, ctx["set_of"], s)
            y = [cat_of[i] for i in test]
            splits.append({
                "seed": s,
                "top1_declaration": float(np.mean([dec[i] == cat_of[i] for i in test])),
                "top1_tie_aware": float(np.mean([credit[i] for i in test])),
                "balanced_declaration": B._macro([dec[i] for i in test], y)[0],
                "balanced_tie_aware": _macro_credit([credit[i] for i in test], y)})
        metrics = [k for k in splits[0] if k != "seed"]
        out[tag] = {
            "w_signature": w,
            "n_tied": int(sum(len(t) > 1 for t in tied.values())),
            "tie_sizes": {str(k): int(v) for k, v in
                          sorted(Counter(len(t) for t in tied.values()).items())},
            "full_set_top1_declaration": float(np.mean([dec[i] == cat_of[i] for i in ids])),
            "full_set_top1_tie_aware": float(np.mean([credit[i] for i in ids])),
            "per_split": splits,
            "mean": {k: _mean([r[k] for r in splits]) for k in metrics},
        }
    metrics = list(out["off"]["mean"])
    out["n_cycles"] = len(ids)
    out["splits_lower_with_signature"] = {
        k: int(sum(a[k] < b[k] for a, b in zip(out["on"]["per_split"],
                                               out["off"]["per_split"])))
        for k in metrics}
    out["on_minus_off"] = {k: out["on"]["mean"][k] - out["off"]["mean"][k] for k in metrics}
    out["definitions"] = {
        "tied": "more than one mode within baselines.TIE_EPS of the rank-1 score",
        "declaration": "ties broken by FMEA declaration order, as retrieve() does",
        "tie_aware": "a cycle scores (true modes in the tied set) / (tied set size)",
    }
    if verbose:
        print(f"\n6. MODE-LEVEL SIGNATURE (w_signature {w_on} vs 0), seeds 0-4")
        for tag in ("off", "on"):
            d = out[tag]
            m = d["mean"]
            print(f"   {tag:>3}: ties {d['n_tied']}/{len(ids)}  full-set top-1 "
                  f"{d['full_set_top1_declaration']:.3f}  split top-1 "
                  f"{m['top1_declaration']:.3f} (tie-aware {m['top1_tie_aware']:.3f})  "
                  f"balanced {m['balanced_declaration']:.3f} "
                  f"(tie-aware {m['balanced_tie_aware']:.3f})")
        print(f"   splits lower with the signature: {out['splits_lower_with_signature']}")
    return out


# ---------------------------------------------------------------------------
# 7. Label budget, paired over twenty splits
# ---------------------------------------------------------------------------
ARMS = ("stage1_only", "prior_at_k", "control", "targeted", "unrestricted")
METRICS = ("top1", "balanced_recall", "balanced_recall_tied", "group_acc")
PAIRED = ("top1", "balanced_recall", "balanced_recall_tied")


def _scores(ctx, test, pred, top1, stage1=None):
    """Top-1, balanced recall and group (cluster) accuracy of one prediction list.

    The group of a prediction is the cluster of its mode; a stage-1 prediction
    of a mode with no voraus category keeps that mode's cluster.
    balanced_recall_tied scores the predictions flagged in `stage1` (the
    retrieval's rank-1 mode, declaration order) by expected credit over the
    rank-1 tied set, as baselines.balanced_recall_expected does, and every
    other prediction by a hard hit. Without `stage1` it equals balanced_recall.
    """
    y = [ctx["cat_of"][i] for i in test]
    grp = [ctx["rep"][ctx["fm_of_cat"][p]] if p in ctx["fm_of_cat"] else ctx["rep"][f]
           for p, f in zip(pred, top1)]
    stage1 = stage1 or [False] * len(test)
    credit = [B._tie_credit(ctx, B._tied(ctx, i), t) if s1 else float(p == t)
              for i, p, t, s1 in zip(test, pred, y, stage1)]
    return {"top1": float(np.mean([a == b for a, b in zip(pred, y)])),
            "balanced_recall": B._macro(pred, y)[0],
            "balanced_recall_tied": B._macro_credit(credit, y)[0],
            "group_acc": float(np.mean([a == ctx["true_cluster"][i]
                                        for a, i in zip(grp, test)]))}


def _paired(a, b, tol=1e-12):
    d = np.asarray(a) - np.asarray(b)
    return {"n": int(len(d)), "mean": _mean(d), "sd": _sd(d),
            "control_wins": int((d > tol).sum()), "control_losses": int((d < -tol).sum()),
            "ties": int((np.abs(d) <= tol).sum())}


def label_budget(ctx, sizes=BUDGETS, seeds=WIDE_SEEDS, verbose=True):
    """repair.label_curve_vs_control's arms, kept per split so they can be paired.

    Per split: the fit half is shuffled with RandomState(seed). The control is
    a flat 12-way classifier on the first k shuffled cycles, all grammar dims.
    'targeted' spends up to k labels inside EACH testable twin group (the first
    k shuffled fit cycles of that group); 'unrestricted' uses only those of the
    first k shuffled cycles that fall in a group. A group gets a resolver
    (mag + sign + step) when its labels contain two classes. As in
    label_curve_vs_control, a k whose labels hold one class is skipped for all
    arms on that split.
    """
    cyc, col, cat_of, rep = ctx["cycles"], ctx["columns"], ctx["cat_of"], ctx["rep"]
    allg = list(R.GRAMMAR)
    rows, full = [], []
    for seed in seeds:
        fit, test = R._grouped_split(ctx["ids"], cat_of, ctx["set_of"], seed)
        top1 = [ctx["top1"][i] for i in test]
        kb = [ctx["cat_of_fm"][f] for f in top1]
        rng = np.random.RandomState(seed)
        shuf = [fit[j] for j in rng.permutation(len(fit))]
        Xte, _ = R.matrix(cyc, test, col, allg)
        Xte_d, _ = R.matrix(cyc, test, col, FIXED)
        stage1 = _scores(ctx, test, kb, top1, [True] * len(test))

        # control on the whole fit half, for reference
        Xf, _ = R.matrix(cyc, fit, col, allg)
        _, pf = R._fit_predict(Xf, np.array([cat_of[i] for i in fit]), Xte)
        full.append(dict(seed=seed, n_fit=len(fit),
                         **_scores(ctx, test, [int(v) for v in pf], top1)))

        for k in sizes:
            if k > len(shuf):
                continue
            bud = shuf[:k]
            ytr = np.array([cat_of[i] for i in bud])
            row = {"seed": seed, "k": k, "n_test": len(test)}
            if len(set(ytr)) < 2:
                row["skipped"] = "one class among the k labels"
                rows.append(row)
                continue
            row["stage1_only"] = stage1
            maj = Counter(ytr.tolist()).most_common(1)[0][0]
            row["prior_at_k"] = _scores(ctx, test, [maj] * len(test), top1)
            Xtr, _ = R.matrix(cyc, bud, col, allg)
            _, pc = R._fit_predict(Xtr, ytr, Xte)
            row["control"] = dict(_scores(ctx, test, [int(v) for v in pc], top1),
                                  labels=k)
            for regime in ("targeted", "unrestricted"):
                resolvers, drawn, fitted, n_sub = {}, 0, 0, {}
                for members, cats in ctx["testable"]:
                    if regime == "targeted":
                        sub = [i for i in shuf if cat_of[i] in cats][:k]
                    else:
                        sub = [i for i in bud if cat_of[i] in cats]
                    drawn += len(sub)
                    if len(set(cat_of[i] for i in sub)) < 2:
                        continue
                    fitted += len(sub)
                    Xc, _ = R.matrix(cyc, sub, col, FIXED)
                    clf, _ = R._fit_predict(Xc, np.array([cat_of[i] for i in sub]), Xc)
                    resolvers[rep.get(sorted(members)[0])] = clf
                    n_sub[rep.get(sorted(members)[0])] = len(sub)
                pred = list(kb)
                from_kb = [True] * len(test)
                applied = 0
                for r, clf in resolvers.items():
                    idx = [j for j, f in enumerate(top1) if rep.get(f) == r]
                    if idx:
                        applied += n_sub[r]
                        for j, v in zip(idx, clf.predict(Xte_d[idx])):
                            pred[j] = int(v)
                            from_kb[j] = False
                row[regime] = dict(_scores(ctx, test, pred, top1, from_kb),
                                   labels_drawn=drawn,
                                   labels_fitted=fitted, labels_applied=applied,
                                   n_resolvers=len(resolvers))
            rows.append(row)

    summary = {}
    for k in sizes:
        ok = [r for r in rows if r["k"] == k and "control" in r]
        if not ok:
            continue

        def col_(arm, m, rs=ok):
            return [r[arm][m] for r in rs]

        def labels(rs):
            return {"control": float(k),
                    "targeted_drawn": _mean(col_("targeted", "labels_drawn", rs)),
                    "targeted_fitted": _mean(col_("targeted", "labels_fitted", rs)),
                    "targeted_applied": _mean(col_("targeted", "labels_applied", rs)),
                    "unrestricted_drawn": _mean(col_("unrestricted", "labels_drawn", rs)),
                    "unrestricted_fitted": _mean(col_("unrestricted", "labels_fitted", rs)),
                    "unrestricted_applied": _mean(col_("unrestricted", "labels_applied", rs))}

        paper = [r for r in ok if r["seed"] in PAPER_SEEDS]
        summary[str(k)] = {
            "n_splits": len(ok),
            "n_skipped": sum(1 for r in rows if r["k"] == k and "skipped" in r),
            "mean": {a: {m: _mean(col_(a, m)) for m in METRICS} for a in ARMS},
            "sd": {a: {m: _sd(col_(a, m)) for m in METRICS} for a in ARMS},
            "control_minus_targeted": {m: _paired(col_("control", m), col_("targeted", m))
                                       for m in PAIRED},
            "control_minus_unrestricted": {m: _paired(col_("control", m),
                                                      col_("unrestricted", m))
                                           for m in PAIRED},
            "labels_mean": labels(ok),
            "paper_splits_mean": {a: {m: _mean(col_(a, m, paper)) for m in METRICS}
                                  for a in ARMS},
            "paper_splits_labels_mean": labels(paper),
        }
    full_paper = [r for r in full if r["seed"] in PAPER_SEEDS]
    out = {
        "seeds": list(seeds), "sizes": list(sizes), "resolver_dims": FIXED,
        "definitions": {
            "control_minus_targeted": "per split, control minus targeted two-stage; "
                                      "sd is the sample sd (ddof=1); control_wins counts "
                                      "splits where the control is higher",
            "labels_drawn": "labels the arm spends in testable twin groups (targeted: up "
                            "to k per group, summed over groups)",
            "labels_fitted": "labels in groups whose resolver was fitted (two classes "
                             "present)",
            "labels_applied": "labels in resolvers that receive at least one held-out "
                              "cycle; no cycle is ever routed to the collision or gripping "
                              "group, so only load-torque labels can change a prediction "
                              "and this count never exceeds k",
            "paper_splits_mean": "the same means over seeds 0-4 only",
            "group_acc": "the predicted mode's twin cluster equals the true one",
            "balanced_recall_tied": "balanced recall with each stage-1 prediction "
                                    "scored by expected credit over the rank-1 tied "
                                    "set (the form the paper uses); resolver and "
                                    "control predictions are hard hits",
        },
        "per_split": rows,
        "summary": summary,
        "control_full_fit": {"per_split": full,
                             "mean": {m: _mean([r[m] for r in full]) for m in METRICS},
                             "paper_splits_mean": {m: _mean([r[m] for r in full_paper])
                                                   for m in METRICS}},
    }
    out["reproduces"] = _budget_reproduction(summary)
    if verbose:
        print(f"\n7. LABEL BUDGET, {len(seeds)} paired leave-setting-out splits "
              "(balanced recall with tied stage-1 modes sharing the credit)")
        print(f"   {'k':>4}  {'control':>8} {'targ':>7} {'unres':>7}   "
              f"{'bal ctl':>7} {'bal tg':>7}   {'d top1':>7} {'W/L/T':>8}   "
              f"{'d bal':>7} {'W/L/T':>8}   {'targ labels':>11}  {'ctl group':>11}")
        print(f"   {'':>4}  {'':>8} {'':>7} {'':>7}   {'':>7} {'':>7}   {'':>7} {'':>8}   "
              f"{'':>7} {'':>8}   {'fit (0-4)':>11}  {'(0-4)':>11}")
        for k, s in summary.items():
            m, d = s["mean"], s["control_minus_targeted"]
            wl = {x: f"{d[x]['control_wins']}/{d[x]['control_losses']}/{d[x]['ties']}"
                  for x in d}
            lab, lab5 = s["labels_mean"], s["paper_splits_labels_mean"]
            grp5 = s["paper_splits_mean"]["control"]["group_acc"]
            print(f"   {k:>4}  {m['control']['top1']:>8.3f} {m['targeted']['top1']:>7.3f} "
                  f"{m['unrestricted']['top1']:>7.3f}   "
                  f"{m['control']['balanced_recall_tied']:>7.3f} "
                  f"{m['targeted']['balanced_recall_tied']:>7.3f}   "
                  f"{d['top1']['mean']:>+7.3f} {wl['top1']:>8}   "
                  f"{d['balanced_recall_tied']['mean']:>+7.3f} "
                  f"{wl['balanced_recall_tied']:>8}   "
                  f"{lab['targeted_fitted']:>5.1f} ({lab5['targeted_fitted']:>4.1f})  "
                  f"{m['control']['group_acc']:.3f} ({grp5:.3f})")
        cf = out["control_full_fit"]["mean"]
        cf5 = out["control_full_fit"]["paper_splits_mean"]
        print(f"   control on the whole fit half: top-1 {cf['top1']:.3f} ({cf5['top1']:.3f})  "
              f"balanced {cf['balanced_recall']:.3f} ({cf5['balanced_recall']:.3f})  "
              f"group {cf['group_acc']:.3f} ({cf5['group_acc']:.3f})")
        print(f"   reproduction of stored curves (max |diff|): {out['reproduces']}")
    return out


def _budget_reproduction(summary):
    """Largest difference against results/label_budget_curve.json, where it overlaps."""
    try:
        with open(os.path.join(_ROOT, "results", "label_budget_curve.json"),
                  encoding="utf-8") as fh:
            st = json.load(fh)
    except OSError:
        return None
    five, twenty = [], []
    pairs = (("control_12way", "control"), ("two_stage_targeted", "targeted"),
             ("two_stage_unrestricted", "unrestricted"))
    for k, s in summary.items():
        for old, new in pairs:
            if k in st.get(old, {}):
                five.append(abs(st[old][k] - s["paper_splits_mean"][new]["top1"]))
            if k in st.get("macro", {}).get(old, {}):
                five.append(abs(st["macro"][old][k]
                                - s["paper_splits_mean"][new]["balanced_recall"]))
        cr = st.get("crossover_robustness", {}).get("per_k", {}).get(k)
        if cr and cr["n"] == s["n_splits"]:
            twenty.append(abs(cr["control_mean"] - s["mean"]["control"]["top1"]))
            twenty.append(abs(cr["two_stage_mean"] - s["mean"]["targeted"]["top1"]))
    return {"label_curve_seeds_0_4": max(five) if five else None,
            "crossover_robustness_seeds_0_19": max(twenty) if twenty else None}


# ---------------------------------------------------------------------------
def main(argv):
    t0 = time.time()
    ctx = _context()
    times = {}
    res = {"protocol": "leave-setting-out (repair._grouped_split); paper splits are "
                       "seeds 0-4",
           "n_cycles": len(ctx["ids"]),
           "group": {"name": GROUP, "members": ctx["g_members"],
                     "categories": {str(c): f for c, f in sorted(ctx["g_cats"].items())}},
           "fixed_dims": FIXED}
    print(f"{len(ctx['ids'])} anomalous cycles; group {GROUP}")

    t = time.time()
    res["searched_dims"] = searched_dims(ctx)
    times["searched_dims"] = time.time() - t

    t = time.time()
    fixed_runs = {s: score_fixed(ctx, FIXED, s) for s in PAPER_SEEDS}
    res["nested_fixed"] = nested_fixed(ctx, res["searched_dims"], fixed_runs)
    res["routing_fixed"] = routing(ctx, fixed_runs)
    res["group_pairs_fixed"] = group_pairs(ctx, fixed_runs)
    times["fixed"] = time.time() - t

    t = time.time()
    res["dim_subsets"] = dim_subsets(ctx)
    times["dim_subsets"] = time.time() - t

    t = time.time()
    res["mode_signature"] = mode_signature(ctx)
    times["mode_signature"] = time.time() - t

    t = time.time()
    res["label_budget"] = label_budget(ctx)
    times["label_budget"] = time.time() - t

    with open(RESULTS_OUT, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten -> {os.path.relpath(RESULTS_OUT, _ROOT)}")
    print("runtime (s): " + ", ".join(f"{k} {v:.0f}" for k, v in times.items())
          + f", total {time.time() - t0:.0f}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
