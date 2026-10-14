# -*- coding: utf-8 -*-
"""Trivial baselines for the deployed retrieval, and the effect of tie-breaking.

WHY THIS MODULE EXISTS
----------------------
Accuracy in this project was first reported against uniform chance: 1/14 =
0.071 top-1 (1/12 = 0.083 over the modes this dataset contains) and 0.150 at
the cluster level.  voraus-AD is skewed -- 156 cycles of category 1 against 10
of category 8 -- so uniform chance is too low a reference.  This module adds
the PREVALENCE PRIOR: rank the categories by how often they occur in the fit
half and give that same ranking for every cycle.  The prior reads no evidence.

On the five leave-setting-out splits the prior scores higher than the deployed
retrieval on top-1, top-3, top-5, MRR and cluster top-1.  The retrieval's top-1
is 0.91x the prior's and 2.37x uniform chance.

WHAT EACH ANALYSIS MEASURES
---------------------------
  1. prior_vs_retrieval -- the prior as a ranker, against the retrieval.

  2. balanced_recall -- mean per-category recall over the categories in the
     test half.  A constant predictor scores 1/K here however skewed the data
     (K = 10 or 11 under leave-setting-out, because three categories have a
     single instantiation and fall wholly on one side of the split), so the
     prior gains nothing from the skew.

  3. tiebreak_interval, tiebreak_interval_grouped -- 514 of 755 cycles have an
     exact rank-1 score tie, and the mode reported for a tied set is the one
     declared first in the knowledge base.  These two functions recompute
     top-1 under other declaration orders (three named orders and 200 random
     ones), on the full set and on the five splits of Table 1.

  4. tiebreak_exact -- the same interval without sampling.  Every rank-1 tied
     set lies inside one of two sets of modes, {losecan, axisweight,
     varweight} and {miscommutation, overtemp}, so top-1 depends only on the
     relative order inside those two sets: 3! x 2! = 12 orders, all of them
     enumerated.  Cluster top-1 is the same under every order.  This holds
     because no rank-1 tied set contains modes of two groups, so every order
     picks a mode from the same group.

  5. tie_robust -- top-1, top-3, top-5 and MRR under all 12 orders against
     the prior, and per-split paired differences under declaration order.

  6. balanced_recall_expected -- balanced recall with each tied set scored by
     expected credit (1/|T| per tied member), which equals the mean over the
     12 orders and so does not depend on the declaration order.

  7. constant_router -- sends every held-out cycle to the load-torque
     resolver, without stage 1, and compares it with the two-stage arm.

  8. set_valued -- takes the rank-1 group, or the rank-1 tied set, as the
     answer and compares how often it contains the true mode with the prior's
     top categories in a list of the same size.

Nothing in this module changes a deployed default; it only measures.  Splits,
features and the ranking are imported from `repair.py` / `line_kg.py`, so there
is one implementation of each.

Run:
    ..\\.venv\\Scripts\\python.exe baselines.py          # full report
    ..\\.venv\\Scripts\\python.exe baselines.py --quick  # skip the learned arms
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import repair as R

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_OUT = os.path.join(_ROOT, "results", "baselines.json")
SEEDS = (0, 1, 2, 3, 4)
TIE_EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def _setup():
    from eval_retrieve import load_cache
    from line_kg import retrieve, FMEA
    from distinguishability import cluster_map
    from integration_test import CATEGORY_TO_FM

    cycles, columns = R.load_features(verbose=False)
    recs = {str(r["cycle_id"]): r for r in load_cache()}
    ids = sorted(set(cycles) & set(recs), key=int)
    ctx = dict(
        cycles=cycles, columns=columns, ids=ids, recs=recs,
        cat_of={i: cycles[i]["category"] for i in ids},
        set_of={i: cycles[i]["setting"] for i in ids},
        fm_order=[f["id"] for f in FMEA],
        rep=cluster_map(),
        fm_of_cat={c: f for c, f in CATEGORY_TO_FM.items()},
    )
    ctx["scores"] = {i: {c["id"]: c["score"]
                         for c in retrieve(recs[i], top_k=len(FMEA))} for i in ids}
    ctx["cat_of_fm"] = {f: R._cat_of_fm(f) for f in ctx["fm_order"]}
    ctx["true_cluster"] = {i: ctx["rep"][ctx["fm_of_cat"][ctx["cat_of"][i]]] for i in ids}
    return ctx


def _tied(ctx, i):
    """The rank-1 tied set, in FMEA declaration order."""
    s = ctx["scores"][i]
    top = max(s.values())
    return [f for f in ctx["fm_order"] if abs(s[f] - top) <= TIE_EPS]


def _ranked_cats(ctx, i):
    """Candidate categories in score order; declaration order breaks ties."""
    s = ctx["scores"][i]
    order = sorted(ctx["fm_order"], key=lambda f: (-s[f], ctx["fm_order"].index(f)))
    return [ctx["cat_of_fm"][f] for f in order]


def _rank_metrics(rank_lists, ys):
    t1 = t3 = t5 = mrr = 0.0
    for rl, y in zip(rank_lists, ys):
        if y not in rl:
            continue
        r = rl.index(y) + 1
        t1 += r <= 1
        t3 += r <= 3
        t5 += r <= 5
        mrr += 1.0 / r
    n = len(ys)
    return dict(top1=t1 / n, top3=t3 / n, top5=t5 / n, mrr=mrr / n)


def _macro(pred, y):
    per = defaultdict(list)
    for p, t in zip(pred, y):
        per[t].append(p == t)
    return (float(np.mean([np.mean(v) for v in per.values()])),
            {int(c): float(np.mean(v)) for c, v in per.items()})


def _macro_credit(credit, y):
    """Macro recall from per-cycle credit in [0, 1] instead of hard hits."""
    per = defaultdict(list)
    for c, t in zip(credit, y):
        per[t].append(c)
    return (float(np.mean([np.mean(v) for v in per.values()])),
            {int(c): float(np.mean(v)) for c, v in per.items()})


def _prior_rank(ctx, fit):
    """Categories by fit-half frequency, then the unseen ones in numeric order."""
    cnt = Counter(ctx["cat_of"][i] for i in fit)
    rank = [c for c, _ in cnt.most_common()]
    return rank + [c for c in sorted(set(ctx["cat_of"].values())) if c not in rank]


def _fit_resolvers(ctx, fit, dims, testable):
    """One within-group classifier per testable group, fitted on FIT only.

    Keyed by the group representative. A group is skipped when FIT holds fewer
    than six of its cycles or only one of its categories.
    """
    resolvers = {}
    for members, cats in testable:
        sub = [i for i in fit if ctx["cat_of"][i] in cats]
        if len(sub) < 6 or len(set(ctx["cat_of"][i] for i in sub)) < 2:
            continue
        Xc, _ = R.matrix(ctx["cycles"], sub, ctx["columns"], dims)
        clf, _ = R._fit_predict(Xc, np.array([ctx["cat_of"][i] for i in sub]), Xc)
        resolvers[ctx["rep"][sorted(members)[0]]] = clf
    return resolvers


# ---------------------------------------------------------------------------
# 1. The prevalence prior, as a ranker
# ---------------------------------------------------------------------------
def prior_vs_retrieval(ctx, verbose=True):
    """Retrieval against a constant ranking by fit-half category prevalence."""
    ours, prior, clus = [], [], []
    for seed in SEEDS:
        fit, test = R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], seed)
        ys = [ctx["cat_of"][i] for i in test]
        ours.append(_rank_metrics([_ranked_cats(ctx, i) for i in test], ys))
        prior.append(_rank_metrics([_prior_rank(ctx, fit)] * len(test), ys))

        maj_cl = Counter(ctx["true_cluster"][i] for i in fit).most_common(1)[0][0]
        clus.append(dict(
            ours=float(np.mean([ctx["rep"][_tied(ctx, i)[0]] == ctx["true_cluster"][i]
                                for i in test])),
            majority=float(np.mean([ctx["true_cluster"][i] == maj_cl for i in test]))))

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    out = {
        "retrieval": {k: mean(ours, k) for k in ("top1", "top3", "top5", "mrr")},
        "prevalence_prior": {k: mean(prior, k) for k in ("top1", "top3", "top5", "mrr")},
        "uniform_chance": dict(top1=1 / 14, top3=3 / 14, top5=5 / 14,
                               mrr=sum(1 / r for r in range(1, 15)) / 14),
        "cluster_retrieval": mean(clus, "ours"),
        "cluster_majority": mean(clus, "majority"),
        "cluster_uniform_chance": 0.150,
    }
    if verbose:
        print("\n1. THE MISSING BASELINE - a constant ranking by category prevalence")
        print("                          top-1    top-3    top-5      MRR")
        for name, key in (("retrieval (deployed)", "retrieval"),
                          ("prevalence prior", "prevalence_prior"),
                          ("uniform chance (paper)", "uniform_chance")):
            d = out[key]
            print(f"   {name:<22}  {d['top1']:>6.3f}   {d['top3']:>6.3f}   "
                  f"{d['top5']:>6.3f}   {d['mrr']:>6.3f}")
        lift_p = out["retrieval"]["top1"] / out["prevalence_prior"]["top1"]
        lift_u = out["retrieval"]["top1"] / out["uniform_chance"]["top1"]
        print(f"   -> the prior wins on all four. The retrieval's lift over it is "
              f"{lift_p:.2f}x,")
        print(f"      where the paper reports {lift_u:.2f}x over uniform chance.")
        print(f"   cluster top-1: retrieval {out['cluster_retrieval']:.3f}  vs  "
              f"majority cluster {out['cluster_majority']:.3f}  "
              f"(paper's chance {out['cluster_uniform_chance']:.3f})")
    return out


# ---------------------------------------------------------------------------
# 2. Balanced recall - the metric a prior cannot game
# ---------------------------------------------------------------------------
def balanced_recall(ctx, quick=False, verbose=True):
    """Micro top-1 and macro recall for prior / retrieval / two-stage / control."""
    dims = ["mag", "sign", "step"]          # the fixed shipping configuration
    testable, _ = R.testable_clusters()
    mi, ma = defaultdict(list), defaultdict(list)
    per_cat = defaultdict(lambda: defaultdict(list))
    const_bound = []          # 1/K, K = categories present in the test half

    for seed in SEEDS:
        fit, test = R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], seed)
        y = [ctx["cat_of"][i] for i in test]
        const_bound.append(1.0 / len(set(y)))
        maj = Counter(ctx["cat_of"][i] for i in fit).most_common(1)[0][0]
        arms = {"prior (constant)": [maj] * len(test),
                "retrieval": [ctx["cat_of_fm"][_tied(ctx, i)[0]] for i in test]}

        if not quick:
            resolvers = _fit_resolvers(ctx, fit, dims, testable)
            two = []
            for i in test:
                top1 = _tied(ctx, i)[0]
                p = ctx["cat_of_fm"][top1]
                clf = resolvers.get(ctx["rep"][top1])
                if clf is not None:
                    Xi, _ = R.matrix(ctx["cycles"], [i], ctx["columns"], dims)
                    p = int(clf.predict(Xi)[0])
                two.append(p)
            arms["two-stage"] = two

            Xtr, _ = R.matrix(ctx["cycles"], fit, ctx["columns"], list(R.GRAMMAR))
            Xte, _ = R.matrix(ctx["cycles"], test, ctx["columns"], list(R.GRAMMAR))
            _, pc = R._fit_predict(Xtr, np.array([ctx["cat_of"][i] for i in fit]), Xte)
            arms["flat control"] = list(pc)

        for name, pred in arms.items():
            mi[name].append(float(np.mean([a == b for a, b in zip(pred, y)])))
            m, per = _macro(pred, y)
            ma[name].append(m)
            for c, v in per.items():
                per_cat[name][c].append(v)

    out = {n: dict(micro=float(np.mean(mi[n])), macro=float(np.mean(ma[n])),
                   per_category={str(c): float(np.mean(v))
                                 for c, v in sorted(per_cat[n].items())})
           for n in mi}
    # a uniform draw over the 14 modes recalls each category with probability
    # 1/14, the same chance model as uniform_chance's top-1
    out["uniform_macro"] = 1.0 / len(ctx["fm_order"])
    out["constant_macro_bound"] = float(np.mean(const_bound))
    if verbose:
        print("\n2. BALANCED RECALL - mean per-category recall; a constant scores "
              f"1/K per split whatever the skew (K = categories in the test half; "
              f"mean {out['constant_macro_bound']:.3f}), uniform over 14 modes scores "
              f"{out['uniform_macro']:.3f}")
        print("                          micro top-1    macro recall")
        for n in ("prior (constant)", "retrieval", "two-stage", "flat control"):
            if n in out:
                print(f"   {n:<22}  {out[n]['micro']:>8.3f}     {out[n]['macro']:>8.3f}")
        if "flat control" in out:
            print("\n   per-category recall - the retrieval is 0.000 on 7 of 12 categories")
            print("     cat    prior   retrieval   two-stage   control")
            for c in sorted(out["retrieval"]["per_category"], key=int):
                print(f"     {c:>3}    " + "       ".join(
                    f"{out[n]['per_category'][c]:.3f}"
                    for n in ("prior (constant)", "retrieval", "two-stage", "flat control")))
    return out


# ---------------------------------------------------------------------------
# 3. The tie-break interval
# ---------------------------------------------------------------------------
def tiebreak_interval(ctx, draws=200, verbose=True):
    """How much of the headline top-1 is decided by an arbitrary list order."""
    ids, cat_of, cat_of_fm = ctx["ids"], ctx["cat_of"], ctx["cat_of_fm"]
    fm_order = ctx["fm_order"]
    tsets = {i: _tied(ctx, i) for i in ids}

    def acc(key):
        return float(np.mean([cat_of_fm[min(tsets[i], key=key)] == cat_of[i]
                              for i in ids]))

    rand = []
    for d in range(draws):
        rng = np.random.RandomState(d)
        perm = {f: rng.rand() for f in fm_order}
        rand.append(acc(lambda f: perm[f]))

    ceiling = float(np.mean([cat_of[i] in [cat_of_fm[f] for f in tsets[i]] for i in ids]))
    cl = {name: float(np.mean([ctx["rep"][min(tsets[i], key=key)] ==
                               ctx["true_cluster"][i] for i in ids]))
          for name, key in (("declaration", lambda f: fm_order.index(f)),
                            ("alphabetical", lambda f: f),
                            ("reverse", lambda f: -fm_order.index(f)))}
    out = {
        "n_tied": int(sum(1 for i in ids if len(tsets[i]) > 1)),
        "n_total": len(ids),
        "tie_sizes": {str(k): int(v) for k, v in
                      sorted(Counter(len(t) for t in tsets.values()).items())},
        "declaration_order": acc(lambda f: fm_order.index(f)),
        "alphabetical": acc(lambda f: f),
        "reverse_declaration": acc(lambda f: -fm_order.index(f)),
        "random_mean": float(np.mean(rand)), "random_sd": float(np.std(rand)),
        "random_min": float(min(rand)), "random_max": float(max(rand)),
        "oracle_ceiling": ceiling,
        "cluster_by_tiebreak": cl,
        "cluster_invariant": len(set(round(v, 6) for v in cl.values())) == 1,
    }
    if verbose:
        print("\n3. THE TIE-BREAK INTERVAL - what the ranking does NOT determine")
        print(f"   {out['n_tied']}/{out['n_total']} cycles tie at rank 1 "
              f"({100 * out['n_tied'] / out['n_total']:.1f}%), tie sizes {out['tie_sizes']}")
        print(f"   declaration order (reported)   {out['declaration_order']:.4f}")
        print(f"   alphabetical by mode id        {out['alphabetical']:.4f}")
        print(f"   reverse declaration            {out['reverse_declaration']:.4f}")
        print(f"   random, {draws} draws             {out['random_mean']:.4f} +/- "
              f"{out['random_sd']:.4f}   [{out['random_min']:.3f}, {out['random_max']:.3f}]")
        print(f"   oracle tie-break (ceiling)     {out['oracle_ceiling']:.4f}")
        print("   -> the range across orders is wider than the split-to-split 95%")
        print("      interval and contains the prevalence prior.")
        print(f"   cluster assignment is tie-break INVARIANT: "
              f"{ {k: round(v, 4) for k, v in cl.items()} }")
        print("      -> no tied set contains modes of two groups.")
    return out


def tiebreak_interval_grouped(ctx, draws=200, verbose=True):
    """The tie-break interval on the protocol of Table 1.

    `tiebreak_interval` above sweeps the full 755-cycle set, whereas the prior
    and the retrieval are compared as five-split leave-setting-out means.  Here
    every ordering is applied to the held-out half of the SAME five splits and
    averaged, so the interval, the prior and the split-to-split CI are one
    quantity on one population.  An ordering is a property of the knowledge
    base, so draw d is the same permutation on every split.  `tiebreak_exact`
    enumerates the 12 relative orders that decide top-1, so it gives the same
    interval without sampling.
    """
    cat_of, cat_of_fm, fm_order = ctx["cat_of"], ctx["cat_of_fm"], ctx["fm_order"]
    tsets = {i: _tied(ctx, i) for i in ctx["ids"]}
    named = {"declaration": lambda f: fm_order.index(f),
             "alphabetical": lambda f: f,
             "reverse": lambda f: -fm_order.index(f)}
    keys = dict(named)
    for d in range(draws):
        rng = np.random.RandomState(d)
        perm = {f: rng.rand() for f in fm_order}
        keys[f"random_{d}"] = (lambda p: lambda f: p[f])(perm)

    top1, clus = defaultdict(list), defaultdict(list)
    oracle, prior, n_test = [], [], []
    for seed in SEEDS:
        fit, test = R._grouped_split(ctx["ids"], cat_of, ctx["set_of"], seed)
        n_test.append(len(test))
        for name, key in keys.items():
            picks = [min(tsets[i], key=key) for i in test]
            top1[name].append(float(np.mean([cat_of_fm[f] == cat_of[i]
                                             for f, i in zip(picks, test)])))
            clus[name].append(float(np.mean([ctx["rep"][f] == ctx["true_cluster"][i]
                                             for f, i in zip(picks, test)])))
        oracle.append(float(np.mean([cat_of[i] in [cat_of_fm[f] for f in tsets[i]]
                                     for i in test])))
        maj = Counter(cat_of[i] for i in fit).most_common(1)[0][0]
        prior.append(float(np.mean([cat_of[i] == maj for i in test])))

    def ci95(v):
        return 2.776 * float(np.std(v, ddof=1)) / np.sqrt(len(v))

    rand = [float(np.mean(top1[f"random_{d}"])) for d in range(draws)]
    every = [float(np.mean(v)) for v in top1.values()]
    within = [max(top1[n][s] for n in keys) - min(top1[n][s] for n in keys)
              for s in range(len(SEEDS))]
    decl = top1["declaration"]
    prior_mean = float(np.mean(prior))
    out = {
        "protocol": "leave-setting-out, five splits, held-out half; same splits as Table 1",
        "seeds": list(SEEDS), "n_test": n_test, "draws": draws,
        "declaration_order": float(np.mean(decl)),
        "alphabetical": float(np.mean(top1["alphabetical"])),
        "reverse_declaration": float(np.mean(top1["reverse"])),
        "random_mean": float(np.mean(rand)), "random_sd": float(np.std(rand)),
        "random_min": float(min(rand)), "random_max": float(max(rand)),
        "all_orderings_min": float(min(every)), "all_orderings_max": float(max(every)),
        "interval_width": float(max(every) - min(every)),
        "oracle_ceiling": float(np.mean(oracle)),
        "declaration_per_split": decl,
        "declaration_ci95": ci95(decl),
        "declaration_split_range": float(max(decl) - min(decl)),
        "within_split_width": within,
        "within_split_width_mean": float(np.mean(within)),
        "prior_top1": prior_mean,
        "frac_orderings_at_or_above_prior": float(np.mean([v >= prior_mean for v in every])),
        "cluster_by_tiebreak": {n: float(np.mean(clus[n])) for n in named},
        "cluster_invariant": all(len(set(round(clus[n][s], 9) for n in keys)) == 1
                                 for s in range(len(SEEDS))),
        "cluster_invariant_over": f"{len(keys)} orderings x {len(SEEDS)} splits",
    }
    if verbose:
        print("\n3b. THE TIE-BREAK INTERVAL ON TABLE 1's PROTOCOL "
              f"(leave-setting-out, {len(SEEDS)} splits, five-split means)")
        print(f"   declaration order (reported)   {out['declaration_order']:.4f}"
              f"   +/- {out['declaration_ci95']:.4f} (t 95% over splits)")
        print(f"   alphabetical by mode id        {out['alphabetical']:.4f}")
        print(f"   reverse declaration            {out['reverse_declaration']:.4f}")
        print(f"   random, {draws} draws             {out['random_mean']:.4f} +/- "
              f"{out['random_sd']:.4f}   [{out['random_min']:.3f}, {out['random_max']:.3f}]")
        print(f"   every ordering tried           [{out['all_orderings_min']:.3f}, "
              f"{out['all_orderings_max']:.3f}]  width {out['interval_width']:.3f}")
        print(f"   oracle tie-break (ceiling)     {out['oracle_ceiling']:.4f}")
        print(f"   prevalence prior               {prior_mean:.4f}   "
              f"({100 * out['frac_orderings_at_or_above_prior']:.1f}% of orderings reach it)")
        print(f"   within-split width, per split  {[round(w, 3) for w in within]}")
        print(f"   cluster assignment invariant over {out['cluster_invariant_over']}: "
              f"{out['cluster_invariant']}  "
              f"{ {k: round(v, 4) for k, v in out['cluster_by_tiebreak'].items()} }")
    return out


# ---------------------------------------------------------------------------
# 4. The exact tie-break interval
# ---------------------------------------------------------------------------
# Load-torque group: its representative in distinguishability.cluster_map.
LOAD_GROUP = "FM_axisweight"


def _splits(ctx):
    return [R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], s) for s in SEEDS]


def _tie_components(ctx):
    """Modes that share a rank-1 tie on some cycle, merged into connected sets."""
    decl = ctx["fm_order"]
    comps = []
    for i in ctx["ids"]:
        t = set(_tied(ctx, i))
        if len(t) < 2:
            continue
        for c in [c for c in comps if c & t]:
            comps.remove(c)
            t |= c
        comps.append(t)
    comps = [sorted(c, key=decl.index) for c in comps]
    return sorted(comps, key=lambda c: (-len(c), decl.index(c[0])))


def _relative_orders(ctx, comps):
    """Every mode order that differs from declaration order only inside components.

    The members of each component are permuted among the positions the component
    holds in the declaration order; every other mode keeps its position. The first
    order returned is the declaration order. Returns [(label, {mode: position})].
    """
    decl = ctx["fm_order"]
    out = []
    for perms in itertools.product(*(itertools.permutations(c) for c in comps)):
        slots = list(decl)
        for comp, perm in zip(comps, perms):
            for k, f in zip(sorted(decl.index(m) for m in comp), perm):
                slots[k] = f
        out.append(("; ".join(" > ".join(p) for p in perms),
                    {f: k for k, f in enumerate(slots)}))
    return out


def _tie_credit(ctx, tied, y):
    """Expected top-1 hit when the tied set is broken uniformly at random."""
    return sum(ctx["cat_of_fm"][f] == y for f in tied) / len(tied)


def tiebreak_exact(ctx, verbose=True):
    """Every top-1 value a tie-break order can produce, on Table 1's protocol.

    A rank-1 tied set only ever contains modes of one tie component, so top-1
    depends only on the relative order of the modes inside each component.
    Enumerating those orders gives the interval exactly; the random orders of
    `tiebreak_interval_grouped` are a sample of the same values.
    """
    cat_of, cat_of_fm, rep = ctx["cat_of"], ctx["cat_of_fm"], ctx["rep"]
    tsets = {i: _tied(ctx, i) for i in ctx["ids"]}
    cross = sum(1 for t in tsets.values() if len({rep[f] for f in t}) > 1)
    assert cross == 0, f"{cross} rank-1 tied sets contain modes of two groups"
    comps = _tie_components(ctx)
    orders = _relative_orders(ctx, comps)

    top1, clus = defaultdict(list), defaultdict(list)
    aware, prior = [], []
    for fit, test in _splits(ctx):
        for lab, pos in orders:
            picks = [min(tsets[i], key=pos.get) for i in test]
            top1[lab].append(float(np.mean([cat_of_fm[f] == cat_of[i]
                                            for f, i in zip(picks, test)])))
            clus[lab].append(float(np.mean([rep[f] == ctx["true_cluster"][i]
                                            for f, i in zip(picks, test)])))
        aware.append(float(np.mean([_tie_credit(ctx, tsets[i], cat_of[i]) for i in test])))
        top_cat = _prior_rank(ctx, fit)[0]
        prior.append(float(np.mean([cat_of[i] == top_cat for i in test])))

    for s in range(len(SEEDS)):
        assert len({round(clus[lab][s], 12) for lab, _ in orders}) == 1, \
            "cluster top-1 depends on the tie-break order"
    means = {lab: float(np.mean(top1[lab])) for lab, _ in orders}
    vals = list(means.values())
    exact_mean = float(np.mean(vals))
    assert abs(float(np.mean(aware)) - exact_mean) < 1e-12
    decl = top1[orders[0][0]]
    prior_top1 = float(np.mean(prior))
    out = {
        "protocol": "leave-setting-out, five splits, held-out half; same splits as Table 1",
        "seeds": list(SEEDS),
        "n_cycles": len(tsets),
        "n_rank1_tied": int(sum(1 for t in tsets.values() if len(t) > 1)),
        "cross_group_ties": int(cross),
        "tie_components": comps,
        "n_relative_orders": len(orders),
        "n_distinct_means": len({round(v, 12) for v in vals}),
        "exact_min": float(min(vals)),
        "exact_max": float(max(vals)),
        "exact_mean": exact_mean,
        "exact_sd_population": float(np.std(vals)),
        "per_order": [{"order": lab, "top1_mean": means[lab], "top1_per_split": top1[lab],
                       "cluster_top1": float(np.mean(clus[lab]))} for lab, _ in orders],
        "declaration_order": float(np.mean(decl)),
        "declaration_per_split": decl,
        "declaration_se": float(np.std(decl, ddof=1) / np.sqrt(len(decl))),
        "declaration_split_range": float(max(decl) - min(decl)),
        "prior_top1": prior_top1,
        "n_orders_at_or_above_prior": int(sum(v >= prior_top1 for v in vals)),
        "cluster_top1": float(np.mean(clus[orders[0][0]])),
        "cluster_invariant": True,
        "tie_aware_top1": {"per_split": aware, "mean": float(np.mean(aware))},
        "note": ("Orders are written component by component, first-listed mode wins "
                 "a tie; modes outside the components keep declaration order. The "
                 "first order is the declaration order. exact_sd_population is the "
                 "population s.d. over the orders. tie_aware_top1 gives 1/|T| credit "
                 "per tied member of the true category and equals exact_mean, because "
                 "over all orders each member of a tied set comes first equally often. "
                 "Cluster top-1 is the same under every order because no rank-1 "
                 "tied set contains modes of two groups."),
    }
    if verbose:
        print(f"\n4. EXACT TIE-BREAK INTERVAL - {len(orders)} relative orders inside "
              f"{len(comps)} tie components, {cross} tied sets across groups")
        for c in comps:
            print(f"   component: {', '.join(c)}")
        print(f"   top-1 over all orders   [{out['exact_min']:.4f}, {out['exact_max']:.4f}]"
              f"  mean {exact_mean:.4f}  s.d. {out['exact_sd_population']:.4f} "
              f"({out['n_distinct_means']} distinct values)")
        print(f"   tie-aware top-1         {out['tie_aware_top1']['mean']:.4f}")
        print(f"   declaration order       {out['declaration_order']:.4f}  "
              f"s.e. over splits {out['declaration_se']:.4f}")
        print(f"   prevalence prior        {prior_top1:.4f}  "
              f"({out['n_orders_at_or_above_prior']}/{len(orders)} orders reach it)")
        print(f"   cluster top-1           {out['cluster_top1']:.4f} under every order")
    return out


# ---------------------------------------------------------------------------
# 5. Ranking metrics under every order, paired against the prior
# ---------------------------------------------------------------------------
def tie_robust(ctx, verbose=True):
    """Does the prior's lead over the retrieval survive every tie-break order?"""
    cat_of, cat_of_fm, rep, decl = ctx["cat_of"], ctx["cat_of_fm"], ctx["rep"], ctx["fm_order"]
    metrics = ("top1", "top3", "top5", "mrr")
    orders = _relative_orders(ctx, _tie_components(ctx))
    tsets = {i: _tied(ctx, i) for i in ctx["ids"]}

    per_order = defaultdict(list)
    prior, decl_rows, cl_ret, cl_maj, mode_cl = [], [], [], [], []
    tb_top1, tb_macro = [], []
    for fit, test in _splits(ctx):
        ys = [cat_of[i] for i in test]
        for lab, pos in orders:
            ranked = [[cat_of_fm[f] for f in sorted(decl, key=lambda f: (-ctx["scores"][i][f],
                                                                         pos[f]))]
                      for i in test]
            per_order[lab].append(_rank_metrics(ranked, ys))
        decl_rows.append(per_order[orders[0][0]][-1])
        rank = _prior_rank(ctx, fit)
        prior.append(_rank_metrics([rank] * len(test), ys))

        maj_cl = Counter(ctx["true_cluster"][i] for i in fit).most_common(1)[0][0]
        cl_ret.append(float(np.mean([rep[tsets[i][0]] == ctx["true_cluster"][i]
                                     for i in test])))
        cl_maj.append(float(np.mean([ctx["true_cluster"][i] == maj_cl for i in test])))
        top_cl = rep[ctx["fm_of_cat"][rank[0]]]
        mode_cl.append(float(np.mean([ctx["true_cluster"][i] == top_cl for i in test])))

        # ties broken by fit-half category frequency, most frequent first
        cnt = Counter(cat_of[i] for i in fit)
        pred = [cat_of_fm[min(tsets[i], key=lambda f: (-cnt.get(cat_of_fm[f], 0),
                                                      decl.index(f)))] for i in test]
        tb_top1.append(float(np.mean([p == t for p, t in zip(pred, ys)])))
        tb_macro.append(_macro(pred, ys)[0])

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    order_means = {lab: {k: mean(per_order[lab], k) for k in metrics} for lab, _ in orders}
    prior_means = {k: mean(prior, k) for k in metrics}
    ref = prior_vs_retrieval(ctx, verbose=False)["retrieval"]
    assert all(abs(order_means[orders[0][0]][k] - ref[k]) < 1e-12 for k in metrics)

    paired = {}
    for k in metrics + ("cluster",):
        if k == "cluster":
            ours, base = cl_ret, cl_maj
        else:
            ours, base = [r[k] for r in decl_rows], [r[k] for r in prior]
        d = [a - b for a, b in zip(ours, base)]
        paired[k] = {"retrieval_per_split": ours, "prior_per_split": base,
                     "diff_per_split": d, "diff_mean": float(np.mean(d)),
                     "retrieval_wins": int(sum(x > 0 for x in d)),
                     "prior_wins": int(sum(x < 0 for x in d)),
                     "ties": int(sum(x == 0 for x in d))}
    out = {
        "n_relative_orders": len(orders),
        "per_order": [{"order": lab, **order_means[lab]} for lab, _ in orders],
        "range_over_orders": {k: {"min": min(order_means[lab][k] for lab, _ in orders),
                                  "max": max(order_means[lab][k] for lab, _ in orders),
                                  "prior": prior_means[k],
                                  "n_orders_at_or_above_prior": int(sum(
                                      order_means[lab][k] >= prior_means[k]
                                      for lab, _ in orders))}
                              for k in metrics},
        "paired_declaration_vs_prior": paired,
        "mode_prior_cluster": {"per_split": mode_cl, "mean": float(np.mean(mode_cl))},
        "prevalence_tiebreak": {"top1_per_split": tb_top1, "top1": float(np.mean(tb_top1)),
                                "macro_per_split": tb_macro, "macro": float(np.mean(tb_macro))},
        "note": ("Rankings sort by score, then by the order, which keeps modes outside "
                 "the tie components in declaration order. Paired differences are "
                 "retrieval under declaration order minus the prior, per split. The "
                 "cluster row compares the retrieval's cluster top-1 with the fit-half "
                 "majority cluster, as prior_vs_retrieval does. mode_prior_cluster is "
                 "the cluster accuracy of the prior's top category, mapped to its mode. "
                 "prevalence_tiebreak breaks each rank-1 tie by fit-half category "
                 "frequency, most frequent first, then declaration order."),
    }
    if verbose:
        print(f"\n5. RANKING METRICS UNDER ALL {len(orders)} ORDERS, against the prior")
        print("            min      max      prior    orders >= prior")
        for k in metrics:
            r = out["range_over_orders"][k]
            print(f"   {k:<6}  {r['min']:.4f}   {r['max']:.4f}   {r['prior']:.4f}   "
                  f"{r['n_orders_at_or_above_prior']}/{len(orders)}")
        print("   paired per split, declaration order minus prior:")
        for k, p in paired.items():
            print(f"   {k:<7}  mean {p['diff_mean']:+.4f}  retrieval wins "
                  f"{p['retrieval_wins']}/{len(SEEDS)}, prior wins {p['prior_wins']}/{len(SEEDS)}")
        print(f"   cluster accuracy of the prior's top mode  {out['mode_prior_cluster']['mean']:.4f}")
        print(f"   ties broken by fit-half prevalence  top-1 "
              f"{out['prevalence_tiebreak']['top1']:.4f}  macro "
              f"{out['prevalence_tiebreak']['macro']:.4f}")
    return out


# ---------------------------------------------------------------------------
# 6. Balanced recall with ties scored by expected credit
# ---------------------------------------------------------------------------
def _resolver_preds(ctx, resolvers, test, dims):
    """{group representative: {cycle: predicted category}} over the test half."""
    Xte, _ = R.matrix(ctx["cycles"], test, ctx["columns"], dims)
    return {g: dict(zip(test, (int(p) for p in clf.predict(Xte))))
            for g, clf in resolvers.items()}


def balanced_recall_expected(ctx, quick=False, verbose=True):
    """Balanced recall that does not depend on the declaration order.

    A cycle scores 1/|T| per member of its rank-1 tied set T that maps to its true
    category. In the two-stage arm, a cycle whose rank-1 group has a resolver gets
    the resolver's prediction (hard credit) and every other cycle is scored like
    the retrieval. Configuration as in `balanced_recall`: dims mag, sign, step.
    """
    dims = ["mag", "sign", "step"]
    testable, _ = R.testable_clusters()
    cat_of, cat_of_fm, rep = ctx["cat_of"], ctx["cat_of_fm"], ctx["rep"]
    orders = _relative_orders(ctx, _tie_components(ctx))
    tsets = {i: _tied(ctx, i) for i in ctx["ids"]}
    arms = ["retrieval"] + ([] if quick else ["two-stage"])

    micro, macro = defaultdict(list), defaultdict(list)
    per_cat = defaultdict(lambda: defaultdict(list))
    by_order = defaultdict(lambda: defaultdict(list))
    const_bound = []
    for fit, test in _splits(ctx):
        y = [cat_of[i] for i in test]
        const_bound.append(1.0 / len(set(y)))
        preds = {} if quick else _resolver_preds(
            ctx, _fit_resolvers(ctx, fit, dims, testable), test, dims)

        credit = {"retrieval": [_tie_credit(ctx, tsets[i], cat_of[i]) for i in test]}
        if not quick:
            credit["two-stage"] = [
                float(preds[rep[tsets[i][0]]][i] == cat_of[i]) if rep[tsets[i][0]] in preds
                else _tie_credit(ctx, tsets[i], cat_of[i]) for i in test]
        for a in arms:
            micro[a].append(float(np.mean(credit[a])))
            m, per = _macro_credit(credit[a], y)
            macro[a].append(m)
            for c, v in per.items():
                per_cat[a][c].append(v)

        for lab, pos in orders:
            hard = {"retrieval": [cat_of_fm[min(tsets[i], key=pos.get)] for i in test]}
            if not quick:
                hard["two-stage"] = [preds[rep[tsets[i][0]]][i] if rep[tsets[i][0]] in preds
                                     else cat_of_fm[min(tsets[i], key=pos.get)] for i in test]
            for a in arms:
                by_order[a][lab].append(_macro(hard[a], y)[0])

    bound = float(np.mean(const_bound))
    out = {}
    for a in arms:
        order_means = [float(np.mean(by_order[a][lab])) for lab, _ in orders]
        for s in range(len(SEEDS)):
            assert abs(np.mean([by_order[a][lab][s] for lab, _ in orders])
                       - macro[a][s]) < 1e-12
        pc = {str(c): float(np.mean(v)) for c, v in sorted(per_cat[a].items())}
        out[a] = {
            "macro_per_split": macro[a], "macro": float(np.mean(macro[a])),
            "micro_per_split": micro[a], "micro": float(np.mean(micro[a])),
            "per_category": pc,
            "n_zero_categories": int(sum(v == 0 for v in pc.values())),
            "n_categories": len(pc),
            "macro_min_over_orders": float(min(order_means)),
            "macro_max_over_orders": float(max(order_means)),
            "n_orders_below_constant_bound": int(sum(m < bound for m in order_means)),
            "declaration_macro_per_split": by_order[a][orders[0][0]],
            "declaration_macro": order_means[0],
        }
    out["n_relative_orders"] = len(orders)
    out["constant_macro_bound_per_split"] = const_bound
    out["constant_macro_bound"] = bound
    out["note"] = ("Expected credit over the rank-1 tied set equals the mean over all "
                   "relative orders, per split. Per-category values are means over the "
                   "splits in which the category is in the test half. declaration_macro "
                   "is the hard-prediction value that balanced_recall reports. "
                   "n_orders_below_constant_bound compares five-split means with "
                   "constant_macro_bound.")
    if verbose:
        print("\n6. BALANCED RECALL WITH EXPECTED CREDIT OVER TIED SETS")
        print("                 expected   min-max over orders   declaration   "
              "zero-recall categories")
        for a in arms:
            d = out[a]
            print(f"   {a:<12}  {d['macro']:.4f}     [{d['macro_min_over_orders']:.4f}, "
                  f"{d['macro_max_over_orders']:.4f}]      {d['declaration_macro']:.4f}"
                  f"        {d['n_zero_categories']}/{d['n_categories']}")
        print(f"   constant predictor bound 1/K, mean {bound:.4f}; orders below it: "
              + ", ".join(f"{a} {out[a]['n_orders_below_constant_bound']}/{len(orders)}"
                          for a in arms))
    return out


# ---------------------------------------------------------------------------
# 7. Constant router: every cycle to the load-torque resolver
# ---------------------------------------------------------------------------
def constant_router(ctx, verbose=True):
    """Skip stage 1 and send every held-out cycle to the load-torque resolver.

    The resolver is the one `balanced_recall` fits (dims mag, sign, step, FIT
    half). The two-stage arm is `balanced_recall`'s, under declaration order.
    """
    dims = ["mag", "sign", "step"]
    testable, _ = R.testable_clusters()
    cat_of, cat_of_fm, rep = ctx["cat_of"], ctx["cat_of_fm"], ctx["rep"]
    load = rep[LOAD_GROUP]
    top = {i: _tied(ctx, i)[0] for i in ctx["ids"]}
    rows = defaultdict(list)
    for fit, test in _splits(ctx):
        y = [cat_of[i] for i in test]
        resolvers = _fit_resolvers(ctx, fit, dims, testable)
        assert load in resolvers, "no load-torque resolver on this split"
        preds = _resolver_preds(ctx, resolvers, test, dims)
        arms = {"router": [preds[load][i] for i in test],
                "two-stage": [preds[rep[top[i]]][i] if rep[top[i]] in preds
                              else cat_of_fm[top[i]] for i in test]}
        for a, pred in arms.items():
            rows[f"{a}_top1"].append(float(np.mean([p == t for p, t in zip(pred, y)])))
            rows[f"{a}_macro"].append(_macro(pred, y)[0])

    out = {}
    for k in ("top1", "macro"):
        r, t = rows[f"router_{k}"], rows[f"two-stage_{k}"]
        d = [a - b for a, b in zip(r, t)]
        out[k] = {"router_per_split": r, "two_stage_per_split": t,
                  "router": float(np.mean(r)), "two_stage": float(np.mean(t)),
                  "diff_per_split": d, "diff_mean": float(np.mean(d)),
                  "n_router_higher": int(sum(x > 0 for x in d)),
                  "n_splits": len(d)}
    out["note"] = ("router: every held-out cycle goes to the load-torque group's "
                   "resolver, fitted on the fit half as in balanced_recall; it can only "
                   "output categories 1, 6 and 7. two_stage: balanced_recall's two-stage "
                   "arm under declaration order.")
    if verbose:
        print("\n7. CONSTANT ROUTER - every cycle to the load-torque resolver")
        for k in ("top1", "macro"):
            d = out[k]
            print(f"   {k:<6} router {d['router']:.4f}  two-stage {d['two_stage']:.4f}  "
                  f"router higher on {d['n_router_higher']}/{d['n_splits']} splits")
    return out


# ---------------------------------------------------------------------------
# 8. Set-valued answers: the rank-1 group and the rank-1 tied set
# ---------------------------------------------------------------------------
def set_valued(ctx, verbose=True):
    """Coverage of the rank-1 group and tied set against a prior list of equal size."""
    cat_of, rep, fm_of_cat = ctx["cat_of"], ctx["rep"], ctx["fm_of_cat"]
    members = defaultdict(list)
    for f in ctx["fm_order"]:
        members[rep[f]].append(f)
    load = rep[LOAD_GROUP]
    kinds = ("group", "tied")
    rows = defaultdict(list)
    for fit, test in _splits(ctx):
        rank = _prior_rank(ctx, fit)
        y = [cat_of[i] for i in test]
        true_fm = [fm_of_cat[c] for c in y]
        sets = {"group": [members[rep[_tied(ctx, i)[0]]] for i in test],
                "tied": [_tied(ctx, i) for i in test]}
        for k in kinds:
            cov = [float(f in s) for f, s in zip(true_fm, sets[k])]
            pri = [float(f in [fm_of_cat[c] for c in rank[:len(s)]])
                   for f, s in zip(true_fm, sets[k])]
            rows[f"{k}_coverage"].append(float(np.mean(cov)))
            rows[f"{k}_size"].append(float(np.mean([len(s) for s in sets[k]])))
            rows[f"{k}_prior_coverage"].append(float(np.mean(pri)))
            rows[f"{k}_balanced"].append(_macro_credit(cov, y)[0])
            rows[f"{k}_prior_balanced"].append(_macro_credit(pri, y)[0])
        maj_cl = Counter(ctx["true_cluster"][i] for i in fit).most_common(1)[0][0]
        rows["majority_is_load_group"].append(maj_cl == load)
        rows["constant_group"].append(float(np.mean([ctx["true_cluster"][i] == load
                                                     for i in test])))

    out = {}
    for k in kinds:
        out[k] = {}
        for m in ("coverage", "size", "prior_coverage", "balanced", "prior_balanced"):
            v = rows[f"{k}_{m}"]
            out[k][m] = float(np.mean(v))
            out[k][f"{m}_per_split"] = v
    out["constant_group"] = {"group": members[load], "size": len(members[load]),
                             "coverage_per_split": rows["constant_group"],
                             "coverage": float(np.mean(rows["constant_group"])),
                             "majority_is_load_group_per_split": rows["majority_is_load_group"]}
    out["note"] = ("group: every mode in the rank-1 mode's group (cluster_map); tied: the "
                   "rank-1 tied set. coverage: the true category's mode is in the set. "
                   "prior_coverage: the prior's top-|S| categories for the same cycle, "
                   "|S| the size of that cycle's set. balanced: mean over categories in the "
                   "test half. Group coverage equals the retrieval's cluster top-1, tied "
                   "coverage equals the oracle tie-break ceiling. constant_group always "
                   "answers the load-torque group; it equals cluster_majority when the "
                   "fit-half majority cluster is that group on every split.")
    if verbose:
        print("\n8. SET-VALUED ANSWERS against a prior list of the same size")
        print("              coverage   size   prior same size   balanced   prior balanced")
        for k in kinds:
            d = out[k]
            print(f"   {k:<8}   {d['coverage']:.4f}   {d['size']:.3f}   {d['prior_coverage']:.4f}"
                  f"            {d['balanced']:.4f}     {d['prior_balanced']:.4f}")
        print(f"   constant load-torque group (size {out['constant_group']['size']})  "
              f"coverage {out['constant_group']['coverage']:.4f}")
    return out


# ---------------------------------------------------------------------------
def main(argv):
    quick = "--quick" in argv
    ctx = _setup()
    print(f"{len(ctx['ids'])} anomalous cycles, {len(SEEDS)} leave-setting-out splits")
    res = {
        "n_cycles": len(ctx["ids"]), "n_seeds": len(SEEDS),
        "prior_vs_retrieval": prior_vs_retrieval(ctx),
        "balanced_recall": balanced_recall(ctx, quick=quick),
        "tiebreak": tiebreak_interval(ctx),
        "tiebreak_grouped": tiebreak_interval_grouped(ctx),
        "tiebreak_exact": tiebreak_exact(ctx),
        "tie_robust": tie_robust(ctx),
        "balanced_recall_expected": balanced_recall_expected(ctx, quick=quick),
    }
    if not quick:
        res["constant_router"] = constant_router(ctx)
    res["set_valued"] = set_valued(ctx)
    res["note"] = ("Trivial baselines the published evaluation omitted. The "
                   "prevalence prior beats the retrieval on every ranking metric; "
                   "balanced recall is where the retrieval and the repair still show "
                   "an effect; the tie-break interval bounds what a single top-1 "
                   "number can mean when 68% of cycles tie at rank 1.")

    # the declaration-order values of the new analyses must match the old ones
    bre, br = res["balanced_recall_expected"], res["balanced_recall"]
    for a in [a for a in ("retrieval", "two-stage") if a in bre]:
        assert abs(bre[a]["declaration_macro"] - br[a]["macro"]) < 1e-12, a
    if not quick:
        assert abs(res["constant_router"]["top1"]["two_stage"] - br["two-stage"]["micro"]) < 1e-12
        assert abs(res["constant_router"]["macro"]["two_stage"] - br["two-stage"]["macro"]) < 1e-12
    te, tg = res["tiebreak_exact"], res["tiebreak_grouped"]
    assert abs(te["declaration_order"] - tg["declaration_order"]) < 1e-12
    assert abs(te["prior_top1"] - res["prior_vs_retrieval"]["prevalence_prior"]["top1"]) < 1e-12
    assert te["exact_min"] <= tg["all_orderings_min"] + 1e-12
    assert te["exact_max"] >= tg["all_orderings_max"] - 1e-12
    sv, pv = res["set_valued"], res["prior_vs_retrieval"]
    assert abs(sv["group"]["coverage"] - pv["cluster_retrieval"]) < 1e-12
    assert abs(sv["tied"]["coverage"] - tg["oracle_ceiling"]) < 1e-12
    if all(sv["constant_group"]["majority_is_load_group_per_split"]):
        assert abs(sv["constant_group"]["coverage"] - pv["cluster_majority"]) < 1e-12

    # the committed results files use CRLF line endings
    with open(RESULTS_OUT, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten -> {os.path.relpath(RESULTS_OUT, _ROOT)}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
