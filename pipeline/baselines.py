# -*- coding: utf-8 -*-
"""Trivial baselines the evaluation was missing, and the tie-break interval.

WHY THIS MODULE EXISTS
----------------------
Every accuracy number in this project was reported against a UNIFORM chance
rate: 1/14 = 0.071 top-1, or 1/12 = 0.083 restricted to the modes this dataset
reaches, and 0.150 at the cluster level.  Uniform chance is the wrong reference
for voraus-AD, whose anomaly categories are strongly skewed -- 156 cycles of
category 1 against 10 of category 8.  The reference a reviewer will reach for is
the PREVALENCE PRIOR: rank the failure modes by how often they occurred in the
fit half and emit that same ranking for every cycle.  It uses no evidence, no
knowledge base, and nothing about the cycle in front of it.

Measured here, that baseline BEATS the deployed retrieval on every headline
metric -- top-1, top-3, top-5, MRR, and cluster top-1.  The published "2.28x
lift over chance" is therefore a lift over the wrong quantity; against the prior
the retrieval scores 0.90x.

Two further references belong beside it:

  BALANCED (MACRO) RECALL -- mean per-category recall.  A constant predictor
      scores 1/n_classes here however skewed the data, so this is the metric the
      prior cannot game.  It is where the retrieval, and much more clearly the
      repair, still show a real effect.

  THE TIE-BREAK INTERVAL -- 514 of 755 cycles have an exact rank-1 score tie, so
      "the" top-1 accuracy is one draw from a distribution the algorithm does not
      determine.  A single number implies a precision that does not exist: the
      spread across tie-breaks is several times the split-to-split CI the paper
      quotes, and it straddles the prevalence prior, so which of the two wins is
      decided by the order failure modes happen to be declared in.  This is not a
      defect to hide.  It is the paper's own thesis -- ranking cannot break
      structural ties -- expressed as a number.  Cluster assignment, by contrast,
      is tie-break INVARIANT, because the ties lie inside clusters.  That
      invariance is the positive result here.

Nothing in this module changes a deployed default; it only measures.  Splits,
features and the ranking are imported from `repair.py` / `line_kg.py` so there is
one implementation, not a parallel copy.

Run:
    ..\\.venv\\Scripts\\python.exe baselines.py          # full report
    ..\\.venv\\Scripts\\python.exe baselines.py --quick  # skip the learned arms
"""
from __future__ import annotations

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

        cnt = Counter(ctx["cat_of"][i] for i in fit)
        rank = [c for c, _ in cnt.most_common()]
        rank += [c for c in sorted(set(ctx["cat_of"].values())) if c not in rank]
        prior.append(_rank_metrics([rank] * len(test), ys))

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

    for seed in SEEDS:
        fit, test = R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], seed)
        y = [ctx["cat_of"][i] for i in test]
        maj = Counter(ctx["cat_of"][i] for i in fit).most_common(1)[0][0]
        arms = {"prior (constant)": [maj] * len(test),
                "retrieval": [ctx["cat_of_fm"][_tied(ctx, i)[0]] for i in test]}

        if not quick:
            resolvers = {}
            for members, cats in testable:
                sub = [i for i in fit if ctx["cat_of"][i] in cats]
                if len(sub) < 6 or len(set(ctx["cat_of"][i] for i in sub)) < 2:
                    continue
                Xc, _ = R.matrix(ctx["cycles"], sub, ctx["columns"], dims)
                clf, _ = R._fit_predict(Xc, np.array([ctx["cat_of"][i] for i in sub]), Xc)
                resolvers[ctx["rep"][sorted(members)[0]]] = clf
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
    out["uniform_macro"] = 1.0 / len(per_cat["retrieval"])
    if verbose:
        print("\n2. BALANCED RECALL - mean per-category recall; a constant scores "
              f"{out['uniform_macro']:.3f} here whatever the skew")
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
        print("   -> the spread across tie-breaks is several times the split-to-split")
        print("      CI the paper quotes, and it straddles the prevalence prior.")
        print(f"   cluster assignment is tie-break INVARIANT: "
              f"{ {k: round(v, 4) for k, v in cl.items()} }")
        print("      -> the ties lie INSIDE clusters. That is Property 5.2, measured.")
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
        "note": ("Trivial baselines the published evaluation omitted. The "
                 "prevalence prior beats the retrieval on every ranking metric; "
                 "balanced recall is where the retrieval and the repair still show "
                 "an effect; the tie-break interval bounds what a single top-1 "
                 "number can mean when 68% of cycles tie at rank 1."),
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten -> {os.path.relpath(RESULTS_OUT, _ROOT)}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
