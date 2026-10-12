"""Held-out re-tuning of the deployed retrieval weights.

WHAT IS TUNED
-------------
Every retrieval number in the paper comes from line_kg.retrieve() with its
shipped weights. Besides the joint-match and family-Jaccard terms (both fixed at
1, and w_signature fixed at 0), it has three free weights: w_evidence, w_phase
and w_coverage, shipped as (0.2, 0.2, 0.15). Those values were set by looking at
all 755 cycles. This script measures what a choice made without the test cycles
gives instead.

MAIN EXPERIMENT (`retune_deployed`)
-----------------------------------
It uses the five leave-setting-out splits of the paper (repair._grouped_split,
seeds 0-4): every fault instantiation (`setting`) falls wholly in the fit half or
wholly in the test half. On each split:
  1. every triple of the fixed grid
         w_evidence in {0, .1, .2, .3, .4, .6}
         w_phase    in {0, .1, .2, .3, .4}
         w_coverage in {0, .15, .3}
     is scored on the FIT half by the MRR of the category ranking (categories in
     score order, FMEA declaration order breaking ties, as in
     baselines._ranked_cats);
  2. the triple with the highest fit-half MRR is kept (on equal MRR, the first
     triple in grid order);
  3. that triple is scored on the TEST half: top-1, top-3, top-5, MRR, balanced
     (macro) recall of the top-1 under declaration order, tie-aware top-1 (the
     expected top-1 when the rank-1 tie is broken uniformly at random) and the
     number of test cycles with a rank-1 tie.
The shipped triple is scored on the same test halves, and the per-split
differences are stored. For context only, the output also holds the triple that
maximises top-1 on each TEST half (an optimistic bound: it is chosen on the data
it is scored on), the prevalence prior from baselines.prior_vs_retrieval and
baselines.balanced_recall, and the range of rank-1 tie counts over the grid.

HISTORICAL EXPERIMENT (`legacy_stratified_retune`)
--------------------------------------------------
The earlier version of this script tuned (w_ev, w_phase) of
eval_retrieve.make_retrieve, an older scorer (family-level IDF, z capped at 6,
no coverage term), on stratified RANDOM 50/50 splits, which put cycles of one
fault instantiation on both sides. It neither tunes the deployed scorer nor uses
the paper's splits. It is kept unchanged and rerun so its numbers stay
reproducible; the output goes under "legacy".

Run:
    ..\\.venv\\Scripts\\python.exe held_out_eval.py
"""
from __future__ import annotations
import functools
import itertools
import json
import os
import random
import time
from collections import Counter

import numpy as np

import baselines as B
import repair as R
from eval_retrieve import load_cache, make_retrieve, CATEGORY_TO_FM
from line_kg import retrieve as retrieve_production, FMEA

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_OUT = os.path.join(_ROOT, "results", "held_out_seeds.json")
SEEDS = [0, 1, 2, 3, 4]
T95 = 2.776          # t quantile, 95% two-sided, 4 degrees of freedom (n = 5)

# deployed scorer: the grid and the shipped weights
GRID_W_EVIDENCE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
GRID_W_PHASE = [0.0, 0.1, 0.2, 0.3, 0.4]
GRID_W_COVERAGE = [0.0, 0.15, 0.3]
GRID = list(itertools.product(GRID_W_EVIDENCE, GRID_W_PHASE, GRID_W_COVERAGE))
SHIPPED = (0.2, 0.2, 0.15)
METRICS = ("top1", "top3", "top5", "mrr", "balanced", "top1_tie_aware", "n_rank1_ties")

# legacy scorer: grid over the two terms added on top of joint+Jaccard
LEGACY_W_EV_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
LEGACY_W_PH_GRID = [0.0, 0.2, 0.3, 0.4]
# the production weights the committed legacy output was generated with, before
# the 2026-07-20 evidence refinement (see the line_kg.retrieve docstring)
PRE_REFINEMENT = dict(w_evidence=0.3, w_phase=0.3, w_coverage=0.0,
                      use_signal_idf=False, use_log_z=False)


def _ci95(v):
    """Half-width of the 95% t-interval over the per-split values."""
    return T95 * float(np.std(v, ddof=1)) / np.sqrt(len(v))


def _w(t):
    return {"w_evidence": t[0], "w_phase": t[1], "w_coverage": t[2]}


def _key(t):
    return f"{t[0]:g}/{t[1]:g}/{t[2]:g}"


# ---------------------------------------------------------------------------
# Main experiment: the deployed scorer on the leave-setting-out splits
# ---------------------------------------------------------------------------
def _grid_scores(ctx):
    """Per-cycle score of every failure mode, for every grid triple."""
    out = {}
    for t in GRID:
        out[t] = {i: {c["id"]: c["score"]
                      for c in retrieve_production(ctx["recs"][i], top_k=len(FMEA),
                                                   w_evidence=t[0], w_phase=t[1],
                                                   w_coverage=t[2])}
                  for i in ctx["ids"]}
    return out


def _metrics(ctx, scores, ids):
    """Ranking metrics of one score table on a set of cycles."""
    c = dict(ctx, scores=scores)
    ys = [ctx["cat_of"][i] for i in ids]
    out = B._rank_metrics([B._ranked_cats(c, i) for i in ids], ys)
    tied = [B._tied(c, i) for i in ids]
    pred = [ctx["cat_of_fm"][t[0]] for t in tied]
    out["balanced"] = B._macro(pred, ys)[0]
    out["top1_tie_aware"] = float(np.mean(
        [(y in [ctx["cat_of_fm"][f] for f in t]) / len(t) for t, y in zip(tied, ys)]))
    out["n_rank1_ties"] = int(sum(len(t) > 1 for t in tied))
    return out


def retune_deployed(ctx, verbose=True):
    """Choose (w_evidence, w_phase, w_coverage) on each fit half, score on the test half."""
    grid = _grid_scores(ctx)
    assert grid[SHIPPED] == ctx["scores"], "grid scorer differs from the deployed scores"

    splits = []
    for seed in SEEDS:
        fit, test = R._grouped_split(ctx["ids"], ctx["cat_of"], ctx["set_of"], seed)
        fit_mrr = {t: _metrics(ctx, grid[t], fit)["mrr"] for t in GRID}
        chosen = GRID[0]
        for t in GRID:
            if fit_mrr[t] > fit_mrr[chosen]:
                chosen = t
        on_test = {t: _metrics(ctx, grid[t], test) for t in GRID}
        best_test = GRID[0]
        for t in GRID:
            if on_test[t]["top1"] > on_test[best_test]["top1"]:
                best_test = t
        ties = [on_test[t]["n_rank1_ties"] for t in GRID]
        splits.append({
            "seed": seed, "n_fit": len(fit), "n_test": len(test),
            "chosen": _w(chosen),
            "fit_mrr_chosen": fit_mrr[chosen], "fit_mrr_shipped": fit_mrr[SHIPPED],
            "held_out_chosen": on_test[chosen],
            "held_out_shipped": on_test[SHIPPED],
            "diff_chosen_minus_shipped": {m: on_test[chosen][m] - on_test[SHIPPED][m]
                                          for m in METRICS},
            "test_chosen_by_top1": {"weights": _w(best_test), **on_test[best_test]},
            "n_triples_at_fit_max": int(sum(fit_mrr[t] == fit_mrr[chosen] for t in GRID)),
            "n_rank1_ties_grid_min": int(min(ties)), "n_rank1_ties_grid_max": int(max(ties)),
        })
        if verbose:
            a, s = on_test[chosen], on_test[SHIPPED]
            print(f"seed {seed}: fit-chosen {_key(chosen):<12} | test top1 {a['top1']:.3f} "
                  f"(shipped {s['top1']:.3f})  top5 {a['top5']:.3f} ({s['top5']:.3f})  "
                  f"MRR {a['mrr']:.3f} ({s['mrr']:.3f})  balanced {a['balanced']:.3f} "
                  f"({s['balanced']:.3f})  ties {a['n_rank1_ties']}/{len(test)} "
                  f"({s['n_rank1_ties']}) | test-chosen {_key(best_test)} "
                  f"top1 {on_test[best_test]['top1']:.3f}")

    def col(name, m):
        return [s[name][m] for s in splits]

    names = ("held_out_chosen", "held_out_shipped", "diff_chosen_minus_shipped",
             "test_chosen_by_top1")
    mean = {n: {m: float(np.mean(col(n, m))) for m in METRICS} for n in names}
    ci95 = {n: {m: _ci95(col(n, m)) for m in METRICS} for n in names}
    d1 = col("diff_chosen_minus_shipped", "top1")

    # rank-1 tie counts on all cycles, for every triple
    full = {t: _metrics(ctx, grid[t], ctx["ids"])["n_rank1_ties"] for t in GRID}
    lo, hi = min(full.values()), max(full.values())

    prior = B.prior_vs_retrieval(ctx, verbose=False)
    bal = B.balanced_recall(ctx, quick=True, verbose=False)
    shipped_ok = (all(abs(mean["held_out_shipped"][m] - prior["retrieval"][m]) < 1e-12
                      for m in ("top1", "top3", "top5", "mrr"))
                  and abs(mean["held_out_shipped"]["balanced"]
                          - bal["retrieval"]["macro"]) < 1e-12)

    out = {
        "protocol": "leave-setting-out 50/50 (repair._grouped_split), seeds 0-4; "
                    "weights chosen on the fit half, scored on the test half",
        "scorer": "line_kg.retrieve; w_joint=1, w_signal=1, w_signature=0 fixed; "
                  "w_evidence, w_phase, w_coverage tuned",
        "selection": "highest fit-half MRR of the category ranking (score order, "
                     "declaration order on ties); first triple in grid order on equal MRR",
        "metric_notes": {
            "balanced": "mean per-category recall of the top-1, declaration order on ties",
            "top1_tie_aware": "expected top-1 when the rank-1 tie is broken uniformly "
                              "at random: 1/|tied set| if the true mode is in it, else 0",
            "n_rank1_ties": "cycles whose rank-1 score is shared by more than one mode",
            "test_chosen_by_top1": "triple with the highest TEST-half top-1 (first in grid "
                                   "order on equal top-1); optimistic, context only",
        },
        "grid": {"w_evidence": GRID_W_EVIDENCE, "w_phase": GRID_W_PHASE,
                 "w_coverage": GRID_W_COVERAGE, "n_triples": len(GRID)},
        "shipped": _w(SHIPPED),
        "seeds": SEEDS,
        "splits": splits,
        "mean": mean,
        "ci95": ci95,
        "n_distinct_chosen": len({_key(tuple(s["chosen"].values())) for s in splits}),
        "shipped_chosen_on_splits": int(sum(tuple(s["chosen"].values()) == SHIPPED
                                            for s in splits)),
        "n_splits_top1_chosen_above_shipped": int(sum(d > 1e-12 for d in d1)),
        "n_splits_top1_chosen_below_shipped": int(sum(d < -1e-12 for d in d1)),
        "n_splits_top1_chosen_equal_shipped": int(sum(abs(d) <= 1e-12 for d in d1)),
        "prevalence_prior": {**prior["prevalence_prior"],
                             "balanced": bal["prior (constant)"]["macro"]},
        "shipped_matches_baselines": bool(shipped_ok),
        "rank1_ties_full_set": {
            "n_cycles": len(ctx["ids"]),
            "min": int(lo), "max": int(hi),
            "argmin": [_key(t) for t in GRID if full[t] == lo],
            "argmax": [_key(t) for t in GRID if full[t] == hi],
            "shipped": int(full[SHIPPED]),
            "per_triple": {_key(t): int(full[t]) for t in GRID},
        },
    }
    if verbose:
        mc, ms, md = mean["held_out_chosen"], mean["held_out_shipped"], \
            mean["diff_chosen_minus_shipped"]
        pp = out["prevalence_prior"]
        print("\n" + "=" * 74)
        print(f"DEPLOYED SCORER, LEAVE-SETTING-OUT (mean over {len(SEEDS)} splits)")
        print("=" * 74)
        print("                      top-1   top-3   top-5    MRR   balanced  tie-aware")
        for name, d in (("fit-chosen", mc), ("shipped", ms), ("chosen - shipped", md),
                        ("test-chosen (bound)", mean["test_chosen_by_top1"])):
            print(f"  {name:<19} {d['top1']:>6.3f}  {d['top3']:>6.3f}  {d['top5']:>6.3f}  "
                  f"{d['mrr']:>6.3f}  {d['balanced']:>7.3f}  {d['top1_tie_aware']:>8.3f}")
        print(f"  {'prevalence prior':<19} {pp['top1']:>6.3f}  {pp['top3']:>6.3f}  "
              f"{pp['top5']:>6.3f}  {pp['mrr']:>6.3f}  {pp['balanced']:>7.3f}")
        print(f"  fit-chosen top-1 95% CI +/-{ci95['held_out_chosen']['top1']:.3f}; "
              f"paired difference {md['top1']:+.3f} +/- "
              f"{ci95['diff_chosen_minus_shipped']['top1']:.3f}")
        print(f"  distinct triples chosen: {out['n_distinct_chosen']}/{len(SEEDS)}; "
              f"shipped chosen on {out['shipped_chosen_on_splits']} split(s)")
        print(f"  rank-1 ties over the grid, all {len(ctx['ids'])} cycles: {lo}..{hi} "
              f"(shipped {full[SHIPPED]})")
        print(f"  shipped column equals the baselines.py retrieval column: {shipped_ok}")
    return out


# ---------------------------------------------------------------------------
# Historical experiment: legacy scorer on stratified random splits
# ---------------------------------------------------------------------------
def _score(fn, recs, top_k=5):
    """Quiet scorer -> (top1, topk, mrr)."""
    h1 = hk = rr = 0.0
    for rec in recs:
        cat = rec["detector"]["ground_truth_category"]
        tf = CATEGORY_TO_FM.get(cat)
        ids = [c["id"] for c in fn(rec, top_k=top_k)]
        rank = (ids.index(tf) + 1) if tf in ids else 0
        h1 += rank == 1
        hk += rank >= 1
        rr += (1.0 / rank) if rank else 0.0
    n = max(1, len(recs))
    return h1 / n, hk / n, rr / n


def _split(recs, seed):
    """Stratified random 50/50 split by ground-truth category."""
    by_cat: dict[int, list] = {}
    for r in recs:
        by_cat.setdefault(r["detector"]["ground_truth_category"], []).append(r)
    rng = random.Random(seed)
    tune, test = [], []
    for cat, rs in sorted(by_cat.items()):
        rs = rs[:]
        rng.shuffle(rs)
        k = len(rs) // 2
        tune += rs[:k]
        test += rs[k:]
    return tune, test


def legacy_stratified_retune(recs, verbose=True):
    """The earlier experiment, kept for reproducibility.

    Tunes (w_ev, w_phase) of the legacy scorer eval_retrieve.make_retrieve by
    TUNE-half MRR on stratified random 50/50 splits, and scores on the TEST half:
    the tuned legacy scorer, line_kg.retrieve as it ships now ("production"),
    line_kg.retrieve with its pre-refinement weights ("production_pre_refinement",
    which is what the committed file held under "production"), and joint+Jaccard
    alone ("baseline").
    """
    pre = functools.partial(retrieve_production, **PRE_REFINEMENT)
    rows, chosen = [], []
    for seed in SEEDS:
        tune, test = _split(recs, seed)
        best = None
        for w_ev in LEGACY_W_EV_GRID:
            for w_ph in LEGACY_W_PH_GRID:
                fn = make_retrieve(w_joint=1, w_jac=1, w_ev=w_ev, w_phase=w_ph)
                _, _, mrr = _score(fn, tune)
                if best is None or mrr > best[0]:
                    best = (mrr, w_ev, w_ph)
        _, w_ev, w_ph = best
        chosen.append((w_ev, w_ph))
        cols = {"tuned": _score(make_retrieve(w_joint=1, w_jac=1, w_ev=w_ev, w_phase=w_ph),
                                test),
                "production": _score(retrieve_production, test),
                "production_pre_refinement": _score(pre, test),
                "baseline": _score(make_retrieve(w_joint=1, w_jac=1, w_ev=0, w_phase=0),
                                   test)}
        rows.append({"seed": seed, "weights": {"w_ev": w_ev, "w_phase": w_ph},
                     **{k: dict(zip(("top1", "top5", "mrr"), v)) for k, v in cols.items()}})
        if verbose:
            print(f"seed {seed}: tune-best w_ev={w_ev} w_phase={w_ph} | TEST tuned top1 "
                  f"{cols['tuned'][0]:.3f} | production {cols['production'][0]:.3f} | "
                  f"pre-refinement {cols['production_pre_refinement'][0]:.3f} | "
                  f"baseline {cols['baseline'][0]:.3f}")

    names = ("tuned", "production", "production_pre_refinement", "baseline")
    mean = {n: {m: float(np.mean([r[n][m] for r in rows])) for m in ("top1", "top5", "mrr")}
            for n in names}
    ci95 = {n: {m: _ci95([r[n][m] for r in rows]) for m in ("top1", "top5", "mrr")}
            for n in names}
    mode_w, mode_n = Counter(chosen).most_common(1)[0]
    out = {
        "protocol": "stratified random 50/50 by category (cycles of one setting on both "
                    "sides), seeds 0-4",
        "scorer": "eval_retrieve.make_retrieve (family-level IDF, z capped at 6, no "
                  "coverage term); tuned: w_ev, w_phase",
        "seeds": rows, "mean": mean, "ci95": ci95,
        "most_chosen_weights": {"w_ev": mode_w[0], "w_phase": mode_w[1], "n_seeds": mode_n},
        "note": "Rerun of the earlier experiment. 'tuned' and 'baseline' reproduce the "
                "committed values. 'production' is line_kg.retrieve as it ships now; the "
                "committed file held line_kg.retrieve before the 2026-07-20 evidence "
                "refinement, which 'production_pre_refinement' recomputes.",
    }
    if verbose:
        print(f"  mean top-1: tuned {mean['tuned']['top1']:.3f} +/- "
              f"{ci95['tuned']['top1']:.3f} | production {mean['production']['top1']:.3f} | "
              f"pre-refinement {mean['production_pre_refinement']['top1']:.3f} | "
              f"baseline {mean['baseline']['top1']:.3f}")
        print(f"  most-chosen weights: w_ev={mode_w[0]} w_phase={mode_w[1]} "
              f"({mode_n}/{len(SEEDS)} seeds)")
    return out


def main():
    t0 = time.time()
    ctx = B._setup()
    print(f"{len(ctx['ids'])} anomalous cycles; grid of {len(GRID)} weight triples, "
          f"{len(SEEDS)} leave-setting-out splits\n")
    res = retune_deployed(ctx)

    print("\nLEGACY: make_retrieve re-tuned on stratified random splits\n")
    res["legacy"] = legacy_stratified_retune(load_cache())

    text = json.dumps(res, indent=1, ensure_ascii=False)
    with open(RESULTS_OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace("\n", "\r\n"))
    print(f"\nwritten -> {os.path.relpath(RESULTS_OUT, _ROOT)}  "
          f"({time.time() - t0:.1f} s)")
    return res


if __name__ == "__main__":
    main()
