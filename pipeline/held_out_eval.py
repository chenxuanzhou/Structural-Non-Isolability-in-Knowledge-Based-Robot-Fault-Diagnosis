"""Held-out validation of the retrieve() ranking weights.

METHODOLOGICAL GAP THIS CLOSES
------------------------------
The headline root-cause numbers (top-1 0.158 / top-5 0.636 / MRR 0.319) were
obtained with retrieve() weights tuned on the FULL 755-anomaly set -- i.e. the
ranker was tuned and reported on the same data (tune-on-test). That risks an
optimistic bias. This script splits the anomalies into a TUNE and a TEST half
(stratified by category, multiple random seeds), re-tunes the evidence/phase
weights on TUNE only, and reports accuracy on the held-out TEST half. The TEST
number is the honest, non-overfit one to put in the paper.

It also reports, on the same TEST half:
  * the PRODUCTION line_kg.retrieve() (current shipped weights), and
  * the pure joint+Jaccard BASELINE (w_ev=w_phase=0),
so we can see (a) whether the tuned weights generalise and (b) how much of the
gain over baseline survives on held-out data.

    ..\\.venv\\Scripts\\python.exe held_out_eval.py
"""
from __future__ import annotations
import random
import numpy as np

from eval_retrieve import load_cache, make_retrieve, CATEGORY_TO_FM
from line_kg import retrieve as retrieve_production

# grid over the two principled terms added on top of joint+Jaccard
W_EV_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
W_PH_GRID = [0.0, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2, 3, 4]


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
    """Stratified 50/50 split by ground-truth category."""
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


def main():
    recs = load_cache()
    print(f"loaded {len(recs)} anomaly records\n")
    print(f"grid search w_ev x w_phase = {len(W_EV_GRID)}x{len(W_PH_GRID)} configs,"
          f" {len(SEEDS)} stratified 50/50 splits\n")

    rows = []  # per-seed test results
    chosen = []
    for seed in SEEDS:
        tune, test = _split(recs, seed)
        # tune: pick weights maximising MRR on the TUNE half
        best = None
        for w_ev in W_EV_GRID:
            for w_ph in W_PH_GRID:
                fn = make_retrieve(w_joint=1, w_jac=1, w_ev=w_ev, w_phase=w_ph)
                _, _, mrr = _score(fn, tune)
                if best is None or mrr > best[0]:
                    best = (mrr, w_ev, w_ph)
        _, w_ev, w_ph = best
        chosen.append((w_ev, w_ph))
        # evaluate the three rankers on the held-out TEST half
        tuned = _score(make_retrieve(w_joint=1, w_jac=1, w_ev=w_ev, w_phase=w_ph), test)
        prod = _score(retrieve_production, test)
        base = _score(make_retrieve(w_joint=1, w_jac=1, w_ev=0, w_phase=0), test)
        rows.append((seed, (w_ev, w_ph), tuned, prod, base))
        print(f"seed {seed}: tune-best w_ev={w_ev} w_phase={w_ph} | "
              f"TEST tuned top1 {tuned[0]:.3f} top5 {tuned[1]:.3f} MRR {tuned[2]:.3f} | "
              f"prod top1 {prod[0]:.3f} | base top1 {base[0]:.3f}")

    def avg(i, j):
        return np.mean([r[i][j] for r in rows])

    def ci95(i, j):
        """Half-width of the 95% t-interval over the per-seed values (n=5, t=2.776)."""
        v = [r[i][j] for r in rows]
        return 2.776 * np.std(v, ddof=1) / np.sqrt(len(v))

    print("\n" + "=" * 64)
    print(f"HELD-OUT TEST (mean over {len(SEEDS)} seeds, stratified 50/50)")
    print("=" * 64)
    print(f"  tuned-on-tune weights : top-1 {avg(2,0):.3f}  top-5 {avg(2,1):.3f}  MRR {avg(2,2):.3f}")
    print(f"  production retrieve() : top-1 {avg(3,0):.3f}  top-5 {avg(3,1):.3f}  MRR {avg(3,2):.3f}")
    print(f"  joint+Jaccard baseline: top-1 {avg(4,0):.3f}  top-5 {avg(4,1):.3f}  MRR {avg(4,2):.3f}")
    print("\n  95% CI (t, n=5):")
    print(f"  tuned   : top-1 ±{ci95(2,0):.3f}  top-5 ±{ci95(2,1):.3f}  MRR ±{ci95(2,2):.3f}")
    print(f"  baseline: top-1 ±{ci95(4,0):.3f}  top-5 ±{ci95(4,1):.3f}  MRR ±{ci95(4,2):.3f}")

    import json
    import os as _os
    _out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "results", "held_out_seeds.json")
    with open(_out, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": [{"seed": r[0], "weights": {"w_ev": r[1][0], "w_phase": r[1][1]},
                       "tuned": {"top1": r[2][0], "top5": r[2][1], "mrr": r[2][2]},
                       "production": {"top1": r[3][0], "top5": r[3][1], "mrr": r[3][2]},
                       "baseline": {"top1": r[4][0], "top5": r[4][1], "mrr": r[4][2]}}
                      for r in rows],
            "mean": {"tuned": {"top1": avg(2, 0), "top5": avg(2, 1), "mrr": avg(2, 2)},
                     "baseline": {"top1": avg(4, 0), "top5": avg(4, 1), "mrr": avg(4, 2)}},
            "ci95": {"tuned": {"top1": ci95(2, 0), "top5": ci95(2, 1), "mrr": ci95(2, 2)},
                     "baseline": {"top1": ci95(4, 0), "top5": ci95(4, 1), "mrr": ci95(4, 2)}},
        }, f, indent=2)
    print("  per-seed results -> held_out_seeds.json")
    # modal chosen weights
    from collections import Counter
    mode_w = Counter(chosen).most_common(1)[0]
    print(f"\n  most-chosen tune weights: w_ev={mode_w[0][0]} w_phase={mode_w[0][1]}"
          f"  ({mode_w[1]}/{len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
