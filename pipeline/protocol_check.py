# -*- coding: utf-8 -*-
"""Protocol check for the distal-collision certificate.

The paper's certificate table reports the collision group at 0.830 under a
random split over cycles (searched certificate, adopted on 5/5 splits) and at
0.776 under leave-setting-out (rejected).  The two figures do not come from the
same procedure.  Under leave-setting-out the grouped inner search cannot run,
so 0.776 is the pre-registered free-cost dimension set; 0.830 is a searched
certificate whose inner CV is ungrouped.  This module changes one thing at a
time (outer split, inner CV, number of folds) and adds setting-level
permutation tests.

The collision group is category 2 (collision_foam, 6 settings, 72 cycles) and
category 4 (collision_card, 3 settings, 22 cycles).  The load-torque group
(categories 1, 6, 7; 32 settings, 310 cycles) is the reference in part 5.

Parts, by key in results/protocol_check.json:
  random_split          1. random 50/50 split over cycles, ungrouped inner CV,
                           seeds 0-4 (the leakage check in repair.py, per seed)
  preregistered_free    2. the pre-registered free-cost set under the random
                           split and under leave-setting-out
  random_grouped_inner  3. random outer split, inner CV grouped by setting
  nine_fold             4. StratifiedKFold(9) over cycles with ungrouped inner
                           CV vs leave-one-setting-out over the 9 collision
                           settings with grouped inner CV, inner seeds 0-9
  setting_permutation   5. category labels permuted across whole settings and
                           the full certify rerun on the same random split
  retrieval_protocol    6. retrieval top-1 (declaration order) over 20 random
                           and 20 leave-setting-out splits

Search, adoption rule and margins are repair.certify and repair.ADOPT_MARGIN
unchanged.  BLAS is limited to one thread in every process, so the numbers do
not depend on --jobs.

Run (from pipeline/):
    python protocol_check.py                    # all parts
    python protocol_check.py --skip-load-perm   # skip the load-torque permutations
    python protocol_check.py --skip-perm        # skip both permutation tests
    python protocol_check.py --jobs 4           # worker processes (default: all cores)
    python protocol_check.py --load-draws 200   # fewer load-torque labellings
Parts that are skipped keep the values already in the JSON file.

Runtime, measured with --jobs 4 on a 4-core machine that was running another
job at the same time: 18 min in total.  Parts 1-4 and 6 took 1 min, the
collision permutations (5 x 84 certify runs) 1.2 min, the load-torque
permutations (5 x 301 certify runs) 16 min.  One certify run costs about 0.5 s
(collision) and 1.5 s (load torque) on one core, so --jobs 1 needs about
45 min.  The times of the last run are written to runtime_s in the JSON.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from collections import Counter

import numpy as np

import repair as R

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_OUT = os.path.join(_ROOT, "results", "protocol_check.json")
REPAIR_RESULTS = os.path.join(_ROOT, "results", "repair_results.json")

SEEDS = (0, 1, 2, 3, 4)
INNER_SEEDS = tuple(range(10))
RETRIEVAL_SEEDS = tuple(range(20))
N_FOLDS = 9
LOOSE_MARGIN = (0.02, 0.01)          # loosest margin in repair.sensitivity()
LOAD_DRAWS = 300
COLL = "collision_card+collision_foam"
LOAD = "axisweight+losecan+varweight"
FOAM, CARTON = 2, 4
CAT_NAME = {1: "axisweight", 2: "collision_foam", 4: "collision_card",
            6: "losecan", 7: "varweight"}
# the pre-registered free-cost set, built the way certify() builds its fallback
PREREG = ["mag"] + [d for d in R.REPAIR_DIMS if R.GRAMMAR[d]["cost"] == "free"]


# ---------------------------------------------------------------------------
# Setup and the certify task (shared by the main process and the workers)
# ---------------------------------------------------------------------------
_FEAT = {}


def _init(cycles=None, columns=None):
    """One BLAS / OpenMP thread; load the feature space once per process."""
    # threadpoolctl only limits libraries already loaded, so load sklearn first
    import sklearn.linear_model, sklearn.model_selection, sklearn.preprocessing  # noqa: E401,F401
    from threadpoolctl import threadpool_limits
    _FEAT["limits"] = threadpool_limits(limits=1)
    if cycles is None:
        cycles, columns = R.load_features(verbose=False)
    _FEAT["cycles"], _FEAT["columns"] = cycles, columns


def _predict(fit, test, y_fit, dims):
    Xtr, _ = R.matrix(_FEAT["cycles"], fit, _FEAT["columns"], dims)
    Xte, _ = R.matrix(_FEAT["cycles"], test, _FEAT["columns"], dims)
    return [int(p) for p in R._fit_predict(Xtr, y_fit, Xte)[1]]


def _cert_task(task):
    """repair.certify on one split; optionally the held-out predictions too."""
    fit, test, y_fit, y_test, groups, seed, name, members, want_pred = task
    cert = R.certify(_FEAT["cycles"], _FEAT["columns"], name, members, fit, test,
                     y_fit, y_test, groups, seed)
    out = {k: v for k, v in cert.items()
           if k not in ("cluster", "members", "adoption_decided_on")}
    if want_pred:
        # same fit as inside certify(); checked against its reported accuracy
        out["pred"] = _predict(fit, test, y_fit, cert["dims"])
        out["pred_mag"] = _predict(fit, test, y_fit, ["mag"])
        assert abs(np.mean(np.array(out["pred"]) == y_test) - cert["acc_test"]) < 1e-12
        assert abs(np.mean(np.array(out["pred_mag"]) == y_test)
                   - cert["acc_deployed_repr"]) < 1e-12
    return out


def _runner(jobs):
    """Map certify tasks serially or over a spawn pool; order is preserved."""
    if jobs <= 1:
        return (lambda tasks: [_cert_task(t) for t in tasks]), None
    import multiprocessing as mp
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"            # read by each worker when it loads BLAS
    pool = mp.get_context("spawn").Pool(jobs, initializer=_init)

    def run(tasks):
        return pool.map(_cert_task, tasks, chunksize=max(1, len(tasks) // (8 * jobs)))
    return run, pool


def _context():
    from baselines import _setup
    ctx = _setup()
    testable, _ = R.testable_clusters()
    ctx["clusters"] = {"+".join(m.replace("FM_", "") for m in members): (sorted(members), cats)
                       for members, cats in testable}
    return ctx


def _split(ctx, name, protocol, seed):
    """The cluster's fit / test cycles exactly as repair.evaluate builds them:
    split all 755 cycles, then keep the cluster's categories."""
    cats = ctx["clusters"][name][1]
    ids, cat_of = ctx["ids"], ctx["cat_of"]
    fit, test = (R._grouped_split(ids, cat_of, ctx["set_of"], seed) if protocol == "setting"
                 else R._stratified_split(ids, cat_of, seed))
    return [i for i in fit if cat_of[i] in cats], [i for i in test if cat_of[i] in cats]


def _task(ctx, name, fit, test, seed, grouped, label_of=None, pred=False):
    label_of = label_of or ctx["cat_of"]
    y_fit = np.array([label_of[i] for i in fit])
    y_test = np.array([label_of[i] for i in test])
    groups = np.array([ctx["set_of"][i] for i in fit]) if grouped else None
    return (fit, test, y_fit, y_test, groups, seed, name, ctx["clusters"][name][0], pred)


def _threshold(c, margin):
    """The inner-CV accuracy a searched set has to exceed to be adopted."""
    if not c["search_validated"]:
        return None
    return max(c["inner_cv_majority"] + margin[0], c["inner_cv_deployed_repr"] + margin[1])


def _adopted_at(c, margin):
    t = _threshold(c, margin)
    return bool(t is not None and c["inner_cv_fit"] > t)


def _fit_majority_class(y_fit):
    vals, cnt = np.unique(y_fit, return_counts=True)     # as in certify()
    return int(vals[np.argmax(cnt)])


def _rates(pred, truth):
    """Accuracy, balanced accuracy and per-class recall of pooled predictions."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    rec = {CAT_NAME[int(c)]: float(np.mean(pred[truth == c] == c)) for c in sorted(set(truth))}
    return {"acc": float(np.mean(pred == truth)),
            "balanced_acc": float(np.mean(list(rec.values()))), "recall": rec}


def _confusion(pred, truth, labels):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return {"labels": [CAT_NAME[c] for c in labels], "rows": "true", "columns": "predicted",
            "matrix": [[int(np.sum((truth == a) & (pred == b))) for b in labels]
                       for a in labels]}


def _summ(v):
    v = [x for x in v if x is not None]
    return {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}


def _f(x):
    return "  n/a" if x is None else f"{x:.3f}"


def _stored():
    with open(REPAIR_RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Random split over cycles, ungrouped inner CV
# ---------------------------------------------------------------------------
def random_split(ctx, run, verbose=True):
    splits = [_split(ctx, COLL, "random", s) for s in SEEDS]
    res = run([_task(ctx, COLL, f, t, s, grouped=False, pred=True)
               for s, (f, t) in zip(SEEDS, splits)])
    rows = []
    for s, (fit, test), c in zip(SEEDS, splits, res):
        y_test = np.array([ctx["cat_of"][i] for i in test])
        pred = np.array(c["pred"])
        own = Counter(y_test.tolist()).most_common(1)[0][1]
        rows.append({
            "seed": s, "n_fit": c["n_fit"], "n_test": c["n_test"],
            "dims": c["dims"], "search_trace": c["search_trace"],
            "inner_cv_selected": c["inner_cv_fit"],
            "inner_cv_deployed": c["inner_cv_deployed_repr"],
            "fit_majority": c["inner_cv_majority"],
            "adoption_threshold": _threshold(c, R.ADOPT_MARGIN),
            "adopted": c["adopted"],
            "adopted_loose_margin": _adopted_at(c, LOOSE_MARGIN),
            "held_out_acc": c["acc_test"],
            "held_out_correct": int(np.sum(pred == y_test)),
            "test_own_majority": own / len(y_test),
            "test_fit_majority_class": c["acc_majority"],
            "deployed_held_out": c["acc_deployed_repr"],
            "permuted_held_out": c["acc_permuted"],
            "confusion": _confusion(pred, y_test, [FOAM, CARTON])})
        assert _adopted_at(c, R.ADOPT_MARGIN) == c["adopted"]
    stored = _stored()["leakage_check"]["per_cluster"][COLL]["random"]["acc"]
    out = {
        "protocol": "random stratified 50/50 split over cycles (repair._stratified_split), "
                    "inner CV ungrouped (groups=None), as in repair.py's leakage check",
        "per_seed": rows,
        "held_out_acc": _summ([r["held_out_acc"] for r in rows]),
        "test_own_majority": _summ([r["test_own_majority"] for r in rows]),
        "inner_cv_selected": _summ([r["inner_cv_selected"] for r in rows]),
        "inner_cv_deployed": _summ([r["inner_cv_deployed"] for r in rows]),
        "deployed_held_out": _summ([r["deployed_held_out"] for r in rows]),
        "permuted_held_out": _summ([r["permuted_held_out"] for r in rows]),
        "adopted": f"{sum(r['adopted'] for r in rows)}/{len(rows)}",
        "stored_leakage_check_acc": stored,
        "matches_stored": bool(abs(np.mean([r["held_out_acc"] for r in rows]) - stored) < 1e-12),
    }
    if verbose:
        print("\n1. RANDOM SPLIT, UNGROUPED INNER CV (collision group)")
        for r in rows:
            print(f"   seed {r['seed']}: dims {'+'.join(r['dims']):24s} inner "
                  f"{r['inner_cv_selected']:.3f} (mag {r['inner_cv_deployed']:.3f}, "
                  f"threshold {r['adoption_threshold']:.3f})  adopted {r['adopted']}  "
                  f"held-out {r['held_out_acc']:.3f} ({r['held_out_correct']}/{r['n_test']})"
                  f"  own majority {r['test_own_majority']:.3f}")
        print(f"   adopted {out['adopted']}; matches repair_results.json: {out['matches_stored']}")
    return out


# ---------------------------------------------------------------------------
# 2. The pre-registered free-cost set under both outer splits
# ---------------------------------------------------------------------------
def preregistered_free(ctx, verbose=True):
    out = {"dims": PREREG}
    for protocol, key in (("random", "random_split"), ("setting", "leave_setting_out")):
        rows = []
        for s in SEEDS:
            fit, test = _split(ctx, COLL, protocol, s)
            y_fit = np.array([ctx["cat_of"][i] for i in fit])
            y_test = np.array([ctx["cat_of"][i] for i in test])
            pred = np.array(_predict(fit, test, y_fit, PREREG))
            pmag = np.array(_predict(fit, test, y_fit, ["mag"]))
            rows.append({"seed": s, "n_fit": len(fit), "n_test": len(test),
                         "held_out_acc": float(np.mean(pred == y_test)),
                         "deployed_held_out": float(np.mean(pmag == y_test)),
                         "test_fit_majority_class": float(np.mean(y_test == _fit_majority_class(y_fit))),
                         "test_own_majority": Counter(y_test.tolist()).most_common(1)[0][1] / len(y_test)})
        out[key] = {"per_seed": rows,
                    "held_out_acc": _summ([r["held_out_acc"] for r in rows]),
                    "deployed_held_out": _summ([r["deployed_held_out"] for r in rows]),
                    "test_fit_majority_class": _summ([r["test_fit_majority_class"] for r in rows])}
    stored = _stored()["certificates"][COLL]["acc_test"]
    out["stored_leave_setting_out_acc"] = stored
    out["leave_setting_out_matches_stored"] = bool(abs(
        out["leave_setting_out"]["held_out_acc"]["mean"] - stored) < 1e-12)
    if verbose:
        print("\n2. PRE-REGISTERED FREE-COST SET " + "+".join(PREREG))
        print(f"   random split       {out['random_split']['held_out_acc']['mean']:.3f}  "
              f"per seed {[round(r['held_out_acc'], 3) for r in out['random_split']['per_seed']]}")
        print(f"   leave-setting-out  {out['leave_setting_out']['held_out_acc']['mean']:.3f}  "
              f"(repair_results.json {stored:.3f}, match {out['leave_setting_out_matches_stored']})")
    return out


# ---------------------------------------------------------------------------
# 3. Random outer split, inner CV grouped by setting
# ---------------------------------------------------------------------------
def random_grouped_inner(ctx, run, part1=None, verbose=True):
    splits = [_split(ctx, COLL, "random", s) for s in SEEDS]
    res = run([_task(ctx, COLL, f, t, s, grouped=True) for s, (f, t) in zip(SEEDS, splits)])
    rows = []
    for s, (fit, _), c in zip(SEEDS, splits, res):
        per_class = {CAT_NAME[k]: len({ctx["set_of"][i] for i in fit if ctx["cat_of"][i] == k})
                     for k in (FOAM, CARTON)}
        rows.append({"seed": s, "fit_settings_per_class": per_class,
                     "inner_folds": min(4, min(per_class.values())),
                     "search_validated": c["search_validated"], "dims": c["dims"],
                     "inner_cv_selected": c["inner_cv_fit"],
                     "inner_cv_deployed": c["inner_cv_deployed_repr"],
                     "fit_majority": c["inner_cv_majority"],
                     "adoption_threshold": _threshold(c, R.ADOPT_MARGIN),
                     "adopted": c["adopted"],
                     "adoption_threshold_loose": _threshold(c, LOOSE_MARGIN),
                     "adopted_loose_margin": _adopted_at(c, LOOSE_MARGIN),
                     "held_out_acc": c["acc_test"]})
    out = {"protocol": "random stratified 50/50 split over cycles, inner CV grouped by "
                       "the setting of each fit cycle",
           "adopt_margin": list(R.ADOPT_MARGIN), "loose_margin": list(LOOSE_MARGIN),
           "per_seed": rows,
           "inner_cv_selected": _summ([r["inner_cv_selected"] for r in rows]),
           "inner_cv_deployed": _summ([r["inner_cv_deployed"] for r in rows]),
           "held_out_acc": _summ([r["held_out_acc"] for r in rows]),
           "adopted": f"{sum(r['adopted'] for r in rows)}/{len(rows)}",
           "adopted_loose_margin": f"{sum(r['adopted_loose_margin'] for r in rows)}/{len(rows)}"}
    if part1 is not None:
        out["ungrouped_inner_cv_selected_mean"] = part1["inner_cv_selected"]["mean"]
        out["ungrouped_adopted"] = part1["adopted"]
        out["ungrouped_adopted_loose_margin"] = (
            f"{sum(r['adopted_loose_margin'] for r in part1['per_seed'])}/{len(part1['per_seed'])}")
    if verbose:
        print("\n3. RANDOM OUTER SPLIT, INNER CV GROUPED BY SETTING")
        for r in rows:
            print(f"   seed {r['seed']}: dims {'+'.join(r['dims']):24s} inner "
                  f"{_f(r['inner_cv_selected'])} (threshold {_f(r['adoption_threshold'])})  "
                  f"adopted {r['adopted']} / loose {r['adopted_loose_margin']}  "
                  f"held-out {r['held_out_acc']:.3f}")
        print(f"   inner mean {out['inner_cv_selected']['mean']:.3f}"
              + (f" vs ungrouped {out['ungrouped_inner_cv_selected_mean']:.3f}" if part1 else "")
              + f"; adopted {out['adopted']} at {R.ADOPT_MARGIN}, "
                f"{out['adopted_loose_margin']} at {LOOSE_MARGIN}")
    return out


# ---------------------------------------------------------------------------
# 4. Nine folds on the collision cycles: over cycles vs over settings
# ---------------------------------------------------------------------------
def _pooled(ctx, folds, results, label="fold"):
    truth, p_sel, p_mag, p_maj, rows = [], [], [], [], []
    for k, ((fit, test), c) in enumerate(zip(folds, results)):
        y_test = [ctx["cat_of"][i] for i in test]
        maj = _fit_majority_class([ctx["cat_of"][i] for i in fit])
        truth += y_test
        p_sel += c["pred"]
        p_mag += c["pred_mag"]
        p_maj += [maj] * len(test)
        row = {label: ctx["set_of"][test[0]] if label == "held_out_setting" else k}
        if label == "held_out_setting":
            row["category"] = CAT_NAME[y_test[0]]
        row.update({"n_test": len(test), "search_validated": c["search_validated"],
                    "adopted": c["adopted"], "dims": c["dims"],
                    "inner_cv_selected": c["inner_cv_fit"], "held_out_acc": c["acc_test"]})
        rows.append(row)
    sel, mag, maj = _rates(p_sel, truth), _rates(p_mag, truth), _rates(p_maj, truth)
    return {"folds": len(folds),
            "searched": sum(c["search_validated"] for c in results),
            "adopted": sum(c["adopted"] for c in results),
            "pooled_acc": sel["acc"], "pooled_balanced_acc": sel["balanced_acc"],
            "pooled_carton_recall": sel["recall"]["collision_card"],
            "pooled_majority_acc": maj["acc"],
            "pooled_deployed_acc": mag["acc"],
            "pooled_deployed_balanced_acc": mag["balanced_acc"],
            "pooled_deployed_carton_recall": mag["recall"]["collision_card"],
            "per_fold": rows}


def nine_fold(ctx, run, verbose=True):
    from sklearn.model_selection import StratifiedKFold
    cats = ctx["clusters"][COLL][1]
    coll = [i for i in ctx["ids"] if ctx["cat_of"][i] in cats]
    y = np.array([ctx["cat_of"][i] for i in coll])
    settings = sorted({ctx["set_of"][i] for i in coll})
    loso = [([i for i in coll if ctx["set_of"][i] != s], [i for i in coll if ctx["set_of"][i] == s])
            for s in settings]

    # (a) stratified 9-fold over cycles, ungrouped inner CV; seed also shuffles folds
    # (b) leave-one-setting-out, inner CV grouped by setting
    folds_a = {s: [([coll[j] for j in tr], [coll[j] for j in te]) for tr, te in
                   StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(np.zeros(len(coll)), y)]
               for s in INNER_SEEDS}
    tasks = [_task(ctx, COLL, f, t, s, grouped=False, pred=True)
             for s in INNER_SEEDS for f, t in folds_a[s]]
    tasks += [_task(ctx, COLL, f, t, s, grouped=True, pred=True)
              for s in INNER_SEEDS for f, t in loso]
    res = run(tasks)
    n = len(INNER_SEEDS) * N_FOLDS
    res_a, res_b = res[:n], res[n:]

    a_rows, b_rows = [], []
    for k, s in enumerate(INNER_SEEDS):
        a = _pooled(ctx, folds_a[s], res_a[k * N_FOLDS:(k + 1) * N_FOLDS])
        b = _pooled(ctx, loso, res_b[k * len(loso):(k + 1) * len(loso)], "held_out_setting")
        a_rows.append(dict(seed=s, **a))
        b_rows.append(dict(seed=s, **b))

    # pre-registered set and deployed set under leave-one-setting-out: no search,
    # no inner CV, so independent of the inner seed
    truth, p_pre, p_mag = [], [], []
    for fit, test in loso:
        y_fit = np.array([ctx["cat_of"][i] for i in fit])
        truth += [ctx["cat_of"][i] for i in test]
        p_pre += _predict(fit, test, y_fit, PREREG)
        p_mag += _predict(fit, test, y_fit, ["mag"])
    pre, mag = _rates(p_pre, truth), _rates(p_mag, truth)

    def agg(rows):
        return {"searched_per_seed": [r["searched"] for r in rows],
                "adopted_per_seed": [r["adopted"] for r in rows],
                "pooled_acc": _summ([r["pooled_acc"] for r in rows]),
                "pooled_balanced_acc": _summ([r["pooled_balanced_acc"] for r in rows]),
                "pooled_carton_recall": _summ([r["pooled_carton_recall"] for r in rows]),
                "pooled_majority_acc": _summ([r["pooled_majority_acc"] for r in rows]),
                "pooled_deployed_acc": _summ([r["pooled_deployed_acc"] for r in rows])}

    out = {
        "n_cycles": len(coll), "n_settings": len(settings), "folds": N_FOLDS,
        "inner_seeds": list(INNER_SEEDS),
        "stratified_cycles": dict(
            protocol="StratifiedKFold(9, shuffle=True, random_state=seed) over the 94 "
                     "collision cycles, inner CV ungrouped", **agg(a_rows), per_seed=a_rows),
        "leave_one_setting_out": dict(
            protocol="leave-one-setting-out over the 9 collision settings, inner CV "
                     "grouped by setting", **agg(b_rows), per_seed=b_rows,
            preregistered={"dims": PREREG, "pooled_acc": pre["acc"],
                           "pooled_balanced_acc": pre["balanced_acc"],
                           "pooled_carton_recall": pre["recall"]["collision_card"],
                           "pooled_foam_recall": pre["recall"]["collision_foam"]},
            deployed={"dims": ["mag"], "pooled_acc": mag["acc"],
                      "pooled_balanced_acc": mag["balanced_acc"],
                      "pooled_carton_recall": mag["recall"]["collision_card"],
                      "pooled_foam_recall": mag["recall"]["collision_foam"]}),
    }
    if verbose:
        print("\n4. NINE FOLDS ON THE COLLISION CYCLES, inner seeds 0-9")
        for key, label in (("stratified_cycles", "(a) 9-fold over cycles  "),
                           ("leave_one_setting_out", "(b) leave-one-setting-out")):
            d = out[key]
            print(f"   {label}: searched {d['searched_per_seed']}  adopted {d['adopted_per_seed']}")
            print(f"      pooled {d['pooled_acc']['mean']:.3f} [{d['pooled_acc']['min']:.3f}, "
                  f"{d['pooled_acc']['max']:.3f}]  majority {d['pooled_majority_acc']['mean']:.3f}"
                  f"  deployed {d['pooled_deployed_acc']['mean']:.3f}")
        p = out["leave_one_setting_out"]["preregistered"]
        m = out["leave_one_setting_out"]["deployed"]
        print(f"   (b) pre-registered: acc {p['pooled_acc']:.3f}  balanced "
              f"{p['pooled_balanced_acc']:.3f}  carton recall {p['pooled_carton_recall']:.3f}")
        print(f"   (b) deployed      : acc {m['pooled_acc']:.3f}  balanced "
              f"{m['pooled_balanced_acc']:.3f}  carton recall {m['pooled_carton_recall']:.3f}")
    return out


# ---------------------------------------------------------------------------
# 5. Setting-level permutation tests under the random split
# ---------------------------------------------------------------------------
def _perm_labels(ctx, name, setting_label):
    cats = ctx["clusters"][name][1]
    return {i: setting_label[ctx["set_of"][i]] for i in ctx["ids"] if ctx["cat_of"][i] in cats}


def _cluster_settings(ctx, name):
    cats = ctx["clusters"][name][1]
    lab = {ctx["set_of"][i]: ctx["cat_of"][i] for i in ctx["ids"] if ctx["cat_of"][i] in cats}
    return sorted(lab), lab


def perm_collision(ctx, run, verbose=True):
    """All C(9,3) = 84 ways to call 3 of the 9 settings carton, per seed."""
    t0 = time.time()
    settings, true = _cluster_settings(ctx, COLL)
    n_carton = sum(v == CARTON for v in true.values())
    labellings = [tuple(c) for c in itertools.combinations(settings, n_carton)]
    identity = tuple(s for s in settings if true[s] == CARTON)
    tasks = []
    for s in SEEDS:
        fit, test = _split(ctx, COLL, "random", s)
        for carton in labellings:
            lab = {x: (CARTON if x in carton else FOAM) for x in settings}
            tasks.append(_task(ctx, COLL, fit, test, s, grouped=False,
                               label_of=_perm_labels(ctx, COLL, lab)))
    res = run(tasks)
    rows = []
    for k, s in enumerate(SEEDS):
        r = res[k * len(labellings):(k + 1) * len(labellings)]
        acc = [c["acc_test"] for c in r]
        obs = acc[labellings.index(identity)]
        n_ge = int(sum(a >= obs - 1e-12 for a in acc))
        rows.append({"seed": s, "observed": obs, "n_ge_observed": n_ge,
                     "p": n_ge / len(labellings),
                     "null_mean": float(np.mean(acc)), "null_sd": float(np.std(acc)),
                     "null_max": float(max(acc)),
                     "null_adopted": int(sum(c["adopted"] for c in r)),
                     "null_acc": acc})
    out = {"cluster": COLL, "labellings": len(labellings), "exhaustive": True,
           "settings": settings, "true_carton_settings": list(identity),
           "carton_settings_per_labelling": [list(c) for c in labellings],
           "p_rule": "fraction of the 84 labellings (identity included) with held-out "
                     "accuracy >= observed",
           "per_seed": rows,
           "p": _summ([r["p"] for r in rows]),
           "null_mean": _summ([r["null_mean"] for r in rows]),
           "runtime_s": time.time() - t0}
    if verbose:
        print("\n5a. SETTING-LEVEL PERMUTATION, collision, all 84 labellings")
        for r in rows:
            print(f"   seed {r['seed']}: observed {r['observed']:.3f}  null mean "
                  f"{r['null_mean']:.3f}  >= observed {r['n_ge_observed']}/84  p {r['p']:.3f}")
    return out


def perm_load(ctx, run, draws=LOAD_DRAWS, verbose=True):
    """Random relabellings of the 32 load-torque settings, counts per category kept."""
    t0 = time.time()
    settings, true = _cluster_settings(ctx, LOAD)
    true_vec = np.array([true[x] for x in settings])
    tasks, n_identity = [], []
    for s in SEEDS:
        fit, test = _split(ctx, LOAD, "random", s)
        rng = np.random.RandomState(s)
        vecs = [true_vec] + [rng.permutation(true_vec) for _ in range(draws)]
        n_identity.append(int(sum(np.array_equal(v, true_vec) for v in vecs[1:])))
        for v in vecs:
            lab = {x: int(c) for x, c in zip(settings, v)}
            tasks.append(_task(ctx, LOAD, fit, test, s, grouped=False,
                               label_of=_perm_labels(ctx, LOAD, lab)))
    res = run(tasks)
    rows = []
    for k, s in enumerate(SEEDS):
        r = res[k * (draws + 1):(k + 1) * (draws + 1)]
        obs, acc = r[0]["acc_test"], [c["acc_test"] for c in r[1:]]
        n_ge = int(sum(a >= obs - 1e-12 for a in acc))
        rows.append({"seed": s, "observed": obs, "observed_adopted": r[0]["adopted"],
                     "n_ge_observed": n_ge, "p": (1 + n_ge) / (1 + draws),
                     "null_mean": float(np.mean(acc)), "null_sd": float(np.std(acc)),
                     "null_max": float(max(acc)),
                     "null_adopted": int(sum(c["adopted"] for c in r[1:])),
                     "identity_draws": n_identity[k], "null_acc": acc})
    stored = _stored()["leakage_check"]["per_cluster"][LOAD]["random"]["acc"]
    out = {"cluster": LOAD, "draws_per_seed": draws, "exhaustive": False,
           "draw_rng": "numpy RandomState(seed), one permutation of the 32 setting labels per draw",
           "settings_per_category": {CAT_NAME[c]: int(np.sum(true_vec == c))
                                     for c in sorted(set(true_vec.tolist()))},
           "p_rule": "(1 + draws with held-out accuracy >= observed) / (1 + draws)",
           "per_seed": rows,
           "p": _summ([r["p"] for r in rows]),
           "null_mean": _summ([r["null_mean"] for r in rows]),
           "observed": _summ([r["observed"] for r in rows]),
           "stored_leakage_check_acc": stored,
           "observed_matches_stored": bool(
               abs(np.mean([r["observed"] for r in rows]) - stored) < 1e-12),
           "runtime_s": time.time() - t0}
    if draws != LOAD_DRAWS:
        out["note"] = f"{draws} labellings per seed instead of {LOAD_DRAWS}"
    if verbose:
        print(f"\n5b. SETTING-LEVEL PERMUTATION, load torque, {draws} labellings per seed")
        for r in rows:
            print(f"   seed {r['seed']}: observed {r['observed']:.3f}  null mean "
                  f"{r['null_mean']:.3f}  >= observed {r['n_ge_observed']}/{draws}  p {r['p']:.4f}")
    return out


# ---------------------------------------------------------------------------
# 6. Retrieval top-1 under both protocols
# ---------------------------------------------------------------------------
def retrieval_protocol(ctx, verbose=True):
    from baselines import _tied
    ids, cat_of = ctx["ids"], ctx["cat_of"]

    def top1(test):
        return float(np.mean([ctx["cat_of_fm"][_tied(ctx, i)[0]] == cat_of[i] for i in test]))

    out = {"seeds": list(RETRIEVAL_SEEDS), "tie_break": "FMEA declaration order",
           "sd_note": "sample standard deviation over splits (ddof=1)"}
    for key, split in (("random_split", lambda s: R._stratified_split(ids, cat_of, s)),
                       ("leave_setting_out", lambda s: R._grouped_split(ids, cat_of, ctx["set_of"], s))):
        tests = [split(s)[1] for s in RETRIEVAL_SEEDS]
        v = [top1(t) for t in tests]
        out[key] = {"per_seed": v, "n_test": [len(t) for t in tests],
                    "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                    "mean_seeds_0_4": float(np.mean(v[:5]))}
    out["difference_of_means"] = out["leave_setting_out"]["mean"] - out["random_split"]["mean"]
    stored = _stored()["leakage_check"]
    out["stored_seeds_0_4"] = {"random_split": stored["random_split"]["point_top1"],
                               "leave_setting_out": stored["leave_setting_out"]["point_top1"]}
    out["seeds_0_4_match_stored"] = all(
        abs(out[k]["mean_seeds_0_4"] - out["stored_seeds_0_4"][k]) < 1e-12
        for k in ("random_split", "leave_setting_out"))
    if verbose:
        print(f"\n6. RETRIEVAL TOP-1, {len(RETRIEVAL_SEEDS)} splits per protocol")
        for k in ("random_split", "leave_setting_out"):
            print(f"   {k:18s} {out[k]['mean']:.4f} +/- {out[k]['sd']:.4f} (sd)")
        print(f"   seeds 0-4 match repair_results.json: {out['seeds_0_4_match_stored']}")
    return out


# ---------------------------------------------------------------------------
def _flag(argv, name, default):
    if name in argv:
        return int(argv[argv.index(name) + 1])
    return default


def main(argv):
    t_all = time.time()
    jobs = _flag(argv, "--jobs", os.cpu_count() or 1)
    draws = _flag(argv, "--load-draws", LOAD_DRAWS)
    skip_perm = "--skip-perm" in argv
    skip_load = skip_perm or "--skip-load-perm" in argv

    ctx = _context()
    _init(ctx["cycles"], ctx["columns"])
    coll_ids = [i for i in ctx["ids"] if ctx["cat_of"][i] in ctx["clusters"][COLL][1]]
    print(f"{len(ctx['ids'])} anomalous cycles; collision group {len(coll_ids)} cycles; "
          f"{jobs} worker process(es)")

    res = {}
    if os.path.exists(RESULTS_OUT):
        with open(RESULTS_OUT, encoding="utf-8") as fh:
            res = json.load(fh)
    runtime = res.get("runtime_s", {})
    run, pool = _runner(jobs)
    try:
        t = time.time()
        res["random_split"] = random_split(ctx, run)
        res["preregistered_free"] = preregistered_free(ctx)
        res["random_grouped_inner"] = random_grouped_inner(ctx, run, res["random_split"])
        res["nine_fold"] = nine_fold(ctx, run)
        res["retrieval_protocol"] = retrieval_protocol(ctx)
        runtime["parts_1_to_4_and_6"] = time.time() - t
        perm = res.get("setting_permutation", {})
        if not skip_perm:
            perm["collision"] = perm_collision(ctx, run)
            # the identity labelling must reproduce part 1
            perm["collision"]["observed_matches_part1"] = all(
                abs(r["observed"] - p["held_out_acc"]) < 1e-12 for r, p in
                zip(perm["collision"]["per_seed"], res["random_split"]["per_seed"]))
            runtime["collision_permutation"] = perm["collision"]["runtime_s"]
        if not skip_load:
            perm["load_torque"] = perm_load(ctx, run, draws)
            runtime["load_permutation"] = perm["load_torque"]["runtime_s"]
        if perm:
            res["setting_permutation"] = perm
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    runtime["last_run_total"] = time.time() - t_all
    runtime["jobs"] = jobs
    res["runtime_s"] = runtime
    res["cluster"] = {"name": COLL, "categories": {str(FOAM): CAT_NAME[FOAM],
                                                   str(CARTON): CAT_NAME[CARTON]},
                      "n_cycles": len(coll_ids),
                      "n_settings": len({ctx["set_of"][i] for i in coll_ids})}
    res["adopt_margin"] = list(R.ADOPT_MARGIN)
    res["note"] = ("Collision-group protocol check. Search, adoption rule and margins are "
                   "repair.certify / repair.ADOPT_MARGIN unchanged. Outer random split = "
                   "repair._stratified_split, leave-setting-out = repair._grouped_split, both "
                   "over all 755 cycles and then restricted to the group, as repair.evaluate does.")
    with open(RESULTS_OUT, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten -> {os.path.relpath(RESULTS_OUT, _ROOT)}  "
          f"({time.time() - t_all:.0f} s)")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
