# -*- coding: utf-8 -*-
"""How many fault instantiations does a repair certificate need?

THE QUESTION THIS ANSWERS
-------------------------
`repair.py` returns three kinds of verdict. Two are scientific claims (adopted;
rejected because the evidence does not contain the distinction) and one is a
confession: INCONCLUSIVE, meaning the grouped inner search could not be
validated on a single split because the dataset has too few instantiations of
the fault. Five voraus-AD categories are in that state -- coll_card and platform
with 3 instantiations, misgrip, entangled and invalidgrip with 1.

"Too few" was never quantified, because with real data we cannot vary it. With
`synth_faults` we can: instantiation count is a knob. So we sweep it and report
the smallest N at which the protocol stops saying inconclusive, and the N at
which its verdict stabilizes.

WHY THIS IS NOT CIRCULAR
------------------------
The injected faults carry signatures we designed, so nothing here may be used to
claim a repair works on real faults (see synth_faults.__doc__). But the quantity
measured here is a property of the PROTOCOL -- how much data it needs before it
will commit -- and that is independent of whether the injected physics is a
faithful rendering of a real dropped part. The verdict content is reported only
to show it stabilizes, never as evidence about the real cluster.

Run:  ..\\.venv\\Scripts\\python.exe synth_power.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import repair
import synth_faults as sf

_ROOT = sf._ROOT
RESULTS_OUT = os.path.join(_ROOT, "results", "synth_instantiation_power.json")

# the gripping structural twin -- the cluster the real dataset reports as
# UNDETERMINED, with exactly one instantiation per member
CLUSTER = ("FM_misgrip", "FM_invalidgrip")
MAX_INST = 24                 # generated once, then subsampled
CYCLES_PER_INST = 8
N_GRID = (2, 3, 4, 6, 8, 12, 16, 20)
SEEDS = (0, 1, 2, 3, 4)


def build_pool(verbose=True):
    """Generate MAX_INST instantiations per member, once, and index them in the
    feature layout repair.py expects (one record's signature block per cycle)."""
    from attribution_bridge import build_reference

    if verbose:
        print(f"generating {MAX_INST} instantiations x {CYCLES_PER_INST} cycles "
              f"for each of {CLUSTER} ...")
    normals, cols = sf._load_normals()
    grid = np.linspace(0, 1, sf.GRID_N)
    ref = build_reference(normals, cols, grid)

    cycles, columns = {}, None
    # label each member with a distinct pseudo-category so repair.certify can
    # treat them as classes, and give each instantiation its own `setting`
    for m, fm in enumerate(CLUSTER):
        recs = sf.generate(fm, MAX_INST, CYCLES_PER_INST, seed=1000 + m,
                           normals=normals, ref=ref, id_base=900000 + m * 100000,
                           verbose=verbose)
        for r in recs:
            sig = r["signature"]
            if columns is None:
                columns = sorted(sig["slots"])
            f = {f"{slot}::{st}": v for slot, vals in sig["slots"].items()
                 for st, v in zip(sig["stats"], vals)}
            cycles[str(r["cycle_id"])] = {
                "category": m,                                   # 0 / 1
                "setting": m * 10_000 + r["synthetic"]["instantiation"],
                "fm": fm, "inst": r["synthetic"]["instantiation"], "f": f}
    if verbose:
        print(f"pool: {len(cycles)} cycles, {len(columns)} record slots")
    return cycles, columns


def _split_instantiations(cycles, n_inst, seed):
    """Leave-instantiation-out: whole instantiations go to fit or to test."""
    rng = np.random.default_rng(seed)
    fit, test = [], []
    for m in (0, 1):
        insts = sorted({c["inst"] for c in cycles.values() if c["category"] == m})
        pick = rng.permutation(insts)[:n_inst]
        k = max(1, len(pick) // 2)
        fit_i, test_i = set(pick[:k].tolist()), set(pick[k:].tolist())
        for cid, c in cycles.items():
            if c["category"] != m:
                continue
            if c["inst"] in fit_i:
                fit.append(cid)
            elif c["inst"] in test_i:
                test.append(cid)
    return fit, test


def sweep(cycles, columns, verbose=True):
    rows = []
    for n_inst in N_GRID:
        got_verdict = validated = adopted = 0
        accs, deps, inner, inner_dep, trials = [], [], [], [], 0
        for seed in SEEDS:
            fit, test = _split_instantiations(cycles, n_inst, seed)
            y_fit = np.array([cycles[i]["category"] for i in fit])
            y_test = np.array([cycles[i]["category"] for i in test])
            if len(set(y_fit.tolist())) < 2 or len(test) < 6:
                trials += 1
                continue
            groups = np.array([cycles[i]["setting"] for i in fit])
            cert = repair.certify(cycles, columns, "misgrip+invalidgrip",
                                  list(CLUSTER), fit, test, y_fit, y_test,
                                  groups, seed)
            trials += 1
            validated += bool(cert["search_validated"])
            if cert["search_validated"]:
                got_verdict += 1
                adopted += bool(cert["adopted"])
                accs.append(cert["acc_test"])
                # The baseline the adoption rule compares against, recorded so a
                # reject at ~0.99 can be read off rather than inferred.
                deps.append(cert["acc_deployed_repr"])
                inner.append(cert["inner_cv_fit"])
                inner_dep.append(cert["inner_cv_deployed_repr"])
        rows.append({"n_instantiations": n_inst, "trials": trials,
                     "search_validated": validated / max(trials, 1),
                     "verdict_rate": got_verdict / max(trials, 1),
                     "adopt_rate_given_verdict": (adopted / got_verdict) if got_verdict else None,
                     "acc_test_mean": float(np.mean(accs)) if accs else None,
                     "acc_test_sd": float(np.std(accs)) if len(accs) > 1 else None,
                     "acc_deployed_mean": float(np.mean(deps)) if deps else None,
                     "inner_cv_fit_mean": float(np.mean(inner)) if inner else None,
                     "inner_cv_deployed_mean": float(np.mean(inner_dep)) if inner_dep else None})
        if verbose:
            r = rows[-1]
            acc = f"{r['acc_test_mean']:.3f}" if r["acc_test_mean"] is not None else "  -- "
            print(f"  N={n_inst:>3}  grouped search runs {r['search_validated']:.0%}"
                  f"   verdict {r['verdict_rate']:.0%}"
                  f"   held-out {acc}"
                  + (f" (deployed {r['acc_deployed_mean']:.3f})"
                     if r["acc_deployed_mean"] is not None else "")
                  + (f"   adopt {r['adopt_rate_given_verdict']:.0%}"
                     if r["adopt_rate_given_verdict"] is not None else ""))
    return rows


def main():
    print("=" * 72)
    print("INSTANTIATION POWER — how much data a repair certificate needs")
    print("=" * 72)
    print("Cluster under test: the gripping structural twin, which the real")
    print("dataset reports as UNDETERMINED (1 instantiation per member).")
    print("Faults are semi-synthetic; only the DATA REQUIREMENT is claimed.\n")

    cycles, columns = build_pool()
    print("\nsweep (leave-instantiation-out, 5 draws per point):")
    rows = sweep(cycles, columns)

    ok = [r for r in rows if r["search_validated"] >= 0.8]
    runs_at = ok[0]["n_instantiations"] if ok else None

    # The verdict is STABLE from the smallest N onward at which every larger N
    # agrees with it. Running is not the same as deciding: just above the
    # mechanical threshold the adoption call flips between draws.
    stable_at = None
    for i, r in enumerate(rows):
        if r["adopt_rate_given_verdict"] is None:
            continue
        tail = [q["adopt_rate_given_verdict"] for q in rows[i:]
                if q["adopt_rate_given_verdict"] is not None]
        if tail and all(abs(v - tail[0]) < 1e-9 for v in tail):
            stable_at = r["n_instantiations"]
            break

    print()
    if runs_at:
        print(f"=> N <= {runs_at - 1}: the grouped inner search CANNOT run "
              f"(StratifiedGroupKFold needs >= 2 instantiations per class in the")
        print(f"   fit half), so the protocol can only return 'inconclusive'.")
        print(f"=> N >= {runs_at}: it runs on every draw.")
    if stable_at:
        unstable = [r for r in rows if r["adopt_rate_given_verdict"] not in (None,)
                    and r["n_instantiations"] < stable_at]
        if unstable:
            lo = unstable[0]["n_instantiations"]
            hi = unstable[-1]["n_instantiations"]
            flips = ", ".join("{:.0%}".format(r["adopt_rate_given_verdict"])
                              for r in unstable)
            print("=> N = {}-{}: it runs, but the ADOPTION verdict flips between "
                  "draws ({}) -- a false-adoption regime.".format(lo, hi, flips))
        print(f"=> N >= {stable_at}: the verdict is stable across every draw and "
              f"every larger N.")
    print()
    print("   voraus-AD has 1 instantiation for each gripping mode and 3 for")
    print("   carton collision and platform. The N >= 4 threshold follows from the")
    print("   50/50 grouped split (floor(N/2) per member for training, two needed")
    print("   by the grouped inner search). Gripping cannot be evaluated by any")
    print("   grouped protocol; collision can under leave-one-setting-out.")
    print()
    print("   CAVEAT: the injected pair separates at ~0.99, far cleaner than a")
    print("   real gripping fault, so the N reported here is a FLOOR. A subtler")
    print("   real fault needs at least this many instantiations, probably more.")

    payload = {"cluster": list(CLUSTER), "cycles_per_instantiation": CYCLES_PER_INST,
               "seeds": list(SEEDS), "sweep": rows,
               "min_instantiations_for_validated_search": runs_at,
               "min_instantiations_for_stable_verdict": stable_at,
               "caveat": "semi-synthetic faults; only the data requirement is "
                         "claimed, never that a repair works on real faults. The "
                         "injected pair is unusually separable, so these counts "
                         "are a floor."}
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwritten -> {RESULTS_OUT}")


if __name__ == "__main__":
    main()
