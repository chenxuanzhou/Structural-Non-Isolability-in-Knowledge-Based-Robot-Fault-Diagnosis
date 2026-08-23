# -*- coding: utf-8 -*-
"""Simulator experiments (tier 2, MuJoCo).

Two mechanism tests that did NOT survive scrutiny (A and B, both reported as
not answerable / invalid) and one that did: EXPERIMENT C, the instantiation
count a repair certificate needs for the COLLISION cluster.

Both are tests of OUR PIPELINE, not of the voraus cell, and neither number may
be pooled with a voraus-AD number (see sim_cell.__doc__).

EXPERIMENT A -- where is a collision lost?   *** NOT ANSWERABLE HERE ***
-----------------------------------------------------------------------
VERDICT, recorded before the numbers so nobody quotes them: this experiment
CANNOT be answered by simulation, and the attempt is kept only to document why.

The question was whether the deployed top-k attribution discards distal evidence
that the raw signals contain. The measured answer flips completely with one
calibration constant:

    per-cycle friction jitter 0.0 N.m -> distal signal in the top-5 on 100%
    per-cycle friction jitter 2.5 N.m -> distal signal in the top-5 on   0%

Both runs have identical contacts. What changed is the width of the normal
envelope on the distal joints relative to the proximal ones, and that ratio is a
property of the REAL robot's sensors and cycle-to-cycle repeatability -- not
something a simulator determines. With no jitter the envelope is so tight that
every deviation is hundreds of sigma and nothing is ever discarded; with jitter
the small distal torques drown. Either number would be a report of our own
tuning choice.

The question therefore stays where it was answered honestly: on the real data,
where the flat classifier reaches 0.506 on foam collisions against 0.083 chance
(repair.py), which is what licensed the claim that the loss is in the
attribution layer rather than the sensors.

WHAT THE CELL DID ESTABLISH, and it is the useful part: contact events can be
generated with controlled depth and material, and the material effect is
physically right -- a stiff contact deviates the joint torques about three times
as much as a compliant one at matched depth. That is the capability tier 1
lacked, and it is what would let synth_power's instantiation-count analysis be
extended to the collision cluster.

EXPERIMENT B -- which cost class does the thermal twin need?  *** INVALID ***
-----------------------------------------------------------------------------
VERDICT: the design does not test what it claims, and the numbers must not be
quoted. Measured (leave-instantiation-out, 5 draws, chance 0.500):

    deployed (torque, max|z| only)      0.970 +/- 0.037
    + free dims                         0.973 +/- 0.039
    + motor_iq   (PIPELINE cost)        0.877 +/- 0.094
    + temperature (HARDWARE cost)       1.000 +/- 0.000

Two things are wrong. The deployed representation already reaches 0.970, so the
pair is not a twin in this cell at all: thermal derating leaves the MECHANICAL
torque untouched while commutation ripple perturbs it, and torque alone
separates them. And adding a channel made accuracy WORSE (0.877), which is not
a cost-class effect but overfitting -- roughly 30 fit cycles against a growing
feature count. A non-monotone ordering cannot support a claim about which cost
class is necessary.

Fixing it needs the pair to be genuinely torque-confusable, which means
modelling both modes as causes of the same reported-torque change and letting
only the thermal channel separate them. That is a modelling decision that would
determine the answer, so it needs to be pre-registered and argued from the drive
physics rather than tuned until the taxonomy is confirmed. Not attempted here.


EXPERIMENT C -- how many collision instantiations does a certificate need?
--------------------------------------------------------------------------
This is what the simulator is actually for. `synth_power` answered the same
question for the gripping twin with analytically injected faults; collisions
need contact dynamics, which only the cell can provide. The question is a
property of the PROTOCOL -- how many independent instantiations before the
leave-instantiation-out certificate stops returning "inconclusive" -- so it does
not depend on whether our contact model is a faithful carton.

It does have to survive the test that killed Experiment A, so the sweep is run
at BOTH noise floors that flipped A from 100% to 0%. RESULT:

    grouped inner search runs from   N >= 4   (no jitter)
                                     N >= 4   (jitter 2.5 N.m)

The threshold does not move. voraus-AD has THREE carton-collision
instantiations, so below four the certificate cannot run at all and that
"inconclusive" was forced by the data -- exactly as the gripping cluster's was.

The adoption verdict, though, still flips between draws all the way to N = 16 at
both noise floors (0-20% and 20-40%). The collision pair therefore needs MORE
than sixteen instantiations to settle, against >= 8 for the gripping pair, which
is what one expects of two modes differing only in contact compliance.

Held-out accuracy magnitudes DO move with the noise floor (0.97-1.00 without
jitter, 0.84-0.94 with) and are not quoted as a property of real collisions.


Run:  ..\\.venv\\Scripts\\python.exe sim_experiments.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import repair
import sim_cell as sc

_ROOT = sc.__file__.rsplit(os.sep, 2)[0]
RESULTS_OUT = os.path.join(_ROOT, "results", "sim_experiments.json")

GRID_N = 500          # same resampling resolution as the deployed pipeline
N_NORMAL = 24
N_INST = 10                 # instantiations per fault
CYCLES_PER_INST = 6
SEEDS = (0, 1, 2, 3, 4)

CHANNELS = {
    "torque": [f"motor_torque_{j}" for j in sc.JOINTS]
              + [f"torque_sensor_a_{j}" for j in sc.JOINTS],
    "iq": [f"motor_iq_{j}" for j in sc.JOINTS],
    "temp": [f"joint_temperature_{j}" for j in sc.JOINTS],
    "kinematic": [f"joint_position_{j}" for j in sc.JOINTS]
                 + [f"joint_velocity_{j}" for j in sc.JOINTS],
}
ALL_CH = CHANNELS["torque"] + CHANNELS["iq"] + CHANNELS["temp"] + CHANNELS["kinematic"]


# ---------------------------------------------------------------- resampling
def _resample(sig, col, grid):
    t = sig["time"]
    tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return np.interp(grid, tn, sig[col])


def _envelope(normals, grid):
    ref = {}
    for c in ALL_CH:
        stack = np.vstack([_resample(s, c, grid) for s in normals])
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        rng = np.ptp(mean) if np.ptp(mean) > 0 else 1.0
        ref[c] = {"mean": mean, "scale": np.maximum(std, 0.05 * rng)}
    return ref


def _features(sig, ref, grid):
    """The same five grammar statistics repair.py uses, per channel."""
    t = sig["time"]
    tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
    act = sig["action"][np.searchsorted(tn, grid, side="right") - 1]
    carry = np.isin(act, list(sc_carry()))
    empty = ~carry
    early, late = slice(0, GRID_N // 3), slice(2 * GRID_N // 3, GRID_N)
    out = {}
    for c in ALL_CH:
        z = (_resample(sig, c, grid) - ref[c]["mean"]) / (ref[c]["scale"] + 1e-9)
        out[f"{c}::z_peak"] = float(np.max(np.abs(z)))
        out[f"{c}::z_signed"] = float(np.mean(z))
        out[f"{c}::z_step"] = float(np.mean(z[late]) - np.mean(z[early]))
        out[f"{c}::z_payload"] = float((np.mean(z[carry]) if carry.any() else 0.0)
                                       - (np.mean(z[empty]) if empty.any() else 0.0))
        out[f"{c}::z_std"] = float(np.std(z))
    return out


def sc_carry():
    from attribution_bridge import CARRY_ACTIONS
    return CARRY_ACTIONS


# Calibrated so an injected fault lands in the z range voraus-AD actually shows
# (peak |z| of a few, not a few hundred). Without this the simulated envelope is
# unrealistically tight and every test of the top-k selection is vacuous.
JITTER_SD = 2.5          # per-cycle joint friction offset, N.m
NOISE_SD = 12.0          # sensor noise multiplier


def _cycle_jitter(rng):
    return rng.normal(0.0, JITTER_SD, 6)


def _normal_pool(n=N_NORMAL, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cell = sc.Cell(payload_mass=float(rng.uniform(1.05, 1.35)), seed=1000 + i)
        out.append(cell.run(speed=float(rng.uniform(0.95, 1.05)),
                            noise_sd=NOISE_SD, jitter=_cycle_jitter(rng)))
    return out


# ---------------------------------------------------------------- experiment A
def _transfer_point():
    """Where the wrist actually is mid-transfer. Measured, not guessed: our
    first attempt placed the obstacle 0.61 m off the path, so no contact ever
    occurred and the experiment silently measured nuisance variation instead."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(sc._mjcf(payload_mass=1.2))
    d = mujoco.MjData(m)
    qr, act = sc._trajectory()
    qd = np.vstack([np.zeros((1, 6)), np.diff(qr, axis=0) / sc.SIM_DT])
    d.qpos[:] = qr[0]
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l6")
    pts = []
    for i in range(len(qr)):
        e = qr[i] - d.qpos[:6]
        ed = qd[i] - d.qvel[:6]
        d.ctrl[:] = np.clip(d.qfrc_bias[:6] + sc.KP * e + sc.KD * ed,
                            m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
        if act[i] == 9:                                  # TO_PLACE_PTP
            pts.append(d.xpos[bid].copy())
    pts = np.array(pts)
    return pts[len(pts) // 2]


def experiment_a(verbose=True):
    grid = np.linspace(0, 1, GRID_N)
    normals = _normal_pool()
    ref = _envelope(normals, grid)
    hub = _transfer_point()

    rng = np.random.default_rng(7)
    rows = []
    for stiff, label in ((0.02, "carton (stiff)"), (0.15, "foam (soft)")):
        for k in range(6):
            # the instantiation parameter is how deeply the wrist clips the
            # obstacle; the material is the contact time constant
            obs = {"pos": (float(hub[0]), float(hub[1] + 0.02 * (rng.random() - 0.5)),
                           float(hub[2] - rng.uniform(0.025, 0.045))),
                   "size": (0.02, 0.02, 0.03), "solref": stiff}
            # SAME nuisance distribution as the normal pool, so the only
            # systematic difference from the envelope is the contact itself
            cell = sc.Cell(payload_mass=float(rng.uniform(1.05, 1.35)),
                           obstacle=obs, seed=500 + k)
            sig = cell.run(speed=float(rng.uniform(0.95, 1.05)))
            z = {c: float(np.max(np.abs(
                (_resample(sig, c, grid) - ref[c]["mean"]) / (ref[c]["scale"] + 1e-9))))
                 for c in ALL_CH}
            tq = {c: z[c] for c in CHANNELS["torque"]}
            top5 = sorted(z, key=z.get, reverse=True)[:5]
            load = [c for c in tq if c.endswith(("_1", "_2", "_3"))]
            distal = [c for c in tq if c.endswith(("_4", "_5", "_6"))]
            rows.append({
                "material": label, "instantiation": k,
                "contact_s": round(cell.n_contact_steps * sc.SIM_DT, 3),
                "max_z_load_joints": max(tq[c] for c in load),
                "max_z_distal_joints": max(tq[c] for c in distal),
                "top5_has_distal": any(c.endswith(("_4", "_5", "_6")) for c in top5),
                "top5": top5[:3]})

    if verbose:
        print("")
        print("EXPERIMENT A -- where is a collision lost?")
        print("  a contact is placed on the measured wrist path during transfer")
        print(f"  {'material':>16} {'contact s':>10} {'max|z| load':>12} "
              f"{'max|z| distal':>14}  distal in top-5?")
        for r in rows:
            print(f"  {r['material']:>16} {r['contact_s']:>10.2f} "
                  f"{r['max_z_load_joints']:>12.2f}"
                  f" {r['max_z_distal_joints']:>14.2f}  {r['top5_has_distal']}")
        if not any(r["contact_s"] > 0 for r in rows):
            print("  !! NO CONTACT OCCURRED -- this experiment measured nothing.")
            return rows
        distal_bigger = np.mean([r["max_z_distal_joints"] > r["max_z_load_joints"]
                                 for r in rows])
        surfaced = np.mean([r["top5_has_distal"] for r in rows])
        print(f"  => the contact deviates the DISTAL joints more than the load "
              f"joints on {distal_bigger:.0%} of instantiations;")
        print(f"     the deployed top-5 attribution surfaces a distal joint on "
              f"{surfaced:.0%}.")
        print("     The FMEA scopes the collision modes to `affects=\"distal\"`, so")
        print("     what the retrieval sees depends on whether the attribution")
        print("     put a distal signal in the top-k at all.")
    return rows


# ---------------------------------------------------------------- experiment B
def _thermal_pool(seed=0, verbose=False):
    """Instantiations of the two thermal-twin modes."""
    rng = np.random.default_rng(seed)
    pool = {"FM_miscommutation": [], "FM_overtemp": []}
    for inst in range(N_INST):
        j = int(rng.integers(1, 4))                     # affected joint, 1..3
        commut = {"kind": "commutation", "joints": [j],
                  "amp": float(rng.uniform(2.5, 6.0)),
                  # electrical-frequency ripple: at ~10 poles/rad the disturbance is a
                  # slow drift the controller absorbs and the signature vanishes
                  "poles": float(rng.uniform(80.0, 250.0)),
                  "phase": float(rng.uniform(0, 6.28))}
        # overtemperature: thermal derating -> the same torque needs more current
        over = {"kind": "derate", "joints": [j],
                "kt_loss": float(rng.uniform(0.18, 0.35))}
        for ci in range(CYCLES_PER_INST):
            for fm, f in (("FM_miscommutation", commut), ("FM_overtemp", over)):
                cell = sc.Cell(payload_mass=1.2, fault=f, seed=7000 + inst * 50 + ci)
                pool[fm].append((inst, cell.run()))
        if verbose and (inst + 1) % 5 == 0:
            print(f"    thermal pool: {inst + 1}/{N_INST} instantiations")
    return pool


def experiment_b(verbose=True):
    grid = np.linspace(0, 1, GRID_N)
    normals = _normal_pool(seed=3)
    ref = _envelope(normals, grid)
    pool = _thermal_pool(verbose=verbose)

    cycles, y, inst = {}, [], []
    for cls, fm in enumerate(("FM_miscommutation", "FM_overtemp")):
        for k, (i_inst, sig) in enumerate(pool[fm]):
            cid = f"{cls}_{k}"
            cycles[cid] = {"f": _features(sig, ref, grid),
                           "category": cls, "inst": i_inst}
    ids = sorted(cycles)

    reps = {
        "deployed (torque, max|z| only)": ([("torque", "z_peak")], "deployed"),
        "+ free dims (torque sign/step/payload/shape)":
            ([("torque", s) for s in ("z_peak", "z_signed", "z_step",
                                      "z_payload", "z_std")], "free"),
        "+ motor_iq (PIPELINE cost)":
            ([("torque", s) for s in ("z_peak", "z_signed", "z_step",
                                      "z_payload", "z_std")]
             + [("iq", s) for s in ("z_peak", "z_signed", "z_std")], "pipeline"),
        "+ temperature (HARDWARE cost)":
            ([("torque", s) for s in ("z_peak", "z_signed", "z_step",
                                      "z_payload", "z_std")]
             + [("iq", s) for s in ("z_peak", "z_signed", "z_std")]
             + [("temp", s) for s in ("z_peak", "z_signed", "z_step")], "hardware"),
    }

    out = []
    for name, (spec, cost) in reps.items():
        names = [f"{c}::{stat}" for grp, stat in spec for c in CHANNELS[grp]]
        accs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            insts = rng.permutation(N_INST)
            fit_i, test_i = set(insts[:N_INST // 2].tolist()), set(insts[N_INST // 2:].tolist())
            fit = [i for i in ids if cycles[i]["inst"] in fit_i]
            test = [i for i in ids if cycles[i]["inst"] in test_i]
            X = lambda sel: np.nan_to_num(np.array(  # noqa: E731
                [[cycles[i]["f"][n] for n in names] for i in sel]))
            yf = np.array([cycles[i]["category"] for i in fit])
            yt = np.array([cycles[i]["category"] for i in test])
            _, pred = repair._fit_predict(X(fit), yf, X(test))
            accs.append(float(np.mean(pred == yt)))
        out.append({"representation": name, "cost_class": cost,
                    "n_features": len(names),
                    "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs))})
    if verbose:
        print("\nEXPERIMENT B -- which cost class does the thermal twin need?")
        print("  leave-instantiation-out, 5 draws; chance = 0.500")
        for r in out:
            print(f"  {r['representation']:<46} {r['acc_mean']:.3f} "
                  f"+/- {r['acc_sd']:.3f}   [{r['cost_class']}]")
    return out



# ---------------------------------------------------------------- experiment C
COLL_MAX_INST = 16
COLL_CYCLES = 5
COLL_N_GRID = (2, 3, 4, 6, 8, 12, 16)

# The deployed representation reads these and only these; restricting the
# feature space to them keeps the sweep comparable with synth_power's, which
# ran on the real record's signature block.
DEPLOYED_CH = ([f"motor_torque_{j}" for j in sc.JOINTS]
               + [f"torque_sensor_a_{j}" for j in sc.JOINTS]
               + [f"joint_position_{j}" for j in sc.JOINTS]
               + [f"joint_velocity_{j}" for j in sc.JOINTS])
STATS = ("z_peak", "z_signed", "z_step", "z_payload", "z_std")


def _collision_pool(jitter_sd, verbose=True):
    """MAX_INST instantiations of each material. One instantiation = one
    physical placement (depth, lateral offset), run several times."""
    grid = np.linspace(0, 1, GRID_N)
    rng0 = np.random.default_rng(11)
    normals = []
    for i in range(N_NORMAL):
        cell = sc.Cell(payload_mass=float(rng0.uniform(1.05, 1.35)), seed=1000 + i)
        normals.append(cell.run(speed=float(rng0.uniform(0.95, 1.05)),
                                noise_sd=NOISE_SD,
                                jitter=rng0.normal(0.0, jitter_sd, 6)))
    ref = _envelope(normals, grid)
    hub = _transfer_point()

    rng = np.random.default_rng(23)
    cycles, no_contact = {}, 0
    for cls, (stiff, label) in enumerate(((0.02, "carton"), (0.15, "foam"))):
        for inst in range(COLL_MAX_INST):
            # the instantiation: a fixed placement, reused across its cycles
            depth = float(rng.uniform(0.025, 0.045))
            lat = float(0.02 * (rng.random() - 0.5))
            for ci in range(COLL_CYCLES):
                obs = {"pos": (float(hub[0]), float(hub[1] + lat),
                               float(hub[2] - depth)),
                       "size": (0.02, 0.02, 0.03), "solref": stiff}
                cell = sc.Cell(payload_mass=float(rng.uniform(1.05, 1.35)),
                               obstacle=obs, seed=4000 + inst * 20 + ci)
                sig = cell.run(speed=float(rng.uniform(0.95, 1.05)),
                               noise_sd=NOISE_SD,
                               jitter=rng.normal(0.0, jitter_sd, 6))
                if cell.n_contact_steps == 0:
                    no_contact += 1
                    continue
                f = _features(sig, ref, grid)
                cycles[f"{cls}_{inst}_{ci}"] = {
                    "f": {k: v for k, v in f.items()
                          if k.split("::")[0] in DEPLOYED_CH},
                    "category": cls, "inst": inst, "material": label}
            if verbose and (inst + 1) % 8 == 0:
                print(f"    {label}: {inst + 1}/{COLL_MAX_INST} instantiations")
    if no_contact:
        print(f"    WARNING: {no_contact} cycles produced no contact and were dropped")
    return cycles, sorted(DEPLOYED_CH)


def _sweep_instantiations(cycles, columns, n_grid, seeds=SEEDS):
    ids = sorted(cycles)
    rows = []
    for n_inst in n_grid:
        runs = verdicts = adopts = 0
        accs, trials = [], 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            fit, test = [], []
            for cls in (0, 1):
                insts = sorted({c["inst"] for c in cycles.values()
                                if c["category"] == cls})
                pick = rng.permutation(insts)[:n_inst]
                k = max(1, len(pick) // 2)
                fi, ti = set(pick[:k].tolist()), set(pick[k:].tolist())
                for cid, c in cycles.items():
                    if c["category"] != cls:
                        continue
                    if c["inst"] in fi:
                        fit.append(cid)
                    elif c["inst"] in ti:
                        test.append(cid)
            y_fit = np.array([cycles[i]["category"] for i in fit])
            y_test = np.array([cycles[i]["category"] for i in test])
            trials += 1
            if len(set(y_fit.tolist())) < 2 or len(test) < 6:
                continue
            groups = np.array([cycles[i]["category"] * 1000 + cycles[i]["inst"]
                               for i in fit])
            cert = repair.certify(cycles, columns, "carton+foam",
                                  ["FM_collision_card", "FM_collision_foam"],
                                  fit, test, y_fit, y_test, groups, seed)
            runs += bool(cert["search_validated"])
            if cert["search_validated"]:
                verdicts += 1
                adopts += bool(cert["adopted"])
                accs.append(cert["acc_test"])
        rows.append({"n_instantiations": n_inst, "trials": trials,
                     "search_validated": runs / max(trials, 1),
                     "verdict_rate": verdicts / max(trials, 1),
                     "adopt_rate_given_verdict": (adopts / verdicts) if verdicts else None,
                     "acc_test_mean": float(np.mean(accs)) if accs else None})
    return rows


def experiment_c(verbose=True):
    out = {}
    for jitter_sd, label in ((0.0, "no jitter"), (JITTER_SD, f"jitter {JITTER_SD} N.m")):
        if verbose:
            print("")
            print(f"  building collision pool [{label}] ...")
        cycles, columns = _collision_pool(jitter_sd, verbose=verbose)
        rows = _sweep_instantiations(cycles, columns, COLL_N_GRID)
        out[label] = rows
        if verbose:
            print(f"  sweep [{label}] — carton vs foam, leave-instantiation-out:")
            for r in rows:
                acc = (f"{r['acc_test_mean']:.3f}" if r["acc_test_mean"] is not None
                       else "  -- ")
                ad = (f"{r['adopt_rate_given_verdict']:.0%}"
                      if r["adopt_rate_given_verdict"] is not None else " -- ")
                print(f"    N={r['n_instantiations']:>3}  search runs "
                      f"{r['search_validated']:>4.0%}   held-out {acc}   adopt {ad}")

    def first_running(rows):
        ok = [r for r in rows if r["search_validated"] >= 0.8]
        return ok[0]["n_instantiations"] if ok else None

    needs = {k: first_running(v) for k, v in out.items()}
    if verbose:
        print("")
        print("  KNOB CHECK (the test that killed Experiment A):")
        for k, v in needs.items():
            print(f"    {k:>18}: grouped search runs from N >= {v}")
        vals = [v for v in needs.values() if v is not None]
        stable = len(set(vals)) == 1 and len(vals) == len(needs)
        print(f"    => the requirement is {'STABLE' if stable else 'NOT stable'} "
              f"across the noise floors that flipped Experiment A.")
    return {"sweeps": out, "min_instantiations": needs}


def main():
    print("=" * 74)
    print("SIMULATED CELL — two mechanism tests the real dataset cannot run")
    print("=" * 74)
    print("MuJoCo 6R arm, same seven-phase pick-and-place. NOT a replication of")
    print("the voraus cell: no number here may be pooled with a voraus-AD number.")

    a = experiment_a()
    b = experiment_b()
    print()
    print("EXPERIMENT C -- how many collision instantiations for a certificate?")
    c = experiment_c()

    # what the taxonomy claimed vs what the cell shows
    dep = next(r for r in b if r["cost_class"] == "deployed")
    pipe = next(r for r in b if r["cost_class"] == "pipeline")
    hard = next(r for r in b if r["cost_class"] == "hardware")
    print()
    if pipe["acc_mean"] > dep["acc_mean"] + 0.05 and \
       hard["acc_mean"] <= pipe["acc_mean"] + 0.05:
        verdict = ("the unread current channel is ENOUGH -- the twin was "
                   "classified as needing hardware, but a pipeline change "
                   "suffices. The cost class was too pessimistic.")
    elif hard["acc_mean"] > pipe["acc_mean"] + 0.05:
        verdict = ("temperature is genuinely required: no cheaper "
                   "representation reaches it, so the HARDWARE cost class holds.")
    else:
        verdict = ("no representation separates them in this cell; the pair is "
                   "undecidable here for reasons other than the missing channel.")
    print("=> " + verdict)

    payload = {"note": "simulated cell; mechanism tests only, never pooled with "
                       "voraus-AD numbers",
               "experiment_a_collision": a, "experiment_b_thermal": b,
               "experiment_c_collision_instantiation_power": c,
               "cost_class_verdict": verdict}
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwritten -> {RESULTS_OUT}")


def main_c_only():
    """Experiment C alone -- A and B are recorded as not-answerable/invalid, so
    re-running them costs minutes and adds nothing."""
    print("=" * 74)
    print("EXPERIMENT C -- collision instantiation requirement (simulated cell)")
    print("=" * 74)
    c = experiment_c()
    out = os.path.join(_ROOT, "results", "sim_collision_power.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "simulated contacts; the DATA REQUIREMENT is the claim, "
                           "never the accuracy magnitudes, which are noise-floor "
                           "dependent and must not be quoted as a property of real "
                           "collisions",
                   **c}, f, indent=2, ensure_ascii=False)
    print("")
    print(f"written -> {out}")


if __name__ == "__main__":
    import sys
    if "--c" in sys.argv:
        main_c_only()
    else:
        main()
