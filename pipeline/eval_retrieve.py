"""Offline sandbox for tuning line_kg.retrieve() ranking.

Builds the 755 anomaly bridge records ONCE (official train-normal envelope,
setting==72) and caches them, then scores any retrieval function on
top-1/3/5 + MRR, overall and per category. This lets us iterate on the
ranking WITHOUT re-running the full pipeline each time.

    python eval_retrieve.py            # build/refresh cache + score baseline
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

from attribution_bridge import (build_reference, attribute, to_bridge_record,
                                 JOINT_SIGNALS, ROBOT_SIGNALS, JOINTS, GRID_N)
from integration_test import DATA_PATH, TRAIN_SETTING, NORMAL_CATEGORY, CATEGORY_TO_FM

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "data", "records_cache.json")


def build_cache():
    from attribution_bridge import measured_read_families, add_residual_columns
    read_chan = [f"{f}_{j}" for f in measured_read_families() for j in JOINTS] + list(ROBOT_SIGNALS)
    chan = [f"{f}_{j}" for f in JOINT_SIGNALS for j in JOINTS] + list(ROBOT_SIGNALS)
    meta = ["time", "sample", "category", "action", "setting"]
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
    df = pd.read_parquet(DATA_PATH, columns=[c for c in read_chan + meta if c in avail])
    add_residual_columns(df)
    cols = [c for c in chan if c in df.columns]
    grid = np.linspace(0, 1, GRID_N)
    ref = build_reference(df[df["setting"] == TRAIN_SETTING], cols, grid)

    recs = []
    anom = df[df.category != NORMAL_CATEGORY]
    for sid, run in anom.sort_values("time").groupby("sample"):
        cat = int(run["category"].iloc[0])
        setting = int(run["setting"].iloc[0]) if "setting" in run.columns else None
        rec = to_bridge_record(attribute(run, ref, grid), sample_id=sid,
                               category=cat, setting=setting)
        recs.append(rec)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(recs, f)
    print(f"cached {len(recs)} records -> {CACHE}")
    return recs


def load_cache():
    if not os.path.exists(CACHE):
        return build_cache()
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


GRAD_CACHE = os.path.join(_ROOT, "data", "records_grad_cache.json")


def _fam_of(sig):
    return sig.rsplit("_", 1)[0] if sig[-1].isdigit() else sig


def build_grad_cache():
    """Like build_cache but also attaches the MVT-Flow per-signal GRADIENT
    attribution (detector cross-check) to each record under
    record['detector']['model_attribution_ranked']."""
    import integration_test as it
    d = it._load_detector()
    ckpt = d["ckpt"]
    chan = [f"{f}_{j}" for f in JOINT_SIGNALS for j in JOINTS] + list(ROBOT_SIGNALS)
    det_chan = ckpt["channels"]
    need = sorted(set(chan + det_chan + ["time", "sample", "category", "action", "setting"]))
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(DATA_PATH).schema_arrow.names)
    df = pd.read_parquet(DATA_PATH, columns=[c for c in need if c in avail])
    cols = [c for c in chan if c in df.columns]
    grid = np.linspace(0, 1, GRID_N)
    ref = build_reference(df[df["setting"] == TRAIN_SETTING], cols, grid)

    recs = []
    anom = df[df.category != NORMAL_CATEGORY]
    for sid, run in anom.sort_values("time").groupby("sample"):
        cat = int(run["category"].iloc[0])
        rec = to_bridge_record(
            attribute(run, ref, grid), sample_id=sid, category=cat,
            setting=int(run["setting"].iloc[0]) if "setting" in run.columns else None)
        X, _ = d["to_tensor"](run, det_chan, ckpt["cfg"]["seq_len"])
        fields = d["fields_fn"](d["model"], ckpt, d["norm"](X)[:1])
        ranked = [{"signal": r["signal"], "joint": r["joint"],
                   "family": _fam_of(r["signal"]), "attribution": r["attribution"]}
                  for r in fields["model_attribution_ranked"]]
        rec["detector"]["model_attribution_ranked"] = ranked
        recs.append(rec)
    with open(GRAD_CACHE, "w", encoding="utf-8") as f:
        json.dump(recs, f)
    print(f"cached {len(recs)} records w/ gradient -> {GRAD_CACHE}")
    return recs


def load_grad_cache():
    if not os.path.exists(GRAD_CACHE):
        return build_grad_cache()
    with open(GRAD_CACHE, encoding="utf-8") as f:
        return json.load(f)


def score(retrieve_fn, recs, top_k=5, name=""):
    rows = []
    for rec in recs:
        cat = rec["detector"]["ground_truth_category"]
        true_fm = CATEGORY_TO_FM.get(cat)
        cands = retrieve_fn(rec, top_k=top_k)
        ids = [c["id"] for c in cands]
        rank = (ids.index(true_fm) + 1) if true_fm in ids else 0
        rows.append({"category": cat, "hit1": int(rank == 1),
                     "hitk": int(rank >= 1), "rr": (1.0 / rank if rank else 0.0)})
    r = pd.DataFrame(rows)
    print(f"\n=== {name} ===")
    print(f"  top-1 {r['hit1'].mean():.3f} | top-{top_k} {r['hitk'].mean():.3f} | MRR {r['rr'].mean():.3f}")
    per = r.groupby("category").agg(n=("rr", "size"), top1=("hit1", "mean"), topk=("hitk", "mean"))
    print(per.to_string())
    return r["hit1"].mean(), r["hitk"].mean(), r["rr"].mean()


import math
from line_kg import FMEA, _expand

ROBOT = {"robot_current", "system_current"}
_DF: dict[str, int] = {}
for _fm in FMEA:
    for _f in set(_fm["manifests"]):
        _DF[_f] = _DF.get(_f, 0) + 1
IDF = {f: math.log(len(FMEA) / df) for f, df in _DF.items()}

# action-name -> coarse phase class; FM phase_hint -> which classes it matches
PHASE_CLASS = {
    "DOWN_PICK_LIN": "pick", "UP_PICK_LIN": "pick",
    "DOWN_PLACE_LIN": "place", "UP_PLACE_LIN": "place",
    "TO_SCAN_PTP": "transfer", "OVER_OBJECT_PTP": "transfer", "TO_PLACE_PTP": "transfer",
}
HINT_CLASSES = {"grip": {"pick", "place"}, "transfer": {"transfer"},
                "any": {"pick", "place", "transfer", None}}


def make_retrieve(w_joint=1.0, w_jac=1.0, w_ev=0.0, w_phase=0.0, zcap=6.0):
    """Hybrid: baseline broad recall (joint_match + family Jaccard) PLUS
    IDF-weighted, z-weighted, joint-gated signal evidence and a phase bonus."""
    def retrieve(rec, top_k=5):
        dom = rec["localization"]["dominant_joint"]
        phase = PHASE_CLASS.get(rec["localization"].get("dominant_cycle_phase"))
        sigs = rec["attributed_signals"]
        rec_fams = {s["family"] for s in sigs}
        out = []
        for fm in FMEA:
            affected = set(_expand(fm["affects"]))
            manifests = set(fm["manifests"])
            spec = 1.0 if fm["affects"] != "any" else 0.5
            joint_match = spec if dom in affected else 0.0
            jac = len(manifests & rec_fams) / max(1, len(manifests | rec_fams))
            ev = 0.0
            for s in sigs:
                f = s["family"]
                if f in manifests and ((f in ROBOT) or (s["joint"] in affected)):
                    ev += IDF.get(f, 0.0) * min(float(s["z"]), zcap)
            ev /= max(1, len(sigs))     # normalize per attributed signal
            ph = 1.0 if (phase is not None and phase in HINT_CLASSES.get(fm["phase_hint"], set())) else 0.0
            score = w_joint * joint_match + w_jac * jac + w_ev * ev + w_phase * ph
            out.append({"id": fm["id"], "score": round(score, 4), "failure_mode": fm["name"]})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]
    return retrieve


def make_retrieve_g(w_joint=1.0, w_jac=1.0, w_ev=0.3, w_phase=0.3, w_grad=0.0, zcap=6.0):
    """make_retrieve + a gradient cross-check term: IDF-weighted, joint-gated sum
    over the MVT-Flow per-signal gradient ranking (normalized by the record's max
    gradient). Rescues distal anomalies the envelope attribution under-weights."""
    base = make_retrieve(w_joint, w_jac, w_ev, w_phase, zcap)

    def retrieve(rec, top_k=5):
        ranked = rec.get("detector", {}).get("model_attribution_ranked", [])
        gmax = max((r["attribution"] for r in ranked), default=0.0) or 1.0
        affected_by = {fm["id"]: set(_expand(fm["affects"])) for fm in FMEA}
        man_by = {fm["id"]: set(fm["manifests"]) for fm in FMEA}
        base_scored = {c["id"]: c for c in base(rec, top_k=len(FMEA))}
        out = []
        for fm in FMEA:
            g = 0.0
            for r in ranked:
                f = r["family"]
                if f in man_by[fm["id"]] and ((f in ROBOT) or (r["joint"] in affected_by[fm["id"]])):
                    g += IDF.get(f, 0.0) * (r["attribution"] / gmax)
            c = dict(base_scored[fm["id"]])
            c["score"] = round(c["score"] + w_grad * g, 4)
            out.append(c)
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]
    return retrieve


# ============================================================================
# SIGNATURE ranking (v2): use the DORMANT information already in the record —
# deviation DIRECTION (sign of observed-expected), torque/velocity time
# COINCIDENCE (contact events), and load-joint SPREAD. The signatures are
# authored from the FMEA effect descriptions (physics), NOT from labels;
# weights are tuned with the same tune/test protocol as the published ranking.
# ============================================================================
TORQUE_FAMS = {"motor_torque", "torque_sensor_a", "torque_sensor_b"}

# per-FM signature: dir = expected torque deviation sign (+1 above envelope,
# -1 below, 0 = mixed/unknown); contact = torque+velocity coincide in time;
# spread = expected number of hot LOAD joints ('single' | 'multi' | None)
SIG = {
    "FM_friction":        dict(dir=+1, contact=False, spread="single"),
    "FM_axisweight":      dict(dir=+1, contact=False, spread="multi"),
    "FM_varweight":       dict(dir=0,  contact=False, spread="multi"),
    "FM_losecan":         dict(dir=-1, contact=False, spread=None),
    "FM_misgrip":         dict(dir=-1, contact=False, spread=None),
    "FM_invalidgrip":     dict(dir=0,  contact=False, spread=None),
    "FM_collision_foam":  dict(dir=+1, contact=True,  spread=None),
    "FM_collision_cable": dict(dir=+1, contact=True,  spread=None),
    "FM_collision_card":  dict(dir=+1, contact=True,  spread=None),
    "FM_cable_routed":    dict(dir=+1, contact=False, spread=None),
    "FM_miscommutation":  dict(dir=0,  contact=False, spread="single"),
    "FM_platform":        dict(dir=0,  contact=False, spread="multi"),
    "FM_backlash":        dict(dir=0,  contact=False, spread="single"),
    "FM_overtemp":        dict(dir=+1, contact=False, spread=None),
}


def _sig_features(rec, zcap=6.0):
    """Record-level signature features from EXISTING fields only."""
    sigs = rec["attributed_signals"]
    # signed torque evidence per attributed torque signal: (joint, sign, weight)
    tq = []
    for s in sigs:
        if s["family"] in TORQUE_FAMS:
            sign = 1.0 if (s["observed"] - s["expected"]) >= 0 else -1.0
            tq.append((s["joint"], sign, min(float(s["z"]), zcap) / zcap))
    # contact: a velocity signal and a torque signal deviating at ~the same
    # cycle progress (the "spike + coincident velocity disruption" fingerprint)
    contact = 0.0
    for v in sigs:
        if v["family"] != "joint_velocity":
            continue
        for t in sigs:
            if t["family"] in TORQUE_FAMS and abs(v["progress_pct"] - t["progress_pct"]) <= 8.0:
                contact = max(contact, min(float(v["z"]), zcap) / zcap)
    # load-joint spread from the z ranking (load families only, always present)
    zr = [j["peak_z"] for j in rec["localization"].get("joint_ranking_by_z", [])]
    n_hot = sum(1 for z in zr if zr and z >= 0.6 * max(zr)) if zr else 0
    return tq, contact, n_hot


def make_retrieve_sig(w_joint=1.0, w_jac=1.0, w_ev=0.3, w_phase=0.3,
                      w_dir=0.6, w_contact=0.6, w_spread=0.2, zcap=6.0):
    """make_retrieve + direction / contact / spread signature terms.
    All three new weights at 0 reproduce make_retrieve exactly."""
    base = make_retrieve(w_joint, w_jac, w_ev, w_phase, zcap)

    def retrieve(rec, top_k=5):
        tq, contact, n_hot = _sig_features(rec, zcap)
        affected_by = {fm["id"]: set(_expand(fm["affects"])) for fm in FMEA}
        base_scored = {c["id"]: c for c in base(rec, top_k=len(FMEA))}
        out = []
        for fm in FMEA:
            sg = SIG.get(fm["id"], {})
            add = 0.0
            d = sg.get("dir", 0)
            if d and tq:
                # z-weighted mean sign agreement over torque evidence on the
                # FM's affected joints; mismatches SUBTRACT (negative evidence)
                num = den = 0.0
                for joint, sign, w in tq:
                    if joint in affected_by[fm["id"]]:
                        num += sign * d * w
                        den += w
                if den > 0:
                    add += w_dir * (num / den)
            if sg.get("contact"):
                add += w_contact * (2.0 * contact - 1.0)   # absence is evidence AGAINST
            sp = sg.get("spread")
            if sp and n_hot:
                if sp == "single":
                    add += w_spread * (1.0 if n_hot == 1 else -min(1.0, (n_hot - 1) / 3.0))
                else:                                       # 'multi'
                    add += w_spread * (1.0 if n_hot >= 3 else (0.0 if n_hot == 2 else -1.0))
            c = dict(base_scored[fm["id"]])
            c["score"] = round(c["score"] + add, 4)
            out.append(c)
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]
    return retrieve


# ============================================================================
# FINE-GRAINED ranking (v3): attack the TIE problem. 65.8% of cycles have an
# exact top1-top2 score tie (broken arbitrarily by FMEA order), with the true
# cause inside the tied group on 163/755 cycles. Three refinements of the
# already-validated evidence term — no labels, no new features:
#   * SIGNAL-level IDF: df over (family x affected-joint) manifest slots
#     (joint_velocity@J5 df=4 vs motor_torque@J1 df=9) — finer, more diagnostic
#   * log z instead of the saturating min(z, cap)
#   * manifest COVERAGE: fraction of the FM's manifest families actually hit
#     on affected joints (continuous completeness term)
# ============================================================================
_SDF: dict[tuple, int] = {}
for _fm in FMEA:
    _aff = set(_expand(_fm["affects"]))
    for _f in set(_fm["manifests"]):
        for _j in (_aff if _f not in ROBOT else {None}):
            _SDF[(_f, _j)] = _SDF.get((_f, _j), 0) + 1
_NSLOT = len(FMEA)
SIG_IDF = {k: math.log(_NSLOT / df) for k, df in _SDF.items()}


def make_retrieve_fine(w_joint=1.0, w_jac=1.0, w_ev=0.3, w_phase=0.3,
                       w_cov=0.3, zcap=6.0, use_sig_idf=True, use_logz=True):
    def retrieve(rec, top_k=5):
        dom = rec["localization"]["dominant_joint"]
        phase = PHASE_CLASS.get(rec["localization"].get("dominant_cycle_phase"))
        sigs = rec["attributed_signals"]
        rec_fams = {s["family"] for s in sigs}
        out = []
        for fm in FMEA:
            affected = set(_expand(fm["affects"]))
            manifests = set(fm["manifests"])
            spec = 1.0 if fm["affects"] != "any" else 0.5
            joint_match = spec if dom in affected else 0.0
            jac = len(manifests & rec_fams) / max(1, len(manifests | rec_fams))
            ev, hit_fams = 0.0, set()
            for s in sigs:
                f = s["family"]
                if f in manifests and ((f in ROBOT) or (s["joint"] in affected)):
                    key = (f, None if f in ROBOT else s["joint"])
                    idf = SIG_IDF.get(key, 0.0) if use_sig_idf else IDF.get(f, 0.0)
                    zt = math.log1p(float(s["z"])) if use_logz else min(float(s["z"]), zcap)
                    ev += idf * zt
                    hit_fams.add(f)
            ev /= max(1, len(sigs))
            cov = len(hit_fams) / len(manifests)
            ph = 1.0 if (phase is not None and phase in HINT_CLASSES.get(fm["phase_hint"], set())) else 0.0
            sc = (w_joint * joint_match + w_jac * jac + w_ev * ev
                  + w_phase * ph + w_cov * cov)
            out.append({"id": fm["id"], "score": round(sc, 4), "failure_mode": fm["name"]})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]
    return retrieve


def audit_gradient_crosscheck(verbose=True):
    """Does the envelope-vs-gradient DISAGREEMENT ring mean anything?

    The interface draws a dashed ring on a joint that one attribution flags and
    the other does not, and requirement R3 calls this "cross-checking two
    independent attributions". We already knew the MVT-Flow gradient is not
    fault-discriminative (folding it into retrieve() dropped top-1 0.158 ->
    0.057). This asks the remaining question: even if it cannot rank causes, does
    disagreeing with the envelope predict that the top-1 is wrong?

    Measured over all 755 cycles from shared/cache/dataset_all.json (the per-joint
    envelopeFlag/modelFlag the UI actually draws): it does not. The ring fires on
    737/755 cycles, and the phi coefficient against top-1 correctness is
    approximately zero at every threshold. Reported so the interface and the
    paper stop describing it as a confidence signal.
    """
    import math
    cache = os.path.join(_ROOT, "shared", "cache", "dataset_all.json")
    if not os.path.exists(cache):
        print("shared/cache/dataset_all.json missing — run shared/data_prep.py --all")
        return None
    with open(cache, encoding="utf-8") as f:
        cycles = json.load(f)["cycles"]
    rows = []
    for c in cycles:
        jm, cands = c.get("joint_map") or [], c.get("candidates") or []
        if not jm or not cands:
            continue
        dis = sum(1 for j in jm if bool(j["envelopeFlag"]) != bool(j["modelFlag"]))
        rows.append((dis, cands[0]["id"] == c.get("true_fm")))
    out = {"n": len(rows), "thresholds": {}}
    for thr in (1, 2, 3):
        a = [r for r in rows if r[0] >= thr]
        b = [r for r in rows if r[0] < thr]
        if not a or not b:
            continue
        n11 = sum(1 for r in a if r[1]); n10 = len(a) - n11
        n01 = sum(1 for r in b if r[1]); n00 = len(b) - n01
        den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) or 1.0
        out["thresholds"][thr] = {
            "n_flagged": len(a), "fire_rate": len(a) / len(rows),
            "acc_when_flagged": n11 / len(a), "acc_when_not": n01 / len(b),
            "phi": (n11 * n00 - n10 * n01) / den}
    if verbose:
        print("\nGRADIENT CROSS-CHECK AUDIT — is the disagreement ring informative?")
        for thr, v in out["thresholds"].items():
            print(f"  disagreeing joints >= {thr}: fires on {v['fire_rate']:.1%} of cycles, "
                  f"top-1 {v['acc_when_flagged']:.3f} vs {v['acc_when_not']:.3f}, "
                  f"phi {v['phi']:+.3f}")
        print("  => no association. The ring is a comparison of the detector's")
        print("     internal sensitivity with the physical envelope, NOT evidence")
        print("     about whether the diagnosis is right. Label it that way.")
    return out


if __name__ == "__main__":
    recs = load_cache()
    from line_kg import retrieve as retrieve_production
    score(retrieve_production, recs, name="PRODUCTION retrieve() (refined, 2026-07-20)")
    score(lambda r, top_k=5: retrieve_production(r, top_k=top_k, w_evidence=0.3, w_phase=0.3,
                                                 w_coverage=0, use_signal_idf=False, use_log_z=False),
          recs, name="legacy published ranking (0.158/0.636/0.319)")
    score(make_retrieve(w_joint=1, w_jac=1, w_ev=0, w_phase=0), recs, name="joint+Jaccard baseline")
    score(make_retrieve_fine(use_sig_idf=True, use_logz=False, w_cov=0.15, w_ev=0.5, w_phase=0.2),
          recs, name="top1-oriented operating point (trades top-5)")
