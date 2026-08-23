"""
Line Knowledge Graph + seed FMEA  (Step 1 of 3)
===============================================
Project: LLM-Augmented Interactive Visual Analytics for Industrial Robot
         Anomaly Diagnosis  |  cell: voraus-AD Yu-Cobot pick-and-place

WHAT THIS IS
------------
A LINE-SCOPED knowledge graph for one cell, plus a seed FMEA, plus a
`retrieve()` that is SEEDED BY THE BRIDGE RECORD (attribution_bridge.py):
  bridge record {dominant_joint, attributed signal families, phase}
        -> enter the graph at the Joint node
        -> rank the failure modes that MANIFEST in those signals on that joint
        -> return causes + recommendations + provenance.

This is the GraphRAG entry point from the technical report (section F, step 3):
the detected/attributed signal is the retrieval seed, not a free-text query.

WHY THIS SEED IS DEFENSIBLE
---------------------------
The 12 voraus-AD anomaly types ARE documented failure modes for this exact
cell, grouped as axis wear / gripping errors / process errors. Encoding them as
FailureMode->Cause gives (a) a realistic cold-start KB and (b) root-cause
ground truth for evaluation (detected anomaly -> does retrieval surface the
correct cause?). Map each mode's `voraus_category` to the dataset's integer
label to turn this into a scored benchmark. Standards alignment: IEC 60812
(FMEA structure), ISO 13374 (advisory generation), ISO 14224 (failure taxonomy).

DEPENDENCIES: networkx (demo graph). Production: load via export_cypher() -> Neo4j.
"""
from __future__ import annotations
import json
import networkx as nx

# ----------------------------------------------------------------------------
# Schema (node labels and edge types) -- mirror this in Neo4j
# ----------------------------------------------------------------------------
# Nodes:  Robot, Joint, Component, Signal, FailureMode, Cause, Effect,
#         Recommendation, CyclePhase
# Edges:  HAS_JOINT, HAS_COMPONENT, EMITS(Component->Signal),
#         MANIFESTS_IN(FailureMode->Signal), AFFECTS(FailureMode->Joint),
#         CAUSED_BY(FailureMode->Cause), CAUSES(FailureMode->Effect),
#         MITIGATED_BY(FailureMode->Recommendation), OCCURS_IN(FailureMode->CyclePhase)

JOINTS = [1, 2, 3, 4, 5, 6]
LOAD_JOINTS = [1, 2, 3]      # proximal / load-bearing in pick-and-place
DISTAL_JOINTS = [4, 5, 6]    # wrist / end-effector region
SIGNAL_FAMILIES = ["motor_torque", "torque_sensor_a", "joint_velocity",
                   "joint_position", "robot_current", "system_current"]

# ----------------------------------------------------------------------------
# Seed FMEA -- grounded in the Yu-Cobot pick-and-place cell (voraus-AD types + generic drive faults)
# affects: explicit joint list | "any" | "load" | "distal"
# manifests: signal families where the deviation shows up (the diagnostic key)
# ----------------------------------------------------------------------------
FMEA = [
    dict(id="FM_friction", name="Additional axis friction", group="axis wear",
         affects="any", manifests=["motor_torque", "torque_sensor_a", "robot_current"],
         cause="Bearing/gearbox wear or loss of lubrication",
         effect="Elevated torque, heat, position-tracking error",
         recommendation="Inspect & lubricate joint bearing; check gearbox backlash",
         phase_hint="any", voraus_category=0),    # AXIS_FRICTION
    dict(id="FM_miscommutation", name="Motor miscommutation", group="axis wear",
         affects="any", manifests=["motor_torque", "robot_current"],
         cause="Encoder/commutation offset or Hall-sensor fault",
         effect="Torque ripple, jerky motion, current spikes",
         recommendation="Recalibrate commutation/encoder offset for the joint",
         phase_hint="any", voraus_category=10),   # MOTOR_COMMUTATION
    dict(id="FM_misgrip", name="Misgrip of can", group="gripping error",
         affects="distal", manifests=["torque_sensor_a", "joint_position"],
         cause="Vacuum seal weak or part mispositioned at pick",
         effect="Lower-than-expected lift load; failed place",
         recommendation="Check vacuum pressure & gripping position/teach point",
         phase_hint="grip", voraus_category=5),    # MISS_CAN
    dict(id="FM_losecan", name="Losing the can mid-transfer", group="gripping error",
         affects="load", manifests=["motor_torque", "torque_sensor_a"],
         cause="Vacuum loss / slip during motion",
         effect="Sudden torque drop on load joints late in cycle; dropped part",
         recommendation="Check vacuum line & seal; verify surface condition",
         phase_hint="transfer", voraus_category=6),  # LOSE_CAN
    dict(id="FM_axisweight", name="Additional axis weight", group="process error",
         affects="load", manifests=["motor_torque", "torque_sensor_a"],
         cause="Unexpected payload/attachment on a joint",
         effect="Sustained elevated torque; accelerated wear",
         recommendation="Verify payload configuration & tool weight settings",
         phase_hint="any", voraus_category=1),    # AXIS_WEIGHT
    dict(id="FM_collision_foam", name="Collision with foam", group="process error",
         affects="distal", manifests=["motor_torque", "joint_velocity"],
         cause="Soft obstacle (foam) in motion path",
         effect="Torque spike + velocity disruption at contact",
         recommendation="Clear workspace; verify path & safety zones",
         phase_hint="transfer", voraus_category=2),  # COLLISION_FOAM
    dict(id="FM_collision_cable", name="Collision with cables", group="process error",
         affects="any", manifests=["motor_torque", "joint_velocity"],
         cause="Loose cables in the motion path",
         effect="Torque spike on contact; possible snag",
         recommendation="Secure/route cables out of the work envelope",
         phase_hint="transfer", voraus_category=3),  # COLLISION_CABLE
    dict(id="FM_collision_card", name="Collision with cardboard", group="process error",
         affects="distal", manifests=["motor_torque", "joint_velocity"],
         cause="Cardboard obstacle in path",
         effect="Torque spike + trajectory deviation",
         recommendation="Clear workspace; inspect fixture layout",
         phase_hint="transfer", voraus_category=4),  # COLLISION_CARTON
    dict(id="FM_varweight", name="Varying can weight", group="process error",
         affects="load", manifests=["motor_torque", "torque_sensor_a"],
         cause="Inconsistent product weight between cycles",
         effect="Torque offset proportional to weight; grip/place variance",
         recommendation="Add weight check / adaptive grip-force control",
         phase_hint="grip", voraus_category=7),    # CAN_WEIGHT
    dict(id="FM_cable_routed", name="Cable routed at robot", group="process error",
         affects="any", manifests=["motor_torque", "torque_sensor_a", "joint_velocity"],
         cause="Cable bundle routed across moving links (snag/resistance)",
         effect="Elevated torque across motion; intermittent resistance",
         recommendation="Re-route dress-pack; add cable management",
         phase_hint="any", voraus_category=8),    # ENTANGLED
    dict(id="FM_invalidgrip", name="Invalid gripping position", group="process error",
         affects="distal", manifests=["joint_position", "torque_sensor_a"],
         cause="Part mislocated or incorrect teach point",
         effect="Position deviation + torque anomaly at grip",
         recommendation="Re-teach grip pose; check fixture/feeder alignment",
         phase_hint="grip", voraus_category=9),    # INVALID_POSITION
    dict(id="FM_platform", name="Unstable platform", group="process error",
         affects=[1, 2], manifests=["motor_torque", "joint_velocity", "torque_sensor_a"],
         cause="Loose base mounting / unstable platform",
         effect="Low-frequency oscillation; broad torque deviation on proximal joints",
         recommendation="Secure base mounting; check platform leveling/bolts",
         phase_hint="any", voraus_category=11),   # WOBBLING_STATION
    # generic drive faults (not in voraus-AD but expected on this hardware)
    dict(id="FM_backlash", name="Gearbox backlash / play", group="axis wear",
         affects="any", manifests=["joint_position", "torque_sensor_a"],
         cause="Worn gearbox / increased transmission play",
         effect="Position overshoot; torque chatter on reversal",
         recommendation="Measure backlash; schedule gearbox service",
         phase_hint="any", voraus_category=None),
    dict(id="FM_overtemp", name="Drive overtemperature (inferred)", group="axis wear",
         affects="any", manifests=["robot_current", "motor_torque"],
         cause="Sustained overload / cooling issue (NO temp sensor -> inferred)",
         effect="Current rise for same torque; thermal derating",
         recommendation="Infer from current/torque trend; add thermal monitoring",
         phase_hint="any", voraus_category=None),
]


# ----------------------------------------------------------------------------
# Physical signature of each failure mode (consumed by the schema-0.2 record)
# ----------------------------------------------------------------------------
# Until schema 0.2 the record carried only max|z| per signal, so two failure
# modes with the same (joints, families) were interchangeable no matter how
# differently they behave physically -- the structural twins. The FMEA always
# knew the difference; it was written in prose in the `effect` field and had
# nowhere to go. This table formalizes exactly that prose:
#
#   direction : expected sign of the SUSTAINED torque deviation
#               +1 above the envelope, -1 below, 0 = either (mode-dependent)
#   temporal  : how the deviation evolves within one cycle
#               "sustained" offset present throughout, no late step
#               "drop"      appears/worsens late, negative step  (payload lost)
#               "rise"      worsens late, positive step          (progressive)
#               "transient" brief spike, little sustained offset (contact)
#               None        the FMEA makes no claim
#
# AUTHORED FROM PHYSICS, NOT FROM LABELS: each entry is quoted from the mode's
# own `effect`/`cause` text above, which predates any of this analysis. It is
# knowledge the KB is supposed to hold, not a fit to the data.
#
# MEASURED ON THE FIXED SLOT SET AND NOT SHIPPED (2026-08-05; reproduce with
# `repair.py`, which prints
# both numbers). Scoring this table in retrieve() moves top-1 0.163 -> 0.148 and
# dissolves 3 of the 4 twin clusters, i.e. it makes the system quieter about its
# own undecidability while making it less accurate. The granularity ablation
# explains why: holding the statistic fixed and varying only its resolution
# gives 0.455 (mode-level scalar, at chance) / 0.526 (per family) / 0.720
# (per joint, load) / 0.860 (per joint, all six). The discriminative structure
# is a joint-resolved SIGNED PATTERN -- added axis weight raises
# torque_sensor_a on J2 (+0.37) while lowering it on J3 (-0.70) -- which prose
# cannot express and an engineer would not reliably author. So `w_signature`
# defaults to 0.0 and the joint-resolved version is learned per cluster in
# repair.py instead. The table stays because reproducing a negative result
# matters: `retrieve()` still reports `signature` and `signature_match` in each
# candidate's provenance, so the claim is inspectable rather than folklore.
# (The Cockpit does not read these yet -- it builds from shared/cache/dataset_all.json.)
FM_SIGNATURE = {
    # "Elevated torque, heat, position-tracking error"
    "FM_friction":        dict(direction=+1, temporal="sustained"),
    # "Torque ripple, jerky motion, current spikes"
    "FM_miscommutation":  dict(direction=0,  temporal="transient"),
    # "Lower-than-expected lift load; failed place"
    "FM_misgrip":         dict(direction=-1, temporal="sustained"),
    # "Sudden torque drop on load joints late in cycle; dropped part"
    "FM_losecan":         dict(direction=-1, temporal="drop"),
    # "Sustained elevated torque; accelerated wear"
    "FM_axisweight":      dict(direction=+1, temporal="sustained"),
    # "Torque spike + velocity disruption at contact"
    "FM_collision_foam":  dict(direction=0,  temporal="transient"),
    # "Torque spike on contact; possible snag"
    "FM_collision_cable": dict(direction=0,  temporal="transient"),
    # "Torque spike + trajectory deviation"
    "FM_collision_card":  dict(direction=0,  temporal="transient"),
    # "Torque offset proportional to weight" -- an offset either way, but a
    # steady one within a cycle (the weight does not change mid-cycle)
    "FM_varweight":       dict(direction=0,  temporal="sustained"),
    # "Elevated torque across motion; intermittent resistance"
    "FM_cable_routed":    dict(direction=+1, temporal="sustained"),
    # "Position deviation + torque anomaly at grip" -- no torque-sign claim
    "FM_invalidgrip":     dict(direction=0,  temporal=None),
    # "Low-frequency oscillation; broad torque deviation"
    "FM_platform":        dict(direction=0,  temporal="transient"),
    # "Position overshoot; torque chatter on reversal"
    "FM_backlash":        dict(direction=0,  temporal="transient"),
    # "Current rise for same torque; thermal derating"
    "FM_overtemp":        dict(direction=+1, temporal="sustained"),
}

# families whose direction/step the signature term reads (a drive fault shows in
# the load channels; velocity/position deviations are downstream consequences)
SIGNATURE_FAMILIES = {"motor_torque", "torque_sensor_a", "torque_sensor_b"}


def _expand(affects):
    if affects == "any":
        return JOINTS
    if affects == "load":
        return LOAD_JOINTS
    if affects == "distal":
        return DISTAL_JOINTS
    return list(affects)


# ----------------------------------------------------------------------------
# Knowledge-base write-back (kb_overrides.json, maintained by kb_feedback.py
# and the engineer console kb_console.py)
# ----------------------------------------------------------------------------
# The feedback loop writes its learned/approved updates HERE rather than editing
# this file:
#   {"priors":   {fm_id: float},                       # learned ranking priors
#    "modes":    {fm_id: {"manifests_add": [...],      # human-approved edits
#                          "manifests_remove": [...], "phase_hint": str}},
#    "new_modes": [{full FMEA entries added by engineers}]}
# reload_kb() applies the file IDEMPOTENTLY at import and at runtime (the apps
# hot-reload on file change): pristine FMEA is restored first, then overrides,
# then the IDF is recomputed IN PLACE so existing importers see the update.
# Absent file => retrieve() reproduces the published numbers exactly.
import json as _json
import os as _os

KB_OVERRIDES_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "kb_overrides.json")
KB_PRIORS: dict[str, float] = {}
_NEW_MODE_DEFAULTS = dict(group="engineer-added", effect="", phase_hint="any",
                          voraus_category=None)
_NEW_MODE_REQUIRED = ("id", "name", "affects", "manifests", "cause", "recommendation")


def _load_kb_overrides() -> dict:
    if not _os.path.exists(KB_OVERRIDES_PATH):
        return {}
    try:
        with open(KB_OVERRIDES_PATH, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Retrieval-ranking priors (derived from the FMEA, not hand-tuned per category)
# ----------------------------------------------------------------------------
import math as _math

ROBOT_FAMILIES = {"robot_current", "system_current"}

# Inverse-document-frequency of each signal family across the FMEA: a family
# that manifests in FEW failure modes (e.g. joint_position) is more diagnostic
# than one shared by many (e.g. motor_torque). This is what lets the ranker stop
# over-predicting the torque-on-load-joint cluster.
_DF: dict[str, int] = {}
for _fm in FMEA:
    for _fam in set(_fm["manifests"]):
        _DF[_fam] = _DF.get(_fam, 0) + 1
FAMILY_IDF = {f: _math.log(len(FMEA) / df) for f, df in _DF.items()}

# SIGNAL-level IDF over (family x affected-joint) manifest slots — finer than
# the family IDF: joint_velocity on a wrist joint appears in few failure modes
# (df 2-4) while motor_torque on a load joint appears in almost all (df 7-9),
# so evidence on rare slots is worth more. Validated held-out (2026-07-20).
SIGNAL_IDF: dict[tuple, float] = {}


def _rebuild_signal_idf():
    sdf: dict[tuple, int] = {}
    for fm in FMEA:
        aff = set(_expand(fm["affects"]))
        for fam in set(fm["manifests"]):
            for j in (aff if fam not in ROBOT_FAMILIES else {None}):
                sdf[(fam, j)] = sdf.get((fam, j), 0) + 1
    SIGNAL_IDF.clear()
    SIGNAL_IDF.update({k: _math.log(len(FMEA) / df) for k, df in sdf.items()})


_rebuild_signal_idf()

# pristine snapshot of the hand-authored KB (captured before any override)
_BASE_IDS = [fm["id"] for fm in FMEA]
_PRISTINE = {fm["id"]: {"manifests": list(fm["manifests"]),
                        "phase_hint": fm["phase_hint"]} for fm in FMEA}


def reload_kb(overrides: dict | None = None) -> dict:
    """(Re)apply kb_overrides.json — idempotent, mutates everything IN PLACE so
    modules that did `from line_kg import FMEA, FAMILY_IDF, KB_PRIORS` see the
    update. Pass an explicit `overrides` dict to trial an edit without touching
    the file (pass {} to restore the pristine KB). Returns the applied dict."""
    kb = _load_kb_overrides() if overrides is None else (overrides or {})

    # 1) restore pristine: drop engineer-added modes, reset edited fields
    FMEA[:] = [fm for fm in FMEA if fm["id"] in _PRISTINE]
    for fm in FMEA:
        p = _PRISTINE[fm["id"]]
        fm["manifests"] = list(p["manifests"])
        fm["phase_hint"] = p["phase_hint"]

    # 2) structural overrides on existing modes (human-approved)
    for fid, o in (kb.get("modes") or {}).items():
        fm = next((m for m in FMEA if m["id"] == fid), None)
        if not fm or not o:
            continue
        man = [m for m in fm["manifests"] if m not in set(o.get("manifests_remove", []))]
        man += [m for m in o.get("manifests_add", []) if m not in man]
        fm["manifests"] = man
        if o.get("phase_hint"):
            fm["phase_hint"] = o["phase_hint"]

    # 3) engineer-added failure modes (validated, id-unique)
    seen = {fm["id"] for fm in FMEA}
    for nm in kb.get("new_modes") or []:
        if not all(nm.get(k) for k in _NEW_MODE_REQUIRED) or nm["id"] in seen:
            continue
        entry = {**_NEW_MODE_DEFAULTS, **nm}
        entry["manifests"] = list(entry["manifests"])
        if isinstance(entry["affects"], list):
            entry["affects"] = [int(j) for j in entry["affects"]]
        FMEA.append(entry)
        seen.add(entry["id"])

    # 4) learned priors + IDF, updated in place
    KB_PRIORS.clear()
    KB_PRIORS.update({k: float(v) for k, v in (kb.get("priors") or {}).items()})
    _DF.clear()
    for fm in FMEA:
        for fam in set(fm["manifests"]):
            _DF[fam] = _DF.get(fam, 0) + 1
    FAMILY_IDF.clear()
    FAMILY_IDF.update({f: _math.log(len(FMEA) / df) for f, df in _DF.items()})
    _rebuild_signal_idf()
    return kb


reload_kb()          # apply any existing overrides at import

# voraus-AD Action-enum names -> coarse task-phase class, and which classes each
# FMEA phase_hint is consistent with (gripping errors show at the pick/place
# linear moves; collisions/slip show during the point-to-point transfers).
PHASE_OF_ACTION = {
    "DOWN_PICK_LIN": "pick", "UP_PICK_LIN": "pick",
    "DOWN_PLACE_LIN": "place", "UP_PLACE_LIN": "place",
    "TO_SCAN_PTP": "transfer", "OVER_OBJECT_PTP": "transfer", "TO_PLACE_PTP": "transfer",
}
HINT_PHASE_CLASSES = {
    "grip": {"pick", "place"}, "transfer": {"transfer"},
    "any": {"pick", "place", "transfer", None},
}


# ----------------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------------
def build_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("Robot:YuCobot", label="Robot", model="Yu-Cobot", task="pick-and-place can")
    for j in JOINTS:
        jn = f"Joint:{j}"
        g.add_node(jn, label="Joint", index=j)
        g.add_edge("Robot:YuCobot", jn, key="HAS_JOINT")
        for comp in ["motor", "gearbox", "torque_sensor"]:
            cn = f"Component:{comp}_{j}"
            g.add_node(cn, label="Component", ctype=comp, joint=j)
            g.add_edge(jn, cn, key="HAS_COMPONENT")
        for fam in ["motor_torque", "torque_sensor_a", "joint_velocity", "joint_position"]:
            sn = f"Signal:{fam}_{j}"
            g.add_node(sn, label="Signal", family=fam, joint=j)
            g.add_edge(jn, sn, key="EMITS")
    for fam in ["robot_current", "system_current"]:
        g.add_node(f"Signal:{fam}", label="Signal", family=fam, joint=None)

    for fm in FMEA:
        fn = f"FailureMode:{fm['id']}"
        g.add_node(fn, label="FailureMode", name=fm["name"], group=fm["group"],
                   voraus_category=fm["voraus_category"], phase_hint=fm["phase_hint"])
        g.add_node(f"Cause:{fm['id']}", label="Cause", text=fm["cause"])
        g.add_node(f"Effect:{fm['id']}", label="Effect", text=fm["effect"])
        g.add_node(f"Reco:{fm['id']}", label="Recommendation", text=fm["recommendation"])
        g.add_edge(fn, f"Cause:{fm['id']}", key="CAUSED_BY")
        g.add_edge(fn, f"Effect:{fm['id']}", key="CAUSES")
        g.add_edge(fn, f"Reco:{fm['id']}", key="MITIGATED_BY")
        for j in _expand(fm["affects"]):
            g.add_edge(fn, f"Joint:{j}", key="AFFECTS")
            for fam in fm["manifests"]:
                target = f"Signal:{fam}_{j}" if fam not in ("robot_current", "system_current") else f"Signal:{fam}"
                if g.has_node(target):
                    g.add_edge(fn, target, key="MANIFESTS_IN")
    return g


# ----------------------------------------------------------------------------
# Signature consistency (reads the schema-0.3 signature block)
# ----------------------------------------------------------------------------
def signature_match(fm: dict, record: dict, affected: set) -> float | None:
    """How well the record's torque DIRECTION and within-cycle STEP agree with
    what this failure mode physically does. Returns a value in [-1, 1], or None
    when the record predates schema 0.3, the mode makes no claim, or no torque
    slot falls on the mode's joints -- callers must treat None as "no
    contribution", never as disagreement.

    Reads the record's `signature` block, i.e. the FIXED slot set, not the
    per-cycle top-k selection in `attributed_signals`: these two statistics are
    only comparable across cycles when every cycle contributes the same slots.
    Evidence is weighted by each slot's peak |z|, so a strong deviation has more
    say about the direction than a marginal one. A mismatch scores NEGATIVE -- a
    mode that predicts a torque drop is argued against by a record showing a
    sustained rise. That was the point: no other term can express evidence
    AGAINST a candidate.
    """
    sg = FM_SIGNATURE.get(fm["id"])
    if not sg:
        return None
    block = record.get("signature")
    if not block:
        return None
    ix = {s: i for i, s in enumerate(block["stats"])}
    ev = []
    for slot, vals in block["slots"].items():
        tail = slot.rsplit("_", 1)
        fam, joint = (tail[0], int(tail[1])) if len(tail) == 2 and tail[1].isdigit() \
            else (slot, None)
        if fam in SIGNATURE_FAMILIES and joint in affected:
            ev.append(vals)
    if not ev:
        return None
    w = sum(abs(float(v[ix["z_peak"]])) for v in ev) or 1.0
    mean_signed = sum(float(v[ix["z_signed"]]) * abs(float(v[ix["z_peak"]])) for v in ev) / w
    mean_step = sum(float(v[ix["z_step"]]) * abs(float(v[ix["z_peak"]])) for v in ev) / w
    peak = sum(abs(float(v[ix["z_peak"]])) ** 2 for v in ev) / w

    parts = []
    d = sg.get("direction", 0)
    if d:
        parts.append(_math.tanh(mean_signed) * d)
    t = sg.get("temporal")
    if t:
        # normalize the step by the sustained offset so the comparison is about
        # SHAPE, not amplitude
        r = mean_step / (abs(mean_signed) + 0.5)
        if t == "drop":
            parts.append(-_math.tanh(r))
        elif t == "rise":
            parts.append(_math.tanh(r))
        elif t == "sustained":
            parts.append(1.0 - min(1.0, abs(_math.tanh(r))) * 2.0)
        elif t == "transient":
            # a spike leaves a large peak but little sustained offset
            parts.append(1.0 - 2.0 * min(1.0, abs(mean_signed) / (peak + 1e-6)))
    return (sum(parts) / len(parts)) if parts else None


def retrieve(record: dict, top_k: int = 5, w_joint: float = 1.0,
             w_signal: float = 1.0, w_evidence: float = 0.2,
             w_phase: float = 0.2, w_coverage: float = 0.15,
             z_cap: float = 6.0, w_prior: float = 1.0,
             use_signal_idf: bool = True, use_log_z: bool = True,
             w_signature: float = 0.0) -> list[dict]:
    """Rank failure modes by match to the bridge record's localization + signals.

    Score = w_joint * (dominant-joint match x specificity)      # coarse localization
          + w_signal * (family Jaccard overlap)                 # broad recall
          + w_evidence * (IDF- & z-weighted, joint-gated signal evidence)
          + w_phase * (cycle-phase consistency)
          + w_coverage * (fraction of the FM's manifest families actually hit)
          + w_prior * (analyst-feedback prior from kb_overrides.json; 0 if absent)
          + w_signature * (physical-signature consistency; schema 0.2 records only)

    The signature term (2026-08-05) is the first term that can score NEGATIVE:
    it compares the record's torque direction and within-cycle step against what
    the failure mode physically does (FM_SIGNATURE, authored from the FMEA's own
    effect text). It is what makes structural twins separable in the ranking
    itself rather than in a side model -- see repair.py. It contributes nothing
    for schema-0.1 records, for modes with no signature, and when no torque
    evidence lands on the mode's joints. Default OFF (w_signature=0.0).

    Evidence refinement (2026-07-20, held-out validated): the evidence term uses
    SIGNAL-level IDF over (family x joint) manifest slots and a log1p(z)
    transform instead of the saturating cap, plus the continuous coverage term.
    Rationale: 66% of cycles used to tie at rank 1 (broken arbitrarily by FMEA
    order) because family-level IDF x capped z collapses to few distinct values.
    Held-out (5 seeds, fixed config): top-1 0.157->0.162, top-5 0.641->0.659,
    MRR 0.320->0.325; stacked with the feedback prior: top-1 0.202, top-5 0.693,
    MRR 0.365. The legacy published ranking (top-1 0.158/755) is reproduced by
    w_evidence=0.3, w_phase=0.3, w_coverage=0, use_signal_idf=False,
    use_log_z=False. Setting w_evidence=w_phase=w_coverage=0 reproduces the
    original joint+Jaccard ranker; without kb_overrides.json the prior is zero.
    """
    dom_joint = record["localization"]["dominant_joint"]
    phase_cls = PHASE_OF_ACTION.get(record["localization"].get("dominant_cycle_phase"))
    sigs = record["attributed_signals"]
    rec_families = {s["family"] for s in sigs}

    results = []
    for fm in FMEA:
        affected = set(_expand(fm["affects"]))
        manifests = set(fm["manifests"])
        specificity = 1.0 if fm["affects"] != "any" else 0.5
        joint_match = specificity if dom_joint in affected else 0.0
        fam_overlap = rec_families & manifests
        fam_score = len(fam_overlap) / max(1, len(manifests | rec_families))

        # IDF- & z-weighted evidence over every attributed signal, gated by the
        # failure mode's affected joints (robot-level families apply globally).
        evidence, ev_signals, hit_fams = 0.0, [], set()
        for s in sigs:
            f = s["family"]
            if f in manifests and ((f in ROBOT_FAMILIES) or (s["joint"] in affected)):
                if use_signal_idf:
                    idf = SIGNAL_IDF.get((f, None if f in ROBOT_FAMILIES else s["joint"]), 0.0)
                else:
                    idf = FAMILY_IDF.get(f, 0.0)
                zt = _math.log1p(float(s["z"])) if use_log_z else min(float(s["z"]), z_cap)
                evidence += idf * zt
                ev_signals.append(s["signal"])
                hit_fams.add(f)
        evidence /= max(1, len(sigs))
        coverage = len(hit_fams) / max(1, len(manifests))
        phase_match = (phase_cls is not None
                       and phase_cls in HINT_PHASE_CLASSES.get(fm["phase_hint"], set()))

        prior = KB_PRIORS.get(fm["id"], 0.0)
        sig_match = signature_match(fm, record, affected) if w_signature else None
        score = (w_joint * joint_match + w_signal * fam_score
                 + w_evidence * evidence + w_phase * (1.0 if phase_match else 0.0)
                 + w_coverage * coverage
                 + w_prior * prior
                 + w_signature * (sig_match or 0.0))
        results.append({
            "failure_mode": fm["name"],
            "id": fm["id"],
            "score": round(score, 4),
            "cause": fm["cause"],
            "recommendation": fm["recommendation"],
            "provenance": {
                "joint_match": bool(joint_match),
                "matched_signal_families": sorted(fam_overlap),
                "evidence_signals": ev_signals,
                "phase_match": bool(phase_match),
                "affects": fm["affects"],
                "phase_hint": fm["phase_hint"],
                "feedback_prior": round(prior, 4),
                "signature": FM_SIGNATURE.get(fm["id"]),
                "signature_match": (None if sig_match is None else round(sig_match, 3)),
            },
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ----------------------------------------------------------------------------
# Neo4j Cypher export (for the production store)
# ----------------------------------------------------------------------------
def export_cypher(path: str = "line_kg.cypher") -> str:
    lines = []
    for fm in FMEA:
        fid = fm["id"]
        lines.append(
            f"MERGE (fm:FailureMode {{id:'{fid}'}}) "
            f"SET fm.name='{fm['name']}', fm.group='{fm['group']}';")
        lines.append(f"MERGE (c:Cause {{id:'{fid}'}}) SET c.text=\"{fm['cause']}\";")
        lines.append(f"MERGE (r:Recommendation {{id:'{fid}'}}) SET r.text=\"{fm['recommendation']}\";")
        lines.append(f"MATCH (fm:FailureMode {{id:'{fid}'}}),(c:Cause {{id:'{fid}'}}) MERGE (fm)-[:CAUSED_BY]->(c);")
        lines.append(f"MATCH (fm:FailureMode {{id:'{fid}'}}),(r:Recommendation {{id:'{fid}'}}) MERGE (fm)-[:MITIGATED_BY]->(r);")
        for j in _expand(fm["affects"]):
            lines.append(f"MERGE (j:Joint {{index:{j}}});")
            lines.append(f"MATCH (fm:FailureMode {{id:'{fid}'}}),(j:Joint {{index:{j}}}) MERGE (fm)-[:AFFECTS]->(j);")
            for fam in fm["manifests"]:
                sid = f"{fam}_{j}" if fam not in ("robot_current", "system_current") else fam
                lines.append(f"MERGE (s:Signal {{id:'{sid}'}}) SET s.family='{fam}';")
                lines.append(f"MATCH (fm:FailureMode {{id:'{fid}'}}),(s:Signal {{id:'{sid}'}}) MERGE (fm)-[:MANIFESTS_IN]->(s);")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ----------------------------------------------------------------------------
# Demo: feed the REAL category-5 bridge record into the KG
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import numpy as np, pandas as pd
    from attribution_bridge import (build_reference, attribute, to_bridge_record,
                                     JOINT_SIGNALS, ROBOT_SIGNALS, JOINTS as BJ, GRID_N)

    g = build_graph()
    print(f"KG built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"{len(FMEA)} failure modes")

    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "voraus_df.pkl"))
    cols = ([f"{f}_{j}" for f in JOINT_SIGNALS for j in BJ]
            + [c for c in ROBOT_SIGNALS if c in df.columns])
    grid = np.linspace(0, 1, GRID_N)
    ref = build_reference(df[df.category == 12], cols, grid)

    for cat in (5,):
        run = df[df.category == cat].sort_values("time")
        rec = to_bridge_record(attribute(run, ref, grid),
                               sample_id=run["sample"].iloc[0], category=cat)
        print("\nBRIDGE RECORD seed -> dominant_joint:",
              rec["localization"]["dominant_joint"],
              "| signal families:",
              sorted({s["family"] for s in rec["attributed_signals"]}))
        print("\nTop candidate causes retrieved from the line KG:")
        for i, r in enumerate(retrieve(rec), 1):
            print(f"  {i}. [{r['score']}] {r['failure_mode']}")
            print(f"       cause: {r['cause']}")
            print(f"       fix:   {r['recommendation']}")
            print(f"       why:   joint_match={r['provenance']['joint_match']}, "
                  f"signals={r['provenance']['matched_signal_families']}")

    export_cypher("line_kg.cypher")
    print("\nNeo4j Cypher written to line_kg.cypher")
