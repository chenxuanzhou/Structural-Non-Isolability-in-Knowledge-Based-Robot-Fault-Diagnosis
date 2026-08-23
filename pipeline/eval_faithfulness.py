"""Citation-faithfulness evaluation of the explanation layer.

The reasoning layer (llm_reasoning.py) imposes hard rules on the explanation:
every causal claim must cite a signal id or a KG node id, and no root cause may
be proposed from outside the retrieved candidate set unless flagged as an
UNVERIFIED PRIOR. This script MEASURES compliance, per explanation engine
(rule-based fallback now; the real LLM once ANTHROPIC_API_KEY is set and
shared/generate_llm.py has been run), so the paper can report faithfulness numbers
instead of asserting the constraint.

Metrics per cycle, aggregated per engine:
  * citation resolution rate  — grounded_in resolves to a legal id
                                (signal / FM_* / Joint / Cause / Reco / phase)
  * evidence anchoring rate   — resolved citations point at THIS cycle's own
                                evidence: an attributed signal, a retrieved
                                candidate, its dominant joint/phase
  * candidate compliance rate — every ranked root cause is inside the top-5
                                candidate set given to the model (or is
                                explicitly flagged UNVERIFIED PRIOR)
  * claim coverage            — fraction of evidence items with a non-empty
                                citation

Run:  .\\.venv\\Scripts\\python.exe eval_faithfulness.py
Writes faithfulness_results.json next to this script.
"""
from __future__ import annotations
import json
import os
import re

from line_kg import FMEA, SIGNAL_FAMILIES, JOINTS
from attribution_bridge import ACTION_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHES = [os.path.join(ROOT, "shared", "cache", "dataset.json"),
          os.path.join(ROOT, "shared", "cache", "dataset_all.json")]
OUT = os.path.join(ROOT, "results", "faithfulness_results.json")

FM_IDS = {fm["id"] for fm in FMEA}
ROBOT_SIGNALS = {"robot_current", "system_current", "io_current", "robot_voltage"}
# any legal signal id in the cell (KG families x joints + robot-level)
LEGAL_SIGNALS = {f"{fam}_{j}" for fam in SIGNAL_FAMILIES for j in JOINTS} | ROBOT_SIGNALS
PHASES = set(ACTION_NAMES.values())


def classify_citation(gid: str):
    """Map a grounded_in string to (kind, canonical) or (None, None)."""
    g = (gid or "").strip()
    if not g:
        return None, None
    if g in LEGAL_SIGNALS:
        return "signal", g
    if g in FM_IDS:
        return "fm", g
    for prefix in ("Cause:", "Reco:", "Effect:", "FailureMode:"):
        if g.startswith(prefix) and g.split(":", 1)[1] in FM_IDS:
            return "kgnode", g
    m = re.fullmatch(r"Joint[:\s]*([1-6])", g)
    if m:
        return "joint", int(m.group(1))
    if g in PHASES:
        return "phase", g
    if g.startswith("Signal:") and g.split(":", 1)[1] in LEGAL_SIGNALS:
        return "signal", g.split(":", 1)[1]
    return None, None


def eval_cycle(cyc):
    """Score one cycle's explanation against its own record + candidate set."""
    llm = cyc.get("llm") or {}
    parsed = llm.get("parsed") or {}
    if not parsed:
        return None
    engine = llm.get("engine", "rule-based")
    if llm.get("model"):
        engine = f"{engine}:{llm['model']}"   # per-model rows for real LLM output

    own_signals = {a["signal"] for a in cyc.get("attributed_signals", [])}
    own_fams = {a["family"] for a in cyc.get("attributed_signals", [])}
    cand_ids = [c["id"] for c in cyc.get("candidates", [])[:5]]   # the set sent to the model
    dom_joint = (cyc.get("localization") or {}).get("dominant_joint")
    dom_phase = (cyc.get("localization") or {}).get("dominant_cycle_phase")

    n_claims = n_cited = n_resolved = n_anchored = 0
    n_causes = n_compliant = 0
    for rc in parsed.get("ranked_root_causes", []):
        n_causes += 1
        fm = rc.get("failure_mode_id", "")
        text = json.dumps(rc, ensure_ascii=False)
        if fm in cand_ids or "UNVERIFIED PRIOR" in text.upper():
            n_compliant += 1
        for ev in rc.get("evidence", []):
            n_claims += 1
            raw = ev.get("grounded_in")
            # LLMs sometimes emit a LIST of ids, a comma-joined string of ids,
            # or wrap ids in backticks — normalize representation, then judge.
            gids = raw if isinstance(raw, list) else [raw]
            gids = [p.strip().strip("`")
                    for g in gids if g
                    for p in re.split(r"[,;]", str(g))]
            gids = [g for g in gids if g]
            if not gids:
                continue
            n_cited += 1
            # a multi-id citation counts once; it resolves/anchors if EVERY id does
            results = [classify_citation(g) for g in gids]
            if any(k is None for k, _ in results):
                continue
            n_resolved += 1
            def _anchor(kind, canon, gid):
                return ((kind == "signal" and (canon in own_signals
                                               or canon.rsplit("_", 1)[0] in own_fams))
                        or (kind in ("fm", "kgnode") and any(cid in gid for cid in cand_ids))
                        or (kind == "joint" and canon == dom_joint)
                        or (kind == "phase" and canon == dom_phase))
            n_anchored += int(all(_anchor(k, c, g) for (k, c), g in zip(results, gids)))
    return {
        "engine": engine,
        "claims": n_claims, "cited": n_cited,
        "resolved": n_resolved, "anchored": n_anchored,
        "causes": n_causes, "compliant": n_compliant,
    }


def main():
    per_engine: dict[str, dict] = {}
    used = []
    seen: set[tuple] = set()          # dedupe cycles present in both caches
    for path in CACHES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        used.append(os.path.relpath(path, HERE))
        for cyc in data["cycles"]:
            r = eval_cycle(cyc)
            if r is None:
                continue
            key = (cyc["cycle_id"], r["engine"])
            if key in seen:
                continue
            seen.add(key)
            agg = per_engine.setdefault(r["engine"], {
                "cycles": 0, "claims": 0, "cited": 0, "resolved": 0,
                "anchored": 0, "causes": 0, "compliant": 0})
            agg["cycles"] += 1
            for k in ("claims", "cited", "resolved", "anchored", "causes", "compliant"):
                agg[k] += r[k]

    print(f"caches: {', '.join(used)}\n")
    report = {}
    for eng, a in per_engine.items():
        cov = a["cited"] / a["claims"] if a["claims"] else 0.0
        res = a["resolved"] / a["cited"] if a["cited"] else 0.0
        anc = a["anchored"] / a["resolved"] if a["resolved"] else 0.0
        cmp_ = a["compliant"] / a["causes"] if a["causes"] else 0.0
        report[eng] = {
            "cycles": a["cycles"], "claims": a["claims"],
            "claim_coverage": round(cov, 4),
            "citation_resolution": round(res, 4),
            "evidence_anchoring": round(anc, 4),
            "candidate_compliance": round(cmp_, 4),
        }
        print(f"=== engine: {eng} ===")
        print(f"  cycles                : {a['cycles']}")
        print(f"  causal claims         : {a['claims']}")
        print(f"  claim coverage        : {cov:.3f}   (claims that carry a citation)")
        print(f"  citation resolution   : {res:.3f}   (citations that resolve to a legal id)")
        print(f"  evidence anchoring    : {anc:.3f}   (resolved citations that point at THIS cycle's evidence)")
        print(f"  candidate compliance  : {cmp_:.3f}   (root causes inside the retrieved top-5, or flagged)")
        print()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"caches": used, "engines": report}, f, indent=2)
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
