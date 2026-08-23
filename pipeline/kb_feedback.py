"""Close the knowledge-base write-back loop.

The interfaces log every analyst decision (accept / reject / redirect) to
shared/cache/feedback.jsonl and study/log.jsonl. This script CONSUMES those logs and
turns them into KB updates, three ways:

  --learn     mechanical: aggregate confirmations/rejections into per-failure-
              mode retrieval PRIORS and write them to kb_overrides.json.
              line_kg.retrieve() adds the prior term on next import; without the
              file the ranking reproduces the published numbers exactly.
  --suggest   AI-assisted: feed the aggregated correction patterns to the LLM
              backend (local Ollama or Anthropic, same as the explanation layer)
              and draft STRUCTURAL edit proposals (manifests / phase hints) as
              kb_suggestions.md. Never auto-applied: a human reviews and merges
              the JSON patch into kb_overrides.json "modes".
  --simulate  closed-loop validation without real analysts: split the 755
              anomalies 50/50 (stratified, 5 seeds), simulate a correct analyst
              on the tune half (accept right top-1s, redirect wrong ones),
              learn priors from THOSE decisions only, and measure retrieval on
              the held-out half with vs without the learned priors.

Prior definition (deliberately simple and bounded):
  prior(fm) = G_RATE * clip(ln(p_fm / u), +-CLIP)        # confirmed base rate
            - G_FDR  * (2*fdr_fm - 1)                    # top-1 false-alarm rate
  p_fm  = Laplace-smoothed share of analyst-confirmed diagnoses that were fm,
  u     = uniform 1/N_modes,
  fdr_fm= smoothed fraction of fm's top-1 appearances the analyst overturned.
Magnitudes are capped well below the joint/evidence terms so feedback biases
ties rather than overruling evidence.

Run:  .\\.venv\\Scripts\\python.exe kb_feedback.py --simulate
      .\\.venv\\Scripts\\python.exe kb_feedback.py --learn
      .\\.venv\\Scripts\\python.exe kb_feedback.py --suggest [--local] [--demo]
"""
from __future__ import annotations
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEEDBACK_FILES = [os.path.join(ROOT, "shared", "cache", "feedback.jsonl"),
                  os.path.join(ROOT, "study", "log.jsonl")]
OVERRIDES = os.path.join(HERE, "kb_overrides.json")   # KB state, lives with the KB
SUGGESTIONS = os.path.join(ROOT, "results", "kb_suggestions.md")   # an output

G_RATE, G_FDR, CLIP, LAPLACE = 0.25, 0.25, 1.5, 2.0
MIN_ROWS = 10


# ---------------------------------------------------------------- feedback IO
def normalize_row(r: dict):
    """-> (confirmed_fm|None, top1_fm|None, top1_was_wrong|None)."""
    if "analyst_verdict" in r:                       # va app / console rows
        v, fm, top1 = r["analyst_verdict"], r.get("analyst_fm"), r.get("model_top1")
        if v == "accept":
            return fm, top1, False
        if v == "redirect":
            return fm, top1, (top1 != fm) if top1 else True
        if v in ("case", "revise"):                  # engineer case label / revision
            return fm, top1, ((top1 != fm) if top1 else None)
        if v == "reject":
            return None, (fm if fm == top1 else top1), (fm == top1) or None
    elif "decision" in r:                            # study log.jsonl
        v, fm, top1 = r["decision"], r.get("chosen_fm"), r.get("ai_top1")
        if v == "accept":
            return top1, top1, False
        if v == "reject":
            return fm, top1, True
    return None, None, None


def load_feedback(paths=FEEDBACK_FILES):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def latest_per_cycle(rows):
    """The engineer's FINAL decision per cycle: later rows (by ts, then file
    order) supersede earlier ones — this is what makes revisions effective."""
    by_cycle: dict = {}
    for i, r in enumerate(rows):
        cid = r.get("cycle_id")
        if cid is None:
            continue
        key = (r.get("ts") or "", i)
        if cid not in by_cycle or key >= by_cycle[cid][0]:
            by_cycle[cid] = (key, r)
    return [v[1] for v in by_cycle.values()]


# ---------------------------------------------------------------- learning
def learn_priors(norm_rows, fm_ids):
    """norm_rows: list of (confirmed_fm, top1_fm, top1_was_wrong)."""
    n = len(fm_ids)
    confirms = {f: 0.0 for f in fm_ids}
    top1_right = {f: 0.0 for f in fm_ids}
    top1_wrong = {f: 0.0 for f in fm_ids}
    for conf, top1, wrong in norm_rows:
        if conf in confirms:
            confirms[conf] += 1
        if top1 in confirms and wrong is not None:
            (top1_wrong if wrong else top1_right)[top1] += 1
    total = sum(confirms.values())
    u = 1.0 / n
    priors = {}
    for f in fm_ids:
        p = (confirms[f] + LAPLACE) / (total + LAPLACE * n)
        rate = max(-CLIP, min(CLIP, math.log(p / u)))
        seen = top1_right[f] + top1_wrong[f]
        fdr = (top1_wrong[f] + 1.0) / (seen + 2.0)   # smoothed; 0.5 when unseen
        priors[f] = round(G_RATE * rate - G_FDR * (2.0 * fdr - 1.0), 4)
    return priors, confirms, top1_wrong, top1_right


def learn_and_write(min_rows: int = MIN_ROWS):
    """Aggregate the CURRENT final decisions (latest per cycle) into priors,
    write kb_overrides.json (preserving modes/new_modes), and hot-reload the
    in-process KB. Returns (n_rows_used, priors) or (n, None) if too few.
    Reused by the CLI and by the engineer console."""
    import line_kg
    fm_ids = [fm["id"] for fm in line_kg.FMEA]
    final = latest_per_cycle(load_feedback())
    rows = [normalize_row(r) for r in final]
    rows = [r for r in rows if r != (None, None, None)]
    if len(rows) < min_rows:
        return len(rows), None
    priors, confirms, wrong, right = learn_priors(rows, fm_ids)
    data = {}
    if os.path.exists(OVERRIDES):
        with open(OVERRIDES, encoding="utf-8") as f:
            data = json.load(f)
    data["priors"] = priors
    data.setdefault("modes", {})
    data.setdefault("new_modes", [])
    data["_meta"] = {"rows_used": len(rows), "source": "kb_feedback.learn_and_write"}
    with open(OVERRIDES, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    line_kg.reload_kb()
    return len(rows), (priors, confirms, wrong, right)


def cmd_learn():
    n, res = learn_and_write()
    if res is None:
        print(f"only {n} usable final decisions (<{MIN_ROWS}) — priors from so "
              f"little feedback are noise; collect more decisions first.")
        return
    priors, confirms, wrong, right = res
    print(f"learned priors from {n} final decisions -> {OVERRIDES} (KB reloaded)")
    for f_, v in sorted(priors.items(), key=lambda kv: -kv[1]):
        print(f"  {f_:18s} {v:+.3f}   (confirmed {int(confirms[f_])}, "
              f"top-1 overturned {int(wrong[f_])}/{int(wrong[f_] + right[f_])})")


def quick_eval(w_prior: float = 1.0):
    """Score the CURRENT in-memory KB over the cached 755 records.
    w_prior=0 gives the no-feedback baseline for comparison."""
    from eval_retrieve import load_cache
    from line_kg import retrieve
    from integration_test import CATEGORY_TO_FM
    h1 = hk = rr = 0.0
    recs = load_cache()
    for rec in recs:
        tf = CATEGORY_TO_FM.get(rec["detector"]["ground_truth_category"])
        ids = [c["id"] for c in retrieve(rec, top_k=5, w_prior=w_prior)]
        rank = (ids.index(tf) + 1) if tf in ids else 0
        h1 += rank == 1
        hk += rank >= 1
        rr += (1.0 / rank) if rank else 0.0
    n = max(1, len(recs))
    return round(h1 / n, 3), round(hk / n, 3), round(rr / n, 3)


# ---------------------------------------------------------------- simulation
def cmd_simulate(seeds=(0, 1, 2, 3, 4)):
    import line_kg
    from eval_retrieve import load_cache
    from line_kg import FMEA, retrieve
    from integration_test import CATEGORY_TO_FM
    fm_ids = [fm["id"] for fm in FMEA]
    recs = load_cache()
    print(f"loaded {len(recs)} anomaly records; simulating the analyst loop "
          f"on a 50/50 stratified split, {len(seeds)} seeds\n")

    def score(half):
        h1 = hk = rr = 0.0
        for rec in half:
            tf = CATEGORY_TO_FM.get(rec["detector"]["ground_truth_category"])
            ids = [c["id"] for c in retrieve(rec, top_k=5)]
            rank = (ids.index(tf) + 1) if tf in ids else 0
            h1 += rank == 1
            hk += rank >= 1
            rr += (1.0 / rank) if rank else 0.0
        n = max(1, len(half))
        return h1 / n, hk / n, rr / n

    agg = {"base": [], "prior": []}
    for seed in seeds:
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

        line_kg.KB_PRIORS.clear()                    # analyst works WITHOUT priors
        norm = []
        for rec in tune:
            tf = CATEGORY_TO_FM.get(rec["detector"]["ground_truth_category"])
            top1 = retrieve(rec, top_k=1)[0]["id"]
            norm.append((tf, top1, top1 != tf))      # accept if right, else redirect
        base = score(test)
        priors, *_ = learn_priors(norm, fm_ids)
        line_kg.KB_PRIORS.update(priors)             # KB updated from feedback
        withp = score(test)
        line_kg.reload_kb()                          # restore on-disk KB state
        agg["base"].append(base)
        agg["prior"].append(withp)
        print(f"seed {seed}: held-out  top-1 {base[0]:.3f} -> {withp[0]:.3f}   "
              f"top-5 {base[1]:.3f} -> {withp[1]:.3f}   MRR {base[2]:.3f} -> {withp[2]:.3f}")

    def mean(key, i):
        return sum(v[i] for v in agg[key]) / len(agg[key])

    print("\n" + "=" * 66)
    print("CLOSED LOOP (mean over seeds; priors learned ONLY from the tune half)")
    print("=" * 66)
    print(f"  without write-back : top-1 {mean('base', 0):.3f}  top-5 {mean('base', 1):.3f}  MRR {mean('base', 2):.3f}")
    print(f"  with learned priors: top-1 {mean('prior', 0):.3f}  top-5 {mean('prior', 1):.3f}  MRR {mean('prior', 2):.3f}")


# ---------------------------------------------------------------- suggestions
SUGGEST_PROMPT = """\
You maintain the FMEA knowledge base of a robot-diagnosis system (Yu-Cobot
pick-and-place cell, 6 joints, vacuum gripper, no temperature sensor).
Analysts have been correcting the system's diagnoses; the aggregated
correction patterns are below.

Propose at most 3 SMALL structural edits to the knowledge base that would
reduce the observed confusions. Allowed edit types per failure mode:
  * manifests_add / manifests_remove — signal families from:
    motor_torque, torque_sensor_a, joint_velocity, joint_position,
    robot_current, system_current
  * phase_hint — one of: any, grip, transfer
Rules: every proposal must cite the correction pattern that motivates it
(e.g. "FM_losecan top-1 overturned 41/60 times"). Do not invent failure modes.
Be conservative: fewer, better-justified edits beat many speculative ones.

Return a short rationale, then a ```json block:
{"proposals": [{"fm_id": str, "edit": {"manifests_add": [...] | "manifests_remove": [...] | "phase_hint": str},
               "motivated_by": str, "expected_effect": str}]}

## Failure modes (current KB)
{kb_json}

## Aggregated analyst-correction patterns
{stats_json}
"""


def draft_suggestions(demo=False):
    """-> (raw_text, model, parsed|None, src) or (None, None, None, reason)."""
    from line_kg import FMEA
    from llm_reasoning import call_llm_auto, parse_response
    fm_ids = [fm["id"] for fm in FMEA]

    if demo:                                         # ground-truth-simulated demo rows
        import line_kg
        from eval_retrieve import load_cache
        from line_kg import retrieve
        from integration_test import CATEGORY_TO_FM
        line_kg.KB_PRIORS.clear()
        norm = []
        for rec in load_cache():
            tf = CATEGORY_TO_FM.get(rec["detector"]["ground_truth_category"])
            top1 = retrieve(rec, top_k=1)[0]["id"]
            norm.append((tf, top1, top1 != tf))
        line_kg.reload_kb()
        src = "SIMULATED (ground-truth analyst over all 755 anomalies) — demo"
    else:
        norm = [normalize_row(r) for r in latest_per_cycle(load_feedback())]
        norm = [r for r in norm if r != (None, None, None)]
        src = f"{len(norm)} real final decisions"
        if len(norm) < MIN_ROWS:
            return None, None, None, f"only {len(norm)} usable decisions (<{MIN_ROWS})"

    _, confirms, wrong, right = learn_priors(norm, fm_ids)
    stats = {f: {"confirmed": int(confirms[f]),
                 "top1_overturned": f"{int(wrong[f])}/{int(wrong[f] + right[f])}"}
             for f in fm_ids if confirms[f] or wrong[f] or right[f]}
    kb = [{"id": fm["id"], "name": fm["name"], "affects": fm["affects"],
           "manifests": fm["manifests"], "phase_hint": fm["phase_hint"]}
          for fm in FMEA]
    prompt = SUGGEST_PROMPT.replace("{kb_json}", json.dumps(kb, indent=1)) \
                           .replace("{stats_json}", json.dumps(stats, indent=1))
    raw, model = call_llm_auto("You are a careful FMEA knowledge-base maintainer.",
                               prompt)
    try:
        parsed = parse_response(raw)
    except Exception:
        parsed = None
    return raw, model, parsed, src


def cmd_suggest(demo=False):
    print("drafting KB edit proposals ...")
    raw, model, parsed, src = draft_suggestions(demo=demo)
    if raw is None:
        print(src + "; run with --demo to draft from a simulated analyst instead.")
        return
    with open(SUGGESTIONS, "w", encoding="utf-8") as f:
        f.write(f"# KB edit proposals (AI-drafted — HUMAN REVIEW REQUIRED)\n\n"
                f"Source: {src} · model: {model}\n\n"
                f"To apply an approved edit, merge its `edit` object into\n"
                f"`kb_overrides.json` under `modes.<fm_id>` (or use the engineer\n"
                f"console kb_console.py, which evaluates before applying).\n\n"
                f"---\n\n{raw}\n")
    print(f"wrote {SUGGESTIONS}  (source: {src})")
    if parsed and parsed.get("proposals"):
        for p in parsed["proposals"]:
            print(f"  * {p.get('fm_id')}: {json.dumps(p.get('edit'), ensure_ascii=False)}"
                  f"  — {p.get('motivated_by', '')[:70]}")


if __name__ == "__main__":
    if "--local" in sys.argv:
        os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")
    if "--learn" in sys.argv:
        cmd_learn()
    elif "--suggest" in sys.argv:
        cmd_suggest(demo="--demo" in sys.argv)
    else:
        cmd_simulate()
