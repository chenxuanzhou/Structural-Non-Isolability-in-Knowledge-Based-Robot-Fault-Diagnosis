"""
LLM reasoning layer  (Step 2 of 3)
==================================
Project: LLM-Augmented Interactive Visual Analytics for Industrial Robot
         Anomaly Diagnosis  |  cell: voraus-AD Yu-Cobot pick-and-place

WHAT THIS IS
------------
The reasoning layer that closes the pipeline:

  detector -> attribution_bridge (record) -> line_kg.retrieve (candidates)
           -> THIS: assemble prompt -> LLM -> cited root-cause explanation

Design follows the technical report (section F step 4 + evaluation):
  * structured evidence in, structured explanation out (parseable for the UI);
  * EVERY causal claim must cite its grounding (a signal id or a KB node id)
    -- this is the faithfulness / provenance guard against hallucination;
  * the model may NOT introduce a cause absent from the retrieved candidates
    unless it explicitly flags it as an UNVERIFIED PRIOR;
  * Anomaly-Detection Chain-of-Thought (LLMAD-style) before the verdict;
  * an explicit confidence + what additional evidence would change it
    (the over-trust mitigation the report calls for).

API: `call_llm()` is drop-in for the Anthropic Messages API. Set ANTHROPIC_API_KEY
and run where there is network. The __main__ demo assembles + prints the prompt
(no network needed) so you can inspect exactly what the model receives.
"""
from __future__ import annotations
import json
import os

# ----------------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a diagnostic reasoning assistant for ONE specific industrial robot cell:
a Yu-Cobot performing a pick-and-place task (vacuum gripper, 6 joints). An
upstream detector has flagged a cycle as anomalous and attributed the deviation
to specific joints/signals. A line-specific knowledge base has retrieved
candidate failure modes for that evidence.

Your job: infer the most likely ROOT CAUSE and recommend action.

Hard rules (these are safety-relevant; follow them exactly):
1. Reason ONLY from the provided attribution evidence and the retrieved
   candidate failure modes. Do not rely on outside assumptions about this cell.
2. Every causal claim MUST cite its grounding: a signal id (e.g.
   "torque_sensor_a_2") or a knowledge-base node id (e.g. "FM_losecan").
   Uncited causal claims are not allowed.
3. Do NOT propose a root cause that is not among the retrieved candidates,
   UNLESS you explicitly label it "UNVERIFIED PRIOR" and explain why the
   candidates seem insufficient.
4. First think step by step (anomaly-detection chain-of-thought): what
   deviated, where in the cycle, which candidates are consistent/inconsistent.
   Then give the verdict.
5. State a calibrated confidence (low/medium/high) and what additional evidence
   would raise or lower it. If the evidence is genuinely ambiguous, say so
   rather than over-committing.

Return BOTH:
- a short human-readable explanation, then
- a JSON block (between ```json fences) with this schema:
{
  "summary": str,
  "reasoning_steps": [str, ...],
  "ranked_root_causes": [
    {"failure_mode_id": str, "cause": str, "likelihood": "low|medium|high",
     "evidence": [{"claim": str, "grounded_in": str}]}
  ],
  "recommended_actions": [str, ...],
  "confidence": {"level": "low|medium|high", "rationale": str},
  "alternatives_considered": [str, ...],
  "additional_evidence_needed": [str, ...],
  "grounding_statement": str
}
"""

USER_TEMPLATE = """\
## Cell context
Robot: Yu-Cobot, 6 joints, vacuum gripper. Task: pick-and-place of cans.
Joints 1-3 are load-bearing (proximal); 4-6 are wrist/end-effector.
Note: this cell has NO temperature sensor; thermal causes can only be inferred
from current/torque trends.

## Detected anomaly (structured attribution from the detector + bridge)
{record_json}

## Candidate failure modes retrieved from the line knowledge base
(each was retrieved because the detected joint/signals match how this failure
mode manifests; "provenance" shows why)
{candidates_json}

## Task
Diagnose the most likely root cause of this anomalous cycle, following the
hard rules. Cite a signal id or KB node id for every causal claim.
"""


def build_prompt(record: dict, candidates: list[dict], top_k: int = 5) -> tuple[str, str]:
    """Assemble (system_prompt, user_prompt) from the bridge record + candidates."""
    # Trim the record to what the model needs (keep it token-lean, per report B3).
    lean = {
        "cycle_id": record["cycle_id"],
        "detector": record["detector"],
        "localization": {
            "dominant_joint": record["localization"]["dominant_joint"],
            "dominant_cycle_phase": record["localization"]["dominant_cycle_phase"],
            "joint_ranking_by_abs_delta": record["localization"]["joint_ranking_by_abs_delta"][:3],
        },
        "attributed_signals": record["attributed_signals"][:top_k],
    }
    user = USER_TEMPLATE.format(
        record_json=json.dumps(lean, indent=2),
        candidates_json=json.dumps(candidates[:top_k], indent=2),
    )
    return SYSTEM_PROMPT, user


# ----------------------------------------------------------------------------
# LLM backends
#   * Anthropic Messages API   (needs network + ANTHROPIC_API_KEY)
#   * any OpenAI-compatible LOCAL server, e.g. Ollama on localhost:11434
#     (call_llm_local, or set LLM_BASE_URL/LLM_MODEL and use call_llm_auto)
# ----------------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str,
             model: str = "claude-sonnet-4-5", max_tokens: int = 1500) -> str:
    """Call the Anthropic Messages API. Returns the model's text response."""
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def call_llm_local(system_prompt: str, user_prompt: str,
                   model: str = "qwen2.5:7b",
                   base_url: str = "http://127.0.0.1:11434/v1",
                   max_tokens: int = 1500, temperature: float = 0.2,
                   timeout: float = 600.0) -> str:
    """Call a local OpenAI-compatible chat endpoint (Ollama, LM Studio,
    llama.cpp server ...). Same prompts, same downstream parse_response()."""
    import httpx  # already a dependency of the anthropic SDK
    r = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # reasoning models (e.g. deepseek-r1 distills) prepend a <think> block
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def call_llm_auto(system_prompt: str, user_prompt: str, **kw) -> tuple[str, str]:
    """Pick a backend: LLM_BASE_URL (+ LLM_MODEL) -> local server, else
    ANTHROPIC_API_KEY -> Anthropic. Returns (text, model_tag)."""
    base = os.environ.get("LLM_BASE_URL")
    if base:
        model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
        return call_llm_local(system_prompt, user_prompt, model=model,
                              base_url=base, **kw), model
    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
    return call_llm(system_prompt, user_prompt, model=model, **kw), model


def parse_response(text: str) -> dict:
    """Extract the JSON block from the model response for the UI to consume."""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    return json.loads(text)


# ----------------------------------------------------------------------------
# Demo: assemble the prompt from the REAL category-5 pipeline output
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np, pandas as pd
    from attribution_bridge import (build_reference, attribute, to_bridge_record,
                                     JOINT_SIGNALS, ROBOT_SIGNALS, JOINTS as BJ, GRID_N)
    from line_kg import retrieve

    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "voraus_df.pkl"))
    cols = ([f"{f}_{j}" for f in JOINT_SIGNALS for j in BJ]
            + [c for c in ROBOT_SIGNALS if c in df.columns])
    grid = np.linspace(0, 1, GRID_N)
    ref = build_reference(df[df.category == 12], cols, grid)

    run = df[df.category == 5].sort_values("time")
    record = to_bridge_record(attribute(run, ref, grid),
                              sample_id=run["sample"].iloc[0], category=5)
    candidates = retrieve(record, top_k=5)

    system, user = build_prompt(record, candidates)
    print("=" * 72, "\nSYSTEM PROMPT\n", "=" * 72, sep="")
    print(system)
    print("=" * 72, "\nUSER PROMPT\n", "=" * 72, sep="")
    print(user)

    # When network + key are available, this runs the real model:
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 72, "\nMODEL RESPONSE\n", "=" * 72, sep="")
        print(call_llm(system, user))
