# -*- coding: utf-8 -*-
"""Engineer console — continuous knowledge-base maintenance (Dash, :8080).

The operational surface of the write-back loop for REAL deployments. Engineers:

  1. 决策台账   review every logged diagnosis decision; the LATEST decision per
                cycle is the effective one — submitting a REVISION supersedes
                the old decision (history is append-only, never rewritten);
  2. 添加案例   label a cycle with its true cause, optionally defining a brand
                NEW failure mode (added to the KB via kb_overrides.json);
  3. 学习与生效 aggregate the current final decisions into retrieval priors and
                hot-apply them (kb_feedback.learn_and_write + line_kg.reload_kb);
                evaluate the effect over the cached 755 anomalies;
  4. AI 建议    have the local LLM draft structural KB edits from the correction
                patterns, EVALUATE each proposal empirically, then adopt or
                dismiss — nothing is ever auto-applied.

The diagnosis app (va/app.py) hot-reloads kb_overrides.json on change, so
adopted updates are live there within one interaction.

Run:  .\\.venv\\Scripts\\python.exe kb_console.py    ->  http://127.0.0.1:8080
"""
from __future__ import annotations
import json
import os
import sys
import time

from dash import (Dash, html, dcc, Input, Output, State, callback, no_update,
                  ctx, ALL, dash_table)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))

# palette only — core also loads the exemplar cache (small)
from core import C, FONT_SANS, FONT_MONO                     # noqa: E402
import line_kg                                                # noqa: E402
from line_kg import FMEA, KB_PRIORS, reload_kb, KB_OVERRIDES_PATH, SIGNAL_FAMILIES  # noqa: E402
import kb_feedback as kbf                                     # noqa: E402

FEEDBACK_MAIN = os.path.join(ROOT, "shared", "cache", "feedback.jsonl")

# default to the local LLM backend for suggestion drafting
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")

_RECORDS: dict | None = None      # lazy cycle_id -> bridge record


def records():
    global _RECORDS
    if _RECORDS is None:
        from eval_retrieve import load_cache
        _RECORDS = {r["cycle_id"]: r for r in load_cache()}
    return _RECORDS


def fm_options():
    return [{"label": f"{fm['name']}  ({fm['id']})", "value": fm["id"]} for fm in FMEA]


def append_row(row: dict):
    os.makedirs(os.path.dirname(FEEDBACK_MAIN), exist_ok=True)
    with open(FEEDBACK_MAIN, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_overrides() -> dict:
    if os.path.exists(KB_OVERRIDES_PATH):
        with open(KB_OVERRIDES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_overrides(data: dict):
    data.setdefault("priors", {})
    data.setdefault("modes", {})
    data.setdefault("new_modes", [])
    with open(KB_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    reload_kb()


def ledger_rows():
    all_rows = kbf.load_feedback()
    final = {id(r): True for r in kbf.latest_per_cycle(all_rows)}
    out = []
    for r in all_rows:
        fm = r.get("analyst_fm") or r.get("chosen_fm")
        out.append({"时间": r.get("ts", ""), "周期": r.get("cycle_id"),
                    "判定": r.get("analyst_verdict") or r.get("decision", ""),
                    "最终原因": fm or "—", "备注": r.get("note", ""),
                    "生效": "●" if id(r) in final else ""})
    return list(reversed(out))


def status_chips():
    n_added = sum(1 for fm in FMEA if fm.get("group") == "engineer-added")
    finals = kbf.latest_per_cycle(kbf.load_feedback())
    mtime = (time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(KB_OVERRIDES_PATH)))
             if os.path.exists(KB_OVERRIDES_PATH) else "无")
    chips = [f"失效模式 {len(FMEA)}（工程师新增 {n_added}）",
             f"学习先验 {sum(1 for v in KB_PRIORS.values() if abs(v) > 1e-9)} 项",
             f"最终决策 {len(finals)} 条", f"覆盖文件 {mtime}"]
    return [html.Span(c, style={"fontFamily": FONT_MONO, "fontSize": "11px",
                                "background": C["panel"], "border": f"1px solid {C['line']}",
                                "borderRadius": "6px", "padding": "5px 10px"}) for c in chips]


_panel = {"background": C["panel"], "borderRadius": "10px", "padding": "14px",
          "border": f"1px solid {C['line']}", "marginBottom": "14px"}
_h = {"fontSize": "13px", "fontWeight": 700, "margin": "0 0 10px 0",
      "paddingBottom": "6px", "borderBottom": f"1px solid {C['line']}"}
_btn = {"border": "none", "padding": "8px 14px", "borderRadius": "6px", "fontWeight": 700,
        "cursor": "pointer", "fontSize": "12px"}
_in = {"padding": "6px 8px", "fontSize": "12px", "border": f"1px solid {C['line']}",
       "borderRadius": "6px", "marginRight": "8px"}

app = Dash(__name__, title="KB 运维台", suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id="proposals", data=None),
    html.Div([
        html.Div([html.H2("知识库运维台", style={"margin": 0, "fontSize": "19px", "fontWeight": 800}),
                  html.Div("案例与决策 → 学习先验 → AI 建议（人审）→ 诊断界面热生效",
                           style={"color": C["muted"], "fontSize": "11px", "marginTop": "2px"})]),
        html.Div(id="chips", children=status_chips(),
                 style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "14px 22px", "borderBottom": f"1px solid {C['line']}"}),

    html.Div([
      html.Div([
        # ---- ledger + revision ----
        html.Div([
            html.Div("决策台账 — 每周期以最新决策生效（●）；修订会追加新行、不改历史", style=_h),
            dash_table.DataTable(
                id="ledger",
                columns=[{"name": c, "id": c} for c in ["时间", "周期", "判定", "最终原因", "备注", "生效"]],
                data=ledger_rows(), page_size=8,
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": FONT_MONO, "fontSize": "11px", "padding": "4px 8px",
                            "textAlign": "left", "maxWidth": "260px", "whiteSpace": "normal"},
                style_header={"fontWeight": 700, "background": C["panel2"]},
            ),
            html.Div([
                html.Span("修订：周期", style={"fontSize": "12px"}),
                dcc.Input(id="rev-cycle", type="number", placeholder="cycle id",
                          style={**_in, "width": "110px"}),
                dcc.Dropdown(id="rev-fm", options=fm_options(), placeholder="改判为…",
                             style={"width": "300px", "fontSize": "12px", "display": "inline-block",
                                    "verticalAlign": "middle", "marginRight": "8px"}),
                dcc.Input(id="rev-note", type="text", placeholder="备注(可选)",
                          style={**_in, "width": "200px"}),
                html.Button("提交修订", id="btn-revise", n_clicks=0,
                            style={**_btn, "background": C["accent"], "color": "#fff"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                      "flexWrap": "wrap", "marginTop": "10px"}),
            html.Div(id="rev-status", style={"fontSize": "11px", "color": C["good"], "minHeight": "15px"}),
        ], style=_panel),

        # ---- add case ----
        html.Div([
            html.Div("添加案例 — 为某周期登记真实原因；可同时定义新的失效模式", style=_h),
            html.Div([
                html.Span("周期", style={"fontSize": "12px"}),
                dcc.Input(id="case-cycle", type="number", placeholder="cycle id",
                          style={**_in, "width": "110px"}),
                dcc.Dropdown(id="case-fm", options=fm_options(), placeholder="真实原因…",
                             style={"width": "300px", "fontSize": "12px", "display": "inline-block",
                                    "verticalAlign": "middle", "marginRight": "8px"}),
                dcc.Input(id="case-note", type="text", placeholder="备注(可选)",
                          style={**_in, "width": "200px"}),
                dcc.Checklist(id="case-newmode", value=[],
                              options=[{"label": " 定义新失效模式", "value": "on"}],
                              style={"fontSize": "12px", "display": "inline-block"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px", "flexWrap": "wrap"}),
            html.Div(id="newmode-form", children=[
                html.Div([
                    dcc.Input(id="nm-id", type="text", placeholder="id（如 belt_wear）",
                              style={**_in, "width": "160px"}),
                    dcc.Input(id="nm-name", type="text", placeholder="名称（如 Belt wear）",
                              style={**_in, "width": "220px"}),
                    dcc.RadioItems(id="nm-affects", value="any", inline=True,
                                   options=[{"label": " 任意关节 ", "value": "any"},
                                            {"label": " 负载关节(1-3) ", "value": "load"},
                                            {"label": " 远端关节(4-6) ", "value": "distal"}],
                                   style={"fontSize": "12px", "display": "inline-block"}),
                ], style={"marginTop": "8px"}),
                html.Div([
                    html.Span("表现信号：", style={"fontSize": "12px"}),
                    dcc.Checklist(id="nm-manifests", inline=True,
                                  options=[{"label": f" {f} ", "value": f} for f in SIGNAL_FAMILIES],
                                  value=[], style={"fontSize": "11px", "display": "inline-block"}),
                ], style={"marginTop": "6px"}),
                html.Div([
                    dcc.Input(id="nm-cause", type="text", placeholder="成因描述",
                              style={**_in, "width": "300px"}),
                    dcc.Input(id="nm-reco", type="text", placeholder="建议检查/处置",
                              style={**_in, "width": "300px"}),
                    dcc.RadioItems(id="nm-phase", value="any", inline=True,
                                   options=[{"label": " 任意相位 ", "value": "any"},
                                            {"label": " 抓取 ", "value": "grip"},
                                            {"label": " 搬运 ", "value": "transfer"}],
                                   style={"fontSize": "12px", "display": "inline-block"}),
                ], style={"marginTop": "6px"}),
            ], style={"display": "none"}),
            html.Button("登记案例", id="btn-case", n_clicks=0,
                        style={**_btn, "background": C["good"], "color": "#06210b", "marginTop": "10px"}),
            html.Div(id="case-status", style={"fontSize": "11px", "color": C["good"], "minHeight": "15px"}),
        ], style=_panel),
      ], style={"flex": "1.2", "minWidth": 0}),

      html.Div([
        # ---- learn & apply ----
        html.Div([
            html.Div("学习与生效 — 聚合当前最终决策 → 先验写入 kb_overrides.json → 热生效", style=_h),
            html.Div([
                html.Button("学习并生效", id="btn-learn", n_clicks=0,
                            style={**_btn, "background": C["accent"], "color": "#fff"}),
                dcc.Checklist(id="learn-force", value=[],
                              options=[{"label": " 少量数据也生效（<10 条，不建议）", "value": "on"}],
                              style={"fontSize": "11px", "display": "inline-block", "marginLeft": "10px"}),
                html.Button("评估当前 KB（755 周期）", id="btn-eval", n_clicks=0,
                            style={**_btn, "background": C["panel2"], "color": C["ink"],
                                   "border": f"1px solid {C['line']}", "marginLeft": "10px"}),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            dcc.Loading(html.Div(id="learn-out", style={"fontSize": "11px", "marginTop": "8px",
                                                        "fontFamily": FONT_MONO})),
        ], style=_panel),

        # ---- AI suggestions ----
        html.Div([
            html.Div("AI 建议（人审门禁）— 起草 → 实测评估 → 采纳/忽略", style=_h),
            html.Div([
                html.Button("起草结构修订建议", id="btn-suggest", n_clicks=0,
                            style={**_btn, "background": C["warn"], "color": "#1a1200"}),
                dcc.Checklist(id="sug-demo", value=[],
                              options=[{"label": " 用模拟反馈演示（无足够真实反馈时）", "value": "on"}],
                              style={"fontSize": "11px", "display": "inline-block", "marginLeft": "10px"}),
                html.Span(" 本地模型约 1 分钟", style={"fontSize": "11px", "color": C["muted"]}),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            dcc.Loading(html.Div(id="sug-out", style={"marginTop": "8px"})),
        ], style=_panel),
      ], style={"flex": "1", "minWidth": 0}),
    ], style={"display": "flex", "gap": "14px", "padding": "16px 22px", "alignItems": "flex-start"}),
], style={"background": C["bg"], "minHeight": "100vh", "color": C["ink"], "fontFamily": FONT_SANS})


# ============================ callbacks ======================================
@callback(Output("newmode-form", "style"), Input("case-newmode", "value"))
def toggle_newmode(v):
    return {"display": "block" if "on" in (v or []) else "none"}


def _refresh():
    return ledger_rows(), status_chips()


@callback(Output("ledger", "data"), Output("chips", "children"),
          Output("rev-status", "children"),
          Input("btn-revise", "n_clicks"),
          State("rev-cycle", "value"), State("rev-fm", "value"), State("rev-note", "value"),
          prevent_initial_call=True)
def on_revise(n, cyc, fm, note):
    if not n or cyc is None or not fm:
        return no_update, no_update, "请填写周期号并选择改判原因"
    prev = [r for r in kbf.load_feedback() if r.get("cycle_id") == cyc]
    top1 = (prev[-1].get("model_top1") or prev[-1].get("ai_top1")) if prev else None
    if top1 is None and cyc in records():
        top1 = line_kg.retrieve(records()[cyc], top_k=1)[0]["id"]
    append_row({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cycle_id": int(cyc),
                "analyst_verdict": "revise", "analyst_fm": fm, "model_top1": top1,
                "note": note or "", "source": "console"})
    led, chips = _refresh()
    return led, chips, f"已修订：周期 {cyc} → {fm}（旧决策保留在历史中，学习时以此为准）"


@callback(Output("ledger", "data", allow_duplicate=True),
          Output("chips", "children", allow_duplicate=True),
          Output("case-status", "children"),
          Input("btn-case", "n_clicks"),
          State("case-cycle", "value"), State("case-fm", "value"), State("case-note", "value"),
          State("case-newmode", "value"), State("nm-id", "value"), State("nm-name", "value"),
          State("nm-affects", "value"), State("nm-manifests", "value"),
          State("nm-cause", "value"), State("nm-reco", "value"), State("nm-phase", "value"),
          prevent_initial_call=True)
def on_case(n, cyc, fm, note, newmode, nid, nname, naff, nman, ncause, nreco, nphase):
    if not n or cyc is None:
        return no_update, no_update, "请填写周期号"
    msg = []
    if "on" in (newmode or []):
        if not (nid and nname and nman and ncause and nreco):
            return no_update, no_update, "新失效模式需要：id、名称、至少一个表现信号、成因、处置"
        fmid = nid if nid.startswith("FM_") else f"FM_{nid}"
        if any(m["id"] == fmid for m in FMEA):
            return no_update, no_update, f"{fmid} 已存在"
        data = load_overrides()
        data.setdefault("new_modes", []).append({
            "id": fmid, "name": nname, "affects": naff, "manifests": nman,
            "cause": ncause, "recommendation": nreco, "phase_hint": nphase})
        save_overrides(data)                      # writes + reload_kb()
        fm = fmid
        msg.append(f"新失效模式 {fmid} 已加入知识库")
    if not fm:
        return no_update, no_update, "请选择真实原因（或定义新失效模式）"
    top1 = None
    if cyc in records():
        top1 = line_kg.retrieve(records()[cyc], top_k=1)[0]["id"]
    append_row({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cycle_id": int(cyc),
                "analyst_verdict": "case", "analyst_fm": fm, "model_top1": top1,
                "note": note or "", "source": "console"})
    msg.append(f"案例已登记：周期 {cyc} → {fm}")
    led, chips = _refresh()
    return led, chips, "；".join(msg)


@callback(Output("learn-out", "children"), Output("chips", "children", allow_duplicate=True),
          Input("btn-learn", "n_clicks"), Input("btn-eval", "n_clicks"),
          State("learn-force", "value"), prevent_initial_call=True)
def on_learn(nl, ne, force):
    if ctx.triggered_id == "btn-eval":
        with_p = kbf.quick_eval(w_prior=1.0)
        no_p = kbf.quick_eval(w_prior=0.0)
        return (html.Pre(f"当前 KB（含先验+结构+新模式）: top-1 {with_p[0]:.3f}  top-5 {with_p[1]:.3f}  MRR {with_p[2]:.3f}\n"
                         f"同一 KB、关闭先验          : top-1 {no_p[0]:.3f}  top-5 {no_p[1]:.3f}  MRR {no_p[2]:.3f}",
                         style={"margin": 0}), status_chips())
    min_rows = 1 if "on" in (force or []) else kbf.MIN_ROWS
    n, res = kbf.learn_and_write(min_rows=min_rows)
    if res is None:
        return f"可用最终决策仅 {n} 条（<{kbf.MIN_ROWS}）；勾选左侧选项可强制生效。", status_chips()
    priors = res[0]
    lines = [f"已从 {n} 条最终决策学习先验并热生效："]
    for f_, v in sorted(priors.items(), key=lambda kv: -abs(kv[1]))[:8]:
        lines.append(f"  {f_:20s} {v:+.3f}")
    return html.Pre("\n".join(lines), style={"margin": 0}), status_chips()


@callback(Output("sug-out", "children"), Output("proposals", "data"),
          Input("btn-suggest", "n_clicks"), State("sug-demo", "value"),
          prevent_initial_call=True)
def on_suggest(n, demo):
    raw, model, parsed, src = kbf.draft_suggestions(demo="on" in (demo or []))
    if raw is None:
        return html.Div(src + "；可勾选“用模拟反馈演示”。", style={"fontSize": "11px"}), None
    props = (parsed or {}).get("proposals") or []
    if not props:
        return html.Div(["模型未给出结构化提案。原始输出已写入 kb_suggestions.md。"],
                        style={"fontSize": "11px"}), None
    items = []
    for i, p in enumerate(props):
        items.append(html.Div([
            html.Div([html.B(p.get("fm_id", "?")), "  ",
                      html.Span(json.dumps(p.get("edit"), ensure_ascii=False),
                                style={"fontFamily": FONT_MONO, "fontSize": "11px"})]),
            html.Div(f"依据：{p.get('motivated_by', '')}", style={"fontSize": "11px", "color": C["muted"]}),
            html.Div([
                html.Button("实测评估", id={"type": "prop-eval", "idx": i}, n_clicks=0,
                            style={**_btn, "background": C["panel2"], "color": C["ink"],
                                   "border": f"1px solid {C['line']}"}),
                html.Button("采纳", id={"type": "prop-adopt", "idx": i}, n_clicks=0,
                            style={**_btn, "background": C["good"], "color": "#06210b"}),
                html.Span(id={"type": "prop-res", "idx": i},
                          style={"fontSize": "11px", "fontFamily": FONT_MONO}),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "4px"}),
        ], style={"borderTop": f"1px dashed {C['line']}", "padding": "8px 0"}))
    head = html.Div(f"来源：{src} · 模型：{model} · 采纳前请先实测评估",
                    style={"fontSize": "11px", "color": C["muted"]})
    return [head] + items, props


def _merged_with(prop):
    data = load_overrides()
    fid, edit = prop.get("fm_id"), dict(prop.get("edit") or {})
    modes = data.setdefault("modes", {})
    cur = modes.setdefault(fid, {})
    for k in ("manifests_add", "manifests_remove"):
        if k in edit:
            cur[k] = sorted(set(cur.get(k, [])) | set(edit[k]))
    if edit.get("phase_hint"):
        cur["phase_hint"] = edit["phase_hint"]
    return data


@callback(Output({"type": "prop-res", "idx": ALL}, "children"),
          Output("chips", "children", allow_duplicate=True),
          Input({"type": "prop-eval", "idx": ALL}, "n_clicks"),
          Input({"type": "prop-adopt", "idx": ALL}, "n_clicks"),
          State("proposals", "data"), State({"type": "prop-res", "idx": ALL}, "children"),
          prevent_initial_call=True)
def on_prop(ne, na, props, cur):
    trig = ctx.triggered_id
    out = list(cur) if cur else [""] * len(props or [])
    if not props or trig is None or not isinstance(trig, dict):
        return out, no_update
    i = trig["idx"]
    if not (ne[i] or na[i]):
        return out, no_update
    if trig["type"] == "prop-eval":
        base = kbf.quick_eval(w_prior=1.0)
        reload_kb(_merged_with(props[i]))
        trial = kbf.quick_eval(w_prior=1.0)
        reload_kb()                                   # restore on-disk KB
        d = trial[0] - base[0]
        out[i] = (f"top-1 {base[0]:.3f}→{trial[0]:.3f} ({d:+.3f}) · "
                  f"MRR {base[2]:.3f}→{trial[2]:.3f}"
                  + ("  ⚠ 变差，建议忽略" if trial[2] < base[2] else "  ✓ 未变差"))
        return out, no_update
    save_overrides(_merged_with(props[i]))            # adopt: persist + reload
    out[i] = "✓ 已采纳并热生效（诊断界面将自动刷新 KB）"
    return out, status_chips()


if __name__ == "__main__":
    app.run(debug=False, port=8080)
