"""Figure builder for the SAFEPROCESS 2027 paper and its long version.

Every figure reads from results/*.json or from pipeline/line_kg.py, so no number
in a figure can drift from the number in the text.  Vector PDF out, sized for a
single column (3.5 in) or the full width (7.0 in).

The six-page SAFEPROCESS paper (safeprocess2027_final.tex) has room for two
figures, so it uses only the most important ones and merges where it can:
    Fig. 1  fig_teaser       tie-break interval + balanced recall, merged
    Fig. 2  fig_signatures   structural signatures
The remaining figures (protocol, certificates, power, granularity, label budget)
belong to the long version; their numbers appear in the short paper's tables
and text instead.

Run from anywhere; name figures to build only those:
    python make_figures.py                  # every figure
    python make_figures.py fig_teaser       # just Fig. 1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE                      # this file now lives at the repository root
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "pipeline"))

# ---------------------------------------------------------------------------
# Palette.  Categorical slots 1-2 of the validated default palette; checked with
# scripts/validate_palette.js against the print-white surface:
#   CVD dE 24.7 (protan), normal-vision dE 33.6, both >= 3:1 contrast, all PASS.
# Only two categorical hues are used anywhere, so the all-pairs floors hold and
# the figures survive greyscale printing on lightness alone.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#dcdbd6"
FILL_NEUTRAL = "#c9c8c3"
FILL_NEUTRAL_2 = "#e4e3de"

# Sequential blue ramp (single hue, light -> dark) for the presence matrix.
BLUE_100 = "#cde2fb"
BLUE_600 = "#184f95"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load(name):
    with open(RESULTS / name, encoding="utf-8") as fh:
        return json.load(fh)


def tidy(ax, xgrid=True):
    """Recessive chrome: no top/right spines, grid behind the marks only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
    else:
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ===========================================================================
# Fig. 1 (SAFEPROCESS) -- the measurement, two panels merged into one figure.
# Left: the tie-break interval against the group answer.  Right: balanced
# recall, the one reading a constant predictor cannot game.  Every value on
# the left is a five-split leave-setting-out mean on the splits of Table 1, so
# the prior rules are the same population as the marks, not a scale reference.
# ===========================================================================
def fig_teaser():
    b = load("baselines.json")
    tb = b["tiebreak_grouped"]
    pr = b["prior_vs_retrieval"]
    br = b["balanced_recall"]

    style = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
             "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
             "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}
    with mpl.rc_context(style):
        fig, (ax, bx) = plt.subplots(
            1, 2, figsize=(7.0, 2.6),
            gridspec_kw=dict(width_ratios=[1.0, 0.72], wspace=0.55))

        # -- left: point answer (top-1) against group answer (cluster top-1) --
        y_pt, y_cl = 1.0, 0.0
        lo, hi = tb["random_min"], tb["random_max"]
        ax.add_patch(Rectangle((lo, y_pt - 0.09), hi - lo, 0.18,
                               facecolor=FILL_NEUTRAL, edgecolor="none", zorder=1))
        for x in (lo, hi):
            ax.plot([x, x], [y_pt - 0.09, y_pt + 0.09], color=INK_2, lw=1.0, zorder=2)
        for key, ms in (("reverse_declaration", 5), ("alphabetical", 5),
                        ("declaration_order", 7.5)):
            ax.plot(tb[key], y_pt, "o", ms=ms, color=BLUE, mec="white",
                    mew=1.2, zorder=4)
        ax.plot(tb["oracle_ceiling"], y_pt, "o", ms=6.5, mfc="white", mec=INK_2,
                mew=1.8, zorder=4)
        ax.annotate("", xy=(lo, y_pt + 0.24), xytext=(hi, y_pt + 0.24),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
        ax.annotate(f"{tb['interval_width']:.3f} wide", ((lo + hi) / 2, y_pt + 0.29),
                    ha="center", va="bottom", color=BLUE)

        ax.plot(tb["cluster_by_tiebreak"]["declaration"], y_cl, "o", ms=7.5,
                color=BLUE, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"same under all {tb['draws'] + 3} orderings",
                    (tb["cluster_by_tiebreak"]["declaration"], y_cl),
                    xytext=(-9, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=7, color=BLUE)

        for x, y in ((pr["prevalence_prior"]["top1"], y_pt),
                     (pr["cluster_majority"], y_cl)):
            ax.plot([x, x], [y - 0.17, y + 0.17], ls=(0, (2.5, 1.5)),
                    color=CRITICAL, lw=1.6, zorder=3)
            ax.annotate("prior", (x, y - 0.19), ha="center", va="top",
                        fontsize=7, color=CRITICAL)

        ax.annotate("point answer (top-1)", (0.01, y_pt + 0.62),
                    xycoords=("axes fraction", "data"), va="center")
        ax.annotate("group answer (cluster top-1)", (0.01, y_cl + 0.48),
                    xycoords=("axes fraction", "data"), va="center")
        ax.set_xlim(0.05, 0.46)
        ax.set_ylim(-0.62, 1.82)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_yticks([])
        ax.set_xlabel("accuracy (five-split mean)")
        ax.set_title("The point answer is not determined; the group is",
                     loc="left", pad=6)
        tidy(ax, xgrid=True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(INK)

        # -- right: balanced recall -------------------------------------------
        arms = [("prevalence prior", "prior (constant)", MUTED),
                ("retrieval", "retrieval", BLUE),
                ("two-stage + repair", "two-stage", BLUE),
                ("flat control", "flat control", ORANGE)]
        ys = np.arange(len(arms))[::-1]
        for y, (lab, key, col) in zip(ys, arms):
            v = br[key]["macro"]
            bx.barh(y, v, height=0.52, color=col, edgecolor="none", zorder=2)
            bx.annotate(f"{v:.3f}", (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left")
        u = br["uniform_macro"]
        bx.axvline(u, color=INK_2, ls=(0, (2.5, 1.5)), lw=1.4, zorder=3)
        bx.annotate("uniform over 12 (1/12)", (u, len(arms) - 0.45), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7,
                    color=MUTED)
        bx.set_yticks(ys)
        bx.set_yticklabels([a[0] for a in arms])
        bx.set_xlim(0, 0.52)
        bx.set_ylim(-0.6, len(arms) - 0.1)
        bx.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        bx.set_xlabel("balanced recall")
        bx.set_title("Balanced recall restores the ordering", loc="right", pad=6)
        tidy(bx, xgrid=True)
        bx.tick_params(axis="y", length=0)
        bx.spines["left"].set_visible(False)
        bx.spines["bottom"].set_color(INK)
        save(fig, "fig_teaser.pdf")


# ===========================================================================
# Fig. 2 -- structural signatures.  Identity, not magnitude: a presence matrix.
# ===========================================================================
def fig_signatures():
    import line_kg as K
    import distinguishability as D

    twins = [set(c) for c in D.twin_classes()]
    fmea = {e["id"]: e for e in K.FMEA}

    # Order: twin clusters first (members adjacent), then the singletons.
    ordered, seen = [], set()
    for cluster in sorted(twins, key=lambda c: (-len(c), sorted(c)[0])):
        for fid in sorted(cluster):
            ordered.append(fid)
            seen.add(fid)
    for e in K.FMEA:
        if e["id"] not in seen:
            ordered.append(e["id"])

    scopes = ["any", "load", "distal", "J1-2"]
    families = ["motor_torque", "torque_sensor_a", "joint_position",
                "joint_velocity", "robot_current"]
    cols = scopes + families

    def scope_of(e):
        aff = e["affects"]
        return "J1-2" if isinstance(aff, list) else aff

    M = np.zeros((len(ordered), len(cols)))
    for r, fid in enumerate(ordered):
        e = fmea[fid]
        M[r, cols.index(scope_of(e))] = 1
        for fam in e["manifests"]:
            if fam in cols:
                M[r, cols.index(fam)] = 1

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for r in range(len(ordered)):
        for c in range(len(cols)):
            on = M[r, c] > 0
            ax.add_patch(Rectangle(
                (c + 0.06, r + 0.06), 0.88, 0.88,          # 2px-equivalent gap
                facecolor=BLUE_600 if on else FILL_NEUTRAL_2,
                edgecolor="white", linewidth=0.7, zorder=2))

    # Bracket each twin cluster and give it the verdict it earns in Table 2.
    verdicts = {
        "FM_axisweight": "repaired",
        "FM_collision_card": "inconclusive",
        "FM_misgrip": "undetermined",
        "FM_miscommutation": "hardware gap",
    }
    row_of = {fid: i for i, fid in enumerate(ordered)}
    for cluster in twins:
        rows = sorted(row_of[f] for f in cluster)
        top, bot = rows[0], rows[-1] + 1
        ax.add_patch(Rectangle(
            (-0.02, top + 0.10), len(cols) + 0.04, bot - top - 0.20,
            fill=False, edgecolor=CRITICAL, linewidth=1.2, zorder=4))
        verdict = next((v for f, v in verdicts.items() if f in cluster), "")
        ax.annotate(f"identical signature\n$\\rightarrow$ {verdict}",
                    xy=(len(cols) + 0.18, (top + bot) / 2),
                    va="center", ha="left", fontsize=6.6, color=CRITICAL)

    ax.set_xlim(-0.1, len(cols) + 2.6)
    ax.set_ylim(len(ordered) + 0.9, -1.9)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(
        ["any", "load", "distal", "J1-2",
         "motor\ntorque", "torque\nsensor a", "joint\nposition",
         "joint\nvelocity", "robot\ncurrent"])
    ax.set_yticks(np.arange(len(ordered)) + 0.5)
    ax.set_yticklabels([fmea[f]["name"] for f in ordered])
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Column-group headers, above the matrix.
    ax.annotate("affected joints", xy=(2.0, -1.35), ha="center",
                fontsize=7.5, color=INK_2)
    ax.annotate("manifesting signal families", xy=(6.5, -1.35), ha="center",
                fontsize=7.5, color=INK_2)
    ax.plot([0.06, 3.94], [-0.95, -0.95], color=MUTED, lw=0.7)
    ax.plot([4.06, 8.94], [-0.95, -0.95], color=MUTED, lw=0.7)

    # Key: the fill is binary, so it needs naming once.
    for i, (col, lab) in enumerate([(BLUE_600, "declared in the knowledge base"),
                                    (FILL_NEUTRAL_2, "not declared")]):
        x = 0.06 + i * 4.6
        ax.add_patch(Rectangle((x, len(ordered) + 0.35), 0.55, 0.42,
                               facecolor=col, edgecolor="white", linewidth=0.7))
        ax.annotate(lab, xy=(x + 0.72, len(ordered) + 0.56), va="center",
                    ha="left", fontsize=7, color=INK_2)

    save(fig, "fig_signatures.pdf")


# ===========================================================================
# Fig. 3 -- the validation protocol.  Before/after on a shared scale: dumbbell.
# ===========================================================================
def fig_protocol():
    d = load("repair_results.json")["leakage_check"]
    pc = d["per_cluster"]

    rows = [
        ("Flat 12-way classifier",
         d["random_split"]["flat_classifier"], d["leave_setting_out"]["flat_classifier"], None),
        ("Two-stage diagnosis",
         d["random_split"]["two_stage"], d["leave_setting_out"]["two_stage"], None),
        ("Point top-1 (retrieval)",
         d["random_split"]["point_top1"], d["leave_setting_out"]["point_top1"], None),
        ("Load cluster",
         pc["axisweight+losecan+varweight"]["random"]["acc"],
         pc["axisweight+losecan+varweight"]["leave_setting_out"]["acc"],
         (pc["axisweight+losecan+varweight"]["random"]["adopted"],
          pc["axisweight+losecan+varweight"]["leave_setting_out"]["adopted"])),
        ("Collision cluster",
         pc["collision_card+collision_foam"]["random"]["acc"],
         pc["collision_card+collision_foam"]["leave_setting_out"]["acc"],
         (pc["collision_card+collision_foam"]["random"]["adopted"],
          pc["collision_card+collision_foam"]["leave_setting_out"]["adopted"])),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ys = np.arange(len(rows))[::-1]
    for y, (label, rand, grp, adopt) in zip(ys, rows):
        ax.plot([grp, rand], [y, y], color=GRID, lw=2.4, zorder=1,
                solid_capstyle="round")
        ax.plot(rand, y, "o", ms=8, color=ORANGE, mec="white", mew=1.2, zorder=3)
        ax.plot(grp, y, "o", ms=8, color=BLUE, mec="white", mew=1.2, zorder=3)
        # When the two dots nearly coincide, stacking the labels above/below
        # is unreadable; push them apart horizontally instead.
        close = abs(rand - grp) < 0.05
        ax.annotate(f"{rand:.3f}", (rand, y),
                    xytext=((14, 7) if close else (0, 9)),
                    textcoords="offset points",
                    ha=("left" if close else "center"), fontsize=7,
                    color=ORANGE)
        ax.annotate(f"{grp:.3f}", (grp, y),
                    xytext=((-14, -12) if close else (0, -14)),
                    textcoords="offset points",
                    ha=("right" if close else "center"), fontsize=7,
                    color=BLUE)
        if adopt is not None:
            r_ad, g_ad = adopt
            txt = ("adopted → adopted" if (r_ad and g_ad)
                   else "adopted → REJECTED")
            col = INK_2 if (r_ad and g_ad) else CRITICAL
            weight = "normal" if (r_ad and g_ad) else "bold"
            ax.annotate(txt, (1.005, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=7, color=col,
                        fontweight=weight)

    ax.axhline(1.5, color=GRID, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, len(rows) - 0.4)   # room for the below-dot value labels
    ax.set_xlabel("accuracy")
    tidy(ax, xgrid=True)
    ax.tick_params(axis="y", length=0)

    ax.plot([], [], "o", ms=7, color=ORANGE,
            label="random split (session leakage)")
    ax.plot([], [], "o", ms=7, color=BLUE, label="leave-setting-out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2)
    save(fig, "fig_protocol.pdf")


# ===========================================================================
# Fig. 4 -- repair certificates against their three baselines.
# ===========================================================================
def fig_certificates():
    certs = load("repair_results.json")["certificates"]
    panels = [
        ("Load-torque cluster\naxis weight / lost payload / variable weight",
         certs["axisweight+losecan+varweight"], "adopted 5/5 splits", GOOD),
        ("Distal-collision cluster\ncarton / foam",
         certs["collision_card+collision_foam"], "inconclusive", CRITICAL),
    ]
    keys = [("acc_test", "repaired (held out)", BLUE),
            ("acc_deployed_repr", "deployed representation", FILL_NEUTRAL),
            ("acc_majority", "majority class", FILL_NEUTRAL_2),
            ("acc_permuted", "permuted labels", FILL_NEUTRAL_2)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.10)
    for i, (ax, (title, cert, verdict, vcol)) in enumerate(zip(axes, panels)):
        ys = np.arange(len(keys))[::-1]
        for y, (k, lab, col) in zip(ys, keys):
            v = cert[k]
            ax.barh(y, v, height=0.62, color=col, edgecolor="white",
                    linewidth=0.8, zorder=2)
            # Value inside the dark bar, outside the pale ones: never a label
            # sitting in the gutter between panels.
            inside = col == BLUE
            ax.annotate(f"{v:.3f}", (v, y), xytext=(-5 if inside else 4, 0),
                        textcoords="offset points", va="center",
                        ha="right" if inside else "left", fontsize=7,
                        color="white" if inside else INK)
        ax.set_yticks(ys)
        if i == 0:
            ax.set_yticklabels([lab for _, lab, _ in keys])
        ax.set_xlim(0, 1.0)
        ax.set_title(f"{title}\n", fontsize=7.8, color=INK, loc="left", pad=4)
        ax.annotate(f"{verdict}   ($n_{{fit}}$={cert['n_fit']}, "
                    f"$n_{{test}}$={cert['n_test']})",
                    (0.0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                    textcoords="offset points", fontsize=7.2, color=vcol,
                    fontweight="bold")
        tidy(ax, xgrid=True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("within-cluster accuracy")
    save(fig, "fig_certificates.pdf")


# ===========================================================================
# Fig. 5 -- how many fault instantiations before the protocol commits.
# ===========================================================================
def fig_power():
    g = load("synth_instantiation_power.json")
    c = load("sim_collision_power.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)

    # -- gripping pair, semi-synthetic on real normal cycles ---------------
    ax = axes[0]
    xs = [p["n_instantiations"] for p in g["sweep"]]
    val = [p["search_validated"] for p in g["sweep"]]
    adopt = [p["adopt_rate_given_verdict"] or 0.0 for p in g["sweep"]]
    ax.plot(xs, val, "-o", color=BLUE, lw=2, ms=6, mec="white", mew=1.0,
            label="grouped search runs", zorder=3)
    ax.plot(xs, adopt, "-s", color=ORANGE, lw=2, ms=6, mec="white", mew=1.0,
            label="adoption rate across draws", zorder=3)
    ax.axvline(g["min_instantiations_for_validated_search"], color=MUTED,
               ls=":", lw=1.0, zorder=1)
    ax.axvline(g["min_instantiations_for_stable_verdict"], color=MUTED,
               ls="--", lw=1.0, zorder=1)
    ax.annotate("search can run\n($N\\geq4$)", (4, 0.84), xytext=(11, 0),
                textcoords="offset points", fontsize=6.8, color=INK_2,
                va="center")
    ax.annotate("verdict stable\n($N\\geq8$)", (8, 0.44), xytext=(7, 0),
                textcoords="offset points", fontsize=6.8, color=INK_2,
                va="center")
    ax.set_title("Gripping pair (semi-synthetic)", fontsize=7.8, loc="left",
                 color=INK)
    ax.set_xlabel("fault instantiations $N$")
    ax.set_ylabel("rate")
    ax.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.0, 0.12))

    # -- collision pair, simulated contacts, two noise floors --------------
    ax = axes[1]
    # Two noise floors are the SAME measure under two conditions, so they are
    # separated by line style, not by a third hue.  Direct-labelled rather than
    # legended: the panel has no room for a legend box that misses the marks.
    styles = {"no jitter": ("-", "o", (11.2, 0.10), "no jitter"),
              "jitter 2.5 N.m": ("--", "^", (7.6, 0.47), "jitter 2.5 N$\\cdot$m")}
    for floor, sweep in c["sweeps"].items():
        ls, mk, (lx, ly), lab = styles[floor]
        xs = [p["n_instantiations"] for p in sweep]
        ax.plot(xs, [p["search_validated"] for p in sweep], ls, marker=mk,
                color=BLUE, lw=2, ms=6, mec="white", mew=1.0, zorder=3)
        ax.plot(xs, [p["adopt_rate_given_verdict"] or 0.0 for p in sweep], ls,
                marker=mk, color=ORANGE, lw=2, ms=6, mec="white", mew=1.0,
                zorder=3)
        ax.annotate(lab, (lx, ly), fontsize=6.8, color=ORANGE, ha="center",
                    va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.axvline(4, color=MUTED, ls=":", lw=1.0, zorder=1)
    ax.annotate("adoption still\nflipping at $N=16$", (16, 0.20),
                xytext=(11.5, 0.80), fontsize=6.8, color=CRITICAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=0.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("Collision pair (simulated contacts)", fontsize=7.8,
                 loc="left", color=INK)
    ax.set_xlabel("fault instantiations $N$")

    for ax in axes:
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        tidy(ax, xgrid=False)

    fig.text(0.5, -0.10,
             "voraus-AD supplies $N=1$ per gripping mode and $N=3$ for carton "
             "collision — below the threshold at which a grouped search "
             "can run at all.",
             ha="center", fontsize=7, color=INK_2)
    save(fig, "fig_power.pdf")


# ===========================================================================
# Fig. 6 -- granularity ablation.  Magnitude, one series, ordered.
# ===========================================================================
def fig_granularity():
    ab = load("repair_results.json")["granularity_ablation"]
    order = ["mode-level scalar", "per-family scalar", "joint-resolved (load)",
             "joint-resolved (all 6)"]
    labels = ["mode-level scalar\n(2 features)",
              "per signal family\n(4 features)",
              "joint-resolved,\nload joints (12)",
              "joint-resolved,\nall six joints (24)"]
    vals = [ab[k]["acc"] for k in order]
    majority = ab[order[0]]["majority"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ys = np.arange(len(order))[::-1]
    for y, v in zip(ys, vals):
        below = v < majority
        ax.barh(y, v, height=0.62, color=FILL_NEUTRAL if below else BLUE,
                edgecolor="white", linewidth=0.8, zorder=2)
        # A bar that falls short of the reference line would collide with it;
        # label those to the LEFT, inside the bar.
        ax.annotate(f"{v:.3f}", (v, y), xytext=(-5 if below else 4, 0),
                    textcoords="offset points", va="center",
                    ha="right" if below else "left", fontsize=7,
                    color=INK if below else INK,
                    fontweight="bold" if below else "normal")
    ax.axvline(majority, color=CRITICAL, ls="--", lw=1.0, zorder=4)
    ax.annotate(f"majority class {majority:.3f}", (majority, -0.62),
                xytext=(4, 0), textcoords="offset points", fontsize=6.8,
                color=CRITICAL, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.1, len(order) - 0.4)
    ax.set_xlabel("within-cluster accuracy")
    tidy(ax, xgrid=True)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_granularity.pdf")


# ===========================================================================
# Fig. 7 -- label budget.  Change over a continuous x: lines, one axis.
# The result is negative for us, so the chart must not flatter: the control
# gets the strong categorical hue and our two arms are plotted against it.
# ===========================================================================
def fig_label_budget():
    d = load("label_budget_curve.json")
    ks = d["sizes"]
    get = lambda key: [d[key][str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    base = d["stage1_only"][str(ks[0])]

    # The tie region has to be visible before any line is, or the eye reads a
    # crossover where the measurement says the arms are indistinguishable.
    rob = d.get("crossover_robustness", {}).get("per_k", {})
    tie = [int(k) for k, v in rob.items() if 0.35 <= v["two_stage_win_rate"] <= 0.65]
    if tie:
        ax.axvspan(ks[0], max(tie), color=MUTED, alpha=0.10, lw=0, zorder=0)
        # Label sits below the axis, not inside the plot: at the top it collided
        # with both the legend and the prior arm's entry.
        ax.annotate("within\nsplit noise", (max(tie) ** 0.5 * ks[0] ** 0.5, 0.108),
                    fontsize=6.2, color=MUTED, ha="center", va="bottom",
                    linespacing=1.15)

    # The prior arm carries no diagnostic content; it is here so that the
    # control's early rise is not misread as learning.
    if "prior_at_k" in d:
        ax.plot(ks, get("prior_at_k"), ":", color=MUTED, lw=1.4, zorder=2,
                label="majority of the $k$ labels")
    ax.plot(ks, get("control_12way"), "-o", color=ORANGE, lw=2, ms=5,
            mec="white", mew=1.0, zorder=4, label="flat 12-way control")
    ax.plot(ks, get("two_stage_targeted"), "-s", color=BLUE, lw=2, ms=5,
            mec="white", mew=1.0, zorder=3, label="two-stage, targeted labels")
    ax.plot(ks, get("two_stage_unrestricted"), "--^", color=BLUE, lw=1.6, ms=5,
            mec="white", mew=1.0, zorder=3, alpha=0.75,
            label="two-stage, unrestricted labels")
    ax.axhline(base, color=MUTED, ls=":", lw=1.2, zorder=1)
    ax.annotate(f"knowledge base alone, no labels ({base:.3f})",
                (ks[-1], base), xytext=(-2, -12), textcoords="offset points",
                fontsize=6.6, color=INK_2, ha="right")

    # Mark where the control overtakes everything.
    cross = next((k for k in ks
                  if d["control_12way"][str(k)] > d["two_stage_targeted"][str(k)]
                  and d["control_12way"][str(k)] > base), None)
    if cross:
        ax.axvline(cross, color=CRITICAL, ls="--", lw=1.0, zorder=2)
        # Right of the rule and low, where no series or legend entry sits.
        ax.annotate(f"control ahead\nfrom $k={cross}$", (cross, 0.335),
                    xytext=(5, 0), textcoords="offset points", fontsize=6.6,
                    color=CRITICAL, va="center", linespacing=1.2)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("labelled cycles $k$")
    ax.set_ylabel("end-to-end top-1 (12-way)")
    ax.set_ylim(0.095, 0.60)
    tidy(ax, xgrid=False)
    ax.legend(loc="upper left", fontsize=6.4, borderpad=0.2,
              labelspacing=0.35, handlelength=1.9)
    save(fig, "fig_label_budget.pdf")


FIGURES = {f.__name__: f for f in (fig_teaser, fig_signatures, fig_protocol,
                                   fig_certificates, fig_power, fig_granularity,
                                   fig_label_budget)}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    OUT.mkdir(exist_ok=True)
    print("building figures ->", OUT)
    for name in wanted:
        FIGURES[name]()
    print("done")
