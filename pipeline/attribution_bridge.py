"""Per-signal attribution and the numeric-to-semantic anomaly record (voraus-AD).

Given an anomalous robot cycle, score which signals/joints deviated from the
normal envelope (how much, where in the cycle) and pack that into the JSON
record that the KG-retrieval and LLM stages downstream consume.

Prototype shortcuts to swap for a full run: the reference profile is built from
the normal runs in the loaded slice (pass all 948 train-normal runs for a real
mean/std envelope, see build_reference); anomaly_score/threshold come from
MVT-Flow; action codes 0..14 map to phase names via ACTION_NAMES. The
attribution here is envelope-deviation; MVT-Flow's per-timestep input gradient
gives a second, model-internal attribution to cross-check it.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
JOINTS = range(1, 7)
GRID_N = 500                      # common normalized-cycle resolution
EPS = 1e-9

# Per-joint signal families used for joint-level attribution.
# The residual families (RESIDUAL_DEFS below) are kept but unused: adding them
# dropped top-1 from 0.158 to 0.152, since residuals sit near zero and tiny
# deviations get spuriously high z that the envelope+Jaccard retrieval can't use.
JOINT_SIGNALS = {
    "motor_torque":    "Motor torque (N\u00b7m)",
    "torque_sensor_a": "Torque sensor A (N\u00b7m)",
    "joint_velocity":  "Joint velocity (rad/s)",
    "joint_position":  "Joint position (rad)",
}
# Robot-level (not per-joint) signals.
ROBOT_SIGNALS = ["robot_current", "system_current"]

# Residual families (per joint: minuend - subtrahend); add a key to JOINT_SIGNALS
# to re-enable. Unused by default, see the note above.
RESIDUAL_DEFS = {
    "torque_residual": ("motor_torque", "computed_torque"),
    "pos_error":       ("joint_position", "target_position"),
    "vel_error":       ("joint_velocity", "target_velocity"),
}


def measured_read_families():
    """Per-joint families to READ from the parquet for attribution: the measured
    display families plus the residual SOURCE families (computed/target)."""
    measured = [f for f in JOINT_SIGNALS if f not in RESIDUAL_DEFS]
    sources = {fam for pair in RESIDUAL_DEFS.values() for fam in pair}
    return sorted(set(measured) | sources)


def add_residual_columns(df, joints=JOINTS):
    """Compute the residual columns (e.g. torque_residual_2 = motor_torque_2 -
    computed_torque_2) in place, where both source columns exist."""
    for name, (a, b) in RESIDUAL_DEFS.items():
        for j in joints:
            ca, cb, cn = f"{a}_{j}", f"{b}_{j}", f"{name}_{j}"
            if ca in df.columns and cb in df.columns:
                df[cn] = df[ca] - df[cb]
    return df

# voraus-AD task-cycle action codes -> human-readable phase.
# Official `Action` IntEnum from vorausrobotik/voraus-ad-dataset (voraus_ad.py):
# odd codes are real moves, even codes are VOID (idle/transition between moves).
ACTION_NAMES = {
    0: "VOID_1",  1: "TO_SCAN_PTP",   2: "VOID_2",  3: "OVER_OBJECT_PTP",
    4: "VOID_3",  5: "DOWN_PICK_LIN", 6: "VOID_4",  7: "UP_PICK_LIN",
    8: "VOID_5",  9: "TO_PLACE_PTP", 10: "VOID_6", 11: "DOWN_PLACE_LIN",
    12: "VOID_7", 13: "UP_PLACE_LIN", 14: "VOID_8",
}


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
def _resample(run: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    """Resample a signal onto a normalized [0,1] task-progress grid."""
    t = run["time"].to_numpy()
    t = (t - t.min()) / (t.max() - t.min() + EPS)
    return np.interp(grid, t, run[col].to_numpy())


def build_reference(df_normal: pd.DataFrame, signal_cols: list[str],
                    grid: np.ndarray) -> dict:
    """Build the expected (normal) profile + scale for each signal.

    Aggregates over every normal run (grouped by 'sample'); with one run the
    std is ~0 so we fall back to a range-based scale (documented below).
    Returns {col: {"mean": arr, "scale": arr}}.
    """
    ref = {}
    samples = df_normal["sample"].unique()
    for col in signal_cols:
        stacks = np.vstack([
            _resample(df_normal[df_normal["sample"] == s].sort_values("time"), col, grid)
            for s in samples
        ])
        mean = stacks.mean(axis=0)
        std = stacks.std(axis=0)
        # Fallback when few normal runs: a robust scale based on signal range,
        # so z-like scores stay finite. Replace with real std on full data.
        rng = np.ptp(mean) if np.ptp(mean) > 0 else 1.0
        scale = np.maximum(std, 0.05 * rng)
        ref[col] = {"mean": mean, "scale": scale}
    return ref


# Cycle thirds used for the within-cycle step statistic. The windows travel with
# the record (see SIGNATURE_WINDOWS) so a consumer never has to guess them.
_EARLY = slice(0, GRID_N // 3)
_LATE = slice(2 * GRID_N // 3, GRID_N)

# The can is held between the end of DOWN_PICK_LIN and its release during
# DOWN_PLACE_LIN, so contrasting these two sets isolates payload-dependent load:
# a dropped part changes torque only while carrying, an added axis weight
# changes it throughout.
CARRY_ACTIONS = (7, 8, 9, 10, 11, 12)
EMPTY_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 13, 14)

# The cycle-level statistics the signature block carries, in slot order.
SIGNATURE_STATS = ("z_peak", "z_signed", "z_step", "z_payload", "z_std")
SIGNATURE_WINDOWS = {
    "grid_n": GRID_N,
    "early": [_EARLY.start, _EARLY.stop],
    "late": [_LATE.start, _LATE.stop],
    "carry_actions": list(CARRY_ACTIONS),
    "empty_actions": list(EMPTY_ACTIONS),
}


def attribute(run: pd.DataFrame, ref: dict, grid: np.ndarray) -> pd.DataFrame:
    """Per-signal attribution for one anomalous run vs the reference.

    Alongside the peak statistics this computes the cycle-level ones the
    representation used to discard: max|z| alone throws away the deviation's
    DIRECTION, its evolution WITHIN the cycle, its dependence on the PAYLOAD,
    and its dispersion. repair.py verified direction and step as the dimensions
    that separate the load-torque failure modes. All four are free -- the same z
    series is already computed here.

    Note the earlier, failed attempt these replace: the peak-INSTANT sign
    (observed-expected at the argmax|z| sample) is a single-sample statistic
    dominated by trajectory-timing deviations and carried no diagnostic signal.
    The cycle-level mean and the late-early step do.
    """
    rows = []
    # action phase aligned to the grid (nearest task-progress sample)
    t = run["time"].to_numpy()
    tn = (t - t.min()) / (t.max() - t.min() + EPS)
    act_on_grid = run["action"].to_numpy()[np.searchsorted(tn, grid, side="right") - 1]
    carry = np.isin(act_on_grid, CARRY_ACTIONS)
    empty = np.isin(act_on_grid, EMPTY_ACTIONS)

    for col, r in ref.items():
        obs = _resample(run, col, grid)
        dev = obs - r["mean"]
        z = dev / (r["scale"] + EPS)
        k = int(np.argmax(np.abs(z)))                 # peak-deviation index
        joint = int(col.split("_")[-1]) if col[-1].isdigit() else None
        family = col.rsplit("_", 1)[0] if joint else col
        rows.append({
            "signal": col,
            "joint": joint,
            "family": family,
            "peak_abs_delta": float(np.abs(dev[k])),
            "peak_z": float(np.abs(z[k])),
            # cycle-level statistics (the signature block)
            "signed_mean_z": float(np.mean(z)),
            "step_z": float(np.mean(z[_LATE]) - np.mean(z[_EARLY])),
            "payload_z": float((np.mean(z[carry]) if carry.any() else 0.0)
                               - (np.mean(z[empty]) if empty.any() else 0.0)),
            "std_z": float(np.std(z)),
            "observed_at_peak": float(obs[k]),
            "expected_at_peak": float(r["mean"][k]),
            "progress_pct": round(float(grid[k] * 100), 1),
            "time_s": round(float(run["time"].min() + grid[k] *
                                  (run["time"].max() - run["time"].min())), 3),
            "action_code": int(act_on_grid[k]),
            "cycle_phase": ACTION_NAMES.get(int(act_on_grid[k])),
        })
    return pd.DataFrame(rows).sort_values("peak_z", ascending=False).reset_index(drop=True)


def to_bridge_record(attr: pd.DataFrame, *, sample_id, category, setting=None,
                     anomaly_score=None, threshold=None, top_k=5) -> dict:
    """Assemble the structured JSON anomaly record (the LLM/KG contract).

    Joint localization is driven by the LOAD signals (torque families): a drive
    fault shows first in torque/current; position/velocity deviations are
    usually downstream consequences and are reported as supporting evidence.
    """
    LOAD = {"motor_torque", "torque_sensor_a"}
    load = attr[attr["joint"].notna() & attr["family"].isin(LOAD)]
    # z-based ranking (authoritative once a real normal-std envelope is used)
    per_joint_z = load.groupby("joint")["peak_z"].max().sort_values(ascending=False)
    # absolute-delta ranking (scale-free; physically grounded; robust to 1 normal run)
    per_joint_abs = (load.groupby("joint")["peak_abs_delta"].max()
                     .sort_values(ascending=False))
    dom_joint = int(per_joint_abs.index[0]) if len(per_joint_abs) else None
    dom_phase = (load.sort_values("peak_abs_delta", ascending=False).iloc[0]["cycle_phase"]
                 if len(load) else None)
    top = attr.head(top_k)

    return {
        # 0.2 put z_signed / z_step on each attributed signal; 0.3 replaced that
        # with the record-level `signature` block (see below) and dropped them,
        # because `attributed_signals` is a per-cycle selection and those two
        # statistics are only meaningful on a fixed slot set. Everything else is
        # additive: a 0.1 consumer still works unchanged.
        "schema_version": "0.3",
        "cycle_id": int(sample_id),
        "detector": {
            "verdict": "anomaly",
            "anomaly_score": anomaly_score,      # <- MVT-Flow NLL
            "threshold": threshold,              # <- calibrated on normal-val
            "ground_truth_category": int(category),   # voraus-AD label (eval only)
            # The physical fault instantiation this cycle came from (eval only).
            # Cycles sharing a setting are repeated runs of ONE configuration, so
            # any learned comparison must group on it or it leaks; carrying it
            # here keeps the leave-setting-out protocol reproducible from the
            # records alone. See repair.py.
            "ground_truth_setting": (None if setting is None else int(setting)),
        },
        "localization": {
            "dominant_joint": dom_joint,
            "dominant_cycle_phase": dom_phase,
            "joint_ranking_by_abs_delta": [
                {"joint": int(j), "peak_delta": round(float(v), 2)}
                for j, v in per_joint_abs.items()
            ],
            "joint_ranking_by_z": [
                {"joint": int(j), "peak_z": round(float(v), 2)}
                for j, v in per_joint_z.items()
            ],
        },
        "attributed_signals": [
            {
                "signal": row["signal"],
                "joint": (int(row["joint"]) if pd.notna(row["joint"]) else None),
                "family": row["family"],
                "observed": round(row["observed_at_peak"], 3),
                "expected": round(row["expected_at_peak"], 3),
                "delta": round(row["peak_abs_delta"], 3),
                "z": round(row["peak_z"], 2),
                "progress_pct": row["progress_pct"],
                "time_s": row["time_s"],
                "cycle_phase": row["cycle_phase"],
            }
            for _, row in top.iterrows()
        ],
        # ---- signature block (schema 0.3) --------------------------------
        # `attributed_signals` above is a per-cycle TOP-K SELECTION: which
        # signals appear differs from cycle to cycle, so its z_signed / z_step
        # cannot be compared across cycles. This block is the same statistics
        # over a FIXED slot set -- every signal the attribution reads, in the
        # same order for every record -- which is what makes a cross-cycle
        # signature well defined. repair.py consumes exactly this.
        #
        # Membership here is the definition of a "free" repair dimension: a
        # dimension is free when it can be computed from the record, costs a
        # pipeline change when it needs a channel the attribution does not read
        # (those never appear here), and costs hardware when the cell lacks the
        # channel entirely. Widening JOINT_SIGNALS therefore widens this block
        # automatically -- the cost classes are structural, not hand-assigned.
        "signature": {
            "stats": list(SIGNATURE_STATS),
            "windows": SIGNATURE_WINDOWS,
            "slots": {
                row["signal"]: [round(row["peak_z"], 3),
                                round(row["signed_mean_z"], 3),
                                round(row["step_z"], 3),
                                round(row["payload_z"], 3),
                                round(row["std_z"], 3)]
                for _, row in attr.sort_values("signal").iterrows()
            },
        },
        "_notes": ("Reference built from normal run(s) in slice; replace with "
                   "948-run envelope. anomaly_score/threshold pending MVT-Flow. "
                   "cycle_phase codes pending voraus-AD action-name mapping."),
    }


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "voraus_df.pkl"))
    signal_cols = ([f"{f}_{j}" for f in JOINT_SIGNALS for j in JOINTS]
                   + [c for c in ROBOT_SIGNALS if c in df.columns])
    grid = np.linspace(0, 1, GRID_N)

    ref = build_reference(df[df.category == 12], signal_cols, grid)

    for cat in (5, 1, 11):
        run = df[df.category == cat].sort_values("time")
        attr = attribute(run, ref, grid)
        rec = to_bridge_record(attr, sample_id=run["sample"].iloc[0],
                               category=cat, anomaly_score=None, threshold=None)
        print("=" * 70)
        print(f"ANOMALY CYCLE  (voraus-AD category #{cat})")
        print(json.dumps(rec, indent=2))
