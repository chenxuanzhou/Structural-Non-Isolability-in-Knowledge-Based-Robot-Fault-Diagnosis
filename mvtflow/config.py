"""MVT-Flow reproduction -- configuration."""
from dataclasses import dataclass, field

# Signal families x joints used as flow input channels. The 46-ch measured set is
# the ACTIVE detector (AUROC 0.854). NEGATIVE RESULT (2026-06-07): expanding to the
# FULL 130 signals (all 21 per-joint families inc. target_*/computed_* commanded
# values) HURT the light CPU flow -> AUROC 0.854→0.713 (curse of dimensionality +
# redundant commanded channels; the paper's 0.936 needs the FrEIA flow + GPU). The
# 130-ch checkpoint is kept at checkpoints/mvtflow_130ch.bak.pt for the record.
JOINTS = [1, 2, 3, 4, 5, 6]
FAMILIES = ["motor_torque", "torque_sensor_a", "torque_sensor_b",
            "motor_iq", "motor_id", "joint_position", "joint_velocity"]
ROBOT_SIGNALS = ["robot_current", "system_current", "io_current", "robot_voltage"]


def default_channels():
    cols = [f"{fam}_{j}" for fam in FAMILIES for j in JOINTS]
    return cols + ROBOT_SIGNALS  # robot-level appended last


@dataclass
class Config:
    parquet_path: str = "../data/voraus-ad-dataset-100hz.parquet"   # full voraus-AD (100 Hz)
    normal_category: int = 12                 # voraus-AD: 12 == anomaly-free
    train_setting: int = 72                   # Variant.PRE_A: official train-normal split
    channels: list = field(default_factory=default_channels)
    seq_len: int = 1024                       # fixed length (resample per cycle)
    n_blocks: int = 4                         # paper: 4 coupling blocks
    hidden: int = 128
    clamp: float = 2.0
    batch_size: int = 32
    epochs: int = 60
    lr: float = 2e-4
    val_frac: float = 0.2
    device: str = "cuda"                      # "cpu" if no GPU
    seed: int = 0
    out_dir: str = "checkpoints"
