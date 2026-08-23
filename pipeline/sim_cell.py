# -*- coding: utf-8 -*-
"""A simulated pick-and-place cell (MuJoCo) — tier 2 of the fault-data plan.

WHY A SIMULATOR AT ALL, AND WHAT IT IS *NOT* FOR
------------------------------------------------
`synth_faults.py` (tier 1) manufactures fault instantiations by perturbing real
normal cycles. It cannot do two things, and those two things are exactly what is
still open in the evaluation:

  1. CONTACT. A collision is a contact event; you cannot inject one analytically
     without a contact model. The distal-collision cluster is the one the real
     dataset cannot decide, and the one whose diagnosis we revised (the loss is
     in the attribution layer, not the sensors -- see repair.py's control).
  2. A CHANNEL THE CELL DOES NOT HAVE. The commutation/overtemperature twin is
     classified as a HARDWARE cost class: undecidable because there is no
     temperature signal. That classification has never been tested, because you
     cannot test it on a robot that lacks the sensor.

This module is NOT a replication of the voraus cell, and nothing here may be
mixed with voraus-AD numbers:

  * The detector's AUROC must keep coming from voraus-AD. A simulated normal
    envelope has no gearbox backlash, no thermal drift and no real sensor noise,
    so any detection number computed here would be optimistic and meaningless.
  * The knowledge base describes a real production line. Simulation does not
    validate it.

What the simulator CAN test is a property of OUR PIPELINE, which is
robot-independent: given a fault whose signature is present in the raw signals,
does the deployed attribution-and-retrieval layer surface it, and does the cost
class we assigned to a repair behave as claimed? Those are mechanism tests, and
they are valid on a generic 6R arm.

THE CELL
--------
A 6R arm (base yaw / shoulder / elbow / wrist roll / wrist pitch / tool roll)
running the same seven-phase pick-and-place the voraus data uses, so the phase
codes and therefore attribution_bridge's carry/empty windows apply unchanged.
The payload is a body whose mass is switched at pick and release rather than a
simulated vacuum grip: grasping dynamics are not what is under test, and a mass
switch is exactly reproducible.

SIGNALS, and the modelling choice behind each
---------------------------------------------
  joint_position / joint_velocity  qpos / qvel, downsampled to 100 Hz
  motor_torque                     actuator force (motor side)
  torque_sensor_a                  actuator + constraint force at the joint
                                   (joint side: this is what sees a contact)
  motor_iq                         torque / Kt, plus current-loop noise
  joint_temperature                I^2 R heating with a per-joint time constant
                                   -- FAULT-AGNOSTIC: identical equations for
                                   every fault, so a fault cannot smuggle its
                                   answer into this channel
"""
from __future__ import annotations

import numpy as np

# 100 Hz to match voraus-AD; the solver runs faster and is decimated
SIM_DT = 0.002
LOG_HZ = 100
DECIM = int(round(1.0 / (LOG_HZ * SIM_DT)))
JOINTS = (1, 2, 3, 4, 5, 6)

# voraus action codes (attribution_bridge.ACTION_NAMES). Same seven moves, so
# CARRY_ACTIONS / EMPTY_ACTIONS carry over untouched.
PHASES = [
    ("TO_SCAN_PTP", 1, 1.0),
    ("OVER_OBJECT_PTP", 3, 1.0),
    ("DOWN_PICK_LIN", 5, 0.8),
    ("UP_PICK_LIN", 7, 0.8),
    ("TO_PLACE_PTP", 9, 1.4),
    ("DOWN_PLACE_LIN", 11, 0.8),
    ("UP_PLACE_LIN", 13, 0.8),
]

# joint waypoints (rad) at the start of each phase, plus the final pose
WAYPOINTS = np.array([
    [0.00, -0.30, 0.60, 0.0, 0.40, 0.0],     # home
    [0.35, -0.20, 0.50, 0.0, 0.30, 0.2],     # scan
    [0.70, -0.05, 0.35, 0.0, 0.20, 0.3],     # over object
    [0.70, 0.25, 0.15, 0.0, 0.10, 0.3],      # down to pick
    [0.70, -0.05, 0.35, 0.0, 0.20, 0.3],     # lift
    [-0.60, -0.05, 0.35, 0.0, 0.20, -0.3],   # transfer
    [-0.60, 0.25, 0.15, 0.0, 0.10, -0.3],    # down to place
    [-0.60, -0.20, 0.50, 0.0, 0.30, -0.2],   # retract
])

KT = np.array([0.9, 1.2, 0.9, 0.5, 0.4, 0.3])          # torque constant, N.m/A
R_WIND = np.array([0.8, 0.6, 0.8, 1.2, 1.4, 1.6])      # winding resistance, ohm
THERM_TAU = np.array([70.0, 90.0, 70.0, 45.0, 40.0, 35.0])   # thermal tau, s
# Heat capacity, J/degC. Chosen so a nominal duty cycle settles tens of degrees
# above ambient (dT_ss = tau * R * I^2 / C), i.e. the regime a thermal fault
# would actually live in. Same value for every fault -- see the module docstring.
THERM_C = np.array([900.0, 1400.0, 900.0, 320.0, 260.0, 200.0])
T_AMB = 25.0

# PD gains: high enough to track, low enough to stay well inside the actuator
# limits, so a fault's torque is not swamped by controller transients.
KP = np.array([90.0, 130.0, 80.0, 26.0, 20.0, 12.0])
KD = np.array([14.0, 20.0, 12.0, 4.0, 3.2, 2.0])


def _mjcf(payload_mass=0.0, obstacle=None):
    """Build the cell. `obstacle` = dict(pos, size, solref) or None.

    solref's first element is the contact time constant: a larger value is a
    softer, more compliant contact (foam), a smaller one is stiffer (carton).
    That single parameter is what distinguishes the two collision modes, which
    is the physical claim the FMEA makes about them.
    """
    obs = ""
    if obstacle is not None:
        px, py, pz = obstacle["pos"]
        sx, sy, sz = obstacle["size"]
        obs = (f'<body name="obstacle" pos="{px} {py} {pz}">'
               f'<geom name="obs" type="box" size="{sx} {sy} {sz}" rgba="0.8 0.4 0.2 1" '
               f'solref="{obstacle["solref"]} 1" solimp="0.9 0.95 0.001" '
               f'contype="1" conaffinity="1"/></body>')
    return f"""
<mujoco model="cell6r">
  <!-- MuJoCo defaults to angle="degree"; without this every joint range below
       is read as degrees and the arm is locked inside +/-3 deg. -->
  <compiler angle="radian"/>
  <option timestep="{SIM_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="2.0" armature="0.05"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.01 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.05" contype="0" conaffinity="0"/>
    {obs}
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.1 3.1"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0 0 0.35" size="0.055" mass="6"/>
      <body name="l2" pos="0 0 0.35">
        <joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.35 0 0" size="0.05" mass="5"/>
        <body name="l3" pos="0.35 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-2.6 2.6"/>
          <geom name="g3" type="capsule" fromto="0 0 0 0.30 0 0" size="0.042" mass="3.2"/>
          <body name="l4" pos="0.30 0 0">
            <joint name="j4" type="hinge" axis="1 0 0" range="-3.1 3.1"/>
            <geom name="g4" type="capsule" fromto="0 0 0 0.10 0 0" size="0.036" mass="1.3"/>
            <body name="l5" pos="0.10 0 0">
              <joint name="j5" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
              <geom name="g5" type="capsule" fromto="0 0 0 0.09 0 0" size="0.032" mass="0.9"/>
              <body name="l6" pos="0.09 0 0">
                <joint name="j6" type="hinge" axis="1 0 0" range="-3.1 3.1"/>
                <geom name="g6" type="capsule" fromto="0 0 0 0.06 0 0" size="0.028" mass="0.5"/>
                <body name="payload" pos="0.09 0 0">
                  <geom name="pay" type="sphere" size="0.035" mass="{max(payload_mass, 1e-6)}"
                        contype="0" conaffinity="0" rgba="0.2 0.6 0.9 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1" ctrlrange="-400 400"/>
    <motor joint="j2" gear="1" ctrlrange="-400 400"/>
    <motor joint="j3" gear="1" ctrlrange="-250 250"/>
    <motor joint="j4" gear="1" ctrlrange="-80 80"/>
    <motor joint="j5" gear="1" ctrlrange="-60 60"/>
    <motor joint="j6" gear="1" ctrlrange="-40 40"/>
  </actuator>
</mujoco>"""


def _trajectory(rng=None, speed=1.0):
    """Per-phase minimum-jerk interpolation between the waypoints -> (q, action)."""
    q_all, act_all = [], []
    for i, (_name, code, dur) in enumerate(PHASES):
        n = max(2, int(round(dur / speed / SIM_DT)))
        s = np.linspace(0.0, 1.0, n)
        blend = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5          # min-jerk
        a, b = WAYPOINTS[i], WAYPOINTS[i + 1]
        q_all.append(a + (b - a) * blend[:, None])
        act_all.append(np.full(n, code, dtype=int))
    return np.vstack(q_all), np.concatenate(act_all)


class Cell:
    """One simulated robot cell. `fault` selects the perturbation, if any."""

    def __init__(self, payload_mass=1.2, obstacle=None, fault=None, seed=0):
        import mujoco
        self.mj = mujoco
        self.fault = fault or {}
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(
            _mjcf(payload_mass=payload_mass, obstacle=obstacle))
        self.data = mujoco.MjData(self.model)
        self.payload_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.payload_mass = payload_mass

    # -- fault hooks ------------------------------------------------------
    def _kt(self):
        """Effective torque constant. FM_overtemp is 'current rise for the same
        torque, thermal derating', i.e. Kt drops -- so the SAME mechanical
        torque is drawn at a HIGHER current, which in turn dissipates more heat
        through the same fault-agnostic thermal equations."""
        kt = KT.copy()
        if self.fault.get("kind") == "derate":
            for j in self.fault["joints"]:
                kt[j - 1] *= (1.0 - self.fault["kt_loss"])
        return kt

    def _extra_torque(self, k, q, qd, t, action):
        """Additive joint torque from the active fault, in N.m."""
        f = self.fault
        kind = f.get("kind")
        if kind == "friction":            # FM_friction: elevated, motion-opposing
            g = np.zeros(6)
            for j in f["joints"]:
                g[j - 1] = -f["coulomb"] * np.tanh(qd[j - 1] / 0.05)
            return g
        if kind == "commutation":         # FM_miscommutation: torque ripple, zero mean
            g = np.zeros(6)
            for j in f["joints"]:
                g[j - 1] = f["amp"] * np.sin(f["poles"] * q[j - 1] + f["phase"])
            return g
        return np.zeros(6)

    def run(self, speed=1.0, noise=True, noise_sd=1.0, jitter=None):
        """Simulate one cycle; return a dict of 100 Hz signal arrays."""
        mj = self.mj
        m, d = self.model, self.data
        q_ref, act = _trajectory(speed=speed)
        # reference velocity for the derivative term
        qd_ref = np.vstack([np.zeros((1, 6)), np.diff(q_ref, axis=0) / SIM_DT])
        m.body_mass[self.payload_bid] = 1e-6            # start empty
        d.qpos[:] = q_ref[0]
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)

        temp = np.full(6, T_AMB)
        rows = []
        self.n_contact_steps = 0
        # Per-cycle systematic variation. A real line varies between cycles in
        # ways a bare simulation does not (joint friction, gain drift, payload
        # seating), and without it the normal envelope is so tight that every
        # fault lands at hundreds of sigma and NOTHING is ever discarded by the
        # top-k selection -- which would make any test of that selection vacuous.
        jit = jitter if jitter is not None else np.zeros(6)
        drop_at = self.fault.get("drop_at")             # FM_losecan, fraction of carry
        carry_seen = 0
        n_carry = int(np.isin(act, [7, 8, 9, 10, 11, 12]).sum())

        for i in range(len(q_ref)):
            a = int(act[i])
            carrying = a in (7, 8, 9, 10, 11, 12)
            if carrying:
                carry_seen += 1
                lost = drop_at is not None and carry_seen > drop_at * max(n_carry, 1)
                m.body_mass[self.payload_bid] = 1e-6 if lost else self.payload_mass
            else:
                m.body_mass[self.payload_bid] = 1e-6
            if a in (5, 6):                              # released until pick completes
                m.body_mass[self.payload_bid] = 1e-6

            # Computed-torque control, as a real cobot does it: gravity and
            # Coriolis feedforward (qfrc_bias) plus a PD correction. Without the
            # feedforward, soft gains lag by ~0.6 rad and the commanded phase
            # structure stops matching the actual motion.
            e = q_ref[i] - d.qpos[:6]
            ed = qd_ref[i] - d.qvel[:6]
            tau = d.qfrc_bias[:6] + KP * e + KD * ed
            tau = tau + jit * np.tanh(d.qvel[:6] / 0.05)   # cycle friction offset
            tau = tau + self._extra_torque(None, d.qpos[:6].copy(), d.qvel[:6].copy(),
                                           i * SIM_DT, a)
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mj.mj_step(m, d)
            if d.ncon:
                self.n_contact_steps += 1

            # thermal state: identical equations for every fault. Only Kt
            # differs, and only for the derating mode, which is what that mode
            # physically IS.
            cur = d.actuator_force[:6] / self._kt()
            temp += SIM_DT * (R_WIND * cur ** 2 / THERM_C
                              - (temp - T_AMB) / THERM_TAU)

            if i % DECIM == 0:
                mot = d.actuator_force[:6].copy()
                joint_side = mot + d.qfrc_constraint[:6]     # contact shows up here
                iq = mot / self._kt()
                if noise:
                    mot = mot + self.rng.normal(0, 0.03 * noise_sd, 6)
                    joint_side = joint_side + self.rng.normal(0, 0.02 * noise_sd, 6)
                    iq = iq + self.rng.normal(0, 0.01 * noise_sd, 6)
                rows.append(dict(
                    time=i * SIM_DT, action=a,
                    qpos=d.qpos[:6].copy(), qvel=d.qvel[:6].copy(),
                    motor_torque=mot, torque_sensor_a=joint_side,
                    motor_iq=iq, temperature=temp.copy()))
        return _to_columns(rows)


def _to_columns(rows):
    out = {"time": np.array([r["time"] for r in rows]),
           "action": np.array([r["action"] for r in rows])}
    for name, key in (("joint_position", "qpos"), ("joint_velocity", "qvel"),
                      ("motor_torque", "motor_torque"),
                      ("torque_sensor_a", "torque_sensor_a"),
                      ("motor_iq", "motor_iq"),
                      ("joint_temperature", "temperature")):
        arr = np.array([r[key] for r in rows])
        for k, j in enumerate(JOINTS):
            out[f"{name}_{j}"] = arr[:, k]
    return out


def to_frame(sig, sample_id):
    """Signal dict -> the DataFrame layout attribution_bridge expects."""
    import pandas as pd
    df = pd.DataFrame(sig)
    df["sample"] = sample_id
    return df


if __name__ == "__main__":
    c = Cell(payload_mass=1.2, seed=0)
    s = c.run()
    n = len(s["time"])
    print(f"one cycle: {n} samples at {LOG_HZ} Hz "
          f"({s['time'][-1]:.2f} s), phases {sorted(set(s['action'].tolist()))}")
    for j in (1, 2, 3):
        print(f"  J{j} torque  mean {s[f'motor_torque_{j}'].mean():+8.2f}"
              f"  sd {s[f'motor_torque_{j}'].std():6.2f}"
              f" | temp end {s[f'joint_temperature_{j}'][-1]:.2f} C")
    err = max(abs(s[f"joint_position_{j}"] - 0).max() for j in (1,))
    print("tracking sanity: |j1| max", round(float(err), 3))
