"""Joint order mappings between the pi0.5 rx101 VLA model output and SONIC teleop reference.

Both spaces use rad, absolute joint angles, same zero pose. Only the ordering differs.

VLA_ORDER: from the rx101_blackbox LeRobot dataset's observation.state / action.wbc names
(see dataset meta/info.json). Layout is l_leg[6] + r_leg[6] + waist[3] + l_arm[6] + r_arm[6].

SONIC_POLICY_ORDER: from ~/Tools/docs/vla-on-sonic-deploy.md §2.4 —
joint_names_policy_order, an IsaacLab BFS order that interleaves left/right/waist at each level.
"""

from __future__ import annotations

import numpy as np

VLA_ORDER: tuple[str, ...] = (
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_roll", "l_wrist_pitch",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch",
)

SONIC_POLICY_ORDER: tuple[str, ...] = (
    "l_hip_pitch", "r_hip_pitch", "waist_yaw",
    "l_hip_roll",  "r_hip_roll",  "waist_roll",
    "l_hip_yaw",   "r_hip_yaw",   "waist_pitch",
    "l_knee", "r_knee", "l_shoulder_pitch",
    "r_shoulder_pitch", "l_ankle_pitch", "r_ankle_pitch",
    "l_shoulder_roll", "r_shoulder_roll", "l_ankle_roll",
    "r_ankle_roll", "l_shoulder_yaw", "r_shoulder_yaw",
    "l_elbow", "r_elbow", "l_wrist_roll",
    "r_wrist_roll", "l_wrist_pitch", "r_wrist_pitch",
)

# MCU wire order (30 channels). From md_control_rx configs `mcu_channel_names`.
# Channels 27-29 are head_0/head_1/reserved — not used by the VLA body state.
MCU_ORDER: tuple[str, ...] = (
    "waist_yaw", "waist_roll", "waist_pitch",                              # 0..2
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",               # 3..5
    "r_elbow", "r_wrist_roll", "r_wrist_pitch",                            # 6..8
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",               # 9..11
    "l_elbow", "l_wrist_roll", "l_wrist_pitch",                            # 12..14
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw",                              # 15..17
    "r_knee", "r_ankle_pitch", "r_ankle_roll",                             # 18..20
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw",                              # 21..23
    "l_knee", "l_ankle_pitch", "l_ankle_roll",                             # 24..26
    "head_0", "head_1", "reserved",                                        # 27..29
)

assert set(VLA_ORDER) == set(SONIC_POLICY_ORDER), "joint sets differ"
assert len(VLA_ORDER) == 27 and len(SONIC_POLICY_ORDER) == 27
assert len(MCU_ORDER) == 30
assert set(MCU_ORDER[:27]) == set(VLA_ORDER), "MCU body channels differ from VLA"

# RX2 29-dof body+head order. Matches `observation.state` / `action.wbc` feature names
# for pi05_rx2_blackbox_drawer_v2 (dataset meta/info.json, confirmed 2026-09-03): the
# first 27 dims are IDENTICAL in name and order to VLA_ORDER above; head_yaw/head_pitch
# are appended, not interleaved into the body. head does NOT go through the SONIC
# encoder — no SONIC config anywhere puts head in joint_names_pnc_order — it routes to
# the MCU's direct head-control channels (PicoAuxGate) instead. See
# examples/rx101_bridge/README.md §"RX2 head/gripper direct control".
RX2_VLA_ORDER: tuple[str, ...] = VLA_ORDER + ("head_yaw", "head_pitch")
assert len(RX2_VLA_ORDER) == 29
assert RX2_VLA_ORDER[:27] == VLA_ORDER


def _permutation(src: tuple[str, ...], dst: tuple[str, ...]) -> np.ndarray:
    """Return idx such that dst_vec = src_vec[idx]."""
    src_pos = {name: i for i, name in enumerate(src)}
    return np.array([src_pos[name] for name in dst], dtype=np.int64)


# Static, computed once. Use with `q_sonic = q_vla[VLA_TO_SONIC]` for a single vector,
# or `q_sonic = q_vla[..., VLA_TO_SONIC]` for a batched (T,27) tensor.
VLA_TO_SONIC: np.ndarray = _permutation(VLA_ORDER, SONIC_POLICY_ORDER)
SONIC_TO_VLA: np.ndarray = _permutation(SONIC_POLICY_ORDER, VLA_ORDER)

# Body 27 subset. mcu_pos_f[MCU_BODY_TO_VLA] -> VLA-ordered 27-vec.
# Skips head_0/head_1/reserved at MCU indices 27..29.
MCU_BODY_TO_VLA: np.ndarray = np.array(
    [MCU_ORDER.index(name) for name in VLA_ORDER], dtype=np.int64
)


def vla_to_sonic(q_vla: np.ndarray) -> np.ndarray:
    """Reorder VLA-ordered joint vector(s) to SONIC policy order. Last axis is 27."""
    if q_vla.shape[-1] != 27:
        raise ValueError(f"expected last-dim=27, got shape {q_vla.shape}")
    return q_vla[..., VLA_TO_SONIC]


def sonic_to_vla(q_sonic: np.ndarray) -> np.ndarray:
    """Inverse of vla_to_sonic (for reading robot state back into VLA-order for obs)."""
    if q_sonic.shape[-1] != 27:
        raise ValueError(f"expected last-dim=27, got shape {q_sonic.shape}")
    return q_sonic[..., SONIC_TO_VLA]
