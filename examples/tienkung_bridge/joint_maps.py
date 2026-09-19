"""Joint mapping utilities for Tienkung humanoid robot.

Maps between VLA model order and Tienkung ROS2 motor IDs, handling the distinction
between arm joints (controlled in radians via /arm/cmd_pos) and hand joints
(controlled in percentage 0-1 via /inspire_hand topics).
"""

import numpy as np

# ============================================================================
# Motor ID definitions (from Tienkung SDK)
# ============================================================================

# Left arm motor IDs (7 DOF)
LEFT_ARM_IDS = [11, 12, 13, 14, 15, 16, 17]

# Right arm motor IDs (7 DOF)
RIGHT_ARM_IDS = [21, 22, 23, 24, 25, 26, 27]

# Hand finger IDs (6 DOF per hand, same IDs for left and right)
# Controlled via separate topics with percentage values
HAND_FINGER_IDS = [1, 2, 3, 4, 5, 6]  # little, ring, middle, fore, thumb_bend, thumb_rotation

# ============================================================================
# VLA model ordering
# ============================================================================

# VLA expects 26-dim state/action: [l_arm(7), l_hand(6), r_arm(7), r_hand(6)]
VLA_LEFT_ARM_SLICE = slice(0, 7)
VLA_LEFT_HAND_SLICE = slice(7, 13)
VLA_RIGHT_ARM_SLICE = slice(13, 20)
VLA_RIGHT_HAND_SLICE = slice(20, 26)

# ============================================================================
# Hand conversion: VLA output <-> Percentage
# ============================================================================

# The model outputs hand joint values in radians (trained on gripper angles).
# Tienkung hands expect percentage in [0, 1] where:
#   0.0 = fully closed (grip)
#   1.0 = fully open
#
# Assuming the model's gripper output range is approximately [-2.25, -0.10] rad
# (matching rx101_bridge GRIPPER_OPEN_RAD / GRIPPER_CLOSED_RAD), we map:
#   model_rad <= GRIPPER_CLOSED_RAD  ->  0.0 (closed)
#   model_rad >= GRIPPER_OPEN_RAD    ->  1.0 (open)
#   linear interpolation in between

GRIPPER_OPEN_RAD = -0.10    # model output for "open"
GRIPPER_CLOSED_RAD = -2.25  # model output for "closed"


def hand_rad_to_percentage(hand_rad: np.ndarray) -> np.ndarray:
    """Convert model hand output (radians) to Tienkung percentage [0, 1].

    Args:
        hand_rad: (6,) array of hand joint values in radians

    Returns:
        (6,) array of percentages in [0, 1]
    """
    # Linear mapping: closed_rad -> 0.0, open_rad -> 1.0
    # percentage = (rad - closed) / (open - closed)
    percentage = (hand_rad - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    return np.clip(percentage, 0.0, 1.0)


def hand_percentage_to_rad(hand_pct: np.ndarray) -> np.ndarray:
    """Convert Tienkung hand percentage [0, 1] to model radians.

    Args:
        hand_pct: (6,) array of percentages in [0, 1]

    Returns:
        (6,) array of hand joint values in radians
    """
    # Inverse mapping
    return GRIPPER_CLOSED_RAD + hand_pct * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


# ============================================================================
# VLA action decomposition
# ============================================================================

def split_vla_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 26-dim VLA action into arm and hand components.

    Args:
        action: (26,) VLA action array

    Returns:
        (left_arm_rad, left_hand_rad, right_arm_rad, right_hand_rad)
        Each arm is (7,) in radians, each hand is (6,) in radians
    """
    return (
        action[VLA_LEFT_ARM_SLICE],
        action[VLA_LEFT_HAND_SLICE],
        action[VLA_RIGHT_ARM_SLICE],
        action[VLA_RIGHT_HAND_SLICE],
    )


# ============================================================================
# ROS2 command builders
# ============================================================================

def build_arm_position_msg(left_arm: np.ndarray, right_arm: np.ndarray,
                           speed: float = 0.5, current: float = 8.0):
    """Build /arm/cmd_pos message payload.

    Args:
        left_arm: (7,) left arm positions in radians
        right_arm: (7,) right arm positions in radians
        speed: desired speed in rad/s (default 0.5)
        current: max current in A (default 8.0)

    Returns:
        List of dicts ready for CmdSetMotorPosition.cmds
    """
    cmds = []
    for motor_id, pos in zip(LEFT_ARM_IDS, left_arm):
        cmds.append({"name": int(motor_id), "pos": float(pos), "spd": speed, "cur": current})
    for motor_id, pos in zip(RIGHT_ARM_IDS, right_arm):
        cmds.append({"name": int(motor_id), "pos": float(pos), "spd": speed, "cur": current})
    return cmds


def build_hand_position_msg(hand_rad: np.ndarray) -> dict:
    """Build /inspire_hand/ctrl/{left,right}_hand message payload.

    Args:
        hand_rad: (6,) hand joint values in radians (from model)

    Returns:
        Dict with 'name' and 'position' lists for JointState message
    """
    hand_pct = hand_rad_to_percentage(hand_rad)
    return {
        "name": [str(i) for i in HAND_FINGER_IDS],
        "position": hand_pct.tolist(),
    }


# ============================================================================
# State assembly for VLA input
# ============================================================================

def assemble_vla_state(arm_status: list, left_hand_state: dict, right_hand_state: dict) -> np.ndarray:
    """Assemble robot state into 26-dim VLA input.

    Args:
        arm_status: List of dicts from /arm/status with 'name' and 'pos' (radians)
        left_hand_state: Dict from /inspire_hand/state/left_hand with 'position' (percentage)
        right_hand_state: Dict from /inspire_hand/state/right_hand with 'position' (percentage)

    Returns:
        (26,) state array: [l_arm(7), l_hand(6), r_arm(7), r_hand(6)]
    """
    # Parse arm positions into a dict {motor_id: pos}
    arm_pos_map = {status["name"]: status["pos"] for status in arm_status}

    # Extract left/right arm in VLA order
    left_arm = np.array([arm_pos_map.get(mid, 0.0) for mid in LEFT_ARM_IDS], dtype=np.float32)
    right_arm = np.array([arm_pos_map.get(mid, 0.0) for mid in RIGHT_ARM_IDS], dtype=np.float32)

    # Convert hand percentages back to radians for VLA (model expects radians)
    left_hand_pct = np.array(left_hand_state["position"][:6], dtype=np.float32)
    right_hand_pct = np.array(right_hand_state["position"][:6], dtype=np.float32)
    left_hand = hand_percentage_to_rad(left_hand_pct)
    right_hand = hand_percentage_to_rad(right_hand_pct)

    return np.concatenate([left_arm, left_hand, right_arm, right_hand])
