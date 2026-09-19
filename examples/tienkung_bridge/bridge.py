"""VLA <-> Tienkung ROS2 bridge.

Connects OpenPI VLA model to Tienkung humanoid robot via ROS2 topics.

Architecture:
  - Subscribes to robot state: /arm/status, /inspire_hand/state/{left,right}_hand,
    /camera_head/image_raw/compressed
  - Publishes control commands: /arm/cmd_pos, /inspire_hand/ctrl/{left,right}_hand
  - Runs VLA inference via websocket_client_policy at configurable rate (default 10 Hz)
  - Optional action smoothing via exponential moving average

Control flow:
  1. Collect latest state (arm joints, hands, camera)
  2. Build observation dict for VLA (TienkungInputs format)
  3. Call VLA inference -> 26-dim action [l_arm(7), l_hand(6), r_arm(7), r_hand(6)]
  4. Optionally smooth action with EMA
  5. Split and publish: arms to /arm/cmd_pos, hands to /inspire_hand topics
"""

from __future__ import annotations

import argparse
import io
import logging
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from PIL import Image

from openpi_client import websocket_client_policy

try:
    from bodyctrl_msgs.msg import MotorStatusMsg, CmdSetMotorPosition, SetMotorPosition
except ImportError:
    MotorStatusMsg = None
    CmdSetMotorPosition = None
    SetMotorPosition = None
    logging.warning("bodyctrl_msgs not found. Install Tienkung ROS2 SDK or run with --mock")

from examples.tienkung_bridge import joint_maps


LOG = logging.getLogger("tienkung_bridge")

DEFAULT_PROMPT = "pick up the apple and place it"
DEFAULT_CONTROL_RATE = 10.0  # Hz
DEFAULT_SMOOTHING_ALPHA = 0.3  # EMA: output = alpha*new + (1-alpha)*prev
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


class TienkungBridge(Node):
    """ROS2 node bridging VLA inference to Tienkung robot control."""

    def __init__(
        self,
        vla_host: str,
        vla_port: int,
        prompt: str,
        control_rate: float = DEFAULT_CONTROL_RATE,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        arm_speed: float = 0.5,
        arm_current: float = 8.0,
        mock: bool = False,
    ):
        super().__init__("tienkung_vla_bridge")
        self.prompt = prompt
        self.control_rate = control_rate
        self.smoothing_alpha = smoothing_alpha
        self.arm_speed = arm_speed
        self.arm_current = arm_current
        self.mock = mock

        # State buffers (guarded by lock)
        self._lock = threading.Lock()
        self._arm_status: list[dict[str, Any]] = []
        self._left_hand_state: dict[str, Any] = {"position": [0.0] * 6}
        self._right_hand_state: dict[str, Any] = {"position": [0.0] * 6}
        self._camera_image: np.ndarray | None = None

        # Previous action for smoothing
        self._prev_action: np.ndarray | None = None

        # VLA policy
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=vla_host, port=vla_port)
        LOG.info("Connected to VLA server at %s:%d", vla_host, vla_port)

        # ROS2 subscribers
        self.sub_arm_status = self.create_subscription(
            MotorStatusMsg if not mock else type("MockMsg", (), {}),
            "/arm/status",
            self._on_arm_status,
            10,
        )
        self.sub_left_hand = self.create_subscription(
            JointState, "/inspire_hand/state/left_hand", self._on_left_hand_state, 10
        )
        self.sub_right_hand = self.create_subscription(
            JointState, "/inspire_hand/state/right_hand", self._on_right_hand_state, 10
        )
        self.sub_camera = self.create_subscription(
            CompressedImage, "/camera_head/image_raw/compressed", self._on_camera_image, 10
        )

        # ROS2 publishers
        self.pub_arm_cmd = self.create_publisher(
            CmdSetMotorPosition if not mock else type("MockMsg", (), {}),
            "/arm/cmd_pos",
            10,
        )
        self.pub_left_hand_cmd = self.create_publisher(
            JointState, "/inspire_hand/ctrl/left_hand", 10
        )
        self.pub_right_hand_cmd = self.create_publisher(
            JointState, "/inspire_hand/ctrl/right_hand", 10
        )

        # Control timer
        self.timer = self.create_timer(1.0 / control_rate, self._control_loop)
        LOG.info("Control loop started at %.1f Hz", control_rate)

    # ========================================================================
    # ROS2 callbacks
    # ========================================================================

    def _on_arm_status(self, msg) -> None:
        with self._lock:
            self._arm_status = [
                {"name": s.name, "pos": s.pos, "speed": s.speed, "current": s.current}
                for s in msg.status
            ]

    def _on_left_hand_state(self, msg: JointState) -> None:
        with self._lock:
            self._left_hand_state = {"name": msg.name, "position": msg.position}

    def _on_right_hand_state(self, msg: JointState) -> None:
        with self._lock:
            self._right_hand_state = {"name": msg.name, "position": msg.position}

    def _on_camera_image(self, msg: CompressedImage) -> None:
        try:
            img = Image.open(io.BytesIO(msg.data)).convert("RGB")
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
            img_array = np.asarray(img, dtype=np.uint8)
            with self._lock:
                self._camera_image = img_array
        except Exception as e:
            LOG.error("Failed to decode camera image: %s", e)

    # ========================================================================
    # Control loop
    # ========================================================================

    def _control_loop(self) -> None:
        """Main control loop: read state, infer, publish commands."""
        # Snapshot state
        with self._lock:
            arm_status = self._arm_status.copy()
            left_hand_state = self._left_hand_state.copy()
            right_hand_state = self._right_hand_state.copy()
            camera_image = self._camera_image

        # Check if we have all required data
        if not arm_status or camera_image is None:
            LOG.debug("Waiting for robot state (arm: %d, camera: %s)",
                     len(arm_status), camera_image is not None)
            return

        # Assemble VLA observation
        state = joint_maps.assemble_vla_state(arm_status, left_hand_state, right_hand_state)
        obs = {
            "state": state,
            "images": {"camera_head": camera_image},
            "prompt": self.prompt,
        }

        # VLA inference
        try:
            t0 = time.monotonic()
            output = self.policy.infer(obs)
            infer_ms = (time.monotonic() - t0) * 1000
            action = np.asarray(output["actions"], dtype=np.float32)
            if action.ndim == 2:
                action = action[0]  # take first timestep if action_horizon > 1
            if action.shape[0] < 26:
                LOG.error("VLA returned %d-dim action, expected >= 26", action.shape[0])
                return
            action = action[:26]
        except Exception as e:
            LOG.error("VLA inference failed: %s", e)
            return

        # Smoothing
        if self._prev_action is not None:
            action = (
                self.smoothing_alpha * action
                + (1.0 - self.smoothing_alpha) * self._prev_action
            )
        self._prev_action = action.copy()

        # Split action into components
        l_arm, l_hand, r_arm, r_hand = joint_maps.split_vla_action(action)

        # Publish arm command
        arm_cmds = joint_maps.build_arm_position_msg(
            l_arm, r_arm, speed=self.arm_speed, current=self.arm_current
        )
        if not self.mock:
            arm_msg = CmdSetMotorPosition()
            arm_msg.header.stamp = self.get_clock().now().to_msg()
            arm_msg.cmds = [
                SetMotorPosition(name=c["name"], pos=c["pos"], spd=c["spd"], cur=c["cur"])
                for c in arm_cmds
            ]
            self.pub_arm_cmd.publish(arm_msg)

        # Publish left hand command
        l_hand_msg_data = joint_maps.build_hand_position_msg(l_hand)
        l_hand_msg = JointState()
        l_hand_msg.header.stamp = self.get_clock().now().to_msg()
        l_hand_msg.name = l_hand_msg_data["name"]
        l_hand_msg.position = l_hand_msg_data["position"]
        self.pub_left_hand_cmd.publish(l_hand_msg)

        # Publish right hand command
        r_hand_msg_data = joint_maps.build_hand_position_msg(r_hand)
        r_hand_msg = JointState()
        r_hand_msg.header.stamp = self.get_clock().now().to_msg()
        r_hand_msg.name = r_hand_msg_data["name"]
        r_hand_msg.position = r_hand_msg_data["position"]
        self.pub_right_hand_cmd.publish(r_hand_msg)

        LOG.info(
            "Published commands: l_arm[0]=%.3f, r_arm[0]=%.3f, l_hand[0]=%.2f%%, infer=%.1fms",
            l_arm[0], r_arm[0], l_hand_msg_data["position"][0] * 100, infer_ms
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tienkung VLA bridge via ROS2")
    parser.add_argument("--vla-host", default="127.0.0.1", help="VLA server host")
    parser.add_argument("--vla-port", type=int, default=8000, help="VLA server port")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Task prompt for VLA")
    parser.add_argument("--rate", type=float, default=DEFAULT_CONTROL_RATE,
                        help="Control loop rate in Hz (default 10)")
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA,
                        help="EMA smoothing: output = alpha*new + (1-alpha)*prev (default 0.3)")
    parser.add_argument("--arm-speed", type=float, default=0.5,
                        help="Arm joint speed in rad/s (default 0.5)")
    parser.add_argument("--arm-current", type=float, default=8.0,
                        help="Arm joint max current in A (default 8.0)")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: skip bodyctrl_msgs import (for testing without robot)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    rclpy.init()
    try:
        bridge = TienkungBridge(
            vla_host=args.vla_host,
            vla_port=args.vla_port,
            prompt=args.prompt,
            control_rate=args.rate,
            smoothing_alpha=args.smoothing_alpha,
            arm_speed=args.arm_speed,
            arm_current=args.arm_current,
            mock=args.mock,
        )
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
