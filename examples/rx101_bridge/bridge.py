"""VLA <-> SONIC (encoder mode 0, ZMQ Protocol v1) bridge for rx_p2.

Dual-rate loop:
  - Action stream (~30 Hz):  wraps the VLA server in `ActionChunkBroker`. Each
    broker.infer(obs) returns ONE frame (29-dim = 27 body + 2 gripper); a real
    VLA call happens every `action_horizon` frames (default 16 = 533 ms), the
    remaining calls are dict-slices from the cache. The VLA-call tick blocks
    ~300 ms; the publisher keeps emitting the last cached frame across the gap.
  - Publish (50 Hz):  read the latest cached single-frame action, apply a
    low-pass filter (output = alpha*action + (1-alpha)*prev), permute VLA -> SONIC
    BFS order, publish a ZMQ pose frame for md_control_rx to ingest into
    SonicRobotMotionBuffer. alpha=0.3 default (~56 ms time constant at 50 Hz).

Robot state feed is real: shm proprio ring (rx_control_fb_raw + aux_feedback)
and HTTP MJPEG (cam8750). See state_source_shm.py and camera_source_http.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import logging
import threading
import time
from typing import Protocol

import numpy as np
from PIL import Image

from openpi_client import websocket_client_policy
from openpi_client.action_chunk_broker import ActionChunkBroker

from examples.rx101_bridge import joint_maps
from examples.rx101_bridge import zmq_pose


# Timing constants
VLA_HORIZON_STEPS = 16                    # model config `action_horizon`
VLA_DT_S = 1.0 / 30.0                     # dataset fps=30 -> 33.33 ms per step
SONIC_DT_S = 0.02                         # SONIC control_dt, 50 Hz publish
STALE_ACTION_AFTER_S = 1.0                # freeze pose (not stop publishing) if no fresh action.
                                          # Must stay below SONIC's teleop_lost_timeout_s (1.0 s).
DEFAULT_SMOOTHING_ALPHA = 0.3             # low-pass filter: output = alpha*plan + (1-alpha)*prev.
                                          # Smaller alpha => more smoothing, more lag.
                                          # 0.3 @ 50 Hz => ~56 ms time constant.
ACTION_STREAM_DT_S = VLA_DT_S             # broker steps one frame per this interval (~30 Hz).
DEFAULT_STAND_VLA = np.array(
    # Roughly matches the SONIC yaml `default_pos`, translated from SONIC BFS order back to VLA order.
    # This is used before any plan arrives and during "hold on stale" episodes.
    [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,        # l_leg (VLA[0..5])
     -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,        # r_leg (VLA[6..11])
      0.0, 0.0, 0.0,                        # waist (VLA[12..14])
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # l_arm (VLA[15..20])
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],        # r_arm (VLA[21..26])
    dtype=np.float32,
)
DEFAULT_HEAD_YAW_PITCH = np.zeros(2, dtype=np.float32)  # neutral head, matches PICO's zero_hold

# From the robot's rx_p2_sonic_gmr_5N_step023800_real.yaml `pico_aux.head.sign: [1.0,-1.0]`
# ([yaw,pitch]). PicoAuxGate multiplies the wire value by this sign internally before
# clamping/rate-limiting to a motor target. We PRE-multiply by the SAME sign before
# sending so sign² = 1 cancels — the wire value we choose IS the desired absolute
# MCU-frame head target, independent of whatever "measured" meant on PICO's own path.
# If the robot's yaml `pico_aux.head.sign` ever changes, this constant MUST change to
# match, or head pitch will drive to the WRONG extreme. Verify with the static
# self-check (README.md §"RX2 head/gripper direct control") before any live VLA test.
HEAD_SIGN_YAW_PITCH = np.array([1.0, -1.0], dtype=np.float32)

# From the same yaml `pico_aux.grippers`. PicoAuxGate only supports binary open/close
# (gripper_closed_mask) — VLA's continuous gripper output must be thresholded.
GRIPPER_OPEN_RAD = -2.25
GRIPPER_CLOSED_RAD = -0.10
GRIPPER_CLOSE_THRESHOLD_RAD = (GRIPPER_OPEN_RAD + GRIPPER_CLOSED_RAD) / 2.0  # -1.175

LOG = logging.getLogger("rx101_bridge")


@dataclasses.dataclass(frozen=True)
class DeployProfile:
    """Selects VLA <-> wire routing for a checkpoint's body layout.

    Both profiles share the SAME 27-dof body ordering/semantics (joint_maps.VLA_ORDER) —
    only head presence differs.
      rx101: state=27 (body only).       model action_dim=29 (body[27] + grip[2]).
      rx2:   state=29 (body[27]+head[2]). model action_dim=31 (body[27]+head[2]+grip[2]).
    """
    name: str
    state_dim: int
    action_dim: int
    has_head: bool


PROFILE_RX101 = DeployProfile(name="rx101", state_dim=27, action_dim=29, has_head=False)
PROFILE_RX2 = DeployProfile(name="rx2", state_dim=29, action_dim=31, has_head=True)
PROFILES: dict[str, DeployProfile] = {"rx101": PROFILE_RX101, "rx2": PROFILE_RX2}


@dataclasses.dataclass(frozen=True)
class RobotState:
    joint_pos: np.ndarray            # (27,) VLA order, rad
    head_yaw_pitch: np.ndarray       # (2,) rad, raw MCU feedback. rx101 profiles ignore this.
    left_gripper_pos: float
    right_gripper_pos: float
    root_quat_wxyz: np.ndarray       # (4,) base_link orientation, wxyz
    timestamp_ns: int


class StateSource(Protocol):
    def read(self) -> RobotState: ...
    def read_ego_jpeg(self) -> bytes: ...


class MockStateSource:
    """Fake robot: default-stand VLA-order pose, blank JPEG, identity root quat."""

    _MIN_JPEG = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb0043000806060706050807"
        "0707090909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e272022"
        "2c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101"
        "011100ffc4001f0000010501010101010100000000000000000102030405060708"
        "090a0bffc4001500010100000000000000000000000000000000ffda0008010100"
        "003f00d2cf20ffd9"
    )

    def read(self) -> RobotState:
        return RobotState(
            joint_pos=DEFAULT_STAND_VLA.copy(),
            head_yaw_pitch=DEFAULT_HEAD_YAW_PITCH.copy(),
            left_gripper_pos=-2.25,
            right_gripper_pos=-2.25,
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            timestamp_ns=time.monotonic_ns(),
        )

    def read_ego_jpeg(self) -> bytes:
        return self._MIN_JPEG


class RealStateSource:
    """On-robot composite source: shm proprio ring + HTTP MJPEG ego camera.

    Uses two dedicated helpers:
      * ShmStateSource - mmap read of md_bus_v2__data_rx__control_fb__raw with
        seqlock guard, plus grippers from ..._aux__feedback. Returns joint
        positions already permuted to VLA order.
      * MjpegEgoCamera - background thread pulling cam8750 MJPEG, exposes
        latest JPEG bytes downsized to dataset ego_view shape (480x640).

    Anchors the RobotState timestamp on `time.monotonic_ns()` so downstream
    plan/publish loops all share one clock. The raw shm timestamp is discarded
    for bridge use; it's only needed for external logging.
    """

    def __init__(self,
                 mjpeg_url: str | None = None,
                 fb_raw_path: str | None = None,
                 aux_fb_path: str | None = None):
        from examples.rx101_bridge.state_source_shm import ShmStateSource
        from examples.rx101_bridge.camera_source_http import (
            MjpegEgoCamera, CAM8750_MJPEG_URL,
        )
        self._shm = (
            ShmStateSource(fb_raw_path or ShmStateSource.__init__.__defaults__[0],
                           aux_fb_path or ShmStateSource.__init__.__defaults__[1])
            if fb_raw_path or aux_fb_path else ShmStateSource()
        )
        self._cam = MjpegEgoCamera(url=mjpeg_url or CAM8750_MJPEG_URL).start()

    def close(self) -> None:
        try:
            self._cam.close()
        finally:
            self._shm.close()

    def read(self) -> RobotState:
        snap = self._shm.read()
        return RobotState(
            joint_pos=snap.joint_pos_vla,
            head_yaw_pitch=snap.head_yaw_pitch,
            left_gripper_pos=snap.left_gripper_pos,
            right_gripper_pos=snap.right_gripper_pos,
            root_quat_wxyz=snap.root_quat_wxyz.astype(np.float32),
            timestamp_ns=time.monotonic_ns(),   # shared clock with publish/plan loops
        )

    def read_ego_jpeg(self) -> bytes:
        return self._cam.read_ego_jpeg()


def _decode_jpeg_hwc(jpeg_bytes: bytes, height: int = 480, width: int = 640) -> np.ndarray:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _build_obs(state: RobotState, jpeg: bytes, prompt: str, profile: DeployProfile) -> dict:
    """Rx101Inputs contract (see src/openpi/policies/rx101_policy.py).

    `state` width matches the checkpoint's `observation.state` feature: 27 (body
    only) for rx101, or 29 (body[27]+head[2]) for rx2. Gripper is never part of
    `state` — the model concatenates left/right gripper server-side.
    """
    if profile.has_head:
        model_state = np.concatenate(
            [state.joint_pos, state.head_yaw_pitch]
        ).astype(np.float32)
    else:
        model_state = state.joint_pos.astype(np.float32)
    return {
        "state": model_state,
        "left_gripper_state": np.array([state.left_gripper_pos], dtype=np.float32),
        "right_gripper_state": np.array([state.right_gripper_pos], dtype=np.float32),
        "images": {"ego_view": _decode_jpeg_hwc(jpeg)},
        "prompt": prompt,
    }


class Bridge:
    """Dual-rate bridge with ActionChunkBroker for chunk lifecycle and low-pass
    filtering on published pose.

    Architecture:
      * `action_stream_loop` (~30 Hz): calls broker.infer(obs).  broker caches a
        16-step (29-dim) chunk from a real VLA infer; each subsequent call
        returns one frame from the cache without hitting VLA.  Every 16 calls
        (~533 ms) broker's cache exhausts and the NEXT call blocks for a real
        VLA infer (~300 ms).  During that block, publish_loop keeps re-publishing
        the last cached frame — publisher never drops out.
      * `publish_loop` (50 Hz): reads the latest single-frame action, low-pass
        filters against the previous published pose, and emits a ZMQ frame.

    Smoothing:
        published = alpha * new_action + (1 - alpha) * previous_published
      alpha=0.3 (default) gives ~56 ms time constant at 50 Hz.
    """

    def __init__(
        self,
        *,
        vla_host: str,
        vla_port: int,
        prompt: str,
        state_source: StateSource,
        zmq_bind: str,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        action_horizon: int = VLA_HORIZON_STEPS,
        profile: DeployProfile = PROFILE_RX101,
    ) -> None:
        self._state = state_source
        self._prompt = prompt
        self._profile = profile
        raw_policy = websocket_client_policy.WebsocketClientPolicy(host=vla_host, port=vla_port)
        # ActionChunkBroker: infer(obs) returns a SINGLE frame (dict, sliced along
        # the first dim of the underlying chunk).  A real VLA call happens every
        # `action_horizon` calls; the rest are dict-slices from the cache.
        self._broker = ActionChunkBroker(raw_policy, action_horizon=action_horizon)
        self._publisher = zmq_pose.PosePublisher(bind_addr=zmq_bind)
        self._alpha = float(smoothing_alpha)
        if not (0.0 < self._alpha <= 1.0):
            raise ValueError(f"smoothing_alpha must be in (0,1], got {self._alpha}")

        # Latest action (profile.action_dim: body+[head]+gripper in VLA order) shared
        # between threads.
        self._latest_action: np.ndarray | None = None
        self._latest_action_ns: int = 0
        self._action_lock = threading.Lock()

        self._prev_pub_sonic: np.ndarray | None = None   # for low-pass + finite-diff
        self._stop = threading.Event()
        self._seq = 0
        self._default_sonic = joint_maps.vla_to_sonic(DEFAULT_STAND_VLA)

    # ---- publisher: 50 Hz, timing-critical ----
    def publish_loop(self) -> None:
        LOG.info("publish loop start (50 Hz -> ZMQ; alpha=%.2f; profile=%s)",
                 self._alpha, self._profile.name)
        period_ns = int(SONIC_DT_S * 1e9)
        next_tick = time.monotonic_ns()
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()

            # Snapshot latest action; fall back to default stand if stale/absent.
            with self._action_lock:
                if (self._latest_action is not None
                    and (now_ns - self._latest_action_ns) / 1e9 < STALE_ACTION_AFTER_S):
                    action = self._latest_action.copy()
                else:
                    action = None
            grip_enable_mask = 0
            grip_closed_mask = 0
            if action is None:
                body_vla = DEFAULT_STAND_VLA
                head_yp = DEFAULT_HEAD_YAW_PITCH
                lg = rg = -2.25
            else:
                body_vla = action[:27].astype(np.float32)
                if self._profile.has_head:
                    head_yp = action[27:29].astype(np.float32)
                    lg = float(action[29])
                    rg = float(action[30])
                    # Both grippers are always "enabled" for external control while
                    # a fresh robot_stream action is driving — PicoAuxGate's
                    # per-side closed_mask bit selects open_rad vs closed_rad.
                    grip_enable_mask = 0x03
                    grip_closed_mask = (
                        (0x01 if lg > GRIPPER_CLOSE_THRESHOLD_RAD else 0)
                        | (0x02 if rg > GRIPPER_CLOSE_THRESHOLD_RAD else 0)
                    )
                else:
                    head_yp = DEFAULT_HEAD_YAW_PITCH
                    lg = float(action[27])
                    rg = float(action[28])

            body_sonic = joint_maps.vla_to_sonic(body_vla).astype(np.float32)
            # Low-pass filter against previously published SONIC-order pose. Damps
            # step discontinuities at chunk boundaries and any VLA jitter.
            if self._prev_pub_sonic is not None:
                body_sonic = (self._alpha * body_sonic
                              + (1.0 - self._alpha) * self._prev_pub_sonic)
            # Velocity is derivative of what we PUBLISH (filtered), not of raw VLA output.
            if self._prev_pub_sonic is None:
                joint_vel_sonic = np.zeros(27, dtype=np.float32)
            else:
                joint_vel_sonic = ((body_sonic - self._prev_pub_sonic) / SONIC_DT_S
                                   ).astype(np.float32)
            self._prev_pub_sonic = body_sonic.copy()

            payload = {
                "joint_pos":   body_sonic[None, :],                             # (1, 27)
                "joint_vel":   joint_vel_sonic[None, :],                        # (1, 27)
                "body_quat":   self._state.read().root_quat_wxyz[None, :],      # (1, 4) wxyz
                "frame_index": np.array([self._seq], dtype=np.int64),           # (1,)
            }
            if self._profile.has_head:
                # Sign-cancellation trick — see HEAD_SIGN_YAW_PITCH comment above.
                head_wire = (HEAD_SIGN_YAW_PITCH * head_yp).astype(np.float32)
                payload["head_yaw"] = np.array([head_wire[0]], dtype=np.float32)
                payload["head_pitch"] = np.array([head_wire[1]], dtype=np.float32)
                payload["gripper_enable_mask"] = np.array([grip_enable_mask], dtype=np.uint8)
                payload["gripper_closed_mask"] = np.array([grip_closed_mask], dtype=np.uint8)
            self._publisher.send(payload)
            self._seq += 1

            next_tick += period_ns
            sleep_ns = next_tick - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            elif sleep_ns < -period_ns:
                LOG.warning("publish loop lag %.3f ms; resync", -sleep_ns / 1e6)
                next_tick = time.monotonic_ns()

    # ---- action stream: ~30 Hz, broker manages chunk lifecycle ----
    def action_stream_loop(self) -> None:
        LOG.info("action stream loop start (~%.1f Hz -> broker.infer)",
                 1.0 / ACTION_STREAM_DT_S)
        period_ns = int(ACTION_STREAM_DT_S * 1e9)
        next_tick = time.monotonic_ns()
        chunk_calls = 0
        while not self._stop.is_set():
            call_start = time.monotonic_ns()
            state = self._state.read()
            jpeg = self._state.read_ego_jpeg()
            obs = _build_obs(state, jpeg, self._prompt, self._profile)
            try:
                out = self._broker.infer(obs)
            except Exception as e:
                LOG.error("broker.infer failed: %s", e)
                time.sleep(0.5)
                continue
            call_ms = (time.monotonic_ns() - call_start) / 1e6
            # out["actions"] is a single-frame after broker slicing.
            action = np.asarray(out["actions"]).reshape(-1).astype(np.float32)
            if action.shape[0] != self._profile.action_dim:
                LOG.error("expected %d-dim action (profile=%s) after broker slice, got %s",
                         self._profile.action_dim, self._profile.name, action.shape)
                continue
            with self._action_lock:
                self._latest_action = action
                self._latest_action_ns = time.monotonic_ns()
            chunk_calls += 1
            # Log every 16 calls; VLA infer blocks happen at these boundaries.
            if chunk_calls % VLA_HORIZON_STEPS == 1 and call_ms > 50:
                LOG.info("VLA infer %.1f ms (chunk call #%d, seq=%d)",
                         call_ms, chunk_calls, self._seq)

            # Rate-limit to ACTION_STREAM_DT_S. The chunk-boundary call spikes to
            # VLA latency (~300 ms); we let it slip in that tick and resync next.
            next_tick += period_ns
            sleep_ns = next_tick - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            elif sleep_ns < -period_ns:
                next_tick = time.monotonic_ns()

    def run(self) -> None:
        pub = threading.Thread(target=self.publish_loop, name="publish", daemon=True)
        stream = threading.Thread(target=self.action_stream_loop, name="stream", daemon=True)
        pub.start()
        stream.start()
        try:
            while pub.is_alive() and stream.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            LOG.info("interrupt, stopping")
        finally:
            self._stop.set()
            pub.join(timeout=2.0)
            stream.join(timeout=2.0)
            self._publisher.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla-host", default="127.0.0.1")
    ap.add_argument("--vla-port", type=int, default=8000)
    ap.add_argument("--prompt", default="take the book from the bookshelf and hand it to the person")
    ap.add_argument("--zmq-bind", default="tcp://*:5556")
    ap.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA,
                    help="Low-pass output = alpha*plan + (1-alpha)*prev. Smaller = smoother.")
    ap.add_argument("--action-horizon", type=int, default=VLA_HORIZON_STEPS,
                    help="ActionChunkBroker cache size; must match trained model.")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="rx101",
                    help="Checkpoint body layout: rx101 (27-dof, no head) or "
                         "rx2 (29-dof, head+gripper direct control via PicoAuxGate).")
    ap.add_argument("--mock-state", action="store_true", help="use MockStateSource (no real robot).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(message)s")

    if args.mock_state:
        state: StateSource = MockStateSource()
    else:
        state = RealStateSource()

    try:
        Bridge(
            vla_host=args.vla_host,
            vla_port=args.vla_port,
            prompt=args.prompt,
            state_source=state,
            zmq_bind=args.zmq_bind,
            smoothing_alpha=args.smoothing_alpha,
            action_horizon=args.action_horizon,
            profile=PROFILES[args.profile],
        ).run()
    finally:
        close = getattr(state, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
