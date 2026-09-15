"""Constant-default-stand ZMQ publisher for the rx101 SONIC static self-check.

Publishes a fixed default-stand joint reference at 50 Hz to
tcp://*:5556 (topic="pose"), using GR00T ZMQ Protocol v1 wire format.

The patched rx101_pnc (with `--target robot_stream`) subscribes here, feeds
the frames into SonicRobotMotionBuffer, and drives the SONIC encoder-mode-0
policy. With a constant reference, the robot should enter RL_RUNNING and
hold the default standing pose without tremor — this validates every part of
the pipeline (wire, buffer, controller branch, encoder input) BEFORE we hook
the real VLA and its non-constant plan.

This publisher does NOT depend on VLA / openpi. It only needs pyzmq + numpy.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

try:
    from examples.rx101_bridge import zmq_pose
except ImportError:
    # When shipped as a flat pair with zmq_pose.py alongside (robot deployment).
    import zmq_pose


# default_pos from rx_p2_sonic_gmr_5N_step023800_real.yaml, SONIC BFS joint order
# (already permuted from the dataset order). This is the robot's squatting standing
# pose used as SONIC's static-stand reference.
DEFAULT_STAND_SONIC = np.array(
    [-0.2, -0.2,  0.0,     # l_hip_pitch, r_hip_pitch, waist_yaw
      0.0,  0.0,  0.0,     # l_hip_roll,  r_hip_roll,  waist_roll
      0.0,  0.0,  0.0,     # l_hip_yaw,   r_hip_yaw,   waist_pitch
      0.4,  0.4,  0.0,     # l_knee, r_knee, l_shoulder_pitch
      0.0, -0.2, -0.2,     # r_shoulder_pitch, l_ankle_pitch, r_ankle_pitch
      0.0,  0.0,  0.0,     # l_shoulder_roll, r_shoulder_roll, l_ankle_roll
      0.0,  0.0,  0.0,     # r_ankle_roll, l_shoulder_yaw, r_shoulder_yaw
      0.0,  0.0,  0.0,     # l_elbow, r_elbow, l_wrist_roll
      0.0,  0.0,  0.0],    # r_wrist_roll, l_wrist_pitch, r_wrist_pitch
    dtype=np.float32,
)
assert DEFAULT_STAND_SONIC.shape == (27,)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz, base_link at rest

# ─────────────────────── --natural-raise experiment ──────────────────────
# Hypothesis (2026-09-08): our frozen/constant reference (same joint_pos value
# repeated across all future frames, joint_vel=0 always) may be an out-of-
# distribution input for the SONIC encoder-mode-0 tracking policy — real
# training references are almost certainly dynamic trajectories, not frozen
# snapshots. Symptom: with the frozen reference, the decoder's raw action for
# l_elbow/r_elbow (SONIC BFS idx 21/22) is consistently ~+6.5..6.7 (near the
# ±10 clip range) regardless of the (zero) reference target, producing a
# ~90° elbow bend the reference never asked for.
#
# This experiment replaces the frozen arm reference with a SMOOTH, physically
# self-consistent ramp (half-cosine ease, matching the same profile used
# elsewhere in this codebase for PREPOSE ramps): shoulder_pitch/elbow move
# from 0 to a target over `--raise-duration-s`, WITH joint_vel set to the
# analytic derivative of that ramp (not zero) — then hold at the target
# (velocity returns to 0). All other joints stay at DEFAULT_STAND_SONIC,
# untouched. If tracking fidelity improves (decoder's target ends up close
# to OUR target, not the same runaway ~1.5rad regardless of what we ask for),
# that supports the "frozen reference is OOD" hypothesis over a wrong-
# checkpoint or wire-protocol explanation.
DEFAULT_RAISE_DURATION_S = 2.0

# SONIC BFS indices (see joint_maps.SONIC_POLICY_ORDER for the full 27-name list).
_L_SHOULDER_PITCH, _R_SHOULDER_PITCH = 11, 12
_L_ELBOW, _R_ELBOW = 21, 22

# (start, end) rad per animated joint. End values are in the SAME direction/
# rough magnitude as the decoder's own observed elbow bias (raw action ~6.5-6.7
# -> ~1.5rad after action_scale=0.23), chosen so the comparison is meaningful:
# does a DYNAMIC ramp toward roughly where the decoder "wants" to go anyway
# produce tighter tracking than a frozen reference stuck at 0 the whole time?
_RAISE_TARGETS: dict[int, tuple[float, float]] = {
    _L_SHOULDER_PITCH: (0.0, -0.5),
    _R_SHOULDER_PITCH: (0.0, -0.5),
    _L_ELBOW: (0.0, 1.2),
    _R_ELBOW: (0.0, 1.2),
}


def raise_pose_and_vel(t_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Half-cosine ramp of the arm joints in _RAISE_TARGETS from start to end
    over `duration_s`, then hold. All other 23 joints stay at DEFAULT_STAND_SONIC
    with zero velocity throughout. Returns (pos[27], vel[27]), both float32.

    Velocity is the ANALYTIC derivative of the position ramp (not a finite-
    difference approximation), so pos/vel are self-consistent by construction —
    exactly what a real captured motion trajectory would give the encoder.
    """
    pos = DEFAULT_STAND_SONIC.copy()
    vel = np.zeros(27, dtype=np.float32)
    phase = float(np.clip(t_s / duration_s, 0.0, 1.0))
    alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
    dalpha_dt = 0.0
    if 0.0 <= t_s <= duration_s:
        dalpha_dt = (np.pi / (2.0 * duration_s)) * np.sin(np.pi * phase)
    for idx, (p0, p1) in _RAISE_TARGETS.items():
        pos[idx] = p0 + alpha * (p1 - p0)
        vel[idx] = dalpha_dt * (p1 - p0)
    return pos.astype(np.float32), vel.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5556",
                    help="ZMQ PUB bind address. Robot subscribes on this port.")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="Publish rate (Hz). Match SONIC control_dt = 20 ms.")
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="Stop after this many seconds. 0 = run forever.")
    ap.add_argument("--enable-aux", action="store_true",
                    help="Also publish constant head_yaw/head_pitch/gripper aux fields "
                         "(§ RX2 head/gripper direct control static self-check). Body-only "
                         "self-check is unaffected when this is NOT set.")
    ap.add_argument("--head-yaw", type=float, default=0.0,
                    help="Desired ABSOLUTE MCU-frame head yaw target, rad. Sign-cancellation "
                         "is applied internally to match bridge.py's HEAD_SIGN_YAW_PITCH — "
                         "pass the value you actually want the head to reach.")
    ap.add_argument("--head-pitch", type=float, default=0.0,
                    help="Desired ABSOLUTE MCU-frame head pitch target, rad. Same convention "
                         "as --head-yaw.")
    ap.add_argument("--grip-left-closed", action="store_true",
                    help="Request left gripper closed (default: open).")
    ap.add_argument("--grip-right-closed", action="store_true",
                    help="Request right gripper closed (default: open).")
    ap.add_argument("--natural-raise", action="store_true",
                    help="Replace the frozen constant arm reference with a smooth "
                         "half-cosine ramp (shoulder_pitch/elbow, non-zero joint_vel) "
                         "over --raise-duration-s, then hold. Tests whether a DYNAMIC "
                         "reference tracks better than a frozen one (see comment above "
                         "_RAISE_TARGETS). All other joints unaffected.")
    ap.add_argument("--raise-duration-s", type=float, default=DEFAULT_RAISE_DURATION_S,
                    help="Ramp duration for --natural-raise, seconds.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    log = logging.getLogger("static_publisher")

    pub = zmq_pose.PosePublisher(bind_addr=args.bind)
    log.info(f"publishing constant DEFAULT_STAND_SONIC at {args.hz:.1f} Hz on {args.bind}")
    log.info(f"joint_pos[:6] (SONIC BFS): {DEFAULT_STAND_SONIC[:6].tolist()}")

    head_wire = np.zeros(2, dtype=np.float32)
    grip_enable_mask = 0
    grip_closed_mask = 0
    if args.enable_aux:
        # bridge.py's HEAD_SIGN_YAW_PITCH sign-cancellation: PicoAuxGate multiplies our
        # wire value by yaml's pico_aux.head.sign=[1.0,-1.0] before applying. We pre-
        # multiply by the SAME sign so --head-yaw/--head-pitch are the ABSOLUTE target
        # you actually get, not PicoAuxGate's internal "measured" convention.
        head_sign = np.array([1.0, -1.0], dtype=np.float32)
        head_wire = (head_sign * np.array([args.head_yaw, args.head_pitch], dtype=np.float32))
        grip_enable_mask = 0x03
        grip_closed_mask = (0x01 if args.grip_left_closed else 0) | (0x02 if args.grip_right_closed else 0)
        log.info(f"aux ENABLED: head_target(abs)=[{args.head_yaw:+.3f},{args.head_pitch:+.3f}]rad "
                f"wire=[{head_wire[0]:+.3f},{head_wire[1]:+.3f}]  "
                f"grip_closed=(L={args.grip_left_closed},R={args.grip_right_closed})")

    # Burst multiple sequential frames per message so the C++ subscriber's
    # ZMQ_CONFLATE=1 (keeps only latest) still delivers a dense window that
    # primes SonicRobotMotionBuffer immediately. Without bursting, conflate
    # would drop intermediate frames -> frame_index jumps -> buffer resets
    # every ingest -> never ready. BURST must exceed delay + future_frame_count
    # (10 + 10 = 20 for default legacy1691 encoder) for future() to be Ready.
    #
    # BURST is ALWAYS 25 — this is the proven, repeatedly-validated body-priming
    # behavior (2026-09 real-hardware regression tests); --enable-aux must NEVER
    # change it. An earlier attempt (2026-09-03) instead dropped BURST to 1 for
    # --enable-aux to fix aux-freshness flicker (see STRIDE comment below) and
    # this coincided with an unexplained "robot turned left" incident — root
    # cause was never confirmed, but changing body's own publish cadence for an
    # aux-only problem was needlessly broad. This version fixes ONLY the aux
    # path and leaves body's BURST=25/no-overlap cadence completely untouched
    # in the default (non-aux) case.
    #
    # aux freshness fix — overlapping windows, body-priming semantics unchanged:
    #   SonicController::kSonicTeleopTimeoutUs (500 ms as of this fix) gates the
    #   head/gripper aux path (via SonicTeleopAux, NOT the body encoder's own
    #   future()-based dense-window check). A 25-frame burst sent only once per
    #   500 ms (STRIDE == BURST, the historical behavior) refreshes that clock
    #   right at its own timeout boundary — too tight a margin. So when
    #   --enable-aux is set, we advance frame_index by a SMALLER stride (5) each
    #   cycle instead of by the full BURST (25): consecutive 25-frame bursts then
    #   overlap by 20 frames. The C++ ring's dup/backward check
    #   (`frame.frame_index <= newest_`) silently drops the 20 overlapping
    #   frames and appends only the 5 genuinely-new ones — so the ring stays
    #   exactly as dense/contiguous as the proven BURST=25 default, while a
    #   burst now *arrives* every STRIDE/hz = 100 ms instead of every 500 ms,
    #   comfortably inside the 500 ms aux timeout (5x margin instead of 1x).
    BURST = 25
    STRIDE = 5 if args.enable_aux else BURST   # overlap only in aux mode
    period_ns = int(1e9 / args.hz * STRIDE)    # publish at (hz / STRIDE) Hz
    next_tick = time.monotonic_ns()
    seq = 0
    start = time.monotonic()
    if args.natural_raise:
        log.info(f"natural-raise ENABLED: shoulder_pitch/elbow ramp over "
                f"{args.raise_duration_s:.2f}s then hold. targets: "
                f"{ {k: v[1] for k, v in _RAISE_TARGETS.items()} }")

    try:
        while True:
            # seq advances by STRIDE each cycle; in aux mode this overlaps the
            # previous burst's tail (see comment above) instead of jumping ahead.
            frame_index_burst = np.arange(seq, seq + BURST, dtype=np.int64)
            if args.natural_raise:
                # Each of the BURST rows is a DIFFERENT point along the ramp — frame_index
                # step k corresponds to trajectory time k/hz seconds (SONIC's assumed
                # frame period), NOT a duplicated snapshot like the frozen-reference path.
                joint_pos_burst = np.empty((BURST, 27), dtype=np.float32)
                joint_vel_burst = np.empty((BURST, 27), dtype=np.float32)
                for i in range(BURST):
                    t_s = (seq + i) / args.hz
                    joint_pos_burst[i], joint_vel_burst[i] = raise_pose_and_vel(
                        t_s, args.raise_duration_s)
            else:
                # Repeat the same constant pose BURST times with sequential frame_index.
                joint_pos_burst = np.broadcast_to(DEFAULT_STAND_SONIC, (BURST, 27)).astype(np.float32)
                joint_vel_burst = np.zeros((BURST, 27), dtype=np.float32)
            body_quat_burst   = np.broadcast_to(IDENTITY_QUAT, (BURST, 4)).astype(np.float32)
            payload = {
                "joint_pos":   joint_pos_burst,
                "joint_vel":   joint_vel_burst,
                "body_quat":   body_quat_burst,
                "frame_index": frame_index_burst,
            }
            if args.enable_aux:
                payload["head_yaw"] = np.full(BURST, head_wire[0], dtype=np.float32)
                payload["head_pitch"] = np.full(BURST, head_wire[1], dtype=np.float32)
                payload["gripper_enable_mask"] = np.full(BURST, grip_enable_mask, dtype=np.uint8)
                payload["gripper_closed_mask"] = np.full(BURST, grip_closed_mask, dtype=np.uint8)
            pub.send(payload)
            seq += STRIDE
            if (seq // STRIDE) % int(max(1, args.hz / STRIDE)) == 0:   # every ~1s
                log.info(f"published up to frame {seq + BURST - 1} (burst={BURST} stride={STRIDE})")
            if args.duration_s > 0 and (time.monotonic() - start) >= args.duration_s:
                log.info("duration reached, stopping")
                break
            next_tick += period_ns
            sleep_ns = next_tick - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            elif sleep_ns < -period_ns:
                next_tick = time.monotonic_ns()  # resync after long stall
    except KeyboardInterrupt:
        log.info("interrupt, stopping")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
