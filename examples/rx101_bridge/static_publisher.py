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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5556",
                    help="ZMQ PUB bind address. Robot subscribes on this port.")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="Publish rate (Hz). Match SONIC control_dt = 20 ms.")
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="Stop after this many seconds. 0 = run forever.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    log = logging.getLogger("static_publisher")

    pub = zmq_pose.PosePublisher(bind_addr=args.bind)
    log.info(f"publishing constant DEFAULT_STAND_SONIC at {args.hz:.1f} Hz on {args.bind}")
    log.info(f"joint_pos[:6] (SONIC BFS): {DEFAULT_STAND_SONIC[:6].tolist()}")

    # Burst multiple sequential frames per message so the C++ subscriber's
    # ZMQ_CONFLATE=1 (keeps only latest) still delivers a dense window that
    # primes SonicRobotMotionBuffer immediately. Without bursting, conflate
    # would drop intermediate frames -> frame_index jumps -> buffer resets
    # every ingest -> never ready. BURST must exceed delay + future_frame_count
    # (10 + 10 = 20 for default legacy1691 encoder) for future() to be Ready.
    BURST = 25
    period_ns = int(1e9 / args.hz * BURST)   # publish at (hz / BURST) Hz
    next_tick = time.monotonic_ns()
    seq = 0
    start = time.monotonic()
    try:
        while True:
            # Repeat the same constant pose BURST times with sequential frame_index.
            joint_pos_burst   = np.broadcast_to(DEFAULT_STAND_SONIC, (BURST, 27)).astype(np.float32)
            joint_vel_burst   = np.zeros((BURST, 27), dtype=np.float32)
            body_quat_burst   = np.broadcast_to(IDENTITY_QUAT, (BURST, 4)).astype(np.float32)
            frame_index_burst = np.arange(seq, seq + BURST, dtype=np.int64)
            pub.send({
                "joint_pos":   joint_pos_burst,
                "joint_vel":   joint_vel_burst,
                "body_quat":   body_quat_burst,
                "frame_index": frame_index_burst,
            })
            seq += BURST
            if (seq // BURST) % int(max(1, args.hz / BURST)) == 0:   # every ~1s
                log.info(f"published {seq} frames (burst={BURST})")
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
