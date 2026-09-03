// SPDX-License-Identifier: Apache-2.0
//
// ZMQ SUB listener for the GR00T-WholeBodyControl Protocol v1 "pose" stream
// (encoder mode 0, joint-based motion tracking) — the wire our VLA bridge
// emits (see openpi-0_5/examples/rx101_bridge/zmq_pose.py).
//
// Wire per gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:
//
//   [topic_bytes] + [1280 B JSON header padded '\0'] + [raw payload]
//
// JSON header:
//   {"v": 1, "endian": "le", "count": 1,
//    "fields": [{"name":..,"dtype":"f32|f64|i32|i64|bool","shape":[..]},...]}
//
// Payload: concatenation of each array's C-contiguous little-endian bytes.
//
// Protocol v1 required fields (this listener; extra fields are ignored):
//   joint_pos   [N, kDof]  f32   rad, SONIC policy_order
//   joint_vel   [N, kDof]  f32   rad/s
//   body_quat   [N, 4]     f32   root_link (base_link) wxyz
//   frame_index [N]        i64   monotonic
//
// Per-listener singleton (mirrors RxSonicTeleopListener / RxDebugCmdListener:
// md_message_bus callbacks and the ZMQ SUB thread do not carry user_data).
#pragma once

#include <cstdint>
#include <string>

namespace rx101 {

class SonicRobotMotionBuffer;

/// GR00T Protocol v1 ZMQ SUB listener. Runs on its own dedicated thread; the
/// dequeued packets feed a SonicRobotMotionBuffer. Config comes from yaml
/// (`policy.sonic_zmq_*`).
class RxRobotMotionZmqSubscriber {
public:
    struct Config {
        /// ZMQ endpoint to subscribe to. Publisher (Python bridge) binds this.
        /// Local loopback is preferred when the bridge runs on the robot board.
        std::string connect_addr = "tcp://127.0.0.1:5556";
        /// Topic prefix (must be a prefix of the frame's leading bytes).
        std::string topic = "pose";
        /// Expected `joint_pos` inner dimension. Packets whose shape mismatches
        /// are dropped + counted (fail-closed; no partial application).
        int expected_dof = 27;
        /// Conflate=true keeps only the newest queued message per socket recv.
        /// The buffer merges by frame_index so overlap is safe; we conflate to
        /// minimize the sub thread's carry when the RT loop stalls.
        bool conflate = true;
    };

    /// Subscribe + start the sub thread. Safe to call with the buffer sink not
    /// yet bound; packets are validated and dropped until BindBuffer() lands.
    /// Returns false if the socket setup failed (already logged).
    static bool Init(const Config& cfg);

    /// Attach the sink. Idempotent. Until this is called, accepted packets are
    /// counted but their contents are discarded.
    static void BindBuffer(SonicRobotMotionBuffer* buffer);

    /// Stop the sub thread and close the socket. Idempotent.
    static void Shutdown();

    /// Accepted / dropped packet counts for the startup summary + tests.
    static uint64_t AcceptedCount();
    static uint64_t DroppedCount();

    /// Most-recent dropped reason (short static string) for the operator log.
    /// Empty if no drops yet. Not stable across concurrent drops; strictly a
    /// diagnostic hint.
    static const char* LastDropReason();
};

}  // namespace rx101
