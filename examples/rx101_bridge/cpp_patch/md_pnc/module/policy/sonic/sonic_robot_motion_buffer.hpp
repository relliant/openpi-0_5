// SPDX-License-Identifier: Apache-2.0
//
// Dense robot-native reference stream buffer (encoder mode 0 / joint-based
// motion tracking). Consumes RobotMotionFrame values written by
// RxRobotMotionZmqSubscriber, produces the (dof_pos_future, dof_vel_future,
// root_quat_future) tuple that SonicController::step() would otherwise
// receive from the SMPL->GMR retargeter or SonicMotionLoader.
//
// Contract matches SonicMotionLoader::futureRefs exactly (10-frame or 4-frame
// horizons, contiguous frame_index at stride=1 = 20 ms), so the g1 encoder's
// input distribution is byte-identical to the training-time clip playback path.
//
// Mirrors SonicTeleopBuffer's playback-cursor model but stores robot-space
// frames instead of SMPL frames. Frame_index gaps → kGap; short-horizon →
// kShortHorizon; consumer falls back to standing reference.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace rx101 {

constexpr int kSonicPolicyJoints = 27;

/// One robot-native dense reference frame.
struct RobotMotionFrame {
    int64_t frame_index = -1;
    std::array<float, kSonicPolicyJoints> joint_pos{};      // rad, SONIC policy_order
    std::array<float, kSonicPolicyJoints> joint_vel{};      // rad/s, finite-diff on publisher
    std::array<float, 4>                  body_quat_wxyz{}; // base_link orientation, wxyz
};

class SonicRobotMotionBuffer {
public:
    enum class Status { kReady, kNotPrimed, kShortHorizon, kGap };

    static constexpr int         kNumFutureFrames = 10;   // must match SonicObsBuilder legacy1691
    static constexpr int         kNativeFutureFrames = 4; // native1187
    static constexpr std::size_t kMaxFrames = 64;

    /// delay_frames should be >= future_frame_count-1 so the future horizon fits below
    /// the newest buffered frame. Native1187 uses 4; legacy1691 uses 10.
    explicit SonicRobotMotionBuffer(int playback_delay_frames = kNumFutureFrames,
                                    int future_frame_count = kNumFutureFrames);

    /// Merge one frame into the ring. Strictly-monotonic on frame_index; dupes/
    /// backwards frames are silently dropped.
    void ingest(const RobotMotionFrame& frame, uint64_t recv_mono_us);

    bool primed() const;
    bool fresh(uint64_t now_us, uint64_t timeout_us) const;
    std::size_t frameCount() const;
    int64_t oldestIndex() const;
    int64_t newestIndex() const;
    int64_t cursor() const;

    /// Fill 10 flat future arrays (or 4 for native1187) from the playback cursor.
    /// The C++ side layout matches the encoder input: joint_pos_future[N*27],
    /// joint_vel_future[N*27], root_quat_future[N*4].
    ///
    /// Preconditions: consumer holds the buffer mutex during this call (same as
    /// SonicTeleopBuffer::smplFuture semantics).
    Status future(std::vector<float>* joint_pos_future,
                  std::vector<float>* joint_vel_future,
                  std::vector<float>* root_quat_future);

    /// Discard everything (called on transport/session reset).
    void reset();

    /// Serialization width — read by callers that pre-size the encoder input.
    int futureFrameCount() const { return future_frame_count_; }
    int delayFrames()      const { return delay_; }

private:
    int delay_;
    int future_frame_count_;

    mutable std::mutex mu_;
    std::deque<RobotMotionFrame> ring_;   // ordered by frame_index, dense at stride=1
    int64_t oldest_ = -1;
    int64_t newest_ = -1;
    uint64_t last_progress_mono_us_ = 0;
    bool primed_ = false;
};

}  // namespace rx101
