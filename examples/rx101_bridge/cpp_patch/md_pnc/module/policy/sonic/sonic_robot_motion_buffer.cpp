// SPDX-License-Identifier: Apache-2.0
#include "sonic/sonic_robot_motion_buffer.hpp"

#include <algorithm>

namespace rx101 {

SonicRobotMotionBuffer::SonicRobotMotionBuffer(int playback_delay_frames, int future_frame_count)
    : delay_(playback_delay_frames), future_frame_count_(future_frame_count)
{
    if (future_frame_count_ != kNativeFutureFrames && future_frame_count_ != kNumFutureFrames) {
        throw std::invalid_argument("SonicRobotMotionBuffer future frame count must be 4 or 10");
    }
    if (delay_ < future_frame_count_ - 1 || delay_ >= static_cast<int>(kMaxFrames)) {
        throw std::invalid_argument("SonicRobotMotionBuffer delay must cover future_frame_count-1 and fit the ring");
    }
}

void SonicRobotMotionBuffer::ingest(const RobotMotionFrame& frame, uint64_t recv_mono_us)
{
    std::lock_guard<std::mutex> lk(mu_);
    // Duplicate or backwards → drop.
    if (primed_ && frame.frame_index <= newest_) {
        return;
    }
    // Non-dense (gap) → reset and re-prime from this frame. This is safer than
    // continuing with holes: we would rather fall back to standing for one
    // window than serve a torn horizon.
    if (primed_ && frame.frame_index != newest_ + 1) {
        ring_.clear();
        primed_ = false;
    }
    ring_.push_back(frame);
    if (!primed_) {
        oldest_ = frame.frame_index;
        primed_ = true;
    }
    newest_ = frame.frame_index;
    while (ring_.size() > kMaxFrames) {
        ring_.pop_front();
        oldest_ = ring_.front().frame_index;
    }
    last_progress_mono_us_ = recv_mono_us;
}

bool SonicRobotMotionBuffer::primed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return primed_;
}

bool SonicRobotMotionBuffer::fresh(uint64_t now_us, uint64_t timeout_us) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!primed_) return false;
    if (last_progress_mono_us_ == 0) return false;
    return (now_us - last_progress_mono_us_) < timeout_us;
}

std::size_t SonicRobotMotionBuffer::frameCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ring_.size();
}

int64_t SonicRobotMotionBuffer::oldestIndex() const {
    std::lock_guard<std::mutex> lk(mu_);
    return oldest_;
}

int64_t SonicRobotMotionBuffer::newestIndex() const {
    std::lock_guard<std::mutex> lk(mu_);
    return newest_;
}

int64_t SonicRobotMotionBuffer::cursor() const {
    std::lock_guard<std::mutex> lk(mu_);
    // Cursor lags the newest by `delay_` frames.
    if (!primed_) return -1;
    return newest_ - delay_;
}

SonicRobotMotionBuffer::Status SonicRobotMotionBuffer::future(
    std::vector<float>* joint_pos_future,
    std::vector<float>* joint_vel_future,
    std::vector<float>* root_quat_future)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (!primed_) return Status::kNotPrimed;
    const int64_t cur = newest_ - delay_;
    if (cur < oldest_) return Status::kShortHorizon;
    // Verify contiguous horizon [cur, cur + future_frame_count_-1].
    const int64_t last_needed = cur + future_frame_count_ - 1;
    if (last_needed > newest_) return Status::kShortHorizon;
    // Ring stride is 1 by construction (any gap resets primed_). Locate cur.
    const size_t cur_idx = static_cast<size_t>(cur - oldest_);
    if (cur_idx + future_frame_count_ > ring_.size()) return Status::kShortHorizon;

    joint_pos_future->resize(static_cast<size_t>(future_frame_count_) * kSonicPolicyJoints);
    joint_vel_future->resize(static_cast<size_t>(future_frame_count_) * kSonicPolicyJoints);
    root_quat_future->resize(static_cast<size_t>(future_frame_count_) * 4);
    for (int f = 0; f < future_frame_count_; ++f) {
        const RobotMotionFrame& fr = ring_[cur_idx + f];
        std::copy(fr.joint_pos.begin(),      fr.joint_pos.end(),
                  joint_pos_future->begin() + f * kSonicPolicyJoints);
        std::copy(fr.joint_vel.begin(),      fr.joint_vel.end(),
                  joint_vel_future->begin() + f * kSonicPolicyJoints);
        std::copy(fr.body_quat_wxyz.begin(), fr.body_quat_wxyz.end(),
                  root_quat_future->begin() + f * 4);
    }
    return Status::kReady;
}

void SonicRobotMotionBuffer::reset()
{
    std::lock_guard<std::mutex> lk(mu_);
    ring_.clear();
    oldest_ = newest_ = -1;
    primed_ = false;
    last_progress_mono_us_ = 0;
}

}  // namespace rx101
