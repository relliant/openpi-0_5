# Applying the VLA robot-native reference stream patch

Adds one new listener + one new buffer + a new branch in `SonicController::step()`
that consumes robot-native joint frames (encoder mode 0) from a ZMQ Protocol-v1
"pose" stream. The Python publisher is at
[examples/rx101_bridge/bridge.py](../bridge.py); this patch is its receiver on the
robot board.

Target: `/home/mondo/allen_he/linux-userspace/md_control_rx` (commit `286598fd3-dirty` per
current build_stamp).

## New files (drop-in)

Copy verbatim into the repo at the paths below.

| From (this dir) | To (in md_control_rx) |
|---|---|
| `md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.hpp` | same |
| `md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.cpp` | same |
| `md_pnc/module/policy/sonic/sonic_robot_motion_buffer.hpp` | same |
| `md_pnc/module/policy/sonic/sonic_robot_motion_buffer.cpp` | same |
| `configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml` | `configs/` (same dir as base yaml) |

The overlay yaml is meant to be applied via `rx101_pnc --target robot_stream`.
It only flips reference source; the base yaml's safety chain, encoder path,
motion clip fallback, `sonic_static_stand`, etc. are untouched.

## Existing files to edit (diffs below)

- `md_pnc/module/foundation/config/robot_config.hpp` — new PolicyConfig fields
- `md_pnc/module/foundation/config/robot_config.cpp` — parse them from yaml
- `md_pnc/module/policy/sonic/sonic_controller.hpp` — buffer member + accessor
- `md_pnc/module/policy/sonic/sonic_controller.cpp` — branch in step()
- `md_pnc/module/app/main.cpp` — start/bind the subscriber
- `md_pnc/module/policy/BUILD` — new cc_library target
- `md_pnc/module/protocol/BUILD` — new cc_library target (links libzmq)

### `robot_config.hpp` — add fields to `PolicyConfig` (next to the other `sonic_*` flags)

```cpp
    // ── Robot-native reference stream (encoder mode 0 via ZMQ Protocol v1).
    //    When true, SonicController consumes joint_pos/vel/body_quat futures
    //    from RxRobotMotionZmqSubscriber → SonicRobotMotionBuffer and bypasses
    //    both the SMPL→GMR retargeter and SonicMotionLoader clip playback.
    //    sonic_enable_smpl_teleop MUST be false when this is true (mutually
    //    exclusive reference sources; validated in loader).
    bool         sonic_via_robot_stream               = false;
    std::string  sonic_zmq_connect_addr               = "tcp://127.0.0.1:5556";
    std::string  sonic_zmq_topic                      = "pose";
    int          sonic_zmq_expected_dof               = 27;
    int          sonic_robot_stream_future_frames     = 10;   // 4 or 10 (native/legacy)
    int          sonic_robot_stream_playback_delay_frames = 10;
```

### `robot_config.cpp` — parse in the sonic-fields block (next to `sonic_smpl_via_gmr` read)

```cpp
    p.sonic_via_robot_stream =
        y["sonic_via_robot_stream"] ? y["sonic_via_robot_stream"].as<bool>() : false;
    if (y["sonic_zmq_connect_addr"])
        p.sonic_zmq_connect_addr = y["sonic_zmq_connect_addr"].as<std::string>();
    if (y["sonic_zmq_topic"])
        p.sonic_zmq_topic = y["sonic_zmq_topic"].as<std::string>();
    if (y["sonic_zmq_expected_dof"])
        p.sonic_zmq_expected_dof = y["sonic_zmq_expected_dof"].as<int>();
    if (y["sonic_robot_stream_future_frames"])
        p.sonic_robot_stream_future_frames = y["sonic_robot_stream_future_frames"].as<int>();
    if (y["sonic_robot_stream_playback_delay_frames"])
        p.sonic_robot_stream_playback_delay_frames = y["sonic_robot_stream_playback_delay_frames"].as<int>();

    // Mutual exclusion (fail-closed):
    if (p.sonic_via_robot_stream && p.sonic_enable_smpl_teleop) {
        throw std::runtime_error(
            "PolicyConfig: sonic_via_robot_stream and sonic_enable_smpl_teleop are "
            "mutually exclusive; set sonic_enable_smpl_teleop: false in the profile.");
    }
```

### `sonic_controller.hpp` — add member + accessor

```cpp
#include "sonic/sonic_robot_motion_buffer.hpp"   // near other sonic_* includes

class SonicController {
    // ... existing members ...
    bool                                        via_robot_stream_ = false;
    std::unique_ptr<SonicRobotMotionBuffer>     robot_motion_buf_;

public:
    /// Non-owning; returns nullptr if via_robot_stream_ was false in yaml.
    SonicRobotMotionBuffer* robotMotionBuffer() { return robot_motion_buf_.get(); }
};
```

### `sonic_controller.cpp` — constructor + step()

In the constructor, after `smpl_via_gmr_ = p.sonic_smpl_via_gmr && smpl_teleop_enabled_;`:

```cpp
    via_robot_stream_ = p.sonic_via_robot_stream;
    if (via_robot_stream_) {
        robot_motion_buf_ = std::make_unique<SonicRobotMotionBuffer>(
            /*playback_delay_frames=*/p.sonic_robot_stream_playback_delay_frames,
            /*future_frame_count=*/p.sonic_robot_stream_future_frames);
        const int encoder_dim = policy_->encoderInputDim();
        const bool ok =
            (p.sonic_robot_stream_future_frames == 10 &&
             encoder_dim == SonicObsBuilder::kEncoderObsDim) ||
            (p.sonic_robot_stream_future_frames == 4 &&
             encoder_dim == SonicObsBuilder::kNativeEncoderObsDim);
        if (!ok) {
            throw std::runtime_error(
                "SonicController: robot-stream future/encoder contract mismatch");
        }
        LOG_INFO("SonicController: robot-native reference stream ENABLED "
                 "(future_frames=%d, delay=%d, dof=%d, encoder=%d)",
                 p.sonic_robot_stream_future_frames,
                 p.sonic_robot_stream_playback_delay_frames,
                 p.sonic_zmq_expected_dof, encoder_dim);
    }
```

In `step()`, add a new branch parallel to the SMPL block (roughly around L580 in the
current source, after the SMPL/GMR window fill but before `enc_ptr = ...`):

```cpp
    bool used_robot_stream = false;
    if (via_robot_stream_ && robot_motion_buf_) {
        // Under the same mutex the SMPL path uses for its buffer swap — but the
        // robot-motion buffer owns its own mutex, so we do not need the SMPL
        // mutex here.
        const auto st = robot_motion_buf_->future(&gmr_dof_pos_f_,
                                                  &gmr_dof_vel_f_,
                                                  &gmr_root_quat_f_);
        if (st == SonicRobotMotionBuffer::Status::kReady) {
            used_robot_stream = true;
        } else if (st != SonicRobotMotionBuffer::Status::kNotPrimed) {
            static uint64_t drop_log = 0;
            if ((drop_log++ % 50) == 0)
                LOG_WARN("SonicController: robot-stream horizon not ready (%d)",
                         static_cast<int>(st));
        }
    }
```

In the encoder-input dispatch (where `used_smpl || hold_smpl` currently gates
`encoderObs(...)` for the g1 head), extend the condition:

```cpp
    if (used_smpl || hold_smpl || used_robot_stream) {
        // ... existing anchor_quat_aligned_smpl setup applies only when used_smpl.
        if (used_robot_stream) {
            // Robot-native stream: root quat came from bridge (base_link), no
            // SMPL yaw-latch. Use the raw anchor_quat.
            enc_ptr = &obs_->encoderObs(
                gmr_dof_pos_f_, gmr_dof_vel_f_, gmr_root_quat_f_,
                anchor_quat, policy_->encoderInputDim());
        } else if (smpl_via_gmr_) {
            enc_ptr = &obs_->encoderObs(
                gmr_dof_pos_f_, gmr_dof_vel_f_, gmr_root_quat_f_,
                anchor_quat_aligned_smpl, policy_->encoderInputDim());
        } else {
            enc_ptr = &obs_->encoderObsSmpl(
                smpl_encoder_index_, smpl_joints_f_, smpl_root_quat_f_,
                wrist_dof_f_, anchor_quat_aligned_smpl);
        }
    }
```

### `main.cpp` — init/bind

After the message bus is up (near existing `RxSonicTeleopListener::Init(bus)` call):

```cpp
    if (cfg.policy.sonic_via_robot_stream) {
        RxRobotMotionZmqSubscriber::Config zmq_cfg{
            .connect_addr  = cfg.policy.sonic_zmq_connect_addr,
            .topic         = cfg.policy.sonic_zmq_topic,
            .expected_dof  = cfg.policy.sonic_zmq_expected_dof,
            .conflate      = true,
        };
        if (!RxRobotMotionZmqSubscriber::Init(zmq_cfg)) {
            LOG_WARN("RxRobotMotionZmqSubscriber::Init failed; robot-stream unavailable");
        }
    }
```

After the SonicController is constructed:

```cpp
    RxRobotMotionZmqSubscriber::BindBuffer(sonic_controller.robotMotionBuffer());
```

At shutdown:

```cpp
    RxRobotMotionZmqSubscriber::Shutdown();
```

### `md_pnc/module/policy/BUILD` — new cc_library target

```
cc_library(
    name = "sonic_robot_motion_buffer",
    srcs = ["sonic/sonic_robot_motion_buffer.cpp"],
    hdrs = ["sonic/sonic_robot_motion_buffer.hpp"],
    includes = ["."],
    deps = [
        "//md_control_rx/md_pnc/module/foundation:foundation",
    ],
    visibility = ["//visibility:public"],
)
```

Add `":sonic_robot_motion_buffer"` to the deps of the existing `sonic` cc_library.

### `md_pnc/module/protocol/BUILD` — new cc_library target

```
cc_library(
    name = "rx_robot_motion_zmq_subscriber",
    srcs = ["rx_robot_motion_zmq_subscriber.cpp"],
    hdrs = ["rx_robot_motion_zmq_subscriber.hpp"],
    includes = ["."],
    linkopts = ["-lzmq"],
    deps = [
        "//md_control_rx/md_pnc/module/foundation:foundation",
        "//md_control_rx/md_pnc/module/policy:sonic_robot_motion_buffer",
        "//md_control_rx:yaml_cpp_system",
    ],
    visibility = ["//visibility:public"],
)
```

Add `":rx_robot_motion_zmq_subscriber"` to whatever cc_library `main.cpp` depends on
(likely the pnc app cc_library — check the `app/BUILD` file).

## Build + deploy

**Where to build:** Neither host has `bazel` installed. Mondo's build wraps
bazel inside their CI Docker image
`xjp-dockerhub-registry.ap-southeast-1.cr.aliyuncs.com/x5/builder:1.0`; the entry
point is `./build.sh` (or `./scripts/build_md_control_rx_onboard.sh`) at the top
of the `linux-userspace` monorepo. This produces the aarch64 binary via
cross-compilation inside the container. Run this on any x86_64 host with Docker
and a checkout of `linux-userspace`.

The GPU workstation (host with Docker access) is the natural build box; the
robot has the checkout but no Docker; running the x86_64 image on the aarch64
robot via qemu emulation is not recommended.

```bash
# ── one-time: get a build checkout of linux-userspace on the workstation ──
cd ~/Project    # or wherever you keep source
git clone git@gitlab.mondorobotics.com:sysdev/linux-userspace.git
cd linux-userspace/md_control_rx
git log -1      # sanity: match the deployed rx101_pnc build_stamp / commit_sha if it matters

# ── apply the patch ──
PATCH_DIR=~/Project/openpi-0_5/examples/rx101_bridge/cpp_patch
cp "$PATCH_DIR"/md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.{hpp,cpp} \
   md_pnc/module/protocol/
cp "$PATCH_DIR"/md_pnc/module/policy/sonic/sonic_robot_motion_buffer.{hpp,cpp} \
   md_pnc/module/policy/sonic/
cp "$PATCH_DIR"/configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml \
   configs/
# The edits to sonic_controller.{hpp,cpp}, robot_config.{hpp,cpp}, main.cpp, and
# both BUILD files are diff blocks in this doc — apply them by hand.

# ── build (Docker will pull once, then run bazel inside) ──
cd ~/Project/linux-userspace
./build.sh md_control_rx      # release build by default; add --debug for symbols

# Binary lands at: bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc  (aarch64)

# ── deploy ──
scp bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc \
    "$PATCH_DIR"/configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml \
    rx101:/tmp/

ssh rx101 <<'EOF'
sudo systemctl stop rx101-pnc
sudo cp /tmp/rx101_pnc /md/bin/rx101_pnc
sudo cp /tmp/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml \
        /md/etc/md_pnc/configs/
# Point the systemd unit's ExecStart at `--target robot_stream` (drop-in override
# under /etc/systemd/system/rx101-pnc.service.d/ is the least invasive way).
# Alternatively run manually:  /md/bin/rx101_pnc --config .../rx_p2_sonic_gmr_5N_step023800_real.yaml --target robot_stream
sudo systemctl daemon-reload
sudo systemctl start rx101-pnc
EOF
```

Troubleshooting the build wrapper:

- `Command 'bazel' not found` when invoked directly → don't call `bazel` bare; use `./build.sh <component>` at the top of `linux-userspace`.
- Docker platform mismatch on Apple Silicon or aarch64 host → `./build.sh` already handles this via `DOCKER_PLATFORM=linux/amd64`; on aarch64 it works but slowly (Rosetta / qemu-user).
- Docker image pull fails → check network to `xjp-dockerhub-registry.ap-southeast-1.cr.aliyuncs.com` and CR credentials (`docker login` if the image is private).

## Golden-diff test (before wiring VLA)

The safest sim2sim is:

1. On the bridge machine, run `python -m examples.rx101_bridge.bridge --mock-state`
   (uses the mock default-stand pose, keeps publishing constant joint refs at 50 Hz).
2. On the robot (fall-harness + physical e-stop mandatory), start pnc with the new
   overlay. Expected behavior: SONIC enters RL_RUNNING and holds the default stand
   pose without tremor.
3. Compare the pnc `data_rx__control_sonic__debug` fields against a run with the
   base yaml (no overlay, `sonic_motion_clip: rx_p2_walk4_ik` playing on loop).
   The encoder input distribution should be numerically close (not identical: our
   stream is constant, the clip walks) but the drive_state / encoder_index / etc.
   metadata should match encoder mode 0 behavior.

Only after the constant-pose stream is validated should you point the bridge at the
live VLA server.

## Known follow-ups

- Real robot state feed for the bridge is still gated on the shm proprio wire dump
  (deploy doc §6). Until then, bridge runs `--mock-state` and outputs whatever the
  VLA predicts from a synthetic obs, which is meaningless for closed-loop control
  but sufficient to validate the wire → buffer → encoder path end-to-end.
- No FK yet: bridge sends `body_quat = state.root_quat_wxyz`, currently identity
  from MockStateSource. When state feed lands, this should come from the IMU. For
  a static manipulation task the drift matters less than for locomotion.
