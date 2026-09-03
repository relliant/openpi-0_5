# Resuming the rx101 VLA / SONIC integration build

This document snapshots the state as of the workstation-side build attempt on
2026-08-31. Everything code-wise is done and applied to `~/Project/linux-userspace`.
The build is blocked on Git LFS permissions.

## What's applied to `~/Project/linux-userspace`

All C++ / YAML / BUILD edits from [APPLY.md](APPLY.md) landed in the workstation copy:

**Modified files** (`git diff` will show them):

- `md_control_rx/BUILD` — added `zmq_system` cc_library
- `md_control_rx/md_pnc/module/app/BUILD` — added `rx_robot_motion_zmq_subscriber` dep
- `md_control_rx/md_pnc/module/app/main.cpp` — added include + Init/BindBuffer/Shutdown
- `md_control_rx/md_pnc/module/foundation/config/robot_config.hpp` — 6 new `PolicyConfig` fields
- `md_control_rx/md_pnc/module/foundation/config/robot_config.cpp` — yaml parsers + mutual-exclusion validation
- `md_control_rx/md_pnc/module/policy/BUILD` — split `sonic_robot_motion_buffer` cc_library, add to `sonic` deps
- `md_control_rx/md_pnc/module/policy/sonic/sonic_controller.hpp` — include + `via_robot_stream_` + `robot_motion_buf_` + `robotMotionBuffer()`
- `md_control_rx/md_pnc/module/policy/sonic/sonic_controller.cpp` — constructor init + step() branch
- `md_control_rx/md_pnc/module/protocol/BUILD` — new `rx_robot_motion_zmq_subscriber` cc_library

**New files**:

- `md_control_rx/configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml` (overlay)
- `md_control_rx/md_pnc/module/policy/sonic/sonic_robot_motion_buffer.{hpp,cpp}`
- `md_control_rx/md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.{hpp,cpp}`
- `md_control_rx/md_pnc/third_party/zmq/` — vendored zmq.h (from libzmq v4.3.4 upstream)
  + real libzmq.so.5.2.4 (copied from the robot's `/lib/aarch64-linux-gnu/`)

**LFS pointer files replaced with real .so** (temporary; will re-appear once LFS pull
works, or can be committed if they change). Copied from robot's `/md/etc/md_pnc/lib/`:

- `md_control_rx/md_pnc/third_party/yaml-cpp/lib/libyaml-cpp.so.0.7.0`
- `md_control_rx/md_pnc/third_party/lcm/lib/{liblcm.so.1.3.4,libglib-2.0.so.0.7200.1,libpcre.so.3.13.3}`
- `md_control_rx/md_pnc/third_party/zlib/lib/libz.so.1.2.11`
- `md_control_rx/md_pnc/third_party/onnxruntime/lib/aarch64/libonnxruntime.so.{1,1.22.0}`

## Why the build is blocked

`./build.sh md_control_rx` reaches the link stage for `libmuml.so`, which needs
`x5_lib/hbre/lib/libalog.so.1.0.1` (a Git LFS-tracked pre-built lib). This file is
NOT in the robot's `/md/etc/md_pnc/lib/` deploy tree either — the deployed
`rx101_pnc` was built on some CI/dev machine that had proper LFS credentials.

`@SiyuLuo` gets `The project you were looking for could not be found` from
`gitlab.mondorobotics.com/sysdev/linux-userspace` on `git lfs pull` — no repo
access. `@Felix` (robot's account) does have access but the robot doesn't have
`git-lfs` installed and its checkout is also LFS-pointer-only.

Proceeding requires **admin adds @SiyuLuo to `sysdev/linux-userspace`** (or issues
a Deploy Token). Once that lands:

```bash
cd ~/Project/linux-userspace
git lfs install --local
git lfs pull
# All third_party/*.so become real .so files, build.sh md_control_rx should
# complete end to end.
```

## Resume checklist (after LFS access is granted)

```bash
# 1. Pull LFS content (~2-3 GB across the whole monorepo; safe to skip large model .onnx by --include)
cd ~/Project/linux-userspace
git lfs pull --include='md_control_rx/**' --include='x5_lib/**' --include='orin_lib/**'
# (the model .onnx under md_audio/ etc. are hundreds of MB — skip if not needed)

# 2. Verify the LFS-pointer .so files are now real ELF
file md_control_rx/md_pnc/third_party/yaml-cpp/lib/libyaml-cpp.so.0.7.0
file x5_lib/hbre/lib/libalog.so.1.0.1
# Expected: "ELF 64-bit LSB shared object, ARM aarch64 ..."
# If still "ASCII text" it means LFS pull didn't cover the file — pull again with a
# broader --include or run `git lfs pull` without filters.

# 3. Confirm the patch files I already applied are intact (git status should show my
# modifications + new files)
git -c 'filter.lfs.process=' status --short | grep -E "md_control_rx|third_party/zmq"

# 4. Build. First run will re-pull the CI Docker image if it's not still cached (it is,
# ~1 GB, from our earlier work).
./build.sh md_control_rx

# 5. Expected artifact:
ls bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc
# should be an aarch64 ELF binary, ~150-200 MB with debug symbols

# 6. Deploy
scp bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc rx101:/tmp/
scp md_control_rx/configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml rx101:/tmp/
ssh rx101 "sudo systemctl stop rx101-pnc && \
    sudo cp /tmp/rx101_pnc /md/bin/rx101_pnc && \
    sudo cp /tmp/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml /md/etc/md_pnc/configs/"

# 7. Flip service to --target robot_stream (drop-in override; least invasive):
ssh rx101 "sudo tee /etc/systemd/system/rx101-pnc.service.d/target-robot-stream.conf <<'EOF'
[Service]
ExecStart=
ExecStart=/md/bin/rx101_pnc --config /md/etc/md_pnc/configs/rx_p2_sonic_gmr_5N_step023800_real.yaml --target robot_stream
EOF
sudo systemctl daemon-reload && sudo systemctl restart rx101-pnc"

# 8. Watch logs for the new banner:
ssh rx101 "sudo journalctl -u rx101-pnc -f | grep -E 'SonicController|RxRobotMotion'"
# Expected: "SonicController: robot-native reference stream ENABLED ..."
#           "RxRobotMotionZmqSubscriber: subscribing to tcp://127.0.0.1:5556 topic=pose dof=27"
```

## Static self-check FIRST (before hooking VLA)

Per deployment doc §5.1: publish a CONSTANT default-stand reference and verify
SONIC enters RL_RUNNING and stays upright without tremor.

On the robot:

```bash
# Install bridge deps (only pyzmq + numpy + pillow; the robot already has python3.10):
python3 -m pip install --user pyzmq numpy pillow websockets

# Copy the bridge to the robot:
scp -r ~/Project/openpi-0_5/examples/rx101_bridge rx101:/tmp/bridge/
# But it needs openpi-client — for the CONSTANT-pose self-check we don't need it.
# Use a stripped-down publisher: publish a fixed default-stand joint vector at 50 Hz.
```

(For the constant self-check, I'd write a `static_publisher.py` that just packs a
DEFAULT_STAND vector and publishes at 50 Hz — much simpler than the full bridge.
Skipping here since it's post-build.)

## If build FAILS after LFS pull

Report the error and I'll iterate. Common issues:

- **New LFS pointer files uncovered**: some `.so` I didn't touch may still be pointer
  files. `find . -name '*.so*' -exec sh -c 'head -c 20 "$1" | grep -q git-lfs && echo "$1"' _ {} \;`
- **My C++ has a typo**: I only compile-tested the ZMQ subscriber; the SonicController /
  robot_config edits were compile-blocked-past. Post-LFS, if there's a compile error,
  it'll be in the surrounding code.
- **Runtime crash / assert on load**: check the config validation in `robot_config.cpp`
  (mutual exclusion `sonic_via_robot_stream` vs `sonic_enable_smpl_teleop`).
