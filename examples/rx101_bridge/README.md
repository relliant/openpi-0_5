# rx101 VLA on SONIC — 部署框架说明

在 rx_p2 人形机器人上部署 openpi 0.5 (pi0.5) VLA 模型，通过 **GR00T-WholeBodyControl ZMQ Protocol v1**（encoder mode 0 = robot-native motion tracking）驱动 SONIC 运控层。

> 本文档描述**当前已跑通**的部署链路。若要"从零开一遍"，按 [§5 Per-session bring-up](#5-per-session-bring-up) 走。

---

## 1. 整体架构

```
                Workstation (GPU)                                Robot (rx101 / Jetson Orin)
    ┌────────────────────────────────────┐        ┌──────────────────────────────────────────┐
    │  scripts/serve_policy.py           │        │  examples/rx101_bridge/bridge.py         │
    │  pi05_rx101_blackbox 29999-step    │        │                                          │
    │                                    │        │  ┌────────────────────────────────────┐  │
    │  input : state[27] + grip[2]       │        │  │ action_stream_loop  ~30 Hz         │  │
    │          + ego_view[480,640,3]     │◄─WS───►│  │   broker.infer(obs) → (29,)/call   │  │
    │          + prompt                  │  8000  │  │   broker 每 16 次触发真 VLA        │  │
    │  output: (16, 29) chunk            │        │  └────────────────┬───────────────────┘  │
    │          = body[27] + grip[2]      │        │                   ▼  latest (29,)         │
    └────────────────────────────────────┘        │  ┌────────────────────────────────────┐  │
                          ▲                       │  │ publish_loop  50 Hz                │  │
                          │                       │  │   low-pass α=0.3 + VLA→SONIC       │  │
                          │ obs = state+jpeg+prompt│  │   permute → ZMQ                    │  │
                          │                       │  └────────────────┬───────────────────┘  │
                ┌─────────┴──────────┐            │                   ▼ tcp://*:5556 loopback │
                │ RealStateSource    │            │  ┌────────────────────────────────────┐  │
                │  = ShmStateSource  │            │  │ rx101_pnc.zmq_stream (patched)     │  │
                │  + MjpegEgoCamera  │            │  │                                    │  │
                └────────────────────┘            │  │  RxRobotMotionZmqSubscriber        │  │
                          ▲                       │  │        ↓                           │  │
                          │                       │  │  SonicRobotMotionBuffer (24f dense,│  │
                          │                       │  │       seqlock-safe)                │  │
                          │                       │  │        ↓                           │  │
                          │                       │  │  SonicController::step()           │  │
                          │                       │  │  (only in RL_RUNNING via gate on   │  │
                          │                       │  │   teleop_active)                   │  │
                          │                       │  │        ↓                           │  │
                          │                       │  │  g1 encoder + decoder → 27-DOF     │  │
                          │                       │  │        ↓                           │  │
                          │                       │  │  V1Client → MCU (30 channels)      │  │
                          │                       │  └────────────────────────────────────┘  │
                          │                       │                                          │
                          │                       │  md_bus shm rings                        │
                          └───────────────────────┼─►  /dev/shm/md_bus_v2__                  │
                                                  │      data_rx__control_fb__raw            │
                                                  │      data_rx__control_aux__feedback      │
                                                  │                                          │
                                                  │  md_webserver :9090                      │
                                                  │      /liveview/cam8750.mjpg (ego cam)    │
                                                  └──────────────────────────────────────────┘
```

**关键设计**：
- **策略层解耦**：VLA server 独立于机器人运行，只吃 obs 吐 action chunk；bridge 负责把 chunk 落地成 50 Hz ZMQ 参考流。
- **双速率**：`action_stream_loop` 30 Hz 拆帧、`publish_loop` 50 Hz 出帧 + 低通。VLA 阻塞（~300 ms/次）只影响 stream，publisher 不断流。
- **原厂 pnc 零改动**：patched binary 独立成 `rx101-pnc-zmq.service`，用 `Conflicts=` 与原 `rx101-pnc.service` 互斥；不改任何原厂 systemd 单元。
- **FSM 门控**：ZMQ 参考流**只在 `FSM=RL_RUNNING` 才生效**（`teleop_active` gate）。STANDING_UP / STANDING_FINISHED 走原厂 SONIC 静立策略。

---

## 2. 组件详解

### 2.1 VLA server（workstation 侧）

- **代码**：`scripts/serve_policy.py`
- **加载**：`policy:checkpoint` 从 orbax 目录反序列化 pi0.5 权重（29999-inference，~6.2 GB）
- **协议**：WebSocket，msgpack + numpy 序列化
- **输入契约**（`src/openpi/policies/rx101_policy.py:Rx101Inputs`）：
  ```python
  {
      "state": np.float32 (27,),                # 27 body joints, VLA order
      "left_gripper_state":  np.float32 (1,),
      "right_gripper_state": np.float32 (1,),
      "images": {"ego_view": np.uint8 (480,640,3)},
      "prompt": str,
  }
  ```
- **输出**：`{"actions": np.float32 (16, 29)}` = 16 步 × (27 body + 2 grip)
- **推理耗时**：steady state ~200-400 ms/次（走 SSH 反向隧道 loopback）

### 2.2 Bridge（robot 侧，本目录）

- **`bridge.py`** — main entry
  - `RealStateSource` = `ShmStateSource` + `MjpegEgoCamera`，实现 `read() / read_ego_jpeg()`
  - `ActionChunkBroker`（openpi_client 官方）：每 `action_horizon` 次 `broker.infer(obs)` 触发一次真 VLA，其余从 cache 切片得单帧
  - `action_stream_loop` (~30 Hz)：拉 obs → broker.infer → 保存最新单帧 (29,)
  - `publish_loop` (50 Hz)：读最新单帧 → 低通滤波 → VLA→SONIC 序 permute → ZMQ send
- **`state_source_shm.py`** — mmap + seqlock 读 `md_bus_v2__data_rx__control_fb__raw` (proprio) + `..._aux__feedback` (gripper)；按 `MCU_BODY_TO_VLA` 排列成 27 维 VLA 序
- **`camera_source_http.py`** — 后台线程用 `cv2.VideoCapture` 拉 `http://127.0.0.1:9090/liveview/cam8750.mjpg`，resize 到 480×640，暴露最新 JPEG 字节
- **`zmq_pose.py`** — GR00T Protocol v1 wire pack/unpack：`[b"pose"] + [1280B JSON header] + [payload]`；字段 `joint_pos[N,27] + joint_vel[N,27] + body_quat[N,4] + frame_index[N]`
- **`joint_maps.py`** — 三序对齐表：VLA / SONIC BFS / MCU wire（30 通道，body 27 + head 2 + reserved 1）
- **`static_publisher.py`** — 静态自检工具：发常量 DEFAULT_STAND_SONIC，不走 VLA。用来在 bridge 未联通时验证 patched pnc + ZMQ + SONIC 通路

### 2.3 Patched md_control_rx（robot 侧，systemd）

**改动点**（相对原厂 `rx101_pnc`）：

| 位置 | 改动 |
|------|------|
| `md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.{hpp,cpp}` | 新增 libzmq 订阅线程 + JSON header 解析 |
| `md_pnc/module/policy/sonic/sonic_robot_motion_buffer.{hpp,cpp}` | 新增稠密帧缓冲（frame_index 严格单调，seqlock 一致性） |
| `md_pnc/module/policy/sonic/sonic_controller.{hpp,cpp}` | `SonicController::step()` 在 `used_robot_stream && teleop_active` 时改从 buffer 取参考 |
| `md_pnc/module/foundation/config/robot_config.{hpp,cpp}` | 新增 6 个 yaml 字段 (`sonic_via_robot_stream` 等) + 与 SMPL 互斥校验 |
| `md_pnc/module/app/main.cpp` | Init / BindBuffer / Shutdown 三件套 |
| `md_pnc/module/app/control_loop.cpp` | 进 RL_RUNNING 时 `setTeleopActive(true)` —— **关键 gate** |

**部署产物**：
- Binary：`/md/bin/rx101_pnc.zmq_stream` (~2.14 MB, aarch64)
- Overlay yaml：`/md/etc/md_pnc/configs/rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml`
- 附加库：`/md/etc/md_pnc/lib/{libalog.so.1,libzmq.so.5,libcamlog.so,libversion.so}`
- Systemd unit：`/etc/systemd/system/rx101-pnc-zmq.service`
  - `Conflicts=rx101-pnc.service`（互斥启动）
  - `ExecStartPre=/bin/sleep 8`（Conflicts= 转换后 MCU/UART 沉降）
  - `ExecStart=/md/bin/rx101_pnc.zmq_stream --config .../real.yaml --target robot_stream`

### 2.4 FSM 状态机

```
       ┌──────────┐   start   ┌─────────────────┐  internal  ┌──────────────────┐
       │ DAMPING  │ ────────► │  STANDING_UP    │ ─────────► │STANDING_FINISHED │
       └──────────┘           │  PREPOSE→POLICY │            └────────┬─────────┘
            ▲                 │  (SONIC static) │                     │ run
            │                 └─────────────────┘                     ▼
            │ halt                                            ┌────────────────┐
            │                                                 │  RL_RUNNING    │
            ├─────────────── halt ─────────────────────────── │  teleop_active │
            │                                                 │   = true       │
            │                                                 │  SONIC 从 buff │
            │  fall / estop / watchdog                        │  取 VLA 参考   │
            │       ┌─────────────────────┐                   └────────────────┘
            └───────┤ EMERGENCY → RECOVERY│
                    └─────────────────────┘
```

命令：`python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py {start|run|halt|recover}`

**重要**：**只有 RL_RUNNING 才会使用 bridge 的 ZMQ 帧**。这是 patched 代码的 gate（`sonic_controller.cpp:672`）。STANDING_UP 期间 SONIC 走原厂静立，不被 bridge 干扰。

---

## 3. 文件布局

### Workstation（本 repo）
```
openpi-0_5/
├── examples/rx101_bridge/          # bridge 源码（本目录）
│   ├── README.md                   # 本文件
│   ├── bridge.py
│   ├── state_source_shm.py
│   ├── camera_source_http.py
│   ├── zmq_pose.py
│   ├── static_publisher.py         # 静态自检（不走 VLA）
│   ├── joint_maps.py
│   └── cpp_patch/                  # md_control_rx patch 材料
│       ├── APPLY.md                # patch 应用步骤
│       ├── RESUME.md               # 断点续传说明
│       └── configs/
│           └── rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml
├── packages/openpi-client/         # WebsocketClientPolicy + ActionChunkBroker
├── src/openpi/policies/rx101_policy.py       # Rx101Inputs/Outputs transforms
├── src/openpi/training/config.py             # TrainConfig("pi05_rx101_blackbox")
├── scripts/serve_policy.py
└── checkpoints/pi05_rx101_blackbox/pi05-rx101-book-v1/29999-inference/
    ├── _CHECKPOINT_METADATA
    ├── assets/                     # norm_stats
    └── params/                     # orbax weights (~6.2 GB)
```

### Robot（rx101）
```
/md/bin/
├── rx101_pnc                        # 原厂 launcher
└── rx101_pnc.zmq_stream             # 我们的 patched binary

/md/etc/md_pnc/
├── configs/
│   ├── rx_p2_sonic_gmr_5N_step023800_real.yaml                         # base
│   └── rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml    # overlay
└── lib/
    ├── libmuml.so, libonnxruntime.so.1, libyaml-cpp.so.0.7, ...       # 原厂
    ├── libalog.so.1, libcamlog.so, libversion.so                       # 补的（patched 依赖）
    └── libzmq.so.5                                                     # 补的

/etc/systemd/system/
├── rx101-pnc.service                # 原厂
└── rx101-pnc-zmq.service            # 我们的（Conflicts=rx101-pnc.service）

/home/mondo/rx101_bridge/            # bridge 部署位置
├── bridge.py
├── state_source_shm.py
├── camera_source_http.py
├── zmq_pose.py
├── joint_maps.py
└── examples/rx101_bridge/           # symlinks 到平级 .py，让 python3 -m 能 import

/home/mondo/openpi-client-src/       # pip install -e 的 openpi_client
/home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py    # FSM 命令工具
```

---

## 4. First-time setup

### 4.1 编译 patched md_control_rx（workstation）

需要 Docker（builder image：`xjp-dockerhub-registry.ap-southeast-1.cr.aliyuncs.com/x5/builder:1.0`）+ Bazel（`build.sh` 自带调用）。

```bash
cd ~/Project/linux-userspace
# 若 patch 未应用，见 examples/rx101_bridge/cpp_patch/APPLY.md
./build.sh md_control_rx
# 产物：bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc (~2.14 MB, aarch64)
```

### 4.2 部署 patched binary（首次）

```bash
# 从 workstation 打包
scp bazel-bin/md_control_rx/md_pnc/module/app/rx101_pnc \
    rx101:/tmp/rx101_pnc.zmq_stream

scp examples/rx101_bridge/cpp_patch/configs/*.yaml \
    rx101:/tmp/

# libalog + libzmq 系列（aarch64）
scp linux-userspace/x5_lib/hbre/lib/lib{alog,camlog,version}* \
    linux-userspace/md_control_rx/md_pnc/third_party/zmq/lib/libzmq.so.5.2.4 \
    rx101:/tmp/lib/

# 在 robot 上 sudo install
ssh rx101 'sudo bash /tmp/rx101_deploy_install.sh'
```

`rx101_deploy_install.sh` 做的事（幂等）：
1. `install /md/bin/rx101_pnc.zmq_stream`
2. `install overlay yaml → /md/etc/md_pnc/configs/`
3. `install libalog/libzmq/libcamlog/libversion → /md/etc/md_pnc/lib/` + symlink
4. `tee /etc/systemd/system/rx101-pnc-zmq.service` + `systemctl daemon-reload`
5. `ldd /md/bin/rx101_pnc.zmq_stream` 验证无 "not found"

### 4.3 安装 bridge 依赖（robot）

```bash
ssh rx101 '
  pip3 install --user pyzmq websockets typing_extensions dm-tree msgpack "numpy>=1.22.4"
'
# openpi_client 从 repo 装（editable）
scp -r ~/Project/openpi-0_5/packages/openpi-client rx101:~/openpi-client-src
ssh rx101 'cd ~/openpi-client-src && pip3 install --user -e .'
```

### 4.4 部署 bridge 代码（robot）

```bash
scp examples/rx101_bridge/{bridge,state_source_shm,camera_source_http,zmq_pose,joint_maps}.py \
    rx101:~/rx101_bridge/

ssh rx101 '
  cd ~/rx101_bridge
  mkdir -p examples/rx101_bridge
  touch examples/__init__.py examples/rx101_bridge/__init__.py
  for f in bridge state_source_shm camera_source_http zmq_pose joint_maps; do
      ln -sf ~/rx101_bridge/$f.py examples/rx101_bridge/$f.py
  done
'
```

---

## 5. Per-session bring-up

假设 §4 都做过。每次上机跑：

### Step 0 —— SSH 隧道

Mac 上确认反向隧道存活（`ssh rx101 → 127.0.0.1:2222`）：
```bash
screen -ls | grep rx101      # 应看到 rx101-bridge screen
# 若死了：screen -wipe 后重启 launch 脚本
```

Workstation 验证：
```bash
ssh rx101 'echo alive'
```

### Step 1 —— 启 VLA server（workstation）

```bash
cd ~/Project/openpi-0_5
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_rx101_blackbox \
    --policy.dir=./checkpoints/pi05_rx101_blackbox/pi05-rx101-book-v1/29999-inference
# 等 ~15 s 到 "server listening on 0.0.0.0:8000"
```

### Step 2 —— 反向隧道 workstation:8000 → robot:8000

```bash
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -R 127.0.0.1:8000:127.0.0.1:8000 rx101 \
  && ssh rx101 'nc -z 127.0.0.1 8000 && echo VLA reachable'
```

### Step 3 —— 启 patched pnc-zmq（robot）

```bash
ssh rx101 'sudo systemctl start rx101-pnc-zmq'
# ExecStartPre sleep 8s + boot ≈ 12 s
sleep 12
ssh rx101 'sudo journalctl -u rx101-pnc-zmq --since=15s | grep -E "robot-native|sub thread up|SonicRobotMotionBuffer"'
```

期望：
```
SonicController: robot-native reference stream ENABLED (future_frames=10 delay=10 dof=27 encoder=1691)
[rx_robot_motion_zmq] sub thread up (connect=tcp://127.0.0.1:5556 topic=pose dof=27)
RxRobotMotionZmqSubscriber: bound to SonicRobotMotionBuffer
```

FSM 应处于 `state=DAMPING`。

### Step 4 —— 启 bridge（robot）

```bash
ssh rx101 '
  cd ~/rx101_bridge
  nohup python3 -m examples.rx101_bridge.bridge \
      --vla-host 127.0.0.1 --vla-port 8000 \
      --prompt "take the book from the bookshelf and hand it to the person" \
      --smoothing-alpha 0.3 \
      > /tmp/bridge.log 2>&1 &
'
sleep 5
ssh rx101 'tail -8 /tmp/bridge.log'
```

期望 bridge 日志：
```
publish loop start (50 Hz -> ZMQ; alpha=0.30)
action stream loop start (~30.0 Hz -> broker.infer)
VLA infer 320.5 ms (chunk call #1, seq=...)
```

### Step 5 —— FSM: DAMPING → STANDING_FINISHED

```bash
ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py start'
sleep 10
ssh rx101 'sudo journalctl -u rx101-pnc-zmq --since=12s | grep "FSM-HB" | tail -2'
```

期望：
```
[FSM-HB] tick=... state=STANDING_FINISHED standing_ready=1 stable=2.02s
```

**此时机器人静立，不应有任何跟 VLA plan 相关的动作**（fix 生效验证）。

### Step 6 —— 安全检查 + 发 `run`

**Halt shell 提前备好**（另开一个窗口）：
```bash
ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py halt'
```

确认无误后：
```bash
ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py run'
# FSM: STANDING_FINISHED → RL_RUNNING
# teleop_active = true → SONIC 开始从 buffer 取 VLA 参考帧
# 机器人按 VLA plan 执行动作
```

监控：
```bash
ssh rx101 'sudo journalctl -u rx101-pnc-zmq -f | grep -E "FSM-HB|EMERGENCY|WATCHDOG|action_norm"'
```

### Step 7 —— 收尾

```bash
# 停 FSM
ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py halt'
# 停 bridge
ssh rx101 'pkill -f "examples.rx101_bridge.bridge"'
# 停服务
ssh rx101 'sudo systemctl stop rx101-pnc-zmq'   # 原厂 rx101-pnc.service 因 Conflicts= 释放后可再启
```

---

## 6. 配置参数速查

### Bridge (`bridge.py` CLI)

| Arg | Default | 说明 |
|-----|---------|------|
| `--vla-host` | `127.0.0.1` | VLA server host（通过反向隧道走本机 loopback） |
| `--vla-port` | `8000` | VLA server port |
| `--prompt` | `"take the book..."` | 任务指令 |
| `--zmq-bind` | `tcp://*:5556` | ZMQ pub bind |
| `--smoothing-alpha` | `0.3` | 低通：`output = α·new + (1-α)·prev`。**小 = 更平滑但滞后大** |
| `--action-horizon` | `16` | broker chunk 长度（必须匹配训练时的 `action_horizon`） |
| `--profile` | `rx101` | 关节布局：`rx101`（27dof 无头）或 `rx2`（29dof，头+夹爪直控） |
| `--mock-state` | off | 用假 obs，网络冒烟用 |
| `--debug-chunk-log` | off | 每 tick 打印原始/发布后的 l/r_elbow（chunk 边界抖动排查用），只加日志量，不改行为 |
| `--debug-web-port` | 不启用 | 见下方"实时诊断仪表盘" |

### 实时诊断仪表盘（`--debug-web-port`，见 `debug_dashboard.py`）

纯只读、不影响控制环路。加这个参数即可：

```bash
ssh rx101b '
  cd ~/rx101_bridge
  nohup python3 bridge.py \
      --profile rx2 --prompt "open the drawer and pick up the black medicine box" \
      --debug-web-port 8765 \
      > /tmp/bridge.log 2>&1 &
'
```

然后在**和机器人同一局域网**的浏览器里直接打开：

```
http://<机器人IP>:<port>/          # 例如 http://10.0.33.164:8765/
```

不需要 SSH 隧道。页面内容：
- 顶部大图 + 关节下拉框：选一个关节看 raw（模型原始输出）/ pub（低通滤波后发布值）/ actual（机器人编码器反馈）三条曲线的细节对比
- "all joints" 网格：27 个 body 关节 + head_yaw/head_pitch，全部同时显示小图，一眼扫全身
- 下方数字读数：夹爪 raw/阈值判断/实际位置，以及机器人当前实际 yaw/roll（从 body_quat 反算）

端口若被占用（比如机器人上 `md_blackbox` 占了 8080），换一个端口即可，比如 `8765`。

### Overlay yaml (`rx_p2_sonic_gmr_5N_step023800_real_overlay_robot_stream.yaml`)

| Key | Value | 说明 |
|-----|-------|------|
| `sonic_enable_smpl_teleop` | `false` | 关闭原 SMPL 遥操路径 |
| `sonic_smpl_via_gmr` | `false` | 关闭 GMR retargeter |
| `sonic_via_robot_stream` | `true` | 启用我们新增的 ZMQ 参考流 |
| `sonic_zmq_connect_addr` | `tcp://127.0.0.1:5556` | 订阅 bridge 的 loopback |
| `sonic_zmq_topic` | `pose` | ZMQ topic |
| `sonic_zmq_expected_dof` | `27` | 校验 payload 维度 |
| `sonic_robot_stream_future_frames` | `10` | encoder 未来帧输入数（legacy1691 encoder） |
| `sonic_robot_stream_playback_delay_frames` | `10` | buffer 播放延迟帧数 |

### Systemd unit (`/etc/systemd/system/rx101-pnc-zmq.service`)

关键项：
- `Conflicts=rx101-pnc.service` — 与原厂服务互斥
- `ExecStartPre=/bin/sleep 8` — Conflicts= 拆原 pnc 后 MCU/UART 沉降时间
- `ExecStart=/md/bin/rx101_pnc.zmq_stream --config .../real.yaml --target robot_stream`

---

## 7. 故障排查

### 隧道断
现象：`ssh rx101 'echo x'` → `Connection reset by peer` 或超时。
```bash
# Mac 端：
screen -ls | grep rx101
screen -wipe   # 清死 socket
# 重启反向隧道 screen（launch 脚本）
```

### VLA server 不可达
```bash
ssh rx101 'nc -z 127.0.0.1 8000'   # 应 succeeded
# 失败：workstation → rx101 反向隧道断了
ssh -f -N -R 127.0.0.1:8000:127.0.0.1:8000 rx101
```

### patched pnc 起不来 / 缺 lib
```bash
ssh rx101 'ldd /md/bin/rx101_pnc.zmq_stream | grep "not found"'
# 若有输出：对应 lib 未部署或不在 LD_LIBRARY_PATH
# 补：libalog.so.1 / libzmq.so.5 应在 /md/etc/md_pnc/lib/
```

### banners 未打
```bash
ssh rx101 'sudo journalctl -u rx101-pnc-zmq | grep "Config: loaded"'
# 应看到 "loaded ... + overlay_robot_stream"
# 若没有 overlay：--target robot_stream 未生效 或 yaml 路径错
```

### `start` 后 PREPOSE 超时
```bash
ssh rx101 'sudo journalctl -u rx101-pnc-zmq -f | grep STAND-HB'
# 关注 prepose_err / prepose_joint
# prepose_err 一直 > tolerance：物理姿态问题（把腿摆正到 default_pos 附近）
# 不是 bridge / VLA 的问题
```

### `run` 前机器人自己动
- **不应该发生**（已 fix）。若发生：
  1. 立刻 halt
  2. 检查 `sonic_controller.cpp` 中 `used_robot_stream` 分支应带 `&& teleop_active`
  3. 检查 `control_loop.cpp` 应在 `sonic_via_robot_stream && case_entry_state==RL_RUNNING` 时 `setTeleopActive(true)`

### 动作一卡一卡
1. Bridge log 里看 `VLA infer X ms`。若 X > 300 ms 是常态，考虑：
   - 减小 `--smoothing-alpha`（如 0.2 或 0.15，更平滑）
   - VLA server 侧用更快 GPU / 关掉其他任务
2. Chunk 边界跳跃：`ActionChunkBroker` 保证 chunk 内连续，chunk 间取决于 VLA 时序一致性 —— 靠低通滤波抹平

### EMERGENCY / UART watchdog
```bash
ssh rx101 'sudo journalctl -u rx101-pnc-zmq --since=1min | grep -E "EMERGENCY|WATCHDOG|SAFETY"'
# 常见触发：
#   - UART_WATCHDOG 在 service restart 后 3s 内 —— 8s ExecStartPre 应挡住
#   - action_norm_abort：VLA 输出幅度过大 —— 检查 obs 是否合理（相机没黑、state 没 NaN）
```

Recovery：
```bash
ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py recover'
# 一次即可。多次会被拒绝并 5s 后 RECOVERY_FAILED → STOP
```

---

## 8. 安全须知

1. **首测必须 fall harness**。VLA 从悬吊姿态起可能输出剧烈动作。
2. **halt 命令随时备好**：
   ```bash
   ssh rx101 'python3 /home/mondo/linux-userspace/soc/md-control-rx/scripts/send_rx101_cmd.py halt'
   ```
3. **物理 E-STOP** 是最终防线。
4. **C++ 侧 SafetyFilter** 已开：`action_clip=1(range=10)`, `action_norm_abort=1(thr=8, frames=3)`, `imu_gate_stop=1`, `mcu_estop_stop=1`, `uart_watchdog=1`。
5. Recovery 失败要重启：`sudo systemctl restart rx101-pnc-zmq`。

---

## 9. 静态自检工具（联调阶段用）

在 patched pnc 联调、bridge 还没写通时，用 `static_publisher.py` 发常量 `DEFAULT_STAND_SONIC` 参考流，验证 ZMQ 通路 + FSM + SONIC 编解码：

```bash
# 在 robot 上（用 static_publisher 代替 bridge）
python3 ~/rx101_bridge/static_publisher.py --bind tcp://*:5556 --hz 50 &
# 然后走 Step 3 → Step 5 → Step 6
# 机器人应保持 default_pos 静立
```

这个走通说明 patched pnc + ZMQ + SONIC 全通，问题必在 bridge / VLA 层。

---

## 10. 已知问题 / 后续 TODO

- **VLA 推理 ~200-400 ms/次**：训练时 `action_horizon=16 @ 30 Hz` 恰好覆盖 533 ms，chunk 边界紧。若 VLA 更慢 → 后续 chunk 有 hold-last 帧，动作可能顿。改进方向：
  - 训练 `action_horizon` 加大（32 / 64）—— 需重训
  - 用更快模型（pi0_fast）或量化
  - Bridge 预取：让 broker 在 chunk 剩余 < N 帧时后台预取下一 chunk（现在 `openpi_client.action_chunk_broker` 是同步的，需自己写 async 包装）
- **`ring publish FAILED topic=wire_frame`** 告警：非致命，md_blackbox 订阅端未及时 re-attach。可忽略。
- **训练分布假设**：数据集里机器人从 "take book from bookshelf" 任务起始姿收集。若在悬吊/极限姿下 `run`，VLA 可能输出不合理动作（OOD）。

---

## 11. RX2 29-DOF head/gripper 直控（checkpoints_rx2）

`pi05_rx2_blackbox_drawer_v2` 等 RX2 checkpoint 用的是 **同一台 rx101/RX-P2 硬件**，但训练数据的 `observation.state`/`action.wbc` 是 29 维：**27 个 body joint（跟 rx101 的 `VLA_ORDER` 逐字段一致）+ `head_yaw` + `head_pitch`**。夹爪仍是独立的 `action.left_gripper`/`action.right_gripper` 字段。

### 11.1 架构决策

body 27 维 **完全复用**现有 SONIC robot_stream 通路，不做任何改动。head + gripper 走机器人上原本就有、但只认 PICO 遥操（`kLcm`/`kDds`）的 **`PicoAuxGate`** 直控通道 —— 这条通道在此之前**跟我们的 ZMQ robot_stream 完全没打通**（gripper 对 rx101 部署也从未真正下发过）。

新增枚举 `SonicTeleopSource::kRobotStream`，让 `PicoAuxGate` 除了接受 PICO 的 `kDds` 之外也接受我们的 ZMQ 通路，复用它已经调好的 rate-limit / deadband / clamp 逻辑，不用重新发明。

```
VLA 29-dim state ← 27 body(现有) + head_yaw/head_pitch(新增，来自 MCU ch 27,28 原始反馈)
VLA 31-dim action → body[0:27](现有 ZMQ→SONIC 通路)
                   → head[27:29] + grip[29:31](新增 ZMQ aux 字段→PicoAuxGate)
```

### 11.2 C++ 改动（`linux-userspace/md_control_rx/`）

| 文件 | 改动 |
|------|------|
| `md_pnc/module/policy/sonic/sonic_teleop_window.hpp` | 新增 `SonicTeleopSource::kRobotStream = 3` |
| `md_pnc/module/policy/sonic/sonic_robot_motion_buffer.{hpp,cpp}` | `RobotMotionFrame` 加 `aux_valid/head_yaw_rad/head_pitch_rad/gripper_*_mask`；新增 `latestAux()` |
| `md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.cpp` | 解析可选的 `head_yaw/head_pitch/gripper_enable_mask/gripper_closed_mask` 字段（缺失时 fail-closed 到 `aux_valid=false`，body 帧仍正常处理，不丢包） |
| `md_pnc/module/policy/sonic/sonic_controller.cpp` | `teleopReadiness()` 在 `via_robot_stream_` 有新鲜 aux 时返回 `source=kRobotStream` |
| `md_pnc/module/app/control_loop.cpp` | `pico_aux_gate.update()` 的 `teleop_active` 参数扩展为 `(FSM==TELEOP) \|\| (sonic_via_robot_stream && FSM==RL_RUNNING)` |
| `md_pnc/module/protocol/gates/pico_aux_gate.cpp` | `source_accepted` 从只认 `kDds` 扩展为 `kDds \|\| kRobotStream` |

### 11.3 Python 改动（本目录）

| 文件 | 改动 |
|------|------|
| `zmq_pose.py` | `_DTYPE_TAG` 加 `uint8→"u8"`（gripper mask 是位掩码，不能走 float32 fallback）；文档补充可选 aux 字段 |
| `joint_maps.py` | `RX2_VLA_ORDER = VLA_ORDER + ("head_yaw","head_pitch")`，29 维 |
| `state_source_shm.py` | `ProprioSnapshot` 加 `head_yaw_pitch`（MCU ch 27,28 原始反馈，不做符号翻转） |
| `bridge.py` | 新增 `DeployProfile`（`PROFILE_RX101`/`PROFILE_RX2`）+ `--profile {rx101,rx2}` CLI；`_build_obs` 按 profile 拼 state；`publish_loop` 按 profile 切 action 并发 head/grip aux 字段 |

### 11.4 关键坑：head 符号翻转

机器人 yaml `pico_aux.head.sign: [1.0, -1.0]`（yaw, pitch）。`PicoAuxGate` 内部会用这个 sign 乘一遍收到的值再下发。我们发送前**用同一个 sign 预先乘一遍**（`bridge.py:HEAD_SIGN_YAW_PITCH`），两次乘负负得正，抵消掉，这样我们发的 wire 值就是"期望的绝对 MCU 目标角"，跟 body 27 维的语义一致。

**如果机器人 yaml 里 `pico_aux.head.sign` 改了，这个常量必须同步改**，否则 pitch 轴会往错误方向转到机械限位。**首次上电务必先跑 §11.5 静态自检，肉眼确认方向对再接 VLA**。

### 11.5 Gripper 二值化

`PicoAuxGate` 的 gripper 只支持开/合二值（`gripper_closed_mask`），不支持连续位置。VLA 输出连续弧度值，bridge 用阈值判断：

```python
GRIPPER_OPEN_RAD = -2.25       # 来自机器人 yaml pico_aux.grippers.open_rad
GRIPPER_CLOSED_RAD = -0.10     # 来自机器人 yaml pico_aux.grippers.closed_rad
GRIPPER_CLOSE_THRESHOLD_RAD = -1.175  # 中点
```

### 11.6 静态自检（上 VLA 前必做）

不用 VLA，手动构造常量 head/grip 目标，确认：
1. `PicoAuxGate` 真的从 `kRobotStream` 激活（看 `[PICO-AUX-EVENT]` 日志 `source=3`）
2. head 转动方向跟指令方向一致（正 yaw → 头右转 or 左转，自己核对物理方向）
3. gripper 二值切换正常

```bash
ssh rx101 'cd ~/rx101_bridge && python3 -c "
import time
from examples.rx101_bridge import zmq_pose
import numpy as np

pub = zmq_pose.PosePublisher(bind_addr=\"tcp://*:5556\")
# 常量 body=default stand, head_yaw=+0.3rad, head_pitch=0, grip 双开
default_sonic = np.zeros((1,27), dtype=np.float32)  # 或用 static_publisher 里的 DEFAULT_STAND_SONIC
for i in range(200):
    pub.send({
        \"joint_pos\": default_sonic,
        \"joint_vel\": np.zeros((1,27), dtype=np.float32),
        \"body_quat\": np.array([[1,0,0,0]], dtype=np.float32),
        \"frame_index\": np.array([i], dtype=np.int64),
        \"head_yaw\": np.array([0.3], dtype=np.float32),
        \"head_pitch\": np.array([0.0], dtype=np.float32),
        \"gripper_enable_mask\": np.array([3], dtype=np.uint8),
        \"gripper_closed_mask\": np.array([0], dtype=np.uint8),
    })
    time.sleep(0.02)
"'
```

配合走 §5 Step 3 → Step 5（`start` → `STANDING_FINISHED`）→ `run`，观察：

```bash
ssh rx101 'sudo journalctl -u rx101-pnc-zmq -f | grep -E "PICO-AUX-EVENT|PICO-AUX-TX-HB"'
```

`source=3` 且 `head=+0.3,+0.0` 跟发的值方向一致，才算通过。**方向错了立刻 halt，别再往下测。**

### 11.7 真机跑 rx2 checkpoint

确认静态自检通过后，Step 4 换成：

```bash
python3 -m examples.rx101_bridge.bridge \
    --vla-host 127.0.0.1 --vla-port 8000 \
    --prompt "open the drawer and pick up the black medicine box" \
    --profile rx2 \
    --smoothing-alpha 0.3
```

Workstation 侧 `serve_policy.py` 换成 rx2 checkpoint：

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_rx2_blackbox_drawer_v2 \
    --policy.dir=./checkpoints_rx2/pi05_rx2_blackbox_drawer_v2/pi05-rx2-drawer-medicine-v2/8000
```

---

## 12. 关键代码索引

- `bridge.py:Bridge.action_stream_loop` — broker 单帧流
- `bridge.py:Bridge.publish_loop` — 50 Hz 低通 + ZMQ（rx2 profile 下额外发 head/grip aux）
- `bridge.py:DeployProfile` — rx101 vs rx2 路由差异
- `state_source_shm.py:ShmStateSource.read` — mmap seqlock + MCU→VLA permute + head 反馈
- `camera_source_http.py:MjpegEgoCamera` — cv2.VideoCapture 后台线程
- `joint_maps.py:MCU_BODY_TO_VLA` / `RX2_VLA_ORDER` — joint 置换表
- `zmq_pose.py:PosePublisher.send` — Protocol v1 wire pack

Patched C++ 关键点（in `linux-userspace/md_control_rx/`）：
- `md_pnc/module/policy/sonic/sonic_controller.cpp` —— `used_robot_stream` 分支 + `teleop_active` gate；`teleopReadiness()` 的 robot_stream aux 分支
- `md_pnc/module/app/control_loop.cpp` —— 进 RL_RUNNING 时 `setTeleopActive(true)`；`pico_aux_gate.update()` 的 `teleop_active` 扩展
- `md_pnc/module/protocol/rx_robot_motion_zmq_subscriber.cpp` —— ZMQ 订阅 + 1280B JSON header 解析 + 可选 head/grip aux 字段
- `md_pnc/module/policy/sonic/sonic_robot_motion_buffer.cpp` —— 稠密帧 buffer + `latestAux()`
- `md_pnc/module/protocol/gates/pico_aux_gate.cpp` —— `source_accepted` 扩展到 `kRobotStream`
- `md_pnc/module/policy/sonic/sonic_teleop_window.hpp` —— `SonicTeleopSource::kRobotStream` 枚举

参考文档：
- `~/Tools/docs/vla-on-sonic-deploy.md` —— 更早期的部署设计笔记
- `examples/rx101_bridge/cpp_patch/APPLY.md` —— C++ patch 应用步骤
- `examples/rx101_bridge/cpp_patch/RESUME.md` —— 断点续传（LFS 权限恢复后）
