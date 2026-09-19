# Tienkung VLA Bridge

ROS2 bridge connecting OpenPI VLA models to Tienkung (天工行者·无疆) humanoid robots.

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────┐
│   VLA Server    │◄──────8000─────────│    Bridge    │
│  (openpi serve) │                    │   (ROS2 node)│
└─────────────────┘                    └───────┬──────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                    /arm/cmd_pos      /inspire_hand/ctrl/*   /camera_head/*
                          │                    │                    │
                    ┌─────▼────────────────────▼────────────────────▼───┐
                    │          Tienkung Robot (ROS2 stack)              │
                    └───────────────────────────────────────────────────┘
```

### Key Features

- **Single-threaded control loop** at configurable rate (default 10 Hz)
- **Arm control** via `/arm/cmd_pos` (14 joints: left 7 + right 7) in radians
- **Hand control** via `/inspire_hand/ctrl/{left,right}_hand` (12 DOF: 6 per hand) in percentage
- **State feedback** from `/arm/status` and `/inspire_hand/state/*`
- **Camera** from `/camera_head/image_raw/compressed`
- **Optional action smoothing** via exponential moving average (EMA)

## Prerequisites

### 1. ROS2 Installation

Tienkung SDK requires ROS2 Humble or later:

```bash
# Ubuntu 22.04
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### 2. Tienkung SDK

Install `bodyctrl_msgs` package from Tienkung ROS2 SDK:

```bash
# Follow https://docs.ubtrobot.com/walker-tienkung/docs/category/sdk文档/
# Ensure these packages are available:
ros2 pkg list | grep bodyctrl_msgs
ros2 pkg list | grep sensor_msgs
```

### 3. OpenPI Dependencies

```bash
cd /path/to/openpi-0_5
pip install -e .
pip install rclpy pillow numpy
```

## Usage

### Step 1: Start VLA Server

Start the OpenPI serving backend:

```bash
# Example: serve pi05_tienkung_pick_place checkpoint
openpi serve \
  --checkpoint checkpoints/pi05_tienkung_pick_place/0719_tienkung_pick_place_full \
  --host 0.0.0.0 \
  --port 8000
```

### Step 2: Prepare Robot

Enter **half-body control mode** on Tienkung:

**Method A** (development):
```bash
# Stop auto-start service
sudo systemctl stop proc_manager.service

# Manually start body control only (no motion control)
ros2 launch body_control body.launch.py
```

**Method B** (operational, requires software >= 2.0.5.2):
1. Press **A** for self-check
2. Land normally
3. Use remote: **G-middle + E-middle + F-down**, then **long-press A**
4. Wait for short beep → half-body control mode active

### Step 3: Run Bridge

On the robot or a machine with ROS2 network access:

```bash
cd /path/to/openpi-0_5
export PYTHONPATH=$(pwd):$PYTHONPATH
source /opt/ros/humble/setup.bash

python examples/tienkung_bridge/bridge.py \
  --vla-host 127.0.0.1 \
  --vla-port 8000 \
  --prompt "pick up the apple and place it" \
  --rate 10.0 \
  --smoothing-alpha 0.3 \
  --arm-speed 0.5 \
  --arm-current 8.0
```

### Arguments

| Argument           | Default                      | Description                                      |
|--------------------|------------------------------|--------------------------------------------------|
| `--vla-host`       | `127.0.0.1`                  | VLA server hostname                              |
| `--vla-port`       | `8000`                       | VLA server port                                  |
| `--prompt`         | `"pick up the apple..."`     | Task instruction for VLA                         |
| `--rate`           | `10.0`                       | Control loop frequency (Hz)                      |
| `--smoothing-alpha`| `0.3`                        | EMA smoothing: output = α×new + (1-α)×prev       |
| `--arm-speed`      | `0.5`                        | Arm joint speed limit (rad/s)                    |
| `--arm-current`    | `8.0`                        | Arm joint current limit (A)                      |
| `--mock`           | `False`                      | Skip bodyctrl_msgs import (testing only)         |

## Robot State Mapping

### VLA Model Order (26 DOF)

```
[0:7]   Left arm    (rad)        Motor IDs: 11-17
[7:13]  Left hand   (rad*)       Finger IDs: 1-6  *converted to % for ROS2
[13:20] Right arm   (rad)        Motor IDs: 21-27
[20:26] Right hand  (rad*)       Finger IDs: 1-6  *converted to % for ROS2
```

### Hand Conversion

The model outputs hand joints in **radians** (trained on gripper angles), but Tienkung expects **percentage [0, 1]**:

- `0.0` = fully closed (grip)
- `1.0` = fully open

Conversion (see `joint_maps.py`):

```python
# Model gripper range: [-2.25, -0.10] rad
percentage = (rad - (-2.25)) / (-0.10 - (-2.25))
percentage = clip(percentage, 0.0, 1.0)
```

Adjust `GRIPPER_OPEN_RAD` / `GRIPPER_CLOSED_RAD` if your model uses different ranges.

## Troubleshooting

### 1. No camera image

Check camera topic:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera_head/image_raw/compressed
```

If missing, verify camera node is running or change topic in `bridge.py:105`.

### 2. Arm not moving

Check arm status:

```bash
ros2 topic echo /arm/status --once
```

Verify motors are online (error ≠ 33072) and within limits.

Test manual control:

```bash
ros2 topic pub /arm/cmd_pos bodyctrl_msgs/msg/CmdSetMotorPosition \
  "{cmds: [{name: 11, pos: 0.1, spd: 0.2, cur: 8.0}]}"
```

### 3. Hand percentage looks wrong

The model's gripper output range may differ. Adjust in `joint_maps.py:35-36`:

```python
GRIPPER_OPEN_RAD = -0.10    # model output for "open"
GRIPPER_CLOSED_RAD = -2.25  # model output for "closed"
```

Verify by echoing raw VLA output and hand state:

```bash
# In bridge.py _control_loop, add before smoothing:
LOG.info("Raw hand output: left=%s, right=%s", l_hand, r_hand)
```

### 4. VLA inference too slow

Lower control rate or run VLA server on GPU:

```bash
python examples/tienkung_bridge/bridge.py --rate 5.0  # 5 Hz
```

Or use action horizon caching (requires modifying bridge to use `ActionChunkBroker`).

## Safety Notes

- **Always supervise** robot during VLA control
- Use **low arm speeds** initially (`--arm-speed 0.3`)
- Keep **E-stop** accessible
- Monitor `/arm/status` for errors and temperature
- Verify hand/arm limits before deploying new checkpoints

## Extending

### Add force/impedance control

Replace `/arm/cmd_pos` with `/arm/cmd_ctrl` (force-position hybrid):

```python
# In bridge.py, use bodyctrl_msgs/msg/CmdMotorCtrl
msg.cmds = [MotorCtrl(name=id, kp=30, kd=10, pos=p, spd=0, tor=0) for id, p in ...]
```

### Use ActionChunkBroker

For action horizons > 1, wrap the policy:

```python
from openpi_client.action_chunk_broker import ActionChunkBroker
self.broker = ActionChunkBroker(self.policy, action_horizon=16)
output = self.broker.infer(obs)  # returns single-frame dict
```

Increase control rate to match dataset FPS (e.g., 30 Hz).

### Add base/leg control

Extend `joint_maps.py` to include waist (ID 31) and legs (IDs 51-56, 61-66). Update `VLA_*_SLICE` and model `action_dim` accordingly.

## Reference

- Tienkung SDK: https://docs.ubtrobot.com/walker-tienkung/docs/category/sdk文档/
- OpenPI repo: https://github.com/Physical-Intelligence/openpi
- ROS2 Humble docs: https://docs.ros.org/en/humble/
