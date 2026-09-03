"""ZMQ Protocol v1 (joint-based motion tracking) publisher + decoder.

Wire format follows GR00T-WholeBodyControl / gear_sonic zmq_planner_sender.py:

    [topic_bytes] + [1280 B JSON header padded '\\x00'] + [raw payload]

The JSON header describes the fields:

    {"v": 1, "endian": "le", "count": 1, "fields":
        [{"name": <str>, "dtype": "f32|f64|i32|i64|bool", "shape": [...]}, ...]}

Payload is the concatenation of each array's C-contiguous little-endian bytes,
in the same order as `fields`. Sent as one `socket.send()` frame (no multipart).

Protocol v1 required fields for SONIC encoder mode 0 (joint-based tracking):

    joint_pos   [N, 27]  f32  IsaacLab / SONIC BFS order, rad
    joint_vel   [N, 27]  f32  rad/s
    body_quat   [N, 4]   f32  root (base_link) orientation, wxyz
    frame_index [N]      i64  monotonic

    (GR00T's public G1 platform is 29-DoF; our rx_p2 is 27-DoF. The C++
    subscriber reads shape from the header — it must accept 27 or match
    the deployed model.)

Optional aux fields (§ RX2 29-dof head/gripper direct-control passthrough,
consumed by PicoAuxGate via SonicTeleopSource::kRobotStream — NOT part of the
SONIC encoder input). Omit all four together for a body-only (rx101 27-dof)
publisher; the C++ subscriber treats partial/absent aux as no-head-control,
never as a malformed packet:

    head_yaw            [N]  f32  rad, robot-frame (same convention as PICO)
    head_pitch          [N]  f32  rad, robot-frame
    gripper_enable_mask [N]  u8   bit0=left, bit1=right
    gripper_closed_mask [N]  u8   meaningful only where enable bit is set

Callers are responsible for producing body values in SONIC BFS joint order
(see `joint_maps.SONIC_POLICY_ORDER`) — this module is transport-only.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from typing import Iterable

import numpy as np
import zmq


HEADER_SIZE = 1280
DEFAULT_TOPIC = b"pose"

_DTYPE_TAG: dict[np.dtype, str] = {
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
    np.dtype("bool"): "bool",
    # gripper_enable_mask / gripper_closed_mask are bitmasks (0x00-0x03), not
    # booleans — must round-trip as raw bytes, never through the float32 fallback.
    np.dtype("uint8"): "u8",
}
_TAG_DTYPE: dict[str, np.dtype] = {v: k for k, v in _DTYPE_TAG.items()}


def _pack_field(arr: np.ndarray) -> tuple[str, bytes]:
    """Force little-endian C-contiguous, return (dtype_tag, bytes)."""
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _DTYPE_TAG:
        # Fallback: cast unknown dtypes to float32, matches gear_sonic behavior.
        arr = arr.astype(np.float32)
    return _DTYPE_TAG[arr.dtype], arr.tobytes()


def pack_pose_message(data: dict[str, np.ndarray], *, topic: bytes = DEFAULT_TOPIC, version: int = 1) -> bytes:
    """Pack a dict of numpy arrays into one ZMQ frame per GR00T Protocol v1.

    Field ordering is dict insertion order — both ends must agree; nothing in
    the header enforces canonicality.
    """
    fields = []
    buffers: list[bytes] = []
    for name, value in data.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"field {name!r} must be a numpy.ndarray, got {type(value).__name__}")
        dtype_tag, buf = _pack_field(value)
        fields.append({"name": name, "dtype": dtype_tag, "shape": list(value.shape)})
        buffers.append(buf)
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"header {len(header_bytes)} B exceeds {HEADER_SIZE} B budget")
    header_padded = header_bytes.ljust(HEADER_SIZE, b"\x00")
    return topic + header_padded + b"".join(buffers)


@dataclasses.dataclass
class DecodedMessage:
    topic: bytes
    version: int
    endian: str
    fields: dict[str, np.ndarray]


def unpack_pose_message(msg: bytes, *, topic_prefix: bytes = DEFAULT_TOPIC) -> DecodedMessage:
    """Inverse of `pack_pose_message`. Raises ValueError on malformed input."""
    if not msg.startswith(topic_prefix):
        raise ValueError(f"topic mismatch: expected prefix {topic_prefix!r}, got {msg[:16]!r}")
    header_start = len(topic_prefix)
    header_end = header_start + HEADER_SIZE
    if len(msg) < header_end:
        raise ValueError(f"message too short ({len(msg)} B) for header window")
    header_raw = msg[header_start:header_end].rstrip(b"\x00")
    header = json.loads(header_raw.decode("utf-8"))
    payload = msg[header_end:]
    offset = 0
    out: dict[str, np.ndarray] = {}
    for f in header["fields"]:
        dtype = _TAG_DTYPE[f["dtype"]]
        shape = tuple(f["shape"])
        n = int(np.prod(shape)) if shape else 1
        nbytes = n * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ValueError(f"payload truncated at field {f['name']!r}")
        arr = np.frombuffer(payload, dtype=dtype, count=n, offset=offset).reshape(shape)
        out[f["name"]] = arr.copy()
        offset += nbytes
    if offset != len(payload):
        raise ValueError(f"trailing {len(payload) - offset} B unaccounted after fields")
    return DecodedMessage(topic=topic_prefix, version=header["v"], endian=header["endian"], fields=out)


class PosePublisher:
    """Thin wrapper: zmq.PUB socket bound to `bind_addr` (default tcp://*:5556).

    Multiple subscribers can connect. We do NOT track subscriber count; a slow
    or absent subscriber has zero effect on publish rate. High-water mark
    defaults to 1 so we drop old frames rather than back-pressuring the sender.
    """

    def __init__(self, bind_addr: str = "tcp://*:5556") -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind_addr)

    def send(self, data: dict[str, np.ndarray], *, topic: bytes = DEFAULT_TOPIC, version: int = 1) -> None:
        self._sock.send(pack_pose_message(data, topic=topic, version=version), zmq.NOBLOCK if False else 0)

    def close(self) -> None:
        self._sock.close(linger=0)


class PoseSubscriber:
    """Thin wrapper: zmq.SUB socket connected to `connect_addr`.

    `conflate=True` (default) keeps only the latest message in the recv queue,
    matching GR00T's `--zmq-conflate` behavior for tracking-latency reasons.
    """

    def __init__(
        self,
        connect_addr: str = "tcp://127.0.0.1:5556",
        *,
        topic: bytes = DEFAULT_TOPIC,
        conflate: bool = True,
    ) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        if conflate:
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, topic)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(connect_addr)
        self._topic = topic

    def recv(self, timeout_ms: int | None = None) -> DecodedMessage | None:
        if timeout_ms is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        try:
            raw = self._sock.recv()
        except zmq.error.Again:
            return None
        return unpack_pose_message(raw, topic_prefix=self._topic)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# CLI: fake receiver for loopback smoke test.
# ---------------------------------------------------------------------------

def _fake_receiver_main(connect_addr: str, n: int) -> None:
    import time

    sub = PoseSubscriber(connect_addr, conflate=False)  # keep all for debug
    print(f"listening on {connect_addr}, waiting for {n} messages...")
    prev_frame_idx: int | None = None
    t_first: float | None = None
    for i in range(n):
        msg = sub.recv(timeout_ms=5000)
        if msg is None:
            print(f"[{i}] timeout")
            continue
        if t_first is None:
            t_first = time.monotonic()
        fs = msg.fields
        fi = int(fs["frame_index"][0]) if "frame_index" in fs else -1
        gap = "" if prev_frame_idx is None else f" Δ={fi - prev_frame_idx}"
        prev_frame_idx = fi
        summary = ", ".join(f"{k}{v.shape}@{v.dtype}" for k, v in fs.items())
        print(f"[{i}] v={msg.version} frame={fi}{gap}  {summary}")
        if i == 0:
            print(f"    joint_pos[0, :6] = {fs['joint_pos'].flatten()[:6]}")
    if t_first is not None:
        print(f"received {n} messages in {time.monotonic() - t_first:.2f}s")
    sub.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://127.0.0.1:5556")
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    _fake_receiver_main(args.connect, args.n)
