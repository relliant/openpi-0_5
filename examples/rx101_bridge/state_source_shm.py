"""Read robot proprio state from md_bus shm rings (bridge side, on-robot).

Two rings:
  * `md_bus_v2__data_rx__control_fb__raw`   — 30-DOF motor + IMU snapshot (~30-60Hz).
  * `md_bus_v2__data_rx__control_aux__feedback` — 2 grippers (position/vel/torque).

Ring layout (reverse-engineered from live segment; see notes below):
  * Segment prefix: 252 B ring metadata (atomics + wake_word etc.). We skip it entirely.
  * Then N slots of `slot_size` bytes each.
  * Each slot: 24 B seqlock-style guard header, then payload per `.msg.yaml` (aligned).
      +0   pub_ts_ns  u64   (seqlock start guard)
      +8   pub_ts_ns  u64   (seqlock end guard — same value when the slot is
                             stable; different value or half-torn write = mid-update)
      +16  seq        u64   (increment counter; even == published, odd == in-flight)
      +24  payload    ...   (per the schema, little-endian, aligned)

To read the latest published frame:
  1. Scan every slot's [0..8) as a u64 candidate timestamp; pick the max — that
     is the newest publisher slot.
  2. On that slot, read guard0 = mm[slot+0..8), payload, guard1 = mm[slot+8..16).
     If guard0 == guard1 and both != 0, payload is torn-free. Otherwise retry.

The writer keeps a fixed slot ring so torn reads should be rare; a couple of
retries hides the residue. We fall back to the last-good snapshot on repeated
tears so downstream never sees NaN.

Publisher is `rx101_pnc` (mapped as writer) and `md_blackbox` also mmaps for
logging; both are compatible readers.
"""

from __future__ import annotations

import dataclasses
import mmap
import os
import struct
import time
from typing import Optional

import numpy as np

from examples.rx101_bridge import joint_maps


# ─────────────────── Ring geometry (control_fb__raw) ────────────────────
FB_RAW_PATH = "/dev/shm/md_bus_v2__data_rx__control_fb__raw"
FB_RAW_HDR_BYTES = 252
FB_RAW_SLOT_BYTES = 1024
FB_RAW_GUARD_BYTES = 24

# rx_control_fb_raw payload offsets (from schema, aligned LE, after the 24 B guard).
# 30 = kMotorChannels; head takes ch 27..28, reserved is 29 — body VLA uses ch 0..26.
_PL_TS_NS   = 0     # u64
_PL_CYCLE   = 8     # u64
_PL_TS_US   = 16    # u64
_PL_POS_I16 = 24    # i16 × 30
_PL_VEL_I16 = 84
_PL_TRQ_I16 = 144
_PL_REF_TRQ = 204
_PL_TEMP    = 264   # i8 × 30
_PL_ERRCODE = 294   # u16 × 30, ends 354
# 2-byte pad here: err_code (u16) ends at 354; next field is f32 array which needs 4-byte alignment
_PL_POS_F   = 356   # f32 × 30   ← what we want (radians), ends 476
_PL_GYRO    = 476   # f32 × 3, ends 488
_PL_ACCEL   = 488   # f32 × 3, ends 500
_PL_QUAT    = 500   # f32 × 4 wxyz, ends 516
_PL_MOT_ERR = 516   # u32
_PL_MCU_ERR = 520   # u32
_PL_MCU_STATE = 524 # u8

# ─────────────────── Ring geometry (control_aux__feedback) ──────────────
AUX_FB_PATH = "/dev/shm/md_bus_v2__data_rx__control_aux__feedback"
AUX_FB_HDR_BYTES = 252     # assumed same md_bus format
AUX_FB_SLOT_BYTES = 128    # payload ~62 B; slot rounded up to 128 (verified at runtime)
AUX_FB_GUARD_BYTES = 24

# rx_control_aux_feedback payload offsets (aligned LE).
_AUX_TS_NS      = 0    # u64
_AUX_FB_TS_US   = 8    # u64
_AUX_HAS_FB     = 16   # u8
_AUX_CAP_FLAGS  = 17   # u8
_AUX_MODE_FLAGS = 18   # u8
_AUX_GRIP_MASK  = 19   # u8
_AUX_GRIP_VALID = 20   # u8
_AUX_HEAD_MASK  = 21   # u8
_AUX_GRIP_POS   = 24   # f32 × 2  ← left,right gripper positions
_AUX_GRIP_VEL   = 32
_AUX_GRIP_TRQ   = 40


# ─────────────────────────────── Data class ──────────────────────────────
@dataclasses.dataclass(frozen=True)
class ProprioSnapshot:
    """Proprio state in VLA joint order + gripper + body pose."""
    joint_pos_vla: np.ndarray            # (27,) rad, body only
    head_yaw_pitch: np.ndarray           # (2,) rad, raw MCU feedback (ch 27,28), no sign flip.
                                          # Only meaningful for RX2 29-dof deploys; rx101
                                          # profiles ignore it. Same "absolute MCU joint
                                          # radians" convention as joint_pos_vla — NOT PICO's
                                          # pre-sign "measured" convention (see joint_maps.py
                                          # RX2_VLA_ORDER / bridge.py head_sign comment).
    left_gripper_pos: float              # rad or normalized (as trained)
    right_gripper_pos: float
    root_quat_wxyz: np.ndarray           # (4,) IMU body_quat
    timestamp_ns: int                    # from control_fb__raw
    cycle: int                           # publisher counter
    mcu_state: int                       # for diagnostics


# ─────────────────────── Ring readers (mmap seqlock) ─────────────────────
class _SeqRingReader:
    """mmap a POSIX shm segment as read-only, scan for freshest slot, seqlock-safe."""

    def __init__(self, path: str, hdr_bytes: int, slot_bytes: int, guard_bytes: int):
        self.path = path
        self.hdr = hdr_bytes
        self.slot = slot_bytes
        self.guard = guard_bytes
        self._fd = open(path, "rb")
        sz = os.fstat(self._fd.fileno()).st_size
        self._mm = mmap.mmap(self._fd.fileno(), sz, prot=mmap.PROT_READ)
        # Estimate slot count. Segment tail holds per-sub cursors + wake_word;
        # scanning slightly-past-end is safe because we bounds-check via cap.
        self.cap = (sz - hdr_bytes) // slot_bytes
        if self.cap <= 0:
            raise RuntimeError(
                f"{path}: segment size {sz} too small for header={hdr_bytes} + slot={slot_bytes}"
            )
        # Struct for the 8-byte pub-ts guard, LE.
        self._u64 = struct.Struct("<Q")

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fd.close()

    def _slot_off(self, k: int) -> int:
        return self.hdr + k * self.slot

    def _read_guarded_slot(self, k: int, retries: int = 4) -> Optional[bytes]:
        """Copy `self.slot` bytes from slot k with seqlock guard check. Return payload
        (guard-stripped) or None if all retries torn."""
        off = self._slot_off(k)
        for _ in range(retries):
            g0 = self._u64.unpack_from(self._mm, off)[0]
            payload = bytes(self._mm[off + self.guard : off + self.slot])
            g1 = self._u64.unpack_from(self._mm, off + 8)[0]
            if g0 == g1 and g0 != 0:
                return payload
        return None

    def latest(self, retries: int = 4) -> Optional[tuple[int, bytes]]:
        """Return (pub_ts_ns, payload_bytes) of the freshest slot.
        None if segment is empty or all newest slots torn beyond `retries`."""
        # Snapshot all slot guard0's, pick the max.
        best_t, best_k = 0, -1
        for k in range(self.cap):
            off = self._slot_off(k)
            t = self._u64.unpack_from(self._mm, off)[0]
            if t > best_t:
                best_t, best_k = t, k
        if best_k < 0:
            return None
        payload = self._read_guarded_slot(best_k, retries=retries)
        if payload is None:
            return None
        return best_t, payload


class ShmStateSource:
    """StateSource that reads the two md_bus proprio rings on the robot loopback."""

    def __init__(self,
                 fb_raw_path: str = FB_RAW_PATH,
                 aux_fb_path: str = AUX_FB_PATH):
        self._fb = _SeqRingReader(fb_raw_path, FB_RAW_HDR_BYTES,
                                  FB_RAW_SLOT_BYTES, FB_RAW_GUARD_BYTES)
        self._aux = _SeqRingReader(aux_fb_path, AUX_FB_HDR_BYTES,
                                   AUX_FB_SLOT_BYTES, AUX_FB_GUARD_BYTES)
        self._last: Optional[ProprioSnapshot] = None

    def close(self) -> None:
        self._fb.close()
        self._aux.close()

    def read(self) -> ProprioSnapshot:
        """Return the freshest proprio snapshot. Falls back to last-good on tear."""
        fb = self._fb.latest()
        aux = self._aux.latest()
        if fb is None:
            if self._last is not None:
                return self._last
            raise RuntimeError(f"{FB_RAW_PATH}: no readable slot (writer up?)")

        _pub_ts, fb_payload = fb
        # pos_f (30 channels, radians) — permute to VLA 27-body order.
        pos_f_mcu = np.frombuffer(fb_payload, dtype="<f4", count=30, offset=_PL_POS_F)
        joint_pos_vla = pos_f_mcu[joint_maps.MCU_BODY_TO_VLA].astype(np.float32, copy=True)
        # head_0/head_1 = MCU channels 27,28 (see joint_maps.MCU_ORDER). Raw feedback,
        # no sign flip — only used by RX2 29-dof deploys (joint_maps.RX2_VLA_ORDER).
        head_yaw_pitch = pos_f_mcu[27:29].astype(np.float32, copy=True)

        quat = np.frombuffer(fb_payload, dtype="<f4", count=4, offset=_PL_QUAT).copy()
        ts_ns = struct.unpack_from("<Q", fb_payload, _PL_TS_NS)[0]
        cycle = struct.unpack_from("<Q", fb_payload, _PL_CYCLE)[0]
        mcu_state = fb_payload[_PL_MCU_STATE]

        left_g, right_g = 0.0, 0.0
        if aux is not None:
            _, aux_payload = aux
            has_fb = aux_payload[_AUX_HAS_FB]
            valid = aux_payload[_AUX_GRIP_VALID]
            if has_fb and valid:
                grip = np.frombuffer(aux_payload, dtype="<f4", count=2, offset=_AUX_GRIP_POS)
                left_g, right_g = float(grip[0]), float(grip[1])

        snap = ProprioSnapshot(
            joint_pos_vla=joint_pos_vla,
            head_yaw_pitch=head_yaw_pitch,
            left_gripper_pos=left_g,
            right_gripper_pos=right_g,
            root_quat_wxyz=quat,
            timestamp_ns=ts_ns,
            cycle=cycle,
            mcu_state=int(mcu_state),
        )
        self._last = snap
        return snap


# ─────────────────────────────── Self-test ───────────────────────────────
def _selftest() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-s", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    src = ShmStateSource()
    prev_ts = 0
    try:
        for i in range(args.n):
            s = src.read()
            wall = time.time_ns()
            age_ms = (wall - s.timestamp_ns) / 1e6
            dt_ms = (s.timestamp_ns - prev_ts) / 1e6 if prev_ts else 0.0
            prev_ts = s.timestamp_ns
            print(
                f"[{i:2d}] cyc={s.cycle}  age={age_ms:+.1f}ms  Δ={dt_ms:+.2f}ms  "
                f"mcu={s.mcu_state}  "
                f"j[:5]={s.joint_pos_vla[:5].round(3).tolist()}  "
                f"head={s.head_yaw_pitch.round(3).tolist()}  "
                f"quat={s.root_quat_wxyz.round(4).tolist()}  "
                f"grip=({s.left_gripper_pos:.3f},{s.right_gripper_pos:.3f})"
            )
            time.sleep(args.dt_s)
    finally:
        src.close()


if __name__ == "__main__":
    _selftest()
