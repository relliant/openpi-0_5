"""Live ego camera via md_webserver's MJPEG endpoint (bridge side, on-robot).

The wide-angle chest camera (`cam8750`) is exposed at
`http://127.0.0.1:9090/liveview/cam8750.mjpg` as a multipart/x-mixed-replace
MJPEG stream (~30 fps). This module runs a background thread that keeps the
socket open, greedily consumes frames, and always exposes the *latest* one via
`read_ego_jpeg()` — the VLA bridge calls at 10 Hz and doesn't want to drain a
buffered queue.

Design:
  * cv2.VideoCapture handles the multipart parsing + JPEG decode into a numpy
    frame. Great for correctness, but cv2 returns decoded RGB/BGR at native
    (4096x3072) which is ~50 MB/frame. We do NOT want to hand that to the VLA
    server every call (the server resizes to 224x224 internally, but that's
    huge JPEG re-encode overhead per obs). Instead we downsize to 480x640
    (dataset's ego_view shape) INSIDE the reader thread, encode as JPEG once,
    hand the bytes to the caller. Caller sends JPEG bytes over websocket, the
    server decodes.
  * If the stream stalls or reconnects, `read_ego_jpeg` blocks briefly (<1s)
    before returning the last-good frame. StaleFrameError only fires if we
    never got a first frame or the stream has been down >5s.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

LOG = logging.getLogger("rx101_bridge.camera")

CAM8750_MJPEG_URL = "http://127.0.0.1:9090/liveview/cam8750.mjpg"

# Dataset stores ego_view at 480x640 (see openpi training config Rx101 comment).
# We resize on the robot side to save network + VLA-side resize cost.
DATASET_EGO_SIZE = (480, 640)   # (H, W)

_STALE_ABORT_S = 5.0             # give up if no frame in this window


class StaleFrameError(RuntimeError):
    pass


class MjpegEgoCamera:
    """Background-thread MJPEG puller. Always exposes latest JPEG bytes."""

    def __init__(self,
                 url: str = CAM8750_MJPEG_URL,
                 downsize_hw: tuple[int, int] = DATASET_EGO_SIZE,
                 jpeg_quality: int = 85):
        self._url = url
        self._h, self._w = downsize_hw
        self._q = jpeg_quality
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_wall_ns: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="EgoCamPuller", daemon=True
        )

    def start(self) -> "MjpegEgoCamera":
        self._thread.start()
        # Wait briefly for first frame so read_ego_jpeg() doesn't crash.
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            with self._lock:
                if self._latest_jpeg is not None:
                    return self
            time.sleep(0.01)
        raise StaleFrameError(
            f"MJPEG source {self._url} produced no frame in 3s (webserver up?)"
        )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def read_ego_jpeg(self) -> bytes:
        """Return the most recent JPEG bytes (already downsized to dataset size)."""
        with self._lock:
            jpeg = self._latest_jpeg
            wall = self._latest_wall_ns
        age = (time.time_ns() - wall) / 1e9 if wall else float("inf")
        if jpeg is None or age > _STALE_ABORT_S:
            raise StaleFrameError(
                f"no fresh JPEG (age={age:.1f}s); MJPEG source stalled?"
            )
        return jpeg

    # ─── worker ────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._consume_stream()
            except Exception as e:
                LOG.warning("MJPEG stream error: %s. Retrying in 0.5s.", e)
                time.sleep(0.5)

    def _consume_stream(self) -> None:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        # Set a small buffer to always get the newest frame after read().
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open {self._url}")
        try:
            while not self._stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError("cap.read() returned no frame (stream ended?)")
                jpeg_bytes = self._encode_downsized_jpeg(frame_bgr)
                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    self._latest_wall_ns = time.time_ns()
        finally:
            cap.release()

    def _encode_downsized_jpeg(self, frame_bgr: np.ndarray) -> bytes:
        # OpenCV BGR -> resize keeping aspect via center-crop? For now match dataset
        # exact shape via cv2.resize (dataset was recorded through some fixed
        # pipeline; the model's resize_with_pad on the server side handles remaining
        # aspect concerns).
        h, w = self._h, self._w
        if frame_bgr.shape[:2] != (h, w):
            resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            resized = frame_bgr
        ok, enc = cv2.imencode(".jpg", resized,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._q])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return enc.tobytes()


# ─────────────────────────────── Self-test ───────────────────────────────
def _selftest() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CAM8750_MJPEG_URL)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dt-s", type=float, default=0.1)
    ap.add_argument("--save", default="/tmp/ego_bridge.jpg")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(threadName)s] %(message)s")

    cam = MjpegEgoCamera(url=args.url).start()
    try:
        prev_len = 0
        for i in range(args.n):
            jpg = cam.read_ego_jpeg()
            print(f"[{i:2d}] jpeg={len(jpg)}B  Δlen={len(jpg)-prev_len:+d}")
            prev_len = len(jpg)
            time.sleep(args.dt_s)
        # Save last frame
        with open(args.save, "wb") as f:
            f.write(jpg)
        # Verify decode
        im = Image.open(args.save)
        print(f"saved: {args.save}  decoded shape: {im.size}  mode: {im.mode}")
    finally:
        cam.close()


if __name__ == "__main__":
    _selftest()
