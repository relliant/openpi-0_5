"""Live telemetry dashboard for rx101_bridge — model-commanded vs robot-actual.

Purely additive/diagnostic: serves a single self-contained HTML page (plain
canvas, no CDN dependency) plus a Server-Sent-Events stream at /stream.
bridge.py pushes one JSON sample per publish tick via `TelemetryHub.publish()`;
nothing here can affect the control loop — publish() is non-blocking and drops
samples for slow/absent clients rather than backing up the publish thread.

Open http://<robot-ip>:<port>/ in a browser on the same network (no SSH
tunnel needed if you're already on the robot's LAN, e.g. the rx101b direct
connection). Pick a joint from the dropdown to compare:
  raw    - model's direct output for this tick (VLA order, before any filter)
  pub    - what was actually published to SONIC (after the low-pass filter)
  actual - the robot's own encoder feedback for that joint
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>rx101_bridge live telemetry</title>
<style>
  body { font-family: monospace; background: #111; color: #ddd; margin: 16px; }
  select, button { font-family: monospace; font-size: 14px; padding: 4px; }
  #status { margin-left: 12px; }
  #status.ok { color: #6f6; }
  #status.bad { color: #f66; }
  canvas { background: #000; border: 1px solid #444; }
  #chart { margin-top: 10px; }
  .legend span { margin-right: 18px; }
  .raw { color: #f66; }
  .pub { color: #6cf; }
  .act { color: #6f6; }
  .readout { margin-top: 14px; line-height: 1.6; }
  .readout b { color: #fff; }
  .grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
  .cell { width: 190px; }
  .cell .label { font-size: 11px; color: #aaa; margin-bottom: 2px; }
  .cell canvas { width: 190px; height: 90px; }
</style>
</head>
<body>
  <h3>rx101_bridge live telemetry: model-commanded vs robot-actual</h3>
  <label>Detail joint:
    <select id="joint"></select>
  </label>
  <span id="status" class="bad">connecting...</span>
  <div class="legend">
    <span class="raw">&#9632; raw (model output)</span>
    <span class="pub">&#9632; pub (published, filtered)</span>
    <span class="act">&#9632; actual (robot feedback)</span>
  </div>
  <canvas id="chart" width="900" height="320"></canvas>
  <div class="readout" id="readout"></div>

  <h4>all joints</h4>
  <div class="grid" id="grid"></div>

<script>
const VLA_ORDER = [
  "l_hip_pitch","l_hip_roll","l_hip_yaw","l_knee","l_ankle_pitch","l_ankle_roll",
  "r_hip_pitch","r_hip_roll","r_hip_yaw","r_knee","r_ankle_pitch","r_ankle_roll",
  "waist_yaw","waist_roll","waist_pitch",
  "l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow","l_wrist_roll","l_wrist_pitch",
  "r_shoulder_pitch","r_shoulder_roll","r_shoulder_yaw","r_elbow","r_wrist_roll","r_wrist_pitch",
];
const HEAD_NAMES = ["head_yaw", "head_pitch"];

const ALL_NAMES = VLA_ORDER.concat(HEAD_NAMES);   // 27 body + 2 head = 29
const N = ALL_NAMES.length;

const sel = document.getElementById("joint");
ALL_NAMES.forEach((name, i) => {
  const opt = document.createElement("option");
  opt.value = i; opt.textContent = name;
  sel.appendChild(opt);
});
sel.value = 17; // l_elbow — default focus of this session's debugging

const MAXPTS = 300;
const bufs = Array.from({ length: N }, () => ({ raw: [], pub: [], act: [] }));
let latest = null;

function pushSample(s) {
  latest = s;
  for (let i = 0; i < N; i++) {
    let r, p, a;
    if (i < 27) {
      r = s.raw[i]; p = s.pub[i]; a = s.act[i];
    } else {
      const j = i - 27;
      r = s.head_raw ? s.head_raw[j] : null;
      p = s.head_pub ? s.head_pub[j] : null;
      a = s.head_act ? s.head_act[j] : null;
    }
    const b = bufs[i];
    b.raw.push(r); b.pub.push(p); b.act.push(a);
    if (b.raw.length > MAXPTS) { b.raw.shift(); b.pub.shift(); b.act.shift(); }
  }
}

function drawSeries(ctx, w, h, buf, showLabels) {
  ctx.clearRect(0, 0, w, h);
  const all = buf.raw.concat(buf.pub, buf.act).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  if (all.length < 2) return;
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 || 0.1;
  lo -= pad; hi += pad;
  const n = buf.raw.length;
  const xstep = w / MAXPTS;
  const y = (v) => h - ((v - lo) / (hi - lo)) * h;
  function line(arr, color) {
    ctx.strokeStyle = color; ctx.lineWidth = showLabels ? 2 : 1.3; ctx.beginPath();
    let started = false;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v === null || v === undefined || Number.isNaN(v)) { started = false; continue; }
      const x = w - (n - i) * xstep;
      if (!started) { ctx.moveTo(x, y(v)); started = true; } else { ctx.lineTo(x, y(v)); }
    }
    ctx.stroke();
  }
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
  line(buf.raw, "#f66");
  line(buf.pub, "#6cf");
  line(buf.act, "#6f6");
  if (showLabels) {
    ctx.fillStyle = "#888"; ctx.font = "11px monospace";
    ctx.fillText(hi.toFixed(3), 4, 12);
    ctx.fillText(lo.toFixed(3), 4, h - 4);
  }
}

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");

// Build one small canvas per joint for the "all joints" grid.
const gridEl = document.getElementById("grid");
const gridCtx = [];
ALL_NAMES.forEach((name, i) => {
  const cell = document.createElement("div");
  cell.className = "cell";
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = name;
  const c = document.createElement("canvas");
  c.width = 190; c.height = 90;
  cell.appendChild(label);
  cell.appendChild(c);
  gridEl.appendChild(cell);
  gridCtx.push(c.getContext("2d"));
});

function draw() {
  drawSeries(ctx, canvas.width, canvas.height, bufs[parseInt(sel.value)], true);
  for (let i = 0; i < N; i++) {
    drawSeries(gridCtx[i], 190, 90, bufs[i], false);
  }
  requestAnimationFrame(draw);
}
draw();

function quatYawRollDeg(q) {
  // q = [w,x,y,z]
  const [w, x, y, z] = q;
  const yaw = Math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) * 180 / Math.PI;
  const roll = Math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * 180 / Math.PI;
  return [yaw, roll];
}

function updateReadout() {
  if (!latest) { requestAnimationFrame(updateReadout); return; }
  const s = latest;
  let html = "";
  html += `<div>seq=<b>${s.seq}</b> t=<b>${s.t.toFixed(3)}</b>s</div>`;
  html += `<div>gripper raw L/R=<b>${s.grip_raw[0].toFixed(3)}/${s.grip_raw[1].toFixed(3)}</b> `
        + `closed_mask=<b>0x${s.grip_closed_mask.toString(16).padStart(2,"0")}</b> `
        + `actual L/R=<b>${s.grip_act[0].toFixed(3)}/${s.grip_act[1].toFixed(3)}</b></div>`;
  if (s.quat_act) {
    const [yaw, roll] = quatYawRollDeg(s.quat_act);
    html += `<div>body actual yaw=<b>${yaw.toFixed(1)}&deg;</b> roll=<b>${roll.toFixed(1)}&deg;</b></div>`;
  }
  document.getElementById("readout").innerHTML = html;
  requestAnimationFrame(updateReadout);
}
updateReadout();

function connect() {
  const es = new EventSource("/stream");
  const status = document.getElementById("status");
  es.onopen = () => { status.textContent = "connected"; status.className = "ok"; };
  es.onerror = () => { status.textContent = "disconnected, retrying..."; status.className = "bad"; };
  es.onmessage = (ev) => { pushSample(JSON.parse(ev.data)); };
}
connect();
</script>
</body>
</html>
"""


class TelemetryHub:
    """Fan-out of the latest samples to any number of /stream clients.

    publish() is called from the bridge's publish_loop (timing-critical) —
    it must never block. Slow/stalled clients get old samples dropped rather
    than backing up the queue.
    """

    def __init__(self, per_client_buffer: int = 64) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._buf = per_client_buffer

    def subscribe(self) -> "queue.Queue[str]":
        q: "queue.Queue[str]" = queue.Queue(maxsize=self._buf)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, sample: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        if not subs:
            return
        line = json.dumps(sample)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(line)
                except queue.Empty:
                    pass


def make_server(hub: TelemetryHub, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:  # noqa: A002
            pass  # silence per-request stderr spam

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                body = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        line = q.get(timeout=5.0)
                        self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (queue.Empty, BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unsubscribe(q)
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    return server
