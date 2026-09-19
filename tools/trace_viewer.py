#!/usr/bin/env python3
"""Live viewer for GOAI policy-server traces (``trace.jsonl`` + camera JPEGs).

Read-only: this script never touches the serving path, has no third-party
dependencies (stdlib only, so ``pixi.lock`` stays untouched), and only ever
reads files that ``pi.inference.goai_trace.TraceWriter`` wrote.

    python tools/trace_viewer.py                      # newest run, port 8099
    python tools/trace_viewer.py --run 0918_exp6...   # pin one run
    python tools/trace_viewer.py --port 9000 --trace-dir /path/to/goai-trace

Then open http://<host>:8099/ (forward the port if the box is remote).

What it shows, per inference call:
  * the three camera frames the model actually received,
  * ``state[14]`` — left arm 0-5, left gripper 6, right arm 7-12, right gripper 13,
  * the returned ``action`` chunk (16 steps x 14 dims), plotted against the state
    that produced it, so the commanded motion reads as a deviation from "now".

The chart is deliberately 14 small multiples rather than 14 coloured lines: past
~8 classes a shared hue scale stops being readable, so each panel carries one
series (the chunk, in the accent hue) plus the current state as a gray reference.
"""

from __future__ import annotations

import re
import json
import time
import argparse
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# 默认跟踪目录:跟随当前用户,不把某台机器的绝对路径写进公开仓库。
DEFAULT_TRACE_DIR = Path.home() / "goai-trace"
DEFAULT_PORT = 8099

# Left arm 0-5, left gripper 6, right arm 7-12, right gripper 13.
_DIM_LABELS = [f"L J{i}" for i in range(1, 7)] + ["L 爪"] + [f"R J{i}" for i in range(1, 7)] + ["R 爪"]
_GRIPPER_DIMS = {6, 13}  # normalized openings: axis is pinned to [0, 1]

_IMAGE_NAME = re.compile(r"^\d{7}_[A-Za-z0-9_]+\.jpg$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


# --------------------------------------------------------------------------- data


def list_runs(trace_dir: Path) -> list[dict]:
    """Every subdirectory holding a trace.jsonl, newest first."""
    runs = []
    for entry in trace_dir.iterdir() if trace_dir.is_dir() else []:
        trace = entry / "trace.jsonl"
        if not entry.is_dir() or not trace.is_file():
            continue
        meta = {}
        meta_path = entry / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        runs.append(
            {
                "run": entry.name,
                "mtime": trace.stat().st_mtime,
                "size": trace.stat().st_size,
                "started_at": meta.get("started_at"),
                "policy_name": meta.get("policy_name"),
                "checkpoint_path": meta.get("checkpoint_path"),
                "execution_horizon": meta.get("execution_horizon"),
                "num_steps": meta.get("num_steps"),
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def read_from(trace: Path, offset: int) -> tuple[list[dict], int]:
    """Append-only read: return complete JSON lines past ``offset``.

    A trailing partial line is left for the next poll, so a line being written
    while we read is never parsed half-formed.
    """
    size = trace.stat().st_size
    if offset > size:  # truncated or replaced underneath us
        offset = 0
    with trace.open("rb") as handle:
        handle.seek(offset)
        blob = handle.read()
    cut = blob.rfind(b"\n")
    if cut == -1:
        return [], offset
    events = []
    for raw in blob[: cut + 1].decode("utf-8", "replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except ValueError:
            continue  # a corrupt line must not kill the live view
    return events, offset + cut + 1


# ------------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    trace_dir = Path(DEFAULT_TRACE_DIR)
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _resolve_run(self, name: str | None) -> Path | None:
        runs = [r["run"] for r in list_runs(self.trace_dir)]
        if name:
            if name not in runs or not _RUN_NAME.match(name):
                return None
            return self.trace_dir / name
        return self.trace_dir / runs[0] if runs else None

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/runs":
            body = json.dumps({"runs": list_runs(self.trace_dir)}, ensure_ascii=False)
            self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/trace":
            run = self._resolve_run(query.get("run", [None])[0])
            if run is None:
                self._send(404, b'{"error":"no such run"}', "application/json")
                return
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                offset = 0
            trace = run / "trace.jsonl"
            if not trace.is_file():
                self._send(404, b'{"error":"no trace.jsonl"}', "application/json")
                return
            with self.lock:
                events, new_offset = read_from(trace, offset)
            body = json.dumps(
                {"run": run.name, "events": events, "offset": new_offset, "eof": True},
                ensure_ascii=False,
            )
            self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/image":
            run = self._resolve_run(query.get("run", [None])[0])
            name = query.get("name", [""])[0]
            if run is None or not _IMAGE_NAME.match(name):
                self._send(400, b'{"error":"bad image request"}', "application/json")
                return
            path = run / "images" / name
            if not path.is_file():
                self._send(404, b'{"error":"no such image"}', "application/json")
                return
            self._send(200, path.read_bytes(), "image/jpeg", cache="public, max-age=31536000")
            return

        self._send(404, b'{"error":"not found"}', "application/json")


# --------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GOAI 推理轨迹</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --axis: #c3c2b7;
    --series-1: #2a78d6;   /* accent: the action chunk */
    --series-dim: #86b6ef; /* same ramp, lighter step: previous chunk */
    --series-state: #e34948; /* categorical red: the state (model input) */
    --border: rgba(11, 11, 11, 0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --axis: #383835;
      --series-1: #3987e5;
      --series-dim: #1c5cab;
      --series-state: #e66767;
      --border: rgba(255, 255, 255, 0.10);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --axis: #383835;
    --series-1: #3987e5;
    --series-dim: #1c5cab;
    --series-state: #e66767;
    --border: rgba(255, 255, 255, 0.10);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--text-primary);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 16px 20px 48px; }

  /* One filter row, above everything it scopes. */
  .controls {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    padding: 10px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
  }
  .controls label { color: var(--text-secondary); font-size: 13px; }
  select, button {
    font: inherit; font-size: 13px; color: var(--text-primary);
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 8px;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--axis); }
  .spacer { flex: 1; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); display: inline-block; }
  .dot.live { background: #0ca30c; }
  .status { color: var(--text-muted); font-size: 12px; }

  /* Episode scrubber - one control row, above everything it scopes. */
  .scrub {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 10px 12px; margin-bottom: 16px;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  }
  .scrub button { min-width: 34px; padding: 4px 8px; }
  .scrub .track { flex: 1; min-width: 200px; display: flex; flex-direction: column; gap: 1px; }
  .scrub input[type="range"] { width: 100%; min-width: 0; accent-color: var(--series-1); }
  /* Episode boundaries: reset events, drawn under the slider track. */
  .scrub .marks { position: relative; height: 9px; margin: 0 8px; }
  .scrub .marks i {
    position: absolute; top: 0; width: 1px; height: 7px;
    background: var(--axis); transform: translateX(-0.5px);
  }
  .scrub .marks i:hover { background: var(--series-1); }
  .scrub input[type="number"] {
    width: 76px; font: inherit; font-size: 13px; padding: 4px 6px;
    color: var(--text-primary); background: var(--page);
    border: 1px solid var(--border); border-radius: 6px;
  }
  .scrub .pos {
    color: var(--text-secondary); font-size: 12px; min-width: 86px; text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .legend { color: var(--text-secondary); font-size: 12px; margin: -2px 0 10px; }
  /* The task instruction the model was given - shown whole, never truncated. */
  .instr {
    display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;
    padding: 10px 12px; margin-bottom: 12px;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  }
  .instr .tag {
    flex: none; font-size: 12px; color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 999px; padding: 1px 9px;
    font-variant-numeric: tabular-nums;
  }
  .instr #instrText { font-size: 14px; color: var(--text-primary); }
  .legend .key { display: inline-block; width: 16px; height: 2px; margin: 0 5px 3px 12px; vertical-align: middle; }
  .legend .key:first-child { margin-left: 0; }
  .key.line-accent { background: var(--series-1); }
  .key.line-dim    { background: var(--series-dim); }
  /* The state key is drawn dashed, matching the reference rule inside the panels. */
  .key.line-state  { height: 0; border-top: 2px dashed var(--series-state); }
  .key.dot-state   { width: 8px; height: 8px; border-radius: 50%; background: var(--series-state); margin-bottom: 0; }
  .ptitle { display: flex; justify-content: space-between; align-items: center; gap: 6px; padding-bottom: 2px; }
  .ptitle .pn { font-size: 11px; color: var(--text-secondary); }
  .ptitle .pv {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; color: var(--text-primary); font-variant-numeric: tabular-nums;
  }
  /* A dot swatch, so the number's identity is carried by a mark - not by red text. */
  .ptitle .pv::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: var(--series-state); flex: none;
  }

  h2 { font-size: 13px; font-weight: 600; color: var(--text-secondary); margin: 24px 0 10px; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
  .tile {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px;
  }
  .tile .k { font-size: 12px; color: var(--text-secondary); margin-bottom: 3px; }
  .tile .v { font-size: 20px; font-weight: 600; }
  .tile .v small { font-size: 12px; font-weight: 400; color: var(--text-muted); }

  /* Two columns: the model's input on the left, its output on the right. */
  .main { display: grid; grid-template-columns: minmax(210px, 320px) 1fr; gap: 18px; align-items: stretch; }
  @media (max-width: 1080px) { .main { grid-template-columns: 1fr; } }
  .col-left, .col-right { display: flex; flex-direction: column; min-width: 0; }
  .col-left h2, .col-right h2 { margin-top: 0; }

  .shots { display: flex; flex-direction: column; gap: 10px; }
  .shot { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .shot img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: var(--page); }
  .shot .cap { font-size: 12px; color: var(--text-secondary); padding: 6px 9px; }

  /* Latency: fills whatever height the camera column leaves. */
  .lat {
    flex: 1; min-height: 220px; display: flex; flex-direction: column; gap: 8px;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
  }
  .lat .latctl { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .lat .latctl label { font-size: 12px; color: var(--text-secondary); }
  .lat .now { font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .lat .now b { color: var(--text-primary); font-weight: 600; }
  .latchart { flex: 1; min-height: 0; }
  .latchart svg { display: block; width: 100%; height: 100%; }
  .lat .hint { font-size: 12px; color: var(--text-muted); }

  .chart { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; position: relative; }
  .grouphead {
    font-size: 12px; color: var(--text-muted); padding: 2px 0 4px;
    grid-column: span 6;
  }
  .grouphead.one { grid-column: span 1; }
  .grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px 8px; }
  .panel { min-width: 0; }
  .panel svg { display: block; width: 100%; height: auto; }
  .tick { font-size: 10px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }

  .tip {
    position: absolute; pointer-events: none; z-index: 5; display: none;
    background: var(--surface-1); border: 1px solid var(--axis); border-radius: 8px;
    padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.14);
    min-width: 168px;
  }
  .tip .th { color: var(--text-secondary); margin-bottom: 5px; font-weight: 600; }
  .tip table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .tip td { padding: 1px 0; }
  .tip td.k { color: var(--text-secondary); padding-right: 10px; }
  .tip tr.grip td.v { color: var(--text-primary); }
  .tip tr.grip td.k { padding-left: 8px; border-left: 2px solid var(--axis); }

  table.data { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; width: 100%; }
  table.data th, table.data td { border-bottom: 1px solid var(--gridline); padding: 3px 6px; text-align: right; white-space: nowrap; }
  table.data th { color: var(--text-secondary); font-weight: 600; text-align: right; }
  table.data th:first-child, table.data td:first-child { text-align: left; color: var(--text-secondary); }
  table.data tr.state td { color: var(--text-primary); font-weight: 600; border-bottom: 1px solid var(--axis); }
  .scroller { overflow-x: auto; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .empty { color: var(--text-muted); padding: 24px 0; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<div class="wrap">

  <div class="controls">
    <label>运行 <select id="run"></select></label>
    <label><input type="checkbox" id="follow" checked> 跟随最新</label>
    <label><input type="checkbox" id="prev"> 显示上一 chunk</label>
    <label><input type="checkbox" id="table"> 表格视图</label>
    <button id="theme" type="button">切换深浅色</button>
    <span class="spacer"></span>
    <span class="status"><span class="dot" id="dot"></span> <span id="status">连接中…</span></span>
  </div>

  <!-- 进度控制：拖动或 ←/→ 浏览整条 episode -->
  <div class="scrub">
    <button id="toFirst" type="button" title="第一帧 (Home)">⏮</button>
    <button id="back1" type="button" title="上一帧 (←)">◀</button>
    <div class="track">
      <input id="slider" type="range" min="1" max="1" value="1" step="1" aria-label="episode 进度">
      <div class="marks" id="marks" title="竖线 = reset，即 episode 分界"></div>
    </div>
    <button id="fwd1" type="button" title="下一帧 (→)">▶</button>
    <button id="toLast" type="button" title="最新 (End)">⏭</button>
    <button id="play" type="button" title="自动播放 (空格)">⏵ 播放</button>
    <span class="pos" id="pos">0 / 0</span>
    <input id="jump" type="number" min="1" placeholder="# 跳转">
  </div>

  <!-- 模型输入的任务指令（完整，不截断） -->
  <div class="instr">
    <span class="tag" id="taskTag">task —</span>
    <span id="instrText">—</span>
  </div>

  <div class="tiles" id="tiles"></div>

  <div class="main">
    <div class="col-left">
      <h2>模型输入 · 三路相机</h2>
      <div class="shots" id="shots"></div>
    </div>

    <div class="col-right">
      <h2>State-action</h2>
      <p class="legend">
        <span class="key line-accent"></span>action chunk（16 步 · 模型输出）
        <span class="key dot-state"></span>state @ 第 0 步（模型输入）
        <span class="key line-state"></span>state 参考线
        <span class="key line-dim"></span>上一 chunk（勾选后）
        &nbsp;·&nbsp; ←/→ 单帧，Shift+←/→ 十帧，空格播放，End 回最新
      </p>
      <div class="chart">
        <div class="grid" id="grid"></div>
        <div class="tip" id="tip"></div>
      </div>

      <h2>延迟</h2>
      <div class="lat">
        <div class="latctl">
          <label>指标 <select id="latMetric"></select></label>
          <span class="now" id="latNow">—</span>
        </div>
        <div class="latchart" id="latchart"></div>
        <div class="hint" id="latHint"></div>
      </div>
    </div>
  </div>

  <div id="tableView" hidden>
    <h2>表格视图</h2>
    <div class="scroller"><table class="data" id="dataTable"></table></div>
  </div>
</div>

<script>
const DIM_LABELS = ["L J1","L J2","L J3","L J4","L J5","L J6","L 爪",
                    "R J1","R J2","R J3","R J4","R J5","R J6","R 爪"];
const GRIPPER_DIMS = new Set([6, 13]);
// Slot order from the policy's INTERFACE.md; the trace itself carries only task_index.
const TASK_SLUGS = ["fill_pen_holder", "put_objects_into_basket", "stack_and_cover_blocks",
                    "stack_bowls", "stand_up_bottles", "insert_charger"];
// 7 columns per row: arm 1-6, then the gripper.
const GROUPS = [
  {label: "左臂 left arm", from: 0, to: 5},
  {label: "左爪", from: 6, to: 6},
  {label: "右臂 right arm", from: 7, to: 12},
  {label: "右爪", from: 13, to: 13},
];

const NS = "http://www.w3.org/2000/svg";
const PAD = {t: 8, r: 10, b: 16, l: 30};
const PW = 150, PH = 100;                 // panel svg size
const PLOT_W = PW - PAD.l - PAD.r, PLOT_H = PH - PAD.t - PAD.b;

const el = (id) => document.getElementById(id);
const state = {
  run: null, offset: 0, calls: [], lastReset: null,
  follow: true, pinned: 0, showPrev: false, hoverStep: null,
  playing: false, timer: null,
  boundaries: [],   // call counts where a reset was seen = episode starts
};

/** The call on screen: the newest while following, else the pinned one. */
function currentCall() {
  if (!state.calls.length) return null;
  if (state.follow) return state.calls[state.calls.length - 1];
  return state.calls[Math.min(state.pinned, state.calls.length - 1)];
}

/** Call indices where an episode begins. 0 is always one; resets add the rest. */
function episodeStarts() {
  const n = state.calls.length;
  const set = new Set([0]);
  for (const b of state.boundaries) if (b > 0 && b < n) set.add(b);
  return [...set].sort((a, b) => a - b);
}

/** 1-based episode number containing the call at index ``i``. */
function episodeOf(i) {
  let k = 0;
  for (const b of episodeStarts()) if (b <= i) k++;
  return k || 1;
}

/* ---------- scale helpers ---------- */

function niceTicks(lo, hi, want) {
  if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
  const raw = (hi - lo) / Math.max(1, want);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  // Pick the 1/2/5 step whose tick count lands closest to what was asked for.
  // Taking the first step that is merely large enough often leaves a single
  // gridline (a 680 ms span at 3 ticks jumped straight to a 500 ms step).
  let best = 1, bestErr = Infinity;
  for (const m of [1, 2, 5, 10]) {
    const count = Math.floor((hi - lo) / (m * mag)) + 1;
    const err = Math.abs(count - want);
    if (err < bestErr) { bestErr = err; best = m; }
  }
  const step = best * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

/** 1-2-5 ticks over decades, thinned until they fit a short panel. */
function logTicks(lo, hi) {
  const build = (mantissas) => {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of mantissas) {
        const v = m * Math.pow(10, e);
        if (v >= lo && v <= hi) out.push(v);
      }
    }
    return out;
  };
  let out = build([1, 2, 5]);
  if (out.length > 7) out = build([1, 5]);
  if (out.length > 7) out = build([1]);
  return out;
}

/** Panel y-domain: grippers are pinned to [0,1]; arms autoscale over the window. */
function domainFor(dim) {
  if (GRIPPER_DIMS.has(dim)) return [0, 1];
  const pool = [];
  for (const c of state.calls) {
    pool.push(c.state[dim]);
    for (const row of c.action) pool.push(row[dim]);
    if (c.prev) for (const row of c.prev) pool.push(row[dim]);
  }
  if (!pool.length) return [-1, 1];
  let lo = Math.min(...pool), hi = Math.max(...pool);
  const pad = (hi - lo) * 0.12 || 0.05;
  return [lo - pad, hi + pad];
}

/* ---------- rendering ---------- */

function svgNode(name, attrs) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function drawPanel(dim, host, domain) {
  const c = currentCall();
  const [lo, hi] = domain;
  const n = c.action.length;

  const x = (i) => PAD.l + (n <= 1 ? PLOT_W / 2 : (i / (n - 1)) * PLOT_W);
  const y = (v) => PAD.t + PLOT_H - ((v - lo) / (hi - lo || 1)) * PLOT_H;

  host.textContent = "";
  // Title carries the dim name and the current state - the model's actual input,
  // which the thin gray rule alone reads as too faint.
  const title = document.createElement("div");
  title.className = "ptitle";
  const pn = document.createElement("span");
  pn.className = "pn";
  pn.textContent = DIM_LABELS[dim];
  const pv = document.createElement("span");
  pv.className = "pv";
  pv.textContent = c.state[dim].toFixed(3);
  title.append(pn, pv);
  host.appendChild(title);

  const svg = svgNode("svg", {viewBox: `0 0 ${PW} ${PH}`, role: "img"});
  svg.setAttribute("aria-label", DIM_LABELS[dim] + " chunk");

  // gridlines + y ticks (solid hairlines, recessive)
  for (const t of niceTicks(lo, hi, 2)) {
    const yy = y(t);
    if (yy < PAD.t - 1 || yy > PAD.t + PLOT_H + 1) continue;
    svg.appendChild(svgNode("line", {
      x1: PAD.l, x2: PAD.l + PLOT_W, y1: yy, y2: yy,
      stroke: "var(--gridline)", "stroke-width": 1,
    }));
    const lab = svgNode("text", {x: PAD.l - 4, y: yy + 3, "text-anchor": "end", class: "tick"});
    lab.textContent = t.toFixed(Math.abs(t) < 10 ? 2 : 1);
    svg.appendChild(lab);
  }
  // baseline
  svg.appendChild(svgNode("line", {
    x1: PAD.l, x2: PAD.l + PLOT_W, y1: PAD.t + PLOT_H, y2: PAD.t + PLOT_H,
    stroke: "var(--axis)", "stroke-width": 1,
  }));
  // x ticks: first / mid / last step
  for (const i of [0, Math.floor((n - 1) / 2), n - 1]) {
    const lab = svgNode("text", {x: x(i), y: PH - 4, "text-anchor": "middle", class: "tick"});
    lab.textContent = i;
    svg.appendChild(lab);
  }

  // The state is the model's INPUT, so it is drawn as an input marker: a dashed
  // rule for the level it sits at, and a filled dot at step 0 — the point the
  // chunk starts from. Drawn under the accent line, its dot redrawn on top.
  const sy = y(c.state[dim]);
  svg.appendChild(svgNode("line", {
    x1: PAD.l, x2: PAD.l + PLOT_W, y1: sy, y2: sy,
    stroke: "var(--series-state)", "stroke-width": 1.5, "stroke-dasharray": "3 3",
  }));

  // previous chunk (same ramp, lighter step) — context, drawn under the accent
  if (state.showPrev && c.prev && c.prev.length) {
    const p = c.prev.map((row, i) => `${i ? "L" : "M"}${x(i)},${y(row[dim])}`).join(" ");
    svg.appendChild(svgNode("path", {
      d: p, fill: "none", stroke: "var(--series-dim)", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }

  // the chunk: one series, so no legend box — the panel title names it
  const d = c.action.map((row, i) => `${i ? "L" : "M"}${x(i)},${y(row[dim])}`).join(" ");
  svg.appendChild(svgNode("path", {
    d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }));

  // end marker: >=8px, with a 2px surface ring so it stays legible over the rule
  const last = c.action[n - 1][dim];
  svg.appendChild(svgNode("circle", {
    cx: x(n - 1), cy: y(last), r: 4,
    fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2,
  }));

  // the state dot last, so the input marker is never buried under the output line
  svg.appendChild(svgNode("circle", {
    cx: x(0), cy: sy, r: 4,
    fill: "var(--series-state)", stroke: "var(--surface-1)", "stroke-width": 2,
  }));

  // crosshair for the hovered step
  if (state.hoverStep !== null && state.hoverStep < n) {
    svg.appendChild(svgNode("line", {
      x1: x(state.hoverStep), x2: x(state.hoverStep), y1: PAD.t, y2: PAD.t + PLOT_H,
      stroke: "var(--text-muted)", "stroke-width": 1,
    }));
  }

  // hit layer: the whole plot area, so hover never needs pixel precision
  const hit = svgNode("rect", {
    x: PAD.l, y: PAD.t, width: PLOT_W, height: PLOT_H, fill: "transparent",
  });
  hit.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * PW;
    const frac = (px - PAD.l) / PLOT_W;
    const step = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
    state.hoverStep = step;
    render();
    showTip(ev, step);
  });
  hit.addEventListener("mouseleave", () => {
    state.hoverStep = null;
    hideTip();
    render();
  });
  svg.appendChild(hit);
  host.appendChild(svg);
}

function showTip(ev, step) {
  const c = currentCall();
  const tip = el("tip");
  const rows = DIM_LABELS.map((lab, d) => {
    const cls = GRIPPER_DIMS.has(d) ? ' class="grip"' : "";
    const cur = c.state[d].toFixed(3);
    const act = c.action[step][d].toFixed(3);
    return `<tr${cls}><td class="k">${lab}</td><td class="v">${act}</td><td class="k" style="padding-left:10px">${cur}</td></tr>`;
  }).join("");
  tip.innerHTML =
    `<div class="th">step ${step} · 目标 / 当前</div>` +
    `<table>${rows}</table>`;
  tip.style.display = "block";
  const host = el("grid").parentElement.getBoundingClientRect();
  let left = ev.clientX - host.left + 14;
  let top = ev.clientY - host.top + 12;
  if (left + tip.offsetWidth > host.width - 8) left = ev.clientX - host.left - tip.offsetWidth - 14;
  if (top + tip.offsetHeight > host.height - 8) top = host.height - tip.offsetHeight - 8;
  tip.style.left = Math.max(4, left) + "px";
  tip.style.top = Math.max(4, top) + "px";
}

function hideTip() { el("tip").style.display = "none"; }

/* ---------- latency ---------- */

/** Only newer traces carry a ``timing`` block; ``interval`` always works. */
function hasTiming() { return state.calls.some((c) => c.timing); }

function latValue(i, key) {
  const calls = state.calls;
  const c = calls[i];
  if (!c) return null;
  if (key === "interval") {
    // A gap spanning a reset is idle time between episodes, not a loop interval:
    // it runs to tens of seconds and would crush the ~1.2 s band onto the axis.
    if (i <= 0 || episodeStarts().indexOf(i) >= 0) return null;
    return (c.t - calls[i - 1].t) * 1000;
  }
  const t = c.timing;
  return t && typeof t[key] === "number" ? t[key] : null;
}

function latLabel(key) {
  return {
    interval: "调用间隔（含动作播放）",
    wall_ms: "推理耗时 · 总计 (wall)",
    infer_ms: "推理耗时 · 内部计时",
    denoise_ms: "推理耗时 · 去噪",
    prepare_ms: "推理耗时 · 观测准备",
    decode_ms: "推理耗时 · 解码输出",
  }[key] || key;
}

function buildLatMetrics() {
  const sel = el("latMetric");
  const want = sel.value || "interval";
  const timing = hasTiming();
  const opts = ['<option value="interval">' + latLabel("interval") + "</option>"];
  if (timing) {
    opts.push('<optgroup label="推理耗时 · 服务端计时">');
    for (const k of ["wall_ms", "infer_ms", "denoise_ms", "prepare_ms", "decode_ms"]) {
      opts.push(`<option value="${k}">${latLabel(k)}</option>`);
    }
    opts.push("</optgroup>");
  }
  const html = opts.join("");
  if (sel.dataset.built !== html) { sel.innerHTML = html; sel.dataset.built = html; }
  const valid = [...sel.options].some((o) => o.value === want);
  sel.value = valid ? want : "interval";
  sel.disabled = !state.calls.length;
}

function latStat(values) {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  return {median: mid, max: s[s.length - 1]};
}

function renderLat() {
  buildLatMetrics();
  const host = el("latchart");
  const key = el("latMetric").value || "interval";
  const n = state.calls.length;
  if (!n) {
    host.textContent = "";
    el("latNow").textContent = "—";
    el("latHint").textContent = "";
    return;
  }

  const series = [];
  for (let i = 0; i < n; i++) {
    const v = latValue(i, key);
    series.push(typeof v === "number" && isFinite(v) ? v : null);
  }
  const pts = series.map((v, i) => ({i, v})).filter((p) => p.v !== null);

  if (!pts.length) {
    host.textContent = "";
    el("latNow").textContent = "—";
    el("latHint").textContent =
      "该 run 的 trace 里没有 timing 字段 —— 推理耗时是服务端算过但未落盘。用打了这个补丁的服务端重跑一次即可显示。";
    return;
  }

  const cur = state.calls.indexOf(currentCall());
  const curV = series[cur];
  const {median, max} = latStat(pts.map((p) => p.v));

  const W = Math.max(220, host.clientWidth || 340);
  const H = Math.max(120, host.clientHeight || 180);
  const P = {t: 14, r: 12, b: 18, l: 54};  // room for 6-digit millisecond labels
  const pw = W - P.l - P.r, ph = H - P.t - P.b;
  const x = (i) => P.l + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);

  const vals = pts.map((p) => p.v);
  const rawHi = Math.max(...vals);
  const rawLo = Math.min(...vals);

  // A metric spanning decades - a 300 ms baseline next to a 50 s stall - is only
  // readable on a log axis; linear buries the band you actually care about.
  const useLog = rawLo > 0 && rawHi / rawLo > 20;

  let lo, hi, y, ticks;
  if (useLog) {
    const lg = Math.log10;
    lo = rawLo / 1.7;
    hi = rawHi * 1.4;
    y = (v) => P.t + ph - ((lg(Math.max(v, lo)) - lg(lo)) / (lg(hi) - lg(lo))) * ph;
    ticks = logTicks(lo, hi);
  } else {
    lo = Math.max(0, rawLo - ((rawHi - rawLo) * 0.18 || 1));
    hi = rawHi + ((rawHi - rawLo) * 0.18 || 1);
    y = (v) => P.t + ph - ((v - lo) / (hi - lo || 1)) * ph;
    ticks = niceTicks(lo, hi, 3);
  }

  // Always milliseconds. Mixing a seconds tick beside a bare millisecond one was
  // worse than either - and a metric that is usually 1.2 s should say 1200.
  const unit = "ms";
  const fmt = (v) => String(Math.round(v));

  el("latNow").innerHTML = (curV === null || curV === undefined ? "当前 —" : `当前 <b>${fmt(curV)} ${unit}</b>`) +
    ` · 中位 ${fmt(median)} ${unit} · 最大 ${fmt(max)} ${unit}`;
  el("latHint").textContent = key === "interval"
    ? "调用间隔 = 动作播放时间 + 推理停顿，不等于推理耗时；勾选服务端计时指标可看真实耗时。"
    : "";

  host.textContent = "";
  const svg = svgNode("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"});
  svg.setAttribute("aria-label", "延迟随调用序号的变化");

  for (const t of ticks) {
    const yy = y(t);
    if (yy < P.t - 1 || yy > P.t + ph + 1) continue;
    svg.appendChild(svgNode("line", {
      x1: P.l, x2: P.l + pw, y1: yy, y2: yy, stroke: "var(--gridline)", "stroke-width": 1,
    }));
    const lab = svgNode("text", {x: P.l - 5, y: yy + 3, "text-anchor": "end", class: "tick"});
    lab.textContent = fmt(t);
    svg.appendChild(lab);
  }
  // The axis unit, stated once, in the corner clear of the tick labels.
  const unitLab = svgNode("text", {x: 4, y: 11, "text-anchor": "start", class: "tick"});
  unitLab.textContent = unit;
  svg.appendChild(unitLab);
  svg.appendChild(svgNode("line", {
    x1: P.l, x2: P.l + pw, y1: P.t + ph, y2: P.t + ph, stroke: "var(--axis)", "stroke-width": 1,
  }));
  for (const i of [0, Math.floor((n - 1) / 2), n - 1]) {
    const lab = svgNode("text", {x: x(i), y: H - 4, "text-anchor": "middle", class: "tick"});
    lab.textContent = i + 1;
    svg.appendChild(lab);
  }

  // The line breaks where a metric is missing rather than interpolating over it.
  let d = "", pen = false;
  for (let i = 0; i < n; i++) {
    const v = series[i];
    if (v === null) { pen = false; continue; }
    d += (pen ? "L" : "M") + x(i) + "," + y(v);
    pen = true;
  }
  svg.appendChild(svgNode("path", {
    d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }));

  // Mark the call currently shown in the panels above.
  if (curV !== null && curV !== undefined) {
    svg.appendChild(svgNode("line", {
      x1: x(cur), x2: x(cur), y1: P.t, y2: P.t + ph, stroke: "var(--text-muted)", "stroke-width": 1,
    }));
    svg.appendChild(svgNode("circle", {
      cx: x(cur), cy: y(curV), r: 4,
      fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2,
    }));
  }

  // Nearest-point hover, so every value is reachable without a tooltip layer.
  const hit = svgNode("rect", {x: P.l, y: P.t, width: pw, height: ph, fill: "transparent"});
  hit.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - P.l) / pw) * (n - 1))));
    if (series[i] === null) return;
    el("latNow").innerHTML = `第 ${i + 1} 帧 <b>${fmt(series[i])} ${unit}</b>` +
      ` · 中位 ${fmt(median)} ${unit} · 最大 ${fmt(max)} ${unit}`;
  });
  hit.addEventListener("mouseleave", () => renderLat());
  svg.appendChild(hit);
  host.appendChild(svg);
}

/** The task instruction the model was handed for this call, shown whole. */
function renderInstr() {
  const c = currentCall();
  const tag = el("taskTag");
  const txt = el("instrText");
  if (!c) { tag.textContent = "task —"; txt.textContent = "—"; return; }
  const slug = TASK_SLUGS[c.task_index];
  tag.textContent = `slot ${c.task_index}` + (slug ? ` · ${slug}` : "");
  txt.textContent = c.instruction || "（该帧没有 instruction）";
}

function renderTiles() {
  const c = currentCall();
  const host = el("tiles");
  if (!c) { host.innerHTML = '<div class="empty">该 run 还没有推理记录。</div>'; return; }
  const idx = state.calls.indexOf(c);
  let gapTxt = "—";
  if (idx > 0) {
    const ms = (c.t - state.calls[idx - 1].t) * 1000;
    gapTxt = ms >= 1000 ? (ms / 1000).toFixed(2) + " <small>s</small>" : Math.round(ms) + " <small>ms</small>";
  }
  const pos = `${c.call}` + (state.follow ? "" : ` <small>· 已固定 ${idx + 1}/${state.calls.length}</small>`);
  const tiles = [
    ["调用序号", pos],
    ["任务", c.instruction ? c.instruction.slice(0, 34) + (c.instruction.length > 34 ? "…" : "") : `slot ${c.task_index}`],
    ["env_idx", `${c.env_id}`],
    ["距上次调用", gapTxt],
    ["左爪 / 右爪", `${c.state[6].toFixed(3)} / ${c.state[13].toFixed(3)}`],
  ];
  host.innerHTML = tiles.map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

const CAM_LABELS = {cam_high: "头部 cam_high", cam_left_wrist: "左腕 cam_left_wrist", cam_right_wrist: "右腕 cam_right_wrist"};
// Display order is deliberate: wrist views above and below the head view, so the
// three frames read top-to-bottom as left arm / scene / right arm.
const CAM_ORDER = ["cam_left_wrist", "cam_high", "cam_right_wrist"];
let shotEls = null, shotKey = "";

/** Cameras in display order, with anything unrecognised appended. */
function orderedCams(images) {
  const known = CAM_ORDER.filter((cam) => images.indexOf(cam) >= 0);
  return known.concat(images.filter((cam) => CAM_ORDER.indexOf(cam) < 0));
}

function renderShots() {
  const c = currentCall();
  const host = el("shots");
  if (!c || !c.images || !c.images.length) {
    host.innerHTML = '<div class="empty">没有图像。</div>';
    shotEls = null; shotKey = "";
    return;
  }
  const cams = orderedCams(c.images);
  // Build the <img> elements once and only swap src afterwards: rebuilding them
  // on every scrub frame makes the images flicker.
  const key = cams.join(",");
  if (!shotEls || shotKey !== key) {
    host.textContent = "";
    shotEls = cams.map((cam) => {
      const box = document.createElement("div");
      box.className = "shot";
      const img = document.createElement("img");
      img.alt = cam;
      const cap = document.createElement("div");
      cap.className = "cap";
      cap.textContent = CAM_LABELS[cam] || cam;
      box.append(img, cap);
      host.appendChild(box);
      return img;
    });
    shotKey = key;
  }
  const tag = String(c.call).padStart(7, "0");
  cams.forEach((cam, i) => {
    shotEls[i].src = `/api/image?run=${encodeURIComponent(state.run)}&name=${tag}_${cam}.jpg`;
  });
}

function renderTable() {
  const c = currentCall();
  const host = el("dataTable");
  if (!c) { host.innerHTML = ""; return; }
  const head = `<tr><th>step</th>${DIM_LABELS.map((l) => `<th>${l}</th>`).join("")}</tr>`;
  const stateRow = `<tr class="state"><td>state</td>${c.state.map((v) => `<td>${v.toFixed(3)}</td>`).join("")}</tr>`;
  const rows = c.action.map((row, i) =>
    `<tr><td>${i}</td>${row.map((v) => `<td>${v.toFixed(3)}</td>`).join("")}</tr>`).join("");
  host.innerHTML = head + stateRow + rows;
}

function render() {
  const grid = el("grid");
  if (!state.calls.length) {
    grid.innerHTML = '<div class="empty">该 run 还没有推理记录。</div>';
    renderTiles(); renderShots(); renderInstr(); renderLat();
    if (!el("tableView").hidden) renderTable();
    syncScrub();
    return;
  }
  const domains = DIM_LABELS.map((_, d) => domainFor(d));
  grid.textContent = "";
  let dim = 0;
  for (let row = 0; row < 2; row++) {
    for (const g of GROUPS.slice(row * 2, row * 2 + 2)) {
      const head = document.createElement("div");
      head.className = "grouphead" + (g.from === g.to ? " one" : "");
      head.textContent = g.label;
      grid.appendChild(head);
    }
    for (let i = 0; i < 7; i++, dim++) {
      const panel = document.createElement("div");
      panel.className = "panel";
      grid.appendChild(panel);
      drawPanel(dim, panel, domains[dim]);
    }
  }
  renderTiles(); renderShots(); renderInstr(); renderLat();
  if (!el("tableView").hidden) renderTable();
  syncScrub();
}

/* ---------- live polling ---------- */

async function loadRuns(select) {
  const res = await fetch("/api/runs");
  const data = await res.json();
  const sel = el("run");
  sel.innerHTML = data.runs.map((r) =>
    `<option value="${r.run}">${r.run}${r.policy_name ? " · " + r.policy_name : ""}</option>`).join("");
  if (!data.runs.length) { el("status").textContent = "没有找到 run"; return; }
  const want = select || state.run || data.runs[0].run;
  sel.value = data.runs.some((r) => r.run === want) ? want : data.runs[0].run;
  if (sel.value !== state.run) resetTo(sel.value);
}

function resetTo(run) {
  state.run = run; state.offset = 0; state.calls = []; state.hoverStep = null;
  state.boundaries = []; state.pinned = 0; state.follow = true;
  el("follow").checked = true;
  el("status").textContent = "加载中…";
  hideTip();
}

let ticks = 0;
async function poll() {
  setDot(false);
  try {
    // New runs appear whenever the server restarts; refresh the list without
    // disturbing the run already on screen.
    if (!state.run || ++ticks % 10 === 0) await loadRuns();
    if (!state.run) return;
    const url = `/api/trace?run=${encodeURIComponent(state.run)}&offset=${state.offset}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    state.offset = data.offset;

    let changed = false;
    for (const ev of data.events) {
      if (ev.event === "call") {
        if (ev.action && ev.action.length) { state.calls.push(ev); changed = true; }
      } else if (ev.event === "reset") {
        state.lastReset = ev.t;
        // A reset marks where the next episode begins. Consecutive resets (the
        // server emits several per boundary) collapse to one; leading ones give 0.
        const at = state.calls.length;
        if (state.boundaries[state.boundaries.length - 1] !== at) {
          state.boundaries.push(at);
          changed = true;
        }
      }
    }
    if (changed) { syncPrev(); render(); }
    setDot(true);
    const c = currentCall();
    el("status").textContent =
      `${state.run} · ${state.calls.length} 次调用` + (c ? ` · 当前 #${c.call}` : "") +
      (state.follow ? " · 已同步" : " · 已固定");
  } catch (err) {
    setDot(false);
    el("status").textContent = "读取失败：" + err.message;
  }
}

function setDot(live) { el("dot").className = "dot" + (live ? " live" : ""); }

/* ---------- episode scrubber ---------- */

/** Coalesce scrub renders into one per frame: dragging fires many input events. */
let raf = null;
function scheduleRender() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = null; render(); });
}

/** Redraw just the latency panel when its box changes size. */
let latRaf = null;
function scheduleLat() {
  if (latRaf) return;
  latRaf = requestAnimationFrame(() => { latRaf = null; renderLat(); });
}

function syncScrub() {
  const n = state.calls.length;
  const c = currentCall();
  const idx = c ? state.calls.indexOf(c) : 0;
  const s = el("slider");
  s.max = String(Math.max(1, n));
  s.value = String(n ? idx + 1 : 1);
  s.disabled = n === 0;

  const starts = episodeStarts();
  el("pos").textContent = n
    ? `${idx + 1} / ${n} · 第 ${episodeOf(idx)}/${starts.length} 轮`
    : "0 / 0";
  el("jump").max = String(Math.max(1, n));
  el("play").textContent = state.playing ? "⏸ 暂停" : "⏵ 播放";
  el("play").disabled = n === 0;

  // Episode starts as ticks under the track (0 is the start, so it gets none).
  const host = el("marks");
  const edges = starts.filter((b) => b > 0);
  if (host.childElementCount !== edges.length) {
    host.textContent = "";
    for (let i = 0; i < edges.length; i++) host.appendChild(document.createElement("i"));
  }
  edges.forEach((b, k) => {
    const tick = host.children[k];
    tick.style.left = ((b / n) * 100).toFixed(3) + "%";
    tick.title = `第 ${k + 2} 轮起点 · 第 ${b + 1} 帧`;
  });
}

/** Move to call ``i``. ``follow`` re-attaches to the newest call instead. */
function setIndex(i, follow) {
  const n = state.calls.length;
  if (!n) return;
  if (follow) {
    state.follow = true;
    state.pinned = n - 1;
  } else {
    state.follow = false;
    state.pinned = Math.max(0, Math.min(n - 1, i));
  }
  el("follow").checked = state.follow;
  scheduleRender();
}

function stepBy(delta) {
  const c = currentCall();
  const cur = c ? state.calls.indexOf(c) : 0;
  const n = state.calls.length;
  const next = cur + delta;
  // Walking off the right end re-attaches to live rather than pinning the last frame.
  if (next >= n - 1 && delta > 0) setIndex(n - 1, true);
  else setIndex(next, false);
}

function togglePlay() {
  state.playing = !state.playing;
  if (state.playing) {
    const c = currentCall();
    if (state.follow || !c || state.calls.indexOf(c) >= state.calls.length - 1) setIndex(0, false);
    state.timer = setInterval(() => {
      const cur = state.calls.indexOf(currentCall());
      if (cur >= state.calls.length - 1) { togglePlay(); return; }
      setIndex(cur + 1, false);
    }, 400);
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  syncScrub();
}

/* ---------- wiring ---------- */

/** The prev-chunk overlay compares each call against the one before it. */
function syncPrev() {
  state.calls.forEach((c, i) => { c.prev = state.showPrev && i > 0 ? state.calls[i - 1].action : null; });
}

el("run").addEventListener("change", (e) => { resetTo(e.target.value); render(); poll(); });
el("follow").addEventListener("change", (e) => {
  state.follow = e.target.checked;
  // Unchecking freezes the call on screen, so a 1.5 s cadence stays readable.
  if (!state.follow) state.pinned = Math.max(0, state.calls.length - 1);
  render();
});
el("prev").addEventListener("change", (e) => { state.showPrev = e.target.checked; syncPrev(); render(); });
el("table").addEventListener("change", (e) => {
  el("tableView").hidden = !e.target.checked;
  if (!e.target.checked) return;
  renderTable();
});
el("theme").addEventListener("click", () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
});

el("slider").addEventListener("input", (e) => setIndex(Number(e.target.value) - 1, false));
el("jump").addEventListener("change", (e) => {
  const v = Number(e.target.value);
  if (Number.isFinite(v) && v >= 1) setIndex(v - 1, false);
  e.target.value = "";
});
el("back1").addEventListener("click", () => stepBy(-1));
el("fwd1").addEventListener("click", () => stepBy(1));
el("toFirst").addEventListener("click", () => setIndex(0, false));
el("toLast").addEventListener("click", () => setIndex(state.calls.length - 1, true));
el("play").addEventListener("click", togglePlay);
el("latMetric").addEventListener("change", () => renderLat());

// The camera column's height settles only once its images load, which resizes
// the latency panel; redraw it then so the chart fills its final box.
if (window.ResizeObserver) new ResizeObserver(scheduleLat).observe(el("latchart"));

document.addEventListener("keydown", (ev) => {
  const tag = (ev.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;
  const n = state.calls.length;
  if (!n) return;
  const d = ev.shiftKey ? 10 : 1;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); stepBy(-d); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); stepBy(d); }
  else if (ev.key === "Home") { ev.preventDefault(); setIndex(0, false); }
  else if (ev.key === "End") { ev.preventDefault(); setIndex(n - 1, true); }
  else if (ev.key === " ") { ev.preventDefault(); togglePlay(); }
});

loadRuns().then(poll);
setInterval(poll, 500);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-dir", type=Path, default=Path(DEFAULT_TRACE_DIR))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--run", help="Pin a run instead of following the newest")
    args = parser.parse_args()

    trace_dir = args.trace_dir.expanduser().resolve()
    if not trace_dir.is_dir():
        parser.error(f"trace 目录不存在: {trace_dir}")

    runs = list_runs(trace_dir)
    if not runs:
        print(f"[warn] {trace_dir} 下没有找到任何 run（缺 trace.jsonl），页面会是空的")

    Handler.trace_dir = trace_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"trace 目录 : {trace_dir}")
    print(f"发现 run   : {len(runs)}" + (f", 最新 = {runs[0]['run']}" if runs else ""))
    print(f"打开       : http://{shown}:{args.port}/")
    if args.run:
        print(f"默认运行   : {args.run}")
    print("Ctrl-C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
