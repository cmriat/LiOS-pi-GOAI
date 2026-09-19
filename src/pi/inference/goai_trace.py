"""Optional on-disk trace of every inference call (observation + action).

Enabled from the server config:

    trace:
      enabled: true
      dir: ~/goai-trace
      image_quality: 90      # JPEG quality for the three cameras
      max_gb: 40             # stop writing (and warn once) past this budget

Layout (one run directory per server start):

    <dir>/<run_id>/meta.json          config/ckpt identity + start time
    <dir>/<run_id>/trace.jsonl        one line per call: state + full action chunk
    <dir>/<run_id>/images/<call>_<camera>.jpg

``trace.jsonl`` is the analysis surface: state[14] and action[16][14] per call make
the gripper trajectory plottable without decoding any image.  The JPEGs are the
context.  Everything is append-only and flushed per call, so the trace survives a
hard kill.

State/action layout follows ``_STATE_KEYS``: left arm 0-5, left gripper 6,
right arm 7-12, right gripper 13.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Mapping

import numpy as np

_DEFAULT_QUALITY = 90
_DEFAULT_MAX_GB = 40.0

# Must match _STATE_KEYS ordering: the flattened 14D vector is state; the same
# layout is reused for every step of the action chunk.
_ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)


def _as_list(value):
    return np.asarray(value).reshape(-1).astype(float).tolist()


def _jsonable(value):
    """Best-effort conversion so an unexpected frame can never abort a write."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class TraceWriter:
    """Append-only recorder; a no-op unless enabled in the config."""

    def __init__(self, model_cfg):
        raw = model_cfg.get("trace") or {}
        if not isinstance(raw, dict):
            raise ValueError("trace 段必须是 map")
        unknown = sorted(set(raw) - {"enabled", "dir", "image_quality", "max_gb"})
        if unknown:
            raise ValueError(f"trace 段有未知字段 {unknown};可选: ['enabled', 'dir', 'image_quality', 'max_gb']")
        self.enabled = bool(raw.get("enabled", False))
        self.quality = int(raw.get("image_quality", _DEFAULT_QUALITY))
        if not 1 <= self.quality <= 100:
            raise ValueError("trace.image_quality 需要在 [1, 100]")
        self.max_bytes = float(raw.get("max_gb", _DEFAULT_MAX_GB)) * (1024**3)
        self.root = None
        self._fh = None
        self._img_dir = None
        self._call = 0
        self._events = 0
        self.bytes_written = 0
        self._capped = False
        self._cv2 = None
        if not self.enabled:
            return
        base = Path(str(raw.get("dir", "~/goai-trace"))).expanduser()
        # Name the run after the checkpoint so `ls` distinguishes rounds without
        # having to open meta.json. `<ckpt>/ema` -> the checkpoint directory name.
        ckpt = Path(str(model_cfg.get("checkpoint_path", "")))
        name = ckpt.parent.name if ckpt.name in ("ema", "raw") else ckpt.name
        self.label = name or "unknown"
        self.run_id = f"{self.label}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.root = base / self.run_id
        self._img_dir = self.root / "images"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (self.root / "trace.jsonl").open("a", encoding="utf-8")
        print(f"[trace] 本次记录写入 {self.root}", flush=True)

    # ------------------------------------------------------------------ helpers

    def write_meta(self, payload):
        """Record run identity once, so a trace is self-describing."""
        if not self.enabled:
            return
        (self.root / "meta.json").write_text(
            json.dumps({"run_id": self.run_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _encoder(self):
        if self._cv2 is None:
            import cv2  # imported lazily: the server only needs it when tracing

            self._cv2 = cv2
        return self._cv2

    def _write_images(self, tag, images):
        cv2 = self._encoder()
        for name, image in images.items():
            # canonical_observation yields RGB; OpenCV encodes BGR, so swap back
            # or the JPEG colors come out wrong.
            bgr = np.ascontiguousarray(np.asarray(image)[..., ::-1])
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                continue
            path = self._img_dir / f"{tag}_{name}.jpg"
            path.write_bytes(buf.tobytes())
            self.bytes_written += len(buf)

    # -------------------------------------------------------------------- write

    def event(self, name, payload=None):
        """Append a non-inference protocol event (prepare_case / reset / trial_end).

        The frame envelope itself (trial_id, step, evaluation_id ...) never reaches
        the model layer - the official server keeps those on the Frame - so only
        what the model hooks actually receive can be recorded here.
        """
        if not self.enabled:
            return
        try:
            line = {"event": name, "t": round(time.time(), 3), "n": self._events}
            self._events += 1
            if payload:
                line.update(_jsonable(payload))
            raw = json.dumps(line, ensure_ascii=False) + "\n"
            self._fh.write(raw)
            self._fh.flush()
            self.bytes_written += len(raw)
        except Exception as exc:
            print(f"[trace] 事件写失败（已忽略）: {type(exc).__name__}: {exc}", flush=True)

    def record(
        self, *, env_id, task_index, obs, actions, raw=None, timing=None, infer_ms=None, e2e_ms=None, action_resample=None
    ):
        """Append one inference call. Never raises into the serving path."""
        if not self.enabled:
            return
        try:
            if self.bytes_written >= self.max_bytes:
                if not self._capped:
                    self._capped = True
                    print(f"[trace] 达到 max_gb 上限，停止写入图像（trace.jsonl 继续）: {self.root}", flush=True)
                images = {}
            else:
                images = obs.get("images") or {}

            self._call += 1
            tag = f"{self._call:07d}"
            line = {
                "event": "call",
                "t": round(time.time(), 3),
                "call": self._call,
                "env_id": int(env_id),
                "task_index": int(task_index),
                # The client's own words, before canonicalisation overwrites them.
                "raw": _jsonable(raw) if raw else None,
                # What the server matched it to and fed the model.
                "instruction": obs.get("instruction"),
                "state": _as_list(obs["state"]),
                "action": [sum((_as_list(step[k]) for k in _ACTION_KEYS), []) for step in actions],
                "n_action": len(actions),
                "images": sorted(images),
                # infer_ms = model forward only.
                # e2e_ms   = gap between the client's two frames (update_obs arrival ->
                #            get_action arrival): network round trip plus the client's own
                #            turnaround, and it EXCLUDES the server-side inference.
                # timing = backend stage breakdown, when the policy exposes one.
                "timing": _jsonable(timing) if timing else None,
                "infer_ms": None if infer_ms is None else round(infer_ms, 1),
                "e2e_ms": None if e2e_ms is None else round(e2e_ms, 1),
            }
            if action_resample is not None:
                line["action_resample"] = _jsonable(action_resample)
            raw = json.dumps(line, ensure_ascii=False) + "\n"
            self._fh.write(raw)
            self._fh.flush()
            self.bytes_written += len(raw)
            if images and not self._capped:
                self._write_images(tag, images)
        except Exception as exc:  # tracing must never take the server down
            print(f"[trace] 写失败（已忽略）: {type(exc).__name__}: {exc}", flush=True)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
