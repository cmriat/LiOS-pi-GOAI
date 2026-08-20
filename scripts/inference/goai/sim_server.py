#!/usr/bin/env python3
"""Serve a Pi05 GOAI checkpoint over the XPolicyLab WebSocket protocol."""

from __future__ import annotations

import os
import http
import json
import time
import asyncio
import logging
import argparse
import datetime as dt
from typing import Any, Mapping
from pathlib import Path

import numpy as np
import websockets
import websockets.asyncio.server as websocket_server

from pi.inference.goai_protocol import decode_frame, encode_frame
from pi.inference.goai_sim_policy import GOAISimPolicy

LOGGER = logging.getLogger(__name__)

REQUEST_RESPONSE_TYPES = {
    "hello": "hello_ack",
    "prepare_case": "prepare_case_ack",
    "reset": "reset_result",
    "infer": "infer_result",
    "trial_end": "trial_end_ack",
    "heartbeat": "heartbeat_ack",
}

FATAL_POLICY_ERROR_MARKERS = (
    "beginAllocateToPool",
    "CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED",
    "CUDA out of memory",
    "device-side assert",
    "illegal memory access",
)


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _reply(request: Mapping[str, Any], message_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "message_type": message_type,
        "message_id": str(request["message_id"]),
        "evaluation_id": str(request["evaluation_id"]),
        "action_case_id": request.get("action_case_id"),
        "trial_id": request.get("trial_id"),
        "repeat_index": request.get("repeat_index"),
        "step": int(request.get("step", 0)),
        "sent_at": _timestamp(),
        "payload": dict(payload),
    }


def _validate_request(frame: Mapping[str, Any]) -> None:
    for key in ("message_type", "message_id", "evaluation_id"):
        if key not in frame:
            raise ValueError(f"WebSocket frame is missing {key}")
    payload = frame.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("WebSocket frame payload must be a map")


_CURRENT_SERVER: "GOAIWebSocketServer | None" = None


def _health_check(_connection: Any, request: Any) -> Any:
    """GET /healthz returns a JSON health payload for ops monitoring."""
    if getattr(request, "path", None) != "/healthz":
        return None
    payload: dict[str, Any] = {
        "status": "ok",
        "time": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "pid": os.getpid(),
    }
    server = _CURRENT_SERVER
    if server is not None:
        payload["uptime_s"] = round(time.monotonic() - server._started_at, 1)
        payload["model"] = server.policy.metadata
        if server._last_bound_task is not None:
            payload["last_bound_task"] = server._last_bound_task
        payload["inference"] = {
            "requests": server._infer_count,
            "mean_ms": round(server._infer_total_ms / server._infer_count, 1) if server._infer_count else None,
            "max_ms": round(server._infer_max_ms, 1) if server._infer_count else None,
        }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    from websockets.http11 import Headers, Response

    return Response(
        http.HTTPStatus.OK.value,
        http.HTTPStatus.OK.phrase,
        Headers([("Content-Type", "application/json")]),
        body,
    )


def _request_task_name(request: Mapping[str, Any]) -> str | None:
    payload = request.get("payload")
    if isinstance(payload, Mapping):
        task_name = payload.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name.strip()
    action_case_id = request.get("action_case_id")
    if isinstance(action_case_id, str) and action_case_id.endswith("_case"):
        return action_case_id.removesuffix("_case")
    return None


class GOAIWebSocketServer:
    """Serve one model with isolated RNG state per WebSocket connection."""

    def __init__(self, policy: GOAISimPolicy, *, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self._policy_lock = asyncio.Lock()
        self._infer_count = 0
        self._infer_total_ms = 0.0
        self._infer_max_ms = 0.0
        self._started_at = time.monotonic()
        self._last_bound_task: dict[str, Any] | None = None
        # v1.0.0 CALL flow: update_obs stores the per-connection observation,
        # get_action reads it back for inference.
        self._session_obs: dict[Any, Mapping[str, Any]] = {}
        self._fatal_event: asyncio.Event | None = None
        self._fatal_error: str | None = None

    @staticmethod
    def _is_fatal_policy_error(error: Exception) -> bool:
        rendered = str(error)
        return any(marker in rendered for marker in FATAL_POLICY_ERROR_MARKERS)

    def _bind_session(self, request: Mapping[str, Any], session: Any) -> None:
        task_name = _request_task_name(request)
        repeat_index = request.get("repeat_index")
        seed = int(repeat_index) if repeat_index is not None else None
        if task_name is not None:
            self.policy.bind_session(session, task_name=task_name, seed=seed)
            # Record the latest binding for ops visibility (exposed via /healthz).
            self._last_bound_task = {
                "name": task_name,
                "index": session.task_index,
                "seed": seed,
                "at": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
            }

    async def _dispatch(self, request: Mapping[str, Any], session: Any) -> dict[str, Any] | None:
        _validate_request(request)
        message_type = str(request["message_type"])
        payload = request.get("payload", {})
        if message_type == "close":
            return None
        if message_type == "hello":
            return _reply(
                request,
                "hello_ack",
                {"ok": True, "server": "pi05_goai", "metadata": self.policy.metadata},
            )
        if message_type == "prepare_case":
            self._bind_session(request, session)
            return _reply(request, "prepare_case_ack", {"ok": True})
        if message_type == "reset":
            self._bind_session(request, session)
            self.policy.reset(session)
            self._session_obs.pop(id(session), None)
            return _reply(request, "reset_result", {"ok": True})
        if message_type == "heartbeat":
            return _reply(request, "heartbeat_ack", {"ok": True})
        if message_type == "trial_end":
            self._session_obs.pop(id(session), None)
            return _reply(request, "trial_end_ack", {"ok": True})
        if message_type == "call":
            # v1.0.0 CALL frame: payload.func_name dispatches to the policy;
            # results go back under payload.result. The CALL flow splits
            # update_obs (store) and get_action (infer from stored obs).
            func_name = payload.get("func_name")
            if not isinstance(func_name, str) or not func_name or func_name.startswith("_"):
                raise ValueError(f"call payload has invalid func_name: {func_name!r}")
            if func_name == "reset":
                self._bind_session(request, session)
                self.policy.reset(session)
                self._session_obs.pop(id(session), None)
                return _reply(request, "call_result", {"result": None, "ok": True})
            if func_name in {"update_obs", "update_obs_batch"}:
                obs = payload.get("obs")
                if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], Mapping):
                    obs = obs[0]  # unwrap single-env batch variants
                if not isinstance(obs, Mapping):
                    raise ValueError(f"{func_name} payload missing obs")
                self._bind_session(request, session)
                self._session_obs[id(session)] = dict(obs)
                return _reply(request, "call_result", {"result": None, "ok": True})
            if func_name in {"get_action", "get_action_batch"}:
                observation = self._session_obs.get(id(session))
                if observation is None:
                    raise ValueError(f"{func_name} called before update_obs")
                self._bind_session(request, session)
                started = time.perf_counter()
                async with self._policy_lock:
                    actions = await asyncio.to_thread(self.policy.infer, observation, session)
                latency_ms = (time.perf_counter() - started) * 1000.0
                self._infer_count += 1
                self._infer_total_ms += latency_ms
                self._infer_max_ms = max(self._infer_max_ms, latency_ms)
                return _reply(request, "call_result", {"result": actions, "latency_ms": latency_ms})
            raise ValueError(f"Unsupported call func_name: {func_name}")
        if message_type != "infer":
            raise ValueError(f"Unsupported WebSocket message_type: {message_type}")
        observation = payload.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("infer payload is missing observation")

        self._bind_session(request, session)
        started = time.perf_counter()
        async with self._policy_lock:
            actions = await asyncio.to_thread(self.policy.infer, observation, session)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._infer_count += 1
        self._infer_total_ms += latency_ms
        self._infer_max_ms = max(self._infer_max_ms, latency_ms)
        if self._infer_count % 50 == 0:
            LOGGER.info(
                "Inference stats: requests=%d mean_ms=%.1f max_ms=%.1f",
                self._infer_count,
                self._infer_total_ms / self._infer_count,
                self._infer_max_ms,
            )
        return _reply(request, "infer_result", {"actions": actions, "latency_ms": latency_ms})

    async def _handler(self, websocket: Any) -> None:
        session = self.policy.create_session()
        LOGGER.info("Connection opened: %s", websocket.remote_address)
        try:
            async for raw in websocket:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                request: dict[str, Any] | None = None
                fatal = False
                fatal_message: str | None = None
                try:
                    request = decode_frame(bytes(raw))
                    response = await self._dispatch(request, session)
                    if response is None:
                        await websocket.close()
                        return
                except Exception as error:
                    LOGGER.exception("GOAI policy request failed")
                    fatal = self._is_fatal_policy_error(error)
                    fatal_message = str(error) if fatal else None
                    if request is None or "message_id" not in request or "evaluation_id" not in request:
                        await websocket.close(code=1002, reason="Invalid XPolicyLab frame")
                        return
                    response = _reply(
                        request,
                        "error",
                        {"code": "infer_failed", "message": str(error), "details": {}},
                    )
                if fatal:
                    self._fatal_error = fatal_message
                    if self._fatal_event is not None:
                        self._fatal_event.set()
                await websocket.send(encode_frame(response))
                if fatal:
                    return
        except websockets.ConnectionClosed:
            pass
        finally:
            self._session_obs.pop(id(session), None)
            LOGGER.info("Connection closed: %s", websocket.remote_address)

    async def run(self) -> None:
        global _CURRENT_SERVER
        _CURRENT_SERVER = self
        LOGGER.info("Starting GOAI policy server on %s:%d", self.host, self.port)
        self._fatal_event = asyncio.Event()
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ):
            await self._fatal_event.wait()
        raise RuntimeError(f"Fatal GOAI policy error: {self._fatal_error}")

    def serve_forever(self) -> None:
        asyncio.run(self.run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--config-name", default="pi05_goai_joint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6060)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=8)  # submission default (closed-loop 8/20)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"),
        default="none",
    )
    parser.add_argument("--norm-mode", choices=("global", "per-timestamp"))
    parser.add_argument("--apply-delta", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    return parser


def _warmup(policy: GOAISimPolicy) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image.copy(),
            "cam_right_wrist": image.copy(),
        },
        "instruction": "stack the bowls",
    }
    session = policy.create_session()
    if session.task_name is None:
        policy.bind_session(session, task_name="stack_bowls")
    started = time.perf_counter()
    policy.infer(observation, session)
    if policy.device.type == "cuda":
        import torch

        torch.cuda.synchronize(policy.device)
    LOGGER.info("Policy warmup completed in %.2fs", time.perf_counter() - started)


def main() -> None:
    args = build_parser().parse_args()
    policy = GOAISimPolicy(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        task_name=args.task_name,
        config_name=args.config_name,
        device=args.device,
        action_horizon=args.action_horizon,
        num_steps=args.num_steps,
        compile_mode=args.compile_mode,
        norm_mode=args.norm_mode,
        apply_delta=args.apply_delta,
        seed=args.seed,
    )
    if args.warmup:
        _warmup(policy)
    GOAIWebSocketServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
