"""Msgpack + NumPy wire codec shared by the GOAI server and test clients.

Kept outside scripts/ to avoid the scripts/inference.py namespace clash
(the repo ships a scripts/inference.py CLI alongside scripts/inference/).
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import numpy as np
import msgspec


def _encode_numpy(value: Any) -> Any:
    """Encode NumPy values using msgpack-numpy's wire representation."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("O", "V", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": b"",
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in ("O", "V", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        return {b"nd": False, b"type": value.dtype.str, b"data": value.tobytes()}
    raise TypeError(f"Unsupported msgpack value: {type(value).__name__}")


def _decode_numpy(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_numpy(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get(b"nd")
    if marker is True:
        if value.get(b"kind") in (b"O", b"V"):
            raise ValueError("Object and structured NumPy arrays are not supported")
        dtype = np.dtype(value[b"type"])
        shape = tuple(int(part) for part in value[b"shape"])
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(shape).copy()
    if marker is False:
        dtype = np.dtype(value[b"type"])
        return np.frombuffer(value[b"data"], dtype=dtype, count=1)[0]
    return {key: _decode_numpy(item) for key, item in value.items()}


def encode_frame(frame: Mapping[str, Any]) -> bytes:
    return msgspec.msgpack.encode(dict(frame), enc_hook=_encode_numpy)


def decode_frame(payload: bytes) -> dict[str, Any]:
    decoded = _decode_numpy(msgspec.msgpack.decode(payload))
    if not isinstance(decoded, dict):
        raise ValueError("WebSocket frame must be a msgpack map")
    return decoded


