"""GOAI real-robot (PiperX) 14D state and action contract.

The GOAI dual-arm policy consumes a 14-dimensional proprioception vector and emits
14-dimensional action chunks; both use the same layout:

    arm_left        0:6     left arm joint positions
    gripper_left    6:7     left gripper opening
    arm_right       7:13    right arm joint positions
    gripper_right  13:14    right gripper opening

Datasets stamp the layout as ``piperx-14d-v1``. Training converts the six arm joints
of every arm to deltas relative to the current state and keeps the grippers
absolute, and a checkpoint records the projection it was trained with so training
and inference cannot drift apart.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Mapping, Sequence

GOAI_POLICY_STATE_CONTRACT_SCHEMA = "goai-policy-state-contract/v1"
# Layout of the state/action columns as they are stored in a GOAI dataset.
GOAI_SOURCE_STATE_SCHEMA = "piperx-14d-v1"
GOAI_SOURCE_STATE_DIM = 14
# Model-visible layout. Source and policy layouts coincide, so the projection
# selects every source index; the schema exists so a checkpoint can declare the
# layout it trained on.
GOAI_POLICY_STATE_SCHEMA_14D = "goai-policy-state-14d-v1"
GOAI_POLICY_STATE_LAYOUT_14D: tuple[tuple[str, int, int], ...] = (
    ("arm_left", 0, 6),
    ("gripper_left", 6, 7),
    ("arm_right", 7, 13),
    ("gripper_right", 13, 14),
)

GOAI_ARM_JOINT_DIM = 6
# Six arm-joint targets per arm are relative to the current state; the grippers
# stay absolute, so they are excluded from the delta transform.
GOAI_JOINT_DELTA_MASK: tuple[bool, ...] = (
    (True,) * GOAI_ARM_JOINT_DIM + (False,) + (True,) * GOAI_ARM_JOINT_DIM + (False,)
)


def _layout_sha256(layout: Sequence[tuple[str, int, int]]) -> str:
    payload = [{"name": name, "start": start, "stop": stop} for name, start, stop in layout]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_goai_policy_state_schema(value: str | None) -> str:
    """Resolve user-facing aliases to the canonical GOAI policy-state schema."""
    normalized = "source" if value is None else str(value).strip().lower()
    aliases = {
        "source": GOAI_POLICY_STATE_SCHEMA_14D,
        "identity": GOAI_POLICY_STATE_SCHEMA_14D,
        "14d": GOAI_POLICY_STATE_SCHEMA_14D,
        "goai14d": GOAI_POLICY_STATE_SCHEMA_14D,
        GOAI_POLICY_STATE_SCHEMA_14D: GOAI_POLICY_STATE_SCHEMA_14D,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported GOAI policy-state schema {value!r}; expected one of: {supported}") from error


def goai_policy_state_indices(value: str | None = None) -> tuple[int, ...]:
    """Return source-state indices retained by the resolved policy-state schema."""
    resolve_goai_policy_state_schema(value)
    return tuple(index for _name, start, stop in GOAI_POLICY_STATE_LAYOUT_14D for index in range(start, stop))


def build_goai_policy_state_contract(value: str | None = None) -> dict[str, Any]:
    """Build the checkpoint-side model-input state contract."""
    state_schema = resolve_goai_policy_state_schema(value)
    source_indices = goai_policy_state_indices(state_schema)
    return {
        "schema": GOAI_POLICY_STATE_CONTRACT_SCHEMA,
        "state_schema": state_schema,
        "source_state_schema": GOAI_SOURCE_STATE_SCHEMA,
        "source_state_dim": GOAI_SOURCE_STATE_DIM,
        "policy_state_dim": len(source_indices),
        "source_indices": list(source_indices),
        "layout_sha256": _layout_sha256(GOAI_POLICY_STATE_LAYOUT_14D),
    }


def validate_goai_policy_state_contract(value: object) -> dict[str, Any]:
    """Validate an archived model-input state contract exactly."""
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint is missing policy_state_contract")
    contract = dict(value)
    expected = build_goai_policy_state_contract(contract.get("state_schema"))
    mismatches = [
        f"{key}={contract.get(key)!r} (expected {expected_value!r})"
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError("GOAI policy-state contract mismatch: " + ", ".join(mismatches))
    return expected
