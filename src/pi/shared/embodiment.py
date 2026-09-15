"""Shared embodiment metadata and indexing contract."""

from __future__ import annotations

from collections.abc import Mapping

EMBODIMENT_METADATA_KEY = "pi:embodiment"
LEGACY_EMBODIMENT_METADATA_KEY = "embodiment"
DEFAULT_EMBODIMENT_NAMES = ("piperX", "airbot")


def _normalized_name(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "")


def embodiment_name(index: int) -> str:
    if index < len(DEFAULT_EMBODIMENT_NAMES):
        return DEFAULT_EMBODIMENT_NAMES[index]
    return f"embodiment_{index}"


def embodiment_index(value: str | int, *, num_embodiments: int) -> int:
    """Resolve a stable embodiment name or numeric index."""
    if isinstance(value, bool):
        raise ValueError("embodiment index must not be boolean")
    if isinstance(value, int):
        index = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("embodiment value must not be empty")
        try:
            index = int(text)
        except ValueError:
            aliases = {"piperx": 0, "piper": 0, "airbot": 1}
            normalized = _normalized_name(text)
            if normalized not in aliases:
                raise ValueError(f"unknown embodiment name: {value!r}") from None
            index = aliases[normalized]
    if index < 0 or index >= num_embodiments:
        raise ValueError(f"embodiment index {index} is outside [0, {num_embodiments})")
    return index


def embodiment_metadata_value(metadata: Mapping[object, object]) -> str | None:
    for key in (EMBODIMENT_METADATA_KEY, LEGACY_EMBODIMENT_METADATA_KEY):
        value = metadata.get(key)
        if value is None:
            value = metadata.get(key.encode())
        if value is not None:
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def resolve_dataset_embodiment(
    metadata: Mapping[object, object],
    *,
    num_embodiments: int,
    fallback: str | int | None = None,
) -> tuple[int, str]:
    """Resolve metadata first, then an explicit fallback, then piperX=0."""
    metadata_value = embodiment_metadata_value(metadata)
    fallback_index = None if fallback is None else embodiment_index(fallback, num_embodiments=num_embodiments)
    if metadata_value is not None:
        metadata_index = embodiment_index(metadata_value, num_embodiments=num_embodiments)
        if fallback_index is not None and fallback_index != metadata_index:
            raise ValueError(
                f"dataset embodiment metadata {metadata_value!r} resolves to {metadata_index}, "
                f"but fallback resolves to {fallback_index}"
            )
        return metadata_index, f"metadata:{EMBODIMENT_METADATA_KEY}"
    if fallback_index is not None:
        return fallback_index, "DATASET_EMBODIMENTS"
    return 0, "default:piperX"


def build_embodiment_contract(*, num_embodiments: int, tokens_per_embodiment: int) -> dict[str, object]:
    return {
        "metadata_key": EMBODIMENT_METADATA_KEY,
        "index_to_name": {str(index): embodiment_name(index) for index in range(num_embodiments)},
        "tokens_per_embodiment": tokens_per_embodiment,
    }
