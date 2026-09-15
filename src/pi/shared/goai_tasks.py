"""Canonical GOAI real-task slots and language-based task alignment."""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass
from collections.abc import Mapping

GOAI_REAL_TASK_INSTRUCTIONS = (
    "Fill the pen holder",
    "Put the objects into the basket",
    "Stack and cover the blocks",
    "Stack the bowls",
    "Stand up the bottles",
    "Insert the charger",
)
# Official full instructions address the same embedding slots as the legacy names.
GOAI_REAL_OFFICIAL_TASK_INSTRUCTIONS = (
    "Pick up the pen holder and place all the pens into it.",
    "Place all the objects on the table into the basket.",
    "Stack the blocks on the table, then cover them with the cup.",
    "Stack the bowls on the table.",
    "Stand the bottle upright.",
    "Insert the charger plug into the power strip, then connect the charging cable to the plug.",
)
GOAI_TASK_MATCH_MIN_SCORE = 0.80
GOAI_TASK_MATCH_AMBIGUITY_MARGIN = 0.05


@dataclass(frozen=True)
class GOAITaskMatch:
    slot: int
    score: float
    method: str
    instruction: str


class GOAITaskMatchError(ValueError):
    """Raised when an instruction cannot be assigned to one canonical slot."""


def normalize_task_instruction(text: str) -> str:
    """Normalize task language without changing word order or semantics."""
    if not isinstance(text, str):
        raise TypeError(f"GOAI task instruction must be a string, got {type(text).__name__}")
    characters = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        if character == "_" or unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


_NORMALIZED_CANONICAL_TASKS = tuple(normalize_task_instruction(text) for text in GOAI_REAL_TASK_INSTRUCTIONS)


def _task_scores(text: str) -> list[tuple[int, float]]:
    normalized = normalize_task_instruction(text)
    return [
        (slot, difflib.SequenceMatcher(None, normalized, canonical).ratio())
        for slot, canonical in enumerate(_NORMALIZED_CANONICAL_TASKS)
    ]


def _score_table(text: str, scores: list[tuple[int, float]]) -> str:
    rows = [f"GOAI task match scores for {text!r}:"]
    rows.extend(f"  slot {slot}: {score:.4f}  {GOAI_REAL_TASK_INSTRUCTIONS[slot]}" for slot, score in scores)
    return "\n".join(rows)


def match_task_details(text: str) -> GOAITaskMatch:
    """Match language to one official real-task slot, rejecting weak or ambiguous matches."""
    normalized = normalize_task_instruction(text)
    exact_slots = [slot for slot, canonical in enumerate(_NORMALIZED_CANONICAL_TASKS) if normalized == canonical]
    if len(exact_slots) == 1:
        slot = exact_slots[0]
        return GOAITaskMatch(slot, 1.0, "exact", GOAI_REAL_TASK_INSTRUCTIONS[slot])

    scores = sorted(_task_scores(text), key=lambda item: (-item[1], item[0]))
    best_slot, best_score = scores[0]
    second_score = scores[1][1]
    table = _score_table(text, scores)
    if best_score < GOAI_TASK_MATCH_MIN_SCORE:
        raise GOAITaskMatchError(
            f"GOAI task instruction did not meet the minimum score "
            f"{GOAI_TASK_MATCH_MIN_SCORE:.2f} (best={best_score:.4f}).\n{table}"
        )
    if best_score - second_score < GOAI_TASK_MATCH_AMBIGUITY_MARGIN:
        raise GOAITaskMatchError(
            f"GOAI task instruction is ambiguous: best margin "
            f"{best_score - second_score:.4f} is below {GOAI_TASK_MATCH_AMBIGUITY_MARGIN:.2f}.\n{table}"
        )
    return GOAITaskMatch(best_slot, best_score, "fuzzy", GOAI_REAL_TASK_INSTRUCTIONS[best_slot])


def match_task(text: str) -> tuple[int, float]:
    """Return the official slot and similarity score for one instruction."""
    result = match_task_details(text)
    return result.slot, result.score


def build_task_remap(tasks: Mapping[str | int, str]) -> dict[int, GOAITaskMatch]:
    """Build a complete local-index to official-slot mapping for one dataset."""
    if not tasks:
        raise GOAITaskMatchError("GOAI Lance metadata has an empty lerobot:tasks_json mapping")

    remap: dict[int, GOAITaskMatch] = {}
    official_to_local: dict[int, int] = {}
    try:
        ordered_tasks = sorted(tasks.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as error:
        raise GOAITaskMatchError("GOAI task metadata contains a non-integer local index") from error
    for local_key, instruction in ordered_tasks:
        local_index = int(local_key)
        if local_index < 0:
            raise GOAITaskMatchError(f"Invalid negative local GOAI task index {local_index}")
        match = match_task_details(instruction)
        previous = official_to_local.get(match.slot)
        if previous is not None:
            raise GOAITaskMatchError(
                f"Local GOAI tasks {previous} and {local_index} both map to official slot {match.slot} "
                f"({match.instruction!r})"
            )
        remap[local_index] = match
        official_to_local[match.slot] = local_index
    return remap
