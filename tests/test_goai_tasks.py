"""Protect official canonical task language and legacy slot compatibility."""

import pytest

from pi.shared.goai_tasks import (
    GOAI_REAL_TASK_INSTRUCTIONS,
    GOAI_REAL_LEGACY_TASK_INSTRUCTIONS,
    build_task_remap,
    match_task_details,
)

OFFICIAL = (
    "Pick up the pen holder and place all the pens into it.",
    "Place all the objects on the table into the basket.",
    "Stack the blocks on the table, then cover them with the cup.",
    "Stack the bowls on the table.",
    "Stand the bottle upright.",
    "Insert the charger plug into the power strip, then connect the charging cable to the plug.",
)


def test_canonical_table_is_official_language():
    assert GOAI_REAL_TASK_INSTRUCTIONS == OFFICIAL


@pytest.mark.parametrize("slot", range(6))
def test_matcher_accepts_official_and_legacy_in_same_slot(slot):
    for instruction in (OFFICIAL[slot], GOAI_REAL_LEGACY_TASK_INSTRUCTIONS[slot]):
        result = match_task_details(instruction)
        assert (result.slot, result.instruction, result.score, result.method) == (slot, OFFICIAL[slot], 1.0, "exact")


def test_dataset_remap_uses_official_language_and_not_local_order():
    tasks = {str(i): instruction for i, instruction in enumerate(reversed(OFFICIAL))}
    remap = build_task_remap(tasks)
    assert [remap[i].slot for i in range(6)] == [5, 4, 3, 2, 1, 0]
