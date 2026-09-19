"""Droid RLDS dataset implementation."""

from enum import Enum, auto


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
