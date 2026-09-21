"""Basic blink-challenge liveness signals from six-point eye landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence


Point = tuple[int, int]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def eye_aspect_ratio(points: Sequence[Point]) -> float | None:
    """Return the standard six-point eye aspect ratio."""
    if len(points) != 6:
        return None
    horizontal = _distance(points[0], points[3])
    if horizontal == 0:
        return None
    vertical = _distance(points[1], points[5]) + _distance(points[2], points[4])
    return vertical / (2 * horizontal)


def average_eye_aspect_ratio(
    landmarks: Mapping[str, Sequence[Point]],
) -> float | None:
    values = [
        value
        for value in (
            eye_aspect_ratio(landmarks.get("left_eye", ())),
            eye_aspect_ratio(landmarks.get("right_eye", ())),
        )
        if value is not None
    ]
    return sum(values) / len(values) if values else None


class BlinkPhase(Enum):
    FIND_OPEN_EYES = "Look at camera"
    WAIT_FOR_BLINK = "Blink now"
    WAIT_FOR_REOPEN = "Open eyes"
    VERIFIED = "Live"


@dataclass
class BlinkChallenge:
    closed_threshold: float = 0.18
    required_open_updates: int = 2
    required_closed_updates: int = 1
    phase: BlinkPhase = BlinkPhase.FIND_OPEN_EYES
    open_updates: int = 0
    closed_updates: int = 0

    @property
    def verified(self) -> bool:
        return self.phase is BlinkPhase.VERIFIED

    @property
    def prompt(self) -> str:
        return self.phase.value

    def update(self, eye_ratio: float | None) -> BlinkPhase:
        if self.verified or eye_ratio is None:
            return self.phase

        eyes_closed = eye_ratio < self.closed_threshold
        if self.phase is BlinkPhase.FIND_OPEN_EYES:
            self.open_updates = self.open_updates + 1 if not eyes_closed else 0
            if self.open_updates >= self.required_open_updates:
                self.phase = BlinkPhase.WAIT_FOR_BLINK
        elif self.phase is BlinkPhase.WAIT_FOR_BLINK:
            self.closed_updates = self.closed_updates + 1 if eyes_closed else 0
            if self.closed_updates >= self.required_closed_updates:
                self.phase = BlinkPhase.WAIT_FOR_REOPEN
        elif self.phase is BlinkPhase.WAIT_FOR_REOPEN and not eyes_closed:
            self.phase = BlinkPhase.VERIFIED
        return self.phase

    def reset(self) -> None:
        self.phase = BlinkPhase.FIND_OPEN_EYES
        self.open_updates = 0
        self.closed_updates = 0
