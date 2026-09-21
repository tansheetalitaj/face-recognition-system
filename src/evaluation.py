"""Metrics for offline open-set face-recognition evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


UNKNOWN_LABEL = "Unknown"
UNKNOWN_DATASET_FOLDER = "_unknown"


@dataclass(frozen=True)
class EvaluationObservation:
    expected: str
    nearest_identity: str
    distance: float


@dataclass
class EvaluationMetrics:
    known_samples: int = 0
    unknown_samples: int = 0
    correct_known: int = 0
    correct_unknown: int = 0
    false_rejections: int = 0
    false_accepts: int = 0
    misidentifications: int = 0
    skipped: int = 0

    def record(self, expected: str, predicted: str) -> None:
        if expected == UNKNOWN_DATASET_FOLDER:
            self.unknown_samples += 1
            if predicted == UNKNOWN_LABEL:
                self.correct_unknown += 1
            else:
                self.false_accepts += 1
            return

        self.known_samples += 1
        if predicted == expected:
            self.correct_known += 1
        elif predicted == UNKNOWN_LABEL:
            self.false_rejections += 1
        else:
            self.misidentifications += 1

    @property
    def false_accept_rate(self) -> float | None:
        if self.unknown_samples == 0:
            return None
        return self.false_accepts / self.unknown_samples

    @property
    def false_reject_rate(self) -> float | None:
        if self.known_samples == 0:
            return None
        return self.false_rejections / self.known_samples

    @property
    def identification_accuracy(self) -> float | None:
        total = self.known_samples + self.unknown_samples
        if total == 0:
            return None
        return (self.correct_known + self.correct_unknown) / total


def metrics_at_threshold(
    observations: Iterable[EvaluationObservation], tolerance: float
) -> EvaluationMetrics:
    metrics = EvaluationMetrics()
    for observation in observations:
        predicted = (
            observation.nearest_identity
            if observation.distance <= tolerance
            else UNKNOWN_LABEL
        )
        metrics.record(observation.expected, predicted)
    return metrics
