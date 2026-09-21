"""Optional, age-only estimation with explicit coarse-band uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


AGE_BANDS = ("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+")


@dataclass(frozen=True)
class AgeEstimate:
    band: str
    confidence: float

    @property
    def uncertainty(self) -> str:
        if self.confidence < 0.35:
            return "very uncertain"
        if self.confidence < 0.60:
            return "uncertain"
        return "moderate confidence"


class AgeEstimator:
    """OpenCV DNN wrapper for the age-only Levi/Hassner eight-band model."""

    def __init__(self, definition_path: Path, weights_path: Path, cv2: Any) -> None:
        if not definition_path.is_file() or not weights_path.is_file():
            raise RuntimeError(
                "Age model files are missing. Run: python setup_models.py --include-age"
            )
        self._cv2 = cv2
        self._network = cv2.dnn.readNet(str(weights_path), str(definition_path))

    def predict(self, frame: Any, box: tuple[int, int, int, int]) -> AgeEstimate | None:
        top, right, bottom, left = box
        height, width = frame.shape[:2]
        padding = round(max(right - left, bottom - top) * 0.12)
        crop = frame[
            max(0, top - padding) : min(height, bottom + padding),
            max(0, left - padding) : min(width, right + padding),
        ]
        if crop.size == 0:
            return None
        blob = self._cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
            crop=False,
        )
        self._network.setInput(blob)
        probabilities = self._network.forward().reshape(-1)
        if probabilities.size != len(AGE_BANDS):
            raise RuntimeError(f"Unexpected age model output: {probabilities.shape}")
        index = int(probabilities.argmax())
        return AgeEstimate(AGE_BANDS[index], float(probabilities[index]))
