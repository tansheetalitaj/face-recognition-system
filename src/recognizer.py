"""Face matching logic kept separate from camera and UI code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .enrollment import KnownFaces


class DistanceBackend(Protocol):
    def face_distance(
        self, known_face_encodings: Sequence[Any], face_encoding: Any
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class RecognitionResult:
    name: str
    distance: float | None
    matched: bool


def identify_face(
    known_faces: KnownFaces,
    face_encoding: Any,
    backend: DistanceBackend,
    tolerance: float,
) -> RecognitionResult:
    distances = list(backend.face_distance(known_faces.encodings, face_encoding))
    if not distances:
        return RecognitionResult(name="Unknown", distance=None, matched=False)

    best_index = min(range(len(distances)), key=distances.__getitem__)
    best_distance = float(distances[best_index])
    if best_distance <= tolerance:
        return RecognitionResult(
            name=known_faces.names[best_index],
            distance=best_distance,
            matched=True,
        )

    return RecognitionResult(name="Unknown", distance=best_distance, matched=False)
