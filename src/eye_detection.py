"""Geometric filtering for noisy Haar-cascade eye detections."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence


EyeBox = tuple[int, int, int, int]
Point = tuple[int, int]


def eye_boxes_from_landmarks(
    landmarks: Mapping[str, Sequence[Point]],
    *,
    frame_scale: float,
    face_box: tuple[int, int, int, int],
) -> tuple[EyeBox, ...]:
    """Build eye boxes from face landmarks, including when the head is tilted."""
    if frame_scale <= 0:
        return ()

    top, right, bottom, left = face_box
    face_width = right - left
    face_height = bottom - top
    if face_width <= 0 or face_height <= 0:
        return ()

    boxes: list[EyeBox] = []
    for name in ("left_eye", "right_eye"):
        points = landmarks.get(name, ())
        if not points:
            continue

        full_x = [round(x / frame_scale) for x, _ in points]
        full_y = [round(y / frame_scale) for _, y in points]
        landmark_width = max(full_x) - min(full_x)
        landmark_height = max(full_y) - min(full_y)
        padding_x = max(3, round(landmark_width * 0.35))
        padding_y = max(3, round(max(landmark_height, landmark_width) * 0.30))

        box_left = max(left, min(full_x) - padding_x)
        box_top = max(top, min(full_y) - padding_y)
        box_right = min(right, max(full_x) + padding_x)
        box_bottom = min(bottom, max(full_y) + padding_y)
        width = box_right - box_left
        height = box_bottom - box_top
        if width > 0 and height > 0:
            boxes.append((box_left - left, box_top - top, width, height))

    return tuple(sorted(boxes, key=lambda box: box[0]))


def select_eye_boxes(
    boxes: Iterable[EyeBox],
    *,
    face_width: int,
    face_height: int,
) -> tuple[EyeBox, ...]:
    """Return at most one plausible left/right eye pair.

    Haar cascades commonly classify glasses frames, reflections, eyebrows, or
    the bridge of the nose as additional eyes. Candidates are restricted to the
    upper face and the most plausible horizontally separated pair is selected.
    """
    if face_width <= 0 or face_height <= 0:
        return ()

    candidates: list[EyeBox] = []
    for raw_box in boxes:
        x, y, width, height = (int(value) for value in raw_box)
        center_y = y + height / 2
        if not (0.08 * face_width <= width <= 0.45 * face_width):
            continue
        if not (0.06 * face_height <= height <= 0.35 * face_height):
            continue
        if center_y > 0.62 * face_height:
            continue
        candidates.append((x, y, width, height))

    if not candidates:
        return ()
    if len(candidates) == 1:
        return (candidates[0],)

    plausible_pairs: list[tuple[float, EyeBox, EyeBox]] = []
    for first, second in combinations(candidates, 2):
        first_center_x = first[0] + first[2] / 2
        second_center_x = second[0] + second[2] / 2
        first_center_y = first[1] + first[3] / 2
        second_center_y = second[1] + second[3] / 2
        horizontal_separation = abs(second_center_x - first_center_x)
        vertical_difference = abs(second_center_y - first_center_y)

        if horizontal_separation < 0.18 * face_width:
            continue
        if vertical_difference > 0.20 * face_height:
            continue

        size_difference = abs(first[2] - second[2]) + abs(first[3] - second[3])
        score = horizontal_separation - vertical_difference - 0.25 * size_difference
        plausible_pairs.append((score, first, second))

    if not plausible_pairs:
        largest = max(candidates, key=lambda box: box[2] * box[3])
        return (largest,)

    _, first, second = max(plausible_pairs, key=lambda item: item[0])
    return tuple(sorted((first, second), key=lambda box: box[0]))
