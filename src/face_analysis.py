"""Head-pose and iris-based gaze estimates from MediaPipe face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


Point3D = tuple[float, float, float]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float

    @property
    def label(self) -> str:
        horizontal = "LEFT" if self.yaw < -15 else "RIGHT" if self.yaw > 15 else "CENTER"
        vertical = "UP" if self.pitch < -12 else "DOWN" if self.pitch > 12 else "LEVEL"
        return f"{horizontal}/{vertical}"


@dataclass(frozen=True)
class GazeEstimate:
    direction: str
    horizontal_ratio: float
    vertical_ratio: float
    confidence: float


@dataclass(frozen=True)
class FaceGeometry:
    box: Box
    head_pose: HeadPose | None
    gaze: GazeEstimate | None


def _point(landmarks: Sequence[Point3D], index: int, width: int, height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark[0] * width, landmark[1] * height


def canonical_pitch(angle: float) -> float:
    """Resolve solvePnP's equivalent upside-down Euler representation."""
    normalized = (angle + 180.0) % 360.0 - 180.0
    if normalized < -90.0:
        normalized += 180.0
    elif normalized > 90.0:
        normalized -= 180.0
    return normalized


def estimate_head_pose(
    landmarks: Sequence[Point3D], width: int, height: int, cv2: Any, np: Any
) -> HeadPose | None:
    """Estimate Euler angles using a generic six-point 3D face model."""
    if len(landmarks) < 468:
        return None
    image_points = np.asarray(
        [
            _point(landmarks, 1, width, height),
            _point(landmarks, 152, width, height),
            _point(landmarks, 33, width, height),
            _point(landmarks, 263, width, height),
            _point(landmarks, 61, width, height),
            _point(landmarks, 291, width, height),
        ],
        dtype=np.float64,
    )
    model_points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.asarray(
        [
            (focal_length, 0.0, width / 2.0),
            (0.0, focal_length, height / 2.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return HeadPose(
        pitch=canonical_pitch(float(angles[0])),
        yaw=float(angles[1]),
        roll=float(angles[2]),
    )


def _ratio(value: float, first: float, second: float) -> float | None:
    low, high = sorted((first, second))
    span = high - low
    if span <= 1e-6:
        return None
    return max(0.0, min(1.0, (value - low) / span))


def estimate_gaze(landmarks: Sequence[Point3D]) -> GazeEstimate | None:
    """Classify image-relative gaze from both iris centres.

    MediaPipe's refined mesh supplies iris landmarks. Direction is deliberately
    image-relative (SCREEN LEFT/RIGHT), avoiding ambiguity from mirrored previews.
    """
    if len(landmarks) < 478:
        return None

    # (iris centre, horizontal corners, upper lid, lower lid)
    eyes = (
        (468, 33, 133, 159, 145),
        (473, 362, 263, 386, 374),
    )
    horizontal_values: list[float] = []
    vertical_values: list[float] = []
    openness_values: list[float] = []
    for iris, corner_a, corner_b, upper, lower in eyes:
        horizontal = _ratio(
            landmarks[iris][0], landmarks[corner_a][0], landmarks[corner_b][0]
        )
        vertical = _ratio(
            landmarks[iris][1], landmarks[upper][1], landmarks[lower][1]
        )
        eye_width = abs(landmarks[corner_a][0] - landmarks[corner_b][0])
        eye_height = abs(landmarks[upper][1] - landmarks[lower][1])
        if horizontal is None or vertical is None or eye_width <= 1e-6:
            continue
        horizontal_values.append(horizontal)
        vertical_values.append(vertical)
        openness_values.append(eye_height / eye_width)

    if len(horizontal_values) != 2 or min(openness_values) < 0.08:
        return None
    horizontal = sum(horizontal_values) / 2.0
    vertical = sum(vertical_values) / 2.0
    agreement = 1.0 - min(1.0, abs(horizontal_values[0] - horizontal_values[1]) * 2.5)
    confidence = max(0.0, min(1.0, agreement * min(1.0, min(openness_values) / 0.18)))
    if confidence < 0.35:
        return None

    horizontal_label = (
        "SCREEN LEFT" if horizontal < 0.38 else "SCREEN RIGHT" if horizontal > 0.62 else "CENTER"
    )
    vertical_label = "UP" if vertical < 0.34 else "DOWN" if vertical > 0.66 else "LEVEL"
    direction = horizontal_label if vertical_label == "LEVEL" else f"{horizontal_label}/{vertical_label}"
    return GazeEstimate(direction, horizontal, vertical, confidence)


def _intersection_over_union(first: Box, second: Box) -> float:
    first_top, first_right, first_bottom, first_left = first
    second_top, second_right, second_bottom, second_left = second
    left = max(first_left, second_left)
    top = max(first_top, second_top)
    right = min(first_right, second_right)
    bottom = min(first_bottom, second_bottom)
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first_right - first_left) * max(0, first_bottom - first_top)
    second_area = max(0, second_right - second_left) * max(0, second_bottom - second_top)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class FaceGeometryAnalyzer:
    """Run one refined face mesh per processed frame and associate its results."""

    def __init__(self, max_faces: int = 4) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for gaze estimation; install project requirements."
            ) from exc
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, rgb_frame: Any, cv2: Any, np: Any) -> list[FaceGeometry]:
        height, width = rgb_frame.shape[:2]
        result = self._mesh.process(rgb_frame)
        analyses: list[FaceGeometry] = []
        for face in result.multi_face_landmarks or ():
            landmarks = tuple((item.x, item.y, item.z) for item in face.landmark)
            xs = [item[0] for item in landmarks[:468]]
            ys = [item[1] for item in landmarks[:468]]
            box = (
                max(0, round(min(ys) * height)),
                min(width, round(max(xs) * width)),
                min(height, round(max(ys) * height)),
                max(0, round(min(xs) * width)),
            )
            analyses.append(
                FaceGeometry(
                    box=box,
                    head_pose=estimate_head_pose(landmarks, width, height, cv2, np),
                    gaze=estimate_gaze(landmarks),
                )
            )
        return analyses

    def match(self, face_box: Box, analyses: Sequence[FaceGeometry]) -> FaceGeometry | None:
        ranked = sorted(
            ((_intersection_over_union(face_box, analysis.box), analysis) for analysis in analyses),
            key=lambda item: item[0],
            reverse=True,
        )
        return ranked[0][1] if ranked and ranked[0][0] >= 0.15 else None

    def close(self) -> None:
        self._mesh.close()
