"""Lucas-Kanade optical-flow projection between recognition frames."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .tracking import Detection


def shift_detection(detection: Detection, dx: float, dy: float) -> Detection:
    top, right, bottom, left = detection.box
    return replace(
        detection,
        box=(
            round(top + dy),
            round(right + dx),
            round(bottom + dy),
            round(left + dx),
        ),
    )


class OpticalFlowProjector:
    def __init__(self) -> None:
        self.previous_gray: Any | None = None
        self.points_by_track: dict[int, Any] = {}

    def reset(self, gray: Any, detections: list[Detection], cv2: Any) -> None:
        import numpy as np

        self.previous_gray = gray.copy()
        self.points_by_track = {}
        height, width = gray.shape[:2]
        for detection in detections:
            if detection.track_id is None:
                continue
            top, right, bottom, left = detection.box
            top, bottom = max(0, top), min(height, bottom)
            left, right = max(0, left), min(width, right)
            if bottom <= top or right <= left:
                continue
            mask = np.zeros_like(gray)
            mask[top:bottom, left:right] = 255
            points = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                maxCorners=30,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=7,
            )
            if points is not None and len(points) >= 3:
                self.points_by_track[detection.track_id] = points

    def project(
        self,
        gray: Any,
        fallback_detections: list[Detection],
        cv2: Any,
    ) -> list[Detection]:
        import numpy as np

        if self.previous_gray is None:
            return fallback_detections
        projected: list[Detection] = []
        next_points: dict[int, Any] = {}
        for detection in fallback_detections:
            track_id = detection.track_id
            old_points = self.points_by_track.get(track_id) if track_id is not None else None
            if old_points is None or len(old_points) < 3:
                projected.append(detection)
                continue
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                old_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if new_points is None or status is None:
                projected.append(detection)
                continue
            valid = status.reshape(-1) == 1
            good_new = new_points.reshape(-1, 2)[valid]
            good_old = old_points.reshape(-1, 2)[valid]
            if len(good_new) < 3:
                projected.append(detection)
                continue
            movement = good_new - good_old
            dx, dy = np.median(movement, axis=0)
            top, right, bottom, left = detection.box
            max_shift = max(5.0, 0.15 * max(right - left, bottom - top))
            dx = float(np.clip(dx, -max_shift, max_shift))
            dy = float(np.clip(dy, -max_shift, max_shift))
            projected.append(shift_detection(detection, dx, dy))
            next_points[track_id] = good_new.reshape(-1, 1, 2).astype(np.float32)

        self.previous_gray = gray.copy()
        self.points_by_track.update(next_points)
        return projected
