"""Lightweight face association and temporal identity confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from .liveness import BlinkChallenge


Box = tuple[int, int, int, int]
EyeBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    box: Box
    name: str
    distance: float | None
    eyes: tuple[EyeBox, ...] = ()
    eye_ratio: float | None = None
    confirmed_identity: str | None = None
    liveness_verified: bool = False
    track_id: int | None = None
    passive_live_score: float | None = None
    head_pose_label: str | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    gaze_direction: str | None = None
    gaze_confidence: float | None = None
    age_band: str | None = None
    age_confidence: float | None = None
    age_uncertainty: str | None = None


@dataclass
class _Track:
    track_id: int
    box: Box
    candidate_name: str
    candidate_hits: int
    confirmed_name: str | None
    missed_updates: int = 0
    velocity: Box = (0, 0, 0, 0)
    blink_challenge: BlinkChallenge | None = None
    last_output: Detection | None = None
    passive_live_hits: int = 0


def intersection_over_union(first: Box, second: Box) -> float:
    first_top, first_right, first_bottom, first_left = first
    second_top, second_right, second_bottom, second_left = second
    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)
    width = max(0, intersection_right - intersection_left)
    height = max(0, intersection_bottom - intersection_top)
    intersection = width * height
    first_area = max(0, first_right - first_left) * max(0, first_bottom - first_top)
    second_area = max(0, second_right - second_left) * max(0, second_bottom - second_top)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class FaceTracker:
    def __init__(
        self,
        confirmations: int = 3,
        *,
        iou_threshold: float = 0.2,
        max_missed_updates: int = 2,
        unknown_grace_updates: int = 1,
        require_liveness: bool = True,
        blink_threshold: float = 0.18,
        require_passive: bool = False,
        passive_threshold: float = 0.70,
        passive_confirmations: int = 2,
    ) -> None:
        self.confirmations = confirmations
        self.iou_threshold = iou_threshold
        self.max_missed_updates = max_missed_updates
        self.unknown_grace_updates = unknown_grace_updates
        self.require_liveness = require_liveness
        self.blink_threshold = blink_threshold
        self.require_passive = require_passive
        self.passive_threshold = passive_threshold
        self.passive_confirmations = passive_confirmations
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_track_ids = set(self._tracks)
        output: list[Detection] = []

        for detection in detections:
            track = self._best_track(detection.box, unmatched_track_ids)
            if track is None:
                track = self._new_track(detection)
            else:
                unmatched_track_ids.remove(track.track_id)
                previous_box = track.box
                track.box = detection.box
                track.velocity = tuple(
                    current - previous
                    for current, previous in zip(detection.box, previous_box)
                )
                track.missed_updates = 0
                if detection.name == track.candidate_name:
                    track.candidate_hits += 1
                else:
                    track.candidate_name = detection.name
                    track.candidate_hits = 1
                    if track.blink_challenge is not None:
                        track.blink_challenge.reset()
                    track.passive_live_hits = 0

            display_name, confirmed_identity, is_live = self._display_state(
                track, detection
            )
            displayed = Detection(
                box=detection.box,
                name=display_name,
                distance=detection.distance,
                eyes=detection.eyes,
                eye_ratio=detection.eye_ratio,
                confirmed_identity=confirmed_identity,
                liveness_verified=is_live,
                track_id=track.track_id,
                passive_live_score=detection.passive_live_score,
                head_pose_label=detection.head_pose_label,
                pitch=detection.pitch,
                yaw=detection.yaw,
                roll=detection.roll,
                gaze_direction=detection.gaze_direction,
                gaze_confidence=detection.gaze_confidence,
                age_band=detection.age_band,
                age_confidence=detection.age_confidence,
                age_uncertainty=detection.age_uncertainty,
            )
            track.last_output = displayed
            output.append(displayed)

        for track_id in unmatched_track_ids:
            self._tracks[track_id].missed_updates += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed_updates <= self.max_missed_updates
        }
        return output

    def predict(self, fraction: float) -> list[Detection]:
        """Project active boxes between recognition updates using recent motion."""
        fraction = max(0.0, min(1.0, fraction))
        predicted: list[Detection] = []
        for track in self._tracks.values():
            if track.last_output is None or track.missed_updates > 0:
                continue
            shifted_box = tuple(
                round(value + velocity * fraction)
                for value, velocity in zip(track.box, track.velocity)
            )
            predicted.append(
                Detection(
                    box=shifted_box,
                    name=track.last_output.name,
                    distance=track.last_output.distance,
                    eyes=track.last_output.eyes,
                    eye_ratio=track.last_output.eye_ratio,
                    confirmed_identity=track.last_output.confirmed_identity,
                    liveness_verified=track.last_output.liveness_verified,
                    track_id=track.track_id,
                    passive_live_score=track.last_output.passive_live_score,
                    head_pose_label=track.last_output.head_pose_label,
                    pitch=track.last_output.pitch,
                    yaw=track.last_output.yaw,
                    roll=track.last_output.roll,
                    gaze_direction=track.last_output.gaze_direction,
                    gaze_confidence=track.last_output.gaze_confidence,
                    age_band=track.last_output.age_band,
                    age_confidence=track.last_output.age_confidence,
                    age_uncertainty=track.last_output.age_uncertainty,
                )
            )
        return predicted

    def _best_track(self, box: Box, available_ids: set[int]) -> _Track | None:
        ranked = sorted(
            (
                (intersection_over_union(box, self._tracks[track_id].box), track_id)
                for track_id in available_ids
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.iou_threshold:
            return None
        return self._tracks[ranked[0][1]]

    def _new_track(self, detection: Detection) -> _Track:
        confirmed = "Unknown" if detection.name == "Unknown" else None
        track = _Track(
            track_id=self._next_track_id,
            box=detection.box,
            candidate_name=detection.name,
            candidate_hits=1,
            confirmed_name=confirmed,
            blink_challenge=BlinkChallenge(closed_threshold=self.blink_threshold),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _display_state(
        self, track: _Track, detection: Detection
    ) -> tuple[str, str | None, bool]:
        if track.candidate_name == "Unknown":
            if (
                track.confirmed_name not in (None, "Unknown")
                and track.candidate_hits <= self.unknown_grace_updates
            ):
                return track.confirmed_name, track.confirmed_name, False
            track.confirmed_name = "Unknown"
            if track.blink_challenge is not None:
                track.blink_challenge.reset()
            track.passive_live_hits = 0
            return "Unknown", None, False

        if track.candidate_hits >= self.confirmations:
            track.confirmed_name = track.candidate_name
        if track.confirmed_name == track.candidate_name:
            assert track.blink_challenge is not None
            if self.require_liveness:
                track.blink_challenge.update(detection.eye_ratio)
                if not track.blink_challenge.verified:
                    return (
                        f"{track.confirmed_name} - {track.blink_challenge.prompt}",
                        track.confirmed_name,
                        False,
                    )

            if self.require_passive:
                score = detection.passive_live_score
                if score is not None and score >= self.passive_threshold:
                    track.passive_live_hits += 1
                else:
                    track.passive_live_hits = 0
                if track.passive_live_hits < self.passive_confirmations:
                    status = "PASSIVE CHECK"
                    if score is not None and score < self.passive_threshold:
                        status = "SPOOF WARNING"
                    return (
                        f"{track.confirmed_name} - {status}",
                        track.confirmed_name,
                        False,
                    )

            if self.require_liveness or self.require_passive:
                return f"{track.confirmed_name} [LIVE]", track.confirmed_name, True
            return track.confirmed_name, track.confirmed_name, False
        return "Verifying...", None, False
