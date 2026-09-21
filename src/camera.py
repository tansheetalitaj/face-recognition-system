"""Webcam capture, frame processing, and visualization."""

from __future__ import annotations

import logging
import os
from typing import Any

from .age_estimation import AgeEstimator
from .anti_spoof import PassiveAntiSpoofDetector
from .config import AppConfig
from .consent import consent_problems, require_active_consent
from .encoding_cache import load_cached_known_faces
from .data_protection import default_data_protector
from .event_log import RecognitionEventLogger
from .face_analysis import FaceGeometryAnalyzer
from .eye_detection import eye_boxes_from_landmarks, select_eye_boxes
from .liveness import average_eye_aspect_ratio
from .optical_flow import OpticalFlowProjector
from .recognizer import identify_face
from .tracking import Detection, FaceTracker


LOGGER = logging.getLogger(__name__)


def _open_camera(cv2: Any, camera_index: int) -> Any:
    if os.name == "nt":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def _process_frame(
    frame: Any,
    *,
    cv2: Any,
    face_backend: Any,
    eye_cascade: Any,
    known_faces: Any,
    config: AppConfig,
    anti_spoof_detector: PassiveAntiSpoofDetector | None,
    geometry_analyzer: FaceGeometryAnalyzer,
    age_estimator: AgeEstimator | None,
) -> list[Detection]:
    import numpy as np

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=config.frame_scale,
        fy=config.frame_scale,
    )
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    full_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    geometry_results = geometry_analyzer.analyze(full_rgb_frame, cv2, np)
    locations = face_backend.face_locations(rgb_frame)
    encodings = face_backend.face_encodings(rgb_frame, locations)
    landmark_sets = face_backend.face_landmarks(
        rgb_frame,
        locations,
        model="large",
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces: list[Detection] = []

    for location, encoding, landmarks in zip(locations, encodings, landmark_sets):
        top, right, bottom, left = (
            int(coordinate / config.frame_scale) for coordinate in location
        )
        result = identify_face(
            known_faces,
            encoding,
            face_backend,
            config.tolerance,
        )

        height, width = gray.shape[:2]
        top = max(0, min(top, height))
        bottom = max(0, min(bottom, height))
        left = max(0, min(left, width))
        right = max(0, min(right, width))

        face_box = (top, right, bottom, left)
        geometry = geometry_analyzer.match(face_box, geometry_results)
        age = age_estimator.predict(frame, face_box) if age_estimator is not None else None
        eyes = eye_boxes_from_landmarks(
            landmarks,
            frame_scale=config.frame_scale,
            face_box=face_box,
        )
        if bottom > top and right > left:
            if len(eyes) != 2:
                roi_gray = gray[top:bottom, left:right]
                eye_boxes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                )
                eyes = select_eye_boxes(
                    eye_boxes,
                    face_width=right - left,
                    face_height=bottom - top,
                )

        detected_faces.append(
            Detection(
                box=(top, right, bottom, left),
                name=result.name,
                distance=result.distance,
                eyes=eyes,
                eye_ratio=average_eye_aspect_ratio(landmarks),
                passive_live_score=(
                    anti_spoof_detector.predict(frame, face_box, cv2).real_score
                    if anti_spoof_detector is not None
                    else None
                ),
                head_pose_label=(
                    geometry.head_pose.label
                    if geometry is not None and geometry.head_pose is not None
                    else None
                ),
                pitch=(
                    geometry.head_pose.pitch
                    if geometry is not None and geometry.head_pose is not None
                    else None
                ),
                yaw=(
                    geometry.head_pose.yaw
                    if geometry is not None and geometry.head_pose is not None
                    else None
                ),
                roll=(
                    geometry.head_pose.roll
                    if geometry is not None and geometry.head_pose is not None
                    else None
                ),
                gaze_direction=(
                    geometry.gaze.direction
                    if geometry is not None and geometry.gaze is not None
                    else None
                ),
                gaze_confidence=(
                    geometry.gaze.confidence
                    if geometry is not None and geometry.gaze is not None
                    else None
                ),
                age_band=age.band if age is not None else None,
                age_confidence=age.confidence if age is not None else None,
                age_uncertainty=age.uncertainty if age is not None else None,
            )
        )

    return detected_faces


def _draw_detections(frame: Any, detections: list[Detection], cv2: Any) -> None:
    for detection in detections:
        top, right, bottom, left = detection.box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        label = detection.name
        if detection.distance is not None:
            label = f"{label} ({detection.distance:.2f})"
        cv2.putText(
            frame,
            label,
            (left, max(20, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        details: list[str] = []
        if detection.head_pose_label is not None:
            details.append(
                f"Pose: {detection.head_pose_label} "
                f"P{detection.pitch:+.0f} Y{detection.yaw:+.0f} R{detection.roll:+.0f}"
            )
        if detection.gaze_direction is not None:
            details.append(
                f"Gaze: {detection.gaze_direction} ({detection.gaze_confidence:.0%})"
            )
        else:
            details.append("Gaze: unavailable")
        if detection.age_band is not None:
            details.append(
                f"Age estimate: {detection.age_band} "
                f"({detection.age_uncertainty}, {detection.age_confidence:.0%})"
            )
        for line_number, detail in enumerate(details, start=1):
            cv2.putText(
                frame,
                detail,
                (left, min(frame.shape[0] - 10, bottom + 24 * line_number)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                2,
            )

        for eye_x, eye_y, eye_width, eye_height in detection.eyes:
            start = (left + eye_x, top + eye_y)
            end = (left + eye_x + eye_width, top + eye_y + eye_height)
            cv2.rectangle(frame, start, end, (0, 255, 0), 2)


def run_camera(config: AppConfig) -> None:
    try:
        import cv2
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "Runtime dependencies are missing. Activate the project virtual environment "
            "and run: python -m pip install -r requirements.txt"
        ) from exc

    cache_protector = (
        default_data_protector() if config.encrypt_encoding_cache else None
    )
    known_faces = load_cached_known_faces(
        config.known_faces_dir,
        face_recognition,
        config.encoding_cache_path,
        rebuild=config.rebuild_cache,
        protector=cache_protector,
    )
    if config.require_consent_manifest:
        require_active_consent(config.consent_manifest_path, known_faces.names)
    else:
        for problem in consent_problems(config.consent_manifest_path, known_faces.names):
            LOGGER.warning("Consent: %s", problem)
    LOGGER.info(
        "Loaded %d enrollment image(s) for %d identity/identities: %s",
        len(known_faces),
        len(set(known_faces.names)),
        ", ".join(sorted(set(known_faces.names))),
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    if eye_cascade.empty():
        raise RuntimeError("OpenCV's eye detection model could not be loaded.")

    camera = _open_camera(cv2, config.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Cannot open camera index {config.camera_index}. "
            "Check camera permissions and whether another application is using it."
        )

    LOGGER.info("Press Q or ESC to quit.")
    tracker = FaceTracker(
        confirmations=config.confirmation_frames,
        require_liveness=config.require_liveness,
        blink_threshold=config.blink_threshold,
        require_passive=config.require_passive_anti_spoof,
        passive_threshold=config.passive_anti_spoof_threshold,
    )
    anti_spoof_detector = (
        PassiveAntiSpoofDetector(
            config.anti_spoof_model_path,
            real_threshold=config.passive_anti_spoof_threshold,
        )
        if config.require_passive_anti_spoof
        else None
    )
    geometry_analyzer = FaceGeometryAnalyzer()
    age_estimator = (
        AgeEstimator(
            config.age_model_definition_path,
            config.age_model_weights_path,
            cv2,
        )
        if config.enable_age_estimation
        else None
    )
    event_logger = (
        RecognitionEventLogger(
            config.event_log_path,
            consent_confirmed=config.event_logging_consent,
            require_liveness=config.require_liveness,
            retention_days=config.event_retention_days,
        )
        if config.event_log_path is not None
        else None
    )
    detections: list[Detection] = []
    optical_flow = OpticalFlowProjector()
    frame_number = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera stopped returning frames.")

            if frame_number % config.process_every_n_frames == 0:
                raw_detections = _process_frame(
                    frame,
                    cv2=cv2,
                    face_backend=face_recognition,
                    eye_cascade=eye_cascade,
                    known_faces=known_faces,
                    config=config,
                    anti_spoof_detector=anti_spoof_detector,
                    geometry_analyzer=geometry_analyzer,
                    age_estimator=age_estimator,
                )
                detections = tracker.update(raw_detections)
                optical_flow.reset(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    detections,
                    cv2,
                )
                if event_logger is not None:
                    event_logger.record(detections)
            else:
                fallback_detections = tracker.predict(
                    (frame_number % config.process_every_n_frames)
                    / config.process_every_n_frames
                )
                detections = optical_flow.project(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    fallback_detections,
                    cv2,
                )

            _draw_detections(frame, detections, cv2)
            cv2.imshow(config.window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            frame_number += 1
    finally:
        geometry_analyzer.close()
        camera.release()
        cv2.destroyAllWindows()
