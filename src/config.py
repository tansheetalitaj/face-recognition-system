"""Application configuration with environment and CLI override support."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}.") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {value!r}.") from exc


@dataclass(frozen=True)
class AppConfig:
    known_faces_dir: Path = PROJECT_ROOT / "known_faces"
    encoding_cache_path: Path = PROJECT_ROOT / ".cache" / "face_encodings.enc"
    encrypt_encoding_cache: bool = True
    camera_index: int = 0
    tolerance: float = 0.6
    frame_scale: float = 0.5
    process_every_n_frames: int = 2
    confirmation_frames: int = 3
    require_liveness: bool = True
    blink_threshold: float = 0.18
    require_passive_anti_spoof: bool = False
    passive_anti_spoof_threshold: float = 0.70
    anti_spoof_model_path: Path = PROJECT_ROOT / "models" / "anti-spoof-mn3.onnx"
    enable_age_estimation: bool = False
    age_model_definition_path: Path = PROJECT_ROOT / "models" / "age_deploy.prototxt"
    age_model_weights_path: Path = PROJECT_ROOT / "models" / "age_net.caffemodel"
    rebuild_cache: bool = False
    event_log_path: Path | None = None
    event_logging_consent: bool = False
    event_retention_days: int = 30
    consent_manifest_path: Path = PROJECT_ROOT / "known_faces" / "consent.json"
    require_consent_manifest: bool = False
    window_title: str = "Face Recognition + Pose/Gaze"

    @classmethod
    def from_environment(cls) -> "AppConfig":
        configured_dir = os.getenv("FACE_KNOWN_FACES_DIR")
        configured_cache = os.getenv("FACE_ENCODING_CACHE")
        config = cls(
            known_faces_dir=(
                Path(configured_dir).expanduser()
                if configured_dir
                else PROJECT_ROOT / "known_faces"
            ),
            encoding_cache_path=(
                Path(configured_cache).expanduser()
                if configured_cache
                else PROJECT_ROOT / ".cache" / "face_encodings.enc"
            ),
            camera_index=_read_int("FACE_CAMERA_INDEX", 0),
            tolerance=_read_float("FACE_TOLERANCE", 0.6),
            frame_scale=_read_float("FACE_FRAME_SCALE", 0.5),
            process_every_n_frames=_read_int("FACE_PROCESS_EVERY_N_FRAMES", 2),
            confirmation_frames=_read_int("FACE_CONFIRMATION_FRAMES", 3),
            blink_threshold=_read_float("FACE_BLINK_THRESHOLD", 0.18),
        )
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        camera_index: int | None = None,
        tolerance: float | None = None,
        frame_scale: float | None = None,
        process_every_n_frames: int | None = None,
        confirmation_frames: int | None = None,
        require_liveness: bool | None = None,
        blink_threshold: float | None = None,
        require_passive_anti_spoof: bool | None = None,
        passive_anti_spoof_threshold: float | None = None,
        enable_age_estimation: bool | None = None,
        known_faces_dir: Path | None = None,
        rebuild_cache: bool | None = None,
        event_log_path: Path | None = None,
        event_logging_consent: bool | None = None,
        event_retention_days: int | None = None,
        encrypt_encoding_cache: bool | None = None,
        require_consent_manifest: bool | None = None,
    ) -> "AppConfig":
        config = replace(
            self,
            camera_index=self.camera_index if camera_index is None else camera_index,
            tolerance=self.tolerance if tolerance is None else tolerance,
            frame_scale=self.frame_scale if frame_scale is None else frame_scale,
            process_every_n_frames=(
                self.process_every_n_frames
                if process_every_n_frames is None
                else process_every_n_frames
            ),
            confirmation_frames=(
                self.confirmation_frames
                if confirmation_frames is None
                else confirmation_frames
            ),
            require_liveness=(
                self.require_liveness
                if require_liveness is None
                else require_liveness
            ),
            blink_threshold=(
                self.blink_threshold if blink_threshold is None else blink_threshold
            ),
            require_passive_anti_spoof=(
                self.require_passive_anti_spoof
                if require_passive_anti_spoof is None
                else require_passive_anti_spoof
            ),
            passive_anti_spoof_threshold=(
                self.passive_anti_spoof_threshold
                if passive_anti_spoof_threshold is None
                else passive_anti_spoof_threshold
            ),
            enable_age_estimation=(
                self.enable_age_estimation
                if enable_age_estimation is None
                else enable_age_estimation
            ),
            known_faces_dir=(
                self.known_faces_dir if known_faces_dir is None else known_faces_dir
            ),
            rebuild_cache=(
                self.rebuild_cache if rebuild_cache is None else rebuild_cache
            ),
            event_log_path=(
                self.event_log_path if event_log_path is None else event_log_path
            ),
            event_logging_consent=(
                self.event_logging_consent
                if event_logging_consent is None
                else event_logging_consent
            ),
            event_retention_days=(
                self.event_retention_days
                if event_retention_days is None
                else event_retention_days
            ),
            encrypt_encoding_cache=(
                self.encrypt_encoding_cache
                if encrypt_encoding_cache is None
                else encrypt_encoding_cache
            ),
            require_consent_manifest=(
                self.require_consent_manifest
                if require_consent_manifest is None
                else require_consent_manifest
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.camera_index < 0:
            raise ConfigurationError("Camera index must be zero or greater.")
        if not 0 < self.tolerance <= 1:
            raise ConfigurationError("Recognition tolerance must be in the range (0, 1].")
        if not 0 < self.frame_scale <= 1:
            raise ConfigurationError("Frame scale must be in the range (0, 1].")
        if self.process_every_n_frames < 1:
            raise ConfigurationError("Frame processing interval must be at least 1.")
        if self.confirmation_frames < 1:
            raise ConfigurationError("Confirmation frames must be at least 1.")
        if not 0 < self.blink_threshold < 1:
            raise ConfigurationError("Blink threshold must be in the range (0, 1).")
        if not 0 < self.passive_anti_spoof_threshold < 1:
            raise ConfigurationError(
                "Passive anti-spoof threshold must be in the range (0, 1)."
            )
        if self.event_log_path is not None and not self.event_logging_consent:
            raise ConfigurationError(
                "Event logging requires explicit --consent-to-event-logging."
            )
        if self.event_retention_days < 1:
            raise ConfigurationError("Event retention must be at least one day.")
