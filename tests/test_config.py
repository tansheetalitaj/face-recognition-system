import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import AppConfig, ConfigurationError


class AppConfigTests(unittest.TestCase):
    def test_reads_environment_values(self) -> None:
        environment = {
            "FACE_CAMERA_INDEX": "2",
            "FACE_TOLERANCE": "0.52",
            "FACE_FRAME_SCALE": "0.25",
            "FACE_PROCESS_EVERY_N_FRAMES": "3",
            "FACE_CONFIRMATION_FRAMES": "4",
            "FACE_BLINK_THRESHOLD": "0.17",
            "FACE_KNOWN_FACES_DIR": "example_faces",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_environment()

        self.assertEqual(config.camera_index, 2)
        self.assertEqual(config.tolerance, 0.52)
        self.assertEqual(config.frame_scale, 0.25)
        self.assertEqual(config.process_every_n_frames, 3)
        self.assertEqual(config.confirmation_frames, 4)
        self.assertEqual(config.blink_threshold, 0.17)
        self.assertEqual(config.known_faces_dir, Path("example_faces"))

    def test_rejects_invalid_scale(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(frame_scale=0).validate()

    def test_rejects_invalid_confirmation_count(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(confirmation_frames=0).validate()

    def test_rejects_invalid_blink_threshold(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(blink_threshold=1).validate()

    def test_rejects_invalid_passive_threshold(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(passive_anti_spoof_threshold=0).validate()

    def test_passive_anti_spoof_is_disabled_by_default(self) -> None:
        self.assertFalse(AppConfig().require_passive_anti_spoof)

    def test_age_estimation_is_opt_in(self) -> None:
        self.assertFalse(AppConfig().enable_age_estimation)

    def test_event_log_requires_consent(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(event_log_path=Path("events.jsonl")).validate()

    def test_rejects_invalid_event_retention(self) -> None:
        with self.assertRaises(ConfigurationError):
            AppConfig(event_retention_days=0).validate()

    def test_cli_overrides_preserve_other_values(self) -> None:
        config = AppConfig(camera_index=1, tolerance=0.6)
        updated = config.with_overrides(tolerance=0.5)

        self.assertEqual(updated.camera_index, 1)
        self.assertEqual(updated.tolerance, 0.5)


if __name__ == "__main__":
    unittest.main()
