import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.event_log import ConsentRequiredError, RecognitionEventLogger
from src.tracking import Detection


class EventLogTests(unittest.TestCase):
    def test_requires_explicit_consent(self) -> None:
        with self.assertRaises(ConsentRequiredError):
            RecognitionEventLogger(Path("events.jsonl"), consent_confirmed=False)

    def test_logs_live_identity_once_during_cooldown(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            logger = RecognitionEventLogger(
                path,
                consent_confirmed=True,
                monotonic_clock=lambda: 100,
            )
            detection = Detection(
                box=(0, 10, 10, 0),
                name="alice [LIVE]",
                distance=0.31,
                confirmed_identity="alice",
                liveness_verified=True,
                track_id=1,
            )

            first_count = logger.record([detection])
            second_count = logger.record([detection])
            lines = path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[0])

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(payload["identity"], "alice")
        self.assertNotIn("image", payload)

    def test_skips_identity_before_liveness_verification(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            logger = RecognitionEventLogger(path, consent_confirmed=True)
            detection = Detection(
                box=(0, 10, 10, 0),
                name="alice - Blink now",
                distance=0.31,
                confirmed_identity="alice",
                liveness_verified=False,
            )

            count = logger.record([detection])

        self.assertEqual(count, 0)
        self.assertFalse(path.exists())

    def test_prunes_expired_records_on_startup(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            old_record = {
                "timestamp": datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat(),
                "identity": "alice",
            }
            path.write_text(json.dumps(old_record) + "\n", encoding="utf-8")

            RecognitionEventLogger(
                path,
                consent_confirmed=True,
                retention_days=30,
            )

            content = path.read_text(encoding="utf-8")

        self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()
