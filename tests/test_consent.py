from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.consent import (
    ConsentError,
    consent_problems,
    load_consent_records,
    record_consent,
    require_active_consent,
)


class ConsentTests(unittest.TestCase):
    def test_records_active_consent_with_retention(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "consent.json"

            record_consent(
                path,
                "alice",
                purpose="local research",
                retention_days=30,
            )
            records = load_consent_records(path)

        self.assertTrue(records["alice"].active)
        self.assertEqual(records["alice"].purpose, "local research")

    def test_missing_consent_is_reported_and_strict_mode_fails(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "consent.json"

            problems = consent_problems(path, ["alice"])
            with self.assertRaises(ConsentError):
                require_active_consent(path, ["alice"])

        self.assertEqual(problems, ["alice: no consent record"])

    def test_expired_retention_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "consent.json"
            expired = datetime.now(timezone.utc) - timedelta(days=1)
            payload = {
                "version": 1,
                "identities": {
                    "alice": {
                        "purpose": "test",
                        "recorded_at": (expired - timedelta(days=1)).isoformat(),
                        "retention_until": expired.isoformat(),
                    }
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            problems = consent_problems(path, ["alice"])

        self.assertEqual(problems, ["alice: retention expired"])


if __name__ == "__main__":
    unittest.main()
