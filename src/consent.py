"""Per-identity consent and retention registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Iterable


CONSENT_VERSION = 1


class ConsentError(RuntimeError):
    """Raised when required consent is missing, invalid, or expired."""


@dataclass(frozen=True)
class ConsentRecord:
    identity: str
    purpose: str
    recorded_at: datetime
    retention_until: datetime

    @property
    def active(self) -> bool:
        return datetime.now(timezone.utc) < self.retention_until


def _read_payload(path: Path) -> dict:
    if not path.exists():
        return {"version": CONSENT_VERSION, "identities": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsentError(f"Consent manifest cannot be read: {path}") from exc
    if payload.get("version") != CONSENT_VERSION:
        raise ConsentError(f"Unsupported consent manifest version: {path}")
    if not isinstance(payload.get("identities"), dict):
        raise ConsentError(f"Consent manifest identities are invalid: {path}")
    return payload


def load_consent_records(path: Path) -> dict[str, ConsentRecord]:
    payload = _read_payload(path)
    records: dict[str, ConsentRecord] = {}
    for identity, raw in payload["identities"].items():
        try:
            records[identity] = ConsentRecord(
                identity=identity,
                purpose=str(raw["purpose"]),
                recorded_at=datetime.fromisoformat(raw["recorded_at"]),
                retention_until=datetime.fromisoformat(raw["retention_until"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConsentError(f"Invalid consent record for {identity!r}.") from exc
    return records


def record_consent(
    path: Path,
    identity: str,
    *,
    purpose: str,
    retention_days: int,
) -> ConsentRecord:
    if not identity.strip():
        raise ConsentError("Identity cannot be blank.")
    if not purpose.strip():
        raise ConsentError("Consent purpose cannot be blank.")
    if retention_days < 1:
        raise ConsentError("Retention period must be at least one day.")
    payload = _read_payload(path)
    now = datetime.now(timezone.utc)
    record = ConsentRecord(
        identity=identity,
        purpose=purpose.strip(),
        recorded_at=now,
        retention_until=now + timedelta(days=retention_days),
    )
    payload["identities"][identity] = {
        "purpose": record.purpose,
        "recorded_at": record.recorded_at.isoformat(),
        "retention_until": record.retention_until.isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return record


def remove_consent(path: Path, identity: str) -> bool:
    payload = _read_payload(path)
    if identity not in payload["identities"]:
        return False
    del payload["identities"][identity]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return True


def consent_problems(path: Path, identities: Iterable[str]) -> list[str]:
    records = load_consent_records(path)
    problems: list[str] = []
    for identity in sorted(set(identities)):
        record = records.get(identity)
        if record is None:
            problems.append(f"{identity}: no consent record")
        elif not record.active:
            problems.append(f"{identity}: retention expired")
    return problems


def require_active_consent(path: Path, identities: Iterable[str]) -> None:
    problems = consent_problems(path, identities)
    if problems:
        raise ConsentError("Consent validation failed:\n  - " + "\n  - ".join(problems))
