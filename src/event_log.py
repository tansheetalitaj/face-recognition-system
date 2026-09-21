"""Consent-gated, metadata-only recognition event logging."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Callable, Iterable

from .tracking import Detection


class ConsentRequiredError(ValueError):
    """Raised when event logging is requested without explicit consent."""


class RecognitionEventLogger:
    def __init__(
        self,
        path: Path,
        *,
        consent_confirmed: bool,
        require_liveness: bool = True,
        cooldown_seconds: float = 30,
        retention_days: int = 30,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not consent_confirmed:
            raise ConsentRequiredError(
                "Event logging requires --consent-to-event-logging."
            )
        self.path = path
        self.require_liveness = require_liveness
        self.cooldown_seconds = cooldown_seconds
        self.retention_days = retention_days
        self.monotonic_clock = monotonic_clock
        self._last_logged: dict[str, float] = {}
        self._prune_expired()

    def _prune_expired(self) -> None:
        if not self.path.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        retained: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                timestamp = datetime.fromisoformat(payload["timestamp"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                retained.append(line)
                continue
            if timestamp >= cutoff:
                retained.append(line)
        content = "".join(f"{line}\n" for line in retained)
        self.path.write_text(content, encoding="utf-8")

    def record(self, detections: Iterable[Detection]) -> int:
        now = self.monotonic_clock()
        events: list[dict[str, object]] = []
        for detection in detections:
            identity = detection.confirmed_identity
            if identity is None:
                continue
            if self.require_liveness and not detection.liveness_verified:
                continue
            last_logged = self._last_logged.get(identity)
            if last_logged is not None and now - last_logged < self.cooldown_seconds:
                continue
            events.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "identity": identity,
                    "distance": detection.distance,
                    "liveness_verified": detection.liveness_verified,
                    "track_id": detection.track_id,
                }
            )
            self._last_logged[identity] = now

        if not events:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as log_file:
            for event in events:
                log_file.write(json.dumps(event, separators=(",", ":")) + "\n")
        return len(events)
