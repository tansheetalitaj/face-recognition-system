"""Safe, versioned JSON cache for enrolled face embeddings."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

from .enrollment import (
    EnrollmentError,
    KnownFaces,
    discover_images,
    identity_for_image,
    ENCRYPTED_SUFFIX,
)
from .data_protection import DataProtectionError, DataProtector


CACHE_VERSION = 1
LOGGER = logging.getLogger(__name__)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cache(
    cache_path: Path, protector: DataProtector | None
) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        raw = cache_path.read_bytes()
        if protector is not None:
            raw = protector.unprotect(raw)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DataProtectionError):
        LOGGER.warning("Ignoring unreadable encoding cache: %s", cache_path)
        return {}
    if payload.get("version") != CACHE_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _write_cache(
    cache_path: Path,
    entries: dict[str, dict[str, Any]],
    protector: DataProtector | None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {"version": CACHE_VERSION, "entries": entries}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if protector is not None:
        raw = protector.protect(raw)
    temporary_path.write_bytes(raw)
    os.replace(temporary_path, cache_path)


def load_cached_known_faces(
    directory: Path,
    backend: Any,
    cache_path: Path,
    *,
    rebuild: bool = False,
    protector: DataProtector | None = None,
) -> KnownFaces:
    image_paths = discover_images(directory)
    if not image_paths:
        raise EnrollmentError(f"No enrollment images found in {directory}.")

    cached_entries = {} if rebuild else _read_cache(cache_path, protector)
    current_entries: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    encodings: list[Any] = []
    problems: list[str] = []
    cache_hits = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(directory).as_posix()
        identity = identity_for_image(image_path, directory)
        digest = _file_digest(image_path)
        cached = cached_entries.get(relative_path)

        if (
            cached
            and cached.get("sha256") == digest
            and cached.get("identity") == identity
            and isinstance(cached.get("encoding"), list)
            and len(cached["encoding"]) == 128
        ):
            import numpy as np

            encoding = np.asarray(cached["encoding"], dtype=float)
            cache_hits += 1
        else:
            try:
                if image_path.name.lower().endswith(ENCRYPTED_SUFFIX):
                    if protector is None:
                        raise DataProtectionError(
                            "Encrypted enrollment requires cache encryption protection."
                        )
                    decrypted = protector.unprotect(image_path.read_bytes())
                    image = backend.load_image_file(io.BytesIO(decrypted))
                else:
                    image = backend.load_image_file(str(image_path))
                detected_encodings = backend.face_encodings(image)
            except Exception as exc:
                problems.append(f"{relative_path}: could not be read ({exc})")
                continue

            if len(detected_encodings) == 0:
                problems.append(f"{relative_path}: no face detected")
                continue
            if len(detected_encodings) > 1:
                problems.append(
                    f"{relative_path}: {len(detected_encodings)} faces detected; expected exactly one"
                )
                continue
            encoding = detected_encodings[0]

        serializable_encoding = [float(value) for value in encoding]
        if len(serializable_encoding) != 128:
            problems.append(
                f"{relative_path}: expected a 128-value face encoding, "
                f"received {len(serializable_encoding)}"
            )
            continue
        current_entries[relative_path] = {
            "identity": identity,
            "sha256": digest,
            "encoding": serializable_encoding,
        }
        names.append(identity)
        encodings.append(encoding)

    if problems:
        details = "\n  - ".join(problems)
        raise EnrollmentError(f"Invalid enrollment images:\n  - {details}")

    if rebuild or current_entries != cached_entries:
        _write_cache(cache_path, current_entries, protector)

    LOGGER.info(
        "Enrollment cache: %d hit(s), %d generated encoding(s)",
        cache_hits,
        len(encodings) - cache_hits,
    )
    return KnownFaces(tuple(names), tuple(encodings))
