"""Inventory and narrowly scoped deletion helpers for biometric project data."""

from __future__ import annotations

import os
from pathlib import Path

from .enrollment import discover_images, identity_for_image
from .consent import remove_consent
from .data_protection import DataProtector
from .enrollment import ENCRYPTED_SUFFIX


class DataManagementError(RuntimeError):
    """Raised when requested biometric data cannot be safely managed."""


def enrollment_inventory(directory: Path) -> dict[str, tuple[Path, ...]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in discover_images(directory):
        identity = identity_for_image(image_path, directory)
        grouped.setdefault(identity, []).append(image_path)
    return {
        identity: tuple(sorted(paths))
        for identity, paths in sorted(grouped.items())
    }


def delete_identity(
    directory: Path,
    identity: str,
    cache_path: Path,
    consent_manifest_path: Path | None = None,
) -> int:
    inventory = enrollment_inventory(directory)
    paths = inventory.get(identity)
    if not paths:
        raise DataManagementError(f"Identity is not enrolled: {identity}")

    parent_directories: set[Path] = set()
    for image_path in paths:
        resolved_image = image_path.resolve()
        resolved_root = directory.resolve()
        if not resolved_image.is_relative_to(resolved_root):
            raise DataManagementError(f"Refusing to delete path outside enrollment: {image_path}")
        parent_directories.add(image_path.parent)
        image_path.unlink()

    for parent in sorted(parent_directories, key=lambda path: len(path.parts), reverse=True):
        current = parent
        while current != directory and current.is_relative_to(directory):
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    if cache_path.exists():
        cache_path.unlink()
    if consent_manifest_path is not None:
        remove_consent(consent_manifest_path, identity)
    return len(paths)


def purge_encoding_cache(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    if not cache_path.is_file():
        raise DataManagementError(f"Cache path is not a file: {cache_path}")
    cache_path.unlink()
    return True


def encrypt_identity_files(
    directory: Path,
    identity: str,
    cache_path: Path,
    protector: DataProtector,
) -> int:
    inventory = enrollment_inventory(directory)
    paths = inventory.get(identity)
    if not paths:
        raise DataManagementError(f"Identity is not enrolled: {identity}")
    plaintext_paths = [
        path for path in paths if not path.name.lower().endswith(ENCRYPTED_SUFFIX)
    ]
    if not plaintext_paths:
        return 0

    prepared: list[tuple[Path, Path]] = []
    for source in plaintext_paths:
        destination = source.with_name(source.name + ENCRYPTED_SUFFIX)
        if destination.exists():
            raise DataManagementError(f"Encrypted destination already exists: {destination}")
        plaintext = source.read_bytes()
        encrypted = protector.protect(plaintext)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(encrypted)
        if protector.unprotect(temporary.read_bytes()) != plaintext:
            temporary.unlink()
            raise DataManagementError(f"Encryption verification failed: {source}")
        prepared.append((source, temporary))

    for source, temporary in prepared:
        destination = source.with_name(source.name + ENCRYPTED_SUFFIX)
        os.replace(temporary, destination)
        source.unlink()
    if cache_path.exists():
        cache_path.unlink()
    return len(prepared)
