"""Discover and validate face enrollment images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ENCRYPTED_SUFFIX = ".dpapi"


def is_supported_image_path(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or any(
        name.endswith(extension + ENCRYPTED_SUFFIX)
        for extension in SUPPORTED_IMAGE_EXTENSIONS
    )


class FaceBackend(Protocol):
    def load_image_file(self, file: str) -> Any: ...

    def face_encodings(self, image: Any) -> Sequence[Any]: ...


class EnrollmentError(RuntimeError):
    """Raised when enrolled face data cannot be loaded safely."""


@dataclass(frozen=True)
class KnownFaces:
    names: tuple[str, ...]
    encodings: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.names)


def discover_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise EnrollmentError(f"Enrollment directory does not exist: {directory}")
    if not directory.is_dir():
        raise EnrollmentError(f"Enrollment path is not a directory: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and is_supported_image_path(path)
    )


def identity_for_image(image_path: Path, enrollment_directory: Path) -> str:
    """Derive identity from a person folder, or the filename for legacy images."""
    relative_path = image_path.relative_to(enrollment_directory)
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    filename = image_path.name
    if filename.lower().endswith(ENCRYPTED_SUFFIX):
        filename = filename[: -len(ENCRYPTED_SUFFIX)]
    return Path(filename).stem


def load_known_faces(directory: Path, backend: FaceBackend) -> KnownFaces:
    image_paths = discover_images(directory)
    if not image_paths:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise EnrollmentError(
            f"No enrollment images found in {directory}. Supported formats: {supported}."
        )

    names: list[str] = []
    encodings: list[Any] = []
    problems: list[str] = []

    for image_path in image_paths:
        try:
            image = backend.load_image_file(str(image_path))
            detected_encodings = backend.face_encodings(image)
        except Exception as exc:
            problems.append(f"{image_path.name}: could not be read ({exc})")
            continue

        if len(detected_encodings) == 0:
            problems.append(f"{image_path.name}: no face detected")
            continue
        if len(detected_encodings) > 1:
            problems.append(
                f"{image_path.name}: {len(detected_encodings)} faces detected; expected exactly one"
            )
            continue

        names.append(identity_for_image(image_path, directory))
        encodings.append(detected_encodings[0])

    if problems:
        details = "\n  - ".join(problems)
        raise EnrollmentError(f"Invalid enrollment images:\n  - {details}")

    return KnownFaces(tuple(names), tuple(encodings))
