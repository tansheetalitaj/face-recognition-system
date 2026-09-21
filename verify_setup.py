"""Verify runtime versions, dependencies, and approved model integrity."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from src.anti_spoof import verify_model
from src.config import PROJECT_ROOT
from setup_models import (
    AGE_DEFINITION_PATH,
    AGE_DEFINITION_SHA384,
    AGE_DEFINITION_SIZE,
    AGE_WEIGHTS_PATH,
    AGE_WEIGHTS_SHA384,
    AGE_WEIGHTS_SIZE,
    _verify_artifact,
)


def locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, expected = stripped.split("==", 1)
        versions[name] = expected
    return versions


def main() -> int:
    problems: list[str] = []
    if sys.version_info[:2] != (3, 10):
        problems.append(f"Expected Python 3.10, found {sys.version.split()[0]}")
    for package, expected in locked_versions(PROJECT_ROOT / "requirements-lock.txt").items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            problems.append(f"Missing package: {package}=={expected}")
            continue
        if actual != expected:
            problems.append(f"{package}: expected {expected}, found {actual}")
    try:
        verify_model(PROJECT_ROOT / "models" / "anti-spoof-mn3.onnx")
    except RuntimeError as exc:
        problems.append(str(exc))
    age_paths = (AGE_DEFINITION_PATH, AGE_WEIGHTS_PATH)
    if any(path.exists() for path in age_paths):
        if not all(path.exists() for path in age_paths):
            problems.append("Age model installation is incomplete.")
        else:
            try:
                _verify_artifact(
                    AGE_DEFINITION_PATH, AGE_DEFINITION_SIZE, AGE_DEFINITION_SHA384
                )
                _verify_artifact(AGE_WEIGHTS_PATH, AGE_WEIGHTS_SIZE, AGE_WEIGHTS_SHA384)
            except RuntimeError as exc:
                problems.append(str(exc))
    if problems:
        print("Setup verification failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Setup verification passed.")
    print("Python, locked packages, and installed model artifacts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
