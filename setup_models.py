"""Download approved model artifacts and verify their published checksums."""

from __future__ import annotations

import logging
import os
import argparse
import hashlib
from pathlib import Path
import urllib.request

from src.anti_spoof import MODEL_SHA384, MODEL_SIZE, verify_model
from src.config import PROJECT_ROOT


MODEL_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "public/2022.1/anti-spoof-mn3/anti-spoof-mn3.onnx"
)
MODEL_PATH = PROJECT_ROOT / "models" / "anti-spoof-mn3.onnx"

AGE_DEFINITION_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/"
    "master/AgeGender/age_deploy.prototxt"
)
AGE_WEIGHTS_URL = (
    "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1"
)
AGE_DEFINITION_PATH = PROJECT_ROOT / "models" / "age_deploy.prototxt"
AGE_WEIGHTS_PATH = PROJECT_ROOT / "models" / "age_net.caffemodel"
AGE_DEFINITION_SIZE = 2_308
AGE_WEIGHTS_SIZE = 45_661_480
AGE_DEFINITION_SHA384 = (
    "09268b6b0bdfb78f3dcdb09d7de5a66e6b907ecbdc4aaeede36c228a123b7026"
    "9a09263143d11ac399f06b546ac8c693"
)
AGE_WEIGHTS_SHA384 = (
    "81327d78e6aa7a6f818f46f7b058a23eb1df379b089673cab2be1b91c65e88bc"
    "6daed9f91a830713f097f04753492ebf"
)


def download_model(destination: Path = MODEL_PATH) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_model(destination)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        verify_model(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _verify_artifact(path: Path, size: int, sha384: str) -> None:
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError(f"Model artifact has an unexpected size: {path}")
    digest = hashlib.sha384(path.read_bytes()).hexdigest()
    if digest.lower() != sha384.lower():
        raise RuntimeError(f"Model artifact checksum mismatch: {path}")


def _download_artifact(url: str, path: Path, size: int, sha384: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _verify_artifact(path, size, sha384)
        return path
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        _verify_artifact(temporary, size, sha384)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def download_age_models() -> tuple[Path, Path]:
    definition = _download_artifact(
        AGE_DEFINITION_URL,
        AGE_DEFINITION_PATH,
        AGE_DEFINITION_SIZE,
        AGE_DEFINITION_SHA384,
    )
    weights = _download_artifact(
        AGE_WEIGHTS_URL,
        AGE_WEIGHTS_PATH,
        AGE_WEIGHTS_SIZE,
        AGE_WEIGHTS_SHA384,
    )
    return definition, weights


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-age",
        action="store_true",
        help="Also download the optional age-only, eight-band model",
    )
    args = parser.parse_args(argv)
    path = download_model()
    logging.info("Verified model: %s", path)
    logging.info("Size: %d bytes", MODEL_SIZE)
    logging.info("SHA-384: %s", MODEL_SHA384)
    if args.include_age:
        definition, weights = download_age_models()
        logging.info("Verified age definition: %s", definition)
        logging.info("Verified age weights: %s", weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
