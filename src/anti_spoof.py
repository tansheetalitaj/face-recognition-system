"""Verified ONNX passive face anti-spoofing inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


MODEL_SHA384 = (
    "6de4534964b723397b3e8c995cadcf43bc007cc2f9930b95a"
    "e25f76adccece5d1d4d058d0b15117b9e4a9f758424f92a"
)
MODEL_SIZE = 12_270_179


class AntiSpoofModelError(RuntimeError):
    """Raised when the passive anti-spoof model is missing or invalid."""


def sha384_file(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected_sha384: str = MODEL_SHA384) -> None:
    if not path.is_file():
        raise AntiSpoofModelError(
            f"Passive anti-spoof model is missing: {path}. Run python setup_models.py."
        )
    if path.stat().st_size != MODEL_SIZE:
        raise AntiSpoofModelError(f"Passive anti-spoof model has an unexpected size: {path}")
    actual = sha384_file(path)
    if actual.lower() != expected_sha384.lower():
        raise AntiSpoofModelError(
            f"Passive anti-spoof model checksum mismatch: {path}"
        )


@dataclass(frozen=True)
class AntiSpoofResult:
    real_score: float
    spoof_score: float


class PassiveAntiSpoofDetector:
    def __init__(self, model_path: Path, real_threshold: float = 0.70) -> None:
        verify_model(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise AntiSpoofModelError(
                "onnxruntime is missing; install the project requirements."
            ) from exc
        self.real_threshold = real_threshold
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: Any, box: tuple[int, int, int, int], cv2: Any) -> AntiSpoofResult:
        import numpy as np

        top, right, bottom, left = box
        frame_height, frame_width = frame.shape[:2]
        face_width = right - left
        face_height = bottom - top
        padding_x = round(face_width * 0.10)
        padding_y = round(face_height * 0.10)
        crop_left = max(0, left - padding_x)
        crop_right = min(frame_width, right + padding_x)
        crop_top = max(0, top - padding_y)
        crop_bottom = min(frame_height, bottom + padding_y)
        crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        if crop.size == 0:
            raise AntiSpoofModelError("Cannot run anti-spoofing on an empty face crop.")

        rgb = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32)
        mean = np.asarray([151.2405, 119.5950, 107.8395], dtype=np.float32)
        scale = np.asarray([63.0105, 56.4570, 55.0035], dtype=np.float32)
        tensor = ((tensor - mean) / scale).transpose(2, 0, 1)[None, ...]
        output = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).reshape(-1)
        if output.size != 2:
            raise AntiSpoofModelError(
                f"Unexpected anti-spoof output shape: {output.shape}"
            )
        if np.any(output < 0) or not np.isclose(float(output.sum()), 1.0, atol=0.05):
            shifted = output - output.max()
            output = np.exp(shifted) / np.exp(shifted).sum()
        return AntiSpoofResult(real_score=float(output[0]), spoof_score=float(output[1]))
