"""Evaluate recognition against labeled known and unknown photographs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import AppConfig, ConfigurationError
from src.data_protection import default_data_protector
from src.encoding_cache import load_cached_known_faces
from src.enrollment import EnrollmentError, discover_images
from src.evaluation import (
    EvaluationMetrics,
    EvaluationObservation,
    UNKNOWN_LABEL,
    metrics_at_threshold,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure recognition errors on a labeled image dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation"),
        help="Dataset containing identity folders and an optional _unknown folder",
    )
    parser.add_argument("--known-faces", type=Path, help="Enrollment directory")
    parser.add_argument("--tolerance", type=float, help="Recognition tolerance")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recalculate enrollment encodings before evaluation",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Compare tolerances from 0.40 through 0.65",
    )
    parser.add_argument(
        "--disable-cache-encryption",
        action="store_true",
        help="Use a plaintext encoding cache (development only)",
    )
    return parser


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def run_evaluation(
    dataset: Path,
    config: AppConfig,
    face_backend,
) -> tuple[EvaluationMetrics, list[EvaluationObservation]]:
    known_faces = load_cached_known_faces(
        config.known_faces_dir,
        face_backend,
        config.encoding_cache_path,
        rebuild=config.rebuild_cache,
        protector=(
            default_data_protector() if config.encrypt_encoding_cache else None
        ),
    )
    image_paths = discover_images(dataset)
    if not image_paths:
        raise EnrollmentError(f"No evaluation images found in {dataset}.")

    metrics = EvaluationMetrics()
    observations: list[EvaluationObservation] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(dataset)
        if len(relative_path.parts) < 2:
            logging.warning("Skipping %s: place it inside an identity folder.", relative_path)
            metrics.skipped += 1
            continue

        expected = relative_path.parts[0]
        try:
            image = face_backend.load_image_file(str(image_path))
            encodings = face_backend.face_encodings(image)
        except Exception as exc:
            logging.warning("Skipping %s: %s", relative_path, exc)
            metrics.skipped += 1
            continue
        if len(encodings) != 1:
            logging.warning(
                "Skipping %s: expected one face, detected %d.",
                relative_path,
                len(encodings),
            )
            metrics.skipped += 1
            continue

        distances = list(
            face_backend.face_distance(known_faces.encodings, encodings[0])
        )
        if not distances:
            logging.warning("Skipping %s: no enrollment encodings.", relative_path)
            metrics.skipped += 1
            continue
        best_index = min(range(len(distances)), key=distances.__getitem__)
        distance = float(distances[best_index])
        nearest_identity = known_faces.names[best_index]
        predicted = nearest_identity if distance <= config.tolerance else UNKNOWN_LABEL
        observation = EvaluationObservation(expected, nearest_identity, distance)
        observations.append(observation)
        metrics.record(expected, predicted)
        print(
            f"{relative_path.as_posix()}: expected={expected} "
            f"predicted={predicted} distance={distance:.3f}"
        )

    return metrics, observations


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        import face_recognition

        config = AppConfig.from_environment().with_overrides(
            known_faces_dir=args.known_faces,
            tolerance=args.tolerance,
            rebuild_cache=args.rebuild_cache,
            encrypt_encoding_cache=not args.disable_cache_encryption,
        )
        metrics, observations = run_evaluation(args.dataset, config, face_recognition)
    except ImportError:
        logging.error("Runtime dependencies are missing; use the project virtual environment.")
        return 1
    except (ConfigurationError, EnrollmentError) as exc:
        logging.error("%s", exc)
        return 2

    print("\nEvaluation summary")
    print(f"Known samples:          {metrics.known_samples}")
    print(f"Unknown samples:        {metrics.unknown_samples}")
    print(f"Correct known:          {metrics.correct_known}")
    print(f"Correct unknown:        {metrics.correct_unknown}")
    print(f"False rejections:       {metrics.false_rejections}")
    print(f"False accepts:          {metrics.false_accepts}")
    print(f"Misidentifications:     {metrics.misidentifications}")
    print(f"Skipped:                {metrics.skipped}")
    print(f"Identification accuracy: {_percentage(metrics.identification_accuracy)}")
    print(f"False-accept rate:       {_percentage(metrics.false_accept_rate)}")
    print(f"False-reject rate:       {_percentage(metrics.false_reject_rate)}")
    if args.sweep:
        print("\nTolerance sweep")
        print("Tolerance  Accuracy  False accept  False reject  Misidentified")
        for tolerance in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
            swept = metrics_at_threshold(observations, tolerance)
            print(
                f"{tolerance:>9.2f}  "
                f"{_percentage(swept.identification_accuracy):>8}  "
                f"{_percentage(swept.false_accept_rate):>12}  "
                f"{_percentage(swept.false_reject_rate):>12}  "
                f"{swept.misidentifications:>13}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
