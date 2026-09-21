"""Command-line entry point for the face recognition application."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.camera import run_camera
from src.config import AppConfig, ConfigurationError
from src.enrollment import EnrollmentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize enrolled faces from a webcam and mark detected eyes."
    )
    parser.add_argument("--camera", type=int, help="Camera index (default: 0)")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Maximum face distance accepted as a match (default: 0.6)",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        help="Recognition frame scale in the range (0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        help="Run recognition every N frames (default: 2)",
    )
    parser.add_argument(
        "--known-faces",
        type=Path,
        help="Directory containing enrollment images",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        help="Consecutive matches required before showing an identity (default: 3)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recalculate every enrollment encoding",
    )
    parser.add_argument(
        "--disable-cache-encryption",
        action="store_true",
        help="Write embedding cache as plaintext (development only)",
    )
    parser.add_argument(
        "--require-consent-manifest",
        action="store_true",
        help="Refuse recognition for missing or expired consent records",
    )
    parser.add_argument(
        "--blink-threshold",
        type=float,
        help="Eye aspect ratio below which eyes count as closed (default: 0.18)",
    )
    parser.add_argument(
        "--disable-liveness",
        action="store_true",
        help="Disable the blink challenge (not recommended for sensitive uses)",
    )
    parser.add_argument(
        "--passive-threshold",
        type=float,
        help="Minimum passive anti-spoof real score (default: 0.70)",
    )
    passive_group = parser.add_mutually_exclusive_group()
    passive_group.add_argument(
        "--enable-passive-anti-spoof",
        action="store_true",
        help="Enable the experimental ONNX passive anti-spoof model",
    )
    passive_group.add_argument(
        "--disable-passive-anti-spoof",
        action="store_true",
        help="Keep passive anti-spoofing disabled (deprecated compatibility flag)",
    )
    parser.add_argument(
        "--enable-age-estimation",
        action="store_true",
        help="Enable optional coarse age-band estimation (model files required)",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="Append consented recognition metadata to this JSONL file",
    )
    parser.add_argument(
        "--consent-to-event-logging",
        action="store_true",
        help="Confirm that affected people consented to recognition event logging",
    )
    parser.add_argument(
        "--event-retention-days",
        type=int,
        help="Delete event records older than this many days (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        config = AppConfig.from_environment().with_overrides(
            camera_index=args.camera,
            tolerance=args.tolerance,
            frame_scale=args.frame_scale,
            process_every_n_frames=args.process_every,
            confirmation_frames=args.confirmations,
            require_liveness=False if args.disable_liveness else None,
            blink_threshold=args.blink_threshold,
            require_passive_anti_spoof=(
                True
                if args.enable_passive_anti_spoof
                else False
                if args.disable_passive_anti_spoof
                else None
            ),
            passive_anti_spoof_threshold=args.passive_threshold,
            enable_age_estimation=args.enable_age_estimation,
            known_faces_dir=args.known_faces,
            rebuild_cache=args.rebuild_cache,
            encrypt_encoding_cache=not args.disable_cache_encryption,
            require_consent_manifest=args.require_consent_manifest,
            event_log_path=args.event_log,
            event_logging_consent=args.consent_to_event_logging,
            event_retention_days=args.event_retention_days,
        )
        run_camera(config)
    except (ConfigurationError, EnrollmentError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        return 130
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
