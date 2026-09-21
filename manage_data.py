"""Inspect and explicitly delete local biometric project data."""

from __future__ import annotations

import argparse
import logging

from src.config import AppConfig, ConfigurationError
from src.consent import ConsentError, load_consent_records, record_consent
from src.data_protection import DataProtectionError, default_data_protector
from src.data_management import (
    DataManagementError,
    delete_identity,
    encrypt_identity_files,
    enrollment_inventory,
    purge_encoding_cache,
)
from src.enrollment import EnrollmentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local enrollment data safely.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="List identities and local data locations")

    delete_parser = subparsers.add_parser(
        "delete-identity", help="Delete every enrollment image for one identity"
    )
    delete_parser.add_argument("identity")
    delete_parser.add_argument(
        "--confirm",
        required=True,
        help="Type the identity exactly to authorize deletion",
    )

    purge_parser = subparsers.add_parser(
        "purge-cache", help="Delete the generated face-encoding cache"
    )
    purge_parser.add_argument(
        "--confirm",
        choices=["PURGE"],
        required=True,
        help="Pass --confirm PURGE to authorize cache deletion",
    )
    consent_parser = subparsers.add_parser(
        "record-consent", help="Record purpose and retention for an enrolled identity"
    )
    consent_parser.add_argument("identity")
    consent_parser.add_argument("--purpose", required=True)
    consent_parser.add_argument("--retention-days", type=int, required=True)
    consent_parser.add_argument(
        "--confirm",
        required=True,
        help="Type the identity exactly to confirm the consent record",
    )
    encrypt_parser = subparsers.add_parser(
        "encrypt-identity",
        help="Replace one identity's plaintext images with user-bound DPAPI files",
    )
    encrypt_parser.add_argument("identity")
    encrypt_parser.add_argument(
        "--confirm",
        required=True,
        help="Type the identity exactly to authorize replacement",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.from_environment()
        if args.command == "status":
            inventory = enrollment_inventory(config.known_faces_dir)
            print(f"Enrollment directory: {config.known_faces_dir}")
            for identity, paths in inventory.items():
                print(f"- {identity}: {len(paths)} image(s)")
            print(
                f"Encoding cache: {config.encoding_cache_path} "
                f"({'present' if config.encoding_cache_path.exists() else 'absent'})"
            )
            records = load_consent_records(config.consent_manifest_path)
            for identity in inventory:
                record = records.get(identity)
                status = "missing"
                if record is not None:
                    status = "active" if record.active else "expired"
                print(f"  consent {identity}: {status}")
            return 0

        if args.command == "record-consent":
            if args.confirm != args.identity:
                logging.error("Confirmation must exactly match the identity.")
                return 2
            if args.identity not in enrollment_inventory(config.known_faces_dir):
                logging.error("Identity is not enrolled: %s", args.identity)
                return 2
            record = record_consent(
                config.consent_manifest_path,
                args.identity,
                purpose=args.purpose,
                retention_days=args.retention_days,
            )
            print(
                f"Consent recorded for {record.identity}; retention until "
                f"{record.retention_until.isoformat()}."
            )
            return 0

        if args.command == "encrypt-identity":
            if args.confirm != args.identity:
                logging.error("Confirmation must exactly match the identity.")
                return 2
            encrypted = encrypt_identity_files(
                config.known_faces_dir,
                args.identity,
                config.encoding_cache_path,
                default_data_protector(),
            )
            print(f"Encrypted {encrypted} enrollment image(s) for {args.identity}.")
            print("Plaintext originals were removed after decryption verification.")
            return 0

        if args.command == "delete-identity":
            if args.confirm != args.identity:
                logging.error("Confirmation must exactly match the identity.")
                return 2
            removed = delete_identity(
                config.known_faces_dir,
                args.identity,
                config.encoding_cache_path,
                config.consent_manifest_path,
            )
            print(f"Deleted {removed} enrollment image(s) for {args.identity}.")
            print("The encoding cache was removed and will rebuild on next run.")
            return 0

        removed = purge_encoding_cache(config.encoding_cache_path)
        print("Encoding cache deleted." if removed else "Encoding cache was already absent.")
        return 0
    except (
        ConfigurationError,
        ConsentError,
        DataManagementError,
        DataProtectionError,
        EnrollmentError,
    ) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
