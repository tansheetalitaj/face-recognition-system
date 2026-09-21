from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data_management import (
    DataManagementError,
    delete_identity,
    encrypt_identity_files,
    enrollment_inventory,
    purge_encoding_cache,
)


class ReversingProtector:
    def protect(self, data: bytes) -> bytes:
        return b"protected:" + data[::-1]

    def unprotect(self, data: bytes) -> bytes:
        if not data.startswith(b"protected:"):
            raise ValueError("not protected")
        return data[len(b"protected:") :][::-1]


class DataManagementTests(unittest.TestCase):
    def test_inventory_groups_person_folder_samples(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            alice = root / "alice"
            alice.mkdir()
            (alice / "front.jpg").touch()
            (alice / "side.jpg").touch()
            (root / "bob.jpg").touch()

            inventory = enrollment_inventory(root)

        self.assertEqual(len(inventory["alice"]), 2)
        self.assertEqual(len(inventory["bob"]), 1)

    def test_delete_identity_removes_only_matching_images_and_cache(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            alice = enrollment / "alice"
            alice.mkdir(parents=True)
            (alice / "front.jpg").touch()
            (alice / "side.jpg").touch()
            bob = enrollment / "bob.jpg"
            bob.touch()
            cache = root / "cache.json"
            cache.write_text("{}", encoding="utf-8")

            removed = delete_identity(enrollment, "alice", cache)

            self.assertEqual(removed, 2)
            self.assertTrue(bob.exists())
            self.assertFalse(alice.exists())
            self.assertFalse(cache.exists())

    def test_delete_unknown_identity_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            root.joinpath("alice.jpg").touch()

            with self.assertRaises(DataManagementError):
                delete_identity(root, "nobody", root / "cache.json")

    def test_purge_cache_reports_whether_file_existed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache.json"
            cache.write_text("{}", encoding="utf-8")

            self.assertTrue(purge_encoding_cache(cache))
            self.assertFalse(purge_encoding_cache(cache))

    def test_encrypt_identity_verifies_then_removes_plaintext(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            enrollment.mkdir()
            source = enrollment / "alice.jpg"
            source.write_bytes(b"jpeg-bytes")
            cache = root / "cache.enc"
            cache.write_bytes(b"cache")

            count = encrypt_identity_files(
                enrollment,
                "alice",
                cache,
                ReversingProtector(),
            )
            encrypted = enrollment / "alice.jpg.dpapi"

            self.assertEqual(count, 1)
            self.assertFalse(source.exists())
            self.assertTrue(encrypted.exists())
            self.assertFalse(cache.exists())
            self.assertEqual(
                ReversingProtector().unprotect(encrypted.read_bytes()),
                b"jpeg-bytes",
            )


if __name__ == "__main__":
    unittest.main()
