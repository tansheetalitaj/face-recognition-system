import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.encoding_cache import CACHE_VERSION, load_cached_known_faces


class FakeEncodingBackend:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_image_file(self, file: str) -> str:
        self.loaded.append(file)
        return Path(file).name

    def face_encodings(self, image: str) -> list[list[float]]:
        value = float(sum(image.encode("utf-8")) % 100) / 100
        return [[value + index / 1000 for index in range(128)]]


class ReversingProtector:
    def protect(self, data: bytes) -> bytes:
        return b"protected:" + data[::-1]

    def unprotect(self, data: bytes) -> bytes:
        return data[len(b"protected:") :][::-1]


class BytesEncodingBackend:
    def load_image_file(self, file) -> bytes:
        return file.read()

    def face_encodings(self, image: bytes) -> list[list[float]]:
        if image != b"jpeg-bytes":
            return []
        return [[index / 1000 for index in range(128)]]


class EncodingCacheTests(unittest.TestCase):
    def test_reuses_unchanged_encodings_and_supports_person_folders(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            alice = enrollment / "alice"
            alice.mkdir(parents=True)
            (alice / "front.jpg").write_bytes(b"front")
            (alice / "side.png").write_bytes(b"side")
            cache_path = root / ".cache" / "encodings.json"
            first_backend = FakeEncodingBackend()

            first = load_cached_known_faces(
                enrollment, first_backend, cache_path
            )
            second_backend = FakeEncodingBackend()
            second = load_cached_known_faces(
                enrollment, second_backend, cache_path
            )

            payload = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(first.names, ("alice", "alice"))
        self.assertEqual(second.names, ("alice", "alice"))
        self.assertEqual(len(first_backend.loaded), 2)
        self.assertEqual(second_backend.loaded, [])
        self.assertEqual(payload["version"], CACHE_VERSION)

    def test_changed_image_invalidates_only_its_cache_entry(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            enrollment.mkdir()
            first_image = enrollment / "alice.jpg"
            second_image = enrollment / "bob.jpg"
            first_image.write_bytes(b"alice-one")
            second_image.write_bytes(b"bob-one")
            cache_path = root / "cache.json"
            load_cached_known_faces(
                enrollment, FakeEncodingBackend(), cache_path
            )
            first_image.write_bytes(b"alice-two")
            backend = FakeEncodingBackend()

            load_cached_known_faces(enrollment, backend, cache_path)

        self.assertEqual([Path(path).name for path in backend.loaded], ["alice.jpg"])

    def test_rebuild_ignores_existing_cache(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            enrollment.mkdir()
            (enrollment / "alice.jpg").write_bytes(b"alice")
            cache_path = root / "cache.json"
            load_cached_known_faces(
                enrollment, FakeEncodingBackend(), cache_path
            )
            backend = FakeEncodingBackend()

            load_cached_known_faces(
                enrollment, backend, cache_path, rebuild=True
            )

        self.assertEqual(len(backend.loaded), 1)

    def test_loads_dpapi_encrypted_enrollment_in_memory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enrollment = root / "known_faces"
            enrollment.mkdir()
            protector = ReversingProtector()
            (enrollment / "alice.jpg.dpapi").write_bytes(
                protector.protect(b"jpeg-bytes")
            )

            faces = load_cached_known_faces(
                enrollment,
                BytesEncodingBackend(),
                root / "cache.enc",
                protector=protector,
            )

        self.assertEqual(faces.names, ("alice",))
        self.assertEqual(len(faces.encodings[0]), 128)


if __name__ == "__main__":
    unittest.main()
