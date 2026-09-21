from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.enrollment import EnrollmentError, discover_images, load_known_faces


class FakeFaceBackend:
    def __init__(self, encodings_by_name: dict[str, list[object]]) -> None:
        self.encodings_by_name = encodings_by_name

    def load_image_file(self, file: str) -> str:
        return Path(file).name

    def face_encodings(self, image: str) -> list[object]:
        return self.encodings_by_name[image]


class EnrollmentTests(unittest.TestCase):
    def test_discovers_supported_extensions_case_insensitively(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for name in ("one.JPG", "two.jpeg", "three.png", "four.jpg.dpapi", "notes.txt"):
                (directory / name).touch()

            names = [path.name for path in discover_images(directory)]

        self.assertEqual(names, ["four.jpg.dpapi", "one.JPG", "three.png", "two.jpeg"])

    def test_loads_one_face_per_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "alice.jpg").touch()
            backend = FakeFaceBackend({"alice.jpg": ["alice-encoding"]})

            known_faces = load_known_faces(directory, backend)

        self.assertEqual(known_faces.names, ("alice",))
        self.assertEqual(known_faces.encodings, ("alice-encoding",))

    def test_person_folder_supports_multiple_samples_for_one_identity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            person_directory = directory / "alice"
            person_directory.mkdir()
            (person_directory / "front.jpg").touch()
            (person_directory / "side.jpg").touch()
            backend = FakeFaceBackend(
                {"front.jpg": ["front-encoding"], "side.jpg": ["side-encoding"]}
            )

            known_faces = load_known_faces(directory, backend)

        self.assertEqual(known_faces.names, ("alice", "alice"))

    def test_rejects_images_without_exactly_one_face(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "empty.jpg").touch()
            (directory / "group.jpg").touch()
            backend = FakeFaceBackend(
                {"empty.jpg": [], "group.jpg": ["first", "second"]}
            )

            with self.assertRaises(EnrollmentError) as context:
                load_known_faces(directory, backend)

        message = str(context.exception)
        self.assertIn("no face detected", message)
        self.assertIn("expected exactly one", message)


if __name__ == "__main__":
    unittest.main()
