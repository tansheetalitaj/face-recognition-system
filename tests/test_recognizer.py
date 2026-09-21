import unittest

from src.enrollment import KnownFaces
from src.recognizer import identify_face


class FakeDistanceBackend:
    def __init__(self, distances: list[float]) -> None:
        self.distances = distances

    def face_distance(self, known_encodings, candidate):
        return self.distances


class RecognizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.known_faces = KnownFaces(
            names=("alice", "bob"),
            encodings=("alice-encoding", "bob-encoding"),
        )

    def test_returns_closest_match_within_tolerance(self) -> None:
        result = identify_face(
            self.known_faces,
            "candidate",
            FakeDistanceBackend([0.58, 0.42]),
            tolerance=0.5,
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.name, "bob")
        self.assertEqual(result.distance, 0.42)

    def test_rejects_closest_face_outside_tolerance(self) -> None:
        result = identify_face(
            self.known_faces,
            "candidate",
            FakeDistanceBackend([0.61, 0.72]),
            tolerance=0.6,
        )

        self.assertFalse(result.matched)
        self.assertEqual(result.name, "Unknown")
        self.assertEqual(result.distance, 0.61)


if __name__ == "__main__":
    unittest.main()
