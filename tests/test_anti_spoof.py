from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.anti_spoof import AntiSpoofModelError, sha384_file, verify_model


class AntiSpoofModelTests(unittest.TestCase):
    def test_sha384_file_is_deterministic(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.onnx"
            path.write_bytes(b"test-model")

            first = sha384_file(path)
            second = sha384_file(path)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)

    def test_missing_model_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.onnx"

            with self.assertRaises(AntiSpoofModelError):
                verify_model(missing)

    def test_wrong_size_is_rejected_before_inference(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.onnx"
            path.write_bytes(b"not-the-approved-model")

            with self.assertRaises(AntiSpoofModelError):
                verify_model(path)


if __name__ == "__main__":
    unittest.main()
