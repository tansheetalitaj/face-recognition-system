import unittest

from src.optical_flow import shift_detection
from src.tracking import Detection


class OpticalFlowTests(unittest.TestCase):
    def test_shift_preserves_metadata_and_moves_box(self) -> None:
        detection = Detection(
            box=(10, 110, 110, 10),
            name="alice [LIVE]",
            distance=0.3,
            confirmed_identity="alice",
            liveness_verified=True,
            track_id=7,
        )

        shifted = shift_detection(detection, dx=4.4, dy=-2.6)

        self.assertEqual(shifted.box, (7, 114, 107, 14))
        self.assertEqual(shifted.confirmed_identity, "alice")
        self.assertTrue(shifted.liveness_verified)


if __name__ == "__main__":
    unittest.main()
