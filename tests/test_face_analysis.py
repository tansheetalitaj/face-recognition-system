import unittest

from src.face_analysis import HeadPose, canonical_pitch, estimate_gaze


def landmarks() -> list[tuple[float, float, float]]:
    points = [(0.5, 0.5, 0.0) for _ in range(478)]
    # Image-horizontal eye corners and open lids.
    points[33] = (0.20, 0.40, 0.0)
    points[133] = (0.40, 0.40, 0.0)
    points[159] = (0.30, 0.38, 0.0)
    points[145] = (0.30, 0.42, 0.0)
    points[362] = (0.60, 0.40, 0.0)
    points[263] = (0.80, 0.40, 0.0)
    points[386] = (0.70, 0.38, 0.0)
    points[374] = (0.70, 0.42, 0.0)
    points[468] = (0.30, 0.40, 0.0)
    points[473] = (0.70, 0.40, 0.0)
    return points


class FaceAnalysisTests(unittest.TestCase):
    def test_center_gaze_uses_both_irises(self) -> None:
        result = estimate_gaze(landmarks())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.direction, "CENTER")
        self.assertGreater(result.confidence, 0.9)

    def test_screen_left_gaze_is_explicitly_image_relative(self) -> None:
        points = landmarks()
        points[468] = (0.24, 0.40, 0.0)
        points[473] = (0.64, 0.40, 0.0)

        result = estimate_gaze(points)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.direction, "SCREEN LEFT")

    def test_closed_eyes_do_not_produce_gaze(self) -> None:
        points = landmarks()
        points[145] = (0.30, 0.385, 0.0)
        points[374] = (0.70, 0.385, 0.0)

        self.assertIsNone(estimate_gaze(points))

    def test_head_pose_label_has_horizontal_and_vertical_parts(self) -> None:
        self.assertEqual(HeadPose(0, 0, 0).label, "CENTER/LEVEL")
        self.assertEqual(HeadPose(-20, -20, 0).label, "LEFT/UP")
        self.assertEqual(HeadPose(20, 20, 0).label, "RIGHT/DOWN")

    def test_equivalent_upside_down_pitch_is_canonicalized(self) -> None:
        self.assertAlmostEqual(canonical_pitch(-178.0), 2.0)
        self.assertAlmostEqual(canonical_pitch(179.0), -1.0)


if __name__ == "__main__":
    unittest.main()
