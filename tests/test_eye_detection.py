import unittest

from src.eye_detection import eye_boxes_from_landmarks, select_eye_boxes


class EyeDetectionTests(unittest.TestCase):
    def test_landmarks_return_both_eyes_when_head_is_tilted(self) -> None:
        landmarks = {
            "left_eye": [(40, 40), (46, 43), (52, 47), (46, 49), (40, 46)],
            "right_eye": [(75, 57), (81, 60), (87, 64), (81, 66), (75, 63)],
        }

        selected = eye_boxes_from_landmarks(
            landmarks,
            frame_scale=0.5,
            face_box=(50, 200, 250, 20),
        )

        self.assertEqual(len(selected), 2)
        self.assertLess(selected[0][0], selected[1][0])

    def test_landmarks_handle_missing_eye_data(self) -> None:
        selected = eye_boxes_from_landmarks(
            {},
            frame_scale=0.5,
            face_box=(50, 200, 250, 20),
        )

        self.assertEqual(selected, ())

    def test_selects_outer_pair_when_glasses_bridge_is_false_positive(self) -> None:
        boxes = [
            (22, 24, 40, 38),
            (70, 25, 42, 38),
            (116, 26, 40, 38),
        ]

        selected = select_eye_boxes(boxes, face_width=180, face_height=180)

        self.assertEqual(selected, (boxes[0], boxes[2]))

    def test_never_returns_more_than_two_boxes(self) -> None:
        boxes = [
            (15, 20, 35, 35),
            (55, 21, 35, 35),
            (100, 20, 35, 35),
            (135, 22, 35, 35),
        ]

        selected = select_eye_boxes(boxes, face_width=180, face_height=180)

        self.assertLessEqual(len(selected), 2)

    def test_rejects_detection_in_lower_face(self) -> None:
        selected = select_eye_boxes(
            [(60, 125, 40, 30)],
            face_width=180,
            face_height=180,
        )

        self.assertEqual(selected, ())


if __name__ == "__main__":
    unittest.main()
