import unittest

from src.liveness import BlinkChallenge, BlinkPhase, eye_aspect_ratio


class LivenessTests(unittest.TestCase):
    def test_eye_aspect_ratio_uses_six_eye_points(self) -> None:
        points = [(0, 0), (2, -2), (8, -2), (10, 0), (8, 2), (2, 2)]

        ratio = eye_aspect_ratio(points)

        self.assertAlmostEqual(ratio or 0, 0.4)

    def test_blink_requires_open_closed_open_sequence(self) -> None:
        challenge = BlinkChallenge(
            closed_threshold=0.18,
            required_open_updates=2,
            required_closed_updates=1,
        )

        self.assertEqual(challenge.update(0.25), BlinkPhase.FIND_OPEN_EYES)
        self.assertEqual(challenge.update(0.25), BlinkPhase.WAIT_FOR_BLINK)
        self.assertEqual(challenge.update(0.10), BlinkPhase.WAIT_FOR_REOPEN)
        self.assertEqual(challenge.update(0.24), BlinkPhase.VERIFIED)
        self.assertTrue(challenge.verified)

    def test_closed_eyes_first_do_not_complete_challenge(self) -> None:
        challenge = BlinkChallenge(closed_threshold=0.18)

        challenge.update(0.10)
        challenge.update(0.10)
        challenge.update(0.25)

        self.assertFalse(challenge.verified)


if __name__ == "__main__":
    unittest.main()
