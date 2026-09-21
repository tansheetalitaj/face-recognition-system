import unittest

from src.tracking import Detection, FaceTracker, intersection_over_union


def detection(
    name: str,
    offset: int = 0,
    eye_ratio: float = 0.25,
    passive_live_score: float | None = None,
) -> Detection:
    return Detection(
        box=(10 + offset, 110 + offset, 110 + offset, 10 + offset),
        name=name,
        distance=0.3,
        eye_ratio=eye_ratio,
        passive_live_score=passive_live_score,
    )


class FaceTrackerTests(unittest.TestCase):
    def test_requires_consecutive_matches_before_showing_identity(self) -> None:
        tracker = FaceTracker(confirmations=3, require_liveness=False)

        first = tracker.update([detection("alice")])[0]
        second = tracker.update([detection("alice", 2)])[0]
        third = tracker.update([detection("alice", 4)])[0]

        self.assertEqual(first.name, "Verifying...")
        self.assertEqual(second.name, "Verifying...")
        self.assertEqual(third.name, "alice")

    def test_single_unknown_frame_does_not_drop_confirmed_identity(self) -> None:
        tracker = FaceTracker(
            confirmations=2,
            unknown_grace_updates=1,
            require_liveness=False,
        )
        tracker.update([detection("alice")])
        tracker.update([detection("alice")])

        brief_unknown = tracker.update([detection("Unknown")])[0]
        repeated_unknown = tracker.update([detection("Unknown")])[0]

        self.assertEqual(brief_unknown.name, "alice")
        self.assertEqual(repeated_unknown.name, "Unknown")

    def test_new_identity_must_be_confirmed(self) -> None:
        tracker = FaceTracker(confirmations=2, require_liveness=False)
        tracker.update([detection("alice")])
        self.assertEqual(tracker.update([detection("alice")])[0].name, "alice")

        changed = tracker.update([detection("bob")])[0]

        self.assertEqual(changed.name, "Verifying...")

    def test_requires_blink_after_identity_confirmation(self) -> None:
        tracker = FaceTracker(
            confirmations=1,
            require_liveness=True,
            blink_threshold=0.18,
        )

        first_open = tracker.update([detection("alice", eye_ratio=0.25)])[0]
        second_open = tracker.update([detection("alice", eye_ratio=0.25)])[0]
        closed = tracker.update([detection("alice", eye_ratio=0.10)])[0]
        reopened = tracker.update([detection("alice", eye_ratio=0.25)])[0]

        self.assertIn("Look at camera", first_open.name)
        self.assertIn("Blink now", second_open.name)
        self.assertIn("Open eyes", closed.name)
        self.assertEqual(reopened.name, "alice [LIVE]")
        self.assertTrue(reopened.liveness_verified)

    def test_predicts_motion_between_processed_frames(self) -> None:
        tracker = FaceTracker(confirmations=1, require_liveness=False)
        tracker.update([detection("alice", offset=0)])
        tracker.update([detection("alice", offset=10)])

        predicted = tracker.predict(0.5)[0]

        self.assertEqual(predicted.box, (25, 125, 125, 25))

    def test_passive_anti_spoof_requires_repeated_real_scores(self) -> None:
        tracker = FaceTracker(
            confirmations=1,
            require_liveness=False,
            require_passive=True,
            passive_threshold=0.70,
            passive_confirmations=2,
        )

        first = tracker.update(
            [detection("alice", passive_live_score=0.80)]
        )[0]
        second = tracker.update(
            [detection("alice", passive_live_score=0.82)]
        )[0]

        self.assertIn("PASSIVE CHECK", first.name)
        self.assertEqual(second.name, "alice [LIVE]")
        self.assertTrue(second.liveness_verified)

    def test_iou_handles_overlap_and_separation(self) -> None:
        self.assertGreater(
            intersection_over_union((0, 100, 100, 0), (10, 110, 110, 10)),
            0.5,
        )
        self.assertEqual(
            intersection_over_union((0, 10, 10, 0), (20, 30, 30, 20)),
            0,
        )


if __name__ == "__main__":
    unittest.main()
