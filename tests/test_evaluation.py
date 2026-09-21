import unittest

from src.evaluation import (
    EvaluationMetrics,
    EvaluationObservation,
    UNKNOWN_DATASET_FOLDER,
    metrics_at_threshold,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_counts_known_outcomes(self) -> None:
        metrics = EvaluationMetrics()

        metrics.record("alice", "alice")
        metrics.record("alice", "Unknown")
        metrics.record("alice", "bob")

        self.assertEqual(metrics.correct_known, 1)
        self.assertEqual(metrics.false_rejections, 1)
        self.assertEqual(metrics.misidentifications, 1)
        self.assertAlmostEqual(metrics.false_reject_rate or 0, 1 / 3)

    def test_counts_unknown_outcomes(self) -> None:
        metrics = EvaluationMetrics()

        metrics.record(UNKNOWN_DATASET_FOLDER, "Unknown")
        metrics.record(UNKNOWN_DATASET_FOLDER, "alice")

        self.assertEqual(metrics.correct_unknown, 1)
        self.assertEqual(metrics.false_accepts, 1)
        self.assertEqual(metrics.false_accept_rate, 0.5)
        self.assertEqual(metrics.identification_accuracy, 0.5)

    def test_threshold_sweep_recalculates_predictions_from_distances(self) -> None:
        observations = [
            EvaluationObservation("alice", "alice", 0.48),
            EvaluationObservation(UNKNOWN_DATASET_FOLDER, "alice", 0.58),
        ]

        strict = metrics_at_threshold(observations, 0.50)
        loose = metrics_at_threshold(observations, 0.60)

        self.assertEqual(strict.correct_known, 1)
        self.assertEqual(strict.correct_unknown, 1)
        self.assertEqual(loose.false_accepts, 1)


if __name__ == "__main__":
    unittest.main()
