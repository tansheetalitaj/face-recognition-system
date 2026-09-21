import unittest

from src.age_estimation import AgeEstimate


class AgeEstimateTests(unittest.TestCase):
    def test_reports_explicit_uncertainty(self) -> None:
        self.assertEqual(AgeEstimate("25-32", 0.20).uncertainty, "very uncertain")
        self.assertEqual(AgeEstimate("25-32", 0.45).uncertainty, "uncertain")
        self.assertEqual(AgeEstimate("25-32", 0.70).uncertainty, "moderate confidence")


if __name__ == "__main__":
    unittest.main()
