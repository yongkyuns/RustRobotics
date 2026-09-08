"""Fast standard-library checks for the development evidence reader."""
import csv
import tempfile
import unittest
from pathlib import Path

from diagnose_ppo_balancing import CAP, classify, f32, read_tsv, summarize, vector


class DiagnosticReaderTests(unittest.TestCase):
    def test_boundary_precedence_matches_physical_limits(self):
        self.assertIsNone(classify([f32(2.4), 0, f32(0.6), 0], 1))
        self.assertEqual(classify([f32(2.4), 0, f32(0.6), 0], CAP), "Timeout")
        self.assertEqual(classify([3, 0, 0, 0], CAP), "Position")
        self.assertEqual(classify([0, 0, -0.7, 0], CAP), "Angle")
        self.assertEqual(classify([-3, 0, 0.7, 0], CAP), "Both")

    def test_vector_roundtrips_the_rust_f32_representation(self):
        self.assertEqual(vector("[0.6, 2.4, -0.01, 0.0]"),
                         [f32(0.6), f32(2.4), f32(-0.01), 0.0])

    def test_malformed_or_nonfinite_vectors_are_rejected(self):
        for text in ("[0, 1, 2]", "[0, 0, 0, True]", "[0, 0, 0, 1e999]", "(0,0,0,0)", "not_data"):
            with self.subTest(text=text):
                with self.assertRaises((AssertionError, SyntaxError, ValueError)):
                    vector(text)

    def test_tsv_rejects_missing_extra_and_empty_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tsv"
            for text in ("a\tb\n1\n", "a\tb\n1\t2\t3\n", "a\tb\n"):
                path.write_text(text)
                with self.assertRaises(AssertionError):
                    read_tsv(path)

    def test_missing_panel_is_not_a_complete_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.tsv"
            path.write_text("controller\ttraining_seed\tupdates\tepisode\nevil\t201\t0\t0\n")
            with self.assertRaisesRegex(AssertionError, "unexpected/duplicate episode"):
                summarize(Path(directory))

    def test_duplicate_and_missing_episodes_are_rejected(self):
        fields = ["controller", "training_seed", "updates", "episode", "evaluation_seed", "return", "discounted_return", "steps", "ending", "abs_force_sum", "near_limit_steps", "initial_value", "initial_observation", "final_state", "max_abs_state"]
        row = ["zero", 201, 0, 0, 0x33000000 + 201 * 1024, -10, -10, 1, "Angle", 0, 0, "none", "[0,0,0,0]", "[0,0,0.7,0]", "[0,0,0.7,0]"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.tsv"
            for copies, message in ((1, "missing episodes"), (2, "unexpected/duplicate episode")):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(fields)
                    writer.writerows([row] * copies)
                with self.assertRaisesRegex(AssertionError, message):
                    summarize(Path(directory))


if __name__ == "__main__":
    unittest.main()
