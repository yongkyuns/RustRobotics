"""The evidence validator must fail closed, not trust a green test marker."""
import itertools
import math
import struct
import unittest

from qualify_ppo_learning import PROTOCOL, SEEDS, TEST, summary, validate_log


def fixture(gains=None):
    gains = [100.0] * 12 if gains is None else gains
    lines = ["running 1 test", f"test {TEST} ...", PROTOCOL, "PPO_CONFIG\tPpoTrainerConfig { synthetic validator fixture }"]
    for seed, gain in zip(SEEDS, gains):
        for phase, score in [("initial", 100.0), ("final", 100.0 + gain)]:
            values = [20.0, 2.0] + [0.0] * 4545
            values[-1] = 0.001 if phase == "final" else 0.0
            raw = struct.pack(">4547f", *values).hex()
            lines.append(f"PPO_POLICY\t{seed}\t{phase}\t{raw}")
            for i in range(32):
                eval_seed = 1_000_000 + (seed - 101) * 1000 + i
                lines.append(f"PPO_EPISODE\t{seed}\t{phase}\t{eval_seed}\t{score:.17f}\t1000\t0\t1")
        lines.append(f"PPO_TRIAL\t{seed}\t128\t65536\t100.00000000000000000\t{100.0 + gain:.17f}\t{gain:.17f}")
    s = summary(gains)
    lines.append("PPO_SUMMARY\t{}\t{:.17f}\t{:.17f}\t{:.17f}\t{:.17f}\t{:.17f}\t{}".format(
        s["wins_above_25"], s["mean_gain"], s["median_gain"], *s["median_interval"], s["p_margin"], int(s["accepted"])))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored;" if s["accepted"]
                 else "test result: FAILED. 0 passed; 1 failed; 0 ignored;")
    return "\n".join(lines) + "\n"


class LearningEvidenceTests(unittest.TestCase):
    def test_complete_success_and_failure_are_recomputed(self):
        result = validate_log(fixture())
        self.assertTrue(result["summary"]["accepted"])
        self.assertEqual(result["summary"]["p_margin"], 1 / 4096)
        failed = validate_log(fixture([0.0] * 12))
        self.assertFalse(failed["summary"]["accepted"])
        self.assertEqual(len(failed["trials"]), 12)

    def test_binomial_probabilities_match_enumeration(self):
        sequences = list(itertools.product((0, 1), repeat=12))
        for wins in range(13):
            result = summary([100.0] * wins + [0.0] * (12 - wins))
            self.assertEqual(result["p_margin"], sum(sum(s) >= wins for s in sequences) / 4096)
        self.assertEqual(summary([100.0] * 10 + [0.0] * 2)["p_margin"], 79 / 4096)

    def test_ties_effect_floor_outliers_and_missing_seeds(self):
        for gains in ([25.0] * 12, [26.0] * 12, [10000.0] + [0.0] * 11):
            self.assertFalse(summary(gains)["accepted"])
        for gains in ([50.0] * 11, [math.nan] * 12, [math.inf] * 12):
            with self.assertRaises(ValueError): summary(gains)

    def test_missing_or_duplicate_episode_is_rejected(self):
        text = fixture()
        record = next(x for x in text.splitlines() if x.startswith("PPO_EPISODE"))
        for broken in (text.replace(record + "\n", "", 1), text + record + "\n"):
            with self.assertRaises(ValueError): validate_log(broken)

    def test_missing_or_duplicate_trial_is_rejected(self):
        text = fixture()
        record = next(x for x in text.splitlines() if x.startswith("PPO_TRIAL"))
        for broken in (text.replace(record + "\n", "", 1), text + record + "\n"):
            with self.assertRaises(ValueError): validate_log(broken)

    def test_wrong_budget_and_seed_are_rejected(self):
        text = fixture()
        for broken in (text.replace("PPO_TRIAL\t101\t128\t65536", "PPO_TRIAL\t101\t127\t65024"),
                       text.replace("PPO_EPISODE\t101\tinitial\t1000000", "PPO_EPISODE\t101\tinitial\t999999")):
            with self.assertRaises(ValueError): validate_log(broken)

    def test_reported_statistics_are_not_trusted(self):
        text = fixture()
        for broken in (text.replace("PPO_SUMMARY\t12", "PPO_SUMMARY\t11"),
                       text.replace("PPO_TRIAL\t101\t128\t65536\t100.", "PPO_TRIAL\t101\t128\t65536\t101.")):
            with self.assertRaises(ValueError): validate_log(broken)

    def test_nonfinite_scores_and_weights_are_rejected(self):
        text = fixture()
        record = next(x for x in text.splitlines() if x.startswith("PPO_EPISODE"))
        parts = record.split("\t"); parts[4] = "nan"
        with self.assertRaises(ValueError): validate_log(text.replace(record, "\t".join(parts), 1))
        policy = next(x for x in text.splitlines() if x.startswith("PPO_POLICY"))
        with self.assertRaises(ValueError): validate_log(text.replace(policy, policy[:-8] + "7fc00000", 1))
        with self.assertRaises(ValueError): validate_log(text.replace(policy, policy[:-8], 1))

    def test_invalid_episode_end_and_impossible_return_are_rejected(self):
        text = fixture()
        record = next(x for x in text.splitlines() if x.startswith("PPO_EPISODE"))
        for column, value in ((5, "999"), (6, "1"), (4, "1001")):
            parts = record.split("\t"); parts[column] = value
            with self.assertRaises(ValueError): validate_log(text.replace(record, "\t".join(parts), 1))

    def test_compiler_failure_or_unexecuted_test_is_not_evidence(self):
        for text in ("error[E0308]: failed to compile", f"test {TEST} ... ok\ntest result: ok. 0 passed; 0 failed; 0 ignored;",
                     fixture().replace("running 1 test", "running 0 tests")):
            with self.assertRaises(ValueError): validate_log(text)

    def test_changed_protocol_and_false_green_summary_are_rejected(self):
        with self.assertRaises(ValueError): validate_log(fixture().replace(PROTOCOL, PROTOCOL + "\tchanged"))
        with self.assertRaises(ValueError): validate_log(fixture([0.0] * 12).replace(
            "test result: FAILED. 0 passed; 1 failed;", "test result: ok. 1 passed; 0 failed;"))

    def test_complete_records_replay_exactly(self):
        first = validate_log(fixture())
        second = validate_log(fixture())
        self.assertEqual(first, second)
        self.assertNotEqual(first, validate_log(fixture([101.0] * 12)))


if __name__ == "__main__":
    unittest.main()
