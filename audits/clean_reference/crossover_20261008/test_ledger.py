import copy
import json
import unittest
from ledger import assert_episode_ledger


class EpisodeLedgerTests(unittest.TestCase):
    def setUp(self):
        self.live = [[(65, 65, -2.25, 1, 0), (321, 256, 120.125, 0, 1)]]
        self.saved = json.loads(json.dumps(self.live))

    def test_original_comparison_fails_but_json_values_match(self):
        self.assertNotEqual(self.live, self.saved)
        self.assertEqual(assert_episode_ledger(self.live, self.saved), self.saved)

    def test_smallest_changed_float_is_rejected(self):
        import math
        changed = copy.deepcopy(self.saved)
        changed[0][0][2] = math.nextafter(changed[0][0][2], 0.0)
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, changed)

    def test_changed_physical_ending_is_rejected(self):
        changed = copy.deepcopy(self.saved)
        changed[0][0][3:] = [0, 1]
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, changed)

    def test_order_change_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, [list(reversed(self.saved[0]))])

    def test_missing_episode_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, [self.saved[0][:-1]])

    def test_missing_environment_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, [])

    def test_nonfinite_reward_is_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.saved)
                changed[0][0][2] = value
                with self.assertRaises(AssertionError):
                    assert_episode_ledger(self.live, changed)

    def test_boolean_counter_not_silently_equal_to_one(self):
        changed = copy.deepcopy(self.saved)
        changed[0][0][3] = True
        with self.assertRaises(AssertionError):
            assert_episode_ledger(self.live, changed)

    def test_empty_ledgers_are_valid(self):
        self.assertEqual(assert_episode_ledger([[]], [()]), [[]])


if __name__ == '__main__':
    unittest.main()
