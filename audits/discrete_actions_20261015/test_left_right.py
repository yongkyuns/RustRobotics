"""Run with: python -m unittest discover -s audits/discrete_actions_20261015."""
import unittest

import gymnasium as gym
import numpy as np

from left_right import LeftRight, discrete_command


class SentinelEnv(gym.Env):
    """An adapter-contract fixture, not replacement cart-pole dynamics."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.calls = []

    def reset(self, *, seed=None, options=None):
        self.calls.append(("reset", seed, options))
        return np.arange(4, dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.calls.append(("step", action.copy()))
        return np.arange(4, dtype=np.float32), 0.125, False, True, {"native": True}


class LeftRightTests(unittest.TestCase):
    def test_left(self):
        np.testing.assert_array_equal(discrete_command(0), [-1.0])

    def test_right(self):
        np.testing.assert_array_equal(discrete_command(np.int64(1)), [1.0])

    def test_one_element_array(self):
        self.assertEqual(discrete_command(np.array([1])).dtype, np.float32)
        np.testing.assert_array_equal(discrete_command(np.array([0])), [-1.0])

    def test_invalid_actions(self):
        for value in (-1, 2, 0.0, 1.0, np.nan, np.inf, True, [0, 1], [[0]], "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                discrete_command(value)

    def test_action_space_and_observation_identity(self):
        native = SentinelEnv()
        wrapped = LeftRight(native)
        self.assertIsInstance(wrapped.action_space, gym.spaces.Discrete)
        self.assertEqual(wrapped.action_space.n, 2)
        self.assertIs(wrapped.observation_space, native.observation_space)

    def test_reset_forwarded(self):
        native = SentinelEnv()
        obs, info = LeftRight(native).reset(seed=123, options={"marker": 7})
        self.assertEqual(native.calls, [("reset", 123, {"marker": 7})])
        np.testing.assert_array_equal(obs, np.arange(4, dtype=np.float32))
        self.assertEqual(info, {"seed": 123})

    def test_reward_and_endings_unchanged(self):
        native = SentinelEnv()
        obs, reward, terminated, truncated, info = LeftRight(native).step(1)
        np.testing.assert_array_equal(native.calls[0][1], [1.0])
        self.assertEqual((reward, terminated, truncated, info),
                         (0.125, False, True, {"native": True}))
        np.testing.assert_array_equal(obs, np.arange(4, dtype=np.float32))

    def test_reject_unnormalized_force_space(self):
        native = SentinelEnv()
        native.action_space = gym.spaces.Box(-20.0, 20.0, (1,), dtype=np.float32)
        with self.assertRaises(ValueError):
            LeftRight(native)

    def test_reject_multiple_actuators(self):
        native = SentinelEnv()
        native.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        with self.assertRaises(ValueError):
            LeftRight(native)

    def test_reject_already_discrete(self):
        native = SentinelEnv()
        native.action_space = gym.spaces.Discrete(2)
        with self.assertRaises(ValueError):
            LeftRight(native)


if __name__ == "__main__":
    unittest.main()
