"""Categorical-action adapter for the existing continuous Rust cart-pole bridge.

The native reference's transform=2 maps normalized -1/+1 to -20/+20 N.
Use this wrapper with a genuine categorical PPO policy, not Gaussian rounding.
Only the action-space contract changes; observations, rewards, noise, resets,
physical integration and episode endings remain owned by the wrapped plant.
This module is an isolated diagnostic, not a production-trainer default.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


def discrete_command(action) -> np.ndarray:
    """Accept one integer category: 0 pushes left, 1 pushes right."""
    value = np.asarray(action)
    if value.shape not in ((), (1,)) or value.dtype.kind not in "iu":
        raise ValueError("expected one integer action index, 0=left or 1=right")
    index = int(value.item())
    if index not in (0, 1):
        raise ValueError("action index must be 0 or 1")
    return np.asarray([-1.0 if index == 0 else 1.0], dtype=np.float32)


class LeftRight(gym.ActionWrapper):
    """Expose Discrete(2) without changing the original physical environment."""

    def __init__(self, env):
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("expected the normalized continuous action space")
        if env.action_space.shape != (1,):
            raise ValueError("expected one cart-force command")
        if not (np.array_equal(env.action_space.low, [-1.0]) and
                np.array_equal(env.action_space.high, [1.0])):
            raise ValueError("expected native bridge commands normalized to [-1, 1]")
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(2)

    def action(self, action):
        return discrete_command(action)
