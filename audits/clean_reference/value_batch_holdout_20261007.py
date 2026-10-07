#!/usr/bin/env python3
"""Fresh seed-only wrapper around the frozen, tested value_batch_compare_20261007.py.

Only the fixed new-seed allowlist changes. The imported module retains its
SB3 PPO, simulator, horizon, gamma, scaling, reward and evaluator semantics.
Protocol: RustRobotics #35 comment 6045659158.
"""
import importlib.util
from pathlib import Path

FROZEN = Path(__file__).resolve().with_name("value_batch_compare_20261007.py")
spec = importlib.util.spec_from_file_location("frozen_value_batch_compare", FROZEN)
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.SEEDS = tuple(range(830101, 830109))

if __name__ == "__main__":
    experiment.main()
