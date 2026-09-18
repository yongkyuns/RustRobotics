#!/usr/bin/env python3
"""Correct the constructor configuration, preserving the original failed replay.

The inherited SB3 collector/train methods and the original comparator, cases and
thresholds are unchanged. Runtime Burn configuration, not an assumed default,
provides the Adam epsilon and betas. This file is deliberately separate so the
previous 1e-8 experiment remains directly reproducible using compare.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stable_baselines3 import PPO as Sb3PPO
import torch

import compare as boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adam = json.loads((args.input / 'adam-config.json').read_text())
    required = {'beta_1', 'beta_2', 'epsilon', 'weight_decay', 'grad_clipping'}
    if set(adam) != required or adam['weight_decay'] is not None or adam['grad_clipping'] is not None:
        raise AssertionError(f'Unexpected native optimizer configuration: {adam}')
    if not (0 < adam['epsilon'] < 1 and 0 < adam['beta_1'] < 1 and 0 < adam['beta_2'] < 1):
        raise AssertionError('Invalid native Adam settings')

    class MatchedPPO(Sb3PPO):
        def __init__(self, *positional, **kwargs):
            config = kwargs['policy_kwargs']['optimizer_kwargs']
            if config != {'eps': 1e-8}:
                raise AssertionError(f'The original comparison recipe changed: {config}')
            config.update(eps=adam['epsilon'], betas=(adam['beta_1'], adam['beta_2']))
            super().__init__(*positional, **kwargs)

    assert MatchedPPO.train is Sb3PPO.train
    assert MatchedPPO.collect_rollouts is Sb3PPO.collect_rollouts
    boundary.PPO = MatchedPPO
    exit_code = 0
    try:
        # Runs all four original cases, persistent Adam, the self-reset negative
        # control, exact inputs and the original predeclared error thresholds.
        boundary.main()
    except SystemExit as error:
        exit_code = int(error.code or 0)

    # Deliberately restore the original incorrect epsilon for a separate control.
    # No positive case weights, training state or results are changed by this.
    boundary.PPO = Sb3PPO
    case = json.loads((args.input / 'default-201.json').read_text())
    wrong_epsilon = boundary.compare_case(case)
    (args.output / 'wrong-epsilon-control.json').write_text(
        json.dumps(wrong_epsilon, indent=2, allow_nan=False) + '\n')
    summary_path = args.output / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['runtime_adam'] = adam
    summary['wrong_epsilon_control'] = {'epsilon': 1e-8, 'detected': not wrong_epsilon['pass']}
    summary['constructor_correction_only'] = True
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('CORRECTED RUNTIME CONFIGURATION SUMMARY', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if exit_code or wrong_epsilon['pass']:
        sys.exit(1)
    print('RUNTIME-MATCHED BOUNDARY REPLAY PASS', flush=True)


if __name__ == '__main__':
    main()
