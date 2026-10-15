# Discrete left/right PPO diagnostic

An isolated comparison for RustRobotics issue #35. The before-outcome protocol is [comment 6200292905](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6200292905).

This does **not** modify the production Rust trainer, its default action distribution, the viewer, the physical task, or deployment. Both learners use the same pinned SB3 PPO implementation against the real native nonlinear Rust cart-pole environment. This deliberately separates the action-space question from Rust-versus-Python optimizer differences.

## What changes

- Continuous: the existing normalized constant-rate reference, with a clipped Gaussian force command.
- Discrete: a true two-logit categorical policy; index 0 maps to -20 N and index 1 to +20 N. No Gaussian rounding, neutral action, or magnitude sweep.

The physical plant, reward, observation/actuation noise, reset distribution, termination rules, 10 ms timestep and training budget stay fixed. Shared actor-hidden tensors and the complete initial critic are paired. The action heads, initial action variances and sampling laws necessarily differ.

Fixed ±20 N incurs the unchanged 0.4-per-step effort penalty, except on true terminal transitions whose reward is -10. Completion and survival are primary diagnostics; equal completion should not be mistaken for equal return across action spaces.

## Fixed experiment

Four exposed development seeds, 201–204. Checkpoints 0, 65,536, 131,072, 262,144 and 524,288 interactions per arm. The 262,144 checkpoint is followed by another 262,144 interactions without clearing Adam. Keep all runs, checkpoints and failures; no outcome-driven early stopping or best-checkpoint substitution.

Use all original 64 ordinary and 64 difficult native-reset cases per seed, in deterministic and stochastic modes, through 1,500 physical ticks. Read ten-second survival from the same stopped traces. Evaluation freezes the normalizer, policy, optimizer, training RNG and live observation. These are exposed development panels, not a new held-out qualification.

Recipe: stock observation-only VecNormalize; gamma .999, GAE lambda 1, 2,048 rollout rows, batch 64, ten epochs, constant learning rate 3e-4, Adam epsilon 1e-5, policy clip .2, global gradient norm .5, entropy coefficient 0, two 64-unit tanh layers. Preserve the reference's 1/1000 reward-unit conversion and corresponding initial critic-head scaling. The native training timeout is 256 steps; physical evaluation uses the original 1,500-tick cap.

## Source and inputs

`code/study.py` is the **exact executed collector** (Git blob `41675eba1b56a9b016c891ce4833210f3684cc62`), including its frozen inline action adapter. `left_right.py` is the reusable adapter with additional normalized-space validation; its contract tests use a sentinel environment, not substitute cart-pole physics. Native equivalence, categorical probability/gradient, actual learning, persistent-Adam and evaluation-isolation controls are in the collector.

The collector requires the retained experiment layout, rather than assuming third-party binaries or old chat sandbox paths exist:

```
discrete_actions_20261015/
  code/study.py
  parent/code/{driver,value_batch_compare_20261007,paired_pool_20261006}.py
  parent/inputs/native.zip
  parent/provenance/{vec_normalize,running_mean_std}.py
  parent/results/selection/...
  results/
  logs/
```

These parent inputs come from `ppo_lr_decay_report_code_20261014.zip`, SHA256 `5cfa6ec278eaf8f2812b2c84952309b18b14190fdd6d0018147b725919b2f5b6`. The complete delivered experiment bundle preserves the needed source and input subset plus the independent reader and supervisor. Do not confuse its subset layout with a complete replay of the previous LR experiment.

The native archive SHA256 is `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`; its production base is PR #38 head `95c670b9f4618a11dc5439b026caf385a39718c8`, and its diagnostic bridge source is `da6623ae4b5878ae5079bf05c18c3d327570bc16`. The loader checks all native payloads and actual PPO/collector/buffer/policy source bytes.

Pinned execution: Python 3.11.14, Torch 2.8.0+cpu, NumPy 2.2.6, SB3 2.9.0, Gymnasium 1.3.0. Recovered runtime: Actions run `38595509112`, artifact `11912632514`, SHA256 `c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1`. The runtime is not redistributed with the evidence bundle.

## Running in a new experiment directory

Do not overwrite recorded results. Supply the pinned interpreter and retained parent inputs, create empty results/logs directories, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/study.py preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/study.py run --seed 201 --arm continuous
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/study.py run --seed 201 --arm discrete
```

The recorded supervisor runs four seed pairs, with continuous first on odd seeds and discrete first on even seeds, sequential within each pair. Every output directory is create-only. A failed run is retained, not automatically retried. Policy/Adam/normalizer exports are **not** complete native-simulator-resume checkpoints.

Adapter-only tests require Gymnasium and NumPy:

```sh
python -m unittest discover -s audits/discrete_actions_20261015 -p 'test_*.py'
```

No production reliability, browser deployment or issue closure follows merely from successful adapter tests or this development experiment.
