# PR #41 / issue #42 — per-step force witness: recovered complete development evidence

**Date: October 16, 2026.** **Evidence status:** The eight native training runs and original evaluation gate **completed successfully**, but the [original workflow #38944679027](https://github.com/yongkyuns/RustRobotics/actions/runs/38944679027) concluded **failure** because its final *Python parser* raised `KeyError: 'before_x'`. The CSV writer introduced leading spaces in the header names of `before_x`, `applied_force_n`, and `rail_breached` when breaking its long Rust string. This is an **artifact-validation bug**, not failed learning or a numerical PPO test.

Immutable source: [`1261f9c6`](https://github.com/yongkyuns/RustRobotics/commit/1261f9c6a3b6f5aa13cf9632327cf2d3bb52634d).
[Full retained artifact #12065457534](https://github.com/yongkyuns/RustRobotics/actions/runs/38944679027/artifacts/12065457534) (GitHub ZIP digest `sha256:93ddc62f27d8f8a9e2d99f434ac45b833277268d10a0570f20b5afa644c24d5f`).
[Reusable verifier](../scripts/verify_cleanrl_force_witness.py) and [verification-only workflow](../.github/workflows/cleanrl-force-evidence-verify.yml) reuse **those exact artifact bytes without retraining**. The independent CI verification had not yet reached a terminal verdict when this audit was written; do not relabel the original failed training workflow as success.

## Experimental protocol and hard evidence boundary

- Already-exposed training seeds **201–204**, both categorical and continuous squashed-Gaussian action heads; 976 PPO updates and 499,712 native transitions for each policy: **3,997,696 training interactions total**, with a persistent optimizer within each seed.
- Same nonlinear RK4 plant, 0.01-s physical step, 64-key shared difficult panel, stochastic/deterministic evaluation at all seven checkpoints, **28,672 preserved nominal evaluation episode records**. Original categorical final common-panel counts reproduced exactly.
- At final update 976, after running the **uninstrumented evaluator on all 64 lanes**, replayed each of **four deliberately post-selected exposed difficult keys** using the same 64-lane batched read-only evaluator. The Rust program asserted **full 64-episode result equality** for each instrumented replay; all 64 such assertions completed without a panic.
- Full CSV artifact contains **64 witness episodes** and **40,525 time-step rows** for the four keys across 4 training seeds × 2 action laws × 2 inference modes.
- The four keys were selected **after examining earlier development failures**. They are mechanistic witnesses, **never held-out confirmation**, and neither model is retrained against them.

## Full-force and initial feedback results

The table shows 15-second completions out of 8 seed/mode combinations per key. Requested force magnitude averages cover the **first 30 ticks = 0.3 seconds**, paired on the same seed, mode, reset key and physical tick.

| Initial reset key | Categorical 15-s completion | Gaussian 15-s completion | Categorical initial mean abs force | Gaussian initial mean abs force | Gaussian median failure time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4100100015` | **8/8** | **0/8** | 20.00 N | **13.64 N** | 0.765 s |
| `4100100021` | **8/8** | **0/8** | 20.00 N | **13.51 N** | 0.845 s |
| `4100100170` | **8/8** | **0/8** | 20.00 N | **14.13 N** | 0.870 s |
| `4100100100` | **1/8** | **0/8** | 20.00 N | **14.92 N** | 0.685 s |

Across **960 paired early ticks** (4 keys × 8 seed/mode pairs × 30 ticks), categorical and Gaussian commanded forces have the **same sign in 937/960 (97.60%)**. For individual keys agreement was 96.67%, 95.42%, 98.33% and 100% respectively. In other words, the Gaussian usually initially chooses the *same direction*, but applies less corrective force.

The archive reports **39 unsuccessful episodes out of 64 selected combinations**: 7 categorical and 32 Gaussian. **All 39 crossed the cart rail** at their terminal tick and had **no simultaneous pole-angle threshold breach** in the recorded true-state traces. Their force/position traces are full source data, not model predictions.

In the three categorical-only recovery keys, categorical requested full **±20 N** action at every 10-ms step, whereas Gaussian generally requested ~**13.5–14.1 N** absolute force during the first 0.3 s. Plant action noise and disturbances change the realized force slightly; actual force is retained separately in the artifact. Categorical's high-frequency bang-bang force reversals also remain an actuator-realism concern.

## What this supports—and does not prove

The evidence makes **insufficient initial corrective-force magnitude** a substantially better-supported hypothesis than a general wrong-sign-action explanation for these frozen witnesses. Importantly, 23 of 960 paired initial commands *did* differ in sign, and later feedback trajectories diverge. The data **do not prove** that merely increasing Gaussian force will fix the failures, nor show that the state is inherently unrecoverable. Direct **evaluation-only** Gaussian force-gain/clamping interventions, keeping policy weights and training unchanged, are an appropriate next causal test; this audit itself does not report such outcomes.

The two laws still differ in exploration/entropy semantics, and no new independent training seed has been tested. **Original production Gaussian PPO and controls remain untouched.** No merge, default change, hardware claim, or independent issue #33 acceptance follows.

## Reproduction and checksums

The raw trace CSV parser must strip **header names only**, not measured data. The earlier verifier read `frame['before_x']` but the true key was `'                 before_x'`. [The new verifier](../scripts/verify_cleanrl_force_witness.py) requires all 28,672 episode records and 64× full witness trajectories, checks checkpoint/seed/keys, pinned source provenance, individual results against the nominal evaluator, rail/angle flags, command bounds and file hashes. It does not re-run PPO.

Independently computed SHA256 of nominal episode CSVs:

- categorical seed 201: `7121b67a85458326e3b790217c1916c19696cba6c0bd581c5e4b032f6e91f88a`
- categorical seed 202: `bc5379e3fba34e00e3a8b2b5f71539ae62c6206f18c25fe3ee9806b03dcb01cc`
- categorical seed 203: `f9b43b2cd380e9196ed2a40f1a2778568160ea419c51ccdad1ca3f1500d5e7db`
- categorical seed 204: `2462df914cd70d689a644ba803b9ae9a0ef91f5c8bf98d264b934c737ab3ab42`
- Gaussian seed 201: `607c708f4fad3058f5d07c58556088fba0a8eec502ca6ede77ff81215605c5df`
- Gaussian seed 202: `3e2b08b7d18cec0ed99c802e2f556d0e437bea1026fd390e987fe9f99f87660a`
- Gaussian seed 203: `0db47f9f5877c3895603f4f7fe6f93a9752827cf40d63a603fd5d4d3f0eba9c4`
- Gaussian seed 204: `dcd7f620bc793951e154cbb86e740449d8f619c717fd3d6976674fbb0d2a1785`

No success-rate claims for unseen reset distributions, physical actuators, or fresh #33 acceptance are made.
