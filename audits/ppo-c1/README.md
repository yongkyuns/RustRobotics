# PPO C1: complete episodes and reward-to-go

**Disposition: joint C1 is rejected as the correction for #35. No production fix is qualified.**

Date: 2026-09-09. Parent cases: [#35](https://github.com/yongkyuns/RustRobotics/issues/35), [#33](https://github.com/yongkyuns/RustRobotics/issues/33).

## Execution and controls

Baseline: `d724c517c27b182145d4054a66d3937563381dfd`. Executed experiment: `08205754d7a4e9f08cca49beb3ec60bf80647140`. [Actions run 34349190569](https://github.com/yongkyuns/RustRobotics/actions/runs/34349190569) completed all 17 jobs successfully. Execution success is not learning success.

The protocol was [posted before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5601407936). The same ordinary `PpoTrainerSession::train_updates` ran every update, with unchanged actor, critic, bounded distribution, optimizer, reward, noise, disturbances and physical termination. No supplied actor, critic pretraining, separate trainer or training-time simulator-state branching was used.

The collector was changed in a hash-verified disposable checkout to require both at least 512 transitions and 16 episode endings, at a completed episode boundary. Lambda 1 was the other intervention. A complete true-terminal lambda-one trace cancels future critic terms, but retains the current-state baseline and finite-sample variance; external timeouts still bootstrap. Lambda 1 on a fixed incomplete 512-step trace is not pure Monte Carlo.

The experiment used all four combinations of those interventions and exposed development seeds 201–204. Fixed checkpoints were initialization and the first completed update reaching 65,536, 262,144 and 1,048,576 actual transitions. Each checkpoint had 32 deterministic-policy noisy 1,000-step episodes and 64 stochastic-policy noisy 1,500-step discounted episodes, using evaluation randomness separate from training and shared across comparisons. All seeds, checkpoints and failures were retained; no training job failed or was retried for a better result.

Before changing collection, the original attribution bundle and all member hashes were verified, all eight offline controls passed, and both original summaries reproduced byte-for-byte. That was reanalysis, not a new simulator experiment. C1 then passed strict Clippy, 64 unit tests, 7 balancing-diagnostic controls, 9 learning-integration controls, release compilation, formatting and diff checks. Four new Rust controls cover episode/transition floors, true-terminal reward-to-go, timeout bootstrapping and collection replay/RNG independence.

Initial workflow run 34349049001 failed validation before any jobs or training. Job environment expressions and bash pipeline failure handling were repaired without changing the numerical protocol. This setup failure was not a failed training seed.

## From-scratch result

Final predeclared checkpoint; means weight each of the four training seeds equally. Completion counts describe their paired evaluation panels, not 128 or 256 independently trained agents. Whole-episode arms overshoot the threshold; actual costs are below.

| Arm | Mean deterministic return | Ten-second completions /128 | Mean discounted stochastic return | Fifteen-second stochastic completions /256 |
|---|---:|---:|---:|---:|
| Unchanged baseline | 531.487 | 57 | 79.173 | 96 |
| Complete episodes only, lambda .95 | 725.119 | 93 | 82.441 | 186 |
| Lambda 1 only, fixed 512 steps | 493.793 | 53 | 78.853 | 108 |
| Joint C1 | 528.345 | 59 | 79.501 | 122 |

Joint C1 is worse on both return scores for every seed at the early checkpoint. At the final checkpoint, deterministic return improves for two seeds and worsens for two; stochastic discounted return improves for only one seed. Joint C1 is not a reliable correction.

Coverage-only is promising but not qualified: its final stochastic score improves for all four seeds, while deterministic return regresses for seed 204. Its early deterministic return is worse for every seed. Final mean paired deterministic improvement is +193.632 with a descriptive pointwise 99% Student interval [-322.439, +709.703]; stochastic improvement is +3.268 with interval [-5.280, +11.816]. Four exposed development seeds and these broad intervals are not new held-out acceptance. No isolated frozen-witness comparison was run for coverage-only.

Actual final training steps, seeds 201/202/203/204:

- Baseline and lambda-one-only: 1,048,576 each.
- Coverage-only: 1,093,492 / 1,104,419 / 1,093,258 / 1,061,764.
- Joint C1: 1,088,111 / 1,073,209 / 1,088,514 / 1,058,324.

## Frozen witnesses

Old actor, old critic and the baseline next actor match the original witnesses byte-for-byte. The unchanged prefix was replayed twice from initialization, preserving true optimizer history rather than treating a weight load as an exact resume. One joint-C1 update was applied after the second prefix. Its first episode can be the remaining part of the live baseline episode; later episodes reset normally.

Each policy had 256 fresh reset evaluations plus paired continuations on the same 32 original physical states, 32 independent innovations per state. First-action tests use the new policy only initially, then the old policy. Full-policy tests use it throughout. Below are candidate-minus-old discounted return changes with approximate pointwise 99% conditional simulation intervals, not simultaneous intervals or new-training-seed uncertainty.

| Witness | Baseline full-policy reset effect | Joint full-policy reset effect |
|---|---|---|
| 201/516 | -2.315414 [-2.916700, -1.714127] | -0.758635 [-1.132618, -0.384652] |
| 202/537 | -0.190128 [-0.460175, +0.079920] | +0.261983 [+0.131261, +0.392704] |
| 203/514 | -0.181607 [-0.375851, +0.012637] | +0.121050 [+0.000380, +0.241719] |
| 204/526 | -0.176725 [-0.244391, -0.109059] | +0.161916 [+0.099003, +0.224828] |

**Seed 201 remains harmful even on the first-action test:** joint change -0.006196, conditional 99% interval [-0.007175, -0.005218]. Its full-policy effect on the fixed-state panel is also negative (-0.590359). Seed 203's fixed-panel first-action effect reverses to positive. Seed 204's joint first-action interval still crosses zero; its precise action-credit cause remains unresolved.

The fresh smaller baseline reset panel does not re-confirm seeds 202/203 at the same precision as the original larger confirmation. Both point estimates remain negative and the actor bytes are identical; do not erase the original results or pretend all four were newly confirmed at equal precision.

This study does not establish the residual cause at seed 201. The joint update changes collected data and update schedule. Its own full rollout arrays and timeout count were not separately exported for a fresh reward/baseline/optimizer decomposition. Harm on the old state panel is not proof of a wrong-sign estimate on the new batch.

## Cost, evidence and disposition

The 16 from-scratch runs used 17,049,699 training transitions. Frozen prefix replays and intervention updates bring total measurement training interaction to 19,206,362, excluding unit controls. Additional diagnostic transitions: 2,103,078 from-scratch evaluation, 756,062 witness resets, 2,093,666 branched continuations. No diagnostic data trained or selected an actor. Joint witness batch sizes were 5,079 / 3,096 / 4,894 / 2,410 versus 512 for each baseline update; these are not equal-cost comparisons.

Independent Python replay verified all 17 raw archives, 452 archived member hashes, identical compiled source, exact checkpoint bytes, compute accounting and 29,696 outcome records. Eight analyzer controls pass and the summary reproduces byte-for-byte. Per-seed/checkpoint results, all termination causes, observed training-call timings and uncertainty calculations are retained.

The self-contained conversation bundle is **`ppo-c1-results.zip`**, SHA256 **`1c8873e05c4e5f2305ff3087aadb83a9e4d2000689ee9b13a53c50d2583aedac`**. It includes all raw C1 artifacts, the entire original attribution bundle, executed compiled source, preparation/harness/workflow files, independent analyzer/tests, full report, summaries and verification logs. Retrieve it by name from saved conversation files; do not assume an old sandbox path or indefinitely retained Actions artifact. Artifacts are named `ppo-c1-controls` and `ppo-c1-{baseline,coverage,returns,joint}-{201,202,203,204}` in run 34349190569; IDs and published SHA256 values are in bundled `artifacts.json`.

Offline replay, with NumPy/SciPy and Python without `-O`:

```sh
cd ppo-c1-results
python unpack.py
cd prior/ppo-update-attribution-results
python unpack.py
cd ../../analysis
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_analyze_c1.py
OPENBLAS_NUM_THREADS=1 python analyze_c1.py \
  --root ../results \
  --prior ../prior/ppo-update-attribution-results/results \
  --output replay-summary.json
cmp summary.json replay-summary.json
```

For new Rust execution, use a disposable checkout of the executed commit, Rust 1.98.1, restored original controls/seed evidence, and the exact preparation/environment/test commands in `.github/workflows/ppo-c1.yml`. The committed source alone is not the prepared compiled source; use the archived source and preparation hashes. Do not rerun selected seeds until favorable.

**Production defaults and protected MuJoCo/control files are unchanged. #35 and #33 remain open; no merge-ready correction PR is delivered.** Heavy frozen learning qualification, full platform/numerical/audit checks, rebuilt WASM/browser checks and a new held-out sustained-balancing protocol were not run for this experimental branch. Diagnostic success does not authorize a merge.

The narrow remaining diagnostic is to recover the exact joint seed-201 batch and separate its reward, current-baseline, sampling and optimizer contributions before choosing another correction. Coverage-only remains a development clue, not a supported workaround or a solved training recipe.
