# Recovery-value calibration and larger-batch PPO comparison

September 18, 2026. Learning issues: #35 / #33. Existing integration correction: PR #37.

## Decision

**Completed; reject the proposed larger-batch schedule as a reliable balancing correction.** Larger batches make the measured whole-batch clipped objective more consistent, but both large-batch arms learn worse policies than the original schedule at the same interaction budget. Eight independent streams outperform the equally large single-stream control on all four seeds, yet remain worse than baseline on three seeds. No production default, actor, critic architecture, reward, or optimizer equation was changed; no merge or held-out qualification was performed.

The preceding observation-only value-calibration diagnostic also has mixed results. Neither its affine nor quadratic correction is installed in training. The results narrow what is insufficient, not establish a unique cause of the remaining failures.

## Protocol and execution

[Calibration protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5735344528) was recorded before its measurements. After inspecting that diagnostic, the [three-arm training protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5735385214) was recorded before training.

Production baseline: `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`. Actual executed audit commit: **`2d3765e2748153dcaff68975fb29023cf5693a60`**. [Actions run **35388608279**](https://github.com/yongkyuns/RustRobotics/actions/runs/35388608279) completed all **13 jobs** successfully: native/reference preflight plus all twelve measurements. No failed training run, rerun, selected best checkpoint, or discarded seed. A successful job means execution and invariants passed, not that sustained balancing passed.

All arms use exposed development seeds 201–204 and the same per-seed random initial actor/critic weights. Each trains for exactly **1,048,576 real simulator transitions**, retaining checkpoints 0, 65,536, 262,144 and 1,048,576.

| Arm | Environments x rollout steps | Minibatch | Epochs | Policy refreshes per run | Paired actor/critic optimizer transactions |
|---|---:|---:|---:|---:|---:|
| Unchanged baseline | 1 x 512 | 128 | 4 | 2,048 | 32,768 |
| Large-single control | 1 x 4,096 | 1,024 | 4 | 256 | 4,096 |
| Pooled candidate | 8 x 512 | 1,024 | 4 | 256 | 4,096 |

Every run has 4,194,304 gradient-sample visits per network. Equal interaction and gradient-sample budgets are **not** equal Adam histories, policy-refresh counts or compute. The large-single versus pooled comparison holds the 4,096-transition update schedule fixed; stream count and trajectory fragmentation differ. Neither large arm versus baseline is a pure coverage intervention.

Everything else is unchanged: existing two-layer 64-unit ReLU networks, Adam epsilon1e-5, learning rate3e-4, gamma0.99, GAE lambda0.95, clipping0.2, four epochs, fixed exploration, actual noisy Rust plant, original resets/rewards/termination and timeout handling. The existing PR37 shared-policy collector and ordinary optimizer perform all training. The audit adds only configuration constructors, an ABI, and observation hooks in a disposable checkout. Protected source restoration is verified.

There are no forced extra resets, privileged-state training inputs, critic pretraining, fitted calibration coefficients, supplied controllers, extra evaluation-selected updates, objective-based rollback, or artificial ReLU derivatives. Actual evaluation uses the unchanged C3/C4 panels: 32 noisy deterministic-policy 1,000-step episodes and 64 noisy stochastic-policy 1,500-step episodes per checkpoint. These are already-exposed panels, not fresh held-out acceptance.

## Final learned-policy results

Completion means reaching the specified cap without true termination. Deterministic describes policy action selection, not a noiseless environment. Stochastic discounted return uses gamma0.99 and differs from the undiscounted deterministic score.

| Arm | 10-second deterministic completions /128 | Deterministic mean return | 15-second stochastic completions /256 | Stochastic discounted mean return |
|---|---:|---:|---:|---:|
| Baseline | **50** | **489.949** | **98** | **77.331** |
| Large-single | 4 | 160.196 | 1 | 68.173 |
| Pooled | 11 | 270.220 | 17 | 72.510 |

### Every final training-seed score

| Seed | Baseline deterministic / stochastic discounted | Large-single deterministic / stochastic discounted | Pooled deterministic / stochastic discounted |
|---|---|---|---|
| 201 | 450.151 / 84.003 | 136.872 / 70.705 | 299.062 / 77.894 |
| 202 | 273.431 / 70.833 | 236.733 / 70.684 | 421.320 / 74.702 |
| 203 | 656.259 / 79.296 | 107.448 / 61.990 | 115.401 / 63.751 |
| 204 | 579.955 / 75.193 | 159.731 / 69.312 | 245.098 / 73.694 |

Large-single is worse than baseline on both scores for all four seeds. Pooled improves both scores only on seed202. Pooled beats large-single on both scores for every seed, so additional streams help within this larger-batch schedule; that does not rescue the proposed schedule as a default.

At the first two fixed checkpoints, mean deterministic return is baseline121.225/167.622, large-single58.835/120.302, pooled57.093/118.385. Stochastic discounted means are baseline64.143/70.296, large-single43.096/64.656, pooled42.905/63.066. All per-seed checkpoint outcomes, including these early regressions, are retained in `results/checkpoints.csv`.

### Paired final effects and uncertainty

| Contrast | Deterministic mean effect; pointwise99% interval | Stochastic discounted effect; pointwise99% interval |
|---|---|---|
| Pooled minus baseline | -219.729; [-1073.235, +633.778] | -4.821; [-28.859, +19.217] |
| Large-single minus baseline | -329.753; [-965.857, +306.351] | -9.158; [-31.493, +13.176] |
| Pooled minus large-single | +110.024; [-124.264, +344.313] | +4.337; [-2.165, +10.840] |

These are descriptive Student intervals over **four paired training seeds**, df3, not pooled-episode intervals, simultaneous guarantees, or equivalence tests. Their width prevents a precise expected-effect claim. It does not justify promoting a candidate with the observed regressions, nor erase the observed all-four-seed pooled-versus-large-single ordering.

## The objective-monitoring result

Observation-only hooks evaluate the actual full-rollout clipped surrogate before optimization and after every epoch. They never select, reject or modify an update. Using a descriptive decrease threshold of1e-6:

| Arm | Final surrogate worse than pre-update | Final surrogate worse than first epoch |
|---|---:|---:|
| Baseline | 276 / 8,192 updates (3.37%) | 1,274 / 8,192 |
| Large-single | 2 / 1,024 (0.20%) | 19 / 1,024 |
| Pooled | 4 / 1,024 (0.39%) | 24 / 1,024 |

Thus the large-batch schedules reduce the C2-style deterioration of the measured batch objective while producing worse learned control. Better optimization of that finite-data objective is insufficient to establish better true return or sustained balancing. This experiment does not independently verify every action-credit sign.

Counts of noisy observations satisfying abs(cart position)>=0.5 and position*velocity>0 are baseline1,065,392, large-single1,199,133, pooled1,309,914 over each arm's 4,194,304 total training transitions. More such samples and more episode endings do not automatically mean more useful recovery diversity: worse policies also terminate more often and visit different states.

Fewer policy/Adam updates are a plausible contributor to the large-arm regressions, but their causal contribution is not isolated from batch size and data correlations. Do not infer that all vectorized PPO is worse, that larger batches cannot work, or that critic undertraining has been proved. An equal-update experiment would have a different interaction budget and is not present here.

## Independent-episode value calibration

The diagnostic uses the four original Rust-e5 final-policy stochastic trajectories from cart-centering run35385827285/artifact10563588646. The full original archive SHA256 is `c3ec6daa968df8c2b1b269a3c4ea19510031c84ef91a8e886a927d901a254a2a`; all79 original manifest entries were checked before analysis.

Actual reward-to-go is reconstructed with gamma=float32(0.99), without bootstrapping from the critic being tested. Observations are sampled at ticks0,25,...,475 while alive. Time-limited paths have at least1,000 subsequent recorded steps; missing later returns remain a finite-horizon limitation, not presumed identically zero. The outward/off-centre subset is defined using noisy policy observations, never privileged simulator state.

Fit a frozen-critic residual on even episodes and evaluate only odd episodes, then reverse folds. Compare affine observation features with affine-plus-all-ten-quadratic-products. Standardization uses only fitting episodes; ridge is fixed at1e-3 in mean-loss units with an unpenalized intercept. No fitted coefficient or diagnostic outcome enters training.

| Seed | Outward-state RMSE: original critic | Episode-cross-fitted affine correction | Episode-cross-fitted quadratic correction |
|---|---:|---:|---:|
| 201 | 2.885 | 2.486 | 2.236 |
| 202 | 5.244 | 6.140 | 9.333 |
| 203 | 3.782 | 4.084 | 3.599 |
| 204 | 13.574 | 11.326 | 13.999 |

Neither correction generalizes consistently across these four policies. This comparison is against noisy realized return, not a noise-free conditional value oracle; it does not prove observation-only Markov sufficiency, isolate all critic bias, or certify action credit. Whole-episode folds prevent same-episode fitting/evaluation leakage but are not new held-out trained agents.

The new pooled policies' outward-state RMSE is5.883/6.613/2.027/7.694 for seeds201–204, versus2.885/5.244/3.782/13.574 for baseline. Some lower errors coexist with substantially worse control. These are **own-policy, own-visited-state** prediction errors, not matched-state causal comparisons: distributions, survival and continuation noise differ. They cannot be used alone to select a correction.

## Verification, reproducibility and limits

Preflight passed strict Clippy, all73 native unit/ABI/audit controls,7 balancing-diagnostic controls and9 ordinary learning-integration controls, compilation/formatting, real-ABI/reference controls, evaluation-wrapper parity and protected-source restoration. The separately ignored heavy tests were not executed. Observer and ordinary twin sessions retain identical networks, metrics and subsequent rollouts; observations do not change training.

All baseline actor/critic checkpoint bytes match C4 exactly, for every seed and all four checkpoints. Baseline evaluation records are exact for201/202/204. For seed203,361/384 records differ in floating fields; maximum undiscounted-return difference0.00200951, discounted difference0.000104210, final-state component difference0.000173718. Episode durations, ending types and panel keys remain identical. These differences are retained with no favorable rerun; their numerical/platform cause is not uniquely attributed. No portable bitwise evaluation claim is made.

Independent analysis verifies13 raw archive digests and320 manifest-listed members,4,608 unique evaluation records,48 actor/critic checkpoint pairs, fixed budgets and epoch coverage, all51,200 epoch records, and final trajectory observation/reward/ending accounting. Actual recorded rewards reproduce episode undiscounted/discounted totals to the declared offline checks. The observer does not export commanded forces, and offline analysis does not claim to reconstruct unexported optimizer gradients or re-simulate the physical reward function. Fourteen independent analyzer/calibration controls pass.

Main costs: **12,582,912 training interactions; 931,492 evaluation interactions; 163,840 paired actor/critic optimizer transactions; 50,331,648 gradient-sample visits per network.** Preflight unit/Gym/reference interactions are additional and not all aggregate-instrumented. Calibration reuses old data with zero new interactions. No speed or all-inclusive compute-budget claim.

## Evidence and status

Self-contained conversation bundle: **`rustrobotics-ppo-recovery-batches-evidence.zip`**. It contains all13 unchanged raw archives, four unchanged C4 baseline archives, digests, exact executed source/version provenance, outcomes/checkpoints, calibration data, independent analyzers/tests, report and offline replay instructions.

To avoid duplicating the148MB cart-centering archive, `calibration-inputs.zip` contains only the four byte-preserved original Rust-e5 stochastic NPZ members, their original member hashes and source-archive provenance. It is explicitly a selected-member package, not the complete original archive. Numerical calibration results reproduce from those same members. Actual native training source/library and dependencies are in the native raw archive; offline replay does not execute the library or retrain.

Keep all supported defaults and master unchanged. This is a completed negative candidate, not a new supported mode. #35/#33 remain unresolved; no new held-out seeds were consumed. The remaining question is how to improve ongoing value/action-credit estimation without sacrificing useful actor update cadence, and it requires a separately declared discriminating test rather than retroactively extending this study's budget.
