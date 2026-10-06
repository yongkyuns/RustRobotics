# Ordinary PPO pool comparison: qualified setup, learning results pending

October 6, 2026. This is a setup/progress record, not a completed learning comparison or a new supported default.

## Why this experiment

The completed conditional-credit report at f1838334663d27d6616b9c7f0821481c30a2983f changes the strongest explanation for frozen stock-SB3 seed203/update54. Independent native continuations show genuine average improvement from the sampled already-balanced states: +0.034989672 stopped discounted return for the new full policy and +0.000382424 for its first action followed by the old policy, at the fixed 2048-step horizon. The same update damages ordinary reset recovery. On the tested panel, return-based and GAE action-credit projections are both positive; imperfect critic calibration does not establish wrong-sign credit for this witness.

This supports testing standard environment-stream diversity before adding a curriculum, an oracle critic or another special recovery mechanism. It is not a universal attribution of every PPO failure.

## Frozen protocol

Registered before new outcomes in [issue #35/comment 6023805256](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6023805256).

Compare ordinary stock SB3 on the same original native nonlinear PendulumEnv:

| Item | One stream | Eight streams |
|---|---:|---:|
| Steps per environment/update | 2048 | 256 |
| Total transitions/update | 2048 | 2048 |
| Total interactions/seed | 1048576 | 1048576 |
| Rollout updates/seed | 512 | 512 |
| Adam steps/seed | 163840 | 163840 |
| Sample visits/network/seed | 10485760 | 10485760 |

Four exposed development seeds201–204. Each pair runs sequentially on the same job/CPU, with identical random initial actor AND critic tensors required. The actual unmodified SB3 PPO.train/collect_rollouts/GAE implementations are used. No parameter averaging or warm-started controller is introduced.

The ordinary tanh64x64 architecture, learned Gaussian standard deviation, Adam epsilon1e-5, learning rate3e-4, gamma.99/lambda.95, batch64, ten epochs, minibatch advantage normalization, norm clipping0.5, noise, rewards, reset distribution and 5000-step episode cap remain unchanged. The change also affects per-stream cutoff frequency, numerical batch shapes and random-draw assignment. It is not perfect isolation of state coverage alone.

All fixed checkpoints0/65536/262144/1048576 are retained. New fixed evaluation parameter keys920000+training_seed are common within each pair and checkpoint:32 deterministic10-second and64 stochastic15-second episodes. No best-checkpoint or seed selection is allowed.

The predeclared developmental screen requires the eight-environment endpoint to reach at least31/32 deterministic and61/64 stochastic completions in EACH seed, no per-seed completion loss against one environment, and positive mean paired stochastic discounted-return improvement. Even passing would only justify a separate new held-out qualification; it would not establish production readiness. Four training seeds, not pooled episodes, are the uncertainty unit.

## Implemented and executed

Diagnostic training script: `audits/clean_reference/paired_pool_20261006.py`, commit150df502dd8f5d78a56d77b81891e66cbd4f8598. Workflow execution source: **77f8624d419666ffdadbc00d7c73f0c5bb3aadd9**.

[Run37519334101](https://github.com/yongkyuns/RustRobotics/actions/runs/37519334101) has completed its preflight successfully. At the latest checked job state, seed202 and203 pairs are executing and seed201 and204 pairs remain queued. No completed paired-result archive has been inspected and no multi-environment learning outcome is claimed here.

The hosted preflight passes six named control groups:

- Independent forward-sum multi-stream GAE/targets, plus a reset-boundary mutation.
- Exact one-environment constructor/training equivalence to the original make_model path.
- Exact observer versus ordinary policy/critic/Adam/RNG/observation/episode equality through two real updates, separately for one and eight environments.
- Equal starting actor/critic tensors across environment counts, first-stream reset identity, and independently checked distinct extra-stream native seed keys.
- A changed-data equality negative control.

The hosted preflight actually executes24576 short-control training interactions. It is not a sustained-balancing measurement. Its original artifact **11437969566**, 2783381bytes, SHA256 **eed51d09fa54573842a7b7ca84af281d7e6baee9b95451547e58013c87f39b2a**, was downloaded and all154 manifest-listed payloads verified. The extracted training script exactly matches the local source and Git blob15a8d34c39a8179e5a7dac7d6b933c0e7dc1658e.

A separate local execution of the same preflight also passes on AMD EPYC9V74 using the transported pinned runtime; the hosted preflight used Intel Xeon Platinum8573C. This adds another24576 short-control interactions, not an independent learning trial. It does not assert across-CPU bitwise equivalence of whole learning trajectories.

Both preflights use Python3.11.16, Torch2.8.0+cpu, NumPy2.2.6, SB32.9.0 and Gymnasium1.3.0. The original native artifact10860907319 and four installed SB3 source modules are byte/hash verified. The transported runtime artifact11434833708 is used only for local execution; its third-party binaries are excluded from the evidence package.

## Reader qualification and remaining work

An independent read-only analyzer is prepared for the complete four-pair comparison. It validates source/native archive identity, all checkpoint tensors and evaluation keys, final Adam histories and budgets, and every per-stream episode start/end against the recorded training history. It computes all checkpoint and casewise changes, four-seed descriptive99% t intervals and the frozen developmental screen.

**26 synthetic/statistical/archive reader tests pass.** Nine additional actual-evidence tests are authored but have NOT run because complete pair archives are not yet available. The analyzer deliberately rejects an incomplete four-pair registry; these tests are not represented as verification of unobserved learning results.

The evidence package retains the original hosted preflight ZIP, local preflight receipt/log, runnable audit/reader/tests, artifact IDs/hashes and reproduction instructions. It is a setup package, not a complete result package. Main planned training cost is8388608 interactions across eight arms; no executed main-work count is claimed before reading those jobs' receipts.

The next continuation must read all original paired artifacts from this SAME fixed run, preserve failed or partial pairs, and finish the reader/actual-data tests before interpreting the development screen. Do not relaunch already-running experiments, replace an unfavorable seed, relax thresholds or select an earlier checkpoint.

Only diagnostic script/workflow/report files were added. No production training/default/master/protected-source changes, production PR merge, or deployment were performed by this work. The tested SB3 configuration must not be represented as a verified Rust default. Issues #33/#35 remain open for a reliable held-out-qualified learning path.
