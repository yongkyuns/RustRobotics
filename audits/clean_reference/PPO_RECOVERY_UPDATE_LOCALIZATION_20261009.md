# PPO recovery: update-level localization (406–464)

**Date:** October 9, 2026. **Status: completed post-hoc development diagnostic, not a corrective trainer or new held-out qualification.**

Registered **before new outcomes** at [issue #35 comment 6074131980](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6074131980). Follows [contemporaneous-policy cross-over](PPO_HISTORICAL_CROSSOVER_RESULTS_20261008.md), [literature review](PPO_LITERATURE_REVIEW_20261008.md), and issue #33 sustained-control acceptance. One original noisy continuous-force cart-pole history, training seed201, one retrospectively selected pair13 hard-reset witness. No intervention on the learner.

## Main findings

**Mean-action function, not exploration sigma.** At the actual later failed physical/noisy state, factor the two frozen learned policies into old/new mean and old/new scalar Gaussian sigma, with **the original32 paired future noise draws**, unchanged native plant/reward/failure and 1500-step horizon:

| Frozen mean | Sigma from incoming406 | Sigma from incoming464 |
|---|---:|---:|
| Incoming406 | **32/32**, mean gamma=.999 return614.482 | **32/32**,614.796 |
| Incoming464 | **8/32**,170.658 | **8/32**,170.750 |

The two learned sigmas are0.15981649 and0.15724076 normalized-force units (~1.6% apart). All24 failures in each later-mean arm hit the cart limit; swapping sigma changes neither completion count. Every original endpoint-policy physical/action/reward trajectory replays **exactly**. On the paired original timeout-source state all four combinations complete32/32. With **zero additional Gaussian action noise** but unchanged native disturbances, incoming406 completes15s while incoming464 fails at the cart limit after112 ticks. Both initially issue -20N, diverge from tick17; at tick100 the earlier policy has cart velocity magnitude0.877m/s versus1.529m/s for the later policy.

**An actual optimizer update seriously damages the same recovery state.** Full reconstruction captured **every one of59 consecutive historical incoming actor/critic policies 406…464**, i.e. **all58 actual adjacent PPO updates**. They were each evaluated from the *same two original pair13 physical/noisy states*, the *same32 Gaussian/plant seeds per state*, native 1500-tick physical stopping, and one additional no-policy-Gaussian case per state.

| Incoming policy | Success on original failed-state panel /32 |
|---:|---:|
|406|32|
|410|26|
|**411**|**7**|
|419|0|
|431|0|
|444|18|
|448|30|
|**449–460**|**32 at every update**|
|461–462|31 each|
|463|19|
|464|8|

This is substantial **non-monotonic** retention, not gradual uniform forgetting.

**Most harmful single adjacent update:** incoming **410→411** changes **26/32→7/32**, **19 losses/0 gains**, mean failure-source gamma=.999 stopped-return **−351.856896**. On the matched nearby timeout-source state completion remains32/32 and return changes−2.146673. The exact archived on-policy rollout u410 contains2048 fitting rows. Under the independent descriptive **whole-rollout-normalized** PPO clipped surrogate, the actual old/new actor change gives **+0.007616** (improvement), and old/new mean conditional latent-Gaussian KL on those sampled rollout states is **0.007264**. At the selected original failed-start observation, the conditional latent-Gaussian KL is **0.073655** (~10× larger). That is *not* a formal KL cap, and latent-action KL is not the same quantity as KL after many-to-one force clipping.

Only **29/2048** original update410 rollout rows satisfy the fixed instantaneous early-outward stratum (|theta|>=.15rad,|omega|>=.25rad/s,same sign,episode age<100ticks). Crucially, these 29 rows' **change** in full-rollout-normalized clipped surrogate from their own old-policy baseline is **+0.028245**—**positive**, not evidence of uniformly suppressing recovery-like sampled actions. Near-center stratum (|x|<=.25m,|theta|<=.05rad) has623 rows and analogous change+0.004967. These are *not* the actual minibatch-normalized SB3 losses; historical64-row permutations were not retained.

Across all58 adjacent updates, **49** have a positive whole-rollout clipped-surrogate change; **34** have lower actual mean paired return on this selected difficult reset; **29** have **both**. These are correlated updates within one previously exposed training history, not independent trained agents or an estimate of prevalence across tasks.

## Exact reconstruction and numerical controls

- Preserved AMD EPYC9V74, Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0, and original Rust native plant.
- Every one of **six** original actor/critic and raw evaluation-reward checkpoint records reproduced, all**512** original diagnostics and original **4,289 completed** episode-ledger rows reproduced exactly (numeric values, order and semantics; list/tuple representation normalized).
- Canonical all512-batch fitting+physical-packet SHA256 **`76e003ff39029b70faabe40c9f85aafba8880fb2c86500576637a2b0e063d0f9`**; selected reset/noisy-packet SHA256 **`d73d4a3f5142183f51352469eb220093c86abcca4c5eb948da99f487a8da5af1`**. Both passed.
- All previous32 actor tensors match element-for-element;59 consecutive incoming policies are captured. 87 unique policy files total. No policy was substituted from a later checkpoint, and there was no fake weight-only historical optimizer resume.
- Fully independent **NumPy-only** reader checks **60 groups /4,150 fixed-noise continuations /4,623,244 native transitions**, including bit-exact native rewards, early physical endings, 0 postterminal padding, original force mapping, full paired Gaussian innovations, gamma=.999 scores, and independent actor commands. Maximum action difference **3.0111e−7**, under3e−6. All **nine fixture/corruption tests pass**.
- PPO collected old log-probabilities are reconstructed with max absolute difference9.7584e−6 across58 rollouts (the original Torch float32 batched re-evaluation also differs by8.5831e−6 for that outlier). A separate **2e−5 log-probability representation check** permits that known float32 batching discrepancy only; it does not alter policy, reward, clipping or any result.
- The native scan used the exact historical **64-case stochastic policy-inference batch shape** plus separately evaluated deterministic cases after a one-ULP numerical mismatch arose from evaluating66 cases together. Original 406 and464 complete trajectories now pass exact equality. Failed preflight/earlier source and the stale source-hash receipt are retained; these were corrected as setup/accounting faults, not retried policies for success.

## Evidence / offline replication

**Complete package:** `ppo_recovery_update_localization_evidence_20261009_final.zip`, **234,709,074 bytes**, SHA256 **`f5f37761202e09faeebd6dfd51e785ee5f007bc7dd388d42c6c24ec5351eefec`**. 306 individually hashed payloads: all59 policy/continuation states and traces, factorization,58 original fitting rollouts, independent verifier, actual history receipt, numerical reports and executed source. A **fresh extraction** passed every payload hash, regenerated **all four numerical outputs byte-for-byte**, and passed nine corruption tests with `OPENBLAS_NUM_THREADS=1 python REPLAY.py`. No native simulator/Torch/PPO/optimizer or network is used for offline replay. Full native regeneration additionally requires separately pinned original binary/runtime archives.

**Compact source/report:** `ppo_recovery_update_localization_report_code_20261009.zip`, **97,514 bytes**, SHA256 **`dcb2708f70412e135ad2d2f2cfecfa47c552e8e9feed8bbbc75730cdfe80fd19`**. Contains summary, all58 contrasts, source/verifier/tests and receipts, **not the full trajectory data**.

### What is established, and what is not

This is a *true policy-version intervention* on two selected identical physical/noisy starts and fixed future innovations, and the largest contrast compares **actual adjacent policies** before/after one optimizer update. It is **not** evidence that the learned value tail is fictitious, that sampled PPO gradients are necessarily wrong on average, that the problem space is physically hard, or that changing the policy's mean globally reduces overall expected return. The classifier of the worst update is **post-hoc descriptive selection among all58**, not a preregistered single-update hypothesis. The whole-rollout clipped surrogate is not the original minibatch PPO training loss.

**Next bounded experiment:** freeze a broad-reset and a targeted hard-recovery *independent* stochastic confirmation panel around actual incoming policies410/411. Measure both overall expected-return improvement and conditional recovery deterioration, plus repeated noisy estimates of the old-data surrogate/gradient direction, without training or tuning. Do not reuse old held-out seed cohort830101–830108 for development or adopt a risk-weighted training rule before proving the mechanism.

**No master, production, default, deployed-policy or runtime changes. No merge. #33/#35 remain open.**
