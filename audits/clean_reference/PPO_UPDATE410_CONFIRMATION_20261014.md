# PPO update410→411: independent broad / recovery-reset confirmation

Date: October 14, 2026. Issues #33/#35 remain open. No production change, PPO correction, merge or deployment.

## Status

**All 3,072 planned trajectories are complete and independently checked. These are provisional compatibility-runtime measurements, not completion of the original pinned-runtime qualification.**

Primary protocol: issue35 comment6079265416, registered October9. Input recovery and exact post-reset RNG cloning: comment6172768446. Explicit local-runtime contingency, registered before any fresh outcome: comment6172899344. No sample size, key, actor, initial-state predicate, noise, horizon or statistical endpoint was changed after outcomes.

## Result

| Panel | Incoming410 completed | Incoming411 completed | Gained / lost | Mean gamma.999 stopped-return delta | Paired two-sided99% Student interval |
|---|---:|---:|---:|---:|---:|
| Broad native resets | 1,001/1,024 | 1,000/1,024 | 0 / 1 | -0.393654691 | [-1.815376941, +1.028067559] |
| Difficult conditional native resets | 402/512 | 392/512 | 0 / 10 | -11.952273921 | [-20.992304279, -2.912243563] |

The difficult-panel completion rate declines from78.515625% to76.5625%, or **1.953125 percentage points**. Every newly lost difficult case terminates at the cart-position boundary, not the pole-angle boundary, at ticks107–126. There are no newly gained cases in either panel.

The earlier selected-state result26/32→7/32 is not a population-wide collapse estimate. The fresh difficult panel indicates recovery-region harm beyond that one selected reset. The broad panel, however, **does not resolve the sign of the overall expected return effect**. It would be incorrect to claim that both panels show resolved negative effects, that the global true policy gradient is wrong, or that a PPO correction has been established.

A prespecified descriptive split of the broad panel provides context:110 difficult initial states yield89→88 completions and mean return delta-5.904696653 (99% interval[-19.320159348,+7.510766042]); the other914 initial states yield912→912 completions and delta+0.269599812 (interval[+0.225448639,+0.313750985]). This is consistent with small improvements on common/easier starts coexisting with a recovery-region regression. The broad total remains unresolved. The two separately sampled panels must not simply be pooled or assigned equal population prevalence.

The intervals condition on the two fixed policies and independent paired reset/noise cases. They do not quantify variability across independently trained agents. These are the declared per-panel intervals, not a familywise analysis or a post-hoc replacement endpoint.

## Measurement contract

- Actual historical incoming policies410 and411, development training seed201. No fitting, supplied teacher, interpolation, rollback or retraining.
- Broad keys:3209102600+i, i=0…1023, original native evaluation reset law.
- Difficult keys:3309102600+j, first512 accepted original native resets in ascending order. True initial predicate:abs(theta)>=.15, abs(omega)>=.25, theta*omega>0. 5,264 keys were considered; final accepted key3309107863. All considered states, observations, reset packets and predicate decisions are retained.
- Both reset panels were frozen before policy trajectories. Original native `rr_env_create(key,1500,0)` then `rr_env_reset`; exact whole-environment/post-reset RNG clones for paired actors. No post-reset reseeding substitution. Original and additive bridge reset packets matched on all6,288 considered resets.
- Original learned sigma per actor, paired standardized innovations from `default_rng(key XOR 0xCE135A79).standard_normal(1500).astype(float32)`, original observed-state noise/disturbed actuation/clipped force mapping.
- Exactly64 inference lanes per actor/group, including inactive lanes. No shrinking batch after termination. First physical failure or1,500-tick horizon, not the short training timeout. Sequential float64 gamma.999 accounting of raw float32 rewards, no endpoint critic bootstrap.
- All24 groups executed once, all3,072 trajectories retained, **4,220,289 fresh native transitions**, zero optimizer transactions or new training rollouts. Full trajectories, endings, early25/50-tick behavior and original100-tick centering metrics retained.

## Runtime and validation

The exact prior localization archive was recovered (SHA256 `f5f37761202e09faeebd6dfd51e785ee5f007bc7dd388d42c6c24ec5351eefec`):306 payload hashes pass, four original numerical reports regenerate byte-for-byte, nine prior corruption tests pass. The recovery-credit parent (`ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412`) passes all648 payload hashes. Its original native ZIP remains `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`.

The original three-day-retention pinned runtime artifact11441687522 now returns404. Runtime-only recovery run38595509112, commit01ae4b7a59fcddd7074d2e38fb036eb35ad08cbf, was still queued at the last check. It requests Python3.11.14, Torch2.8.0+cpu, NumPy2.2.6, SB3 2.9.0, Gymnasium1.3.0; no training is requested.

Fresh measurements instead used the explicitly registered provisional adapter: **Python3.13.5 / Torch2.10.0+cpu / NumPy2.3.5**, with no SB3 dependency, the original frozen actor parameters, the same two64-unit tanh layers and scalar Gaussian head, and the unchanged native library. Before new outcomes, it reproduced **every array byte of all132 archived410/411 trajectories**, preserving original64-stochastic/2-deterministic inference shapes:153,438 native transitions. An additional522 native transitions checked original/additive bridge and post-reset RNG clone equivalence. These controls are not a substitute for the original pinned-runtime/source gate.

The new independent NumPy/SciPy reader uses no Torch, SB3, simulator, optimizer or pickle. It checks full panel/key/group coverage, initial selections, actor identities, Gaussian innovations, post-terminal padding, first physical endings, force mapping, raw float32 reward bits, observation bounds, sequential scores and independent float64 actor commands. Maximum normalized-command discrepancy: **2.713064669590537e-7**, below the inherited3e-6 arithmetic tolerance. All25 new genuine-fixture/corruption tests pass. **Both new numerical reports regenerate byte-for-byte from the full package.** The reader does not independently reintegrate the plant or regenerate private native RNG draws; native provenance and exact controls cover those components.

A shell frontend timeout occurred between completed broad groups; existing completion receipts were inspected before proceeding, and no fresh group was repeated. A read-only full-reader invocation timed out before producing its report and was repeated with a larger tool timeout; no simulation or optimization was repeated in that retry. These events are retained in EXECUTION_NOTES.json.

## Reproduction and next gate

The strict `confirm.py` entry point remains version-gated; `provisional.py` is separately labeled. `pinned_runtime_qualified` is **false** in the measurement/verification reports. The included `compare_pinned.py` requires a completed strict run and byte-compares all fresh selection and trajectory arrays. Downloading a runtime package alone does not qualify these results. Same-key deterministic replay is a compatibility check, not another independent sample; mismatches must be retained, not replaced by favorable outcomes.

The full evidence package contains all raw data, source, original required native/source/checkpoint subsets, manifests,25 tests and `REPLAY.py`. The smaller report/code package contains the reports, source, original actor weights, archived physical fixture, tests and provenance, but omits the fresh raw trajectories. The actual executed source is retained in these packages, not implied to have been installed into the production trainer. Native/source subsets preserve original manifests and document omitted historical files. A byte-identical alias restores the historical selection filename expected by the unchanged old runner; it does not alter inference, states or outcomes.

Artifacts delivered with this continuation:

- `ppo_update410_confirmation_evidence_20261014.zip` —202,193,130bytes; SHA256 `717d6f72d39dca2e16afd922911b58f51ae914ced10c15282d034f272a63cefb`.
- `ppo_update410_confirmation_report_code_20261014.zip` —7,254,316bytes; SHA256 `2c62bed244215e50dd09a05c2bb2882a12901f3558c0f75f7022678c0dd6ec2a`.

Read-only reproduction: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python REPLAY.py`, using the recorded reader environment Python3.13.5 / NumPy2.3.5 / SciPy1.17.0. Tests alone: `python -m unittest discover -s code -p test_verify.py -v`.

The next substantive mechanism question is **finite step size versus estimated direction in the recovery region**. This result does not choose between them, justify a new hyperparameter sweep, or qualify a correction. Complete the explicit runtime gate before upgrading the result's status. Any eventual correction still requires separate sustained-learning and task-level qualification under #33.

Master remains8514317e3b5cd57dc37bee2aa4c950af674751e1. No protected user-owned files, production defaults, reward/reset/noise laws, held-out training seeds830101–830108, production PRs, master or deployment were changed.
