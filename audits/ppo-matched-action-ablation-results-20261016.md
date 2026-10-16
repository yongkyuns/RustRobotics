# PR #41 / issue #42 — matched action-distribution PPO: executed four-seed audit

**Date:** October 16, 2026  
**Disposition:** Both experimental PPO distributions learned from scratch on the native nonlinear plant. **Categorical PPO has a substantial advantage on difficult recovery**, and the continuous squashed-Gaussian reference is not as reliable. The production continuous-action PPO is **not** proven fixed; PR #41 stays draft/unmerged.

## Immutable, executable evidence

- Source commit: [`105b531cd431670bbea63edc1591df27f5cc7856`](https://github.com/yongkyuns/RustRobotics/commit/105b531cd431670bbea63edc1591df27f5cc7856).
- [Completed full action-space experiment #38917148306](https://github.com/yongkyuns/RustRobotics/actions/runs/38917148306), conclusion **success** (experiment *execution*, not policy-quality acceptance).
- [Native Rust validation #38917155170](https://github.com/yongkyuns/RustRobotics/actions/runs/38917155170), conclusion **success**; all **21 targeted tests**, strict Clippy and rustfmt passed.
- [Artifact #12054364162](https://github.com/yongkyuns/RustRobotics/actions/runs/38917148306/artifacts/12054364162) contains eight per-seed CSVs, raw training logs, summary and source provenance.
- All **eight 976-update** training budgets completed: 8 × **499,712 interactions** = **3,997,696** native environment transitions. All **28,672** individual evaluation records were validated: 2 action laws × 4 development seeds × 7 checkpoints × 4 reset panels × 2 inference modes × 64 keys. All eight recorded CSV SHA256 digests **independently match** the archived CSV bytes. The artifact retains every checkpoint and failed episode; no best-seed/checkpoint selection.

This is an already-exposed **development-seed** comparison. Training seeds 201–204 are independent RNG initialization streams, but evaluation reuses identical reset keys, and deterministic/stochastic evaluations are correlated. Pooling 512 observations does *not* mean 512 independent training trials or establish a population confidence interval.

## Frozen differences and invariants

**Common:** 64×64 tanh actor/critic; **two actor output units in both arms**; seeded actor/critic initialization; 4 persistent rollout environments × 128 steps; minibatch 128 × 4 epochs; 976 PPO updates; one persistent Adam epsilon `1e-5`; PPO policy/value clipping, GAE gamma `0.99`, lambda `0.95`; global gradient norm `0.5`, entropy weight `0.01`, linearly annealed LR from `2.5e-4`; timeout/failure boundary semantics; the same nonlinear RK4 plant, observation/actuation noise, rewards and 15-second evaluation panels.

**Action law differs by design:** categorical selects precisely `-20` or `+20 N` using two logits and categorical Shannon entropy; continuous interprets the same two output channels as Gaussian latent mean/log-standard-deviation, clamps logstd to [-2,1], and applies `20*tanh(z)` N. The stored latent `z` is used for replay likelihoods so the tanh Jacobian cancels in PPO ratios. Its entropy bonus is **pre-tanh Gaussian differential entropy**, not true squashed-force entropy. Therefore identical coefficients **do not imply equivalent effective entropy pressure or action magnitudes**, and trajectories diverge after sampling. This holds the training *recipe* fixed more closely than the previous production-vs-categorical comparison, but cannot isolate a unique physical cause.

## 976-update results: shared, difficult 64-key panel

15-second completions out of 64 for each policy and mode:

| Training seed | Categorical deterministic | Categorical stochastic | Gaussian deterministic | Gaussian stochastic |
| ---: | ---: | ---: | ---: | ---: |
| 201 | 63 | 63 | 54 | 53 |
| 202 | 63 | 63 | 56 | 55 |
| 203 | 63 | 63 | 59 | 58 |
| 204 | 64 | 63 | 48 | 44 |
| **Pooled** | **253/256** | **252/256** | **217/256** | **210/256** |

Combined across modes: **505/512 (98.63%) categorical** versus **427/512 (83.40%) Gaussian**, a **15.23 percentage-point difference** on these fixed correlated panels.

Shared ordinary panel: **512/512 categorical** versus **510/512 Gaussian** (seed-204 deterministic/stochastic each 63/64).

At 976 updates, for *each* (training seed, reset key, policy-evaluation mode) pairing:
- **427** outcomes: both policies completed.
- **78**: categorical completed but Gaussian failed.
- **0**: Gaussian completed but categorical failed.
- **7**: both failed.

The paired comparison is especially useful diagnostically because both policies face the same initial reset keys; it remains **development evidence**, not independent inference.

## Failure distribution and root-cause boundary

At the final checkpoint on common difficult keys:
- All **7 categorical failures** were the **same reset key `4100100100`**, spanning 7/8 seed-mode pairs.
- Gaussian had **85 failures across 20 distinct reset keys** (including `4100100100`). Five keys failed in *all eight* Gaussian seed-mode conditions: `4100100015`, `4100100021`, `4100100100`, `4100100170`, `4100100556`.
- All **85 Gaussian failures** and all **7 categorical failures** had **negative recorded minimum cart clearance**; the cart crossed the **±2.4 m rail boundary**. The CSV does not report final pole angle separately, so a simultaneous angle failure cannot be ruled out from this dataset alone.
- The median minimum rail clearance across all Gaussian common difficult episodes was **0.414 m**, versus **0.808 m** for categorical. The categorical controller used full `±20 N` commands, so stronger margins need not mean lower actuator effort.
- Most failing Gaussian trajectories crossed the rail after approximately **0.66–1.11 seconds**; an isolated failure survived 4.65 seconds. Exact per-step corrective actions/force magnitudes were **not recorded by this run**. Claims that insufficient early applied force or policy saturation *caused* these failures remain hypotheses pending trajectory probes.

## Training progression and stability

Difficult 15-second completion counts, deterministic + stochastic out of 128:

| Seed | Categorical at 512 / 768 / 976 | Gaussian at 512 / 768 / 976 |
| ---: | --- | --- |
| 201 | 126 / 127 / 126 | 42 / 108 / 107 |
| 202 | 126 / 126 / 126 | 0 / 109 / 111 |
| 203 | 124 / 126 / 126 | 17 / 108 / 117 |
| 204 | 126 / 127 / 127 | **96 / 72 / 92** |

The continuous arm **does learn**, including on difficult cases. But it learns later than the categorical arm for most seeds and has observable late-checkpoint instability. In seed 204, 24 previously successful seed/mode/key pairs at update 512 failed at 768, with **zero** reverse recoveries in that interval; 20 returned by update 976. That is a concrete non-monotonic learning trajectory, not proof of the earlier production's collapse mechanism.

At update 976 on the common difficult panel, Gaussian policies achieved a **higher average raw reward (1106.9) than categorical (800.4)** despite worse completion and mean survival (**1265.6 vs 1480.5** of 1500 steps). The same reward includes an **action² cost**, so reward and completion measure different objectives when action magnitudes differ; it would be incorrect to select the nominal winner from reward alone.

## Follow-up experiment and release boundary

1. **Targeted actuator/action trace:** on *frozen final* categorical and Gaussian policies, replay known discriminating development keys (e.g. `4100100015`, `4100100021`, `4100100170`) that pass categorical but fail Gaussian. Record per-step noisy observations, true cart/angle, **desired/applied force**, exact rail/angle termination causes, and transient recovery margins. Compare action amplitudes and polarity *without tuning on these already exposed keys*.
2. **Continuous-policy diagnosis:** assess whether failure is driven by mean-force amplitude, logstd/clamping/squash, entropy regularization or incomplete state/rollout coverage. Do not change more than one factor per registered experiment. Retain no-change baselines and unfavorable results.
3. **Compare against the actual existing production Gaussian PPO path**, not merely against this new experimental Gaussian reference, on matched physical conditions/budgets. The existing production algorithm has documented action-credit and reset-distribution weaknesses on #35; changing recipe and action law together cannot identify cause.
4. For #33 independent sustained-balancing acceptance, freeze new training seeds and evaluation panels plus effort/rail constraints **before** observing confirmation data. Four exposed dev seeds, and any chosen failing traces, cannot count as fresh confirmation. Full-network Torch↔Burn gradients/global clipping/Adam parity and production/browser/platform regressions remain outstanding.

**No merge, production trainer/default change, or hardware performance claim follows from this diagnostic.**
