# Fresh from-scratch recovery qualification: gains confirmed, robustness failed

## Decision and status correction

**The completed fresh cohort verifies substantial learning gains from random initialization, but FAILS its original absolute simulator-robustness criteria.** The recovery-data candidate completes 502/512 five-minute deterministic trials, 504/512 five-minute stochastic trials, and 435/512 sixty-second outward-recovery trials. Every remaining final candidate failure is a cart-position-limit failure. No production change or deployment is promoted.

This verification recovered a study already executed in GitHub; it did not start or retry training. The previous conversation's preparation-only status did not account for this separately registered experiment. The unexecuted preparation package described 24 runs with seeds **49001–49008**. This report instead concerns the **32-run cohort using 41001–41008**, registered on September 19, 2026 at 15:49:06 UTC and executed from 15:52:51 to 16:21:40 UTC. The two protocols have different criteria and must not be substituted or conflated. No duplicate cohort was launched during verification.

- Registration before execution: [issue #35 comment 5743221035](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5743221035).
- Executed source: `123a14ede4ce3a7e33f85e0c36781b353079ba89`, [run 35453178361](https://github.com/yongkyuns/RustRobotics/actions/runs/35453178361), first attempt, all 33 jobs successful: preflight plus 32 measurements.
- Evidence-preservation run `35453720451` only downloaded existing artifacts; it did not rerun a policy.
- Production baseline remains PR38 `4739f370558b9443708c920ac30614f86e3c07bb`; the inherited recipe is the previously tested repeated-recovery implementation.

Successful execution is not successful qualification. PR38 remains draft/open/unmerged. This report changes no production source, supported lambda 0.95, master, or deployed asset.

## Fixed study

All eight seeds train four configurations directly from `PpoTrainerSession::new_seeded`, with no checkpoint import, warmup, supplied controller, or pretrained critic. Initial actor/critic bytes and all initial evaluation records match across arms for each seed.

| Arm | Lambda | Fitting and optimization |
|---|---:|---|
| baseline95 | 0.95 | Ordinary supported PPO, 512 rows, batch128, four epochs |
| ordinary | 1 | Ordinary PPO, same small-batch schedule |
| near-union | 1 | Mean-four sampled cutoff future plus 512 near-state reward-labelled rows; 1,024-row union, batch256, four epochs |
| recovery-union | 1 | Identical union recipe, supplemental streams start from ordinary task resets/recovery states |

Every union adds eight 576-step frozen-old-policy streams, fitting only their first 64 rows and using the remaining rewards as future support. True-terminal paths remain terminal; live cutoffs bootstrap; timeout lookahead starts at the pre-reset endpoint. Failures are not filtered. Raw advantages normalize once over the union. The prior KL imitation penalty and extra critic epochs are inactive. Networks receive only existing noisy observations.

The corrected nonlinear RK4 plant, two 64-unit ReLU layers, learning rate 0.0003, Adam epsilon 1e-5, gamma 0.99, fixed exploration, force transformation, rewards and noise remain unchanged. Each run executes **4,096 updates**, **2,097,152 main interactions**, and **65,536 actor plus 65,536 critic Adam steps**. Checkpoints 0/256/1024/4096 are retained; only 4096 qualifies the final policy. Equal Adam steps do not imply equal data or compute.

Every checkpoint has 64 episodes in each short panel: noisy deterministic ordinary reset, noisy stochastic ordinary reset, and stochastic outward starts, capped at 20.48 seconds. Final evaluation additionally has 64 five-minute deterministic nominal, 64 five-minute stochastic nominal and 64 sixty-second stochastic outward trials per seed/arm. Keys use seed + 0x01000000 and disjoint panel tags. Different panels are not extensions of the same episodes. Failures terminate trials immediately.

Outward starts use cart displacement 0.8–1.2 m, same-sign outward velocity 0.4–0.8 m/s, pole angle ±0.15 rad and angular velocity ±0.3 rad/s. These stress outcomes are not training labels. Centred/upright survival additionally requires abs(x) ≤0.5 m and abs(theta) ≤0.1 rad throughout the final ten seconds, not necessarily the entire episode.

## Final sustained results

| Policy | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Ordinary PPO, lambda 0.95 | 290 | 285 | 200 |
| Ordinary PPO, lambda 1 | 309 | 294 | 162 |
| Matched near-state union | 361 | 354 | 285 |
| **Recovery-data union** | **502** | **504** | **435** |

Every candidate long-panel survivor also meets the final-ten-second centring/upright condition. However, **10 deterministic nominal, 8 stochastic nominal and 77 outward trials fail**, all at the position limit. Survivor-only centring cannot remove failures from the reliability denominator.

| Candidate seed | Five-minute deterministic /64 | Five-minute stochastic /64 | Sixty-second outward /64 |
|---|---:|---:|---:|
| 41001 | 64 | 62 | 62 |
| 41002 | 60 | 63 | 41 |
| 41003 | 63 | 64 | 59 |
| 41004 | 63 | 63 | 59 |
| 41005 | 63 | 63 | 47 |
| 41006 | 63 | 63 | 49 |
| 41007 | 62 | 62 | 60 |
| 41008 | 64 | 64 | 58 |

The worst recovery seed completes only 41/64 (64.1%). No candidate passes all panels. Rail containment remains a measured limitation; these outcomes do not uniquely establish which critic estimate, update, or physical recovery margin causes every failure.

## Original qualification gate: FAIL

The actual registration requires, for EACH final long panel: pooled completion ≥99%; every seed ≥95%; a one-sided Student lower bound over eight seed completion proportions ≥95%, with alpha = 0.01/3 for the three-panel family; and ≥99% of survivors satisfying the final centring condition. Missing runs or failed invariants also preclude qualification.

Pooled 99% requires at least **507/512** successes; each-seed 95% requires at least **61/64**.

| Panel | Pooled completion | Worst seed | Registered family-adjusted lower bound | Missed conditions |
|---|---:|---:|---:|---|
| Deterministic nominal | 98.047% | 93.750% | 95.352% | Pooled completion; weakest seed |
| Stochastic nominal | 98.438% | 96.875% | 96.848% | Pooled completion |
| Outward recovery | 84.961% | 64.062% | 68.914% | Pooled completion; weakest seed; lower bound |

All centring-among-survivors conditions pass, but all three absolute panel gates fail. No threshold is relaxed, no seed is excluded, and no earlier checkpoint rescues qualification. These Student bounds are approximate across-training-seed uncertainty measures, not hardware or distribution-free safety guarantees.

## Comparative gains on the new cohort

The candidate improves BOTH final short deterministic mean return and stochastic discounted mean return over all three controls on **every one of the eight fresh training seeds**.

| Policy | Short deterministic complete /512 | Deterministic mean | Short stochastic complete /512 | Stochastic discounted mean | Short outward complete /512 |
|---|---:|---:|---:|---:|---:|
| Ordinary lambda 0.95 | 315 | 1255.398 | 313 | 79.884 | 190 |
| Ordinary lambda 1 | 313 | 1273.172 | 311 | 80.179 | 143 |
| Matched near-state union | 370 | 1480.442 | 359 | 79.538 | 273 |
| Recovery-data union | 504 | 1986.075 | 501 | 86.166 | 434 |

Predeclared secondary intervals use eight paired training-seed differences, df7, NOT 512 independently trained agents. They are pointwise 99% intervals, not a simultaneous six-comparison guarantee.

| Candidate minus comparator | Deterministic mean effect [99% interval] | Stochastic discounted effect [99% interval] |
|---|---|---|
| Ordinary lambda 0.95 | +730.677 [+404.243,+1057.111] | +6.282 [+2.527,+10.038] |
| Ordinary lambda 1 | +712.903 [+483.209,+942.596] | +5.988 [+3.925,+8.051] |
| Matched near-state union | +505.632 [+210.687,+800.577] | +6.629 [+1.647,+11.611] |

The warm-start gain therefore generalizes to learning from scratch on this cohort. Improving a weak comparator is still different from meeting an absolute operational reliability requirement.

Candidate short completion trajectories at updates 0/256/1024/4096 are deterministic 0/0/396/504, stochastic 0/0/406/501, and outward 0/0/236/434, each out of 512. Earlier contrary results remain: at update256 seed41003 trails near-union on both return scores and seed41008 trails on stochastic discounted return. At1024 and4096 all eight candidates lead all three controls on both scores. Four checkpoints do not prove every intermediate update improves or that indefinite training preserves these gains.

## Verification and limits

Preflight passes **92 native unit/audit tests**, seven ordinary balancing controls, nine ordinary learning/evaluator controls, strict Clippy, formatting and protected-source restoration. Six audit endpoints remain ignored during preflight; the new fresh endpoint is explicitly run for measurements. Two separately ignored historical heavy integration tests were not run. The production browser/platform matrix was not rerun for this audit-only study.

Raw preservation artifact **10587737569** has SHA256 **9bf160c3d9b54bfc3b9819b7b0f2c8bf55b65cc4f82ec75bbc58946bf170ac36**. Independent verification checks all39 outer payload hashes, all33 original ZIP digests and **6,781 original payload hashes**. Original job/arm/seed lists are complete and first-attempt only.

Checks cover **30,720 evaluation records**, **131,072 update records**, and **128 regular actor/critic checkpoint pairs**: exact initial networks and initial episode records across arms, source/build binding, declared recipe and budget, evaluation keys, finite fields, ending labels and counted failures. Native tests cover persistent Adam, ordinary-loop equivalence, sampling/evaluation isolation, and terminal/timeout semantics.

Detailed reconstruction covers predefined updates1/256/4096 for each run:49,152 primary rows,221,184 supplemental return-support rows,and1,536 optimizer transactions. Sequential-f32 GAE/raw advantages/global normalization reproduce bit-for-bit. Maximum independent discrepancies are4.741e-8 actor loss,1.759e-4 value loss,and6.173e-7 post-step means, within retained tolerances. Selected optimizer endpoints at256/4096 match regular checkpoint bytes.

All **288 predetermined full evaluation traces (1,727,387 transitions)** reconstruct nonlinear dynamics, action transformation, rewards, endings, force RMS and final-ten-second centring. Max errors: dynamics2.352e-7,actor inference4.341e-7,command2.039e-6,reward2.234e-7,episode-return totals9.095e-13. Other episodes retain aggregates, not complete traces. No independent replay of every unexported gradient, Adam moment, RNG sequence or supplemental physical transition is claimed.

The offline verifier passes **41 tests** covering corruption/missing/unlisted members, invalid or duplicated outcomes, wrong domains, nonfinite values, survivor/failure handling, pooled averages hiding weak seeds, exact registered statistical rules and numerical boundary controls. It uses Python/NumPy/SciPy, does not execute the archived native runner, and does not train. Measurement used Rust1.98.1/Burn0.20.1 with the production lockfile. No numerical tolerance, measured field or acceptance threshold was changed to obtain the result.

## Costs and disposition

Across32 runs:67,108,864 main interactions,118,429,387 sampled-tail interactions,and301,989,888 supplemental interactions: **487,528,139 training-data interactions**. Evaluation adds **109,849,004**. Actor and critic each execute2,097,152 Adam steps and402,653,184 gradient-sample visits. Generic preflight and additional inference are extra and not fully aggregate-instrumented.

The candidate averages **28,607,277 training-data interactions per seed**, versus2,097,152 for ordinary PPO; matched near-union averages28,139,436.375. Equal main counters and Adam steps are not equal sampling cost. No sample-efficiency superiority over ordinary PPO is established.

**Retain the exact recipe as a genuinely improved from-scratch learner, but do not call it robust enough for unattended or hardware use.** The 77/512 outward failures and weakest-seed41/64 outcome fail the registered qualification. If the recipe is changed using these outcomes, this cohort becomes development evidence and a later confirmation must be newly registered.

Complete downloadable evidence is named `rustrobotics-ppo-fresh-cohort-verified-evidence.zip`; numerical tables and the independent verifier are also supplied separately. It contains the unchanged raw archive collection and exact sources, all current numerical reports, verification/test logs, and offline replay instructions. No duplicate training, default promotion, PR38/master/deployment modification or merge occurred during this verification.