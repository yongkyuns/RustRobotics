# Scaled PPO: 2048 vs 8192 unique samples per update — October 7, 2026

Development experiment for [RustRobotics #35](https://github.com/yongkyuns/RustRobotics/issues/35), under the **pre-registered protocol in [comment 6038398543](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6038398543)**.

**Decision: the pre-registered exposed-development screen PASSES, but the evidence does not establish reliable superiority, faster learning, or production readiness.**

## Source and reproducibility

- [Completed GitHub Actions run 37660544338](https://github.com/yongkyuns/RustRobotics/actions/runs/37660544338) at executed head `0f41bb0703b4904b400e644b0c120ea0017016e3`; historical gate plus all 4 paired seed jobs completed successfully.
- Five immutable ZIP artifacts:
  - historical gate: `11502840847`, SHA256 `92c03e00b9636197c8d4b050fab783379c274b3faf90faa648fabb1fe8f0203d`;
  - seed 201: `11504926318`, SHA256 `11d9efaeb4225ef70449f7672bc72accc0a0001bffcf3d2f31a814ba25c2b4a2`;
  - seed 202: `11505538213`, SHA256 `1cd8601420abe822cbd8749eaaaa94a44412464a9dcfa2789e128cae1cad0d9e`;
  - seed 203: `11504253927`, SHA256 `77fc7c9fdadcb2e2107d8aa77bdd51e7f75277e5b1108ac803c0728ea3577a4f`;
  - seed 204: `11504407281`, SHA256 `74003b46345bc01384b802630972ede886684973236921225cc3c42a44c1fa3a`.
- Independent Python-only offline reducer verified **all 5 outer SHA256 digests, all 3,511 inner manifest payload hashes, and all 4,608 evaluation records**, including both discount factors, raw reward traces, exact per-case pairing, recorded ending categories, diagnostic progression and receipts.
- The CI historical control checks reproduce **published completion counts** on the inherited fixed evaluation domain; these are NOT byte-exact reference-policy replay gates. The distinction is retained.
- Earlier first-run SyntaxError and preflight-only fixes are preserved in [comment 6043406761](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6043406761). No favorable result was chosen or retried.

## Controlled intervention

Both arms use the same native pendulum, reward/noise/reset law, seeded initialization, value scale `S=1/(1-.999)`, gamma `.999`, GAE lambda 1, 256-step training timeout, batch 64, 10 epochs, lr 3e-4, clip .2, vf_coef .5, ent_coef 0, global grad norm .5, and the same actor/critic architecture, optimizer, and evaluator.

Only `n_steps` changes:

| | scaled-2048 | scaled-8192 |
|---|---:|---:|
| Unique samples per PPO update | 2,048 | 8,192 |
| PPO updates per seed | 512 | 128 |
| Training interactions per seed | 1,048,576 | 1,048,576 |
| Adam minibatch transactions per seed | 163,840 | 163,840 |
| Gradient-sample presentations per seed | 10,485,760 | 10,485,760 |

All initial actor and critic weights match per pair, checkpoint-zero evaluation matches, and all eight full measurement receipts satisfy the matched budgets.

## Pooled held-at-checkpoint outcomes

Each checkpoint aggregates 4 independently trained **exposed development seeds**, not 256 independent trained policies. Deterministic panels have 32 episodes/seed, stochastic 64/seed. Evaluations are physical-noise/failure-stopped.

| Training interactions | 2048 deterministic /128 | 8192 deterministic /128 | 2048 stochastic /256 | 8192 stochastic /256 | Mean paired stochastic return change (gamma=.999) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 131,072 | 128 | 127 | **249** | 238 | **-67.558** |
| 262,144 | 126 | 128 | 249 | **253** | **-62.340** |
| 524,288 | 126 | 128 | 248 | **253** | -4.315 |
| 786,432 | 127 | 127 | 252 | **253** | +2.811 |
| 1,048,576 | 126 | **128** | **252** | **252** | **+1.398** |

The candidate lags materially in early stochastic completion and especially in gamma-.999 return. By 524k it catches up in completion; at the final checkpoint it ties stochastic completion. Thus a larger batch does **not** establish faster learning or across-training dominance.

## Final checkpoint by trained seed

| Seed | 2048 det/stoch | 8192 det/stoch | Paired mean stochastic gamma-.999 return change |
|---:|---:|---:|---:|
| 201 | 30/32 · 62/64 | **32/32 · 63/64** | +15.595 |
| 202 | 32/32 · 63/64 | **32/32 · 62/64** | **-16.940** |
| 203 | 32/32 · 64/64 | 32/32 · 64/64 | +3.082 |
| 204 | 32/32 · 63/64 | 32/32 · 63/64 | +3.856 |
| **Pooled** | **126/128 · 252/256** | **128/128 · 252/256** | **+1.398** |

Paired case identities on the final stochastic panel show **one lost success and one gained success** across all seeds (not necessarily same seed).

The candidate meets every seed's final >=31/32 deterministic and >=61/64 stochastic thresholds; at recorded checkpoints >=262,144, once a seed clears both thresholds it never falls below either. Final pooled stochastic completions tie the control. Mean across-seed gamma-.999 endpoint return difference is positive. Hence **all four fixed development-screen gates pass**.

But mean +1.398 has a **descriptive across-training-seed two-sided 99% Student interval [-38.027,+40.823]** (n=4); it is neither precise superiority nor equivalence evidence. Do not pool evaluation cases as independent trained-agent samples.

## Numerical conditioning and interpretation

At the final PPO update, average second-hidden-layer tanh saturation `|a|>.99` is **9.44% (2048) vs 12.29% (8192)**. Neither resembles the original severely saturated raw critic, and **0%** of recorded targets exceed the current value-head ceiling in either arm at the endpoint. This remains a conditioned-value experiment.

The larger batch is useful because all 4 exposed seeds pass the registered preservation threshold. But it has **no pooled stochastic-success improvement**, an imprecise positive endpoint return effect, and significant early checkpoint deficits, so no production/default correction, merge, or sustained-reliability claim is justified. A new, preregistered held-out training cohort with stronger multi-checkpoint and long-horizon screens is the appropriate next independent qualification before any default decision.

## Verification limits

The independent offline reducer replays immutable numeric evidence without any new simulation/training/optimization; it does **not** reconstruct unexported future RNG draws or native trajectory dynamics. The original historical policy artifacts are not available for byte-exact old/new optimizer/policy equality against the prior study, so the historical checks are outcome-count reproduction only.

No seed, checkpoint, failed outcome, acceptance rule, optimizer configuration, reward/noise/reset law, production default, or master branch was changed to obtain this result. **Issues #33 and #35 remain open.**
