# Held-out scaled PPO 8192 versus 2048 — October 7, 2026

**Final disposition: reliability screen PASSES; return superiority FAILS; early-learning screen FAILS. Overall held-out candidate qualification FAILS.** No default/master/deployment change or merge.

## Protocol and execution

Pre-registered *before* new training outcomes in [issue #35 comment 6045659158](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6045659158). The prior exposed-development result passed its own weaker screen but did not show precise superiority: [development report](https://github.com/yongkyuns/RustRobotics/blob/audit/value-batch-8192-20261007/audits/clean_reference/VALUE_BATCH_8192_RESULTS_20261007.md).

- Fresh paired training seeds: **830101–830108**; no omitted seed, no best checkpoint, no outcome-driven retry.
- [GitHub Actions run 37677984741](https://github.com/yongkyuns/RustRobotics/actions/runs/37677984741), executed head **f94d0d667337c8e08b8fe7bb765c0b44221e7656**; historical gate and all eight paired jobs completed successfully.
- Frozen stock PPO/scaled-value experiment source blob **505e87f38794aae47545927f8f47906e93a9f7aa**, verified native ZIP SHA256 `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`, pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0.
- Both arms/seed: **1,048,576 training interactions**, **10,485,760 gradient sample presentations**, **163,840 Adam minibatch transactions**. Same random seed/initial weights and native plant/reward/observation/action noise, episode 256 timeout, critic scaling S=1/(1-.999), gamma .999, GAE lambda 1, optimizer, actor/critic model, 64 minibatch size and 10 epochs/sample.
- Only experimental dimension: **2048 unique on-policy transitions/update (512 updates)** versus **8192 (128 updates)**.
- Fixed paired evaluation: **32 deterministic 1000-tick plus 64 stochastic 1500-tick** per seed/arm at 0, 131072, 262144, 524288, 786432, 1048576 interactions. Native physical-failure stopping. Evaluation always uses raw physical rewards; main registered return gamma .999, legacy score gamma float32(.99).

## Independent offline verification

A separate reader checked the **9 outer ZIP SHA256 digests, all 5,408 payload hashes, all 9,216 saved evaluation episodes**, full native reward-trace replay at gamma 1 / legacy .99 / protocol .999, unique paired case identities, every summary and ending count, per-update diagnostic sequence, per-seed exact optimizer/interaction budgets, initial weights, and checkpoint-zero equality. The reader uses no native simulator, PPO or optimizer code. It confirms the historical gate reproduces the published **completion counts**, not prior actor/optimizer bytes.

The 9 immutable GitHub Actions artifact IDs (gate then all eight seeds) are:
`11508797419`, `11510448663`, `11510832759`, `11511527839`, `11509628820`, `11512430007`, `11509628995`, `11511224650`, `11509587316`.
All outer and member digests are recorded by artifact in the independent `VERIFIED.json`.

A read-only verifier correction changed only a SciPy-derived NumPy scalar to a Python float for JSON serialization; the earlier attempt completed source verification but failed before writing final output. No measurement or acceptance rule was changed. Reproducing this offline reducer is **not** a rerun of any training or evaluation.

## Pooled checkpoint outcomes

Each row pools **eight independent trained agents** with fixed paired evaluation episodes; pooled episode counts are not eight times more independent training seeds.

| Interactions | Control deterministic /256 | 8192 deterministic /256 | Control stochastic /512 | 8192 stochastic /512 | Mean paired gamma-.999 stochastic return difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0.000 |
| 131,072 | 255 | 250 | **503** | 485 | **-65.540** |
| 262,144 | 256 | 256 | 508 | 507 | **-84.387** |
| 524,288 | 256 | 256 | 510 | 509 | -15.303 |
| 786,432 | 254 | 256 | 502 | 509 | +9.258 |
| 1,048,576 | 255 | 255 | **507** | **507** | **+4.025** |

The candidate is materially worse early. Its **131,072 stochastic completion loss is 18/512**, violating the predeclared allowance of at most 8/512. Return lags are still large at 262,144 interactions even though completion counts nearly tie. At the endpoint stochastic completion is exactly tied.

## Final endpoint by independent training seed

| Seed | 2048 deterministic /32 | 8192 deterministic /32 | 2048 stochastic /64 | 8192 stochastic /64 | Paired gamma-.999 mean return change |
|---:|---:|---:|---:|---:|---:|
| 830101 | 31 | 32 | 63 | 64 | +26.922 |
| 830102 | 32 | 32 | 63 | 64 | +16.108 |
| 830103 | 32 | 32 | 64 | 64 | +3.912 |
| 830104 | 32 | 32 | 64 | 64 | +1.810 |
| 830105 | **32** | **31** | **64** | **62** | **-20.733** |
| 830106 | 32 | 32 | 62 | 62 | +0.503 |
| 830107 | 32 | 32 | 64 | 64 | +4.433 |
| 830108 | 32 | 32 | 63 | 63 | -0.753 |

Candidate stochastic pairing at endpoint includes **two lost and two gained** completions, both losses at 830105 and gains at 830101/830102, net zero. All contrary results retained.

Across the 8 independent training-seed mean paired stochastic returns (not 512 treated as independent trained policies): **+4.025365**, SD **13.750422**, SE **4.861508**, preregistered two-sided **99% Student interval [-12.987402, +21.038133]**. The lower bound is not positive. This fails the required superiority screen; it neither establishes superiority nor equivalence.

## Predeclared screens

| Screen | Outcome | Evidence |
|---|---|---|
| Endpoint reliability, each of 8 candidate seeds >=31/32 deterministic and >=61/64 stochastic | **PASS** | all eight |
| Continued-training preservation once both thresholds reached at a checkpoint >=262144 | **PASS** | no recorded post-qualification breach |
| Endpoint pooled stochastic >= paired control | **PASS** | 507/512 versus 507/512 |
| **Reliability composite** | **PASS** | all three above |
| **Independent stochastic-return superiority** | **FAIL** | 99% lower bound -12.987 |
| **Early-learning nonregression** (at 131072 and 262144, no more than 8/512 below control) | **FAIL** | -18 at 131072; -1 at 262144 |
| **All screens required for advancement** | **FAIL** | 2 of 3 composites not met |

The critic remains numerically far better conditioned than the original raw value-learning failure at the final update: mean second-hidden tanh saturation (|activation| >.99) across eight seeds is **13.07% control versus 10.30% candidate**, and **0%** of final update targets exceed the current value-head ceiling in either arm. Low saturation does not prove accurate conditional continuation value estimates.

## Interpretation and next work

The outcome challenges the claim that reducing rollout gradient variance by quadrupling unique samples is sufficient to improve general reliability or learning efficiency. The larger batch leads to fewer policy improvements per environment transition and different freshness/credit dynamics; this study **does not isolate** which factor causes early deficits. The fixed-budget comparison does not establish that larger batches are universally worse or that the 2048 arm is production qualified.

**Do not promote 8192 as the new default.** Keep the existing development baseline and archived candidate unchanged. Further investigation should return to the 2048-step update cadence and directly test an explicitly preregistered variance-reduction or policy-step reliability mechanism on development-only data, rather than sweep 4096/16384 or reuse these now-exposed held-out seeds for tuning. Any potential fix still needs separately registered long-horizon and task-level qualification under #33.

No production/master modifications or merge, and no claim of indefinite balancing, real-world robustness, or hardware qualification. Issues #33 and #35 remain open.
