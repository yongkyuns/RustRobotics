# PR #41 / #42 — Gaussian output-force gain during PPO training: completed experiment

**Date:** October 16, 2026. **Result:** Fixed-gain changes to the physical action mapping during **both training and evaluation** improved the experimental from-scratch Gaussian PPO's difficult-reset survival at 976 updates. The benefit was **smaller than applying an output gain only after training**, and hard resets still failed. This is exposed **development** evidence, not new-seed confirmation or a repair of production continuous PPO.

## Immutable evidence and controls

- Source SHA: [`c5a20cd701829f043ad75af2f8f26eb55b7174d4`](https://github.com/yongkyuns/RustRobotics/commit/c5a20cd701829f043ad75af2f8f26eb55b7174d4).
- [Full fixed-budget workflow #38979375155](https://github.com/yongkyuns/RustRobotics/actions/runs/38979375155): **SUCCESS** including formatting, **23 targeted Rust tests**, strict Clippy, release compilation, 12 completed training runs, full record verification and artifact upload.
- [Separate native Rust CI #38979380523](https://github.com/yongkyuns/RustRobotics/actions/runs/38979380523): **SUCCESS**.
- [Full artifact #12080975648](https://github.com/yongkyuns/RustRobotics/actions/runs/38979375155/artifacts/12080975648): all original per-seed CSVs, logs, immutable source provenance and generated summary.
- **12 × 976 PPO updates = 11,712 updates, 5,996,544 native physical training interactions**; **43,008 episode records** (3 gains × 4 exposed training seeds × 7 fixed checkpoints × 4 panels × 2 evaluation modes × 64 reset keys).
- **Hard baseline reproducibility:** The gain=1.0 CSVs, with only the first field renamed back to `distribution=squashed_gaussian`, are **byte-for-byte SHA256-identical** to the four prior immutable Gaussian full-budget CSVs from [action-space experiment #38917148306](https://github.com/yongkyuns/RustRobotics/actions/runs/38917148306). This covers all 7 checkpoints and every episode, not merely aggregate scores.
- All **12 actual CSV SHA256** digests in the artifact provenance were independently recomputed from the downloaded files and matched. Every unsuccessful episode and intermediate checkpoint was retained.

## What the intervention held fixed

The same squashed-Gaussian latent mean/logstd head (2 outputs), 64×64 tanh actor and critic, persistent Adam, PPO likelihood and clipping, GAE, 4 environments × 128 steps, minibatch/epoch schedule, reward/effort cost, physical plant, observation/actuation noise, timeout handling, 976-update training budget and predeclared 1500-tick panels were used.

The **sole intervention** was the physical force mapping of a sampled latent z:
- **1.0× baseline**: `20*tanh(z)` N;
- **1.5×**: `clamp(1.5 * 20*tanh(z), -20, 20)` N;
- **2.0×**: `clamp(2.0 * 20*tanh(z), -20, 20)` N.

The same gain was used in **training rollouts and checkpoint evaluations**, rather than changed after training. PPO ratios remain defined on the sampled **latent** z; because output clipping is many-to-one, this is **not** a density for the physical commanded force or its true entropy. Logstd clamp and pre-tanh Gaussian entropy remain as before. No categorical trainer, production PPO or default config changed.

## Fixed checkpoint 976 — common 64-key difficult panel

Each seed has 64 deterministic and 64 stochastic 15-second evaluations. All training seeds 201–204 are **already-exposed development seeds**, not fresh confirmation.

| Gain | Seed 201 det/stoch | Seed 202 det/stoch | Seed 203 det/stoch | Seed 204 det/stoch | Total |
| --- | --- | --- | --- | --- | ---: |
| **1.0×** | 54/53 | 56/55 | 59/58 | 48/44 | **427/512 (83.40%)** |
| **1.5×** | 56/55 | 57/56 | 61/61 | 54/53 | **453/512 (88.48%)** |
| **2.0×** | 61/61 | 62/59 | 61/60 | 56/54 | **474/512 (92.58%)** |

All held-out ordinary cases completed for 1.5× and 2.0× (**512/512** each), compared with baseline **510/512**.

Paired outcomes on identical (seed, mode, key) tuples, always comparing to 1.0×:
- **1.5×:** 26 formerly failed cases **recovered**, **zero** formerly successful cases lost; 59 failures remain.
- **2.0×:** 49 formerly failed cases **recovered**, but **2** formerly successful cases lost; 38 failures remain. Newly failing pairs are seed **202 stochastic**, key `4100100232`, and seed **203 stochastic**, key `4100100233`. These must remain in future regression panels.

Under 2.0× gain the remaining **38 failure evaluations cover 11 distinct keys** (vs 85 failures / 20 distinct keys baseline). The original hard key `4100100100` fails in **all 8** seed/mode cases even when trained at 2.0×. The key `4100100015` still fails in 7/8 at 2.0×. Of the 38 2.0× failures, 36 recorded negative cart clearance and **2** did not; this CSV alone does not prove the other two are pole-only terminations, because final pole state was not recorded.

| Gain | Mean survival ticks (of 1500) | Median min cart clearance across all difficult episodes | Mean raw return |
| --- | ---: | ---: | ---: |
| 1.0× | 1265.61 | 0.414 m | 1106.86 |
| 1.5× | 1337.26 | 0.531 m | 1180.25 |
| 2.0× | 1394.61 | 0.626 m | 1202.65 |

These returns mix survival and a squared-force penalty; **reward alone** is not an actuator quality or hard-case success metric.

## Training progression and stability

At checkpoint 512/768/976, difficult-panel completions out of 128:

| Seed | Gain 1.0× | Gain 1.5× | Gain 2.0× |
| ---: | --- | --- | --- |
| 201 | 42 / 108 / 107 | 7 / 107 / 111 | 104 / 108 / 122 |
| 202 | 0 / 109 / 111 | 75 / 113 / 113 | 47 / 117 / 121 |
| 203 | 17 / 108 / 117 | 41 / 119 / 122 | 0 / 114 / 121 |
| 204 | 96 / **72** / 92 | 79 / 104 / 107 | 27 / 109 / 110 |

Higher force **does not guarantee faster learning**: seed 203 with 2.0× still has zero completions at update 512. The baseline seed-204 late collapse from 96→72 at 512→768 disappears in the two higher-gain experimental runs, but 2.0× shows some other seed/key regressions during late training. **No monotonic/global convergence theorem follows.**

## Compare with evaluation-only amplification (different training distributions)

The earlier [evaluation-only study](ppo-gaussian-force-gain-results-20261016.md) held the **1.0× trained Gaussian weights fixed** and amplified their action *after training*:

| Gain | Original Gaussian, evaluation-only gain | PPO trained and evaluated with same gain |
| --- | ---: | ---: |
| 1.0× | 427/512 | 427/512 |
| 1.5× | **497/512** | **453/512** |
| 2.0× | **502/512** | **474/512** |

**Inference:** The learned policy responds to the gain used during training. Its action outputs, resulting state distribution and reward trade-offs change. More force is beneficial, but optimizing under the amplified mapping does not automatically reproduce the much stronger fixed-weight evaluation-only result. Neither the difference nor any training-policy adaptation is a direct measurement of learned latent-action amplitude: the current 12-policy CSV stores final outcomes and force sign switches, **not per-tick latent or commanded force magnitudes**.

## Remaining gates and next diagnostic

1. **Frozen train-gain × evaluation-gain crossover:** Evaluate each fully trained gain=1.0/1.5/2.0 Gaussian policy under every **absolute** test gain=1.0/1.5/2.0 (nine cells), without further training or changing rewards/keys. Use the exact same 64-key panels, modes and seeds; verify diagonal cells against these immutable prior results. This separates **training adaptation** from **evaluation force mapping**. Retain any negative/off-diagonal results.
2. Record actual post-gain requested forces for a fixed set of already-exposed witness keys. Compare force magnitude/timing rather than asserting adaptation solely from completion counts.
3. Test a candidate with the **actual production continuous PPO implementation** under matched training recipe and noise, without silently changing defaults. #35 targets a reliable supported continuous PPO path, not a separate categorical or Gaussian experimental workaround.
4. Freeze **fresh** confirmation training seeds, unseen resets, fixed hard-case survival and actuator/bandwidth acceptance criteria before evaluating any candidate for #33. These four exposed development seeds and repeated keys are correlated; 512 evaluated rows are **not 512 independent learned policies**.

**Status:** experimental PR #41 stays **draft/unmerged**; issues #33 and #42 remain open; no hardware/production readiness claim.
