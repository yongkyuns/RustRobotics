# PR #41 — Four-seed native PPO evidence audit (2026-10-16)

**Disposition: learning diagnostic PASSED; production PPO promotion NOT qualified.** This is an opt-in categorical `±20 N` reference, not the existing continuous-action PPO. All claimed measurements below come from completed [workflow #38848301728](https://github.com/yongkyuns/RustRobotics/actions/runs/38848301728), [artifact #12025132799](https://github.com/yongkyuns/RustRobotics/actions/runs/38848301728/artifacts/12025132799), source SHA `83f0530fdab6f71360a6fab8e4b1aec857313fbe`, and cross-checks of its raw per-episode CSVs.

## Completion and reproducibility

- Formatting, 17 targeted Rust tests, Clippy, release build, four training runs and CSV verification **passed**. Independent [reference CI #38848306007](https://github.com/yongkyuns/RustRobotics/actions/runs/38848306007) also passed.
- Seeds **201, 202, 203, 204** all completed **976 PPO updates / 499,712 actual training environment interactions per seed**: total **3,904 updates / 1,998,848 interactions**.
- Four independent *training RNG seeds*, but these are **exposed development seeds**, not fresh confirmation seeds. Each recorded 7 checkpoints × 4 evaluation panels × 2 action-selection modes × 64 initial keys = 3,584 rows, total **14,336 rows**.
- Evaluation is 1,500 steps at `dt=0.01 s` = **15 seconds**, with actual native nonlinear dynamics and configured actuation/observation noise.
- The original seed-201 single-run evidence in [#38844353112](https://github.com/yongkyuns/RustRobotics/actions/runs/38844353112) has **1,792 development-panel rows matching exactly** the same 1,792 rows in the extended four-seed cohort. The two new evaluation panels did not change those rows.
- The CSV SHA256 digests recorded by the workflow were independently verified against the four downloaded CSV byte streams:
  - seed-201: `f6b06d0242837dc2ceec165b340c5a03cb2f384da677af122d8722e1a1dab10d`
  - seed-202: `d2f56fcbfd574d12f71ae4053c91529d45b3f44bb93a384c44d9ba7948adb60a`
  - seed-203: `9c611e85f402ef3952d473838a3714123575236eda2401c8305b412b6508a501`
  - seed-204: `9bd303cf4266fddf8d69001c82a29e139bb8e651e59cc4eaa522d93e199661d5`

## Final 976-update results: common held-out panels

All entries count **completed 15-second episodes / 64**. They are paired on identical reset keys across the four training seeds.

| Seed | Ordinary deterministic | Ordinary stochastic | Difficult deterministic | Difficult stochastic |
| ---: | ---: | ---: | ---: | ---: |
| 201 | 64/64 | 64/64 | 63/64 | 63/64 |
| 202 | 64/64 | 64/64 | 63/64 | 63/64 |
| 203 | 64/64 | 64/64 | 63/64 | 63/64 |
| 204 | 64/64 | 64/64 | 64/64 | 63/64 |

Across the same common difficult keys and four policies, that is **505/512 nominal evaluations**, but **NOT 512 independent draws**: keys, dynamics/noise seeds, training setups and modes are correlated. Do not quote 98.6% as a statistically qualified out-of-sample population success probability.

At updates **768** and **976**, ordinary held-out success was 64/64 for every seed and mode. Difficult held-out results remained at least 63/64 at both checkpoints. Earlier learning varied greatly: at update 256, seed 202 reached 59/64 difficult deterministic but only 13/64 stochastic, whereas seed 203 still had 0/64 in both modes. This is a multi-seed learning result, **not a proof of monotonic improvement**.

## Residual failures — one key dominates

All **seven failed final common-difficult evaluations** (out of 512 evaluated seed/mode/key combinations) share **reset key `4100100100`**. Seed 204's deterministic evaluation was the lone success at this key.

Initial true simulator state for this key:
- cart position **+0.123918 m**, cart velocity **+0.397964 m/s**
- pole angle **−0.232578 rad**, angular velocity **−0.433484 rad/s**

The seven failures ended at **step 62–83 (0.62–0.83 s)**, all `terminated=true`, and each recorded a **negative minimum cart-rail clearance** (`−0.000057` to `−0.014574 m`). Thus the cart *did cross the position boundary*. The CSV does not record final angular state, so it does **not** prove that rail crossing was the *only* termination condition or that recovery is physically impossible.

Across *all* development and common evaluation panels at update 976 there were **12 failures out of 2,048 episodes**, attributable to only four unique reset keys (three seed-specific difficult keys with 2, 2 and 1 failures, plus the common difficult key with 7 failures). No ordinary-panel final failures.

A targeted reachability/policy diagnostic should preserve this fixed witness and distinguish unavoidable actuator-limited states from action-selection mistakes. Never remove the witness from scoring after seeing it fail.

## High-frequency force reversal — a critical caveat

`ReferenceEvaluationEpisode.force_switches` explicitly counts each change between categorical left/right action commands. The native environment uses `dt=0.01` seconds and a direct `±20 N` force command with no actuator lag/rate-limit dynamics. For **completed final** episodes:

| Held-out panel | Mode | Median fraction of action transitions reversing sign |
| --- | --- | ---: |
| Ordinary | Deterministic | **81.5%** |
| Ordinary | Stochastic | **61.3%** |
| Difficult | Deterministic | **80.5%** |
| Difficult | Stochastic | **60.2%** |

Approximately **60–82 commanded full-scale force reversals per second** under 100 Hz control are needed by these policies in this idealized environment. This may approximate a pulse-width/duty-cycle controller in simulation, but is **not demonstrated safe, efficient or realizable on real actuators**. This metric is not a measure of the noisy applied force itself; it is the requested `−20 N ↔ +20 N` changes.

The reward/success criterion does not currently enforce actuation bandwidth, force slew, switching wear, or power consumption. Raw 15-second survival is consequently **insufficient** to qualify a deployable controller.

## Consequences and next gates

1. **No merge/promotion:** This is an experimental *categorical action* control. #35 concerns failures in ordinary *continuous Gaussian PPO*, and #33 requires an independently predeclared sustained-balancing acceptance protocol with *new* training and evaluation seeds. Four exposed dev seeds do not close either.
2. **Causal ablation:** Compare categorical vs continuous action under matched native physics, compute budget, number of envs, rollout length, optimizer/loss, noise, and held-out evaluation. Switching action space and training recipe simultaneously cannot identify which change fixed PPO learning.
3. **Actuator realism:** Quantify behavior with finite slew/bandwidth, discrete command delay or action hold, and an explicit switching/effort budget. Do this as a separate evaluation-only diagnostic before changing training rewards/defaults. Keep failure key `4100100100` in the regression set.
4. **Fresh confirmation:** Freeze acceptance (train-seed-level success and uncertainty, horizon, reset-key distribution, failure/rail margin, action effort), then use new train and evaluation seeds. Do not cherry-pick checkpoints or tune on confirmation failures.
5. **Numerical parity:** Full-network Torch↔Burn gradients, joint global clipping and persistent Adam comparison remains outstanding; earlier operator-level fixtures and Rust unit tests are narrower.
6. **Reproducible frozen inference:** Store safely restorable trained checkpoints and exact evaluator config if stress and counterfactual replay are needed without retraining. Existing artifacts provide CSV/logs, **not loadable model snapshots**.

This audit intentionally makes no claim of fixing production continuous PPO, proving control robustness under delay/lag, or achieving hardware-ready performance.
