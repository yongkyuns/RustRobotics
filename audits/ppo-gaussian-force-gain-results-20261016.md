# PR #41 / issue #42 — completed Gaussian force-amplification evaluation (October 16, 2026)

**Finding:** A bounded corrective-force deficit is a plausible, strongly supported contributor to the experimental squashed-Gaussian PPO's *development* reset failures. Applying larger forces to **unchanged learned policies** sharply improved 15-second recovery. This does **not** establish a training-time remedy, a unique causal mechanism, or a fix for original production continuous PPO.

## Verified executed sources

- Source revision: [`905a9d4a57ab4ece7fb84306d99609ee609cc7c9`](https://github.com/yongkyuns/RustRobotics/commit/905a9d4a57ab4ece7fb84306d99609ee609cc7c9).
- Full experiment [Actions #38954417339](https://github.com/yongkyuns/RustRobotics/actions/runs/38954417339): **SUCCESS**. Release build, four 976-update native Gaussian development policies, original-baseline reproduction, full evidence verification, artifact upload all passed.
- Native Rust [CI #38954423590](https://github.com/yongkyuns/RustRobotics/actions/runs/38954423590): **SUCCESS** (Rust test, strict Clippy and rustfmt).
- [Artifact #12070766798](https://github.com/yongkyuns/RustRobotics/actions/runs/38954417339/artifacts/12070766798): **4096 episode CSV rows**, **25,298 witness per-tick records**, provenance, source SHA, summary, original raw logs. File SHA256 independently verified:
  - `episodes.csv`: `1073df33881e1e2e61bdd9a3c552abac4c6192a3f0abc09f280c8d18d8fc2bb8`
  - `witness-traces.csv`: `165c66caca8f8ee654f50ffd1019d7c4f505dc399544408576dd8e6445625d0d`

## Frozen causal intervention

Four already exposed training seeds **201, 202, 203, 204** were retrained *once per seed* with identical source and full **976 PPO updates / 499,712 interactions** per seed. All have the same Gaussian actor, original noisy nonlinear native plant and original PPO training recipe. **No training steps, weights, rewards or actor observations were changed between actuator conditions.**

Only evaluation-time physical force transformations differed, all clipped at ±20 N:
- `ideal`: output `F = 20*tanh(z)`.
- `boost_1p5`: output `clamp(1.5*F, -20, 20)`.
- `boost_2p0`: output `clamp(2.0*F, -20, 20)`.
- `bang_bang_sign`: output `±20 N` according to sign of `F` (zero maps to zero).

Each condition evaluated the identical common **64 ordinary and 64 difficult reset keys**, deterministic and stochastic actions, fixed **1500 steps × 0.01 s = 15 seconds**. Reset RNG streams were paired; trajectories and subsequent noise consumption can diverge when force interventions change the state.

## Results: whole common difficult panel

| Force condition | 201 (det/stoch) | 202 (det/stoch) | 203 (det/stoch) | 204 (det/stoch) | Total complete |
| :-- | :-- | :-- | :-- | :-- | --: |
| Ideal | 54/53 | 56/55 | 59/58 | 48/44 | **427/512 (83.40%)** |
| 1.5× capped | 60/61 | 63/63 | 63/63 | 62/62 | **497/512 (97.07%)** |
| 2× capped | 62/62 | 63/63 | 63/63 | 63/63 | **502/512 (98.05%)** |
| Bang-bang sign | 62/62 | 63/63 | 63/63 | 63/64 | **503/512 (98.24%)** |

On the paired 512 seed/mode/key tuples:
- **1.5×:** **70** baseline failures recovered, **zero** baseline successes lost. Remaining 15 failures span four keys.
- **2×:** **75** recovered, **zero** lost. Remaining **10 failures** span only **two** keys: `4100100100` (8 seed/mode failures) and `4100100015` (2).
- **Bang-bang:** **76** recovered, **zero** lost. Remaining nine failures span the same two keys.

Common ordinary completion was **510/512 at ideal** and **512/512** for all three stronger-force variants. No unfavorable completion outcomes were suppressed.

For comparison with earlier [matched action-space results](ppo-matched-action-ablation-results-20261016.md), the distinct categorical PPO policies had achieved **505/512** difficult completions. The boosted *Gaussian* policies approached this level without changing PPO training, but they are not the same type of trained controller and no claim of equivalent physical safety is made.

## Survival, physical margins, reward and switching

Aggregated over the 512 common difficult episodes:

| Condition | Mean steps survived (of 1500) | Median min cart clearance, all episodes | Mean raw return |
| :-- | --: | --: | --: |
| Ideal | 1265.61 | 0.414 m | 1106.86 |
| 1.5× | 1458.35 | 0.600 m | 1258.62 |
| 2× | 1472.20 | 0.633 m | 1246.60 |
| Bang-bang | 1475.14 | 0.651 m | 668.10 |

The reward includes a force-squared penalty. It is **not** a safe cross-actuator metric by itself; compare survival, margins, actual applied effort, and actuator bandwidth. For completed 15-second difficult episodes, measured *policy-command sign changes* averaged about **42.1/s ideal**, **46.7/s 1.5×**, **49.1/s 2×**, and **72.8/s bang-bang**. These counts do not directly model motor torque slew, current or wear, nor measure the actual noise-perturbed force's sign changes. Bang-bang retains severe chattering.

## Fixed development witness 4100100015

Of eight seed/mode evaluations at this one preselected key:
- Original Gaussian: **0/8** survive, cart rail breached at **0.70–0.80 s**.
- 1.5×: **4/8** survive.
- 2×: **6/8** survive. The two seed-201 modes still fail around **0.90 s**.
- Bang-bang: **6/8** survive; the same two seed-201 modes still fail.

Per-step traces for all 32 combinations (seed × mode × force condition) passed record-count and terminal consistency checks. Failures on this witness had cart rail breaches and **no simultaneous angle breach** in their recorded last frames.

During the first 30 ticks, the original Gaussian's mean commanded magnitude across the eight witness evaluations is **13.645 N**. The boosted conditions modify trajectories and subsequent policy outputs, so their *desired* mean force differs too; the 2× condition's mean actuator-command magnitude during that window is about **15.22 N**, while bang-bang uses **20 N**. These are descriptive measurements, not evidence of a globally optimal force gain.

## Interpretation, limitations, and disposition

This is unusually strong *evaluation-level causal intervention* evidence: boosting only the actuator command magnitude while keeping trained policy parameters fixed recovered up to **76 of 85** exposed difficult failures with **no paired new failures**. It directly establishes that force mapping can materially affect recovery, with the same action-sign decisions generally driving it. It does **not** establish that force magnitude alone caused every failure: changing forces changes future states, timing and stochastic feedback, and a hard reset remains failed under full force.

**No merge or production default change.** All four training seeds and reset keys are development-exposed; 512 episodes are correlated repeated measurements, not 512 independent policy training trials. The real production Gaussian PPO has its own state-coverage/action-credit issues (issues #35/#33), and no hardware actuator with slew or time delay was qualified. This ablation changed only evaluation, so **a fixed-force-gain training-and-evaluation candidate must be tested separately**, preserving the entire policy update record, original continuous and categorical baselines and unfavorable seeds. Freeze fresh training/evaluation confirmation only when a candidate and acceptance criteria are settled.

This audit documents measured evidence, not a determination of reliable production control.
