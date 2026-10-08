# Early recovery trajectories: matched historical failures versus 2.56-second timeouts
**October 8, 2026 — completed read-only development diagnostic; not a PPO correction**

Pre-registered before new reduction: [RustRobotics #35 comment 6061604313](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061604313). This continues [the actual training-timeout calibration](https://github.com/yongkyuns/RustRobotics/blob/36a5fb4e72cdc6f1a915fa257216a3ccf5f09b7e/audits/clean_reference/PPO_RECOVERY_TIMEOUT_CALIBRATION_20261007.md).

## Result

Within the actual difficult-recovery training episodes of **scaled-2048 development seed 201**, failed trajectories generally show larger pole-angle error by 0.25 seconds and less force opposing cart motion over 0.25–0.50 seconds than nearby trajectories that reach the 2.56-second training cutoff. This is an **observational comparison**, not evidence that changing a single action/force will rescue the episode.

All **428** previously defined difficult outward-moving starts are retained: **120 physical failures**, **308 256-tick timeouts**. Timeout is *not* proof of completed recovery.

## Immutable data, provenance and replay

- `ppo_recovery_credit_evidence_20261007.zip`, SHA256 `ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412`: original exactly reconstructed 512 training batches / 1,048,576 physical packet transitions, native rewards, actor actions/advantages.
- `ppo_recovery_episode_credit_analysis_20261007.zip`, SHA256 `34148c3f1fea74925d1abbde8dac3e80e92802869114ee01024845c737d420eb`: all original episode boundaries and classifications.
- Both outer SHA256 digests and all **648 + 26 member hashes** verified before the reduction. All 4,289 complete episode ranges bind to their initial/final physical states, exact 0-based episode ages, physical termination/256-tick timeout, reward prefixes and stored initial advantages. For contained true-terminal episodes, the observed discounted return matches their zero-tail target within 0.001 physical reward units.
- New work performs **zero native transitions, zero PPO optimizer transactions**.

| Original difficult outcome | Episodes | Alive through 0.25 s | 0.50 s | 0.75 s | 1.00 s |
|---|---:|---:|---:|---:|---:|
| Physical failures | 120 | 120 | 113 | 93 | 40 |
| 256-tick timeouts | 308 | 308 | 308 | 308 | 308 |

Later-window averages are conditional on being alive and subject to survivorship bias.

## Fixed nearest-neighbor observational comparison

For every failure choose a timeout start **in the same original training quarter** minimizing maximum absolute difference in initial physical [position,velocity,angle,angular velocity] after dividing coordinates by reset half-widths [.2,.4,.25,.5]. Tie-break by earlier training-step index. **Controls are reused**.

Of all 120 failures, **77** are within normalized distance ≤0.25, **3** within ≤0.10, all within ≤0.50. The primary 77 close pairs belong to quarters 37/5/15/20; median time separation between their source training transitions is **119,852**—their incoming policy weights need not match. Only **74** close pairs survive through 0.50s in both arms.

| Prespecified early metric | Number of close pairs | Failure mean | Timeout mean | Paired mean failure − timeout |
|---|---:|---:|---:|---:|
| Absolute pole angle at 0.25s, radians | 77 | **0.18397** | **0.08663** | **+0.09734** |
| Absolute cart position at 0.25s, metres | 77 | 0.38143 | 0.51150 | −0.13006 |
| Fraction with force opposing cart velocity, 0–0.25s | 77 | **0.19779** | 0.06787 | **+0.12992** |
| Same proxy, 0.25–0.50s | 74 | **0.50269** | **0.64270** | **−0.14001** |
| Force projected along negative initial pole-angle sign, 0.25–0.50s, N | 74 | +1.11204 | −2.65389 | +3.76593 |
| Mean original physicalized GAE, 0–0.25s | 77 | −247.85 | +75.34 | −323.19 |

**73/77** failure pairs have larger absolute pole angle at0.25s; **59/74** have a lower brake-proxy fraction during0.25–0.50s. This pattern has the same direction within every original training quarter:

| Quarter | Close pairs | Failure − timeout, |angle| at0.25s | Failure − timeout, brake proxy 0.25–0.50s |
|---:|---:|---:|---:|
| 1 | 37 | +0.16264rad | −0.21531 |
| 2 | 5 | +0.05323rad | −0.05600 |
| 3 | 15 | +0.04343rad | −0.08800 |
| 4 | 20 | +0.02798rad | −0.07200 |

The braking proxy is the fraction of observed samples with `|cart velocity| >.25m/s` and `clipped commanded force × cart velocity <0`. It is not actual disturbed applied force, true cart acceleration or an optimal-action oracle. Observed sampled PPO actions contain exploration noise.

**Retain the contradictory first-window evidence:** failing trajectories actually have a higher brake-proxy fraction over 0–0.25s. No simple “PPO never brakes” mechanism is proven. And the failed group is not systematically closer to the cart rail at0.25s despite worse pole angle. The early GAE difference reflects actual stored baseline-relative PPO credit; timeout targets include estimated future returns, so it is not a binary success-label or proof the PPO gradient is correct.

## Causal interpretation and limitations

This analysis compares different realizations of stochastic trajectories, often under different incoming policies separated by many optimizer updates. The nearest-neighbor match does not control plant noise, action noise or policy age, nor ensure latent-state equivalence. Some controls appear more than once, early physical state matching is imperfect, and group membership at later times is survivor-conditioned. **No causal treatment effect, independence-based significance or reliability guarantee follows.**

The early-response difference is a concrete target for a separate intervention: freeze contemporaneous incoming policies at predeclared historical difficult states, compare same-physical-state/noisy-observation continuations under common future noise, then explicitly distinguish policy changes from trajectory randomness. Prior diagnostics already show why one-step LQR labels cannot stand in for an entire coordinated recovery maneuver.

## Reproducibility and status

Compact reproducible analysis bundle `ppo_early_recovery_observational_audit_20261008.zip`: **198,404 bytes**, SHA256 **af7c6f1e489a261cc712277e9d90d79dc1db0ee4745139892b53f20f05797c64**. Contains the standalone NumPy-only analyzer, genuine small packet fixtures/tests, complete all-428-episode derived records, all120 matches, quarter summaries, checks and this report. The two original source ZIPs are separate immutable dependencies, not repackaged.

A fresh extraction verified both archives and all674 source payload hashes; **10 embedded binding checks and10 fixture corruption/identity tests passed**, and all five report JSON outputs replayed **byte-for-byte**. Code invokes no Torch, native simulator, PPO optimizer or network.

**No production/default/master/protected-source changes, new trained policy or merge. #33/#35 remain open.**
