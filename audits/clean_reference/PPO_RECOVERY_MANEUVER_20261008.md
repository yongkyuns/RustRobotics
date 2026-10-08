# PPO difficult-start recovery: early maneuver and recorded credit

**Date:** October 8, 2026 · RustRobotics issue #35 · **Completed exploratory development analysis**. No new learning, simulation, policy, or production change.

## Finding

Difficult recovery failures change character as the one retained PPO training history progresses. In the first quarter, failing trajectories often lack a timely reversal of the pole's outward angular motion. In the fourth quarter, even failing trajectories reverse that motion, but retain higher cart speed and smaller rail clearance during the next phase. The historical PPO advantages already distinguish many failing and 2.56-second-surviving responses; this is not evidence that PPO never credits recovery.

**This is observational evidence from one exactly reconstructed, exposed development seed (201), not a causal or population-level result.** A training timeout is 256 ticks, not demonstrated complete recovery; the earlier separate timeouts study showed that many such paths could later finish recovery under the same policy.

## Registered protocols and inputs

- Primary before-new-results plan: [issue #35 comment 6061559491](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061559491).
- Separately registered post-first-results cart momentum addendum: [comment 6061654892](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061654892); the first reduction's already-seen pole-turn/speed facts were disclosed in that addendum. New cart metrics were not retroactively relabeled as primary.
- Original, immutable actual-training evidence: `ppo_recovery_credit_evidence_20261007.zip` SHA256 `ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412`, captured on the frozen 2048-step scaled PPO trajectory.
- Previous, immutable episode-credit ledger: `ppo_recovery_episode_credit_analysis_20261007.zip` SHA256 `34148c3f1fea74925d1abbde8dac3e80e92802869114ee01024845c737d420eb`.
- **All** 512 batches / 1,048,576 raw physical-packet rows, 4,289 complete episode identities, and all 428 difficult starts retained; no previously held-out training seeds used.

The existing difficult-start definition is `abs(theta0)>=.15 rad`, `abs(omega0)>=.25 rad/s`, and `theta0*omega0>0`. Its 428 starts include **120 physical-limit failures** and **308 episodes reaching the 256-tick training timeout**. Across all groups, 25-tick metrics are eligible for all 428; 50-tick metrics exclude the seven failures that ended earlier. All bounds use the recorded **true** physical states; policies trained on noisy observations.

## 1. First-quarter versus final-quarter pole response

| Metric | Quarter 1 failed (57) | Quarter 1 timeout (65) | Quarter 4 failed (29) | Quarter 4 timeout (70) |
|---|---:|---:|---:|---:|
| Pole outward-angular-rate reversal by tick25 | 28/57 | 65/65 | 29/29 | 70/70 |
| Mean reduction in abs pole angle at tick25, rad | -0.045 | 0.123 | 0.082 | 0.091 |
| Mean fraction of force commands at |u|>=19.9 N, first25 | 0.401 | 0.670 | 0.505 | 0.427 |
| Mean first25 **whole-rollout-normalized** stored advantage | -0.670 | 1.076 | -2.981 | 0.742 |

In quarter 1, timed-out paths typically swing the pole back promptly, using a stronger initial cart excursion/force pattern than failing paths. By quarter4, nearly all recorded trajectories—including failures—achieve the early angular-rate reversal. This is a **phase shift in the failure signatures**, not proof that training reliably learns a safe controller.

## 2. Cart-momentum risk after early pole correction

At **tick50**, quarter4 failures have mean absolute cart speed **2.734 m/s**, against **2.146 m/s** for timeouts. Median rail clearance is **1.122 m** versus **1.409 m**.

For a transparent *kinematic proxy only*, classify outward motion when `x*v>0` and `abs(v)>.1m/s`, then compute `(2.4-abs(x))/abs(v)` as constant-speed time to rail. It **ignores ongoing applied acceleration, feedback, disturbances and collision prediction**.

| Quarter | Failure paths with proxy <=0.5s at tick50 | Timeout paths with proxy <=0.5s at tick50 |
|---:|---:|---:|
| 1 | 10/50 | 45/65 |
| 2 | 13/13 | 49/93 |
| 3 | 21/21 | 23/80 |
| 4 | **24/29** | **11/70** |

The opposite quarter1 direction is important: many **successful timeout paths** approach the rail aggressively before braking. A short kinematic time-to-rail alone is not a failure classifier. Later failures increasingly combine an early pole reversal with excessive outward cart momentum.

## 3. Predeclared close-start descriptive matching

For each of 120 failed difficult starts, match the closest timeout start from **the same training quarter** under normalized physical-state Chebyshev distance, scaling `[x,v,theta,omega]` by original reset half-widths `[.2,.4,.25,.5]`, with replacement and earliest-timestamp tie breaking. **77 of 120** matches are within the primary predefined distance <=.25, involving **57 distinct timeout episodes** (reused controls are not independent matches).

Within those 77 near-start pairs, the timeout reduces absolute pole angle more at tick25 in **69/77** pairs, with mean timeout-minus-failure difference **+0.085 rad**. Its first25 whole-rollout-normalized advantage is higher in **63/77**, difference **+3.515**.

For quarter4's **20** close-start pairs at tick50, **18/20** failing branches versus **6/20** matched timeout branches have a <=0.5s rail-time proxy. By contrast, quarter1's **20/37** timeout paths versus **3/37** failed paths meet that proxy at tick25, reinforcing the early aggressive-excursion distinction.

Pairs differ in actual policy checkpoint, actuator/noise draws and later states, and timeout controls are reused. These are **not** same-state counterfactual interventions; no independent trained-policy confidence interval or causal superiority is claimed.

## 4. What PPO credit says, and what it cannot say

The first25 normalized stored advantage is strongly lower in difficult failures than in timeouts in each quarter (e.g. final quarter **-2.981 vs +0.742**), indicating that this **recorded credit readout** broadly differentiates early trajectory outcomes. The historical PPO implementation normalized advantage within shuffled 64-row minibatches; the whole-2048-rollout-normalized quantity here is explicitly **descriptive and not its actual optimizer objective**. The archive lacks every intermediate actor/Adam snapshot/minibatch permutation, so we cannot infer that a specific recovery action's probability actually increased or decreased due to these credits.

This study does **not** show missing exposure to the difficult-start category, a uniformly wrong advantage sign, or that any simple LQR one-action substitution must improve return. The preceding direct diagnostic already showed that successful recovery can require sustained coordinated control.

## 5. Verification, scope, next work

- Outer SHA256 and all **648 + 26** package member hashes verified.
- Bound all **1,048,576 original physical/observation/action/reward rows**, actual episode starts and endings, age, step continuity, true termination or training timeout, force/latent-action identities, reward reconstruction **bit-for-bit**, and the prior complete ledger.
- **12 real-fixture and deliberate corruption tests pass**. No new simulator, PPO training, optimizer or policy weights generated.
- Reproduction scripts: `analyze.py` for the pre-registered 428-episode analysis; `cart_momentum.py` for the explicitly post-result addendum; `write_report.py` exports this summary and all episodes/matches to CSV. Original 143MB+2.6MB evidence archives are required to recompute the raw readout, but **not copied** into this compact result bundle.

The most promising NEXT causal diagnostic is a **same-state paired intervention on actual late-stage outward-moving cart states after pole rotation has already reversed**, comparing the existing PPO continuation with a coherent early braking response over multiple actions. It needs exact contemporaneous policy/cutoff states and paired physics/noise, not single-action teacher labels. Predeclare that experiment before any new rollout or PPO training.

**No production/default/master change, LQR fallback, deployed weights, or merge. Issues #33/#35 remain open.**