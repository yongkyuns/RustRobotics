# PPO recovery exposure and coordinated control — October 7, 2026

**Completed development diagnosis, not a new qualified PPO correction.** Protocol: issue35/comment6048736471. The full-LQR comparator was added under post-result addendum6048807131, before its new outcomes. No production/default/master/protected-source/deployment change or merge.

## Main finding

Difficult outward-moving starts are present in actual training, but the fine neighborhoods of the four failed evaluation starts are sparse. The first action of a successful recovery controller is not necessarily beneficial when followed by the existing failing policy. Here those isolated actions genuinely worsen return; keeping the recovery controller active from an earlier state succeeds. This distinguishes missing experience, local coverage, conditional value error and multi-step control. It does not establish a universal PPO failure mechanism or justify LQR teacher labels.

## Exact historical reconstruction

Replayed **development seed201**, scaled-2048, from random initialization through1,048,576 training interactions. This seed was selected before new measurements because it accounts for four of the six previously observed endpoint failures. None of the eight previously held-out seeds was used.

Input: original run37660544338, seed201 artifact11504926318, SHA25611d9efaeb4225ef70449f7672bc72accc0a0001bffcf3d2f31a814ba25c2b4a2. All764 member hashes and recovered input bytes match. Pinned native archive58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0; Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0 on AMD EPYC9V74.

At **all six archived checkpoints** (0,131072,262144,524288,786432,1048576), actor/critic arrays and every original evaluation record/reward trace match exactly. All512 original rollout diagnostic records and the complete training-episode ledger match. A separate plain-versus-captured preflight verifies policy, Adam, RNG and live episode equality. The prior archive does not provide every intermediate actor/Adam state; no all-step historical optimizer-byte claim is made.

Every actual training transition is retained with physical state, noisy observation, command, raw reward, age/endings and rollout values/returns/advantages. Only the historical baseline was reconstructed; no new deployment candidate was trained.

## Training exposure

The region was fixed before replay: |angle|>=.15rad, |angular_velocity|>=.25rad/s and their product>0. The original training/evaluation reset half-widths remain(.2m,.4m/s,.25rad,.5rad/s).

**4,290 training starts include428 difficult starts and120 observed difficult-start physical failures.**

| Training interval | Starts | Difficult starts | Difficult256-tick timeouts | Difficult physical failures |
|---|---:|---:|---:|---:|
| 0–262144 | 1178 | 122 | 65 | 57 |
| 262144–524288 | 1033 | 106 | 93 | 13 |
| 524288–786432 | 1038 | 101 | 80 | 21 |
| 786432–1048576 | 1041 | 99 | 70 | 29 |

Intervals classify by episode start; one final ordinary episode is unfinished. Different quarters contain different states/noise/policies, so these counts alone are not a matched causal deterioration test. Reaching the256-tick timeout is only2.56seconds, not a sustained-balancing certificate.

Nearby-reset counts use the predeclared maximum coordinate distance normalized by the original reset half-widths:

| Failed seed201 case | Distance<=.10 | Distance<=.25 | Last-quarter distance<=.25 |
|---|---:|---:|---:|
| Deterministic13 | 1 | 6 | 2 |
| Deterministic19 | 0 | 5 | 0 |
| Stochastic5 | 1 | 2 | 1 |
| Stochastic24 | 0 | 9 | 1 |

Thus neither “PPO never saw difficult starts” nor “the relevant fine neighborhoods are well covered” is justified. These are descriptive coverage probes, not a viability region or an exhaustive support test.

A **post-result descriptive reduction** additionally finds113 of the120 difficult failed episodes fully contained inside one rollout. Their initial physicalized stored return targets agree with the actual gamma-.999 terminal-stopped reward-to-go within0.000642 maximum absolute difference. Seven rollout-crossing episodes are excluded from this direct-target statement. Observing failures and their terminal returns still does not prove discovery of a successful coordinated maneuver.

## Same-state conditional experiment

The four original failure paths passed exact96-case short-panel replay. Twelve anchors use exact physical states and already-observed noisy inputs at ages25/50/75ticks. The additive state-branch ABI matches the original library for4,096 transitions plus state-transplant controls; environment/plant source files match.

Each anchor has32 independent paired future-noise replicas. Original registered arms: unchanged PPO; fixed LQR mean for one action then PPO; fixed LQR mean for25actions then PPO. All use the current PPO Gaussian sigma and noisy observations. Even anchors drawn from deterministic evaluation have stochastic continuations here. The fixed gain is[-12.326826,-18.432033,148.100494,61.271130], with no retuning.

Full LQR thereafter was registered **after** the initial three-arm results and before its own outcomes. Earlier measurements were not rerun. All four arms retain the same anchors and future-key pairing.

| Switch age from original reset | PPO /128 | LQR1 then PPO /128 | LQR25 then PPO /128 | LQR thereafter /128 |
|---|---:|---:|---:|---:|
| .25s | 0 | 0 | 4 | 128 |
| .50s | 0 | 0 | 0 | 0 |
| .75s | 0 | 0 | 0 | 0 |

Success here is survival to8192ticks, nominally81.92seconds. These are four selected failed paths of **one trained policy**, not128 independent trained policies. Failure of this fixed LQR from later states does not prove that every possible controller would fail there.

### The single-action credit is correctly negative

Across the fixed12-state panel, the new-first-action minus PPO stopped-return effect is **-1.071464**, approximate pointwise99% interval **[-1.095845,-1.047082]** across32 independent replica means. Every anchor mean is negative. Effects are identical at1024/4096/8192ticks because all these trajectories terminate earlier.

The simulated training-clock prefixes have231/206/181 remaining ticks. **Every PPO and one-action branch terminates before its prefix cutoff**, so its bootstrap contribution is exactly zero. The short-prefix and long stopped-return differences agree exactly. This particular rejection of an LQR first action is not caused by an erroneous critic tail.

The25-action panel effect is-6.923055, approximate pointwise99% interval[-17.700330,+3.854220]; its overall sign is unresolved. Four survivors occur only at deterministic13's earliest anchor. Full per-anchor results and all horizons are retained.

These are finite conditional policy substitutions, not actual PPO parameter-gradient derivatives. They do not prove a formal optimization barrier or rule out a network update coordinating changes across many states. They do show that a successful full controller cannot automatically supply valid one-step advantage labels.

## Conditional value errors remain

At the.25s anchors, physicalized critic prediction versus independently measured continued-PPO return:

| Case | Critic prediction | Mean continued-PPO return |
|---|---:|---:|
| Deterministic13 | 111.001 | 16.748 |
| Deterministic19 | 536.274 | 12.357 |
| Stochastic5 | 227.094 | 8.603 |
| Stochastic24 | -124.468 | 14.670 |

Second-hidden-layer tanh saturation over the12 inputs is6.38%, not near-total saturation. Unit conversion is independently verified. Conditioning is not calibration. These estimates condition on physical states and noisy inputs, not an integrated observation-only latent-state posterior.

A baseline error can matter to fitting/finite-sample variance even when a completed trajectory contains no tail bootstrap, but these results do not uniquely attribute historical policy learning to that error. The full-LQR arm's optional prefix-plus-PPO-critic number is **not an unbiased full-LQR value estimate**: its continuation policy differs. That arm is a feasibility comparator, never a source of training labels.

## Verification and disposition

Independent NumPy/SciPy-only reader checks **1,536 conditional trajectories /1,142,191 transitions**, bit-exact sequential-float32 reward reconstruction, first physical endings/no padding, all stopped-return horizons, state/key/observation/age identity, zero terminal bootstrap, actor/LQR commands and physical critic predictions. Maximum command discrepancy2.98e-7 and value discrepancy2.39e-4 are below fixed tolerances3e-6/.003.

Historical verification checks **1,048,576 fitting rows** and **1,048,064 interior float32 GAE recursions**. Live final bootstrap values and individual timeout predictions were not separately exported; the reader does not claim independent reconstruction of those boundary predictions or every Adam transaction. Native private RNG draws and complete nonlinear trajectories are not independently regenerated offline. Commanded force is distinct from internally disturbed applied force.

**18 corruption/boundary tests pass.** A fresh extraction verifies648 package payload hashes, reproduces all three numerical reports byte-for-byte and reruns all18 tests. Verification executes no native simulator, Torch, PPO optimizer or pickle loading.

Total new native transitions: **2,962,465**, including1,048,576 historical replay training,8,192 recorder-control training,630,280 historical evaluation,8,192 original/bridge comparisons,86 state-transplant control steps,123,406 anchor short-replay steps,1,542 conditional-control steps, and1,142,191 conditional measurements. Historical optimizer transactions163,840; recorder controls1,280. No new production candidate or efficiency claim.

Complete evidence: **ppo_recovery_credit_evidence_20261007.zip**,143347568bytes,SHA256 **ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412**. Compact report/code: **ppo_recovery_credit_report_code_20261007.zip**,69889bytes,SHA256 **cd3e96866e16db0a4e2f1f6545c29ced80456ec3c23dd7c86d6a4db70a7f3efc**. Extract the complete package and run `OPENBLAS_NUM_THREADS=1 python REPLAY.py` with NumPy and SciPy. The compact package omits measured inputs and is not standalone replay data. Pinned native runtime installers are excluded from both.

Next focused work should analyze the retained actual successful/failed recovery episodes and their fitting credit, rather than assume a missing recovery class, install LQR, use its first actions as oracle labels, or sweep batch sizes again. This report queues no new experiment. Issues33/35 remain open; no merge or production changes.
