# PPO update 54: exact localization of a damaging ordinary update

October 6, 2026 · RustRobotics issue #35 · Diagnostic result, not a production correction

## Result

The first predeclared local-stability-loss event after 65,536 interactions is **stock-SB3 seed 203, rollout update 54**, after collecting 110,592 interactions. The incoming actor was trained through 108,544 interactions. The update consumes one 2,048-row rollout in ten epochs: 320 ordinary Adam minibatches.

This same update improves the collected-data PPO surrogate and reduces critic error against the stored targets, but damages actual native-simulator behavior on both the original exposed evaluation panel and a separately keyed panel fixed before the new results. The actor, critic, Adam state and update random state are reproduced exactly by both plain and instrumented frozen-batch replays.

No production hyperparameter, reward, model, noise, curriculum, or supported training path was changed. No PR was merged and nothing was deployed by this continuation. This is one exposed training seed and one selected update, not population-level reliability qualification.

## Protocol, source and execution

The protocol was registered before new outcomes in [issue #35/comment 6021910555](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6021910555).

Successful [Actions run 37505417538](https://github.com/yongkyuns/RustRobotics/actions/runs/37505417538) executed commit **0e1246bbea16894e29f6161d177a864b6008e1bb**. It uses the original native binary from clean-reference run 36129074098, source da6623ae4b5878ae5079bf05c18c3d327570bc16 and production base 95c670b9f4618a11dc5439b026caf385a39718c8. The original `make_model`, `evaluate`, SB3 `PPO.train`, and `collect_rollouts` implementations remain unchanged. The bounded driver retains the original learn-call boundaries at 0, 65,536 and 262,144 interactions, then stops at the declared prefix budget.

Two from-scratch prefixes use the same ordinary stock-SB3 seed 203 recipe and continuing live learner/environment state. Both have read-only complete environment-output/action/reset hashing. The observed run additionally snapshots before/after updates. It does not import saved weights into the live training prefix or reset Adam. Diagnostic NumPy calculations and snapshots are checked not to consume the training RNGs.

Python 3.11, Torch 2.8.0+cpu, NumPy 2.2.6, SB3 2.9.0 and Gymnasium 1.3.0 are pinned. Four installed SB3 source modules are byte-checked against the original build artifact before training. Native inputs and every manifest payload are hash-verified.

## Historical and observer controls: exact at the recorded prefix checkpoints

Both new prefixes reproduce the archived **complete policy tensors and all 288 evaluation records** at 0, 65,536 and 262,144 interactions exactly. The recorded training-episode prefix also equals the archived prefix exactly.

The control and observed runs equal each other at all recorded checkpoints, final actor/critic/Adam/RNG state, every recorded training episode, all evaluation records, and the hash of all 262,144 training-step outputs/actions and reset outputs. Their common training-output digest is:

`c48b8d6caef121996229e9fb65de82d0b0759acbdd5f4c608387ae7a095ca252`

This is a historically matched reconstruction at all available prefix receipts, with exact within-build observer controls. Original optimizer states and every original intermediate update were not archived, so equality at those unrecorded points cannot be independently checked against September 25.

The earlier *standalone saved-checkpoint* seed-203 replay mismatch remains an unexplained, retained result of that different execution route. It was not erased or converted to a pass. The current training-prefix route matches the original records exactly and therefore does not require a relaxed historical replay tolerance. The 1,048,576-interaction endpoint is outside this new prefix audit.

## A better training score produces a worse controller

| Metric | Before update 54 | After update 54 |
|---|---:|---:|
| Mean clipped PPO surrogate on the same actual minibatch partitions | 0.000000008 | 0.003486083 |
| Critic MSE against the same stored targets | 0.007398704 | 0.002969943 |
| Sampled approximate KL | 0 | 0.001924346 |
| Nearest upright-equilibrium spectral radius | 0.996493434 | 1.000824038 |
| Nearest upright-equilibrium position, m | -0.192623 | -0.421786 |
| Position component of the force Jacobian, N/m, at those equilibria | +1.286100 | -0.457201 |
| Learned normalized-action standard deviation | 0.08353520 | 0.08166102 |

The surrogate is recomputed on the frozen 2,048 rows using all 320 actual recorded minibatch index sets and SB3's minibatch advantage normalization. It is an endpoint re-evaluation on fixed data, not an assertion that every minibatch loss improved monotonically. An independent float64 saved-weight calculation agrees with the recorded surrogate to maximum absolute discrepancy **1.17e-8**.

The critic MSE improvement is against bootstrapped/GAE targets, not an independent demonstration of correct action credit. A small sampled KL is likewise not a guarantee that the closed-loop feedback law or behavior outside these fitting states is preserved.

### Actual native-simulator evaluation

Deterministic episodes last up to 10 seconds; stochastic episodes up to 15 seconds. The original native environment, noise, termination rules and evaluator are used. The separately keyed panel uses the registered seed-key parameter 910203; it is a new evaluation draw, not a new independent training seed or a held-out recipe-qualification cohort.

| Panel | Deterministic completions | Stochastic completions | Mean stochastic discounted return |
|---|---:|---:|---:|
| Original exposed keys (203) | 18/32 → 14/32 | 36/64 → 16/64 | 79.03197 → 78.31972 |
| Separate fixed keys (910203) | 17/32 → 16/32 | 38/64 → 16/64 | 80.45442 → 79.85350 |

All 47 lost completions across these panels are retained; neither panel gains a newly completed case. Separate-panel stochastic mean discounted return falls **0.600915**, with paired-episode standard error **0.180417** conditional on this fixed policy pair and evaluation sampling. This is not a training-seed-level uncertainty estimate. The separate-panel deterministic discounted change is -0.438565 with standard error 0.335869; that smaller panel does not provide the same precision.

This update is therefore not merely a local spectral event with unchanged practical behavior, nor only a discrepancy between deterministic completion and stochastic discounted evaluation. The frozen finite panels exhibit actual return and completion degradation.

## What was in the fitting batch?

The entire rollout is **one uninterrupted near-upright segment**. It contains no episode starts, true terminations or timeouts. The preceding timeout is at interaction 107,211 and the next at 112,211; update 54's collection covers interactions 108,545–110,592. Its 2,048 transitions are not 2,048 independent recovery attempts.

| Noisy observation coordinate | Minimum | Maximum |
|---|---:|---:|
| Cart position, m | -0.259803 | -0.069577 |
| Cart velocity, m/s | -0.204217 | +0.200815 |
| Pole angle, rad | -0.012707 | +0.014714 |
| Angular velocity, rad/s | -0.085918 | +0.080867 |

The old near-origin equilibrium is inside the sampled position range, while the post-update detected equilibrium is outside it. The raw advantage mean is +0.070870 and its standard deviation 0.048747. Rewards are high throughout (0.950268–0.998745). The batch thus heavily represents already-balanced behavior rather than the reset/recovery conditions where evaluation failures occur.

This makes limited trajectory coverage and interference with recovery feedback concrete suspects. **It does not yet distinguish biased/noisy GAE action credit from genuinely beneficial sampled-state policy changes that fail to generalize.** No independently estimated Q/V oracle was used or measured in this audit.

## Minibatch and optimizer localization

Both ordinary and instrumented calls to the actual `PPO.train` reproduce update 54's full actor/critic, Adam state and final RNG states exactly. Every epoch visits all 2,048 rows once; all 320 index sets and actor snapshots are retained. Incoming Adam step counters are 16,960 and outgoing counters 17,280 for all 13 parameter tensors.

The first detected stable-to-unstable nearest-equilibrium change occurs after **minibatch 17**, within epoch one. Stability subsequently returns and is lost again; crossing minibatches are 17, 62, 115, 163, 242, 262, 286 and 298. The final minibatch remains unstable. This is not a single isolated minibatch that permanently explains the entire later training history.

The root structure also changes: before the update the grid locates one stable central upright equilibrium and two unstable outer equilibria; afterward it locates only one upright equilibrium in the scanned interval and that one is unstable. Reported gains are evaluated at each actor's own detected equilibrium, not at one fixed identical state. Root disappearance and changes in the nearest-root identity must not be misdescribed as continuous tracking of a single root.

A contrary diagnostic finding: **critic gradients are not the dominant clipping contribution in most minibatches of this update.** The median critic share of squared pre-clipping gradient norm is only **0.003329** (about 0.33%). Median actor/critic/std gradient norms are 1.64883 / 0.08778 / 0.08837. Clipping is active on 285/320 minibatches, with median scale factor 0.302885. Critic gradients exceed half the squared norm in 2/320 minibatches (maximum share 83.32%); at the first crossing their share is 37.27%, and it is below half at all eight crossing minibatches. This does not support systematic critic-norm domination, but it does not isolate the causal effect of clipping or critic estimates.

Clearing Adam history in a separate frozen-batch negative control changes the resulting parameters (maximum absolute difference 0.0227742) and still yields an unstable detected equilibrium (radius 1.00033568). This demonstrates why weight-only restarts are not faithful replays; it is not a proposed reset-Adam remedy.

## Validation, costs and limits

Independent analytic-MLP and full nonlinear held-command RK4 finite differences verify the before/after stability calculations. Maximum force-Jacobian discrepancy is 3.54e-8 and map-Jacobian discrepancy 4.04e-10. These are float64 local/noiseless diagnostics of the exact stored float32 actor weights. The sign-bracket grid can miss tangent roots; no global/noisy/hardware stability guarantee follows.

The independent read-only analyzer verifies the original archive digest and **210 manifest-listed payloads**, rechecks actual checkpoint/final state equality, historical weights/records/training episodes, frozen Adam budgets, declared first-event selection, complete minibatch coverage, final captured actor, gradient norm partition, and independent surrogate arithmetic. **19 reader tests pass**: 13 synthetic/corruption controls and six actual-evidence mutations. The actual-evidence mutations reject a later selected event, missing update, changed checkpoint, corrupted final Adam, duplicated minibatch row, and changed incoming frozen Adam history. A first test invocation emitted an unclosed-reader ResourceWarning; closing that reader removes the warning without changing numerical results or acceptance criteria.

Cost: **524,288 training interactions** across two exact same-seed prefixes, not two independent training trials; **621,033 evaluation steps** across 960 episode records (576 prefix records plus 384 before/after records); three frozen-update executions reuse the existing batch and add no environment collection. There are no Monte Carlo-oracle value estimates, extra critic-fitting passes or production interventions.

The read-only reduction ran locally under Python 3.13.5, NumPy 2.3.5 and Torch 2.10.0+cpu. Training and frozen-update execution occurred only on the pinned runner, not in that local analysis environment.

Original result artifact **11431717887**, 9,723,610 bytes, SHA256:

`b1ddcbd2d4a31addec42b6a1c9961ec1655420177a59dd500ec2b35a0f261895`

The evidence package retains that original ZIP unchanged, runnable analyzer/tests, result JSON, report, source identities and validation logs. It contains a *frozen optimizer transaction*, not a serialized resumable native environment checkpoint; its prefix can be rebuilt through the recorded source and seeds.

## Next bounded question

Measure independently correct action credit for this exact frozen rollout in the native simulator. That separates a wrong/noisy critic/GAE signal from policy changes that improve sampled-state behavior but damage recovery outside the fitting distribution. The current result identifies and faithfully replays a damaging ordinary update; it does not yet provide a tested correction. Do not introduce a curriculum, rollback, critic-repair phase or evaluation-selected stopping rule merely to hide the witness. Issues #33/#35 remain open for a reliable, predeclared held-out-qualified training path.
