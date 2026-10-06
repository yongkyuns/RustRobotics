# Update 54: real sampled-state improvement, worse recovery

**RustRobotics PPO investigation · October 6, 2026**

Diagnostic evidence, not a new training recipe or sustained-balancing qualification.

## Conclusion

For frozen stock-SB3 seed203/update54, independent native-simulator continuations show **genuine average improvement from the tested training-trajectory states**. The same update previously worsened ordinary reset-distribution evaluation. The tested conditional action-credit estimates do **not** support a wrong-sign-credit explanation.

This moves the strongest explanation for this witness toward **limited trajectory coverage and damage to recovery outside the fitted region**. It does not prove every harmful PPO update shares that cause, imply the critic is perfectly calibrated, or establish a working correction. No oracle labels were fed to training; production/defaults/PRs/master remain unchanged.

## Protocol and source

Fixed protocol: issue35/comment6022369302, before new outcomes. Retained preparation failures and numerical-environment adjustments: comments6022625868/6022775498. Exact capture completion: comment6023018213.

Original witness artifact11431717887, successful run37505417538, execution source0e1246bbea16894e29f6161d177a864b6008e1bb. ZIP SHA256:

`b1ddcbd2d4a31addec42b6a1c9961ec1655420177a59dd500ec2b35a0f261895`

The native environment derives from clean-reference sourceda6623ae4b5878ae5079bf05c18c3d327570bc16, production base95c670b9f4618a11dc5439b026caf385a39718c8, the unmerged #38 including #37. The base conditional audit and additive ABI were committed in22689c446958df2bd660d505d3296dad91eaa5b8/c2bb94c5b903e0bc87d9784b5c121477d435ce88. Equal-shaped actor inference is specified by1332fbacff5c0cfea5e93faacb9c909d216fced6. The executed local independent-group launcher isb2bac76967231ba723baefb10afc1866c2a1c9fe. Actual sources, compiled library and hashes are retained in the evidence package.

## Experiment

Fixed64 rows:16+32*k, k0..63, from the exact2048-row fitting rollout;64 independent future draws per row. Four common-random branches: old throughout; new first action then old; recorded training action first then old; new throughout.

Each branch begins with the captured true physical state and recorded noisy observation. Later policy inputs are normal noisy observations; hidden state never enters the controller. The actual Rust PendulumEnv supplies unchanged dynamics, clipping, noise, disturbances, reward and physical termination. An additive diagnostic ABI provides from-state initialization, fresh future-RNG seeding, peek and clone. It does not reimplement the physical equations.

The training collection clock is removed for counterfactual continuation under the continuing-task timeout interpretation. Physical failure still terminates; the audit imposes a new external horizon. Returns are stopped, **unbootstrapped** sums at512/1024/2048 steps (5.12/10.24/20.48s), gamma=.99 from the PPO recipe. These are not exact infinite-horizon oracles. The older reset evaluator uses its archived f32-promoted.99; historical and new scores are reported separately, not pooled.

Incoming-critic GAE is independently resampled with lambda=.95 and each row's original remaining rollout length2048-row. True terminal removes bootstrap; nonterminal cutoff uses the final pre-reset observation. The expanded estimator is checked against an independent TD-residual sum.

## Real returns improve from the sampled states

Effects are new-minus-old means on this fixed state panel. Intervals are approximate descriptive99% **Monte Carlo** intervals conditional on these states and this policy pair.

|Horizon|New first action, old thereafter|New policy throughout|
|---|---:|---:|
|512|+0.00038106 [+0.00035836,+0.00040376]|+0.03330793 [+0.03298205,+0.03363381]|
|1024|+0.00038242 [+0.00035969,+0.00040515]|+0.03497618 [+0.03464991,+0.03530244]|
|2048|**+0.00038242 [+0.00035969,+0.00040516]**|**+0.03498967 [+0.03466343,+0.03531591]**|

At2048 steps mean return is99.029319 old versus99.064309 new. Both complete4096/4096 continuations from these already-balanced starts. All four branches complete the horizon in the main panel; true-termination handling is exercised separately.

Not every state improves:62/64 row-level mean full-policy effects are positive and two negative. First-action effects are positive at56/64 rows and negative at eight. Every row and replicate is retained. There is no adverse-row exclusion or adaptive sampling.

The same frozen policy pair previously produced the separate fixed reset-panel result (keys910203): stochastic15s completions38/64→16/64 and mean discounted return80.45442→79.85350, change−0.600915, conditional paired-episode SE0.180417. Original exposed reset keys also deteriorate. These panels have different starts/horizons; counts and uncertainties are not combined. Sampled-state benefit is positive at every registered return horizon.

Thus the training-region improvement is not just a false surrogate signal: independent simulation measures it too, while recovery behavior worsens. The result supports coverage/interference as a concrete explanation for this witness, not a formal universal attribution.

## Action-credit decomposition

For the recorded actions, project credit onto the old-policy likelihood derivative in the exact policy-parameter update direction, including log-std. Independent finite differences and float64 Torch automatic differentiation check that derivative.

|Quantity on64-row panel|Estimate|
|---|---:|
|Original recorded raw GAE direction|+0.000213489|
|Return-based centered action-credit direction|**+0.000448759**;99%[+0.000423188,+0.000474329]|
|Resampled action-centered GAE direction|+0.000249849;[+0.000234735,+0.000264962]|
|Resampled absolute GAE direction|+0.000056455;[+0.000015495,+0.000097415]|
|State-dependent baseline contribution to absolute GAE|−0.000193393;[−0.000235116,−0.000151671]|

GAE underestimates the centered return-based directional signal by0.000198910 but does not reverse its sign. The independent first-action distribution comparison agrees: centered GAE predicts+0.000197496 versus return-based+0.000382424. The old critic underpredicts the stopped old-policy return by0.378857 on average, so it is not perfectly calibrated; that does not make the action effect negative here.

These are **raw directional diagnostics**, not a replacement computation of the normalized/clipped PPO objective. The recorded current action at each row is fixed; its future is resampled. The panel does not integrate every possible action/state or validate all2048 rows. No population policy-gradient or global correctness claim follows.

Each replicate index forms a whole fixed-state-panel average; its64 independent future-draw averages give the standard error. The temporally correlated states are fixed conditions, not independent training runs. Intervals are descriptive normal approximations, not multiplicity-adjusted recipe-acceptance tests.

## Small output changes conceal feedback changes

A separate read-only saved-weight analysis evaluates both actors at the **same2048 recorded observations**, rather than at different equilibrium locations.

|Diagnostic|Value|
|---|---:|
|RMS deterministic command change|**0.092583N**|
|Maximum absolute command change|0.256073N|
|Mean Gaussian KL old→new|0.00212950|
|Median position force-Jacobian component, old|**1.309692N/m**|
|Median position force-Jacobian component, new|**0.062553N/m**|
|Reduction in that median component|**95.22%**|
|Inputs with negative position component|0→398/2048|

Near-agreement in actions on a narrow visited trajectory does not imply preservation of feedback derivatives. This reinforces the coverage/recovery concern. All feedback components change: this is not a simulated intervention or proof that the position component alone causes failure. Float64 finite-difference controls agree within3.63e−8 across65 fixed observations per actor.

## Exact capture and numerical-environment failures

First hosted run37508853346 passed strict learner Clippy,74/74 native tests, original/additive ABI256-transition equality, clone/RNG and explicit-state/reseed controls. It failed the initial policy checkpoint before any training update or conditional outcome. Host AMD EPYC7763 differed from the successful reference audit's Intel Xeon Platinum8573C, despite identical installed package lists. This is a numerical-environment lead, not a proven one-variable cause.

Provisioning attempts37510159923 and37510560165 also received non-Intel hosts and stopped at the CPU-family gate before dependency installation/training. All failures remain preserved; none is relabeled successful.

Runtime transport37510903158 packages Python3.11/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0 without training. ZIP and inner tar hashes were verified. The interpreter/package binaries are intentionally excluded from the user evidence package; versions and recovery identifiers remain recorded.

On the local Intel Xeon Platinum8573C, capture passed exact initial/65536/262144 policy/evaluation equality, incoming/outgoing frozen54 actor/critic/Adam/rollout/RNG checks, final state equality, and full training-transition digest:

`c48b8d6caef121996229e9fb65de82d0b0759acbdd5f4c608387ae7a095ca252`

All2048 captured observations/actions/rewards equal the original fitting batch exactly. Physical states, observations and episode ages are retained. No trained weights were imported to bypass the prefix.

Both actors use identical inference batch shapes before branch selection. A same-policy identity continuation produces exactly equal returns, GAE, steps and endings. A separate forced one-step physical-terminal control confirms no later reward/bootstrap. Eight fixed groups execute with at most four independent spawned processes and one numerical-library thread each; their seeds, groups, sample counts and thresholds do not change.

## Verification, costs and recovery

The independent reader anchors the original witness ZIP and all210 original manifest payloads; checks captured physical identity, group membership, future random keys/draw hashes, complete array shapes, finite/boundary/budget invariants and step receipts; and independently reproduces all reductions. Maximum reduction discrepancy is1.36e−19. **21 evidence/statistic tests and10 estimator tests pass.** A fresh extraction of the final archive reproduces the independent analysis JSON byte-for-byte and passes all31 tests. Likelihood derivatives agree with independent float64 Torch automatic differentiation to1.74e−17 on checked inputs.

Two validation-side effects are retained. Python regenerated one archived reference bytecode cache; the first immutable-input check rejected it. Its generated bytes/log were retained and the original cache restored from the unchanged ZIP. All source/model/optimizer files were unchanged. The first fresh-package mutation-test invocation also missed the newly added terminal-control subdirectory in its temporary fixture copy. Copying that directory fixes the test setup; no numerical assertion, threshold, outcome, state or sample changed. Later validation disables bytecode writes.

Cost:262144 additional historical-prefix training interactions and315 resets, plus288 existing checkpoint-evaluation episodes; **33554432 main native continuation steps**;16384 identity-control steps and8 forced-terminal steps. Small separate ABI/learner controls add their recorded work. No oracle fitting, curriculum, rollback, new default, production merge or deployment.

Simulation uses pinned Python3.11.16/Torch2.8.0+cpu/NumPy2.2.6; independent read-only reduction uses Python3.13.5/Torch2.10.0+cpu/NumPy2.3.5. The conditional panel completed **locally**; the earlier hosted conditional runs remain failed preparation attempts, not green end-to-end measurement runs.

Evidence package **ppo-credit54-evidence-20261006.zip**, **41817959 bytes**, SHA256:

`4f3b5ef4db61f019bf0d590d1f8fde7e0f3df4d43b2807176fbe9fec1519acb8`

It contains303 checksummed payloads, including264 result-folder payloads, the unchanged original input/failure archives, captured physical trajectory, all state/replicate/arm estimates, contrary rows, executable audit/analysis/tests, reports, hashes and reproduction instructions. It deliberately excludes the transported third-party runtime. Recover the named conversation package; do not assume another conversation's sandbox path. Hosted preparation artifact11434825396 has SHA25601f1d0f1ed6b9c89734ad4a57e73f3100b7b35815524b3dc5db0e28d1568c455. CPU-gate artifacts11434741084/11434841718 and runtime artifact11434833708 are indexed in ARTIFACTS.json.

## Next bounded test

Test **ordinary multi-environment PPO at fixed total rollout size** rather than an oracle critic or recovery curriculum: e.g.one environment×2048 steps versus eight×256, preserving interaction budget,2048-row updates,minibatch size,epochs,model,reward,noise and evaluation rules. Shorter per-environment segments also change cutoff frequency, so report that tradeoff rather than claiming perfect one-factor isolation.

The hypothesis is that normal independent streams provide more representative experience and reduce loss of recovery feedback. It is not guaranteed; successful streams may still concentrate in similar regions. No such learning comparison was launched by this report. Any supported default still requires a fixed candidate and new predeclared held-out sustained-balancing qualification. #33/#35 remain unresolved; production PRs/master were not changed by this diagnostic work.
