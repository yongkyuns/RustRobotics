# Eight environments, one narrow state range: a harmful PPO update reproduced

RustRobotics · October 6, 2026 · Seed 201, eight environments, update 133

## Result

**A real eight-environment PPO update is now localized and verified.** Update 133 improves its actual collected-data PPO surrogate and critic fitting loss, yet substantially worsens independent noisy-reset control. Its entire 2,048-row batch contains no reset and occupies one small, already-balanced region on the left side of the track. Parallel streams did not provide representative recovery experience in this update.

This establishes a harmful update with narrow training support and much larger off-batch policy changes. It does **not** yet establish that the advantages give correct action credit on this eight-environment batch. The earlier seed203/update54 conditional-credit result cannot be transferred to this different update. No new training recipe or production default is proposed here.

## Selection and exact historical replay

Protocol fixed before new measurements: issue35/comment6024599337. Seed201 was chosen as the first numeric development seed. The selector was the first update after128 whose deterministic-command position derivative at zero observation crosses from positive to nonpositive. It selected **133**, without inspecting new evaluation outcomes. This is not a claim to have found the first harmful update in training, nor a geometric stability test.

Input is original pair201 artifact11441435840 from run37519334101, executed77f8624d419666ffdadbc00d7c73f0c5bb3aadd9. Its SHA256 is `22ea3b48e60b8199131fd5379e859017cc7b5212d58bbb5c06cb5b52166ce86c`. The previous complete comparison package was also independently replayed; its four outputs reproduce byte-for-byte and its70 test executions pass again. Those are prior-result verification, not new learning trials.

One full seed201 eight-environment history was replayed from random initialization, retaining live Adam, environment and random-number state and the original learn-call boundaries. **Every original coverage record, all four original actor/critic checkpoints, and the complete exported final policy/Adam/RNG/observation/episode state match exactly.** No checkpoint-only warm start, optimizer reset, replacement seed, or selected rerun was used.

Recorded direct versions are Python3.11, Torch2.8.0+cpu, NumPy2.2.6, SB3 2.9.0, Gymnasium1.3.0. The original worker is AMD EPYC7763; this replay uses AMD EPYC9V74 with AVX2 math dispatch chosen from the original CPU capability before replay. Actual byte equality, not a claim that different CPUs normally match, establishes this history's identity. The original native library and SB3 source files are hash-bound and unmodified.

All513 initial/updated actor-critic snapshots, all384 actual batches from updates129–512, actual applied actions for all eight streams, and the selected incoming/outgoing warm-Adam state are retained. Later position derivatives can return positive; the archive does not suggest a monotonic decline or select a deployment checkpoint.

## Independent before/after performance

Fresh ordinary noisy resets, common environment/action innovations within each policy pair, separate domains from training and prior evaluations. Deterministic tests last10seconds; stochastic tests15seconds. Inference always uses32 lanes, including inactive/padded lanes, avoiding policy-dependent matrix dimensions. There are256 deterministic and512 stochastic cases per policy; failure stops the episode.

| Metric | Incoming policy, update132 | Outgoing policy, update133 |
|---|---:|---:|
| Deterministic completions /256 |104|41|
| Stochastic completions /512 |217|57|
| Deterministic mean discounted return |79.867385|78.597483|
| Stochastic mean discounted return |79.703388|78.541571|
| Deterministic mean duration, seconds |5.843438|4.464961|
| Stochastic mean duration, seconds |8.204199|5.479902|

There are **63 deterministic and160 stochastic old-success/new-failure cases**, and **zero old-failure/new-success cases** in either panel. This is completion dominance on these cases, not a statement that every individual return declines. The incoming policy was already unreliable; it was not a previously perfect controller.

New-minus-old stochastic discounted return is **-1.161817**, with paired pointwise99% Student interval **[-1.341897,-0.981737]**. Stochastic completion changes by **-31.25 percentage points**, interval **[-36.5514,-25.9486]**. Deterministic discounted return changes by-1.269902[-1.529434,-1.010370]; completion by-24.6094percentage points[-31.6097,-17.6091]. These intervals concern independently drawn episodes conditional on this selected policy pair. They are not independent trained-agent replication, simultaneous familywise bounds, or held-out training qualification.

**Every failure in both policies and both panels reaches the pole-angle limit, not the position limit.** Among newly lost cases, deterministic failures occur3.05–9.87seconds after reset (median6.38), stochastic failures3.52–14.91seconds (median7.69). Thus this is not exclusively an immediate first-tick reset mistake. The position derivative was a selector, not proof that this coefficient alone causes the loss.

## The optimizer succeeds on its batch while control gets worse

The selected update's320 original Adam transactions were replayed from its saved warm optimizer state and actual batch. The outgoing actor, critic, Adam and random-number state match exactly. Original shuffle permutations and per-minibatch normalization are retained. No evaluator result enters the update.

| Collected-data quantity | Before | After |
|---|---:|---:|
| Clipped PPO surrogate, averaged over the320 actual normalization groups |approximately0|+0.002798477|
| Critic MSE against this batch's fixed targets |0.008488066|0.002520374|

Surrogate change is+0.002798471; critic target-fitting error drops about70%. Critic MSE is against bootstrapped fitting targets, not independent true-value calibration. The surrogate uses each actually drawn64-row minibatch's normalization, not a substituted whole-rollout normalization.

Separate NumPy float64 network/Gaussian/loss evaluation agrees with Torch: maximum log-probability discrepancy1.63e-6, value discrepancy1.65e-5, per-minibatch surrogate discrepancy2.34e-7. All320 post-clipping gradient vectors and outgoing parameter vectors are retained. Maximum recorded gradient norm is0.4999999883, consistent with the unchanged0.5 clipping limit. No new optimizer arithmetic defect was identified by these checks.

## What the actual eight-stream batch contains

All2,048 observations are on the negative-position side. All eight streams have zero episode starts; none terminates or times out in the batch, and all eight rollout cutoffs bootstrap.

| Observed component | Minimum | Maximum |
|---|---:|---:|
| Position, metres |-0.328444|-0.174703|
| Velocity, metres/second |-0.120037|+0.108218|
| Pole angle, radians |-0.006744|+0.009057|
| Angular velocity, radians/second |-0.053980|+0.048608|

Every pole angle is within approximately0.52degrees of upright. There are zero rows with absolute angle above0.05rad or0.1rad. The ordinary reset distribution permits angles up to0.25rad and angular velocities up to0.5rad/s. These bounds describe the unchanged task, not a new curriculum. Independent random streams have converged into essentially the same narrow balancing state range, rather than furnishing broad initial recovery experience.

A reset-free batch need not lack all recovery-like states, and a reset-containing batch need not be sufficient. Here the full observed ranges, not reset count alone, establish narrow angular/velocity support and one-sided position support. Raw advantage mean/std are0.075905889/0.052214563.

## Small fitting-domain changes, larger off-batch changes

Define the deterministic command as `u(o)=20*clip(mu(o),-1,1)`, not the expectation of a clipped random action. At zero observation its position derivative changes **+0.428602 to-0.406822N/m**; its command changes+0.160335 to-0.089051N. Other feedback derivatives also change, so this is not an isolated position-gain intervention. Zero observation is not assumed to be an equilibrium.

The RMS change in this command is **0.043456N across the2,048 training observations**, versus **0.554068N across all768 fresh initial reset observations**: approximately **12.75times larger**. Maximum changes are0.148697N versus1.557748N. This measures genuine state-dependent policy drift; it does not by itself prove correct action credit within the training region.

An additional explicitly **post-result exploratory** Gaussian-policy KL calculation uses exactly those same two state sets. Mean `KL(old Gaussian || new Gaussian)` is **0.00171722 on training observations versus0.16164026 on reset observations**, approximately94times larger. This is pre-command-clipping Gaussian KL; it is not labelled the exact divergence of the clipped action distribution. Gaussian standard deviation changes0.05015696→0.04883403 in normalized-force units. The KL formula agrees with independent Torch distribution KL to4.45e-16. This diagnostic does not alter the selection rule, evaluation panels, or primary analysis.

## Shorter rollouts also increase cutoff sensitivity

GAE and targets reproduce bit-for-bit from each stream's actual boundary data. An independently ordered forward sum agrees to3.83e-5 in absolute advantage for the selected batch. The exact backward reconstruction also checks all786,432 retained late-training rows.

For a row withL remaining rewards before a live rollout cutoff, the final-value coefficient in GAE is `gamma*(gamma*lambda)^(L-1)`. With gamma=.99/lambda=.95, the selected batch's mean coefficient is **0.06499474**. The analogous uninterrupted2,048-step stream has mean coefficient **0.00812434**: eight times smaller. In the actual256-step streams,304 rows have coefficient>0.1 and600 have coefficient>0.01.

This is **sensitivity to a unit error in the final critic value**, not measurement of such an error. The large absolute cutoff-value term must not be equated with a large error: other GAE terms can cancel it. No alternative bootstrap target, oracle critic, or training intervention was applied. This documents the cutoff confound rather than claiming to have isolated it as the cause.

## Verification, costs and scope

New validation includes **14 numerical/control tests and six actual-record corruption tests**. Altered observations, advantages, policy weights, minibatch permutations, missing evaluation records and reset Adam counters are rejected. The observer/plain short-training control matches policy, optimizer, RNG and live observation/episode records exactly. A128-record same-policy evaluation control yields exact casewise zero differences. The original source, archive hashes, complete checkpoint bindings and per-stream ending ledger are retained.

The offline reader independently checks513 saved networks,384 late batches/786,432 rows, all320 optimizer transactions and all1,536 primary evaluation records. It reconstructs retained complete trace scores and observation continuity for16 predetermined trajectories/11,786 transitions. Other evaluations retain complete case aggregates and initial observations, not every transition. Offline verification executes no native code or training. It does not regenerate every original gradient or unexported stochastic trace, and it does not independently rerun all native physics.

Costs of this continuation: **1,048,576 exact historical training transitions**, **8,192 short-control training transitions**, **964,521 primary evaluation transitions**, and **86,824 same-policy control evaluation transitions**. The selected optimizer-only replay adds320 Adam transactions but zero environment transitions. Generic construction/inference work is separate. There is no sample-efficiency or new trained-cohort claim.

Setup limitations are retained: direct local runtime installation first failed due to network access; diagnostic-only runtime workflow37524712255 subsequently supplied pinned offline dependencies. The initial analyzer test exposed an unintended float64 Boolean-mask promotion in its GAE reconstruction. Matching SB3's explicit float32 mask fixed the analyzer; no training data, algorithm, evaluator, or tolerance was changed. The original failed test log and pre-fix analyzer are retained. The full historical replay and new evaluation were each executed once, without a favorable-outcome retry.

**No production reward, noise, model, default, protected MuJoCo/control source, master branch, deployment or production PR was changed by this work.** No new held-out training cohort or production browser/platform qualification was run. Issues33/35 remain open.

## Remaining distinction

We now have an exact eight-environment harmful-update witness, not only separated checkpoints. Its narrow support and off-batch drift are concrete. The next bounded attribution is independent first-action and full-policy return measurement from this update's actual sampled states, using fresh future randomness. That distinguishes wrong action credit inside the batch from genuine on-batch improvement that generalizes badly. Those conditional-credit measurements were **not** performed here; neither the previous single-environment credit result nor the observed surrogate gain substitutes for them.

The companion `ppo-eight-env-update133-evidence-20261006.zip` contains executable capture/analysis/test sources, original pair201 input, all retained updates/batches/snapshots/actions, warm optimizer anchors, transactions, evaluation cases, reports and validation logs. Its offline reproduction instructions are in README.md. Pinned third-party runtime installers are not part of the evidence package.

## Completed package verification

The complete evidence ZIP is **100,219,085 bytes**, SHA256 **`6c0bb6f1d0ee03252e4127e354d59bb55c4b1d861627f750e623dcc2674a976c`**. A fresh extraction verifies **1,433 payload hashes**, reruns all **20 new tests**, and regenerates both `OFFLINE_VERIFICATION.json` and `gaussian-drift.json` **byte-for-byte**. This offline replay performs zero native or training transitions. The detached verification receipt records these checks and logs; the original archive digest remains unchanged.
