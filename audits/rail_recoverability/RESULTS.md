# Rail recoverability: many failures are avoidable, neither reference is universally reliable

September 19, 2026. Issues #35/#33. Production baseline remains PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**A fixed bounded reference avoids 65 of the 85 learned-policy failures in the two tested panels without losing any of the learned policy's successes.** Disabling policy action sampling does not change aggregate completion counts and changes only two outward case classifications, one in each direction. The residual failures are therefore not adequately explained by exploration noise or by assuming every initial condition is physically unrecoverable.

This is a diagnostic result, NOT a replacement learner, installed controller, safe-switching policy, or deployment qualification. The better reference still fails two nominal and eighteen outward cases. A failed reference or rescue is NOT proof that no other feasible control sequence can recover. No controller was fitted to these outcomes or supplied to PPO. There are **zero training transitions** in this study, and production defaults/PR38/master/deployment remain unchanged.

## Fixed protocol and execution

[Protocol before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5743895660). Branch `audit/ppo-rail-recoverability-20260919`. Successful executed head **b26194a192d57598445eb5484e42348eef2d7329**, [run35458514720](https://github.com/yongkyuns/RustRobotics/actions/runs/35458514720). All nine jobs complete successfully: repaired preflight and eight frozen-policy measurements. Each full measurement runs once. Earlier failures occurred before any measurement and remain archived.

Inputs are the final recovery-union policies of fresh-study seeds41001–41008 from run35453178361. Those seeds are now exposed development evidence, not a newly held-out cohort here. Original candidate ZIP and member hashes are verified before use. All1,024 historical stochastic episode records reproduce exactly: initial/noise keys, lengths, endings, undiscounted/discounted returns, excursions, centring flags and force RMS.

Two original panels are replayed, each with64 cases per seed: five-minute stochastic nominal starts, and sixty-second stochastic outward starts. Outward displacement is0.8–1.2m, same-sign velocity0.4–0.8m/s, angle±0.15rad and angular velocity±0.3rad/s. No reset masks failure. Each initial condition and environmental random stream is paired across four fixed modes:

1. `learned-stoch`: original frozen learned policy and original action innovations.
2. `learned-mean`: the same network's mean action, without policy sampling; observation noise, actuator noise and disturbances remain.
3. `lqr-discount`: clipped linear feedback derived from the actual local reward with gamma=f32(0.99).
4. `lqr-undiscount`: the same derivation with gamma=1 and identical cost weights.

Reference controllers use only the ordinary NOISY observation vector. They do not receive true physical state, future disturbances, success labels or learned recovery actions. Physical state is used only to initialize/clone the simulator in the postmortem branches. Commands remain bounded at20N. The environment's existing actuation noise/disturbance is added afterward, so applied force can reach21.15N; that contract is unchanged, not an unnoticed actuator-limit relaxation.

## Analytic reference, not a tuned controller

The exact upright RK4 Jacobian is used, not a different continuous-time step or a linear replacement for the evaluation plant. The environment charges successor-state cost, so the discrete cost includes the state/action cross term: Qs=A'QA, Rs=R+B'QB, N=A'QB. Q=diag(f32(.2),f32(.02),1,f32(.05)), R=f32(.001). A generalized discrete Riccati solution is derived in f64 before outcomes. No gain sweep or post-outcome adjustment is performed.

For u=clip(-K*noisy_observation,-20,20), the gains are:

- Discounted: [-5.513626381392378,-10.400541538255077,105.59570546384029,42.01035972313336].
- Undiscounted: [-13.261027299599009,-19.480814855267923,153.4860483684778,63.73442126616264].

The Rust preflight independently derives both gains by symmetric Riccati iteration. The offline verifier uses SciPy's generalized DARE solver and checks residuals and nominal local eigenvalues. These local calculations do not certify clipped noisy nonlinear global stability. Actual evaluation always runs the original nonlinear RK4 environment.

## Full-policy results on identical cases

| Fixed controller | Five-minute nominal /512 | Sixty-second outward /512 |
|---|---:|---:|
| Learned stochastic |504|435|
| Learned mean, sampling disabled |504|435|
| Discounted LQR reference |510|494|
| Undiscounted LQR reference |509|487|

Every survivor in these panels also satisfies the final-ten-second centring/upright condition, abs(x)<=0.5m and abs(theta)<=0.1rad. That does not certify these tighter limits for the whole episode. All failures, including both references' failures, are position-limit failures.

### Paired outcomes, not just pooled totals

| Alternative versus learned stochastic | Learned failure rescued, nominal | Learned success lost, nominal | Learned failure rescued, outward | Learned success lost, outward |
|---|---:|---:|---:|---:|
| Learned mean |0|0|1|1|
| Discounted reference |6|0|59|0|
| Undiscounted reference |5|0|53|1|

Thus the discounted reference gives constructive feasible control sequences for65 of85 learned failures under the same tested noisy-observation/action/plant contract. The remaining20 cases are unresolved, not certified impossible. The undiscounted reference is NOT uniformly better: it loses one outward trial the learned policy completes, and has more failures than the discounted reference overall. Increasing feedback aggressiveness or removing discounting is not established as a reliable correction.

Removing exploration improves mean discounted return on every seed, but does not repair rail containment. Its paired seed-level mean effect is+0.409527 [0.387052,0.432003] for nominal and+0.389154 [0.361797,0.416510] for outward. There are still eight nominal failures and77 outward failures. This is a deployment-action-sampling comparison, not a test of how exploration affected training.

Discounted reference effects versus learned are+1.132605 [0.824650,1.440560] nominal and+3.446715 [1.515455,5.377976] outward. Undiscounted effects are-0.199649 [-0.470287,0.070989] nominal and+0.491298 [-1.661891,2.644487] outward. These are descriptive pointwise99% Student intervals across eight paired seed means, not pooled episodes treated as independently trained agents, simultaneous family bounds, or hardware safety guarantees.

### Every outward-panel seed

| Seed | Learned stochastic /64 | Learned mean /64 | Discounted reference /64 | Undiscounted reference /64 |
|---|---:|---:|---:|---:|
|41001|62|62|64|63|
|41002|41|41|63|62|
|41003|59|59|61|59|
|41004|59|59|62|61|
|41005|47|48|59|58|
|41006|49|48|63|63|
|41007|60|60|61|60|
|41008|58|58|61|61|

The worst learned seed41002 is especially informative: mean-only actions leave41/64 completions unchanged, while the unchanged discounted reference reaches63/64 on those exact cases. No reference was tuned specifically for that seed.

## These failures occur during initial recovery

All85 original stochastic failures occur within0.46–11.84seconds, with pooled median1.55seconds. Nominal failures occur0.78–6.32seconds after reset (median0.94s); outward failures occur0.46–11.84seconds (median1.59s). Thus the present frozen-policy deficit is concentrated in the initial recovery transient, rather than a policy surviving for minutes and then drifting into the rail. This does not invalidate the separate earlier finding of deterioration across TRAINING checkpoints.

This duration analysis is descriptive over all original failures, not a selection rule for which episodes to report. It is consistent with the known difference between near-centre reset trajectories used for supplemental fitting and the more extreme outward test starts. It does not uniquely establish inadequate state coverage or identify a particular erroneous advantage.

## Postmortem recovery branches: successful early does not imply successful late

For every original learned-stochastic failure, keep its exact state, noisy observation and random-generator cursors at0%,25%,50%,75%,90% of the time to failure. At each saved anchor, run the FIXED undiscounted reference for60seconds or first failure. Only the external evaluation clock restarts. Physical state, observation and future environmental RNG continue from the actual learned trajectory.

These intervention times require hindsight and are not a deployable switch rule. Selection is conditional on a known failure; the branches are not independent evaluation episodes or estimates of a general recovery probability. Every predefined branch is retained, whether successful or not.

| Fraction of original failure time already elapsed | Nominal failures recovered /8 | Outward failures recovered /77 | Total recovered /85 |
|---|---:|---:|---:|
|0%|5|53|58|
|25%|5|31|36|
|50%|2|28|30|
|75%|0|8|8|
|90%|0|1|1|

All successful rescue branches also finish centred/upright. The60second nominal rescue at time0 is a different horizon from the original five-minute nominal reference; it must not be treated as a new five-minute pass. For outward cases, both time0 horizons are60seconds and the scores reproduce the corresponding reference episode exactly.

The counts show loss of recoverability UNDER THIS PARTICULAR REFERENCE as the learned trajectory evolves. They are not exact viability boundaries. There are no observed rescue successes among cases this reference failed at time0, but that does not prove mathematical monotonicity or rule out another controller.

### Recorded example, not a selected acceptance case

Seed41002, outward replication0, is already a predetermined full-trace case. The learned policy hits the rail at6.28seconds. With the same undiscounted reference from the start, it survives60seconds and settles at the centre. Switching at the50% anchor (3.14s) also survives, with maximum cart displacement1.785869m.

At75% (4.71s), its saved physical state is approximately[1.774930m,0.439496m/s,-0.002925rad,0.066354rad/s]. The reference still recovers, reaching at most2.290128m and surviving the complete60second branch. At90% (5.65s), the same reference fails after another18ticks,0.18seconds. The associated plot shows the original trajectory and the retained first10seconds of the successful rescue; the complete60second survival is supplied by the verified aggregate outcome, not fabricated by extending the plotted trace.

This is a constructive late recovery witness and a nearby unsuccessful intervention, not proof that5.65seconds is the last possible recovery time for every admissible controller. One different case (41006/outward48) recovers even at its90% anchor; that contrary late success is retained.

## Verification and retained failures

The final preflight passes **73 native unit/diagnostic tests**, seven ordinary balancing controls and nine ordinary learning/evaluator controls, strict Clippy, formatting, compiled-runner discovery and protected-source restoration. One new diagnostic endpoint is ignored at preflight and explicitly invoked in eight measurement jobs. Two separately ignored historical heavy integration tests are not counted as executed. The production platform/browser matrix is not rerun for this evaluation-only branch.

The initial run35458239722/714300159f051408153c3c4cdb3438eb1710554b stops before measurements on strict Clippy's fixed-size byte-decoding rule. Equivalent array-chunk decoding corrects it. The next run35458325070/bac9c3d69518969bfd33b23eb4cf00c80c47cb29 passes Clippy but its independent Riccati iteration fails convergence. Enforcing the expected symmetry of the quadratic-form iterate fixes the numerical oracle; reference gains and original convergence/gain tolerances remain unchanged. Both failed preflight artifacts remain unchanged (68 and69 payload members). No full policy measurement is selectively repeated.

All eight result ZIP digests and645 payload member hashes verify; the successful build has73 verified members, and eight nested original policy archives have1,936 verified members. Every original stochastic record matches exactly. Checks cover4,096 fixed-policy episodes,425 rescue branches and573 exported trajectories/prefixes totaling1,300,894 transitions. Every rescue anchor matches its original saved physical state/observation and timing. Native clone controls cover continuing RNG state; hashes alone are not claimed to regenerate hidden random streams.

Independent reconstruction checks all exported nonlinear transitions, noisy-observation bounds, learned/reference action transformations, rewards, terminal/timeout flags and available full-episode aggregates. Maximum errors: dynamics2.501e-7, commanded force2.112e-6,reward1.591e-7. The largest reference/actor-output discrepancy is3.713e-6 (includes f32 display of an unsaturated f64 reference force). Full recorded reward/discounted/RMS sums reconstruct exactly. The remaining trials and rescue suffixes retain aggregates, not every step; no regeneration of all unexported trajectories or Adam histories is claimed. There is no optimizer execution in this study.

The offline verifier passes **36 tests**, including mutation of actual recorded commands, physics, rewards, return totals, historical binding and random-domain identities. One earlier unit test exposed a missing snapshot byte-length check before NumPy decoding; the explicit check was added, with its initial log retained. No measured data, acceptance threshold or reconstruction tolerance changed. Offline Python/NumPy/SciPy versions are recorded separately from native Rust1.98.1/Burn0.20.1. Offline replay does not execute the native binary or train.

## Costs and disposition

Zero training transitions. Full-policy evaluation uses **71,960,552 simulator transitions**; rescue branches add **803,927**. Generic preflight/control simulation is additional and not completely aggregate-instrumented. Reference design is not a learned sample-efficiency comparison.

Keep the constructive reference successes and early/late recovery anchors as diagnosis. They rule out inevitable physics for many tested failures and reject disabling policy sampling as a sufficient fix. They do not qualify either reference, justify hardware switching, or identify one universal cause for the remaining failures. The next learner-level correction should target the initial off-centre/outward recovery transient while preserving nominal balance, with an explicit bounded sampling change and a new confirmation cohort after development.

Self-contained evidence: `rustrobotics-ppo-rail-recoverability-evidence.zip`. It includes unchanged successful/failed current archives, original policy evidence nested inside, exact compiled-source provenance, all outcomes and retained trajectories, independent verifier/tests, analytic derivation, numerical tables, example plot and offline replay instructions. No production defaults, PR38/master/deployment, or learned weights were modified.