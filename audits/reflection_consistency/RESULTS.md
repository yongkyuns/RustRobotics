# Frozen reflection consistency: modest gains, not robust control

September 20, 2026 UTC. RustRobotics issues #35 / #33.

## Decision and completed execution

**The frozen odd projection increases outward-recovery completion from 474/512 to 483/512, but none of the six registered paired-history contrasts is statistically resolved and the original absolute reliability requirements fail in all three panels.** Some previously failed cases are repaired; two previously successful cases fail under the projection. No inference wrapper, training change or production controller is promoted.

The original GitHub run **35486221478** completed successfully on its first attempt at 03:59:20 UTC. All nine jobs passed: native preflight plus all eight frozen-policy evaluations. Executed source is **d11b9fbf99edd0d221c4dd179a421ca46466b6eb**, branch `audit/ppo-reflection-consistency-20260919`. Registration [5747262659](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5747262659) predates execution. Pre-execution clarification5747271701 fixes only the arithmetic count of the explicitly specified grid: 225 points, not675. No grid coordinates, policies, cases or statistical rules changed.

Earlier status/progress documents describe intermediate queued or partial states and are superseded by this complete report. No missing history was excluded from the final analysis. A separately declared LOCAL execution failed strict historical replay and is retained below; it is neither a successful CI measurement nor additional comparative evidence.

## What was held fixed

Freeze all eight final global-normalization actors, seeds41001–41008, from source-normalization run35482824488, source a799c6d35c3753266fa382230d734e2d7de18312, checkpoint512. No weight, critic or optimizer is trained. All observations, rewards, force limits, noise, disturbances and the corrected nonlinear RK4 plant remain those of PR38 source4739f370558b9443708c920ac30614f86e3c07bb.

For observation o=[x,velocity,angle,angular_velocity], compare three predetermined latent means:

- **Original:** mu(o).
- **Reflected:** -mu(-o), negating all four coordinates.
- **Odd projection / symmetric combination:** [mu(o)-mu(-o)]/2.

The existing latent Gaussian innovation is then added, followed by one20*tanh transform. This is a combination of latent means, not an average of already-squashed forces. There is no fitted coefficient, reference controller, teacher, reward modification, privileged observation, adaptive switching or evaluation-selected action. The odd mean is exactly antisymmetric by construction; individual stochastic commands need not be opposite when the SAME rather than negated innovation is used.

A deployed odd mapping would require two network evaluations per action versus a usual original actor's one. This diagnostic computes both means for every mode to record them, so it is NOT a relative-latency benchmark. Averaging nonlinear controllers is not guaranteed safe or equivalent to learning a symmetric controller from scratch.

Native preflight confirms the actual noiseless environment's reflected one-step states, observations, rewards and endings match, including clipping/boundary cases. This supports the symmetry premise for the tested plant. It does not claim unchanged stochastic noise realizations themselves produce reflected paths or that every state has a unique optimal action.

## Fixed evaluation and provenance

Each seed/mapping has64 noisy five-minute deterministic nominal,64 noisy five-minute stochastic nominal and64 sixty-second stochastic outward trials: **4,608 total episodes**. Deterministic refers to mean policy actions, not a noiseless environment. Initial conditions and environmental/action keys pair the mappings within each panel. Different panels use different keys and are not extensions of one another.

These are the EXISTING source-normalization cases on evaluation domain seed+0x04000000, not new held-out qualification. All1,536 original-policy episode keys, caps, durations, endings, return totals, excursions, force RMS and centring reproduce their immutable historical records exactly in CI. True failure ends the trial immediately; no reset conceals failure.

Complete replication0 trajectories are retained for every seed/panel/mapping. Other episodes retain aggregates. The eight trained actors remain eight history units; reflected/odd mappings and repeated episodes are not additional independently trained agents.

## Full sustained results

| Frozen mapping | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Original |504|503|474|
| Reflected |504|503|479|
| **Odd projection** |**505**|**504**|**483**|

Odd completion rates are98.633%,98.438% and94.336%. Every survivor in every mode meets the final-ten-second bounds |x|<=0.5m and |angle|<=0.1rad. These tighter bounds were not required throughout the whole episode and do not remove failures from the denominator.

All44 odd-projection failures are cart-position-limit failures:7 deterministic nominal,8 stochastic nominal and29 outward. Original failures are7position+1angle in deterministic nominal,8position+1angle in stochastic nominal,and38position in outward. Eliminating the two observed angle failures under projection is not proof of pole stability throughout the operating region.

### Every history

Each cell is deterministic nominal / stochastic nominal / outward completions, each out of64.

| Seed | Original | Reflected | Odd projection |
|---|---|---|---|
|41001|64 /64 /52|64 /64 /54|64 /64 /60|
|41002|64 /63 /58|63 /63 /59|64 /63 /58|
|41003|61 /62 /63|62 /63 /63|62 /63 /63|
|41004|64 /64 /62|64 /64 /62|64 /64 /62|
|41005|63 /63 /61|63 /63 /62|63 /64 /62|
|41006|64 /64 /59|64 /64 /61|64 /64 /59|
|41007|64 /64 /59|64 /64 /59|64 /64 /60|
|41008|60 /59 /60|60 /58 /59|60 /58 /59|

Odd outward completion improves three histories, regresses one and ties four. Seed41001's eight-case gain supplies most of the net nine-case outward improvement. Seed41008 loses one stochastic nominal and one outward success. The projection therefore is not a universally harmless inference correction.

### Paired repairs and losses

| Odd versus original, same cases | Previously failed, now succeeds | Previously successful, now fails |
|---|---:|---:|
| Deterministic nominal |1|0|
| Stochastic nominal |2|1|
| Outward recovery |10|1|

Across all1,536 tested original cases,13 failures are repaired and2 successes are lost, net11. The reflected mapping instead repairs12 outward failures while losing7 outward successes, net5. Counts with the same total can conceal changed identities; all paired identities are retained.

Initial-side outward counts are descriptive: left starts have260 attempted cases and right starts252. Original succeeds243left/231right; reflected236left/243right; odd243left/240right. These are not matched mirror pairs and do not establish a universal side-specific advantage.

## Registered paired-history statistics

The six primary contrasts are odd-minus-original completion and discounted return in each of the three panels. Intervals are two-sided99% Student intervals with Bonferroni adjustment over those six contrasts, using eight paired history means, df7. Reflected contrasts are retained as descriptive controls, not additional primary tests.

| Odd minus original | Mean effect | Adjusted interval |
|---|---:|---|
| Deterministic nominal completion |+0.195 percentage points|[-0.770,+1.161] pp|
| Deterministic nominal discounted return |+0.321329|[-0.196404,+0.839061]|
| Stochastic nominal completion |+0.195 percentage points|[-1.555,+1.946] pp|
| Stochastic nominal discounted return |+0.257650|[-0.080317,+0.595617]|
| Outward completion |+1.758 percentage points|[-6.028,+9.544] pp|
| Outward discounted return |+0.995419|[-0.841067,+2.831904]|

**All six intervals cross zero.** The observed gains are retained, but no contrast establishes a resolved history-level improvement under this registered analysis. This is not proof of zero effect, equivalence, or certain population harm. Approximate finite-history intervals are not safety certificates.

The reflection protocol was a descriptive diagnostic, not the previous normalization experiment's advancement rule. No new acceptance rule or winner selection is introduced here.

### Original absolute reliability screen, separately reported

Each panel requires at least507/512 pooled completions, every history at least61/64, a three-panel-family one-sided history-level lower bound of95%, and at least99% centring among survivors.

| Odd-projection panel | Completed | Weakest history | History lower bound | Result |
|---|---:|---:|---:|---|
| Deterministic nominal |505/512|60/64|95.568%|FAIL|
| Stochastic nominal |504/512|58/64|94.085%|FAIL|
| Outward recovery |483/512|58/64|90.620%|FAIL|

All panels fail pooled and weakest-history requirements; the latter two also fail the lower-bound requirement. Survivor-only centring passes but cannot override these failures. This reused cohort would not be fresh or hardware qualification even if the numerical thresholds passed.

## Concrete recorded benefit and its limits

In the predeclared full trace for seed41005, stochastic nominal episode0, the original mapping hits the rail after111 steps (1.11s); the reflected mapping hits it after75 steps (0.75s). Their odd combination survives all30,000 steps (300s), with maximum cart excursion2.274005m and final-ten-second centring.

All three start from the same state and receive paired environmental/action innovations. This supplies an actual bounded learned-policy combination that succeeds on a case where both component mappings fail. It does not show that averaging always helps, that one changed action explains recovery, or that the combination is a trained robust controller. The provided plot displays the first five seconds of the retained full trajectories; the 300-second outcome is verified from the complete trace.

The original fixed225-point grid measurements also reproduce in native output: all eight actors have nonzero antisymmetry defects, while the odd mean removes them algebraically. A symmetry defect is not an error against a known optimal action. This experiment establishes both a representation property and its limited observed control effects, not a complete causal explanation of the earlier PPO failures.

## Verification and retained runtime failure

CI preflight passes73 native unit/audit tests, seven ordinary balancing controls, nine ordinary learning/evaluator controls, strict Clippy, formatting and protected-source restoration. The current diagnostic endpoint is ignored in preflight and explicitly executed by the eight measurement jobs. Two historical heavy integration endpoints remain ignored. No new production browser/platform matrix was run for this audit-only source.

Independent primary verification checks all9 published ZIP digests and221 payload hashes, all4,608 outcomes,1,800 grid probes,and72 full traces containing1,429,579 steps. Original actor/evaluation files bind to retained predecessor manifests. Full predecessor ZIPs were hash-verified in native preparation but are not duplicated in the current result archives. Maximum independent errors are actor5.790e-7,command1.984e-6,dynamics2.353e-7,reward1.406e-7,and aggregate return9.664e-13, within unchanged tolerances. Other episode paths and unexported random draws are not independently regenerated. No optimizer transactions exist in this frozen study.

The offline suite passes **37 tests**:25 original property/unit tests,10 additional synthetic trace-corruption tests,and2 regression tests for the separate failed-local assertion reader. Synthetic fixtures are not measured controller episodes.

### The separate local attempt is NOT comparative evidence

While CI child jobs were queued, comment5747362636 declared a separate execution before local episode outcomes. It used the exact published executable (SHA256 a287ba162d9e5ff1ffb9f7675b0bd07811f7df171cb8f71d2008fa88174da0ac) on local x86_64/glibc2.41, with no recompilation or experiment change. Its73 native tests passed. The original GitHub workflow was untouched.

All eight local seeds stopped at the FIRST original-policy episode's strict historical-return equality assertion, before any reflected or odd episode. Each reached the same30,000-step cap and matched the asserted historical key/cap/duration/ending, but total returns differed by approximately1.4e-6 to7.35e-5. No tolerance was relaxed, no seed was rerun, and no local outcome enters the primary comparison.

Recorded traces place the first difference in commanded force, while that step's physical state, observation, mean, latent and action innovation still agree. The differences are approximately0.95–1.91e-6N. This localizes the initial discrepancy to numerical force computation; a runtime math difference is plausible, but its precise library/CPU cause is not uniquely established. It is not a diagnosis of the RL robustness failure.

All eight failed local original-only traces independently reconstruct their own actions, dynamics, rewards and aggregate totals within the unchanged offline tolerances. That does not override the FAILED stricter native historical equality. An initial postmortem reader reversed the expected/actual operands in Rust's assert_eq output; it was corrected against immutable native line216, and the original failed reader log and regression tests are retained. No measured values or tolerance were modified. These local records are explicitly separate and never called GitHub artifacts or extra independent episodes. Comment5747397126 records the failure.

## Accounting and disposition

Primary CI diagnostic: **99,326,189 evaluation transitions, zero main training transitions**. Failed local execution: **240,000 additional original-only evaluation steps**, zero reflected/odd comparisons and zero main training transitions. Generic native tests/preflight experience and uninstrumented inference are additional, not a claimed all-inclusive efficiency benchmark.

The complete evidence retains all9 unchanged CI archives,8 separately identified failed-local archives, their bound historical trace inputs, numerical reports, exact native sources/build, independent verifier/tests, logs and replay instructions. Offline replay executes no native binary, performs no training and contacts no service. All contrary outcomes and failed assertions remain visible.

**No production adoption.** Keep global normalization and the existing unwrapped learned policy behavior unchanged. Symmetry may be a useful representation property to test in learning, but this frozen post-processing result does not qualify it as a controller replacement or establish better from-scratch training. PR38 remains draft/open at4739f37; master, deployed assets, learned weights and production defaults were not changed or merged. Issues #33/#35 remain unresolved.