# State-local baseline repair: selective correction fails on both witnesses

23 September 2026. **The fixed outward-state-only correction fails on both exposed witnesses. At update 4195 it removes 99.67% of measured squared baseline error yet worsens recovery. Full correction improves recovery on both new external panels, but also loses a small number of nominal successes. No tested correction passes the local no-regression screen on these new domains.**

Protocol: issue #35 comment 5804954005, posted before the candidate replays and new evaluation. This is a numerical full-step and independent-simulator diagnostic, not a new native training run, affordable learned baseline or from-scratch qualification. Earlier native causal results remain valid for their original cases.

## Fixed intervention

Two witnesses: seed41006/update4140 and seed41004/update4195. The previously used observation-only proxy is `abs(o[0]) >= 0.5 && o[0]*o[1] > 0`, computed in float32. It selects204/1024 and237/1024 rows. The complement includes inward-moving and other recovery states and must not be called simply nominal or safe. The gate uses no future outcome, baseline-error label, advantage sign or start provenance.

For each witness retain the incoming actor and compare four complete numerical updates: original, full Monte Carlo baseline correction, outward-only correction and complement-only correction. Corrected rows use float32(original_return - float32(mean sampled V)); untouched rows retain original captured raw GAE bits. Recompute the unchanged whole-batch normalization. Untouched RAW advantages therefore do not imply untouched normalized advantages or an unchanged policy in those states. Full correction retains the earlier baseline-only formula, not a new critic fit.

All16 minibatches and their order, incoming actor Adam moments/counters, observations, sampled actions, old likelihoods, learning rate, clipping and critic targets stay fixed. No critic is fitted or installed. Simulator baselines retain actual pre-action states and far-cutoff critic bootstrap: diagnostic estimates, not exact conditional expectations or deployable observation-only values.

## New external behavioral results

Unchanged NumPy/PCG64 evaluator, fixed new roots2026092321/2026092322,512 paired cases per panel,2048 ticks at dt.01. Deterministic means no policy sampling, not a noiseless environment. These are numerical actors and external random domains, not Rust/StdRng parity or new independently trained agents. All15360 records are retained.

| Witness | Actor/update | Nominal det | Nominal stoch | Outward recovery |
|---|---|---:|---:|---:|
|41006/4140|Incoming|507|508|408|
||Original|507|508|369|
||Full correction|506|507|436|
||Outward-only|507|508|365|
||Complement-only|507|508|378|
|41004/4195|Incoming|510|510|405|
||Original|508|508|404|
||Full correction|509|510|429|
||Outward-only|510|510|392|
||Complement-only|507|507|404|

Outward-only loses43 recovery cases versus incoming in the first witness and13 in the second, gaining zero in both. Complement-only loses30 and1. At4195 outward-only preserves both incoming nominal success sets and improves nominal discounted returns, while worsening recovery. A state-selective baseline repair does not imply a beneficial localized policy change.

Outward discounted-return differences versus incoming, ordinary descriptive99% mean+/-2.586*SE intervals:

|Witness|Correction|Mean|Interval|
|---|---|---:|---|
|41006/4140|Full|+.511018|[+.392769,+.629268]|
||Outward-only|-.473451|[-.567902,-.378999]|
||Complement-only|-.131934|[-.222980,-.040887]|
|41004/4195|Full|+.196045|[+.063283,+.328807]|
||Outward-only|-.189583|[-.283454,-.095711]|
||Complement-only|-.046435|[-.143817,+.050946]|

Full correction gains28/24 aggregate outward completions versus incoming, but paired changes are29 gained/1 lost and26 gained/2 lost. It loses one nominal deterministic success in both witnesses and one nominal stochastic success in the first. Full correction consequently also fails the unchanged no-aggregate-completion-loss screen on these domains. All its nominal discounted-return intervals versus incoming include zero. Prior native absence of lost cases and these new external losses are finite-case evidence, not universal safety or population harm. Completion, total reward and discounted reward remain separately retained; intervals are not multiplicity-adjusted or training-seed bounds.

## Remaining error and complete update

At4195 the outward proxy contains99.6716% of measured squared baseline error. Correcting only it reduces residual baseline-error RMS from15.0845 to.86448. That remaining error is still27.454 times paired action-credit RMS.0314883. On the common original normalization scale, the complement's baseline-error gradient contribution has norm.070452 versus.033513 for action credit. Removing almost all squared error does not remove every influential finite-batch gradient error.

At4140 only31.2033% of squared baseline error is in the outward proxy. Its complement retains68.7967%, RMS2.17510, about124.286 times action-credit RMS. Its baseline-error gradient component has cosine-.91643 against action credit. Outward start provenance and being outward-moving now are not interchangeable coverage criteria.

|Witness|Original full-displacement credit projection|Full correction|Outward-only|Complement-only|
|---|---:|---:|---:|---:|
|41006/4140|-.062012|+.091235|-.055765|-.019605|
|41004/4195|+.006317|+.077583|-.027411|+.007445|

At4195 outward-only initially improves credit-gradient cosine from+.06551 to+.24584, but its COMPLETE optimizer displacement has negative credit projection. Full correction gives initial cosine+.95267. These are sampled local derivatives, not measured returns; behavior is measured separately above.

Residual-gradient decomposition is post-outcome interpretation and changed no gate/candidate/threshold. Global normalization, shared policy parameters and multistep Adam couple regions. This experiment does not isolate these from the discontinuous gate, remaining calibration errors, continuation noise or bootstrap bias. It does not prove every alternative state-dependent correction fails, or that lower value RMSE alone guarantees a useful actor step.

## Verification, cost and preservation

25 new actual-data/mutation tests pass with warnings as errors, including the fresh extracted package. Both original recorded actor/critic controls reproduce64 Adam transactions bit-for-bit from exported gradients/moments/counters/weights. Each main replay executes128 numerical actor transactions. All128 independent manual NumPy float64 gradient comparisons pass the unchanged1e-4 relative gate; maximum6.142924e-6. The32 original numerical actor checkpoints plus16 late full-correction checkpoints agree with their native references within the fixed2e-6 absolute parameter gate; maximum8.605421e-7. Numerical actors are not claimed bit-identical to native actors.

A complete numerical replay repeats171 files byte-for-byte. Both outward panels, all five actors, were rerun: all5120 repeated episode records and four panel files match exactly. This is reproducibility, not a new statistical sample. The four nominal panels were not rerun in full. Tests independently reconstruct all15360 summaries and paired contrasts from raw records.

Original-input verification checks976 declared payload hashes across the old Adam archive, independent-futures archive, nested paired-credit archive and late-result package. Self-contained inputs retain334 consumed original members with origin records; three inherited code bodies are separately bound to their original members. The package is not a substitute for every original historical archive.

New evaluation executes29,524,237 active simulator transitions; the two outward repeats add8,705,296. Replay, repeated replay and tests perform numerical optimizer work. No new Monte Carlo baseline estimation, critic fitting, continuing learner history, native CI job or from-scratch cohort. Original baseline-estimation cost remains inherited, not free.

Conversation bundle `rustrobotics-state-local-baseline-20260923.zip`:25,788,763bytes; SHA256295a7a028d2b7dee4c3f8ad456030ab0d7770a9bcb96e7dda7281c1fac695439;550 manifest-bound payloads. Contains fuller report, source/tests, all actor transactions/models, self-contained witness inputs, all evaluations, source receipts and repeat logs. Separate readable report: `rustrobotics-state-local-baseline-report.md`.

This fixed observable outward gate is not a sufficient simplification of the demonstrated full baseline repair. A practical method must be judged on the complete actor update and all behavioral panels, not merely removed squared value error. No gate, clipping rule or automatic head refresh is validated. Only this evidence report is added outside workflow trigger paths; production code/defaults/master/deployment remain unchanged.
