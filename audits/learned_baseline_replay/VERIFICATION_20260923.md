# Learned-baseline behavioral verification and native replay repair

23 September 2026. Diagnostic only; no production defaults, deployed policies, or supported training recipe changed.

## Native replay unblocked

Old run **35771100199**, executable source `24b31e6ea8e24afd1f8bbdc59dff573bab85a155`, succeeded at reproducing both registered critics and their exact float32 prediction hashes. Its preflight failed solely at Clippy's 10-argument `run_candidate` helper. Measurement was skipped; the failed artifact is retained.

Repair **206d4f2d27c5a78c68560894df7ccb16c608b5bb** groups the repeated shared references into immutable borrowed `CandidateReplay`. No lint suppression. The candidate computation and every assertion after argument unpacking are byte-identical between the old/new prepared sources. The workflow is byte-identical; all other common prepared source files match. No candidate, prediction hash, optimizer rule, evaluation case or criterion changes.

New run **35841601153** passes strict Clippy, format/source restoration, **117 native unit/audit tests**, **7 balancing controls**, and **9 learning controls**. Twelve heavy library endpoints and two historical heavy integration endpoints are explicitly ignored in the ordinary passes; the current heavy replay endpoint is invoked separately. Its values job also passes, with both critic binaries, both prediction arrays and the source identity record byte-identical to the earlier successful values artifact.

At the recorded status check, measurement job **107118711045** was executing the actual replay after successful build/value/historical-input verification. No completed native prefix/sham result, native candidate scores, or 6,144 native evaluation records is claimed here. Preflight success is not policy qualification.

## Completed independent behavioral verification

Recovered the already-completed external study from issue comments **5785029668**, **5785034261**, **5785089283**, report commit `baba5a7abf5214184aebacee5961d27c863b610d`. Reran its original evaluator with the exact four archived actors and fixed cases. All **6,144 rows and the full report reproduce byte-for-byte**. This is a reproducibility rerun, not another independent statistical sample.

All four actor byte streams also match their actual source members: incoming/original in Adam artifact10671125548, and final learned-uniform/early-balanced numerical actors in the prior full-step bundle. They are numerical actors, not yet actors emitted by the pending native replay.

The external evaluator uses the fixed nonlinear RK4 task/reward but NumPy/PCG64 rather than Rust/StdRng. Each panel has512 paired cases, cap2048 steps, dt.01 (~20.48s). Deterministic means no policy sampling; environmental noise remains. This is not native numerical parity.

| Actor | Nominal deterministic | Nominal stochastic | Outward stochastic | Mean outward discounted return |
|---|---:|---:|---:|---:|
| Incoming |509/512|508/512|415/512|57.55691|
| Original update4140 |509/512|509/512|382/512|57.12924|
| Uniform learned baseline |509/512|508/512|415/512|57.75496|
| Early-balanced learned baseline |509/512|508/512|412/512|57.79334|

All observed failures are position-limit failures. Versus original, uniform gains33 outward completions and loses0; early-balanced gains30 and loses0. **Versus incoming, uniform gains2 and loses2**, so equal pooled completion is not case-by-case safety. Early-balanced loses3 and gains0 versus incoming. Under the same no-completion-regression criterion used for the native local screen, uniform passes on this external panel; early-balanced does not. Neither candidate is selected or altered for native execution.

Independent standard-library recomputation from all raw rows reproduces the NumPy summaries and paired contrasts. Outward discounted-return differences, with ordinary descriptive99% intervals using the existing mean +/-2.586*SE rule:

| Comparison | Mean | Interval |
|---|---:|---:|
| Original minus incoming |-.42766|[-.48525,-.37008]|
| Uniform minus incoming |+.19805|[+.17802,+.21809]|
| Early-balanced minus incoming |+.23643|[+.21585,+.25702]|
| Uniform minus original |+.62572|[+.55883,+.69260]|
| Early-balanced minus original |+.66410|[+.60454,+.72366]|

Both candidates have positive mean return differences on both nominal panels as well. Better mean return/earlier gradient cosine does not erase early-balanced's extra recovery failures. Intervals are not multiplicity-adjusted, training-seed confidence bounds, or a population noninferiority result. This remains one exposed training history and one external random domain, not continual critic tracking or sustained-balancing qualification.

## Verification and cost

**16 new local tests pass with warnings treated as errors.** They cover actual row/statistic reconstruction; duplicate/missing/invalid/nonfinite data; permutation and paired accounting; terminal precedence; and equal aggregate versus individual outcomes. Independent synthetic controls check the derivative against a separately solved coupled mass matrix, mechanical power balance, symmetry/equilibrium, strict boundaries, first-failure stopping, input immutability, and commanded-force versus disturbance-force reward semantics. These do not replace native qualification.

The reproducibility run repeats **11,803,802 active environment transitions** on the6144 fixed cases. Vectorized masked calculations are additional CPU work. No local critic/actor fitting. The prior expensive frozen-policy critic fit remains inherited cost; GitHub's values jobs reproduce that fixed pilot, and native measurement replays history and executes the prescribed candidate/control updates. No zero-optimizer-work claim is made for those jobs.

## Retained provenance

All five outer archive hashes and276 declared payload hashes verify:
- failed build10728816711,124payloads: `b605c37310249ecc722584c2cddeb1e4006af9598374a4d16b27d7564f37d506`;
- earlier values10728205396,7payloads: `bbc8f1194b118598caf8cad9524f874e20c990f47a83ce15e2a0406e94ba3c2d`;
- repaired build10741398129,130payloads: `4573aaf6939816ef9b8c20d440867915c805785b11482e1c655a5982d9a971ad`;
- repaired values10741895559,7payloads: `43121ad51e3586dfe8c8794adeefe2d22f3d6f5262504a62aa1fd559515f0a47`;
- external archive,8declaredpayloads: `358c56caccf4408f4e79b15306b0981540c13a5fe8a69f5f80d7f73263bf3259`.

The external manifest does not enumerate its two logs; the outer digest covers them. The two large actor-source archives are bound by outer hashes and exact actor-member bytes here, not counted as a fresh all-nested-payload audit.

Unchanged baseline prediction hashes: uniform `8d2cdd410a6247ae08e88eb228e21baf08349a932db8941a2062c892e7d8a937`; early-balanced `e62cfcc96f3458cc74ff7e17110f1c55ac759d63f132381a27c274c741ecf12d`. No new explanation is asserted for the prior cross-session arithmetic mismatch.

The conversation verification bundle retains the original external archive, four CI archives, original evaluator/actors/results, repeated outcomes, independent verifier/tests, logs and receipts. Native measurement remains the outstanding gate; no new learning sweep or production merge is authorized by this report.
