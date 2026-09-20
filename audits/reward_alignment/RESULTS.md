# Reward versus survival: usually aligned, with one concrete exception

September20,2026. Issue #35. Retained-data audit complete; selected-case native replay is still queued at this status check. No new trained controller, changed reward/discount, or robustness pass is claimed.

## Decision

**The current discounted reward prefers survival in all15 primary original-versus-symmetric comparisons where survival differs.** A secondary original-versus-reflected comparison contains one real exception: a300-second survivor receives less discounted return than a1.2-second rail failure. Neither a blanket reward explanation nor universal reward/safety alignment follows. Do not promote a gamma change on this evidence.

Analysis registration: #35 comment5749512184, before the accounting. Narrow replay addendum5749522580, after the aggregate exception was found and before additional trajectory outcomes. Original evidence: reflection run35486221478 atd11b9fbf99edd0d221c4dd179a421ca46466b6eb, all eight previously exposed actors41001–41008. All4608 outcomes and the72 predetermined replication0 traces are included. No controller was retrained or rerun during the completed offline audit.

## All discordant outcome pairs

| Mapping pair | Exactly one survives | Current reward prefers survivor | Current reward prefers failure |
|---|---:|---:|---:|
|Original vs symmetric — PRIMARY|15|15|0|
|Original vs reflected — secondary|23|22|1|
|Reflected vs symmetric — secondary|14|14|0|

The52 comparisons overlap across26 unique initializations; they are not52 independent trials. The15 primary cases span five histories and include all13 repaired failures AND the two lost successes. In those lost cases, the original survivor receives higher reward. Primary survivor-minus-failure discounted differences range+6.566167 to+25.649971.

This examines recorded finite-horizon return, not a critic prediction. It does not establish conditional expected action value, optimizer correctness, population reliability, or which single action caused a failure. Same-outcome pairs are retained but do not answer this particular survival-ranking question.

## The single secondary counterexample

Seed41003, long-stoch, episode45; identical original initial state/random key:

| Mapping | Ending | Duration | Gamma0.99 return | Undiscounted return |
|---|---|---:|---:|---:|
|Original|position failure|1.20s|27.464364|31.662562|
|Reflected|survives|300s|15.554230|29402.219491|
|Symmetric|survives|300s|38.657315|29681.381316|

The reflected survivor scores11.910134 below the failure, while the symmetric survivor scores11.192951 above it. This is a concrete realized disagreement between survival and the configured objective, not evidence the failing original has higher expected return over independent noise replications.

The original episode45 retains aggregates, not its full reward path. Its exact retrospective replay has been submitted as **run35508137816**, source **114fe450771f4fb9d2e3853d4a2c5757abf94891**, using all three unchanged mappings. Every historical field must match before the full paths are interpreted. At the last check job106071288661 remains queued with no executed steps. No new Rust compilation/test pass, replay outcome, time-local cost explanation, or alternative-gamma score for that missing path is asserted.

## Discount in physical time

Actual float32 dt is0.009999999776482582s and gamma0.9900000095367432. Reward weights after100/200/500/1000 ticks are0.366033/0.133980/0.00657051/0.0000431717. The half-life is0.690s and exponential time constant0.995s. This is not a hard horizon.

A−10 terminal reward at tick1000 contributes approximately−0.000436 to the initial score. It is NOT the entire effect of failure: termination also removes later rewards, and nonterminal state/control costs remain important.

The unchanged nonterminal objective is1−.2x²−.02v²−theta²−.05omega²−.001u_clipped²; true terminals replace it by−10. Timeout truncation is not a task failure. Discounted cost/benefit and survival-first qualification are different criteria, not a newly discovered force-sign or integration bug.

## All retained full-trace sensitivity checks

All72 replication0 trajectories are independently re-scored atgamma0.99/.995/.999/1. These are unchanged recorded rewards, not policies trained at different gamma values. Only two unique initializations in this subset have different survival outcomes, yielding four pair comparisons. All four prefer survival under every listed gamma. Episode45 is absent and is not reconstructed from its aggregate.

For41005/stochastic nominal0, original fails1.11s and symmetric survives300s. Survivor-minus-failure scores are+7.879341,+55.175380,+760.473650,+29588.278364 under the four listed gammas. Gamma changes return scale as well as horizon; these numbers are not comparable learning effect sizes and do not justify selecting a new hyperparameter.

Fixed-tick cumulative scores and per-trace discounted cost components are archived. Float32 subtraction rounding is accounted separately, not misassigned to physical costs.

## Verification and limits

All9 original ZIP digests/221 payload hashes, build/actor bindings,4608 outcome keys/denominators/initial pairs and original historical scores verify. All72 full traces/1429579 transitions reconstruct policy/action/dynamics/rewards/endings/centering. The reward implementation additionally reproduces BIT-FOR-BIT with sequentialfloat32 operations on every exported step. Current-gamma accumulation agrees within1e−10; inherited aggregate discrepancy is at most9.664e−13. Other replications have aggregate outcomes only.

**51 offline tests pass**:35 inherited reflection/trace controls plus16 new accounting/pairing tests. They cover terminal override, timeout, clipping, float32 operation order, missing/duplicate/mismatched pairs, retained lost successes, and discount indexing. Synthetic tests are not learning outcomes. The predecessor's native preflight is not a pass for the new queued endpoint.

Two missing loose CI archives were recovered from exact`raw/` paths in the attached predecessor evidence, not its separately failed-local folder. Hashes match both the enclosing manifest and published digests. No controller outcome was rerun or replaced.

## Disposition

The completed audit adds ZERO simulator/training interactions. The selected replay, when executed, is at most90000 additional diagnostic steps, separately counted; no fresh cohort or candidate training is involved. Its result is not yet available.

No production gamma, reward, global normalization, policy weights, PR38, master, or deployment changed. Most tested survival repairs already earn higher current reward, so learning/retaining better recovery cannot be replaced by a blanket reward-misalignment explanation. The single counterexample is retained rather than hidden; it does not alone establish the next learning correction.

A self-contained conversation package supplies all9 unchanged source artifacts, complete numerical tables, standalone verifier/tests, and offline replay. The full README/status distinguishes the complete retained-data audit from the queued selected-case replay.