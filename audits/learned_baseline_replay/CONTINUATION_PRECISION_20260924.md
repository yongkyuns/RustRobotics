# Continuation precision: baseline sampling changes the update; paired credit is not automatically safe

24 September 2026. **Completed numerical full-step and independent-simulator diagnostic on the two exposed witnesses. At update4195, disjoint four-draw baseline estimates produce436 versus395 recovery completions. Direct paired action credit reaches496, but loses nominal deterministic successes and makes a much larger policy step. No candidate passes the fixed no-regression screen in both histories.**

Protocol: issue35/comment5811217525. A numerical-environment clarification5811275425 was posted after the first original control stopped, before any new half/paired candidate or evaluation. Its exact scope is documented below; the initial failed identity comparison is retained, not claimed passed. This is not native policy qualification, a deployable critic, a new training history or an online fix.

## Fixed estimators and controls

Reuse the exact state-local source bundle SHA256295a7a028d2b7dee4c3f8ad456030ab0d7770a9bcb96e7dda7281c1fac695439. Witnesses remain41006/update4140 and41004/update4195. Original1024 observations, latent actions, old likelihoods,16 minibatch indices/order, incoming actor Adam moments/counters, learning rate and PPO clipping remain fixed. No fitting action is resampled. Critic targets are unchanged and no critic is fitted or installed.

Five full16-transaction numerical updates, plus the incoming actor:

- original: captured original raw/normalized GAE;
- full8: original return minus float32(mean sampled V over all8 draws);
- halfA: original return minus float32(mean V over draws0–3);
- halfB: original return minus float32(mean V over draws4–7);
- paired8: float32(mean(Q-V) over all8 paired draws).

Means accumulate sequentially in float64 before rounding. R-minus-baseline subtraction is float32; all arms use the original sequential whole-batch float32 normalization. The paired-credit method already existed in the investigation; it is not claimed as a new algorithm. This comparison adds the direct-paired update at4195 and a fixed, disjoint-baseline precision comparison on both witnesses. No favorable draw subset or checkpoint is selected.

The existing Q/V estimates use actual simulator pre-action states and a far-cutoff critic bootstrap. They are not exact conditional expectations or validated observation-only estimates. Disjoint draw halves measure sampling sensitivity; they do not expose common bootstrap bias. Only two half estimates are evaluated, not a population distribution of all possible baseline estimates.

## New external results

Unchanged NumPy/PCG64 evaluator. New roots2026092431 for4140 and2026092432 for4195. Each panel contains512 paired cases capped at2048 ticks,dt.01,approximately20.48seconds. Deterministic means no policy sampling; environmental noise remains. All18432 episode records are retained, with total and discounted reward. These are numerical actors and external random domains, not Rust/StdRng parity or independently trained agents.

| Witness | Actor/update | Nominal deterministic | Nominal stochastic | Outward recovery |
|---|---|---:|---:|---:|
|41006/4140|Incoming|511|510|408|
||Original|511|510|379|
||Full8|510|510|440|
||HalfA|511|510|435|
||HalfB|510|510|440|
||Paired8|510|510|447|
|41004/4195|Incoming|512|508|394|
||Original|509|507|396|
||Full8|510|508|429|
||HalfA|511|509|436|
||HalfB|510|508|395|
||Paired8|509|509|496|

At4195, halfA gains42 recovery cases/loss0 versus incoming; halfB gains1/loss0. HalfA minus halfB is41 extra completions, all gained/no lost, and discounted-return difference+.414201, descriptive99% interval[+.360136,+.468267]. HalfB still has an excellent initial credit-gradient cosine but delivers little recovery benefit. No rule choosing halfA from these outcomes is adopted.

At4140, both halves improve recovery: halfA gains28/loses1, halfB gains34/loses2 versus incoming. The sensitivity is not equally severe on both witnesses. Full8 gains34/loses2 there and35/loses0 at4195. The previous native/external results remain valid for their original domains.

Paired8 gains41/loses2 recovery cases at4140, and105/loses3 at4195, versus incoming. At4195 it gains70/loses3 recovery cases versus full8, net67. Yet it loses3 incoming nominal deterministic successes at4195 and1 at4140. Its higher recovery count is not casewise safety.

Outward discounted-return changes versus incoming, original descriptive mean +/-2.586*SE intervals:

|Witness|Full8|HalfA|HalfB|Paired8|
|---|---|---|---|---|
|4140|+.488689[+.273604,+.703774]|+.482620[+.369439,+.595801]|+.530102[+.315117,+.745086]|+1.149352[+.887731,+1.410973]|
|4195|+.309698[+.263188,+.356207]|+.408907[+.350019,+.467795]|-.005294[-.014015,+.003426]|+1.216751[+.948918,+1.484585]|

The fixed screen remains no incoming-relative aggregate completion loss in any panel and no discounted-return interval entirely below zero. Only halfA passes on4140; it fails4195 due to one nominal deterministic loss. Every other corrected arm fails nominal deterministic completion in at least one panel of each witness. HalfB also has confidently negative nominal-return contrasts at4195. **No corrected estimator passes both histories.** These pointwise, unadjusted intervals are conditional paired-episode descriptions, not training-seed confidence bounds, equivalence tests or general safety guarantees.

## Sampling uncertainty remains larger than the action signal

RMS across original fitting rows:

| Quantity |4140|4195|
|---|---:|---:|
|Paired mean action credit|.017501|.031488|
|Original return minus mean Q|.173179|.158190|
|Estimated standard error of mean8 V|.047443|.088695|
|Estimated standard error of mean8(Q-V)|.005427|.012505|
|Disjoint half-baseline disagreement|.088282|.171334|

Standard errors use sample variance across the8 draws divided by8, then aggregate their RMS over states. They describe this finite continuation experiment, not calibrated confidence bounds for true values. V-estimate uncertainty is roughly2.7–2.8 times the action-credit RMS. The halves differ only by which continuation draws form V; their effects on the complete actor update can therefore differ materially even with the original fitting actions fixed.

Within-row Q/V fluctuations are highly correlated under paired future randomness: pooled correlations .99344 and .99001. Subtracting paired returns cancels much of their shared randomness. The resulting lower standard errors do not prove exact credit or an unbiased baseline. Absolute mean V bootstrap contributions have RMS1.15854/1.13794; the much smaller paired bootstrap differences do not remove the possibility of common bias.

The realized-return residual also need not oppose the correct direction. On a common full8 normalization scale, its gradient has cosine+.88808 with the paired-credit component at4140 but-.51851 at4195. Its norm versus the credit component is3.03198 versus1.75734 at4140 and.96769 versus3.09466 at4195. Removing residual return noise is not automatically removing a harmful gradient in every frozen sample. Rounding residuals are explicitly retained in the additive decomposition; maximum gradient reconstruction residual is1.06e-15.

## Initial agreement is not a full-update guarantee

At4195, halfA has initial cosine+.91359 against the OTHER half's paired-credit reference; halfB has+.96417. Nevertheless, halfA produces436 recovery completions and halfB395. Their full-displacement projections against the all8 reference are+.08159 and+.03437. These are local derivatives, not expected returns. Paired8's initial cosine against the same paired8 reference is1 by construction and is not independent validation.

|Witness|Original fitting-state KL|Full8|HalfA|HalfB|Paired8|
|---|---:|---:|---:|---:|---:|
|4140|.005863|.010616|.008488|.011346|.045475|
|4195|.004772|.008298|.009054|.003787|.073690|

The direct-paired step has about4.3x and8.9x the full8 KL. Raw advantage SD falls from.164842 to.017499 at4140, and from.158163 to.031488 at4195, before each estimate is normalized separately. The estimator, normalized gradient scale, clipping trajectory and response of the continuing optimizer are coupled. The experiment did NOT match final policy movement or vary normalization rules. Therefore the recovery gain cannot be attributed solely to more precise credit, and the nominal losses cannot yet be attributed solely to a larger step. A fixed comparison controlling policy movement is a distinct next causal question, not a validated change to defaults.

## Numerical environment and retained failures

The first original-control replay stopped at cross-session byte identity. The UNCHANGED archived replay function produced exactly the same new weights, while its previous-session numerical actor differed by2.98e-8. Raw/normalized advantages, incoming parameters/moments/counters and minibatches were identical. ATen capability probes did not recover prior bytes; MKL arithmetic-path probes changed last bits but did not recover them. Precise cause remains unresolved. This is not evidence for a PPO arithmetic failure.

Clarification5811275425 explicitly replaced the cross-session numerical-byte prerequisite BEFORE any new half/paired candidate or evaluation with the pre-existing2e-6 native/reference absolute parameter gate, exact current-runtime agreement with the unchanged function, and exact current repeats. No numerical tolerance was enlarged. All four original/full controls pass: maximum previous-numerical-actor discrepancy5.96e-8, with exact current-function identity. The original failed prerequisite is retained, not represented as passed. Recorded native gradients through independent float32 Adam still reproduce all64 original actor/critic transactions exactly across the two witnesses.

159/160 new float64 manual gradient comparisons pass1e-4; the retained exception is4195/halfA/transaction13. A first-layer activation at fitting row548/unit11 is0 in float32 but+1.65e-9 in float64, changing its ReLU derivative. The independent manual float32 derivative passes at6.82e-7 relative. Independent manual float32 backpropagation was then checked on ALL160 transactions: maximum relative discrepancy2.17e-6, below the unchanged1e-4 gate. The float64 failure is retained and diagnosed rather than its threshold relaxed. This is a precision-sensitive verifier comparison, not an identified native training bug.

## Verification, repeatability and cost

19 new actual-data tests pass with warnings as errors. They check source/member identities, original controls, exact advantage formulas, disjoint halves, all160 Adam recurrences and minibatch histories, all160 independently differentiated gradients, evaluated-final-actor identity, malformed/missing/duplicate/nonfinite outcomes, paired intervals/accounting, random-domain records, and both-history screen interpretation.

All550 prior package payload hashes verified. A compact support set retains348 manifest-bound files, including334 consumed original witness members and unchanged inherited replay/evaluator/checker code, the prior numerical controls and origin metadata. This is not a new audit of every historical archive. The main comparison executes160 actor transactions; unchanged-function controls, recorded-gradient controls, tests and a complete replay repeat are additional work. The repeat reproduces301 output files byte-for-byte. Both outward panels repeat all6144 episode records and four panel files exactly. The four nominal panels were NOT rerun in full. Repetition is reproducibility, not another statistical sample.

New evaluation executes36,006,818 active simulator transitions; the outward repeats add10,946,562. No new Q/V baseline estimation, critic fitting, native CI job, continuing learner or from-scratch cohort was run. Existing simulator estimates retain their inherited collection cost. Source, tests, all candidate optimizer records/actors, outcomes, repeats and retained failure evidence are included in the conversation evidence package.

Only this report is added to the repository, outside workflow triggers. Production/defaults/master/deployment remain unchanged. The result shows sensitivity to continuation sampling and a remaining return-noise/update-size confound; it does not establish a practical safe actor update or invalidate the earlier native baseline-causality findings.
