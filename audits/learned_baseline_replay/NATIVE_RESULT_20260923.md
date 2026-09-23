# Native learned-baseline full-step result

23 September 2026. **Both fixed candidates pass the registered local native screen. This is one exposed update, not a practical online training fix or sustained-balancing qualification.**

Run **35844254183**, executable **09e559af70556df2d547c80c2a3a921c61c089de**, completed successfully. It replayed seed41006 through original update4140, executed the original/sham and both prescribed learned-baseline interventions with incoming live optimizer history, and produced all **6,144 native evaluation records**, domain **0x61000000**. No candidate, checkpoint, prediction hash, evaluation case or acceptance rule was changed during verification.

## Native results

Every panel contains512 paired cases capped at2048 steps, dt=.01 (~20.48 seconds). Deterministic means no policy sampling; environmental noise remains. Evaluation discount is the existing f64::from(0.99_f32). The historical development training recipe remains gamma.995/support1024, not a changed production default.

| Actor | Nominal deterministic | Nominal stochastic | Outward stochastic | Mean outward discounted return |
|---|---:|---:|---:|---:|
| Incoming |508/512|503/512|412/512|58.773693|
| Original update4140 |508/512|505/512|381/512|58.311740|
| Uniform learned baseline |508/512|505/512|418/512|59.005065|
| Early-balanced learned baseline |508/512|503/512|416/512|59.036455|

The original update loses31 outward completions. Uniform gains37 versus original and6 versus incoming; early-balanced gains35 versus original and4 versus incoming. Neither candidate loses an incoming-success case on these native panels. Uniform also gains2 nominal stochastic successes. Early-balanced has the incoming nominal success sets, so it loses2 nominal stochastic successes relative to the historical original. All failures are position-limit failures.

| Outward contrast | Mean return difference | Ordinary descriptive99% interval |
|---|---:|---:|
| Original minus incoming |-0.461953|[-0.554465,-0.369441]|
| Uniform minus incoming |+0.231372|[+0.153954,+0.308790]|
| Early-balanced minus incoming |+0.262762|[+0.187294,+0.338229]|
| Uniform minus original |+0.693325|[+0.618570,+0.768080]|
| Early-balanced minus original |+0.724715|[+0.660196,+0.789233]|

Both candidates also have positive candidate-minus-incoming return intervals on both nominal panels. Uniform nominal-det delta+.117353 [.078370,.156337], nominal-stoch+.192639 [.085094,.300184]; early-balanced+.099055 [.070666,.127444] and+.118481 [.082224,.154738].

The original local screen requires no completion panel below incoming and no confidently harmful return contrast. Both pass. Intervals retain the prescribed mean +/-2.586*SE rule. They are not multiplicity-adjusted, independent-training-seed bounds, or formal noninferiority evidence. Finite observed absence of lost cases is not a safety guarantee.

## Exact controls and independent verification

All703 declared payload hashes verify:255 result,130 build,7 values,311 embedded historical. Outer digests are checked before members. All72 original-transaction files match historical reference/update-4140 byte-for-byte, including fitting/support data, raw/normalized advantages, optimizer records and step weights. Twelve actor/critic prefix checkpoint files match. The reference wrapper's four additional metadata files are separately bound: incoming actor/critic, gamma and support length.

All48 sham optimizer files match the original exactly. All32 candidate critic-step weight files match their original counterparts, as do minibatch indices and critic losses. Every epoch is a1024-row permutation. Candidate observations, latent actions, old likelihoods and original critic targets are unchanged. Candidate raw and normalized advantages exactly match independent sequential-float32 construction using the original registered predictions. Each evaluated candidate is its final16th native checkpoint.

The learned critic is NOT installed in the trainer: only its frozen baseline predictions form actor advantages. Actual candidate definition remains normalize(original_return - fixed_learned_baseline). The completed native endpoint checks incoming live-optimizer restoration, shuffle-stream equality and session immutability. This archive exports step weights/records, not a fresh export of candidate moment tensors; no nonexistent moment export is claimed.

Independent recomputation reproduces published summaries, contrasts and local screens exactly. All6144 identities, paired keys, lengths, boundary classifications and finite fields verify. All12 retained replication0 traces reproduce total and discounted return exactly. Full trajectories for every episode were not exported and are not claimed independently reconstructed.

The previous independently computed numerical actor trajectories closely reproduce the new native optimizer trajectories. Maximum absolute parameter discrepancy across all16 checkpoints: original2.9802322387695312e-8, uniform5.960464477539063e-8, early-balanced2.9802322387695312e-8. This is not a claim of bit-identical cross-framework candidate weights or moments.

## Retain contrary evidence and limitations

The prior independent NumPy/PCG64 panel gave outward incoming415, original382, uniform415, early-balanced412. Uniform gained2/lost2 cases versus incoming there, and early-balanced lost3. Those results remain retained; the favorable native panel does not erase them. Distinct simulator/RNG implementations and random domains are not independent training histories and should not be pooled as training-seed reliability evidence.

Together these findings support critic-baseline miscalibration as a causal contributor to this particular destructive update, rather than only a correlation between critic RMSE and actor quality. Sparse coverage versus inadequate critic tracking/optimization and target quality remains unresolved upstream.

The prior native attempt35841601153 stopped before candidates at an invalid exact inverse assertion: R=float32(A+V) does not imply float32(R-V)==A. Actual forward identity holds on all1024 rows; inverse differs on1011, maximum7.62939453125e-6, normalized difference4.5299530029296875e-6. Repair09e559af binds actual original files to history instead. Captured advantages remain authoritative for the sham; learned-candidate arithmetic is unchanged. No tolerance replaced an identity check.

## Tests and cost

Current native preflight passes119 unit/audit tests,7 balancing and9 learning controls. Twelve heavy library endpoints and two historical heavy integration endpoints remain explicitly ignored in ordinary passes; the current heavy replay was separately executed and passed. No new full production/browser/platform qualification is claimed.

**29 new local verifier tests pass with warnings treated as errors.** They exercise actual native evidence, corruption rejection, missing/duplicate/nonfinite/unpaired cases, false timeouts, boundaries, row permutation, paired gains/losses, normalization and the exact rounding counterexample. A second full offline verification reproduces report.json byte-for-byte. The verifier executes no optimization or simulation. An initial verifier probe rejected four reference-only wrapper metadata files; the final version checks those explicitly rather than ignoring them. That preliminary log is retained.

Native evaluation executed11,799,016 transitions. The native job also replayed the original training prefix and executed original/sham/candidate updates; complete.json training_updates:0 means no candidate continuation, NOT zero optimizer work. The values job repeated the fixed pilot. That frozen-policy pilot used4096 critic transactions per arm,256 times the original per-update budget. This cost is not hidden or proposed as mandatory pretraining.

The next mechanism question is bounded-cost ONLINE critic tracking on representative ongoing-policy data as the actor changes, followed by a fixed from-scratch comparison. This result does not authorize another reward/gamma/lambda sweep, a production merge, or a claim that sustained balancing is solved. No new learning cohort was run in this verification.

## Provenance and reproduction

The conversation evidence bundle retains the three original native archives, detailed report, computed report.json, offline verify_native.py, test_verify_native.py, logs and repeat receipt. The larger prior numerical replay ZIP is hash-bound but not duplicated. Artifact retention is finite; retrieve actual retained archives rather than assuming prior sandbox paths.

- Native result10743785980 SHA256: 47e03b279ff896c34242cbc079ff3e26a60cb9c4a3d5e153b6b51f288b2c2b68.
- Build10742538449: 9590eaa9acacf859fc97eb96548fd7a1ead277a0655aaeef0433e83d4aa8ab64.
- Values10742667402: bc60568d1674300ea4d50825fcefc03b72b1b50c56e337fcb9f64e629081de52.
- Embedded historical: 65e071b04fa23f550a8be4cc306c05ccfe7d76eec0dfc6e7e1bab21cb3698ece.
- Prior numerical bundle: 6b3ea5671725a139e78f52988f86373de762d7b424e456213e8122a564dfe223.

Only this evidence report is added to the repository in this continuation. Native source, workflows, production/defaults, master and deployment remain unchanged. The report-only path does not retrigger the completed native experiment.
