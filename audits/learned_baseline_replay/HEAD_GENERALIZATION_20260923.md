# Native head numerics pass; four-stream calibration does not generalize reliably

23 September 2026. Diagnostic only. **The Rust fitting kernel is qualified, but the fixed four-stream calibration rule produces severe held-out prediction regressions at two of four additional historical checkpoints. This is not a measured actor-policy regression or a closed-loop training result.**

## Native qualification

Run35859938686, source b84295f6e25b8d1505d739f47ea128e7ea286ee5, passed nine Rust unit tests, strict Clippy, formatting and both actual-system comparisons against independent augmented SVD. For the original four/24-trajectory systems the float32 heads are bit-identical to SVD, standardized correction discrepancies are1.0085e-11/5.4925e-11, and independent equation residuals7.4212e-15/1.1888e-13. Ordered float32 predictions agree exactly on supplied features. This qualifies the fitting kernel, not a native policy update.

Recovered artifact10750173878, SHA256 ae927dca39df7d76ceb773245e083c2fed83756a3714228048f6aff758dd9208; all40 payload hashes pass. The local runtime has no Rust compiler. Subsequent local native executions use this verified Linux executable, not a claimed local build.

## Four additional anchors fixed before outcomes

Protocol issue35/comment5795119279 preceded fitting. Reuse the BASELINE data from stopped online-pilot archives10749166623/10748303328: histories41004/41006 at checkpoints4097 and4128. These are all available complete independent calibration panels in those stopped archives. Their final4224 holdout panels were not written; no missing outcome is inferred.

Each fit uses its current outgoing critic's own hidden representation and head. Fit exactly four fresh current-policy streams[0,1,4,5],512 rows each: two ordinary and two outward TRAINING starts. This mixed subset was fixed explicitly; the first four streams in this corpus would all be nominal. It is a new test of the same fitting rule, not the earlier pilot's identical four trajectories.

Keep the original65-variable objective: ridge0.001 on population-standardized features, scale floor1e-8, unpenalized intercept, correction around the incoming output head, float64 solve, float32 folded coefficients. The4480 hidden parameters remain byte-identical. No sweep, outcome-dependent subset or checkpoint selection.

All eight separate holdout streams are scoring-only,4096 selected observations per anchor. Holdout returns use2048-step streams and zero far bootstrap; training streams have1536 steps and the original frozen far bootstrap. Gamma is float32(.995); true terminals stop return propagation across resets. Labels are sampled finite-horizon returns, not exact conditional values. Current actor and critic snapshots bind to the saved outgoing ordinary optimizer checkpoint. No calibrated baseline is fed back into actor learning here.

## Results

| History/checkpoint | Fitting RMSE before -> after | Independent holdout RMSE before -> after | Outward-start holdout RMSE before -> after |
|---|---:|---:|---:|
|41004/4097|19.3554 ->2.8504|26.1764 ->54.0285|36.6131 ->76.3892|
|41004/4128|8.3870 ->1.1688|8.5204 ->4.8986|11.9478 ->6.7786|
|41006/4097|8.6998 ->1.3896|4.5198 ->4.2508|5.7379 ->5.9254|
|41006/4128|1.7167 ->0.7148|9.5042 ->54.3127|13.3774 ->76.7963|

Every fitting loss improves. Two pooled holdout errors deteriorate severely. The modest pooled improvement at41006/4097 hides an outward-start regression. Each anchor has at least two individual holdout streams with increased error. Every per-stream, early/late, outward-early and signed-mean error is retained in the detailed artifact; no adverse group is dropped.

## Solver error ruled out for these outputs

Executed the qualified Rust binary on the EXACT four new feature/target systems, twice each. All four native float32 heads match the independently fitted/evaluated heads and SVD heads byte-for-byte. Maximum standardized correction discrepancy versus SVD2.9679e-10; maximum independent equation residual5.7903e-14. The original numerical gates pass unchanged. Ordered prediction checks on supplied features are exact. Both old CI fixtures were also replayed locally and all four files per fixture reproduce exactly.

Heldout full-network inference uses the checked float32 reference, not a claim that full native actor evaluation ran. Incoming reference-network predictions differ from archived native values by at most6.1036e-5 per anchor, below the fixed1e-4 gate. All recorded float32 return recurrences reproduce exactly, including holdout zero bootstrap. These numerical differences are tiny relative to the observed generalization failures.

## Post-outcome extrapolation diagnostics

At41004/4097 the corrected network predicts as high as435.466 while heldout returns peak at199.087. With the task's nonnegative quadratic reward penalties, reward is at most1, giving a derived infinite-horizon return ceiling approximately200.00019 at the actual gamma. Thus435 is not simply disagreement with a noisy return label. The reward source is baseline4739f370/rust_robotics_train/src/env.rs. No clipping rule was introduced.

At41006/4128 predictions reach-542.440; no universal lower-bound claim is made. The worst-error recovery row has a hidden feature roughly20298 training standard deviations beyond its fitting range; the analogous41004/4097 departure is about2826. Posthoc regularized-design leverage also becomes very large out of sample. This supports inadequate coverage/extrapolation as a failure mode, not a unique separation of coverage, target noise, representation and regularization effects. No new uncertainty threshold or fit was selected from these diagnostics.

## Stale-snapshot check

At4128, the entire frozen4097 calibrated snapshot, including its original encoder, gives holdout RMSE70.2223 for41004 and10.6067 for41006. Fresh4128 calibration gives4.8986 and54.3127 respectively. No old head was grafted onto a different encoder. Neither unconditional refresh nor indefinite reuse is validated; these are retained comparisons, not a rule choosing the best after evaluation.

## Verification and scope

31 local tests pass with warnings as errors, covering all actual anchors, SVD/native-head identity, native repeats, source bindings, disjoint random keys, full-tail float32 returns, malformed/nonfinite data, heldout bootstrap, immutable inputs, whole-snapshot stale inference, permutation/negative-stride input, intercept and independent scalar metric reconstruction.

Both original stopped archives pass outer checks and all3206 payload hashes,1603 each. A complete repeat reproduces164 result files byte-for-byte. The compact evidence package retains88 exact relevant source members and both original manifests, plus native qualification source/executable/evidence, scripts, results and logs. A separate retained-source reproduction validates those88 members and reconstructs all four model/prediction/metric results; this is not a claim of rehashing the omitted90MB complete archives.

There are four distinct calibration candidates,8192 fitting rows from24576 archived selected-stream transitions. All64 source streams' full tails total114688 archived steps per validation. Repeats/tests add numerical work. Ten local native solves comprise two old-fixture controls and four new systems twice. Zero new actor updates, simulator transitions or policy-performance episodes are executed by this offline diagnostic.

Original archive identities:
- 41004/artifact10749166623: f461c1dc45a5a1976587d7280fc7c66e8a352a15fc463d10136b3cab1b160258;
- 41006/artifact10748303328: 3270a09418f659e94f9bbcad1963441c96eb4cc40beb4d47c478e9bf67d529df.

## Separate online pilot

The repaired full-network online pilot35861559555/source cfe683840094a2eea13dbd020b53e6b588ed511d has passed preflight. At the recorded status check41006 was executing and41004 was queued. No completed treatment result is available. That experiment was not duplicated, retuned or cancelled, and its simulation/optimizer cost is separate from this offline test's zero-new-simulation accounting.

The earlier successful4140 baseline intervention remains valid. These additional anchors show why it must not be generalized into an automatic four-stream online correction yet. A fixed independently tested treatment of recovery-state coverage and out-of-sample calibration is still required; this report does not establish such a guard or alternative algorithm. No from-scratch reliability or sustained-balancing claim follows.

Only this evidence report is added. Production/defaults/master/deployment, the qualified kernel and active online experiment remain unchanged. This report path is outside the head-qualification workflow trigger paths. Detailed source, tests and evidence are retained in rustrobotics-head-generalization-results-20260923.zip.
