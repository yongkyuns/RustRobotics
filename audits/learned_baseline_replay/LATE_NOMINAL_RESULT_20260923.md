# Late nominal witness 4195: baseline-only correction restores nominal completion and improves recovery

23 September 2026. The fixed native causal diagnostic completed. On independent confirmation cases the original update loses nominal successes and discounted return; replacing only its actor-advantage baseline restores the incoming nominal success sets and improves recovery. This supports a baseline-error contribution at this selected step. It is not an online training recipe, an explanation of every update in the window, or a from-scratch qualification.

## Execution and fixed selection

Run **35904599021**, executable **fc994acfc455d0ea099b0d32b5af67f59bef0af2**, completed preflight and measurement. Protocol: issue35/comment5800483298. It reconstructs seed41004's original prefix through4096 and the unchanged extra-fresh full-critic continuation through4224, with original live Adam and all original extra-fitting data. No candidate, stop point, evaluation domain or selection rule changed during verification.

All97 checkpoints4128–4224 were evaluated on64 fixed cases per short panel,18624 localization records. The largest positive consecutive nominal-deterministic discounted-return drop selects **update4195**, drop **0.4878131317401255**. Independent confirmation does not reselect the witness. There are47 positive drops in the window; the selected step is not the entire explanation for its net nominal-det change of-1.0574148641. Later recovery in the same curve remains retained.

## Independent native confirmation

New domain0x74000000, three fixed actors,512 paired cases per panel, cap2048 steps at dt.01 (about20.48s):4608 confirmation records. Deterministic means deterministic policy actions, not absence of environmental noise. Evaluation discount remains float32(.99), promoted to f64; training and paired-continuation discount remain float32(.995).

| Actor | Nominal deterministic | Nominal stochastic | Outward recovery |
|---|---:|---:|---:|
| Incoming4194 |512/512|510/512|392/512|
| Original4195 |510/512|509/512|396/512|
| Baseline-only4195 |512/512|510/512|424/512|

The original loses2 deterministic and1 stochastic nominal successes, gains4 outward successes, and loses none outward. The corrected actor restores exactly the incoming nominal success sets. It gains32 outward cases and loses0 versus incoming; versus original it gains2/1/28 cases and loses0 across the three panels. Finite observed absence of lost cases is not a general safety guarantee.

Mean discounted-return contrasts and the unchanged ordinary descriptive99% paired intervals, mean +/-2.586*SE:

| Panel | Original minus incoming | Baseline-only minus incoming | Baseline-only minus original |
|---|---|---|---|
| Nominal deterministic |-0.543665 [-0.693278,-0.394052]|+0.007848 [-0.033964,+0.049661]|+0.551513 [+0.386768,+0.716258]|
| Nominal stochastic |-0.460922 [-0.607266,-0.314577]|-0.030608 [-0.079220,+0.018005]|+0.430314 [+0.287036,+0.573592]|
| Outward recovery |-0.027952 [-0.045215,-0.010690]|+0.301330 [+0.254776,+0.347884]|+0.329282 [+0.289643,+0.368921]|

The original nominal return regression independently reproduces. The correction clearly improves all three contrasts versus the original. Nominal return versus incoming is approximately restored, NOT demonstrated equivalent or uniformly improved: both nominal intervals include zero, and the stochastic point estimate remains negative. These pointwise intervals are not multiplicity-adjusted, training-seed confidence bounds or formal noninferiority tests.

Undiscounted mean episode-total changes versus incoming are+1.272471 deterministic,+1.098852 stochastic,+105.492416 outward for the correction. Their respective intervals are[+0.997872,+1.547070],[+0.818582,+1.379123],[+73.770861,+137.213970]. All outcomes remain retained; completion and discounted return are not conflated.

## What changed

Eight paired1024-step continuations at each of the1024 original fitting states estimate conditional Q and V under the SAME incoming policy. There are8192 recorded pairs. The only candidate change is:

    advantage = normalize_float32(original_return - float32(mean_sampled_V))

Original observations, latent actions, old likelihoods, critic targets, minibatch order, learning rate, clipping and incoming actor/critic Adam state are retained. The learned critic is not replaced or refitted by this intervention; its ordinary optimizer transactions are identical. Captured original normalized advantages, not an inverse reconstruction from rounded returns, remain authoritative for the sham.

This is a diagnostic simulator baseline using actual pre-action states, not an observation-only estimator that has been shown deployable. Far-cutoff bootstrap remains: survivor coefficient0.00589979355 and RMS mean V bootstrap contribution1.13794. The RMS paired action-credit bootstrap contribution is0.00003067, but this small paired difference does not establish that the absolute baseline is unbiased. Sampled credit and values are not exact conditional expectations.

## Independent action-credit analysis

On this selected batch:

| Component | RMS |
|---|---:|
| Paired conditional action credit Qmean-Vmean |0.0314883|
| Original return minus Qmean |0.158190|
| Vmean minus incoming recorded critic |15.084466|

Baseline error is approximately479 times action-credit RMS. Its contribution to the initial actor gradient on the common original normalization scale has norm1.99907, versus0.03351 for action credit (about60 times larger). **98.65% of baseline-error variance lies between contiguous episode/stream groups.** One57-row recovery segment has mean Vmean-minus-critic error-63.2862: the critic substantially overestimates those returns. These groups and signed errors, including contrary groups, are retained.

The exact decomposition retains a separate return-rounding residual, maximum7.6294e-6, rather than pretending float32(R-V) reconstructs original raw GAE exactly.

Initial actor-gradient alignment with normalized paired action credit is **+0.06551 originally and+0.95267 after correction**. Unlike the earlier4140 witness, the original direction is weakly aligned, not reversed. Using disjoint four-draw halves for baseline correction and action-credit reference gives corrected cosines+0.91359/+0.96418, versus original+0.07423/+0.05704. These are independent continuation-draw checks on the SAME exposed states, not new trained histories. RMS draw standard errors are0.08870 for baseline estimates and0.01250 for paired credit.

For the COMPLETE native actor displacement, initial-credit-gradient dot displacement improves from+0.006317 to+0.077584. Mean policy KL on original fitting observations is0.004772 original versus0.008299 corrected; weight-displacement norms0.026788 versus0.023148. Thus the result is not merely an almost-zero policy step. This is a local derivative, not measured return; the behavioral confirmation above is the separate evidence. The globally scaled nominal-start supplemental contribution to that local projection changes from-0.001245 to+0.014398, while outward-start contribution changes+0.007481 to+0.062898. Individual groups are not all positive after correction.

## Exact verification

The original result archive's outer SHA and all9071 payload hashes verify. The build archive's outer SHA and137 payloads verify. The prior online archive's outer SHA and1579 consumed members are independently checked, plus six prefix snapshots inside its hash-bound historical archive. This is not a fresh full audit of every member in all old nested archives.

All128 continuation update records,1024 fresh-stream files,256 fitting-data/extra-optimizer-log files,288 detailed ordinary files,14 available actor/critic checkpoint comparisons and384 localization endpoint records match the original online evidence. No original continuation was changed to obtain the witness.

The previously prepared transaction checker now runs end-to-end on the ACTUAL late result. All16 actor and16 critic transactions in each of sham and candidate reproduce bit-for-bit from exported native gradients, moments, counters and weights:64 transactions. Actor counters start67104; critic counters start71808, correctly including previous extra critic work. All48 sham optimizer files equal the original. Candidate critic gradients, moments, counters, step weights and critic losses are unchanged. Final evaluated actor bytes match the16th optimizer checkpoint.

Independent PyTorch float32 backpropagation checks all64 recorded gradients; maximum relative L2 discrepancy3.8767e-6, below the unchanged1e-4 check. Manual NumPy float64 derivatives agree within that gate on63/64 transactions. The remaining candidate actor transaction14 differs by0.0012539 relative because a hidden preactivation at fitting row836/unit12 is+1.6722e-8 in float64 but exactly0 in float32. The float32 derivative matches native to1.5666e-6 relative. This is a precision-sensitive ReLU kink in the verifier comparison, not an identified native training bug. The failed float64 check is preserved rather than silently loosening its threshold.

All8192 pair identities, exact random keys, state/action bindings, continuation discount coefficients, bootstrapped estimates and the registered mean-baseline bytes verify. All18624 localization and4608 confirmation cases, summary fields, paired contrasts, caps and boundary classifications recompute exactly. Confirmation contains332 position failures and1 angle failure, not exclusively position failures. All21 retained traces (12 credit and9 confirmation) reproduce total/discounted returns, first-ending semantics and state/observation continuity; confirmation maxima, applied-force RMS and centering also match. Full trajectories for every episode were not exported, and no independent simulator rerun is claimed.

Twenty-two new regression/mutation tests pass with warnings treated as errors, using actual evidence. A second complete offline verification reproduces the computed report byte-for-byte. The original failed float64-gradient comparison and an initial RMS-check failure caused by summation order are retained; the latter was corrected to the native sequential multiply-and-add order. No candidate, evaluation result or acceptance threshold changed. Local analysis versions are Python3.13.5/NumPy2.3.5/Torch2.10.0+cpu, with one numerical thread; these are not the native build versions.

## Scope, cost and next boundary

This demonstrates that the selected late nominal loss is not an unavoidable price of recovery improvement: a baseline-only change restores incoming nominal completion while improving recovery more than the original step. It does not show that all other47 dropping steps have the same cause, that the entire128-update treatment is repaired, or that a cheap critic can supply this baseline online. Previous cheap-head extrapolation failures and the original online treatments' failed no-regression screens remain valid.

Native work includes36,409,873 localization transitions,15,868,968 paired-credit transitions,8,972,001 confirmation transitions, plus the original prefix and128-update continuation. Prefix collection totals52,441,749 transitions; ordinary continuation1,703,936; extra fresh collection1,572,864, with6144 extra critic minibatches. The verification performs no new training or policy simulation; numerical forward/backward calculations and hashing are real work. The costly simulator baseline is diagnostic, not a practical training-speed result.

Only this evidence report is added to the repository, outside the experiment's workflow trigger paths. Executable, production defaults, master and deployment are unchanged. No new workflow or from-scratch cohort was launched.

Raw result artifact10774320351,325783421 bytes, SHA256 **6358bbf59ab09d1021b5599415e78ae73ed9879cac97015bea010204ac43e5b8**. Build artifact10773071363 SHA256 **61ab382524972ff7aae5c07fd02b4c52e2f0299e269e886b8a9cce13b7056881**. Original online artifact10752868007 SHA256 **b4e3364d19ea79f334e4a6e9734e562cbd02cbba09e3e0deec15436dfdc02687**. Verifier source, tests, computed metrics, original manifests, selected raw evidence and failure/repeat logs are retained in the conversation package. Full historical/all-member verification requires the original complete archives, not the compact subset.
