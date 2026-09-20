# Outward-mixture trade-off: coupled learning signals and lost transient control

September 19, 2026 (America/Toronto). Issues #35/#33. This is attribution of the already completed outward-mixture experiment, not another training sweep or a promoted correction.

## Decision

**There is measurable cross-coupling between the retained nominal samples and the new outward-start samples. The frozen seed41008 policy also loses a specific acceleration-to-braking maneuver, producing a large cart/pole overshoot.** These findings are related diagnostic leads, not proof that advantage normalization alone caused the entire regression.

No new candidate was trained, no normalization/weighting change was installed, and no production default, PR38, master or deployed asset was changed. The existing outward-mixture development/absolute screens remain failed. A successful replay or analyzer is not controller acceptance.

Analysis plan before new calculations: [issue35 comment5746308632](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5746308632). Additional exact historical trace recovery declared before execution: [comment5746332262](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5746332262).

## Sources and scope

The source experiment is run35461070693, executed3d9e29a1f19157a926f0864ed244f9fcbcb7edec, report494a6ab8. Its eight exposed histories41001–41008 compare reset-only versus half-outward supplemental starts after4096 historical recovery-union updates. The original continuation has512 updates; detailed optimizer records exist only for updates1/128/512.

Use all eight histories, both arms, and all three archived detailed updates:48 batches, each1024 fitted rows. Group rows by fixed provenance:512 main,256 ordinary-reset supplemental,256 final-four-stream supplemental. The final group has outward starts in the candidate, ordinary starts in control. A group label does not assert that every later observation remains outward-moving; failures reset ordinarily.

All768 recorded post-minibatch mean vectors are checked. Incoming weights are available for736 parameter displacements: all16 at update1, but only steps2–16 at updates128/512. The first incoming weights at those later updates were NOT exported. We exclude those32 displacements from gradient attribution rather than substituting an outgoing checkpoint. Full-batch initial objectives remain calculable from the actual stored collection means.

The NumPy differentiation implements the same clipped surrogate and frozen squashed-Gaussian likelihood from the pinned production source. It does not execute Adam or manufacture optimizer state. The actual recorded parameter displacements remain the observations to be explained.

## 1. Changing outward samples changes the unchanged nominal training signal

At update1, both arms have identical incoming actor/critic bytes. The first768 observations, actions, old likelihoods, returns and RAW advantages match exactly. Only the final256 supplemental rows differ.

Nevertheless, normalizing once across the changed1024-row union changes the signs of **2,041 of those6,144 otherwise unchanged nominal advantages** across the eight histories. This is33.22% of the retained nominal rows in this particular first-update comparison, not a training-population failure rate.

| History | Identical nominal rows whose normalized sign changes /768 |
|---|---:|
|41001|548|
|41002|7|
|41003|145|
|41004|573|
|41005|123|
|41006|407|
|41007|181|
|41008|57|

The direction of the nominal-only actor gradient also changes, not just its scale. The cosine between control and candidate nominal gradients is **-0.468712 for41004** and **-0.485257 for41006**. Negative cosine means the two nominal gradient vectors point into opposing half-spaces even though their underlying nominal samples, raw targets and incoming network are unchanged. For the main512 rows alone,41001 and41006 are nearly reversed: cosines-0.997243 and-0.998881. All other histories and group results are retained.

At the incoming policy, no sample is beyond the clipping boundary. For a retained group N, the actor-gradient expression is schematically:

    g_N = (g_raw,N - global_mean_advantage * g_sample_score,N) / global_std_advantage

The finite sample-score gradient need not be zero. Changing the other rows changes the global mean and therefore can rotate the nominal gradient; positive standard-deviation scaling alone cannot rotate it. Diagnostic group-centering calculations and numerical derivative controls are retained but do not enter learning.

**This is not a newly discovered arithmetic bug, nor proof that every sign change is incorrect action credit.** Raw advantages themselves are estimates. A constant baseline has different properties in an expectation than in one correlated finite batch. Crucially,41006 later has one of the largest observed recovery gains despite its reversed first-update nominal gradient. This rules out treating the reversal count as a simple causal harm score.

## 2. Twenty-five percent of rows can dominate update weighting

Across the24 captured candidate batches, the final256 outward-origin rows account for a median **81.42% of the sum of squared globally normalized advantages**, versus42.07% for the corresponding final-quarter rows in the control. Candidate shares range14.10–98.12%; domination is not universal.

This is an advantage-magnitude diagnostic, NOT a percentage of Adam's update. Directly differentiated minibatches provide a related observation: the median norm of the final-quarter gradient divided by the combined first768-row gradient is **2.142** in the candidate, versus **0.656** in control. These are correlated steps from selected archived updates, not independent statistical samples.

At seed41008/update512, all512 main-stream raw advantages are positive, but global mean subtraction makes all512 negative. The outward-origin quarter contributes83.48% of squared normalized advantage magnitude. A nominal sample count alone does not preserve a nominal learning signal or its relative strength.

Group gradients oppose each other in172/368 known candidate steps, versus149/368 control steps. The actual parameter displacement has a negative local nominal-objective projection in133/368 candidate steps and122/368 controls. Conflicts also exist without outward mixing; these counts do not uniquely identify the cause of a controller failure.

Whole-batch accounting gives concrete trade-offs. In41005's first candidate update, the final-quarter surrogate contribution improves by+0.002868 while the retained nominal contribution changes by-0.001170, leaving a positive total+0.001697. In41008/update512, the corresponding changes are+0.003815 and-0.000072. These are objective contributions on the recorded data, not independent expected-return effects.

There are also outright full-batch objective regressions:5/24 captured candidate updates and1/24 control updates end below their incoming surrogate. We retain these; not every update successfully improves even the full empirical objective. Neither an aggregate gain nor a lower training loss can substitute for the nominal and recovery tests.

## 3. The exact failure trajectories identify a lost recovery maneuver

The archived evaluation contains twelve nominal angle-failure records, all for seed41008: two at checkpoint128 and ten at512. At32, the candidate has no nominal angle failures. Thus the final recorded update512 cannot explain when the first failures appeared; the change begins somewhere in the unrecorded interval32–128.

New native replay **20d7c39034b03707ac41dec57f6dc68d0dd6dd97**, [run35478304373](https://github.com/yongkyuns/RustRobotics/actions/runs/35478304373), uses the unchanged shared nonlinear PendulumEnv and exported policy inference. It runs the incoming, matched reset-only, and candidate frozen policies on ALL twelve existing failure cases, with their original reset, action-innovation and disturbance keys. This is retrospectively selected historical diagnosis, not new qualification or independent outcomes.

**All36 episode results reproduce the original fields exactly. Both incoming and matched reset-only policies survive every case; the candidate fails all12 at the original1.72–1.92-second times.** Full trajectories are now retained instead of only aggregate failure records. No controller is supplied, modified or switched during the replay.

On each candidate trajectory, an exploratory common snapshot at0.25seconds shows7.14–14.19N MORE force in the original recovery-push direction than the matched control would command on those SAME noisy observations and action innovation. This comparison does not execute a mid-trajectory switch or prove that changing only one action would recover the episode.

Over the same duration up to candidate failure, candidate peak cart speed is5.545–8.341m/s; the matched control's peak is2.997–3.952m/s. All12 candidates show the larger peak. The trajectories catch the initial pole fall, carry too much momentum through the return toward upright relative to the surviving control, reverse sharply, and hit the opposite pole-angle limit. This is not a slow drift after several minutes of stable balancing.

### Concrete example: the first indexed final long deterministic failure

Seed41008, final checkpoint512, long-det episode10:

- Both incoming and control policies complete the original300-second trial. Candidate fails at1.80seconds.
- At0.25seconds on the CANDIDATE trajectory: x=-0.562412m, velocity=-3.672127m/s, angle=+0.071172rad, angular velocity=-1.096447rad/s.
- Candidate commands **-9.146535N**, continuing the leftward push.
- The matched control evaluated on the EXACT SAME noisy observation would command **+5.048381N**, braking that leftward motion. These same-observation values must not be confused with control actions on its own already-different trajectory.
- Candidate later reaches about-0.43rad, reverses through upright, and crosses+0.6rad. Its peak cart speed is5.789m/s versus2.997m/s for control over the same1.80seconds.

The accompanying plot uses this first indexed long-panel failure as an illustration; all12 selected failures and every contrary training history are included in the tables. It is not a cherry-picked independent significance test.

## What is and is not established

There are two concrete findings: (1) the mixture modifies retained nominal update signals through shared normalization and parameter gradients, and (2) the resulting41008 actor has lost the timing of a difficult nominal recovery maneuver. The analysis does NOT establish that the normalization mechanism alone produced all12 failures, locate the single first harmful update inside32–128, or prove that separate normalization is the correct long-run fix.

Both original learning histories and the reference recoverability experiments still matter. Some nominal gradient reversals accompany useful learning; the unchanged control also has within-batch conflicts; and discarded/improperly reweighted recovery information could undo its gains. No counterfactual optimizer or corrected candidate was trained here.

The next discriminating correction test should hold the actual data mixture fixed while isolating cross-source centering/weighting and checking both nominal maneuver retention and outward recovery. Increasing the number of outward starts again would not isolate this mechanism.

## Verification and artifacts

Input binding checks all nine original published ZIP digests and their payload manifests:99 build plus3816 result members. The new replay ZIP is artifact10595171555,60,154,372bytes,SHA2560978e5dbfd7eb8b194e4c8a2f58a162b87c011e2a330b28156ff907aa275009e;106 payload members, including an unchanged copy of the original41008 archive.

Independent calculations retain48 fitted batches, all768 post-step mean checks,736 available pre/post gradients/loss checks, all group projections, all normalized-sign comparisons,675 fixed observation probes for each of56 checkpoint policies, and every original15,360 evaluation outcome. The probe-grid actions are descriptive network queries, not simulated success tests.

Maximum discrepancies: recorded post-step means7.185e-7; recorded actor losses2.875e-7; central directional derivatives along actual parameter displacements2.288e-10. The missing32 predecessor snapshots are explicitly excluded, not silently reconstructed. No complete unexported Adam moments, gradients or random streams are regenerated.

Native replay preflight passes **70 unit tests** (68 inherited plus2 new), seven ordinary balancing controls and nine ordinary learning/evaluator controls, strict Clippy, formatting and protected-source restoration. The replay endpoint is explicitly invoked after preflight. Two historical heavy integration tests remain ignored. All new workflow steps succeed on their first attempt. Production browser/platform workflows were not rerun for this test-only audit.

All36 complete traces contain330,808 transitions. Independent scalar/network and nonlinear-dynamics reconstruction verifies actions, rewards, boundaries and episode sums: maximum mean error4.437e-7,command5.445e-6,state transition4.574e-7,reward1.558e-7,aggregate sum3.638e-12. Cross-policy commands on the same stored observations are calculated, not applied.

**34 independent analyzer tests pass**, including analytic-versus-finite-difference gradients, input Jacobians, group additivity, clipping cases, centering/scale controls, malformed data and corrupted/missing/duplicate/unsafe archive members. Synthetic tests are not controller measurements.

No candidate training experiment was run. Main replay cost is330,808 simulator transitions. Ordinary preflight/regression tests perform additional small training/control work, not fully aggregate-instrumented; zero candidate training does not mean zero total test-suite interaction. Runtime/source versions and all reports are retained in the offline evidence.

Evidence package: `rustrobotics-ppo-outward-tradeoff-evidence.zip`, with unchanged original and new raw archives, exact executed replay source, independent NumPy analysis/tests, numerical CSV/JSON tables, plot and replay instructions. A small analysis-only package is also supplied. Offline replay never executes Rust/native code, contacts GitHub, or trains. Existing controller-qualification failures remain failures; no merge, fallback, reward change or production lambda change is made.