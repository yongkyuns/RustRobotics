# Sampled-tail target correction: better cutoff values, harmful updates remain

September 19, 2026. Issues #35/#33. Production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The bounded correction test is complete. Averaging four sampled reward tails improves the cutoff-value estimate on all four tested histories. It makes two updates less harmful, but does not produce a confidently beneficial update relative to the incoming policy on any seed. Three corrected updates still have confidently negative independent reset-return effects. Do not promote this as a reliable-controller fix.**

The correction successfully changes the intended value/target quantity; it is not a disabled intervention. This separates a real bootstrap-value error from the broader failure of a finite-batch update to generalize. It does not establish that cutoff errors never matter, that all action-credit rankings have now been repaired, or that recovery coverage is the uniquely proven remaining cause.

No production defaults, PR38, master, deployed assets or MuJoCo/control files were changed. There is no merge, new supported training mode, selected deployment checkpoint, or hardware qualification.

## Protocol and execution

[Protocol before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5741354484). Audit branch: `audit/ppo-tail-support-20260919`. Executed commit **21f7071011bce472237d65c2c784983100f34039**, [run **35439763116**](https://github.com/yongkyuns/RustRobotics/actions/runs/35439763116). All five jobs passed on their first attempt: native preflight and four measurements. Execution success is not policy acceptance. No measurement was retried or excluded for a favorable score.

This intervention uses the four exposed lambda1 training histories from the prior actual-rollout diagnostic. Each ordinary native trainer is replayed through update8191, preserving environment, partial episode, RNGs and both Adam histories. The captured512-transition batch for update8192, incoming actor/critic, original updated actor/critic, every original optimizer transaction and historical evaluation records reproduce the previous evidence byte-for-byte.

The candidate is therefore conditional on the SAME actual lambda1-trained incoming learner; it is not a new from-scratch lambda0.95 policy, a comparison of selected best checkpoints, or an exact-resume file loaded from bare weights.

## Exactly what changes

From the original rollout-end physical state and its original noisy observation, draw four512-step continuations using the same frozen old actor and actual corrected nonlinear `PendulumEnv`. Draw0 clones the unconsumed environment/action RNG cursors; the other three use separate predetermined domains. Sampling cannot mutate the live training session. The physical state and noise/reward/force model are unchanged. Only the external collection clock is re-established for the512-step diagnostic cap; a true failure ends a tail without bootstrap or reset.

Each tail value is its sampled discounted rewards plus the old critic's final-state value when it survives512 steps. All16 measured tails in this study survive512 steps; true-terminal behavior is separately checked by native controls. The four-tail mean replaces only the bootstrap at the original batch's final live-path boundary. The existing lambda1 GAE and advantage normalization then run, followed by the unchanged ordinary optimizer. Four single-tail variants are prespecified sensitivity controls, not candidates selected after seeing scores.

The optimizer receives EXACTLY the original512 observations, latent actions, old log probabilities and minibatch permutations. Suffix actions are never added to either network's training minibatches. Every variant starts from an internal clone of the same modules, parameter identities, Adam records and shuffle cursor. There are still16 actor and16 critic steps in each candidate update; no change to architecture, losses, reward, gamma, learning rate, exploration or normalization.

This requires a cloneable simulator and additional interactions. It is not free data, a hardware-model-free operation, or a demonstrated repeated-update collector. The suffix contains ordinary policy/environment samples, not an evaluation oracle or supplied controller. Independent evaluation returns never enter training or policy selection.

For a sample with h rewards remaining in the original live path, the unnormalized lambda1 correction obeys

`A_tail(t) - A_original(t) = gamma^h * (sampled_tail_value - old_cutoff_value)`.

Independent reconstruction verifies this identity up to the expected float32 recurrence differences, maximum5.724e-6. The raw-return/advantage recurrences and normalized outputs reproduce bit-for-bit using sequential float32 arithmetic. Normalized advantage signs change on10/512,23/512,78/512 and33/512 rows for mean-four at seeds201–204. Correcting the cutoff does not remove the original current-state value baseline or global mean/standard-deviation normalization.

## Cutoff-value prediction actually improves

An independent256-trajectory old-policy assay starts at the same cutoff physical state/noisy observation and runs2048 steps or true failure. It uses distinct action/environment random streams and supplies no training labels. These are finite discounted realized returns conditional on the selected state, not an exact infinite-horizon or observation-only value oracle.

| Seed | Original critic cutoff value | Four-tail replacement | Independent mean return | Original absolute error vs mean | Four-tail absolute error vs mean |
|---|---:|---:|---:|---:|---:|
|201|99.438164|99.503021|99.503380|0.065216|0.000358|
|202|99.502144|99.536469|99.541379|0.039235|0.004911|
|203|98.405533|97.411255|97.432731|0.972802|0.021476|
|204|98.897598|99.016327|99.011617|0.114018|0.004710|

All four replacement estimates fall inside their independently measured reference intervals; all four original estimates fall outside. Reference adjusted99% intervals are201[99.491411,99.515348],202[99.529750,99.553008],203[97.406241,97.459221],204[98.997216,99.026017]. Error columns are differences from noisy sample means, not known exact expected errors or a guarantee that four-tail averaging always improves a critic.

The remaining512-step bootstrap is retained and the diagnostic evaluates the existing portable policy path, independently checked against the tensor-built weights. No claim of exact cross-backend floating arithmetic is made beyond the actual replay controls.

## Independent full-policy effects

For each seed, seven frozen policies are evaluated on256 NEW paired noisy resets: incoming policy, original update, four single-tail updates, and the mean-four update. The horizon is2048 steps (20.48 seconds), not the prior five-minute robustness screen. All branches share each reset/innovation key; failures end trials immediately without auto-reset masking.

The fixed48-contrast99% Student/Bonferroni family covers44 policy-return contrasts and four cutoff-reference means. Intervals describe repeated evaluation draws for these frozen policies; four suffixes are not four independently trained agents, and four reused histories are not fresh held-out learning qualification. Undiscounted scores, completion counts and ending types are retained descriptively.

### Primary candidate versus the incoming policy

| Seed | Original update effect | Mean-four corrected effect | Adjusted interval for corrected effect | Corrected classification |
|---|---:|---:|---|---|
|201|-0.273872|-0.275541|[-0.399604,-0.151478]|harmful|
|202|-0.183273|-0.178755|[-0.353509,-0.004002]|harmful|
|203|+0.037463|+0.024868|[-0.019857,+0.069592]|inconclusive|
|204|-0.158481|-0.065532|[-0.115539,-0.015525]|harmful|

These original-update numbers use the new256-pair panel. The separate old128-pair panel reproduces the previous experiment exactly; it is not pooled into the new estimates. Different numerical effects from the earlier report are not changed policies or overwritten measurements.

### Improvement relative to the original update is not improvement over the incoming policy

| Seed | Mean-four minus original update | Adjusted interval |
|---|---:|---|
|201|-0.001669|[-0.019137,+0.015800]|
|202|+0.004517|[+0.002139,+0.006896]|
|203|-0.012596|[-0.030074,+0.004883]|
|204|+0.092949|[+0.039668,+0.146229]|

The correction makes202 and204 less harmful with resolved positive differences relative to their original updates. It does NOT turn either into a beneficial update relative to its own incoming policy. Neither difference is resolved for201 or203.

Fresh-reset completions, incoming/original/mean-four out of256, are201:152/145/145;202:191/186/186;203:206/209/209;204:197/195/196. Gains in cutoff-value accuracy do not automatically preserve the reset controller. Both angle and position failures remain; the full ending tables are retained.

Every mean-four update still improves its own scalar-reconstructed full-batch clipped surrogate: gains0.004993,0.009076,0.000933 and0.000374 for201–204. Loss reconstruction checks agree with the executed optimizer. No surrogate score is used to accept or reject a training update. Fitting/optimizing corrected targets on one batch is not a substitute for independent control performance.

## Single-tail sensitivity retained

The four single-tail values are not uniformly accurate. The mean-four value check must not be presented as every individual continuation being reliable. For201 and202, ALL four single-tail updates remain confidently harmful on the fresh panel. For203, all four single-tail effects are inconclusive. For204, two are harmful and two inconclusive; the point effects range from-0.134106 to-0.005802. None of the16 single-tail updates has a confidently positive effect relative to its incoming policy under the fixed comparison family.

No single-tail variant, seed or alternative estimator was selected after observing those outcomes. The mean-four intervention was declared in advance. This limited test does not estimate an optimum tail length or number of sampled continuations.

## Additional descriptive coverage check

This section is post-hoc diagnosis, not a changed acceptance rule or additional training intervention. The four captured batches contain no internal terminal or timeout. Their maximum absolute physical pole angles are respectively0.019309,0.015661,0.010658 and0.009251 radians. Maximum observed angles, including observation noise, are0.020995,0.017146,0.011949 and0.010850 radians.

By comparison, the unchanged training/evaluation physical reset distribution permits angles up to0.25 radians and angular velocities up to0.5rad/s. The batch fitting occurs on near-upright trajectories, while the fresh-reset test includes substantially larger excursions. Adding futures from the SAME cutoff does not broaden the actor's original training-observation set.

This documents a coverage mismatch, not proof that it uniquely causes the harmful update. The original baseline/current-value errors, finite correlated advantages, normalization and simultaneous changes across the network remain possible contributors. This experiment does not remove any of them or supply independent action-ranking labels at all unvisited states. It also does not rerun the earlier64 first-action-ranking witnesses with every candidate; its direct test is the actual target substitution and whole-policy reset effect.

## Verification and scope

Native preflight passes76 unit/ABI/audit tests, seven ordinary balancing controls, nine ordinary learning/evaluator controls, strict Clippy, formatting, compiled endpoint discovery and source restoration. The two diagnostic heavy endpoints are explicitly invoked only as prescribed; separately ignored historical integration tests are not claimed as executed. Existing anchor controls check not only identical restored weights but the next identical optimizer step. Tail controls check captured-state isolation, original-cursor future parity, terminal bootstrap removal, surviving-cutoff behavior and unchanged-bootstrap target identity.

All five current ZIP digests are verified:76 successful-build payload members and1328 measurement members. The four unchanged nested preceding horizon artifacts have652 members; their directly used historical lambda1 artifacts have100 members. Every incoming batch, original actor/critic endpoint, and all16 original per-minibatch records/weights reproduce the preceding experiment exactly. Every historical reset record also reproduces exactly for every seed.

Independent analysis checks9216 new/replayed outcome records,2048 original training rows, all384 recorded optimizer transactions across six paths per seed, and all8192 sampled-tail physical transitions. Forty predetermined first-replication evaluation traces retain74899 transitions; other evaluations retain aggregate outcomes, not all steps. Physics, force/reward semantics, action/value inference, discounted totals and credit decompositions are independently reconstructed on the exported full traces. Maximum evaluation-trace errors are2.088e-7 for dynamics,1.654e-7 for reward,1.408e-7 for actor mean,3.866e-5 for critic prediction,6.111e-13 for episode totals and2.985e-13 for credit. Recorded minibatch actor/value losses independently reconstruct within4.155e-8 and4.522e-6; post-step mean discrepancy is below4.648e-8.

All24 independent analyzer controls pass, including corrupted/unlisted/missing evidence, duplicate records, f32 decoding, terminal/combined endings, cutoff arithmetic, paired statistics and historical mismatch reporting. Neither measured fields nor tolerances were relaxed. Main CI had no failed preflight or measurement. No favorable rerun occurred.

The analyzer does not independently replay every unexported evaluation trajectory, RNG draw, gradient or Adam moment. Moment preservation is established by actual native anchor/sham/next-step controls and historical weights, not inferred from weight-only files. Offline replay never executes the archived native binary.

Actual training/measurements use Rust1.98.1, Burn0.20.1 and the pinned lockfile. Offline analysis uses Python3.13.5, NumPy2.3.5 and SciPy1.17.0, recorded separately. The full platform/browser workflow matrix was not rerun because production code and PR38 were untouched.

## Costs and disposition

Main replay cost:16,777,216 original-history training interactions. Additional sampled tails:8192 simulator interactions. Fresh reset evaluations:11,048,326 interactions; historical binding panel:1,590,432; independent cutoff-value assay:2,097,152. Per network there are524,224 prefix Adam steps and448 target steps, including all sensitivity/sham executions. Generic preflight experience is additional and not fully aggregate-instrumented. These are not new independently trained successes or an efficiency comparison.

Retain the sampled-tail intervention as evidence that correcting the scalar cutoff value is insufficient to repair the entire update. Do not install mean-four tails as a supported default from these results. The next useful correction must preserve performance beyond the near-upright fitting batch, rather than assume an improved value prediction makes every policy update safe.

Self-contained conversation bundle: `rustrobotics-ppo-tail-support-evidence.zip`. It contains all five unchanged current raw archives, historical inputs nested inside, exact prepared source/native runner, all candidate targets/weights, all retained outcomes/traces, independent analyzer/tests, supplementary coverage results, hashes and offline replay instructions. Production lambda remains0.95; PR38/master/defaults/deployment are unchanged. No from-scratch repeated-tail learner, fresh held-out training cohort, five-minute candidate robustness screen or hardware qualification was performed.
