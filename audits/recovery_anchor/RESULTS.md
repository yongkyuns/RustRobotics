# Recovery-policy preservation: reduced drift, incomplete protection

September 19, 2026. Issues #35/#33. Corrected nonlinear baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The bounded experiment is complete. Preserving the old policy on reset/recovery observations reduces independently measured policy drift and retains substantial progress on the original training batch. It improves some control comparisons, including a beneficial seed204 outward-recovery update. However, seed201's reset-return update remains confidently harmful, and seed203 regresses relative to the unregularized update. The predeclared necessary safety gate fails. No production default or deployment is promoted.**

This is one conditional update from each of four previously exposed histories, not a from-scratch or repeated-update learner. Protecting the existing policy can also suppress useful changes and cannot supply correct recovery actions when the incoming policy is already inadequate. No claim of a uniquely established root cause or hardware reliability is made.

## Protocol and executed code

[Protocol fixed before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5742117446). Branch `audit/ppo-recovery-anchor-20260919`. Executed commit **0df5333fd4cc9e33b40f7eb059876ae94322d143**; [run35444939466](https://github.com/yongkyuns/RustRobotics/actions/runs/35444939466). All five jobs passed on their first attempt: preflight and four measurements. Successful execution is not a policy-quality pass.

Replay the actual lambda1 training histories201–204 through update8191, then capture the same512-row batch for update8192, ending at4,194,304 training transitions. Keep live actor/critic modules, parameter identities, both Adam histories, environment and RNGs. Recreate the previous mean-four sampled-tail correction with identical tail draws. The incoming networks, original batch, four tails, corrected targets, unregularized endpoint and every original optimizer transaction reproduce the preceding tail-support experiment exactly.

### Four fixed policies

1. Incoming old policy.
2. Original mean-four corrected update, with no new regularizer.
3. The same mean-four update plus a policy-preservation penalty on the original near-upright512-row training batch (control).
4. The same update plus the identical penalty on512 independently collected reset/recovery observations (candidate).

Both regularized arms add beta=1.0 times mean((new latent mean - old latent mean)^2/(2*sigma^2)), with fixed latent sigma0.1, at each original actor minibatch. This is the analytic Gaussian KL for equal fixed variance; the same bounded-force transformation is unchanged. It is a soft penalty, not a hard trust-region bound, stability proof or evaluation-based rejection rule. Beta was not tuned after results were seen.

Only the actor objective receives this test-only addition. All original PPO observations/actions/old likelihoods/advantages, corrected critic targets,16 actor/critic optimizer steps, minibatch permutations and optimizer histories remain fixed across paths. Critic weights match exactly at every step between all three update paths. Gamma0.99,lambda1 for this diagnostic history,rollout512,minibatch128,four epochs,Adam epsilon1e-5,learning rate0.0003,2x64ReLU networks,reward/noise/plant/exploration remain unchanged. Supported production lambda remains0.95.

### Fitting and validation observations

Collect32 independent streams from the original reset distribution under the incoming stochastic policy,64 simulator steps per stream. Retain the noisy observation at steps0,4,...,60, giving512 fitting observations. A separate identically constructed512-observation bank uses a disjoint RNG domain and is never used in updates. The original512-row training batch supplies the equal-sized control bank. This changes which states the regularizer sees, not its coefficient or number of observations.

The banks use the actual `PendulumEnv`, original noise and reset semantics, and independent action/environment RNGs. Native fingerprints require that collection leave the live trainer and its random cursors unchanged. Only noisy observations and incoming policy means enter the preservation penalty. Rewards, actions and ending flags are archived for audit/costs, not selected by score or used as training labels. All measured64-step bank streams survived; reset-after-failure handling remains in the collector but was not exercised by these measured banks.

The fitting-bank maximum observed pole angles are0.2744,0.3803,0.2473 and0.3372 radians across seeds201–204, substantially beyond the near-upright original batches. These are early original-reset trajectories, not a uniformly sampled recovery basin. In particular, the more severe outward-start evaluation distribution is not used for fitting.

## 1. Preservation works without freezing the batch update

The independent validation bank measures KL from the incoming policy. It has never contributed a gradient.

| Seed | Unregularized validation KL | Recovery-regularized validation KL | Reduction | Original corrected-batch surrogate gain retained |
|---|---:|---:|---:|---:|
|201|0.00268440|0.00160760|40.1%|101.3%|
|202|0.00807172|0.00250490|69.0%|90.2%|
|203|0.00034635|0.00012260|64.6%|95.7%|
|204|0.00052783|0.00022318|57.7%|45.4%|

All four candidates retain more than the predeclared25% minimum of the unregularized positive surrogate gain. This rules out labeling an almost unchanged actor as the whole success. Seed201 slightly exceeds100%; KL regularization changes the optimization path, so it does not necessarily trade exactly one-for-one with the PPO objective.

The same-coefficient near-batch control retains82.5%,74.2%,99.3% and47.7% of the original gains. Its validation KL values are0.00121881,0.00295638,0.00021721 and0.00052147. Recovery-bank regularization produces lower validation KL than the near-bank control on three seeds, but not201. Equal coefficients are not equal achieved distribution changes. These comparisons do not isolate a universal state-coverage mechanism.

## 2. Independent returns: a useful component, not complete protection

Each seed evaluates all four FULL frozen policies on256 new paired noisy original resets and256 paired outward-recovery starts. The cap is2048 steps, or20.48 seconds. Failures stop immediately, without automatic reset masking. This is not the previous60-second or five-minute qualification screen.

Outward starts use random sign,cart displacement0.8–1.2m and same-sign outward velocity0.4–0.8m/s,angle uniform±0.15rad and angular velocity±0.3rad/s. All policies share each initial condition, observation and random-innovation key. Fitting-bank, validation-bank and both evaluation domains are separate. Full traces cover the predetermined first replication of every panel/policy; remaining replications retain aggregate outcomes.

The48-comparison99% paired Student/Bonferroni family comprises four seeds x two panels x six contrasts: each update versus incoming, each regularized update versus unregularized, and recovery-specific versus near-batch. These are approximate intervals over evaluation draws of fixed policies, not confidence about a population of newly trained agents. Completion counts and undiscounted returns are descriptive. No favorable seed or coefficient is selected.

### Candidate versus incoming policy

Negative means the update loses discounted return. The unregularized effects below use the same NEW panel as the candidate, not the old report's different panel.

| Seed | Original-reset unregularized effect | Original-reset recovery-preserved effect [adjusted interval] | Outward-recovery unregularized effect | Outward-recovery recovery-preserved effect [adjusted interval] |
|---|---:|---|---:|---|
|201|-0.281435|-0.123704 [-0.201475,-0.045933]|-0.336967|-0.201629 [-0.454086,+0.050829]|
|202|-0.159077|-0.063262 [-0.131762,+0.005239]|-0.118788|-0.040962 [-0.131464,+0.049540]|
|203|+0.005539|-0.028191 [-0.065853,+0.009470]|+0.020522|-0.001966 [-0.011933,+0.008001]|
|204|-0.088203|+0.001810 [-0.138066,+0.141686]|-0.039718|+0.036038 [+0.012008,+0.060067]|

**Seed201 still has a confidently harmful reset update. Seed204 has a confidently beneficial outward-recovery update. The remaining six candidate-versus-incoming comparisons are inconclusive.** Loss of statistical significance is not proof that harm has been eliminated or that the candidate is noninferior. Consequently, the necessary no-confirmed-harm criterion fails even though the surrogate-retention criterion passes.

### Relative improvements and regressions must both be retained

Recovery preservation versus the unregularized update:

- Seed201 improves reset return by+0.157731 [+0.052892,+0.262570] and outward-recovery return by+0.135339 [+0.032816,+0.237862]. The reset update is less harmful, not beneficial relative to incoming.
- Seed204 improves outward-recovery return by+0.075756 [+0.038516,+0.112996], turning that observed negative update into a positive one relative to incoming.
- Seed203 worsens reset return by-0.033730 [-0.054570,-0.012890] and outward-recovery return by-0.022488 [-0.043644,-0.001332]. Preserving behavior can remove a useful change.
- Other differences versus unregularized are inconclusive.

The equal-weight near-batch control also helps some comparisons. Recovery-specific versus near-batch is confidently better only for seed202 resets,+0.067305 [+0.008013,+0.126596], and confidently worse for seed203 resets,-0.012921 [-0.023372,-0.002470]. All other contrasts between the two regularizers are inconclusive. Do not interpret reduced drift or isolated gains as proof that the recovery-bank distribution is universally superior to ordinary update regularization.

### Completion totals, with all failures included

| Panel, pooled over four existing policies | Incoming | Unregularized mean-four | Near-batch KL | Recovery-bank KL |
|---|---:|---:|---:|---:|
|Original noisy resets /1024|715|707|713|716|
|Outward-recovery starts /1024|454|440|448|447|

The candidate slightly exceeds incoming pooled nominal completions but still has fewer pooled recovery completions. These totals are not1024 independently trained controllers. The incoming policies themselves are far from robust; matching their actions can preserve their limitations. A preservation penalty does not provide improved action-credit targets on recovery states, and no repeated-application or long-run training experiment was performed.

## Verification

All five native jobs complete on their first attempt with no failed setup or favorable retry. Preflight passes80 native unit/audit tests,seven ordinary balancing controls,nine ordinary learning/evaluator controls,strict Clippy,formatting and source restoration. Three large audit endpoints remain ignored during preflight; measurements explicitly invoke only the new recovery-preservation endpoint. Separately ignored historical heavy integration tests are not counted as executed.

New controls verify identical-policy zero KL,analytic Gaussian value/mean derivative,bank selection/reproducibility/live-state isolation,and zero-beta exact ordinary optimization including the subsequent update with warm Adam. Existing anchor tests retain their optimizer-history/sham/next-update checks. The active penalty uses real Burn autodiff within the original optimize loop; it does not substitute a Python PPO implementation.

Offline verification checks all five published ZIP digests,80 build members,724 current measurement members,and1328 members of the four directly used nested historical tail-support archives. Every original batch/target/tail/incoming weight/unregularized endpoint and all16 unregularized optimizer transactions reproduce historical bytes. All three paths have identical minibatch permutations and critic weights at every step.

Checks cover8192 evaluation records,2048 original rollout rows,4096 fitting/validation observations,16384 recorded bank transitions,and192 exported actor/critic transaction pairs. Thirty-two predetermined first-replication full evaluation traces contain43580 transitions. Independent reconstructions include reward/end/force semantics,nonlinear dynamics,policy draws and value/credit sums on those full traces. Other evaluation repetitions retain aggregate records,not complete trajectories.

Maximum reconstruction errors: actor minibatch loss5.34e-8,critic minibatch loss2.69e-6,added KL loss1.42e-9,post-step training means4.35e-8. Selected evaluation traces: dynamics2.66e-7,actor means1.72e-7,critic values3.75e-5,rewards1.54e-7,episode totals3.27e-13 and credit2.14e-13. These are independent floating computations with fixed tolerances, not complete regeneration of unexported gradients,Adam records or random sequences. Full optimizer-state preservation is checked by the native controls,not inferred from bare weight snapshots.

All26 independent analysis tests pass,including corruption/unlisted/missing evidence,invalid weights,f32 decoding,paired statistics,Gaussian derivative and lambda-boundary identities,and mutations of real outcome identities,domains,lengths and record counts. No observed field or tolerance was changed to obtain acceptance. There were no native or offline analysis failures requiring a measurement rerun.

Actual training/evaluation uses Rust1.98.1/Burn0.20.1 with the production lockfile. Offline analysis uses Python3.13.5,NumPy2.3.5,SciPy1.17.0. The offline verifier never loads the archived native executable or starts training. The full platform/browser matrix is not rerun for this audit-only branch; production source on PR38 is unchanged.

## Compute and disposition

Main costs:16,777,216 replayed training interactions;8192 sampled-tail interactions;8192 fitting-bank and8192 validation-bank interactions;10,207,033 evaluation interactions. Each network has524224 prefix Adam steps plus256 target steps,including unregularized,two regularized and zero-beta replay paths. The two regularized updates add65536 actor KL sample-visits across the study,besides ordinary PPO forward/backward work. Teacher/validation inference and generic preflight interactions are additional; no equal-compute or all-inclusive efficiency claim is made.

The candidate passes the nontrivial-batch-progress check but fails preservation of independent performance on every tested case. Keep it as a bounded,partially helpful component,not a supported default. This does not uniquely resolve coverage,critic bias,correlated advantages,normalization or unseen-state generalization. No actor-reward labels from the new recovery observations were tested.

Evidence bundle: `rustrobotics-ppo-recovery-anchor-evidence.zip`,containing all unchanged current raw artifacts,previous inputs nested inside,actual prepared source/native runner,all banks/targets/weights/outcomes,traces,independent analyzer/tests,full numerical reports,provenance and offline replay instructions. No new held-out training cohort,five-minute candidate robustness screen,production/default/PR38/master modification,merge or deployment occurred.
