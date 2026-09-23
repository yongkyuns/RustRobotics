# Online critic tradeoff attribution

23 September 2026. **Completed offline analysis of the existing native pilot, not a new training experiment.** The nominal return loss is mostly present in episodes both policies complete. Discounted return and whole-episode reward can rank the same successful policies oppositely. Saved actor-fitting batches also retain strong trajectory offsets and highly uneven advantage magnitude.

Source: run35861559555, executable cfe683840094a2eea13dbd020b53e6b588ed511d, preceding result report89085532. Procedure recorded in issue35/comment5799555256 after final pilot outcomes were exposed but before these new calculations. No policy, checkpoint, training budget, acceptance threshold, reward or gamma changed. No new simulator episode or optimizer update.

## 1. Most nominal loss is not caused by the extra failures

History41004, final extra-fresh minus baseline;512 paired cases per panel. Each contribution is a sum of paired discounted-return changes divided by all512 cases.

| Panel | Whole-panel change | Both complete | Both fail | Newly lost |
|---|---:|---:|---:|---:|
| Nominal deterministic |-1.603394|**-1.389572**,506 cases|-0.112605,4 cases|-0.101218,2 cases|
| Nominal stochastic |-1.643971|**-1.381493**,506 cases|-0.037294,2 cases|-0.225184,4 cases|

Thus **86.7% and84.0%** of the respective deficit comes from shared-success episodes. It is not merely an average dominated by two/four newly failed episodes. This partition is descriptive and conditioned on outcomes, not a new population significance test. All complement partitions remain retained.

Old-batch's smaller41004 deterministic loss has a different composition: shared successes contribute-0.06870 and four shared failures-0.08203, with no gained/lost cases. The two interventions need not have the same failure mechanism.

## 2. Successful episodes expose a temporal tradeoff

Within the506 shared-success cases, both trajectories last2048 ticks:

| Extra-fresh minus baseline,41004 | Discounted-return mean change | Undiscounted episode-total mean change | Applied-force RMS mean change | Peak absolute pole-angle mean change |
|---|---:|---:|---:|---:|
| Nominal deterministic |**-1.40605**|**+21.06094**|+0.37277N|+0.02668rad|
| Nominal stochastic |**-1.39787**|**+22.57258**|+0.24754N|+0.02758rad|

Mean peak absolute cart excursion decreases by0.03040m/0.03126m respectively. The fresh policy is not uniformly worse by every measure: higher total reward and slightly smaller peak cart displacement coexist with more force, larger pole excursions and lower discounted reward. Opposite signs of weighted/unweighted reward differences necessarily imply a time-dependent tradeoff. Full traces for all506 cases were not exported; particular first-second penalty causes are not inferred for the population.

The evaluator uses float32(.99) promoted to f64 at dt.01. Reward weight after5s is0.0065705; after10s0.00004317. Training uses the unchanged float32(.995), whose5s weight is0.0815721. This explains possible ranking differences but does not prove that evaluation discount caused the learned changes. No gamma change or new reward rule was tested.

There is a direct failure-versus-return ranking counterexample in41006, fresh versus old-batch: **all12 lost nominal deterministic cases and all10 lost nominal stochastic cases have HIGHER discounted return under the policy that fails**. Failure times are4.28–9.26s and4.83–11.50s respectively. All have lower undiscounted totals. Similarly,12 of13 old-batch outward cases lost versus baseline have higher discounted return despite failure. Original acceptance screens remain unchanged; a return improvement is not relabeled as successful balancing.

## 3. Uneven finite-batch credit persists

All24 detailed ordinary updates were analyzed: seeds41004/41006, three arms, updates4097/4128/the witness4137 or4140/4224. Every one of16 saved actor checkpoints per update was examined:384 total. The first ordinary update is identical across arms in each history, so duplicate controls are not independent training histories.

Each ordinary batch contains512 primary continuing-rollout observations and64 selected rows from each of8 supplemental streams. During this interval streams0–3 use nominal starts;4–7 use outward TRAINING starts. Start provenance is not a recoverability label. A separate fixed noisy-observation partition uses |x|>=.5 and x*v>0 as an outward-moving-state proxy.

A concrete final example:41004/extra-fresh/update4224.

| Provenance | Rows | Raw advantage mean | Raw advantage SD | Normalized mean | Fraction of squared normalized advantage magnitude |
|---|---:|---:|---:|---:|---:|
| Primary continuing rollout |512|+0.00842|0.19854|-0.19819|**2.00%**|
| Nominal-start supplement |256|+0.63359|2.34304|-0.11745|2.63%|
| Outward-start supplement |256|+5.52120|14.58957|+0.51382|**95.37%**|

Global normalization mean1.54291, SD7.74253. All512 primary normalized advantages are negative; their within-primary SD after scaling is about.02564. About73.1% of full-batch variance lies between contiguous episode/stream groups. Under the separate physical proxy,268 rows carry95.93% of squared magnitude, so this example is not merely a provenance-label artifact.

**Squared advantage magnitude is NOT parameter-gradient share.** Network Jacobians, sampled innovations, clipping and Adam matter. Negative normalized advantages do not establish that these actions should instead be reinforced. Group offsets may contain real return variation, value error and return noise. No new independent action-credit oracle was executed, so offsets are not automatically labeled critic error.

The final whole-batch clipped-surrogate gain is+.00240183. Contributions, on the same1/1024 denominator: outward-start+.00242912, primary-.00000124, nominal-start supplement-.00002606. The primary term is effectively near zero, not a demonstrated substantial behavioral loss. Pooled sampled-objective improvement does not establish benefit in every region.

Contrary case: old-batch fitting in41004/final lowers between-group variance to18.0%, yet outward-start rows still carry93.64% of squared advantages and both other provenance groups have negative final surrogate contributions. An improved offset statistic is not a complete safety condition.

The exact additive function-space score is mean[A*(latent-mu_old)/sigma^2*(mu_new-mu_old)]. Splitting A into group mean and within-group residual reconstructs the total on the original global normalization scale. This is a sampled first-order objective diagnostic, NOT true action credit, a parameter-space gradient cosine or an independent expected-return prediction. All384 checkpoint decompositions, KLs, clipped changes and group partitions are retained in report.json. Missing incoming parameter snapshots and optimizer moments are not invented.

## 4. The nominal deficit emerges later in the retained window

Use the SAME first64 cases across checkpoints, not earlier64 versus final512 totals.41004 fresh-minus-baseline discounted-return changes:

| Panel |4097|4128|4224|
|---|---:|---:|---:|
| Nominal deterministic |0|+.06527|**-1.45739**|
| Nominal stochastic |0|-.00611|**-1.03209**|

The sizable deficit appears between the retained4128 and4224 observations. This does not identify one causative update inside that interval. Each arm's policy and state distribution evolve, so later cross-arm fitting batches are not a matched-state causal comparison.

On identical saved baseline nominal-deterministic replication0 observations, the final41004 fresh actor's mean absolute command difference is.865N, versus.130N for old-batch. That example change is not explained by different observations. It remains one trace, not an estimate over all episodes.

## 5. Example traces must not substitute for the full panel

All36 retained replication0 traces were decomposed into survival reward, cart position/velocity, pole angle/angular velocity, commanded-force penalty, terminal reward and an explicit float32 arithmetic residual. Fixed windows:first1s,1–5s,after5s,full. Applied noise force is not substituted into the commanded-force reward penalty.

The41004 nominal deterministic and stochastic replication0 examples actually IMPROVE under fresh fitting (+.06135/+.06675), opposite to the whole-panel means. They cannot be presented as representative illustrations of the mean regression. The outward example improves+1.87374: a+4.88814 position-penalty benefit is partly offset by-1.87893 control,-.59548 velocity and other changes; its first-second change is negative.41006's outward replication0 is an early failure, also contrary to its aggregate improvement. All examples and contradictory components remain retained.

## Interpretation and next causal boundary

The prior baseline-only4140 intervention remains causal evidence. This analysis adds two separable concerns: uneven and offset-dominated sampled action credit, and metrics that rank safety/completion and discounted behavior differently. It does not justify treating every nominal deficit as catastrophic forgetting, every offset as critic error, or every recovery gain as improved value estimation.

The next causal test should reconstruct the later41004 nominal-regression interval, capture the actual incoming optimizer state and batch at the deteriorating step, and independently measure nominal/recovery action credit at that SAME policy. A baseline-only intervention could then separate bad credit from a real task tradeoff without changing reward, gamma, mixture weights and optimizer history simultaneously. Subtracting trajectory means or adopting a new normalization/head-refresh rule from this exposed diagnostic is not a demonstrated fix. No new native run or treatment was launched.

## Verification and preservation

**19 new local tests pass with warnings treated as errors.** Controls cover all24 sequential-float32 normalizations and behavior-row identities; all384 saved checkpoints' mean checks and additive decompositions; independent Torch versus NumPy inference/first-order derivatives/clipped objectives at all24 final actors; all13824 evaluation identities; all36 trace return reconstructions; paired accounting, terminal semantics, corruption and malformed input. Maximum f64-inference discrepancy versus saved native f32 actor means6.29343e-7. Very small subgroup objective signs are not treated as strong behavioral effects.

Both original outer result hashes pass; **1052 consumed member hashes**,526/history, are reverified. This is not another all11035-nested-member audit. Retained source subsets carry complete original manifests and exact-member indices. A retained-input rerun reproduces report.json and both retained-source ZIPs byte-for-byte. An extended extraction first exceeded a20-second command timeout; its log is retained and completed/repeated executions use unchanged data/formulas.

No new fitting, optimizer transaction, simulated transition or evaluation episode. Numerical inference and analysis are real work; prior native study costs remain. Only this report is added to the repository, outside workflow triggers. Source,19 tests, full computed report, all1052 exact consumed source members, manifests and logs are in the conversation bundle. Production/defaults/master/deployment and original screen decisions remain unchanged.

Original outer SHA256s:
-41004:b4e3364d19ea79f334e4a6e9734e562cbd02cbba09e3e0deec15436dfdc02687.
-41006:0462dbc96e11bc4c3730e8d20532b9258629bcba7ebfa7f016215b50bda0ce32.
Computed report.json SHA256:4900e5923635ab66311bcd387b4bbd9d6bec0e9a8763ddf6d0efcc1fcde2edd0.

Methodological context: PPO optimizes a sampled surrogate through minibatch updates, not an episode-completion count. Schulman et al., Proximal Policy Optimization Algorithms, arXiv1707.06347. Numerical findings here derive from retained native artifacts, not that paper.
