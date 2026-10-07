# PPO recovery episode-credit ledger — October 7, 2026

**Completed read-only development diagnostic. No new simulation, optimization, candidate training, production change or merge.**

Protocols: issue35/comment6049064732 fixed the episode reduction before new statistics; addendum6049103834 fixed all five retained outgoing-snapshot comparisons after the first ledger reduction and before their likelihood analysis. Input is the preceding exactly reconstructed scaled-2048 development seed201 history, not the previous eight held-out seeds.

## Main finding

The308 difficult episodes reaching the256-tick training timeout are not verified completed recoveries: none achieved the previously defined100-consecutive-tick centring window. For282 of those episodes wholly contained in one rollout, an average **85.04% of the initial return target comes from the critic's estimated tail**, not reward observed during that episode.

This identifies where the learner relies on bootstrapping. It does **not** establish that the tail estimate is wrong, that the timeout implementation is defective, or that a longer training episode fixes PPO. Correct bootstrapping can legitimately value an unfinished recovery. No new continuation outcomes were measured here.

## Complete retained episode accounting

Verified1,048,576 actual training rows,512 fitting batches,4,290 starts and4,289 completed episodes. The final ordinary episode is unfinished after136 recorded steps. Difficult starts retain the original definition: |theta|>=.15rad, |omega|>=.25rad/s and theta*omega>0.

| Difficult-start outcome | Episodes | Negative initial stored advantage | Positive initial stored advantage | Negative mean advantage, first25 actions | Completed centring window |
|---|---:|---:|---:|---:|---:|
| Physical failure |120|93|27|92|0|
| Training timeout at256ticks |308|87|221|132|**0**|

The centring window requires100 consecutive post-step samples with |x|<=.25m and |theta|<=.05rad. It is descriptive, not a safety certificate. A separately inspected final-state statistic finds74/308 timeout episodes inside both bounds on the last sample, but still without a complete100-sample window. This is not proof that all are far from equilibrium or will eventually fail.

Among ordinary starts,1,712 of3,593 timeout episodes complete the same window. Thus the training horizon includes settled experience in the broader ordinary-start group; the failure to observe that full window is specific to the difficult starts here.

Fixed quarter difficult-timeout counts65/93/80/70 have11/27/32/17 negative initial advantages. Difficult-failure counts57/13/21/29 have18/3/2/4 positive initial advantages. Different states/noise/policies occur in different quarters; these are not matched causal trend estimates.

## Observed prefix versus estimated continuation

For fully contained episodes, lambda1 allows direct target accounting from the stored reward buffer. True physical termination has zero tail; an external timeout has the original critic augmentation. No label was changed.

For282/308 difficult timeouts fully inside one rollout:

| Initial-target component, physical reward units | Mean |
|---|---:|
| Actually observed discounted reward prefix |**130.68**|
| Discounted tail implied by recorded timeout augmentation |**735.93**|
| Sum |**866.61**|

The per-episode tail fraction averages85.04%, median84.70%, range73.57%–95.31%. Implied timeout value before discounting averages950.76. This is algebraic reconstruction of the reward-buffer augmentation, **not an independently measured true continuation value**. The26 crossing timeout episodes remain in the full ledger but are excluded from this one-rollout decomposition.

For113/120 fully contained difficult physical failures, zero-tail targets match observed terminal-stopped reward-to-go within0.000642 physical units. The other7 failures cross a rollout boundary; their initial fragments can contain a bootstrap and the policy may change before eventual failure. They are not silently treated as fixed-policy terminal targets.

Across all contained ordinary and difficult episodes,919,748 rows reproduce the stored float32 advantage/target recurrence exactly. The separate float64 prefix-plus-tail accounting differs by at most0.003921 over all timeouts (0.003460 over the282 difficult timeouts). This is recorded rounding/accumulation difference, not a modified tolerance or target.

## Credit is relative to the critic, not binary survival

Most failures already receive negative initial credit. Conversely,87 short-horizon survivors have negative initial credit, and27 failures positive credit. Neither count alone proves mistaken expected policy-gradient directions.

A retrospectively selected illustrative failure after the first quarter starts at training transition410230, lies inside update201, and fails after73ticks. Observed stopped return3.6225, stored initial value-14.8309, stored advantage+18.4535. No future bootstrap is present. Its positive sign follows from the low baseline; a single realized trajectory does not reveal the expected return of alternative actions or prove that baseline caused the historical failure.

A surviving episode can similarly have lower return than its current-policy baseline. The historical64-row minibatch permutations are not retained. Whole-rollout centring in the exported ledger is explicitly descriptive, not claimed to be the historical minibatch-normalized credit.

## Five actual outgoing snapshots: mixed, not uniform recovery suppression

The outgoing actor is retained at checkpoints131072/262144/524288/786432/1048576, and its immediately preceding2048-row batch stores sampled latent actions and old log-probabilities. All five updates64/128/256/384/512 were fixed before analyzing those outgoing likelihoods. This examines10,240 actual fitting rows without another optimizer run.

All five **whole-rollout-normalized** clipped-surrogate readouts improve. These are not the unrecorded minibatch-normalized training losses. The difficult-timeout subgroup worsens on this descriptive surrogate at update128; it improves at the other saved updates containing difficult rows. Update384 contains none.

For the1,170 difficult-timeout rows present in those five batches:

| Raw stored-credit group | Sampled-action density increases | Density decreases |
|---|---:|---:|
| Positive advantage,429rows |215|214|
| Negative advantage,741rows |369|372|

No subgroup directions fall within the declared2e-5 ambiguity band. Independent float32/float64 NumPy Gaussian inference has maximum new-log-probability discrepancy3.38e-6 over all10,240rows. Likelihoods use the **unclipped latent Gaussian actions**, not clipped force commands.

This does not show that recovery is uniformly discarded. A credit sign is not a guarantee of an isolated density change under a shared-network update. These five batches contain no physical-failure episode rows, so they cannot attribute how optimization treats the120 historical failures. Incoming actor weights, actual minibatch permutations and every intermediate policy are unavailable; no Adam-displacement, causal performance or all-update claim is made.

## Verification and evidence

Immutable input `ppo_recovery_credit_evidence_20261007.zip`,143347568bytes,SHA256 **ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412**. All648 input payload hashes verify. The prior three numerical reports and18 tests replay exactly before the new reductions.

New checks cover all episode/packet/action/reward bindings, physical endings and timeout logic, episode continuity, reset identities, six original checkpoint tensor identities, original diagnostic bytes and full contained-episode recurrence. **20 new genuine-fixture corruption/identity/boundary tests pass**, including rewards/targets/endings/timeout augmentation/ages/ledger data, centring windows, float precision and unclipped action-density semantics.

The companion `ppo_recovery_episode_credit_analysis_20261007.zip` retains exact analyzer/snapshot source, tests, small genuine fixtures, all episode rows, numerical summaries, per-snapshot readouts, environment and logs. Replay uses the original143MB input ZIP: `OPENBLAS_NUM_THREADS=1 python REPLAY.py --archive /path/to/ppo_recovery_credit_evidence_20261007.zip`. It executes no Torch, native simulator, optimizer or pickle loading. The analysis environment is recorded separately from the historical training runtime.

## Next causal question, not a claimed fix

Test whether the critic's tail predictions at **actual difficult training timeouts** accurately value finishing recovery. That needs the same incoming policy/critic that produced each target, its exact cutoff state/noisy observation, and independent longer continuations. A later saved policy cannot silently substitute for those missing contemporaneous incoming snapshots.

This ledger narrows the next measurement; it does not establish that bootstrapping is the cause or authorize a longer-horizon/teacher/reset/optimizer workaround. No new continuation trial is queued. Production/master/defaults/deployed weights remain unchanged; issues33/35 remain open.
