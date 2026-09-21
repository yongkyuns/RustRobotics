# Historical regression window complete: update 4,140 damages recovery

Verified September 21, 2026. Issue #35. Registration comment5754579087 preceded all new outcomes. Executed source **a3276e1efa3071d381480bc61f1abe1ebd818a54**, workflow **35554192268**, branch `audit/ppo-regression-window-20260920`. All three jobs completed successfully on their first attempt at September21 02:54:38UTC.

## Decision

**Seed41006/update4140 reduces independently evaluated outward-recovery completions from208/256 to196/256.** The registered adjusted interval excludes zero. The same update improves its fitted PPO objective and lowers critic MSE against sampled targets. This identifies an actual harmful transaction, not merely a gap between distant checkpoints.

No particular erroneous target, gradient component, or corrective algorithm is yet established. The selected update in improving comparison history41004 also lowers independent discounted return, although its completion effects are inconclusive. No new candidate, rollback, reward/discount/normalization change, merge, deployment or default promotion occurred. The longer-support rejection stands.

## Exact historical reproduction

Both deliberately selected exposed histories41004/41006 were reconstructed from original random actor/critic initialization through4224. Gamma.995, support1024, lambda1, global normalization, existing4096 ordinary-reset recovery-union then half-outward recipe, plant, rewards, noise, networks, exploration and Adam schedule remain unchanged.

Live Adam moments, parameter identities, environment, unfinished episode and RNG streams persist during native replay. Historical files are comparisons, never warm starts. All4224 update records per history reproduce. Both networks match0/1024/4096/4128/4224 exactly, and all historical episode fields at the window endpoints match. Weight files are not portable complete optimizer-resume checkpoints.

Every update4129–4224 retains incoming/outgoing networks, primary batch, sampled futures, supplemental support trajectories, returns/advantages and all16 optimizer transactions. Evaluation does not alter training.

## Selection, then independent evaluation

Every one of97 window snapshots receives the same three original20.48-second panels with64 cases. The fixed rule selects the largest consecutive outward-completion decrease, earliest tie. It does not select a production checkpoint or roll training back.

|History|Selected update|Original-panel outward before→after|
|---|---:|---:|
|41006, regression target|**4140**|**55/64→48/64**|
|41004, improving comparison|**4137**|**47/64→43/64**|

41004/update4138 also loses four cases; earliest-tie selection correctly retains4137. All96 changes per history remain.

Each selected before/after pair then receives256 paired cases per panel using independent domain0x20000000, distinct from selection domain0x10000000. These cases enter neither fitting nor selection. Environment noise remains active and failure immediately ends a trial.

Intervals are the preregistered paired Student two-sided99% intervals adjusted across12 contrasts:2 fixed selected updates×3 panels×completion/discounted return. They concern independent evaluation draws conditional on selected policies, not training-population reliability. Binary-effect intervals are the registered approximation, not exact safety bounds.

### Seed41006/update4140: independently confirmed recovery loss

|Panel|Before /256|After /256|Completion effect|
|---|---:|---:|---|
|Nominal mean-action|252|252|No observed outcome change|
|Nominal stochastic|253|253|No observed outcome change|
|**Outward stochastic**|**208**|**196**|**−4.6875 percentage points [−9.1638,−0.2112]**|

There are12 lost outward successes and zero repairs. Every lost case ends at the cart-position limit. Nominal success/failure identities are unchanged, not merely counts. Zero observed paired variation is not proof of zero future risk or equivalence; the resulting empirical zero-width interval must not be treated as a safety certificate.

Common evaluation-discount return effects:

|Panel|Mean effect|Adjusted99% interval|Classification|
|---|---:|---|---|
|Nominal mean-action|+0.030356|[−0.029563,+0.090275]|Inconclusive|
|Nominal stochastic|+0.003670|[−0.026008,+0.033348]|Inconclusive|
|**Outward stochastic**|**−0.436941**|**[−0.542803,−0.331079]**|**Harmful**|

The common evaluation discount is.99 while training uses.995. The return effect is not claimed as an unbiased estimate of the training-discount expected objective. The independent completion loss does not depend on that scoring distinction.

### Seed41004/update4137: harmful local return inside an improving history

|Panel|Completions before→after /256|Completion interval, percentage points|Discounted effect and adjusted interval|
|---|---:|---|---|
|Nominal mean-action|244→239|[−4.8837,+0.9775]|−0.076335[−0.119687,−0.032984]|
|Nominal stochastic|241→240|[−1.7116,+0.9304]|−0.060567[−0.105371,−0.015763]|
|Outward stochastic|185→179|[−5.5477,+0.8602]|−0.251072[−0.335904,−0.166240]|

All three return losses are resolved; all three completion effects are inconclusive. There are5/1/6 lost successes, zero repairs, all lost cases ending at the rail. These distinct panels are not one population of independently trained policies.

## The largest drop is not the whole collapse

41006 starts the window at59/64 outward and ends33/64. It has43 consecutive count decreases,24 increases and29 ties on the reused panel. These are not43 statistically confirmed harmful updates. Counts at4139/4140/4141/4142 are55/48/46/52: recovery rebounds after the selected step, with later losses and gains. The selected update explains a local loss, not an irreversible one-step explanation of the entire96-update decline.

41004 improves37/64→59/64 across the interval, yet has27 count decreases,30 increases and39 ties. Its selected local harm is retained, not excluded because the eventual trend is favorable.

## What the harmful update optimizes

|Recorded quantity|41006/update4140|41004/update4137|
|---|---:|---:|
|Full-batch clipped-surrogate gain|**+0.008415**|**+0.002704**|
|Incoming critic MSE against sampled targets|7.067007|18.636265|
|Final critic MSE against those targets|**3.555546**|**16.502789**|
|Final mean fitting-batch KL|0.005863|0.007071|
|Final maximum fitting-batch KL|0.025321|0.121666|

These are reconstructed finite-batch quantities, not true-value errors or robustness guarantees. This diagnostic did not identify an optimizer arithmetic bug. It establishes that empirical improvement in this transaction does not preserve independently measured recovery.

For41006, surrogate gain contributions are approximately−.000030 primary,+.004904 ordinary-reset supplemental,+.003541 outward-origin supplemental, all divided by1024. Ordinary supplemental samples, not only outward samples, contribute to the fitted gain. The outward quarter supplies34.21% of squared normalized-advantage magnitude, ordinary supplemental61.25%, primary4.54%; the selected41004 outward share is93.00%. Outward dominance is not a necessary signature in these two cases. Energy shares are not percentages of Adam's displacement.

41006/update4140 trajectory-segment means account for79.39% of raw-advantage variance. Their incoming gradient norm is3.01× the within-segment component, with cosine−.452 between components. Both have a positive projection on the actual final displacement. These local derivatives are not counterfactual Adam updates or proof that subtracting segment means fixes learning; improving41004 also has strong mean components.

Across all192 updates,19 finish below their incoming full-batch surrogate (9 in41004,10 in41006). Both selected harmful updates finish above it. All contrary updates and3072 transactions remain.

### No internal-timeout route in the selected target batch

41006/update4140 has no primary true failure or internal timeout. All eight supplemental first trajectories and four cutoff futures provide their requested support without true termination. The largest final-value coefficient for fitted rows is.0058703, not an unextended.995 internal-timeout bootstrap. That coefficient's magnitude is not its estimation error, and this does not certify every critic estimate or target. In selected41004, one supplemental path truly terminates and64 fitting rows use that boundary. Neither terminal presence nor absence uniquely explains the effects.

## Verification and limitations

Native preflight:111 unit/audit tests,7 ordinary balancing controls,9 ordinary learning controls,Clippy,formatting and source restoration. Ten heavy audit endpoints were ignored in the regular pass; current replay separately invoked. Two historical heavy integration endpoints and the production browser/platform matrix were not rerun.

The unchanged109-test offline suite passes on the completed outputs. All3 current ZIP digests/15165 payload hashes and2 nested historical archives/548 payload hashes verify. Both histories'8448 update records,20 historical actor/critic file comparisons and768 historical endpoint evaluations reproduce. All400 top-level weight files are bound to the corresponding incoming/outgoing transactions.

Complete coverage:37248 historical-window+3072 independent=40320 evaluation records;192 detailed updates;98304 primary and1671168 supplemental-support rows;sampled tails;3072 optimizer transactions. Returns/global normalization reproduce bit-for-bit. Recorded loss differences stay within unchanged scale-aware checks (max6.36e−8 actor,1.84e−4 critic).

All24 predetermined full evaluation trajectories/40327 transitions reconstruct actions,nonlinear dynamics,rewards,endings and scores; rewards are sequentialfloat32 exact. Maximum actor/force/dynamics/score differences3.21e−7/2.14e−6N/1.29e−7/4.69e−13. Other replications retain aggregates; no missing paths are invented.

All analytical/finite-difference transaction comparisons are retained. Maximum full-window difference4.73e−5; selected-update maxima below4.83e−9. The unchanged checker logs possible nonsmooth ReLU/clipping crossings rather than certifying every residual as a smooth derivative. No tolerance changes. Unexported Adam moments,RNG draws and native gradients are not independently regenerated.

A streaming container request was unavailable before execution. Identical numerical checks completed in noninteractive segments; no training or evaluation was retried locally. Full source/measurement hashes,all intermediate batch reports and offline instructions are supplied in the conversation evidence package.

## Costs and disposition

Native reconstruction uses107624755 training-data interactions to reproduce8448 old updates, not train a new candidate. Historical-window and independent checks add75173727 evaluation transitions; generic preflight work is additional. No efficiency or hardware-qualification claim.

A concrete, independently confirmed recovery-regression case is now available at update4140. Proposed corrections can be tested at this identified incoming learner state rather than inferred from distant checkpoints. No same-input counterfactual optimizer intervention or repeated-learning fix was tested here.

Production gamma.99/lambda.95,rewards,normalization,learned policies,PR38,master,deployment and fallback remain unchanged. No merge or controller adoption. The prior1024-support rejection stands.