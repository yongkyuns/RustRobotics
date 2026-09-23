# Bounded online critic pilot: recovery gains, but neither fixed treatment passes

23 September 2026. **The native two-history experiment is complete. Extra critic fitting improves recovery completion in both histories, but neither treatment passes the predeclared two-history screen. Fresh current-policy fitting is not uniformly better than equal-work old-batch fitting.** This is not a production recipe, a from-scratch result, or a result for the separate output-layer calibration method.

## Fixed experiment and execution

Protocol: issue #35 comment 5793892853; implementation note 5794076791; input-repair notes 5794908769 and 5794993042. Run **35861559555**, executable **cfe683840094a2eea13dbd020b53e6b588ed511d**, passes preflight and both native measurement jobs. All **13,824 registered evaluation records** are present, 6,912 per history. No case, threshold, candidate, training budget, or stop point changed during this verification.

Seeds 41004 and 41006 reconstruct their original gamma .995/lambda 1/support1024 histories through update4096, then fork into three arms for exactly128 updates through4224. Baseline uses unchanged learning. Extra-old-batch performs the ordinary16 actor/16 critic minibatches, then48 additional critic-only Adam minibatches on the current ordinary update's original1024 fitting rows and frozen return targets. Extra-fresh performs the same48 extra critic minibatches on4096 selected rows from independent trajectories of the just-updated actor. Extra fitting is AFTER the actor update; stored actor advantages are not recomputed mid-update. The resulting critic is carried into subsequent ordinary value/GAE/bootstrap computation with its live optimizer state.

This trains the full existing critic, not the cheap output-layer solver. It imports no fitted critic, resets no optimizer, performs no policy rollback, and selects no favorable checkpoint. Fresh fitting combines current-policy targets and representative starts; it does not isolate coverage from target freshness.

Both treatments have the same12288 extra critic row visits/update:12 passes over1024 old rows or3 passes over4096 fresh rows. Both have64 total critic transactions/update versus baseline16; actor transactions remain16. All three arms collect8 independent1536-step streams/update, including baseline and old-batch arms that discard them for fitting. Four streams start nominally; four use the existing outward TRAINING ranges, not the evaluation-start ranges. Only the first512 observations per stream supply fitting rows, with long returns and the original frozen far bootstrap. This matched-collection experiment is not interaction-cost equality with unaugmented production.

## Final short-panel completions

Each cell is out of512 paired evaluation episodes, cap2048 steps at dt .01 (about20.48s). Deterministic refers to policy action selection; environmental noise remains. Evaluation discount is float32(.99) promoted to float64, distinct from the unchanged training gamma .995.

| History | Arm | Nominal deterministic | Nominal stochastic | Outward stochastic |
|---|---|---:|---:|---:|
|41004|baseline|508|510|489|
|41004|extra-old-batch|508|510|503|
|41004|extra-fresh|506|506|498|
|41006|baseline|468|464|273|
|41006|extra-old-batch|499|500|425|
|41006|extra-fresh|494|493|373|

Descriptive pooled outward completions are762/1024 baseline,928/1024 old-batch and871/1024 fresh: gains166 and109. These pooled episodes are NOT independent training seeds, and cannot erase a regression in either history.

## Registered screen: BOTH fail, for different reasons

The fixed screen requires no short-panel completion loss versus matched baseline in EITHER history, a strictly positive pooled outward completion gain, and no confidently harmful short-panel discounted-return contrast (the existing descriptive mean +/-2.586*SE interval).

**Extra-fresh fails on41004:** nominal deterministic completion loses2 and stochastic loses4, with no compensating gained cases in those panels. Discounted-return differences are -1.603394, interval[-1.929043,-1.277745], and -1.643971, interval[-2.021871,-1.266071]. Its outward improvement of9 completions does not erase these nominal regressions. On41006 it instead gains26/29/100 short-panel completions, with no lost baseline-success case in those three panels.

**Extra-old-batch fails only the41004 nominal-deterministic return condition:** delta -0.150724, interval[-0.290298,-0.011150]. Its short completion counts do not regress in either history. This is a small adverse contrast compared with its substantial recovery gains; it is nevertheless a failure under the original rule, not grounds to move the threshold after outcomes. The interval is ordinary, pointwise and unadjusted, not a simultaneous population-harm conclusion.

Outward discounted-return differences versus baseline:

| History | Treatment | Mean delta | Descriptive99% interval |
|---|---|---:|---|
|41004|extra-old-batch|+1.234146|[+0.980626,+1.487667]|
|41004|extra-fresh|+0.785629|[+0.527116,+1.044142]|
|41006|extra-old-batch|+3.899174|[+3.349478,+4.448871]|
|41006|extra-fresh|+4.857936|[+4.286500,+5.429372]|

All earlier checkpoints, paired gains/losses, and contrary results remain in the detailed report. Neither treatment is promoted or retuned from these results.

## Equal-work fresh versus old-batch comparison

Fresh fitting has FEWER outward completions than old-batch in both histories:498 versus503 on41004, and373 versus425 on41006. On41004 its outward return is also lower: delta -0.448517, interval[-0.626264,-0.270771]. On41006 its outward mean discounted return is HIGHER by+0.958762, interval[+0.476410,+1.441114], despite52 fewer completions. The paired41006 difference comprises32 gained and84 lost cases relative to old-batch. Higher discounted return is therefore not a substitute for recovery completion.

Old-batch versus baseline on41006 gains165 outward successes but loses13, net+152. Fresh versus baseline gains100 and loses0 there, despite its lower aggregate completion than old-batch. Finite absence of lost cases is not a safety guarantee.

## Long-horizon results are retained

Each final panel contains64 paired episodes. Nominal deterministic/stochastic caps are30000 steps (five minutes); outward cap6000 steps (60seconds).

| History | Arm | Five-minute deterministic | Five-minute stochastic | 60-second outward |
|---|---|---:|---:|---:|
|41004|baseline|64|64|63|
|41004|extra-old-batch|64|64|64|
|41004|extra-fresh|63|64|63|
|41006|baseline|57|59|40|
|41006|extra-old-batch|62|62|51|
|41006|extra-fresh|60|62|47|

Every long-panel survivor is marked centered under the existing final-window criterion. On41004, fresh loses one five-minute deterministic case and has negative nominal long-panel discounted-return contrasts (-1.965108 deterministic, interval[-3.644573,-0.285642]; -1.451581 stochastic, interval[-2.424964,-0.478198]). It improves both nominal long-panel mean returns in41006. Old-batch has no aggregate long-panel completion loss versus baseline, but41006 outward-long gains12 and loses1 case. These results do not establish sustained-balancing reliability across new training histories.

## Critic tracking: fitting improvement is not uniform calibration

Independent calibration streams score the critics immediately before/after an extra phase under the SAME outgoing actor and same noisy observations. The2048-step holdout streams have reward-only finite-horizon targets with ZERO far bootstrap. They are not exact conditional V labels. Training labels retain their original far bootstrap. After policies diverge, between-arm RMSE comparisons are not matched-state causal comparisons.

| History/checkpoint | Old-batch holdout RMSE before -> after | Fresh holdout RMSE before -> after |
|---|---:|---:|
|41004/4097|26.1764 ->19.3269|26.1764 ->13.6373|
|41004/4128|7.4373 ->7.4799|7.2856 ->3.7349|
|41004/4224|1.0198 ->1.2406|1.9242 ->1.6432|
|41006/4097|4.5198 ->3.1992|4.5198 ->3.3331|
|41006/4128|8.3813 ->8.2243|7.9390 ->8.4296|
|41006/4224|2.2941 ->2.2658|6.7775 ->6.0790|

Old-batch fitting loss decreases on256/256 extra phases; fresh on255/256 (the retained exception is41006/update4156). Pooled holdout RMSE decreases on4/6 old-batch panels and5/6 fresh panels. Neither is uniform. Fresh41004/final pooled improvement hides an early-phase RMSE increase3.006 ->3.435. Fresh41006/4128 pooled and outward-start errors worsen. All per-stream, early/late, outward-start, outward-early and signed-mean errors are retained.

The one-off native baseline-causality result and the separate cheap-head generalization failures remain valid. This experiment demonstrates that bounded extra critic work can materially help some continuing histories, but does not establish fresh-data superiority or a no-regression online rule. The next mechanism to isolate is the nominal/recovery tradeoff in resulting actor updates, not another unbounded critic-loss sweep. No from-scratch cohort or automatic head refresh was launched in this verification.

## Offline verification and limitations

Three original outer archives and their four explicitly verified nested historical/upstream archives pass **11035 payload hash checks**:4867+4867 result members,131 build,311+311 historical,274+274 upstream. Embedded ZIP members are also hash-bound as byte streams at the outer level; these counts are not unique independent sources.

Verified32 exported historical actor/critic snapshots across the two histories, including both final4224 pairs. Each baseline harmful-witness ordinary transaction subtree has72 files identical to history,144 total. Initial4097 ordinary transaction files, fresh streams and independent holdout streams are byte-identical across all arms before the extra phase changes subsequent learning. Baseline continuation records' retained fields reproduce the original128 rows per history; native assertions additionally check every original prefix/continuation update record.

All768 continuation updates,24576 extra critic minibatch records, complete sampled-index permutations and fitting-data shapes verify. Fresh fitting rows bind exactly to reconstructed returns on all256 fresh-treatment updates. Old-batch row identity is independently bound at the four detailed updates/history/arm; unexported ordinary batches are NOT claimed reconstructed. Checked6288 stream return recurrences, including training terminal resets and holdout zero bootstrap, all registered stream keys/start distributions, and training/holdout/evaluation separation. The stream rewards themselves are archived native outcomes, not a separately rerun simulator.

All13824 case identities, caps, boundary classifications, paired contrasts, summaries and calibration summaries recompute exactly. All36 retained rep0 traces reproduce total/discounted returns, state/observation continuity, boundary flags, maxima, force RMS and centring exactly. Complete trajectories for every evaluation episode were not exported and are not claimed independently replayed. Across ALL checkpoints there are1011 position and5 angle failures, with no combined failures; this is not an all-position-failure experiment.

Independent float64 inference through saved critic weights agrees with archived float32 calibration predictions within7.62e-5 maximum absolute difference. This is a numerical diagnostic, not bitwise native prediction identity or a new optimized tolerance. Candidate optimizer moments are NOT exported by this study; continuity/isolation is enforced by native code/tests, not a fresh independent reconstruction of nonexistent moment files.

Native preflight passes120 unit/audit tests,7 balancing controls and9 learning controls, followed by both separately invoked heavy endpoints. Twelve historical heavy library endpoints and two historical heavy integration endpoints are explicitly ignored in ordinary passes. No full product/browser qualification is claimed.

**33 new offline verifier tests pass with warnings as errors.** Tests use actual retained outcomes and cover both-history screening, pooled-regression masking, return-only rejection, duplicate/missing/invalid/nonfinite cases, paired gain/loss accounting, keys, horizons, return boundaries, holdout data, calibration identities, trace mutation, archive integrity/path rejection and signed-zero identity. A full repeat reproduces both verified-history JSONs and the combined report byte-for-byte. An initial verifier exact-match probe multiplied the interval scale before dividing by sqrt(n); the final version follows the published standard-error-then-scale order. That failed verification log is retained. No candidate/result/threshold changed.

## Cost and provenance

Per arm:128 continuation updates and2048 actor transactions. Baseline has2048 ordinary critic transactions; each treatment has8192 total critic transactions. Extra critic row visits per treatment/history are1572864. Additional corpus collection totals9437184 transitions across the six arms, calibration294912, and evaluation48653424. Original-history reconstruction adds104216883 transitions, and ordinary continuation collection adds10223616. Equal pilot collection is not an efficiency claim versus the ordinary learner. The offline verifier runs no optimization or simulation; native study costs are not represented as zero work.

- Result41004/artifact10752868007: b4e3364d19ea79f334e4a6e9734e562cbd02cbba09e3e0deec15436dfdc02687.
- Result41006/artifact10752344226: 0462dbc96e11bc4c3730e8d20532b9258629bcba7ebfa7f016215b50bda0ce32.
- Build/artifact10750404497: 239cc11c5f95973dfc3c8d89cce1cd66f6feb46efebcf99a1645416154c47eb6.

The conversation package retains verifier/tests, computed reports, source/logs, original manifests and selected exact source members for independent outcome/calibration reproduction. The two large complete result archives are separately retained conversation attachments, not silently replaced by partial files. Full stream/minibatch verification requires those original ZIPs. Production/defaults/master/deployment and both experimental algorithms are unchanged; this repository addition is the evidence report only, outside the active workflow's trigger paths.
