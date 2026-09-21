# Return-support regression attribution

September 20, 2026 Toronto / September 21 UTC. Issue #35. Registered retained-data plan: comment5754293590. Source study35548063439, executed0a4445f6c5ab83650133a696faaeb764fd5b6c26, predecessor resultd8bf6183.

## Decision and scope

No new controller fix is established; support1024 remains rejected. This step analyzes ALL eight paired histories, both arms, and all three archived updates1/4097/4608. It adds zero simulator or training interactions. No reward, discount, support length, normalization, teacher, inference wrapper, controller selection or production setting changed. The cited poor policies are development histories, not new independent qualification agents.

## First update unchanged; later performance loss is localized

Every first main/corrected/union batch is byte-identical between support512 and support1024, including return targets and normalized advantages. All16 actor and16 critic optimizer snapshots match too. Extra support is not changing the first fitted update.

The first differing retained policy-loss/value-loss/episode-count record is update186/261/246/93/292/162/205/220 for seeds41001–41008. These are first differing METRICS, not recovered first weight divergences or demonstrated harmful updates. Detailed targets/weights at those early differing records were not exported. All occur before outward-start training begins at4097.

A later window is much more useful for explaining loss of recovered behavior. For the long-support41006 policy, the fixed short evaluation counts are:

|Update|Nominal mean-action /64|Nominal stochastic /64|Outward /64|
|---|---:|---:|---:|
|1024|15|7|0|
|4096|64|63|49|
|4128|64|63|59|
|4224|58|59|33|
|4608|54|56|30|

The outward drop59->33 happens within the96 updates4129–4224. Long-support41003 retains64/64 in both short nominal panels through4224, then ends60/64 and59/64 at4608. Distinct panels have distinct cases; these short counts are not substituted for five-minute outcomes. Three archived detailed updates do not identify a damaging transaction inside either missing interval. Weight-only checkpoint loading cannot reproduce persistent Adam history.

## Two different recorded recovery failures

All144 predetermined full evaluation traces are checked. The final nominal-mean replication0 traces for41003 and41006 contain actual new-long-support failures versus successful short-support counterparts, not newly simulated trials.

|Seed/support|Outcome|Peak absolute cart speed|Maximum absolute pole angle|
|---|---|---:|---:|
|41003/512|survives300s|4.132m/s|.317rad|
|41003/1024|rail failure1.70s|7.792m/s|.577rad|
|41006/512|survives300s|2.684m/s|.220rad|
|41006/1024|rail failure3.13s|2.757m/s|.220rad|

41003 drives left to about−2.031m at.77s, reverses, then crosses the opposite rail at+2.401m with+5.802m/s. At its recorded.24s observation the long policy commands−16.797N versus−9.241N when the short policy is queried on the SAME noisy observation. Its own actor immediately BEFORE update4608 already commands−16.483N. Thus that difference is not created entirely by the very last update.

41006 initially catches the pole. At1s it is nearx=−1.173m,v=−.048m/s,theta=−.000811rad, then drifts and accelerates toward the same left rail, endingx=−2.400m,v=−1.343m/s,theta=+.159rad. At the recorded2.99s observation, the long policy commands−2.614N versus the short policy's+4.339N; its pre-final-update actor already commands−2.471N.

Same-observation queries are NOT simulated switches or proof that one substituted action rescues the case. The pre-final-update policies were not evaluated on these exact cases. Other replications retain aggregate outcomes only; no missing path was invented.

## Smaller target scatter does not qualify learning

At the final archived update, raw-advantage SD changes8.422->2.632 for41003 and11.507->7.904 for41006 (short->long). Incoming critic MSE against those sampled targets changes74.391->8.661 and132.612->65.164. These later actors/batches differ, so this is NOT a same-state variance intervention or true-value error oracle. It contradicts a blanket claim that longer support necessarily produces larger observed batch advantage scatter; it does not estimate independent sampling variance.

Across the16 later archived updates per arm, median raw-advantage SD is7.301short versus4.727long. Mean absolute final-value contributions to targets are typically smaller as intended. Their magnitude is not their estimation error.

## Trajectory offsets and finite-batch objectives

Split normalized advantages into the mean on each fitted same-episode trajectory segment and deviations around it, honoring all recorded boundaries. No such centering was applied to training. At incoming on-policy ratios, the actor gradient decomposes linearly into those contributions.

For long-support later updates,69.8% of raw-advantage squared deviations is the median between-segment mean contribution. The median segment-mean gradient norm is2.05times the within-segment gradient norm. Final ratios are3.54for41003 and2.17for41006; the improving41004 also has3.12. Similar dominance occurs in short controls. These offsets are not automatically wrong: real trajectory quality, state baselines and correlated sampling can all contribute. Automatically subtracting them would change the estimator and may introduce bias; that correction was NOT tested.

The outward-origin quarter contributes a median79.0% of squared normalized advantage magnitude in long-support later updates. This is not79% of Adam's displacement, and the prior failed source-normalization intervention is not revived by this descriptive accounting.

All48 selected updates and768 minibatch transactions are retained. Four end with a lower full-batch clipped surrogate: long41001/41002/41005 at4097, and short41002at4097. Both highlighted long final updates improve it:41003+0.008611,41006+0.004852, with mean batch KL.008702/.006068. For41006 the final gain consists of−.000621primary,+.000909ordinary supplemental,+.004565outward-origin contributions, each divided by1024. This is a recorded objective trade-off, NOT proof that the last update caused a whole-policy survival regression.

Gradient dot recorded displacement is a local calculation, not a counterfactual Adam replay, equilibrium shift or expected-return guarantee. All768 directional finite-difference comparisons are retained; largest absolute difference3.811e−7. No unexported optimizer moments or gradients were claimed to have been recovered.

## Internal timeouts remain direct bootstraps

The sampled-tail substitution extends the LAST collection cutoff. An earlier nonterminal timeout in a primary batch retains its original critic bootstrap. Four selected updates contain this: long41004/4608atrow31, short41005/4097atrow110, short41008/4097atrow494, long41008/4097atrow418 (zero-based).

Immediately before such a timeout, the coefficient remainsgamma=.995, notgamma^1024. In long41004's last update that is about.995*198.279=197.288 of value contribution, NOT197.288 of measured error. This is ordinary timeout bootstrapping, not credit crossing a true failure. It limits any claim that all bootstraps are bounded by the long-window coefficient.

None of the selected detailed updates for41003/41006 has an internal timeout; the improving41004 does. Therefore these four cases do not explain the two target failures. The original protocol extended the selected last cutoff, not every intermediate time limit.

## Verification and evidence

All17 original source ZIP digests and4494 payload hashes verified. The unchanged predecessor verifier checks build/configuration/history bindings,21504 evaluations,73728 update rows,96 checkpoint pairs and all48 detailed return/normalization/optimizer paths. All144 full traces/1170196 steps reconstruct actions,physics,rewards,endings and scores under unchanged tolerances. No additional CI or native measurement was executed.

81 offline tests pass:63 inherited and18 new derivative/clipping/partition/terminal/timeout controls. The first all-history analysis process hit a45-second tool limit; its partial log is retained and identical calculations completed per seed. This is computation on immutable records, not retried training, resampled outcomes or relaxed thresholds.

Conversation report: rustrobotics-support-regression-results.md. Self-contained evidence: rustrobotics-support-regression-evidence.zip,205138731bytes,SHA256990bedc3428367c9858d77544755df4b70bcb7bc51e8f3e138e4f3fdbe7d93b2. It contains all17 unchanged raw archives,analysis/tests,ten numerical reports,logs and replay instructions. A separate analysis-only package omits raw archives. Offline commands never run native code,network,simulator or learning.

## Next discriminating measurement

Instrument the41006 regression interval4129–4224 with complete persistent optimizer history, retaining an improving history as a control. This is more specific than increasing the horizon again or blaming the final recorded update for earlier damage. That missing-interval replay has NOT been launched in this step.

Production gamma.99/lambda.95,reward,normalization,learned weights,PR38,master,deployment and fallback behavior remain unchanged. No merge. The prior longer-support rejection stands; these diagnostics do not establish a qualified replacement controller.