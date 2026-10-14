# Transaction60: advantage normalization changes do not rescue the recovery loss

October14,2026. Fixed before-outcome protocol6179668160. Continues0629b36e. Conditional intervention COMPLETE; no production correction, new training cohort, PR merge or deployment.

## Decision

All tested replacements of minibatch advantage statistics retain the same lost recovery cases. Rollout-wide centering slightly improves mean reward on both exposed panels, but does not rescue either added failure. That positive result is retained. The study rejects these particular replacements as a local rescue, not every normalization-related effect throughout training or the possibility of a different correction.

This concerns the historical stock-SB3 reference transaction60/global Adam step130940 in update410, not a newly measured production Rust optimizer update. There is no fresh from-scratch or held-out qualification.

## Exact warm-state intervention

Warm ZIP e2558823223e70309fa3a884329e788e51d4c3bfb43ad8799c427f42c7a414ce and9 payload hashes pass. Runtime ZIP c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 payload hashes pass. AMD EPYC9V74, Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0; actual native/SB3 source gates pass.

The unchanged portable320-transaction replay first reproduces final policy/Adam/all RNG exactly. A second replay with observation hooks also matches exactly and captures the original row permutations and intermediate states. The isolated LOCAL transaction60 computation matches its original normalized advantages, raw/clipped gradients, policy and Adam arrays byte-for-byte. No initial weights, fresh Adam or approximate reconstruction are substituted; no million-interaction history needed rebuilding this time.

Fixed2×2, all from the SAME actual pre59 warm state and SAME64 sampled rows: mean from minibatch or full2048-row frozen rollout; sample standard deviation from minibatch or full rollout. ddof1,1e-8 epsilon. LOCAL, ROLLOUT_MEAN_ONLY, ROLLOUT_STD_ONLY, ROLLOUT_BOTH. Each applies exactly one Adam transaction. No critic target, action, old likelihood, value coefficient, gradient-clipping threshold, moment/counter/lr/beta/epsilon or physical reward change.

## All-case results

All1024 original ordinary and512 original difficult cases, previously exposed. OLD410 controls action ticks0–63; the actor below controls tick64 onward. No reset/reseeding at switch. Original full post-reset native/RNG clones and innovations,64 lanes including inactive lanes, noisy nonlinear plant/bounded commands,1500 ticks,gamma.999 stopped raw rewards,no critic bootstrap. Exactly7680 new trajectories; zero new independent reset/noise cases or trained agents.

|Actor after tick64|Ordinary complete /1024|Difficult complete /512|
|---|---:|---:|
|PRE59, no transaction|1001|400|
|LOCAL, original transaction|1000|399|
|ROLLOUT_MEAN_ONLY|1000|399|
|ROLLOUT_STD_ONLY|1000|399|
|ROLLOUT_BOTH|1000|399|

Every treatment loses the SAME one ordinary and one difficult case versus PRE59, gains none. No alternative gains or loses any completion versus LOCAL; this is checked casewise, not inferred from totals. All physical failures are cart-limit, not pole-angle.

Selected witness3309102759: PRE59 completes15s, return618.649299; LOCAL fails tick109, return20.821598. All alternatives fail tick109; mean-changing variants return20.821397. This exposed selected witness is not a population effect estimate.

### Small positive reward effect, not a rescue

Mean gamma.999 return changes versus LOCAL, descriptive pointwise paired99% Student intervals:

|Alternative|Ordinary|Difficult|
|---|---:|---:|
|ROLLOUT_MEAN_ONLY|+.0114514 [+.0110838,+.0118189]|+.0144037 [+.0133022,+.0155053]|
|ROLLOUT_STD_ONLY|+.000000095 [−.000000151,+.000000340]|+.000000118 [−.000000410,+.000000646]|
|ROLLOUT_BOTH|+.0114514 [+.0110838,+.0118189]|+.0144036 [+.0133021,+.0155052]|

For context, LOCAL versus PRE59 has ordinary−.587801 [−2.016268,+.840667] and difficult−1.197959 [−4.216744,+1.820825]. These overall signs remain unresolved despite concrete lost cases. No equivalence margin was declared; intervals spanning zero are not equivalence. These exposed fixed-policy intervals are not simultaneous/familywise or trained-agent-population/held-out guarantees.

## The intervention was not a no-op

Minibatch advantage mean .00040889165 versus rollout−.00000646380; sample stds .00243576989 versus .00223379173. Changing the mean changes8/64 advantage signs:27 positive/37 negative becomes35 positive/29 negative. The clipped actor-gradient cosine versus LOCAL is.989953, and the actor increment differs by4.1417% of LOCAL's increment norm. The failed rescue is not explained by unchanged labels or parameters.

Scale-only changes are largely removed by global norm clipping in this actor-dominated transaction. Clip factor .1173307→.1076015; actor increment differs by only.000233% of its original norm,13 actor entries differ at most7.45e−9. This is not a general claim that scale never matters with Adam/joint clipping.

All variants improve their own sampled actor loss, but those losses use different normalized labels and should not be compared directly across arms. Under the original LOCAL objective, all improve from−.02432819: LOCAL/scale-only→−.02606708, mean-changing→−.02605324. None establishes true action-credit correctness or a control guarantee.

## Verification and actual code

37 NEW genuine-data/mathematical tests plus18 retained trace tests pass:55 distinct tests. Corruptions cover normalization factors/statistics, samples, advantages, gradients/clipping, Adam state/clock, policy/loss, scheduled identity, keys/noise, reward bits including one representable float, endings/padding/scores and receipts. Twelve finite-difference checks cover hand-derived actor gradients across four variants.

The independent reader checks all320 reference transactions' losses/clipping/Adam equations and hand-derives each intervention gradient. Maximum intervention raw-gradient discrepancy4.735e−7; parameter-update discrepancy5.461e−8. No tolerance relaxation. The exact executed reader is committed in normalization410_20261014/code/verify_optimizer.py at726cbab5; it expects the artifact input layout, not missing in-repository measurements.

Five original native controls reproduce OLD/NEW group00 in both panels plus the retained temporal fixture byte-for-byte:320 control trajectories,436565 transitions. Independent physical/inference validation covers all7680 new paths, complete cases/arms, original reset bindings, exact64-step prefixes, innovations, physical endings/raw reward bits/force mapping/sequential scores. Max independent normalized-command discrepancy2.71306467e−7<3e−6.

Fresh FULL extraction verifies1651 payload hashes,55 tests and all FIVE numerical reports byte-for-byte. A combined extraction/replay tool invocation hit its frontend limit after tests; no simulator/optimizer ran. The unchanged behavior reader and other numerical readers then completed separately on that same fresh extraction. Fresh COMPACT replay verifies1155 hashes,55 tests and3 reports; it explicitly skips full-panel behavior reduction. Independent readers do not reintegrate native dynamics/private noise.

Complete collector, guarded observer-loader seam, intervention code, all readers/tests, raw data and source hashes are in the evidence packages. They are not falsely described as integrated into production or routine CI.

## Costs and interruptions

Known completed native work11000341 transitions =10563776 primary +436565 controls. Optimizer work644 transactions =320 original portable +320 instrumented identical +4 fixed alternatives. No new training-environment interactions or independent case cohort.

A streaming request was rejected before execution. A later frontend timeout completed broad06/07 and the first four arms of broad08 but interrupted its ROLLOUT_BOTH arm without archive/receipt. All four completed receipts were verified and not rerun; only the missing fixed-key arm was recovered. Its94589 completed transitions are included. The interrupted attempt's unknown cost is bounded by96000 transitions, not claimed zero. Logs/guarded recovery script retained; no outcome-selected retry. Final offline timeout/reduction logs are bound by the detached receipt.

## Artifacts and status

- Full ppo_normalization410_evidence_20261014.zip:793007343bytes, SHA2568dad924914121913ad68603e2714df40093d2bac0b5a7457557cc795aad79f67.
- Compact ppo_normalization410_report_code_20261014.zip:134206832bytes, SHA256df4570c66863ca64faecb4961e6a6ef77954be15de3d508c7b130be27fc986f2.
- Detailed ppo_normalization410_report_20261014.md and detached ppo_normalization410_verification_20261014.json accompany them.

REPLAY.py checks payloads/tests and regenerates numerical outputs without Torch/native/optimizer/pickle. Recorded reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Compact includes the full optimizer trace and genuine fixtures, not the full behavior panel. Original manifests/omission notices and native/source provenance are retained. Third-party runtime installers/fonts are excluded. README gives separate strict native regeneration; mismatches must not be waived.

No production/default/task/reward/reset/noise/physics/protected-file/held-out-training-seed/master/deployment change or PR merge. #33/#35 remain open. These normalization alternatives are not adopted; a correction still needs from-scratch performance/retention comparison and sustained-balancing qualification.
