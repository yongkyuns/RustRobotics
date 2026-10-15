# Observation-normalization crossover: learned weights retain the late regression

October14,2026, America/Toronto. Protocol issue35/comment6182687929 fixed before new outcomes. Continues b2dcaace. Diagnostic complete; no new training, production correction, PR merge or deployment.

## Decision

Changing the saved normalization statistics alone does NOT reproduce the large late recovery losses. Final learned weights retain the loss under either statistics set. Restoring halfway statistics at inference makes the final policy worse in the two most regressed seeds. The earlier observation-normalization candidate remains rejected under its original development rule.

This separates direct effects of the saved preprocessing map from final actor parameters in a fixed factorial. It does NOT establish that evolving statistics had no indirect effect on training, that weights/statistics did not co-adapt, or that freezing statistics throughout learning is a tested remedy. Deliberately crossed pairs are diagnostic counterfactuals, not valid original training checkpoints or deployment candidates.

## Fixed2x2 and complete difficult-start result

All four exposed normalized-training seeds201–204, checkpoints524288 and1048576. W0/W1 = halfway/final complete actor parameters including learned log-sigma; N0/N1 = corresponding frozen mean/variance/epsilon/clip. Evaluate all four pairs. Each controls its entire episode from reset, no time-switched hybrid. Critic outputs do not enter control or return.

Original64 ordinary+64 difficult cases per seed, both deterministic/stochastic modes. Same original native reset keys/physical law, noisy observations/disturbed actuation/clipped Gaussian force,64 lanes including inactive lanes,1500 ticks, first physical failure, raw/gamma.999 stopped returns and signed minimum cart clearance. No evaluation statistic update, fitting, teacher, reward shaping, selected data or optimizer step. All arms/seeds/modes fixed before outcomes.

Each cell below is completions/64:

|Seed|Mode|W0N0|W0N1|W1N0|W1N1|
|---:|---|---:|---:|---:|---:|
|201|deterministic|64|64|59|61|
|201|stochastic|64|64|58|60|
|202|deterministic|58|58|58|58|
|202|stochastic|58|58|58|58|
|203|deterministic|64|64|36|40|
|203|stochastic|64|64|35|37|
|204|deterministic|63|64|45|50|
|204|stochastic|63|64|43|50|

Seed203 retains64/64 with halfway weights and either statistics set; final weights give35/64 stochastic with halfway stats and37/64 with final stats. Seed204 final stochastic improves43→50/64 with final statistics, whereas halfway weights give63/64→64/64. Updating statistics alone loses NO difficult completion at either fixed weight set, across any seed/mode; all gains/losses are checked casewise, not inferred from equal totals.

Pooled difficult stochastic:249/256,250/256,194/256,205/256 in the column order above. These are case counts, not256 independent agents. The parameter-loss pattern also occurs under deterministic mean-action control, so it is not only increased exploration noise. This does not factor every role of the mean network/sigma during training.

## Ordinary results and clearance

Ordinary completion/64:

|Seed|Mode|W0N0|W0N1|W1N0|W1N1|
|---:|---|---:|---:|---:|---:|
|201|deterministic|63|64|63|63|
|201|stochastic|63|64|63|63|
|202|deterministic|64|64|64|64|
|202|stochastic|64|64|64|64|
|203|deterministic|64|64|63|63|
|203|stochastic|64|64|63|63|
|204|deterministic|63|63|62|63|
|204|stochastic|63|63|60|63|

Difficult stochastic completion WITH50mm clearance:

|Seed|W0N0|W0N1|W1N0|W1N1|
|---:|---:|---:|---:|---:|
|201|63|64|58|60|
|202|57|57|58|57|
|203|64|64|33|36|
|204|62|63|43|49|

50mm remains the original descriptive diagnostic category, not a changed failure boundary or a qualified hardware safety requirement. Failed cases stay in denominators. Per-case minimum signed clearance includes initial and every recorded physical post-state, excluding padding; closest tick and successful-case quantiles remain explicit. All4096 raw/discounted returns/endings and individual lost/gained keys are retained.

## Conditional effects, interaction and uncertainty

Report statistics effects at BOTH fixed weight sets, weight effects at BOTH fixed statistics sets, total change and J11−J10−J01+J00 interaction. Do not assume additive/separable effects.

Difficult stochastic RAW-return differences, pointwise99% Student intervals over the FOUR training-seed means:

|Contrast|Mean|99% interval|
|---|---:|---:|
|Statistics change at halfway weights|−.932|[−34.587,+32.723]|
|Statistics change at final weights|+58.559|[−132.274,+249.393]|
|Weight change at halfway statistics|−298.063|[−1138.625,+542.499]|
|Weight change at final statistics|−238.572|[−1000.454,+523.310]|
|Actual final versus halfway|−239.504|[−997.727,+518.720]|
|Interaction|+59.491|[−101.831,+220.813]|

All intervals above include0. The case-specific counterfactual losses are concrete, but four exposed development seeds do not establish a population-wide expected-effect sign. Modes are not eight independent agents. Per-seed fixed-case intervals are separate from four-seed pooled uncertainty; no new equivalence/acceptance threshold. All ordinary/difficult, deterministic/stochastic contrasts appear in RESULT.json.

Statistics effects are not universally positive in reward. For example seed202 has a small negative statistics-replacement return effect without changed completion. Small effects and all contrary cases remain. The conclusion is not that normalization is irrelevant: simply reverting to old statistics at inference is NOT a rescue of the observed final-weight losses. Freezing statistics throughout training would change the learning path and has NOT been tested here.

## Identity and verification

Current Intel Xeon8370C differs from original training8573C. This difference was disclosed before outcomes; all four cells are remeasured on the same host with no archived endpoint substitution.

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0. Runtime archive c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 payload hashes pass. Parent compact61b564896e00fcf794d5a5fd0ceb8dc8db31198a73316792636af24414e40311:267 hashes,26 prior tests,57 saved identity arrays/metadata and original failed progression decision reproduce. Original native58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0 and142 payloads verify; actual original PPO/buffer/policy/collector AND normalizer/RMS source bytes match. No current online implementation substituted.

ALL2048 diagonal endpoint case summaries match the original exactly: endings,lengths,raw/discounted return,clearance,closest tick and50mm category. Original compact omits main raw trajectories, so full raw-array identity is NOT claimed for those unavailable records. This is same-case reproduction, not another independent cohort or a cross-host guarantee.

Before main outcomes, unchanged original evaluator versus frozen adapter versus duplicate adapter:192 trajectories, all6 array fields byte-exact, frozen policy/statistics/RNG/live observation/empty optimizer checks pass. No learner step during model construction/evaluation.

28 NEW tests pass:10 factorial/statistical controls and18 genuine-record/receipt/normalizer/physical tests. Reject altered parameters/statistics, checkpoint/arm mismatch, one-representable-float reward, wrong action/noise/keys/initial observation/endings/padding/timing dtype/count/nonfinite values/collector receipt. Validate factorial interaction, four-seed units, paired gains/losses rather than totals, and actual nonzero inference changes when statistics change. Prior26 tests remain separate, not28+26 newly introduced.

Independent NumPy/SciPy reader verifies all4096 paths,64 complete groups, source/checkpoint/statistic/selection/policy/collector binding, conditional reset predicate, innovation mapping, independent float64 actor inference, force and raw float32 reward bits, physical endings,padding,sequential returns and margins. Max command discrepancy4.191864715e−7<unchanged3e−6. It does not independently reintegrate native dynamics/private disturbance RNG. No tolerance relaxed.

Fresh final FULL ZIP extraction verifies469 hashes, all28 new and26 retained tests, original saved57 control arrays plus three new evaluator controls, and THREE new numerical reports byte-for-byte. Fresh COMPACT extraction verifies407 hashes and all tests/controls but explicitly skips full4096-path reduction. Offline checking invokes no Torch/native/optimizer/pickle. Complete source patch applies in a fresh Git directory and all28 tests pass with the retained genuine fixture.

## Costs, artifacts and scope

4096 main trajectories/5774443 native transitions;192 controls/288000 transitions. Total4288 trajectories/**6062443 native transitions**. Zero training/optimizer transactions, new independent agents or reset-noise cases. All64 main groups completed once, all four process exits0; no interrupted/retried simulation, outcome-selected extension or unrecorded partial cost. New28 tests pass first invocation.

- Full ppo_obsnorm_crossover_evidence_20261014.zip:259960777bytes, SHA2563ef53836e5288731f82cff77b562ff74db550001b8c40c3e1306fb30c724696a.
- Compact ppo_obsnorm_crossover_report_code_20261014.zip:22735278bytes, SHA25687cf5a8a47a0a72c96ca6c2ba6affbe00e75e164088528303cc5342c9bbf170d.
- Complete audit patch38193bytes: ppo_obsnorm_crossover_audit_20261014.patch, SHA25604eef25abd3e2f6d8f553d24f58379f7137bf25549521f23d01bffc59517fd1c.
- Detailed report and detached verification/log receipt accompany them.

Collector committed6088b817 under obsnorm_crossover_20261014/code/crossover.py; it expects the artifact's parent/checkpoint/input layout, not absent repository measurements. Complete collector/reader28 tests/inherited metric core/README/replay are in the applicable patch and evidence; not falsely represented as production or routine-CI integration. Full REPLAY.py checks all payloads/tests and regenerates RESULT.json, CASES.json, ENDPOINT_COMPARISON.json byte-for-byte. Reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Compact omits most new raw groups. Original parent manifests and omission notices preserved; runtime installers/fonts excluded.

Master rechecked8514317e3b5cd57dc37bee2aa4c950af674751e1. No production/default/task/reward/noise/reset/physics/protected-file/deployment change, PR merge or issue closure. Observation-normalization candidate remains rejected; a supported correction still needs successful from-scratch retention and #33 held-out sustained-balancing qualification.
