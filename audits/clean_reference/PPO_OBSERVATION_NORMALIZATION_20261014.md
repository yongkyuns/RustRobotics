# Vanilla PPO observation normalization: faster early learning, retention screen failed

October 14, 2026 (America/Toronto). Issue #35 protocol6181775559, registered before outcomes. Continues the retained recovery-margin audit6181696111. Eight complete new from-scratch runs; no production correction or deployment.

## Decision

**Observation normalization does not pass the declared development progression rule.** It gives a substantial early-learning gain on these four exposed development seeds, but late recovery losses remain and two seeds finish worse than the reference. No threshold is relaxed and no intermediate checkpoint is selected as a deployment policy.

These are NEW paired trainings on the same Intel host, not an approximate replay of historical AMD optimizer trajectories. Each arm starts with equal fresh actor/critic tensors and empty Adam. Every evaluated actor controls its complete episode from reset; there is no late-switch hybrid, warm trained actor, teacher, rollback or safety override.

## All fixed checkpoints

Difficult stochastic panel,64 original-native-reset cases per training seed,256 total. Completion requires survival to1500 ticks. Completion+50mm additionally requires minimum signed cart clearance at least50mm. This is a declared diagnostic category, not a changed physical boundary or qualified hardware safety requirement.

|Interactions per run|Reference complete|Normalized complete|Reference complete+50mm|Normalized complete+50mm|
|---:|---:|---:|---:|---:|
|0|0/256|0/256|0/256|0/256|
|65536|81/256|227/256|77/256|223/256|
|262144|243/256|235/256|241/256|233/256|
|524288|195/256|249/256|194/256|246/256|
|1048576|194/256|205/256|194/256|202/256|

The halfway normalized result249/256 falls to205/256 at the fixed endpoint. Its seed203 declines64/64 to37/64 and seed204 declines63/64 to50/64 over that interval. Reference training also has late losses and recoveries; for example seed201 falls64/64 to23/64 while seed202 recovers13/64 to48/64. All checkpoints are retained, not a monotonic-learning claim or favorable-checkpoint selection.

## Final results: every seed and evaluation mode

All completion denominators are64. Arrows denote reference to normalized, not changes over training time.

|Seed|Mode|Ordinary completions|Difficult completions|Difficult complete+50mm|Ordinary raw-return delta|
|---:|---|---:|---:|---:|---:|
|201|deterministic|56 to63|25 to61|25 to61|+134.642217|
|201|stochastic|54 to63|23 to60|23 to60|+176.676416|
|202|deterministic|63 to64|49 to58|49 to57|+22.216523|
|202|stochastic|63 to64|48 to58|48 to57|+19.217600|
|203|deterministic|64 to63|64 to40|64 to36|−34.435888|
|203|stochastic|64 to63|64 to37|64 to36|−41.176060|
|204|deterministic|63 to63|59 to50|59 to50|−9.839726|
|204|stochastic|63 to63|59 to50|59 to49|−9.956768|

Pooled final totals improve modestly, but conceal substantial seed variation. Four training seeds—not thousands of correlated evaluation trajectories—are the uncertainty unit. Pointwise99% Student intervals over four seed differences are retained in RESULT.json, separately by mode. The final mean-effect intervals span zero; no population-average benefit is certified.

### Frozen progression rule: FAIL

- No final difficult-completion reduction in either mode for any seed: **fails** for203 and204.
- No final ordinary mean-raw-return reduction in either mode for any seed: **fails** for203 and204.
- Mean difficult completion-with50mm gain across eight seed/mode pairs at least5percentage points: measured **+2.9296875percentage points**, so **fails**.

A pass would only have made the recipe promising for further testing. This failure does not prove observation normalization is useless; the early benefit is real on this development panel. It means normalization alone is not established as a reliable retention correction. This does not diagnose whether the remaining late loss comes from changing statistics, learned weights, coverage or advantage estimation.

## Single change and unchanged contracts

Candidate: stock SB3 VecNormalize(norm_obs=True,norm_reward=False,clip_obs=10,epsilon=1e-8). Only training observations update running statistics. Evaluation uses the saved mean/variance/epsilon with no statistic update. Both actor and critic receive normalized noisy observations; physical state is used by the environment and evaluator only, never extra actor inputs.

Both arms retain the existing scaled-2048 SB3 reference recipe: original nonlinear native plant/reward/noise/reset and physical failures;256-step external training timeout; inherited fixed reward-unit scale1/1000 and divided initial critic output head; gamma.999,lambda1,2048 rollout rows,batch64,ten epochs,lr.0003,Adam epsilon.00001,clip.2,global gradient norm.5,entropy coefficient0,two64-unit tanh layers and learned Gaussian action scale. Original PPO.train, rollout collection and GAE methods execute unchanged. No reward normalization, shaping, curriculum, supplied controller, additional data or optimizer passes.

This is the SB3 reference branch of the RustRobotics investigation, not a claim that production Rust defaults already use this recipe or that a Rust port has passed platform/browser checks.

## Fixed sampling, inference and source provenance

Four exposed development training seeds201–204. Each arm runs exactly1048576 interactions with checkpoints0,65536,262144,524288,1048576. Four concurrent single-threaded process slots; each seed's two arms execute sequentially, raw first for odd seeds and normalized first for even seeds. No throughput comparison is claimed.

All eight source panels froze before main training. Per seed:64 ordinary and64 difficult native resets, the latter first64 ascending keys meeting abs(theta)>=.15,abs(omega)>=.25,theta*omega>0 (max65536 considered). Base key4200142600+seed*1000000+panel_index*100000. Each checkpoint evaluates both deterministic and stochastic policies on all cases. Stochastic innovations use default_rng(key XOR0x5E1410AA).standard_normal(1500).astype(float32). Both arms/checkpoints share the same reset and standardized innovation keys; each policy keeps its own learned sigma. Inference uses64 lanes including inactive lanes.

Each evaluation stops at its first original physical ending or1500ticks, no critic bootstrap. Signed clearance includes initial and every physical post-step state through the terminal transition, excluding padding. The actual float32 cart boundary is promoted without rounding. Failed cases remain in completion denominators; successful-case margin quantiles are explicitly conditional.

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0, Intel Xeon8573C. Original native ZIP SHA25658af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0 and all142 payload hashes pass. Actual original PPO/buffer/policy/collector modules match archived bytes. Runtime ZIPc7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 runtime payloads pass. Exact VecNormalize and RunningMeanStd source and their MIT license are retained, with hashes matching all eight runtime receipts. No historical CPU mismatch is accepted as an exact replay.

## Verification

Ten live preflight checks pass before main training: same-host exact raw-versus-monitored policy/Adam/RNG/live-state replay, disabled-normalizer identity, independent running-statistics formula, normalized pre-reset timeout observation, untouched raw rewards, enabled learner smoke and frozen-evaluation isolation. A separate retained control repetition exports57 numeric arrays and episode metadata; unobserved, observed and disabled-wrapper histories match exactly. This repeated control is not another independent trial.

**26 read-only metric/decision tests pass.** Genuine-fixture corruptions cover one-float32 reward changes, actions, policy/statistics, noise/keys, endings, padding, missing seeds, negative progression effects and clearance logic. Physical failure is not redefined by the50mm readout.

The full independent NumPy/SciPy reader validates **10240 evaluation trajectories /11102201 recorded evaluation transitions**, every checkpoint/model/statistics/selection binding, paired initial tensors, all fixed interaction budgets and named Adam clocks, training-only RMS counts, completed training episode ledgers, native reward bits, observation bounds, independent actor inference, first endings, sequential raw/discounted returns and signed clearance. Maximum independent command discrepancy3.782532171925368e-7 is below the fixed3e-6 tolerance. No tolerance changed after execution.

Fresh final FULL extraction verifies **426 payload hashes**, all26 tests,57 control arrays/metadata and regenerates **RESULT.json, CHECKPOINTS.json and CASES.json byte-for-byte**. Fresh COMPACT extraction verifies267 hashes,26 tests,the57 control arrays and the original decision; it explicitly skips main raw-path reduction. Offline replay performs no Torch/native/optimizer/pickle execution and does not independently reintegrate dynamics/private noise or reconstruct unretained training trajectories.

All40 checkpoints include policy.npz, named adam.npz and NORMALIZER.json. Normalization statistics are essential inference state. These are not full native-simulator-resume checkpoints. The final unfinished training episode is not counted as completed; its remaining interaction count is explicit from the final ledger gap.

## Costs and retained setup issues

Main training8388608 native interactions and1310720 Adam transactions. Main evaluation11102201 native transitions. Preflight16384 training+4079 evaluation; retained control repetition12288 training. **Total19523560 native transitions and1315200 optimizer transactions.** Eight new paired development training runs; no new held-out qualification.

All eight main runs completed once, each with create-only outputs. No outcome-driven retry, seed replacement or adaptive budget. Initial streaming, log-permission and missing-source launch failures happened before any main training and are retained. No main numerical run was repeated. The source patch applies in a fresh Git directory and passes all26 tests with the genuine fixture; git apply emits one nonblocking blank-line-at-EOF warning in retained source.

## Artifacts, code and scope

- Full ppo_obsnorm410_evidence_20261014.zip:467052337bytes; SHA256165a7ed043e6ea52db1ddaeb7295d32ee5a71a47b312d198502f46732eccd175.
- Compact ppo_obsnorm410_report_code_20261014.zip:7876744bytes; SHA25661b564896e00fcf794d5a5fd0ceb8dc8db31198a73316792636af24414e40311.
- Complete source patch ppo_obsnorm410_audit_20261014.patch:87219bytes; SHA256a726788161cb13dc7f506c106edd2159382ac9070dca9ad7566f83ada9687e0d. Patch base93fae69289820ce0ec5796d71b44e23ac97a3cc6; later separately published identical metrics.py must not be added twice.
- Detailed report and detached verification receipt/logs accompany them.

Protocol committed1cf84ac2, progression code93fae692, exact executed independent metric reader86686340. Complete driver, inherited pinned loader/recipe, controls, all readers/tests and reproduction are in the evidence and tested patch; not all are claimed integrated into routine CI or production. Native/source subsets retain provenance; duplicated extracted native libraries are omitted in favor of the original ZIP. Runtime installers and fonts are excluded.

Offline: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python REPLAY.py. README documents separate strict-runtime native regeneration. Reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. The compact package retains all model/statistic checkpoints and every case summary, but not main raw paths; it cannot replace full path verification.

No production/default/reward/noise/reset/physics/protected-file/held-out-training-seed/master/deployment change, production PR or merge. #33/#35 remain open. Faster initial learning is not accepted as stable long-term retention or sustained-balancing qualification.
