# Gamma0.995 with longer sampled return support

Registered before new outcomes in issue35 comment5753874883. Executable source **0a4445f6c5ab83650133a696faaeb764fd5b6c26**, workflow **35548063439**, branch **audit/ppo-return-support-20260920**.

**Native preflight PASSED on its first attempt; learning jobs have started.** At this progress check thirteen of the sixteen training jobs are executing and three are queued. There is no completed, independently verified learning comparison or controller-quality result yet. Documentation-only commits do not start another measurement.

## Fixed comparison

Hold training gamma0.995 fixed from random actor/critic initialization and compare only sampled support lengths512 and1024. There are sixteen runs, paired seeds41001–41008, all already-exposed development histories. Every run performs4096 ordinary-reset recovery-union updates followed by512 half-outward updates with continuous Adam, environment, unfinished episode and random state. No learned weights are imported.

| Quantity | support512 control | support1024 candidate |
|---|---:|---:|
| Primary fitting samples/update |512|512|
| Supplemental streams |8|8|
| Fitting samples per supplemental stream |64|64|
| Collected steps per supplemental stream |576|1088|
| Maximum steps per primary-cutoff future |512|1024|
| Primary-cutoff futures |4|4|
| Total fitting samples/update |1024|1024|
| Adam steps per network/update |16|16|

Extra supplemental observations after the selected64 samples provide future rewards but do not enter actor/critic fitting. True terminal masks prevent credit from crossing failure/reset boundaries. Only the test-scoped tail and supplemental lengths change. Plant, noise, reward, global normalization, lambda1, architecture, action distribution, initial-state mixture, Adam settings, minibatch256 and four epochs remain fixed. No teacher, imitation, symmetry wrapper, extra critic passes or safety fallback.

The unfinished-cutoff value coefficient falls from0.995^512≈0.07681 to0.995^1024≈0.00590. This tests whether more sampled support helps; it does not establish that bootstrap error caused the earlier recovery regression. Longer support also changes realized sampling variance and the later bootstrap state.

## Controls and outcomes required

The512-support control must reproduce BOTH actor and critic from the preceding gamma995 archives at checkpoints0/1024/4096/4128/4224/4608. All4608 update records after their arm label and final cost records must also match. Archives are bound to immutable published digests. Historical values are comparison inputs, never warm starts. Any mismatch stops and is retained. Both new arms must have identical initial networks and first fitted physical data; subsequent training paths may differ.

All checkpoints are retained; only4608 determines advancement. Detailed updates1/4097/4608 retain incoming weights, fitting rows, full sampled support, targets and all16 optimizer transactions. Evaluation uses new fixed offset0x10000000 and common score-gamma0.99 for both arms. Each checkpoint has64 short nominal-mean, nominal-stochastic and outward trials per seed/arm. Final evaluation adds64 five-minute nominal-mean,64 five-minute nominal-stochastic and64 sixty-second outward trials. Failure ends a trial immediately. Rep0 full trajectories at4096/final; other evaluations retain aggregate outcomes. Nothing from evaluation enters fitting.

Development advancement requires a positive lower bound for outward-completion gain using eight paired-history two-sided99% Student intervals adjusted overthree long-completion comparisons, both candidate nominal pooled counts>=control, and every candidate nominal history>=61/64. The separate absolute screen retains507/512 pooled per long panel,61/64 per history,three-panel-family one-sided99% history lower bound>=95%,and99% final-ten-second centring among survivors. Missing histories cannot pass. No best-checkpoint selection, favorable retries, threshold changes or excluded weak seeds. This exposed cohort is not fresh or hardware qualification.

Per arm/history:2359296 main interactions,21233664 versus40108032 supplemental interactions,and up to9437184 versus18874368 cutoff futures. Each network receives73728 Adam steps and18874368 fitting-sample visits in both arms. Evaluation and preflight are additional. This is equal fitting/optimizer work, not equal total simulator work or a sample-efficiency result.

## Verification completed

The native preflight passed **108 unit/audit tests**, **7 ordinary balancing controls**, **9 ordinary learning controls**, strict Clippy, formatting and protected-source restoration. The nine heavy audit endpoints were ignored during the regular unit pass; the new learning endpoint is explicitly invoked by the separate training jobs. Two historical heavy integration endpoints and the production browser/platform matrix were not run.

Seven new native tests validate support scope/thread isolation and unwind restoration,unchanged512 dispatch plus subsequent optimizer-history parity,first main/selected supplemental physical-data identity,tail-prefix identity,terminal/cutoff bootstrap handling,and reproducible1024-support updates with unchanged fitting work. These tests passed; they are not sustained-control results.

Build artifact10617012511, SHA256 **5fa2e313fdc55d9779ff91c0b665cf8e47a17af7ffd8e71f3fb42223651d4e74**, contains the actually compiled source/runner/logs. Its published ZIP digest and all110 payload hashes were independently verified. No local recompilation or alternative simulator was substituted.

The prepared offline verifier passes **63 synthetic/property tests**. Empty-input control requires all17 result/build archives and emits INCOMPLETE/exit2 without scores. Generalized512-support numerical reconstruction also matches the inherited verifier exactly on archived gamma995 histories41001/41004. This is compatibility checking on existing data, not a new learning result. Source-length transformations passed exact reversibility checks on the actual archived collectors.

The verifier is ready to bind all new artifacts and checkpoints, reconstruct both support lengths' GAE/tails/returns/normalization and selected optimizer outputs, and check exported evaluation dynamics/actions/rewards. It has not yet verified new training outcomes. Unexported gradients,Adam states and remaining trajectories are not independently regenerated. Offline replay executes no native code, simulator, training or network.

**No production gamma0.99/lambda0.95, reward, normalization, learned policy, PR38, master or deployment changed.** No merge or controller adoption. The study has cleared native preflight and begun training; the question of improved recovery is still unanswered.