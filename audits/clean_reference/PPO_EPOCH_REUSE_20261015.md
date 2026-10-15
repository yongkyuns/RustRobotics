# PPO rollout reuse: four passes preserve the measured tail, but the overall screen fails

October15,2026. Registered protocol6194736723, resumption/current-host disclosure6199979919. Completes the existing design at9537deb6. Twelve main training executions finished once at their fixed budgets. No production correction, PR merge, deployment or issue closure.

## Decision

The four-pass candidate passes the declared continued-learning retention conditions: no previously completed difficult case is lost during the final262144 interactions, in either evaluation mode or any of the four seeds. No difficult completion-with50mm case is lost either. Its difficult stochastic count rises234/256 to239/256, while ten-pass decay falls248/256 to246/256. All4546 actor parameters change in each four-pass learner during this tail; the improvement is not obtained by freezing weights.

However, four-pass PPO still fails the original overall development screen. Ordinary mean raw-return non-reduction fails at both required checkpoints. Final difficult recovery is also lower than ten-pass decay:239/256 versus246/256 stochastic,241/256 versus249/256 deterministic. The direct four-versus-ten comparison helps seeds201/202 and hurts203/204. Fewer passes are not adopted as an unqualified improvement or a reliable production fix.

This is a new execution of the historical SB3 reference recipe on the Rust native environment, not a production Rust PPO integration. These are four exposed development training seeds, not a held-out cohort or twelve independent random agents. The two ten-pass reference histories repeat the earlier same-host numerical results exactly; they are not additional independent evidence.

## All fixed difficult-stochastic checkpoints

Each denominator256 is64 evaluation cases per training seed. Completion+50mm means survival with minimum signed cart clearance at least50mm, a descriptive category rather than a changed physical boundary or hardware safety specification.

|Interactions per run|CONSTANT10|DECAY10|DECAY4|DECAY4 complete+50mm|
|---:|---:|---:|---:|---:|
|0|0|0|0|0|
|65536|204|192|199|193|
|262144|234|250|239|236|
|524288|238|221|240|238|
|786432|211|243|213|210|
|1048576|218|248|234|231|
|1310720|177|246|239|237|

The four-pass240→213 decline before786432 is retained. Passing one finite continued-learning tail is not monotonic learning at every checkpoint or indefinite retention. No intermediate checkpoint is selected as the final policy.

## Every final seed and mode

All entries are completions/64. Each saved actor controls the whole evaluation episode from reset; there is no policy-switching hybrid.

|Seed|Mode|CONSTANT10 difficult|DECAY10 difficult|DECAY4 difficult|DECAY4 difficult+50mm|DECAY4 ordinary|
|---:|---|---:|---:|---:|---:|---:|
|201|deterministic|57|64|64|64|63|
|201|stochastic|55|63|64|64|63|
|202|deterministic|45|59|62|62|64|
|202|stochastic|44|58|62|62|64|
|203|deterministic|50|64|59|58|64|
|203|stochastic|50|64|59|58|64|
|204|deterministic|28|62|56|55|63|
|204|stochastic|28|61|54|53|63|

## Original progression conditions, unchanged

At BOTH1048576 and1310720, each decay arm is compared with the newly executed CONSTANT10: no per-seed/mode difficult-count reduction; no per-seed/mode ordinary raw-return reduction; mean difficult completion-with50mm gain at least5percentage points. Independently require no reduction in the candidate's own difficult completion or50mm counts during the positive-floor tail. The direct DECAY4−DECAY10 contrast is separately reported and does not replace the registered comparator.

DECAY4 passes difficult-count and margin-gain conditions at both endpoints (margin gains+7.421875pp and+24.4140625pp), and both tail conditions. It fails ordinary-return non-reduction at both endpoints. At1048576 its ordinary stochastic mean return is lower than CONSTANT10 in all four seeds, with deterministic reduction in204 too. At1310720, seed203 remains lower by5.283609 deterministic and26.245306 stochastic raw-return units despite both arms completing64/64 ordinary cases. These deficits are not waived.

DECAY10 reproduces its prior failed screen: all comparison conditions pass at1048576; ordinary-return non-reduction fails for203 at1310720, and204 loses difficult completion and50mm counts during the tail. This result is not relabeled as a new independent confirmation.

### Four-pass positive-floor tail

Exactly16384 additional Adam transactions per run at lr.00003; all4546 actor coordinates change in every seed. Casewise gains/losses are checked, not inferred from totals.

|Seed|Mode|Difficult before→after|Gains|Losses|50mm before→after|
|---:|---|---:|---:|---:|---:|
|201|deterministic|64→64|0|0|64→64|
|201|stochastic|64→64|0|0|64→64|
|202|deterministic|62→62|0|0|61→62|
|202|stochastic|61→62|1|0|60→62|
|203|deterministic|58→59|1|0|58→58|
|203|stochastic|58→59|1|0|58→58|
|204|deterministic|54→56|2|0|52→55|
|204|stochastic|51→54|3|0|49→53|

## Isolated effect and uncertainty

DECAY4−DECAY10 at1310720, pointwise two-sided99% Student intervals over FOUR paired training-seed differences, separately by mode. Modes/cases/checkpoints are not independent agents and these intervals are not simultaneous or held-out guarantees.

|Panel/mode|Mean completion difference, per64 [99% interval]|Mean raw-return difference [99% interval]|
|---|---:|---:|
|Ordinary/deterministic|0 [0,0]|−2.210 [−7.159,+2.739]|
|Ordinary/stochastic|0 [0,0]|−28.980 [−36.870,−21.091]|
|Difficult/deterministic|−2.000 [−14.390,+10.390]|−45.759 [−326.705,+235.187]|
|Difficult/stochastic|−1.750 [−16.713,+13.213]|−66.620 [−390.335,+257.094]|

Identical ordinary completion differences produce a degenerate empirical interval, not proof of population equivalence. Difficult-effect intervals remain wide and cross0. Ordinary stochastic return is lower in every seed with a negative four-seed interval on this development sample. Reduced reuse changes both actor and critic fitting; this experiment does not uniquely diagnose actor overfitting, critic calibration, normalization dynamics or coverage.

## Fixed implementation

CONSTANT10 uses lr.0003 and ten epochs. DECAY10/DECAY4 share lr(t)=.0003*(1−.9*min(t/1048576,1)), reaching the nonzero.00003 floor; only n_epochs differs10 versus4. The schedule uses absolute completed interactions and never restarts at checkpoint learn calls. All actual Adam-step rates/clocks are checked.

All arms retain observation-only VecNormalize (training-only statistics, frozen evaluation, clip10/epsilon1e-8), the original noisy nonlinear native plant, physical reward/reset/failure laws,256-step external timeout, inherited reward-unit scale1/1000 and divided initial critic head, gamma.999,lambda1,2048-row rollouts,batch64,Adam epsilon1e-5,clip.2,gradient norm.5,entropy0,two64-unit tanh layers and stock PPO.train/collection/GAE. No teacher, curriculum, reward shaping, extra data, optimizer reset, rollback or selected actor.

Each arm runs1310720 interactions/640 rollouts. Ten-pass arms perform204800 optimizer transactions; four-pass performs81920,60% fewer. This is not equal optimizer work or a measured60% wall-clock speedup. Four fixed single-threaded worker slots follow the original per-seed three-arm order. All12 initial policies/critics are fresh, equal within each seed, with empty Adam.

Evaluation retains all original64 ordinary+64 difficult native resets per seed, both modes, keys/standardized innovations,64 inference lanes including inactive lanes,1500 ticks and original first physical termination. Raw/gamma.999 returns have no bootstrap. Signed clearance includes initial and every physical post-step state but no terminal padding. Failed cases remain in denominators. Normalizer statistics and model checkpoints must stay paired.

## Provenance and executed verification

The initial interrupted registration named Intel8370C; no completed epoch-reuse results were recovered. Current AMD EPYC9V74 was disclosed before main outcomes in6199979919. All three arms are newly executed on this same current host; no archived comparator substituted. All56 ten-pass checkpoints reproduce prior policy arrays, named Adam arrays and saved normalization statistics exactly. These numerical repeats are not additional independent training seeds.

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0; original native/PPO/collector/buffer/policy/normalizer/RMS source gates pass. Runtimec7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 payloads, observation-normalization parent61b564896e00fcf794d5a5fd0ceb8dc8db31198a73316792636af24414e40311 and267 payloads, LR parent5cfa6ec278eaf8f2812b2c84952309b18b14190fdd6d0018147b725919b2f5b6 and607 payloads were verified. Thirty prior tests and saved control arrays replayed before main setup.

Nine fresh live gates pass, including five exact56-array/metadata pairs: unmodified versus observed CONSTANT10/DECAY10/DECAY4 and grouped versus checkpoint-split DECAY10/DECAY4. Equal initialization, empty Adam, frozen evaluation and independent reward/action/clearance controls pass.

**All34 new scientific design/decision/genuine-record/consumer-integrity tests pass.** Independent reduction checks21504 evaluation trajectories/26354108 physical transitions,336 groups,84 model/Adam/normalizer checkpoints,640 rate records per run, complete training ledgers, RMS counts, source/model/selection identities, actions, force/raw float32 reward bits, first endings,padding,margins and all paired case outcomes. Maximum independent normalized-command discrepancy4.0009887003478184e-7 < unchanged3e-6. No tolerance relaxed.

FRESH final FULL extraction verifies1884 payload hashes, all34 tests, five56-array controls and regenerates all FIVE numerical reports byte-for-byte. FRESH COMPACT extraction verifies1549 hashes, tests/controls/checkpoints/rates and the decision, explicitly skipping its omitted336 main raw groups. Hydrating it with the four seed-specific raw volumes reproduces the full1884-entry manifest, then independently reruns all five reports byte-for-byte. Missing/altered manifest payloads are rejected. Offline replay runs no Torch/native/optimizer/pickle; it does not independently reintegrate physical dynamics/private disturbances or reconstruct unretained training paths.

## Costs and retained setup issues

Main training15728640 native interactions/1966080 Adam transactions. Main evaluation26354108 transitions. Completed controls add86016 training+3474 evaluation and9984 Adam transactions. **Known total42172238 native transitions and1976064 optimizer transactions**, plus an interrupted control-only remainder bounded by8192 training interactions/1280 optimizer steps. No efficiency claim.

The first synchronous control invocation timed out after three completed pairs; completed arrays were reused and checked, not rerun. An import-path collision in the control-completion supervisor failed before numerical execution, then was repaired. All21 main source hashes were frozen before launch and remain unchanged. All12 main exits are0, no main retry/interruption/outcome-selected extension, and no learner remains running. An initial compact-packaging omission filter included one required preflight fixture; review caught337 rather than336 exclusions, and the selector was corrected before final replay. It changes no experiment, outcome or tolerance. Initial source/logs/receipts are retained.

## Source and recoverable artifacts

Registered design/experiment source9537deb6 matches blobsaa6f0f104ab4e75f747d3cca27a8525292899615 and57c797eba899904517202d4790190483e298867f. Decision, tests and exact schedule published in2b2bf32f,c13718b5,cf0481b2. Complete36-module source patch applies againstcf0481b2 and all34 tests pass in a fresh test directory with the genuine evidence fixtures. Numeric fixtures are not silently embedded in the text patch. The complete implementation and original input layout are in the evidence, not claimed production/routine-CI integration.

- Full ppo_epoch_reuse_evidence_20261015.zip:1127945770bytes; SHA2565378a475cd4f904f5601327a4dc309c4f9f5dadcc5a558a115ced79211d1e230.
- Compact ppo_epoch_reuse_report_code_20261015.zip:37996794bytes; SHA256ebe5966e10ac6fdf1694734886ff554a93fb2f5b5714492ff158a49af4de3eaf.
- Complete audit patch207911bytes; SHA256adae8032ed82cc20d01f4943d1f725e3674b169c619b2425c27f4013a0857645.
- Four raw-data volumes (84 groups each): seed201273518807bytes/e85ce3ed383a839e92e6a64f2bead27b11ad448d3724bdbede70f7d72a9de37c; seed202272474881bytes/e5ff15652567a3e15b345bba47b7e87b80ec88d27e571e9a99bc4a8ec268f0d7; seed203274255522bytes/b73ad82b7063a518952986c2a817846dd1183589dc75fdb8467b6372fc856264; seed204271000525bytes/9255c4e94a804bcf77d80fc128e70211872e01586fb62f1a699bfcc1c5ea6025.

The compact package plus all four raw volumes is an independently verified alternative to the1.13GB full ZIP, not another sample. README/hydrate_raw.py enforce exact input/manifest identity. Full REPLAY.py regenerates RESULT.json,CHECKPOINTS.json,CASES.json,RETENTION.json,PAIRED_CASES.json. Recorded reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Checkpoints are not complete native-simulator resumes. Runtime installers are not bundled; source/version gates remain mandatory for native regeneration. Parent manifests/subset scope, setup failures and detached verification logs are preserved.

Master remains8514317e3b5cd57dc37bee2aa4c950af674751e1. No production trainer/default, physical task/reward/noise/reset, protected source, held-out training seed, deployment or master change. No PR merge or issue closure. Improved measured tail retention is retained as evidence, but the ordinary-return deficits and lower final recovery prevent promoting this candidate as the reliability fix.
