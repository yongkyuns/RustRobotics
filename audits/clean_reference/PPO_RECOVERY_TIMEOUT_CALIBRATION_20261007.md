# PPO recovery: calibration at actual training timeouts

**Completed development diagnostic. No corrective PPO training, production/default change, deployment or merge.** Protocol: issue35/comment6049976573, fixed before new replay and continuation outcomes.

## Main result

The large bootstrap share was **not evidence of fictitious recovery success in this fixed sample**. At32 actual difficult training timeouts, the contemporaneous critic predicted mean future return **947.250**. Independent continuation with the same contemporaneous policies measured **953.698** at8192 additional ticks. **All1024/1024 stochastic continuations survived.**

Across the32 cutoff means: signed prediction error **-6.448**, MAE **10.017**, RMSE **14.501** in physical reward units. The largest underprediction is48.298; largest overprediction22.456. These errors are not zero, but this is not an approximately950-point critic estimate attached to continuations which immediately fail.

This contradicts a blanket bad-timeout-bootstrap account, not the earlier measured value errors at failing early-recovery states. It does not establish unbiased action advantages, universal stability or irrelevance of the training horizon.

## Fixed panel and exact historical reconstruction

Use only the preceding scaled-2048 **development seed201** history. No previously held-out830101–830108 data is used. Original recovery evidence SHA256 ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412; episode-ledger analysis SHA25634148c3f1fea74925d1abbde8dac3e80e92802869114ee01024845c737d420eb.

The ledger contains282 difficult256-tick timeout episodes wholly inside one rollout. Eligible counts by training quarter:56/86/73/67. Sort each quarter by starting interaction and select8 zero-based ranks floor((j+.5)*N/8),j=0..7. This fixed32-cutoff panel does not select on new calibration error or survival. It is deterministic/time-stratified, not a random sample of all282; equal quarter weighting is not a population mean. The26 crossing timeouts remain excluded.

Replay starts at random initialization and reaches1048576 interactions in pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0 on AMD EPYC9V74. Native input SHA25658af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0.

Every one of512 actual fitting batches and1048576 physical-packet rows matches the previous capture, as do all512 diagnostic records, reset/episode ledgers, six original actor/critic checkpoints and every original evaluation record/reward trace. The incoming actor/critic is saved BEFORE each selected fitting update. These are32 different contemporaneous models, incoming updates42–504, not a later final policy substituted retrospectively.

At every selected cutoff, scalar prediction on the exact final noisy observation reconstructs the collector's float32 timeout reward augmentation **bit-for-bit**, and the initial value matches its stored value exactly. The old archive lacks every intermediate actor/Adam snapshot; no comparison to nonexistent all-update optimizer bytes is claimed.

## Continuation contract

Each cutoff gets32 fresh stochastic futures. Keys are2107201000+1000*panel_index+replica,replica0..31; Gaussian action innovations use the separately keyed XOR0xBC79A412 stream. Original/additive ABI and state-transplant controls precede measurement.

Start from the exact cutoff physical state AND already-observed noisy input. Reset only the external diagnostic clock. Follow the same incoming policy, learned Gaussian sigma, ordinary native observation/actuation disturbances and physical-failure rules for8192 additional ticks. Fixed inference shape32x4. No LQR, teacher, optimizer update, physical reset or reapplication of the256-tick training timeout.

Returns at1024/4096/8192 ticks are raw-reward, gamma.999, failure-stopped sums with **no learned endpoint bootstrap**. These estimates condition on physical state and noisy observation, not an observation-only latent-state posterior.

## Calibration

| Training quarter | Mean prediction | Mean measured8192-tick tail | Signed error | RMSE across cutoff means | Survivors |
|---|---:|---:|---:|---:|---:|
| First |881.631|902.528|-20.897|25.165|256/256|
| Second |969.392|969.544|-0.152|9.156|256/256|
| Third |968.762|967.557|+1.205|5.582|256/256|
| Fourth |969.214|975.163|-5.949|9.634|256/256|
| **Fixed panel** |**947.250**|**953.698**|**-6.448**|**14.501**|**1024/1024**|

Nine cutoff means are overpredicted and23 underpredicted. Twelve absolute errors exceed10, three exceed25 and none exceed50. The largest error is incoming update42:793.425 predicted versus841.723 measured. Largest overprediction is incoming247:997.863 versus975.407. Every row is retained.

Approximate fixed-panel99% Student interval for mean signed error: **[-6.556,-6.340]**, across32 independent future-replica panel means. This is Monte Carlo uncertainty for fixed states/policies, not uncertainty over independently trained agents or a guarantee excluding rare failures missed by32 futures at a state.

### Finite-horizon sensitivity

| Additional ticks | Mean measured return | Prediction minus finite return |
|---:|---:|---:|
|1024|610.010|+337.240|
|4096|938.052|+9.197|
|8192|953.698|-6.448|

The short-horizon gaps include omitted future rewards and are not automatically critic bias. Since each native reward is at most1, the maximum positive remaining contribution beyond8192 is .999^8192/(1-.999), approximately0.276. This is not a general two-sided trajectory-error or stability bound.

## Recovery and centring

All1024 survive an additional81.92 nominal seconds, or84.48 seconds including the recorded256-tick prefix. All enter the prior descriptive100-consecutive-tick centring window: |x|<=.25m and |theta|<=.05rad. Latest first window completion: **21.70 nominal seconds from reset**.

However, **300/1024 later leave at least one tight centring bound**, without a physical-limit failure. Counts by quarter92/39/169/0. Do not call all earlier policies permanently tightly centred. This does not contradict the prior final-policy five-minute finding: these are earlier models. The descriptive window does not constrain velocity and is not a safety certificate.

## Effect on the initial target

With the original observed256-tick prefix held fixed, replace only its estimated tail with the independently measured same-policy conditional mean:

`calibrated_scalar_target = recorded_prefix + .999**256 * measured_tail_mean`

The average absolute initial-target difference is **7.752**; maximum **37.384** physical units. The sign relative to the unchanged recorded initial critic baseline changes for **4/32** cases:3 negative-to-positive,1 positive-to-negative.

These are scalar conditional calibration checks, NOT action advantage oracles. They do not integrate over alternative actions, establish that PPO's gradient was wrong, or show that fitting these replacements improves learning. Initial baseline calibration is not established here.

## Verification and preserved failure

Before measurement, both previous packages'648/26 payload hashes, numerical reports and18+20 tests replay; the runtime's31 payloads verify.

The new independent NumPy/SciPy reader verifies32 model/cutoff identities, actual float32 timeout augmentation, all1024 trajectories and **8388608 recorded transitions**. It reconstructs every native reward bit-for-bit, all three return sums, physical ending/no-padding rules, command mapping, bounded observation noise, exported Gaussian innovations, actor commands, physicalized critic values and centring metrics. Maximum independent normalized-command error3.58e-7 and physical-value error2.39e-4 are below unchanged inherited tolerances3e-6/.003.

**All20 new genuine-record corruption/boundary tests pass.** The reader runs no Torch/native/optimizer/pickle code. It does not independently integrate all nonlinear dynamics, regenerate every private native disturbance draw or reconstruct every Adam transaction.

First observer fixture failed because its untrained policy never reached a timeout, so scalar timeout-value capture was not exercised. Failed source/logs are retained. Corrected fixture loads an archived policy with fresh Adam in BOTH plain/recorded controls only to exercise real timeouts. Main historical training still begins at random initialization. No main replay or continuation outcome was retried. A streaming-shell request failed before execution; supervised ordinary container processes completed all work.

| Work | Native transitions |
|---|---:|
|Historical training reconstruction|1048576|
|Historical checkpoint evaluations|630280|
|First invalid observer fixture|8192|
|Corrected observer fixture|8192|
|ABI/state-transplant controls|8279|
|Independent cutoff continuations|8388608|
|**Total**|**10092127**|

Historical replay and observer controls use166400 optimizer transactions. Continuations use zero. No new corrective candidate trained.

## Evidence and disposition

Complete incremental evidence: **ppo_recovery_timeout_calibration_evidence_20261007.zip**,394364996bytes,SHA256 **224667e0204c95577756466688cd924dd2668dd23dc792361cc8cebdeecc453b**. It retains every new trace/model, selected original batch, full selection ledger, sources, fixtures, reports and failure logs. Fresh extraction verifies **562 package payloads** and original selected-input bindings, regenerates VERIFIED.json/CUTOFFS.json/cutoffs.csv **byte-for-byte**, and passes all20 tests. Command: `OPENBLAS_NUM_THREADS=1 python REPLAY.py` with Python,NumPy,SciPy. No simulator/optimizer executes in offline replay.

Compact report/source/fixture package: **ppo_recovery_timeout_calibration_report_code_20261007.zip**,2110976bytes,SHA256 **cdfd60472f1c3c13872c23f0314c0a0fc0fd5df1f7cf64667a8efc988992b544**. It is not the complete calibration data; full REPLAY needs the complete archive. Native regeneration additionally requires the prior143MB archive and separately pinned runtime.

**Do not turn the previous85% estimated-tail statistic into a claim of fabricated recovery success.** Here the same policies really complete the interrupted maneuvers and the estimated tails are reasonably close, with modest average underprediction. Earlier failing recovery states and their large conditional value errors remain unresolved; surviving-timeout calibration cannot clear them.

Next causal target remains acquisition/retention of the early recovery response—not another batch-size sweep, automatic timeout extension or LQR first-action labels. No master/default/protected-source/deployment change, fallback or merge. No new experiment queued. Issues33/35 remain open.
