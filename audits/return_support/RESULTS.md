# Longer return support at gamma 0.995 — completed, not adopted

September 20, 2026 Toronto / September 21 UTC. Issue #35.

Executed source **0a4445f6c5ab83650133a696faaeb764fd5b6c26**, workflow **35548063439**, branch `audit/ppo-return-support-20260920`. All 17 jobs completed successfully on the first attempt at September 21 01:00:00 UTC (September 20 21:00 Toronto). Registration comment **5753874883** precedes all new outcomes. This closes the previously submitted experiment; verification did not launch duplicate training.

## Decision

**Do not adopt the 1,024-step support recipe.** With training gamma held at 0.995, doubling sampled future support adds five long outward-recovery successes but loses thirteen deterministic and ten stochastic nominal successes. Both the registered development-advancement screen and the separate absolute reliability screen fail. No best-checkpoint selection, omitted weak seed, favorable measurement retry or changed threshold was used.

The production gamma 0.99/lambda 0.95 defaults remain unchanged. This is a test of the experimental recovery-data recipe, not a replacement production controller or a fresh hardware qualification.

## Fixed intervention

Sixteen policies train from random initialization: eight paired, previously exposed seeds 41001–41008, across support512 and support1024. Both actor and critic learn gamma 0.995 from the first target; no stale-critic warm start. Each run performs 4,096 ordinary-reset recovery-union updates, then 512 half-outward supplemental-start updates. Adam, environment, partial episode and random state persist throughout.

| Per-update quantity | 512-step control | 1,024-step candidate |
|---|---:|---:|
| Primary fitting samples |512|512|
| Supplemental streams |8|8|
| Selected fitting samples per stream |64|64|
| Collected transitions per stream |576|1,088|
| Maximum steps per cutoff future |512|1,024|
| Cutoff futures |4|4|
| Total fitting samples |1,024|1,024|
| Adam transactions per network |16|16|

The additional supplemental observations supply rewards, not extra fitting rows. True terminal masks prevent credit crossing failure/reset boundaries. Rewards, corrected nonlinear plant, noise, network, exploration, global advantage normalization, experimental lambda 1, minibatch 256, four epochs, learning rate 0.0003 and Adam epsilon 1e-5 remain unchanged. No teacher, imitation, symmetry wrapper, extra critic fitting or safety fallback is introduced.

Every support512 actor/critic checkpoint at 0/1,024/4,096/4,128/4,224/4,608 matches the preceding gamma995 archives exactly. All 4,608 update rows per short-support run, after the arm label, and final cost records match too. Historical inputs are hash-bound comparison files, never loaded to initialize training. Both arms' initial main batches and first 576 physical supplemental transitions match, including the 64 fitted rows. Short-tail prefixes match long tails; an actual terminal prevents extension. Later learning trajectories may differ.

Evaluation uses new fixed random draws at offset 0x10000000, common scoring gamma 0.99 for both arms, and the existing undiscounted/survival measures. Deterministic describes policy action selection: environmental noise remains active. Failure ends every trial immediately. Different panels have different cases; long trials are not extensions of specific short trials. Only final update 4,608 determines advancement.

## Final results

| Support length, both training gamma 0.995 | Five-minute deterministic | Five-minute stochastic | Sixty-second outward recovery |
|---|---:|---:|---:|
|512 steps|507 / 512|510 / 512|450 / 512|
|1,024 steps|494 / 512|500 / 512|455 / 512|
|Candidate minus control|-13|-10|+5|

Candidate completion rates are 96.48%, 97.66% and 88.87%; the control rates are 99.02%, 99.61% and 87.89%. All 87 remaining candidate failures are position-limit failures: 18 deterministic nominal, 12 stochastic nominal and 57 outward. The control has five/two/62 failures respectively, including three nominal angle failures. Avoiding those angle failures does not compensate for the candidate's additional rail failures.

Every long-trial survivor in both arms stays within +/-0.5 m and +/-0.1 rad throughout its final ten seconds. This does not certify tighter bounds throughout the episode or remove failed cases from the denominator.

The control's new 450/512 outward result is not substituted for the previous 443/512 on a different random domain. Historical weights and update histories reproduce exactly; new evaluation scores need not.

### All paired history counts

|Seed|Outward control|Outward candidate|Deterministic nominal control -> candidate|Stochastic nominal control -> candidate|
|---|---:|---:|---:|---:|
|41001|59 / 64|62 / 64|64 -> 64|64 -> 64|
|41002|61 / 64|62 / 64|64 -> 64|64 -> 64|
|41003|57 / 64|59 / 64|63 -> 57|64 -> 60|
|41004|51 / 64|63 / 64|62 -> 64|63 -> 64|
|41005|61 / 64|59 / 64|64 -> 63|64 -> 63|
|41006|40 / 64|32 / 64|63 -> 55|64 -> 59|
|41007|59 / 64|54 / 64|64 -> 63|64 -> 63|
|41008|62 / 64|64 / 64|63 -> 64|63 -> 63|

Five histories improve outward completion and three regress: twenty gained successes versus fifteen lost. Seed 41004 improves substantially. Seed 41006 worsens to 50% outward completion and also loses nominal reliability. Seed 41003 gains two outward successes while losing six deterministic and four stochastic nominal successes. These are descriptive observed effects, not independently resolved causal claims per history.

The candidate is already weaker on early short panels. At update 1,024, nominal deterministic counts are 360 control versus 288 candidate, stochastic 366 versus 268. At update 4,096, before the outward mixture, nominal counts are 501 versus 497 and 500 versus 493; short outward is 438 versus 413. At update 4,128 the candidate nominal counts drop to 481 and 480. These checkpoints locate observed behavior, not a causal optimizer transaction.

Final short outward completion is 456/512 control versus 443/512 candidate. The modest long-panel gain is not consistent across distinct evaluation panels. All checkpoints remain in the evidence.

## Registered acceptance and uncertainty

Development advancement requires a positive outward-completion lower bound, no pooled nominal loss and at least 61/64 nominal successes for every candidate history. Intervals use eight paired history averages, Student t with seven degrees of freedom, and two-sided 99% coverage adjusted over three completion contrasts.

|Candidate minus control|Mean percentage-point effect|Adjusted 99% interval, percentage points|
|---|---:|---:|
|Deterministic nominal|-2.539|[-10.966, +5.888]|
|Stochastic nominal|-1.953|[-7.057, +3.151]|
|Outward recovery|+0.977|[-13.456, +15.409]|

All intervals include zero. That is not evidence of equivalence or certain population-wide harm. The required positive outward lower bound is absent and nominal retention also fails. **Development screen: FAIL.** Repeated episodes are not independently trained controllers.

The separate absolute screen retains at least 507/512 per panel, every history at least 61/64, a three-panel-adjusted one-sided 99% history lower bound at least 95%, and at least 99% centring among survivors.

|Candidate panel|Completed|Weakest history|History lower bound|Absolute result|
|---|---:|---:|---:|---|
|Deterministic nominal|494 / 512|55 / 64|88.884%|FAIL|
|Stochastic nominal|500 / 512|59 / 64|93.605%|FAIL|
|Outward recovery|455 / 512|32 / 64|66.720%|FAIL|

Only centring passes in every candidate panel. Exposed training seeds are not fresh confirmation, even if a numerical criterion had passed.

Common-gamma final outward mean return changes only from 59.864628 to 59.951322 discounted, and 5,170.876130 to 5,221.049074 undiscounted. Nominal total means decline from 29,594.778432 to 28,850.462488 deterministic and 29,634.961669 to 29,067.175630 stochastic. These are descriptive, not additional significance tests.

## Interpretation and cost

The actual float32 gamma gives a cutoff coefficient of 0.0768101 after 512 steps and 0.00589979 after 1,024. Independent reconstruction verifies the longer support reaches the intended targets. **Lowering that coefficient does not yield a robust learner.** This rejects the tested support-length change as a sufficient correction; it does not prove bootstrap errors are irrelevant or that all longer-horizon estimators fail. Sampling variance, later bootstrap states, value targets and subsequent learning paths also change. None was independently isolated here.

Actual aggregate training-data interactions:

|Source|Control, eight histories|Candidate, eight histories|
|---|---:|---:|
|Primary|18,874,368|18,874,368|
|Supplemental|169,869,312|320,864,256|
|Cutoff futures|68,900,359|133,093,042|
|Total|257,644,039|472,831,666|

The candidate uses approximately 1.835 times the simulator experience, averaging 59.10 million versus 32.21 million per policy. Both arms retain 589,824 Adam steps and 150,994,944 fitting-sample visits per network across eight histories. The study adds 93,306,588 evaluation transitions; generic preflight/inference work is additional. No equal-total-compute or sample-efficiency improvement is established.

## Verification and reproducibility

All 17 jobs pass on their first attempt. Preflight passes 108 unit/audit tests, seven ordinary balancing and nine ordinary learning controls, strict Clippy, formatting and source restoration. Nine heavy audit endpoints are ignored in the normal unit pass; the current endpoint is explicitly measured. Two historical heavy integration endpoints and the production browser/platform matrix are not rerun.

The **unchanged 63-test offline suite passes**. Checks cover all 17 published ZIP digests and 4,494 payload hashes; 224 bound historical inputs; 21,504 evaluation outcomes; 73,728 update rows; 96 regular actor/critic checkpoint pairs; first paired physical data; and historical-control equality. Detailed updates 1/4,097/4,608 cover 319,488 support rows with reconstructed GAE and 768 optimizer transactions. GAE, sampled tails, cutoff substitution, returns and normalization reproduce bit-for-bit.

All 144 exported evaluation trajectories, 1,170,196 transitions, reconstruct actions, nonlinear dynamics, rewards, endings and scores. Reward arithmetic is exact float32. Maximum discrepancies: actor mean 5.146e-7, force 2.095e-6 N, dynamics 2.396e-7, aggregate return 7.816e-13. Recorded optimizer-loss discrepancies are at most 4.042e-8 actor and 3.983e-4 critic under unchanged scale-aware checks. No tolerance changes.

Unexported gradients, Adam moments, random draws and remaining full trajectories are not independently regenerated. Native controls, not weight-only inference, establish historical training reproduction. An unavailable interactive container invocation was replaced with a noninteractive run of the unchanged verifier; no simulator outcome was retried.

**Fresh-unpack replay verifies 35 outer payload hashes, reruns all 63 tests and regenerates all five numerical reports byte-for-byte**, including both failed decisions. Complete evidence `rustrobotics-return-support-evidence.zip`: 204,886,816 bytes, SHA256 `1955a53546f9fff5683c27a43df15ce45c5b0e65e101a1098124317d5c70d0af`. It includes all original raw archives, exact compiled source/logs, verifier/tests, five numerical reports and instructions. The detached verification receipt and analysis-only package retain the final replay logs. Offline replay executes no network, native binary, simulator or training.

## Disposition

No production gamma, lambda, reward, normalization, learned weights, PR #38, master, deployment or fallback changed. No merge. Keep the production defaults unchanged; gamma 0.995/lambda 1 and longer support remain experimental. This study is complete with a negative adoption decision, not pending training.