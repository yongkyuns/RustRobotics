# Source-wise advantage normalization: historical cases improve, robustness does not

September 20, 2026 UTC (September 19 in Toronto). RustRobotics issues #35 / #33.

## Decision

**The registered source-standardize candidate fails the development advancement screen and all three absolute robustness panels.** It repairs eight of twelve previously exposed angle-failure cases, but on the new evaluation domain it completes 465/512 sixty-second outward trials versus 474/512 for unchanged global normalization. Five-minute deterministic and stochastic completions also decline, from 504/503 to 503/500. No production normalization change is promoted.

The existing run was completed and verified; this verification did not dispatch duplicate training or retry a measurement. [Registration 5746872084](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5746872084) preceded execution. Executed source: `a799c6d35c3753266fa382230d734e2d7de18312`, [run 35482824488](https://github.com/yongkyuns/RustRobotics/actions/runs/35482824488), first attempt, all nine jobs successful. Submitted at 02:00:06 UTC, it finished at 02:55:58 UTC after the earlier queued status. Successful execution is not successful controller qualification.

## Fixed comparison

Eight exposed histories, seeds 41001–41008, replay the original recovery-union prefix through 4,096 updates. Incoming actor/critic bytes and prefix costs reproduce the preceding archives exactly. Complete live sessions, including Adam histories, random streams, environment and unfinished episodes, are cloned into three descendants. Each receives 512 additional updates; checkpoints 0/32/128/512 are fixed, with only 512 used for advancement.

Every descendant uses the SAME four-of-eight outward-start mixture. The first 768 fitting rows are primary512 plus ordinary-reset supplemental256; the last256 originate from outward-start streams. These are source labels, not claims about each sample's instantaneous physical state.

| Arm | Subtracted mean | Divisor |
|---|---|---|
| global | Whole1024-row mean | Whole-batch standard deviation |
| source-center | Separate768/256-group means | Original whole-batch standard deviation |
| source-standardize, preregistered candidate | Separate group means | Separate group standard deviations |

Moments use the existing sequential-float32 population formula and 1e-6 standard-deviation floor. Only actor advantages are changed. Raw return targets, critic loss, collection, reward, nonlinear RK4 plant, noise, exploration, architecture and optimizer settings are unchanged. There is no teacher, imitation loss, gradient projection, reward change, extra critic fitting or evaluation-based update rejection. Experimental lambda remains1; the supported production default remains0.95.

Both normalization interventions use the same first-update observations/actions/old likelihoods/raw advantages/returns as global. Every one of the sixteen first-update critic snapshots matches exactly across all three arms. Subsequent actors naturally collect different trajectories; later critics are NOT claimed identical across arms. The global arm reproduces all historical half-outward actor/critic checkpoints at32/128/512 exactly.

## New-domain sustained results

Each panel contains64 trials per history. Deterministic means policy mean actions, not a noiseless plant. True failures end trials immediately. Different panels have different keys; a five-minute trial is not an extension of a specific short trial. Evaluation seed offset is0x04000000, separate from fitting and previous evaluation domains.

| Policy | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Incoming, before continuation |506|503|441|
| Global normalization |504|503|474|
| Separate means, common scale |504|503|460|
| Separate means and scales |503|500|465|

The candidate achieves98.242%,97.656% and90.820%, respectively. Its nine deterministic failures comprise eight position-limit and one angle-limit failure; its twelve stochastic nominal and47 outward failures are position-limit failures. Every long-panel survivor meets the final-ten-second bounds |x|<=0.5m and |theta|<=0.1rad. Survivor-only centring does not remove failed trials from the denominator.

The incoming441/512 and global474/512 outward totals belong to this NEW evaluation domain. They do not replace the previous437/472 totals on different episodes, despite identical incoming/global policy weights.

### Every final history

Each cell is deterministic nominal / stochastic nominal / outward completions, each out of64.

| Seed | Global | Separate means | Separate means and scales |
|---|---|---|---|
|41001|64 /64 /52|64 /63 /59|64 /63 /59|
|41002|64 /63 /58|63 /63 /58|63 /63 /57|
|41003|61 /62 /63|63 /64 /63|63 /63 /63|
|41004|64 /64 /62|64 /64 /62|64 /64 /62|
|41005|63 /63 /61|63 /63 /61|63 /63 /61|
|41006|64 /64 /59|64 /62 /55|64 /64 /56|
|41007|64 /64 /59|63 /64 /42|62 /60 /47|
|41008|60 /59 /60|60 /60 /60|60 /60 /60|

Candidate outward completions improve on one history, regress on three and tie on four. Seed41001 gains7 successes, but41007 loses12. Seed41007 also loses four stochastic nominal successes,64->60. Separate mean subtraction alone is not a universal alternative: its41007 recovery count falls to42/64. It remains a mechanistic control, not a post-hoc replacement candidate.

The candidate still improves outward completion over the incoming policy,441->465, but continuing the existing global recipe improves it more,441->474. This experiment rejects the normalization change as an improvement over matched continued training, not the broader recovery-data recipe.

## Registered decisions

Advancement requires positive lower limits for BOTH candidate-minus-global long-outward completion and discounted-return effects, using two-sided paired-history Student99% intervals Bonferroni-adjusted for two contrasts. It additionally requires pooled five-minute completions no lower than global OR incoming and at least61/64 nominal completions in every history.

| Candidate minus global | Mean | Adjusted interval |
|---|---:|---|
| Outward completion | -1.758 percentage points | [-13.442,+9.926] percentage points |
| Outward discounted return | +0.003443 | [-1.845793,+1.852679] |

Neither recovery lower limit is positive. Both nominal retention rules also fail:503<504 and506 for deterministic,500<503 for stochastic;41008 has60/64 deterministic, and41007/41008 each have60/64 stochastic. All four advancement components fail.

These are eight paired training-history effects, not512 independently trained agents. The intervals are approximate finite-cohort uncertainty estimates. They do not prove zero effect, equivalence or certain expected harm; they fail the registered advancement requirement. Outward discounted point estimates improve on two histories and worsen on six despite their near-zero pooled difference.

The separate original absolute screen requires at least507/512 pooled completions, every history at least61/64, a three-panel-family one-sided history-level lower bound>=95%, and>=99% centring among survivors.

| Candidate panel | Pooled | Worst history | Family-adjusted lower bound | Gate |
|---|---:|---:|---:|---|
| Deterministic nominal |98.242%|93.750%|95.391%|FAIL|
| Stochastic nominal |97.656%|93.750%|94.285%|FAIL|
| Outward recovery |90.820%|73.438%|80.134%|FAIL|

All three fail pooled and weakest-history requirements. The latter two also fail the lower-bound requirement. No threshold, seed, or final-checkpoint rule was changed. This reused cohort would not be fresh held-out confirmation even if its numerical screen passed.

## The intervention worked mathematically, but that was insufficient

Across the24 selected batches per arm, the outward-origin quarter's median share of squared normalized advantage magnitude changes from81.4175% under global normalization to25% under separate means/scales. With separate means but common scale, it remains86.7671%. These are magnitude accounting, NOT percentages of Adam's update or independent statistical observations.

Independent reconstruction confirms the formulas bit-for-bit. At the first common-data update, separate mean subtraction changes2,080 of6,144 nominal-origin normalized signs compared with global normalization. The separate-scale candidate makes nominal normalized values independent of the other group's values, as intended and tested. This does not establish that all changed signs are correct. Source-standardization changes relative objective weighting; unit group variance does not guarantee aligned gradients, calibrated action credit or safe control.

The observed failure therefore cannot be dismissed as an inactive hook or different first-update data. Removing this coupling is not sufficient to repair the learned controller. The experiment does not prove normalization caused every previous failure, nor identify whether loss of useful weighting, estimator noise or later changed trajectories explains every regression.

## Historical angle-failure witnesses, separate from new evaluation

All twelve previously exposed seed41008 nominal angle-failure cases are evaluated at their original checkpoint/key with the three frozen descendants. They were selected because they had already failed and are correlated retrospective diagnostics, not fresh acceptance episodes.

| Normalization | Survive /12 | Position failures | Angle failures |
|---|---:|---:|---:|
| Global historical replay |0|0|12|
| Separate means |6|4|2|
| Separate means/scales |8|2|2|

Global outcomes reproduce every original field exactly. The candidate repairs eight witnesses, but two remain angle failures and two become rail failures. Repairing the selected examples does not override worse whole-panel completion. Two repaired candidate cases are from checkpoint128; the other six are from512. Both interventions repair the previously illustrated final long-det episode10, but neither repairs every original nominal maneuver.

All36 witness traces were additionally reconstructed offline: dynamics, actor outputs, action transformation, noise bounds, rewards, endings, totals, force RMS, initial pairing and final centring. They contain256,330 steps. This extension does not simulate new episodes or feed any evaluation label into training.

## Earlier checkpoints retained

Short stochastic outward completions at0/32/128/512 are:

- global:426/449/460/467 out of512;
- separate means:426/440/459/472;
- separate means/scales:426/429/422/474.

The candidate falls below its incoming outward count at128 before improving by512. Its final short outward count exceeds global, while its separately sampled long outward count is lower. Different panel keys prevent interpreting this as observed failure specifically after20.48seconds. All nominal and return trajectories, including contrary seed effects, remain in the tables. No best checkpoint was selected.

## Verification, provenance and limitations

Native preflight passes102 unit/audit tests, seven ordinary balancing controls and nine ordinary learning/evaluator controls, plus strict Clippy, formatting and protected-source restoration. Eight heavy audit endpoints are ignored in preflight; the current endpoint is explicitly measured. Two separately ignored historical heavy integration tests were not run. No production browser/platform matrix was rerun for this audit-only study.

All nine published current ZIP digests and6,093 payload hashes verify:105 build plus5,988 result payloads. The80 retained predecessor inputs are bound to their original manifests; full predecessor archives were verified in native preparation but are not duplicated in this new evidence bundle. The current eight initial actor/critic pairs and all global descendant checkpoints match those retained historical bytes.

Offline checks cover21,504 new-domain episode records,36 historical witness records,12,288 continuation-update records and80 regular actor/critic checkpoint pairs. Detailed updates1/128/512 cover36,864 primary rows,331,776 supplemental return-support rows and1,152 optimizer transactions. GAE, corrected/raw return targets and each normalization formula reconstruct bit-for-bit in sequential float32. Maximum independent discrepancies: actor loss5.281e-8, critic loss2.840e-5, updated action means7.775e-7.

All192 preselected new-domain full traces contain2,044,359 steps; maximum dynamics discrepancy2.358e-7, command2.138e-6, reward1.432e-7 and aggregate9.664e-13. The36 witness traces add256,330 steps; maximum dynamics4.574e-7, command2.315e-6 and aggregate3.198e-13. Other evaluation replications retain aggregates, not full paths. Complete unexported gradients, Adam moments and random sequences are not independently regenerated; native cloning/isolation/history controls establish those specific preservation properties.

The unchanged core offline verifier retains its59 tests. Eight added synthetic witness-reconstruction controls bring the total to67 passing tests. Synthetic fixtures are not controller measurements. Two whole-suite tool invocations hit60/120-second execution limits; their partial logs are retained. The unchanged analysis then completed with all histories and traces. No measured data, formula, statistical rule or numerical tolerance was changed; no training run was repeated.

Measurement used Rust1.98.1/Burn0.20.1 with the pinned production lock. Offline analysis used Python3.13.5/NumPy2.3.5/SciPy1.17.0 and never executes the archived native runner or trains. Successful evidence verification explicitly retains the failed controller screens.

## Cost and disposition

The shared prefix replays use228,858,216 training-data interactions. The three continuation arms use29,355,665 (global),29,355,804 (separate means),and29,356,033 (separate means/scales), for316,925,718 training-data interactions total. New-domain evaluation adds161,872,567 interactions and historical witnesses256,330. Generic preflight and uninstrumented inference are additional.

Each branch/history has262,144 main plus2,359,296 supplemental interactions and up to1,048,576 tails;8,192 actor and8,192 critic Adam steps;2,097,152 sample visits per network. This is matched continuation compute/sampling, not evidence of sample efficiency against ordinary PPO.

**Retain global normalization for the existing recipe; neither tested source-wise variant is qualified as its replacement.** The old witness improvements remain useful mechanistic evidence, not a robust-control result. No production default, PR38, master, deployment or controller fallback was changed or merged. PR38 remains draft/open at4739f37; #33/#35 remain unresolved.

The complete current evidence bundle contains all nine unchanged raw archives, selected historical comparison inputs, exact executed source/runner, results, independent analysis/tests and replay instructions. A small analysis-only archive and readable report accompany it. No duplicated training or favorable outcome selection occurred during this verification.
