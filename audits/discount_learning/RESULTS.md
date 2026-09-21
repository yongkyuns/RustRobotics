# Discount learning: nominal gains, weaker outward recovery

Experiment completed September 20, 2026. Issue #35. **Do not adopt training gamma 0.995 as a replacement for 0.99 in this recipe.** The candidate passes the two nominal absolute screens on this exposed cohort, but fails outward recovery and registered development advancement. This is an actual from-scratch learning comparison, not rescoring old paths.

## Execution and fixed protocol

Registration: issue #35 comment5749864155, posted12:41:47UTC before dispatch12:45:17UTC. Executed source **6516572add5509ccb338e4ecb55d572733f99bd2**, run **35511524659**, finished13:39:44UTC on its FIRST attempt. All17 jobs pass: preflight and16 training runs. Verification retrieved the completed run; it did not dispatch duplicate training or retry measurements.

Eight paired EXPOSED development seeds41001–41008 train gamma0.99 and0.995 from random initialization. Discount is set before the first target. Paired initial actor/critic weights and initial physical paths match exactly; computed return targets differ. No warm-start weights, changed objective in an already-trained critic, teacher, reflection wrapper, imitation penalty or controller fallback. Live Adam/environment/unfinished-episode/RNG history persists throughout training.

Both arms execute4096 ordinary-reset recovery-union updates followed by512 half-outward supplemental-start updates. Each update fits512 primary+512 selected supplemental rows;8x576 supplemental transitions with first64 fitted per stream;four up-to512-step sampled futures at the primary cutoff. Global advantage normalization, experimental lambda1, nonlinear RK4 plant, reward, noise, networks, exploration, minibatches/epochs and Adam settings are unchanged. This does not modify the supported lambda0.95 production default.

Gamma0.99 actor AND critic reproduce all archived checkpoints4096/4128/4224/4608 exactly for every seed. These checkpoints are comparisons, never loaded to initialize training. All fixed checkpoints0/1024/4096/4128/4224/4608 are retained; only4608 decides advancement.

## Final sustained results

Each panel has64 evaluations per trained policy,512 per arm. Deterministic means mean-action selection, not a noiseless plant. True failure ends each trial. Different panels use different keys. Both discounts are evaluated with the SAME gamma0.99 score, alongside common undiscounted return and survival.

| Training discount | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
|0.99 control|508|502|477|
|0.995 candidate|**509**|**507**|**443**|

Candidate rates are99.414%,99.023%,86.523%. Every long survivor meets final-ten-second centring/upright bounds, |x|<=.5m and |theta|<=.1rad. That is not a guarantee throughout the episode or an excuse to exclude failures.

Control failures: deterministic2position+2angle; stochastic7position+3angle; outward35position. Candidate: deterministic2position+1angle; stochastic4position+1angle; outward**69position**. No simultaneous failures occur. The new evaluation domain is offset0x08000000; control477 outward must not be substituted for474 in the earlier reflection domain despite identical weights.

### Every final training history

Cells are deterministic nominal / stochastic nominal / outward successes, each out of64.

|Seed|Gamma0.99|Gamma0.995|
|---|---|---|
|41001|64 /63 /56|64 /63 /63|
|41002|64 /62 /60|64 /64 /60|
|41003|63 /63 /62|64 /64 /62|
|41004|64 /64 /60|63 /63 /40|
|41005|64 /63 /59|64 /64 /59|
|41006|64 /64 /60|63 /62 /37|
|41007|64 /62 /58|64 /63 /58|
|41008|61 /61 /62|63 /64 /64|

Two outward histories improve,two regress,four tie.41004/41006 lose43 successes combined;41001/41008 gain9,net loss34. The weakest candidate completes37/64=57.8% outward trials. The nominal improvements are retained rather than hidden.

## Registered decisions

Development requires a positive lower bound for candidate-minus-control outward completion, using paired-history two-sided99% Student intervals adjusted across three long-completion contrasts. Nominal pooled counts must not fall below control; every candidate history must have>=61/64 in both nominal panels.

|Completion contrast|Mean percentage points|Adjusted interval, percentage points|
|---|---:|---|
|Deterministic nominal|+0.195|[-2.189,+2.580]|
|Stochastic nominal|+0.977|[-2.868,+4.821]|
|Outward recovery|**-6.641**|**[-32.957,+19.676]**|

Nominal retention passes; outward-improvement condition fails, so **development_screen=False**. All intervals include zero. This does not prove universal expected harm or equivalence; it fails to establish the required consistent improvement. The units are8 paired training histories, not512 independent agents.

Absolute rules additionally require>=507/512 pooled,each history>=61/64,family-adjusted one-sided history completion lower bound>=95%,and>=99% centring among survivors.

|Candidate panel|Successes|Worst history|Family-adjusted lower bound|Result|
|---|---:|---:|---:|---|
|Deterministic nominal|509/512|63/64|98.326%|**PASS**|
|Stochastic nominal|507/512|62/64|97.459%|**PASS**|
|Outward recovery|443/512|37/64|64.164%|**FAIL**|

**absolute_screen=False overall.** Two nominal passes are not hardware or fresh held-out qualification. Seeds were already exposed; no criterion, weak seed or final-checkpoint rule was changed.

## Scores and earlier checkpoints

Final outward common-gamma discounted mean return decreases61.035184->60.193122; undiscounted mean decreases5478.644015->5099.932635. These descriptive scores are not an additional preregistered acceptance test. They cannot be inflated by using candidate gamma for only its own evaluation.

Short stochastic outward completions at0/1024/4096/4128/4224/4608:

- control:0/260/429/433/451/462 out of512;
- candidate:0/174/426/431/439/448 out of512.

The two severe recovery weaknesses appear BEFORE the outward-start phase. At4096,41004 short outward completes55/64 control versus43/64 candidate;41006 completes45/64 versus25/64. The last512 mixed-start updates therefore cannot be the sole origin. This locates earlier weakness, not its causal optimizer/critic mechanism. All checkpoint returns and contrary cases remain in the supplied tables.

## Interpretation

Longer discounting changed the ranking of particular fixed paths in the preceding audit. It does not follow that a learner trained with that discount discovers better recovery. Here nominal balancing improves slightly while observed outward recovery worsens substantially for two seeds.

Support lengths were intentionally fixed. A512-step surviving-path bootstrap has weight0.005824 atgamma0.99 versus0.076810 at0.995,about13.2x greater dependence. This was anticipated in the protocol. It is a plausible next diagnostic, NOT proof that bootstrap error caused the weak policies. Objective,target scale,visited states and later critic fits all change. No matched-return-support intervention or new value oracle was performed. Do not promote a gamma change or assume increasing it again fixes this failure.

## Verification and limits

All17 published current ZIP digests and4344 payload hashes verify(104 build+4240 result).128 retained predecessor comparison inputs are bound to their original manifests; full predecessor archives were checked by native preparation and need not be duplicated here. Current source/build/runner identities match the registered head.

Checks cover21504 evaluation outcomes,73728 training-update records,96 regular actor/critic checkpoint pairs,and768 selected optimizer transactions. Detailed updates1/4097/4608 cover24576 primary and221184 supplemental support rows. Main GAE,sampled-tail returns,cutoff substitution,supplemental targets and global normalization reconstruct BIT-FOR-BIT under each arm's actual float32 training gamma. All paired first-update observations/actions/likelihoods and original target inputs agree before learning diverges.

All144 predetermined full traces/1243991 transitions reconstruct actions,nonlinear dynamics,rewards,endings,force RMS and score sums. Rewards reproduce BIT-FOR-BIT in float32. Max independent errors:actor5.174e-7,command1.892e-6N,dynamics2.230e-7,aggregate7.958e-13. Training-loss reconstruction maxima:actor3.877e-8,critic3.983e-4;updated means7.391e-7. Existing absolute-plus-relative tolerances were unchanged. Other episodes retain aggregates. Unexported full gradients,Adam moments and RNG sequences are not independently regenerated; native history/isolation controls cover their stated contracts.

Native preflight:101 unit/audit tests,7 ordinary balancing+9 learning/evaluator controls,Clippy,format,protected-source restoration. Eight heavy audit endpoints ignored in preflight; current endpoint explicitly invoked by16 measurement jobs. Two historical heavy integration endpoints and full production browser/platform matrix were not rerun.

The unchanged prepared offline verifier passes48 synthetic/property tests. Synthetic fixtures are not controller outcomes. No new native binary was executed locally. An unavailable interactive container-session request was replaced by noninteractive execution of the same verifier; no formula,data,tolerance or measurement changed. The standalone package supplies a full fresh-unpack replay command and numerical reports; its receipt is retained with the downloadable evidence.

## Cost and disposition

Training-data totals258213881(control) and257644039(candidate),approximately32.28M/32.21M per policy;515857920 overall. Evaluation94767479 transitions. Generic preflight/inference additional. Per arm/history2359296 primary+21233664 supplemental+up to9437184 tails;73728 Adam steps and18874368 gradient-sample visits PER network. Tail counts differ with failure. This is matched prescribed work,not a sample-efficiency comparison with ordinary PPO.

**Retain gamma0.99 for this recipe.** No production gamma/lambda/reward/normalization,learned weights,PR38,master,deployment or fallback was changed or merged. PR38 remains draft/open at4739f37. The experiment is complete with a negative adoption decision,not queued or pending training. Full evidence contains17 unchanged raw archives,actual executed sources/logs,all checkpoints and tables,and an offline verifier that uses no network,native executable,simulator or training.
