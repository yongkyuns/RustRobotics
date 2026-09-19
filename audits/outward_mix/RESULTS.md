# Outward-start mixture: pooled recovery improves, advancement screen fails

September 19, 2026. Issues #35/#33. Corrected nonlinear production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision and study identity

**The executed outward-start candidate improves pooled sixty-second recovery from 433/512 to 472/512 compared with continued ordinary-reset training, but FAILS its registered development screen.** Gains are concentrated in two histories, three histories lose recovery completions, and seed41008 loses nominal pole recovery. The original absolute robustness screen also fails in all three long panels. No default change, production merge or deployment is justified.

This verification recovered an already-completed experiment; it did not dispatch, retry or retune training. The previous preparation-only conversation update did not account for this separately registered run. Do not conflate it with the unused local prepared patch:

| Setting | Executed, registered study | Unexecuted local proposal |
|---|---|---|
| Outward position magnitude | 0.4–1.4 m | 0.4–1.2 m |
| Pole-angle range | ±0.2 rad | ±0.25 rad |
| Side choice for four changed starts | Independent random sign | Two starts on each side |
| Evaluation seed offset | 0x02000000 | 0x03000000 |
| Intermediate repetitions per panel | 64 | 32 |
| Advancement criterion | Registered recovery-completion AND return gains, plus nominal retention | Different proposed non-inferiority rules |

Only the executed registration supplies this report's decision rules. No unused proposal criterion was substituted after observing results.

Registration: [issue35 comment5744302237](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5744302237), September19 at18:21:11 UTC. Executed head **3d9e29a1f19157a926f0864ed244f9fcbcb7edec**, [run35461070693](https://github.com/yongkyuns/RustRobotics/actions/runs/35461070693),18:24:05–18:41:21 UTC. First attempt; all nine jobs succeeded: preflight and eight seed measurements. Execution success is not controller qualification.

## Fixed intervention and comparison

Use eight now-exposed development histories41001–41008. Each original recovery-union policy is replayed from random initialization through4096 updates, preserving live Adam, environment, episode state and random streams. BOTH endpoint networks and prefix sampling costs reproduce the original fresh-cohort archive exactly. Historical weights are comparison inputs, not loaded optimizer state.

The complete live session is internally cloned into two descendants:

- **Reset-only control:** continue the measured recovery-union recipe with all eight supplemental streams starting from ordinary task resets.
- **Half-outward candidate:** retain ordinary starts for streams0–3; replace only the first state of streams4–7 with independent outward starts. Position magnitude is uniform[0.4,1.4), same-sign outward velocity magnitude[0.2,1.0), angle[-0.2,0.2), angular velocity[-0.5,0.5). Ordinary observation noise remains.

Original initialization consumes its existing environmental draws before replacement. Candidate start/observation sampling uses a separate fixed domain; original action and transition streams are not consumed by it. After a supplemental failure, subsequent resets remain ordinary. This changes four of eight STARTS, not every subsequent state or all512 supplemental fitting rows. No failure is filtered or feasibility-tested.

Both descendants perform512 additional updates, retaining checkpoints0/32/128/512. The512 main fitting rows, mean of four up-to512-step cutoff futures, eight576-step supplemental streams, first64 fitted rows per stream, global union normalization,1024-row batches,minibatch256,four epochs and actual Rust/Burn optimize loop remain unchanged. The first four supplemental streams and the first main/corrected/tail data reproduce exactly across the two initial updates.

The nonlinear RK4 plant,2x64ReLU networks, learning rate0.0003, Adam epsilon1e-5, gamma0.99, lambda1 for this experimental recipe, fixed exploration, reward/noise and force limits are unchanged. No LQR/teacher, imitation loss, extra critic epochs, privileged network input, reward shaping, controller shield or evaluation-selected update enters training. The supported production lambda remains0.95.

The arms have the same prescribed sampling schedule and optimizer cadence. Actual tail costs can differ because true failures truncate futures. This is continued development on exposed histories, not fresh held-out confirmation or a from-scratch test of the changed mixture.

## Independent evaluation

All checkpoints use new evaluation domains, seed+0x02000000, with paired keys across policies/checkpoints. Short panels contain64 noisy trials per history: deterministic nominal, stochastic nominal and stochastic outward, capped at20.48seconds. Incoming0 and final512 additionally contain64 five-minute deterministic nominal,64 five-minute stochastic nominal and64 sixty-second stochastic outward trials per history/policy. There are15,360 retained episode outcomes.

Outward evaluation starts0.8–1.2m off-centre,0.4–0.8m/s outward,angle±0.15rad and angular velocity±0.3rad/s. These individual draws and outcomes are excluded from fitting. A true failure ends the trial immediately. Deterministic refers to policy action selection, not a noiseless environment. Different panels use different starting/randomness keys, so long trials are not continuations of selected short trials.

Centred/upright success requires survival and |x|≤0.5m,|theta|≤0.1rad throughout the final ten seconds. It does not require those tighter bounds throughout the entire trajectory. Initial evaluation is counted once per history, not as two independent trained agents.

## Final sustained results

| Policy | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Incoming, before continuation |504|491|437|
| Continued reset-only control |502|495|433|
| **Half-outward candidate** |**502**|**503**|**472**|

Outward failures decrease from79 to40 compared with the matched control, a net39-case gain. The candidate's outward completion rate is92.188%, versus84.570% for control. All candidate long-panel survivors also meet final-ten-second centring/upright bounds.

The incoming437/512 outward result is from this experiment's new evaluation domain. It does not replace or contradict the earlier435/512 fresh-cohort outcome on different cases.

### Every history: control → candidate

| Seed | Nominal deterministic /64 | Nominal stochastic /64 | Outward recovery /64 |
|---|---|---|---|
|41001|64 →64|64 →64|60 →53|
|41002|60 →63|60 →62|37 →58|
|41003|64 →62|64 →63|63 →61|
|41004|63 →64|63 →63|62 →63|
|41005|64 →64|64 →64|61 →61|
|41006|60 →63|58 →62|33 →61|
|41007|63 →63|59 →64|56 →56|
|41008|64 →59|63 →61|61 →59|

Two weak controls improve substantially:41002 gains21 recovery completions and41006 gains28. But41001 loses7,41003 loses2,and41008 loses2;41004 gains1 and two histories tie. The mean gain is not uniform improvement.

Seed41008 is a separate nominal regression: candidate five-minute deterministic completion drops from64/64 to59/64, and all five failures are pole-angle-limit failures. Neither the incoming nor continued control has nominal angle failures on this panel. Thus broader recovery-start exposure reintroduces a pole-recovery failure in one history; these measurements do not uniquely establish which loss contribution or parameter change causes it.

Across final candidate long panels, deterministic nominal has5 position and5 angle failures; stochastic nominal has9 position failures; outward has40 position failures. The candidate also has three angle failures in short deterministic reset and two in short stochastic reset, all in41008. Remaining failures cannot now all be described as rail failures alone.

### Short fixed panels

| Policy | Deterministic complete /512 | Deterministic mean return | Stochastic complete /512 | Stochastic discounted mean | Outward complete /512 |
|---|---:|---:|---:|---:|---:|
|Incoming|498|1962.902453|509|86.902554|416|
|Reset-only|494|1950.093708|503|86.683611|420|
|Half-outward|494|1945.399144|504|86.358899|455|

Improving outward completion does not imply improving both nominal return metrics. At final512, candidate nominal means are lower than control in this aggregate comparison. All per-history scores and endings remain in the numerical tables.

Candidate short outward completions progress416→423→443→455 at checkpoints0/32/128/512; control progresses416→421→414→420. Candidate short nominal deterministic counts are498→494→497→494 and stochastic counts509→501→506→504. Intermediate regressions are not removed, and only the final prescribed checkpoint determines the development decision.

## Registered development decision: FAIL

The original registration requires BOTH final outward-completion and outward-discounted-return gains to have positive lower bounds under paired-history Student99% intervals with Bonferroni adjustment for these two contrasts. It also requires pooled nominal survival no worse than control OR incoming and at least61/64 nominal successes for every candidate history in both five-minute panels.

Intervals use eight paired history effects,df7,not512 independently trained agents. They are two-sided99% family intervals over the two recovery contrasts, an approximate finite-history model rather than a distribution-free safety statement.

| Candidate minus control | Mean effect | Registered adjusted interval |
|---|---:|---|
|Outward completion proportion|+0.076172|[-0.201956,+0.354300]|
|Outward discounted return|+0.694544|[-4.004074,+5.393162]|

Neither lower bound is positive. The completion result equals a gain of7.617percentage points with an interval of[-20.196,+35.430]percentage points. Failure to resolve improvement is not proof of no effect; it fails the criterion for advancing this fixed candidate.

The nominal deterministic rule also fails:502 pooled successes are below incoming504, and seed41008 has59/64 rather than the required61. The nominal stochastic retention rule passes:503 exceeds both control495 and incoming491, and every seed has at least61.

Do not relax the rules, replace them with the unused preparation package's criteria, or select a subset of improved seeds. This fixed four-of-eight mixture is not a qualified universal correction.

## Original absolute screen: also FAIL

Report the original robustness thresholds separately, without treating this exposed cohort as fresh confirmation. Each long panel requires at least507/512 pooled successes,every history≥61/64,the three-panel-family one-sided seed-level lower bound≥95%,and≥99% centring among survivors.

| Candidate panel | Pooled success | Worst history | Family-adjusted lower bound |
|---|---:|---:|---:|
|Deterministic nominal|98.047%|92.188%|94.538%|
|Stochastic nominal|98.242%|95.313%|95.875%|
|Outward recovery|92.188%|82.813%|85.352%|

All three fail the pooled threshold. Deterministic nominal and outward recovery also fail the weakest-history and lower-bound conditions. Centring among survivors passes, but it cannot erase failures from the attempted-episode denominator. No hardware safety or full attraction-region guarantee follows.

## Verification and limits

All nine native jobs completed on their first attempt. Preflight passes **96 native unit/audit tests**,seven ordinary balancing controls,nine ordinary learning/evaluator controls,strict Clippy,formatting,compiled endpoint discovery and protected-source restoration. Seven heavy audit endpoints remain ignored in preflight; measurements explicitly invoke the outward-mixture endpoint. Two separately ignored historical heavy integration tests were not run. The production browser/platform matrix was not rerun for this audit-only study.

Independent verification checks all9 published ZIP digests,99 build+3816 result payload hashes,and1936 nested historical payloads. It validates15,360 evaluation identities/keys/durations/endings/finite values and attempted-episode denominators;8192 continuation-update records and budgets;56 regular actor/critic checkpoint pairs;historical initial weight bytes AND prefix costs;and first-update main/tail/unchanged-half equality.

Detailed numerical checks cover predefined updates1/128/512 of both arms:24,576 main rows,221,184 supplemental return-support rows,and768 optimizer transactions. Sequential float32 GAE,original/refitted targets and union normalization reconstruct bit-for-bit. Maximum independent actor-loss discrepancy is4.196e-8,value-loss discrepancy2.840e-5,post-step means7.185e-7. Final selected optimizer snapshots match the regular128/512 checkpoints. The audited mixture starts satisfy their registered ranges; individual signs are sampled, not forced two-per-side.

All144 predetermined full evaluation traces,covering1,684,161 transitions,reconstruct initial-state pairing,action innovations,nonlinear dynamics,force transformation,rewards,endings,return sums,force RMS and final-ten-second centring within fixed inherited tolerances. Maximum discrepancies: dynamics2.322e-7,actor inference6.114e-7,command2.149e-6,reward1.583e-7,episode return8.669e-13. Other episodes retain aggregate outcomes, not complete trajectories. Supplemental GAE is reconstructed, but not every supplemental physical transition is independently re-simulated.

**49 offline analyzer tests pass**, covering missing/corrupt/unlisted archives,invalid outcome domains/endings/nonfinite fields,failed episodes mislabelled centred,weak histories hidden by pooled averages,the actual registered statistical family,normalization/GAE boundaries and snapshot decoding. Synthetic test fixtures are not controller results. Full unexported optimizer moments,gradients and RNG streams are not independently regenerated; native clone/isolation/next-update controls support those preservation claims,not weight files alone.

Measurement uses Rust1.98.1/Burn0.20.1 with the pinned production lock. Offline verification uses Python3.13.5/NumPy2.3.5/SciPy1.17.0 and never executes the archived native runner or trains. No measured field,numerical tolerance,acceptance criterion or seed was changed during verification.

## Costs and disposition

Replaying the eight existing histories consumes228,858,216 training-data interactions:16,777,216 main+61,086,056 cutoff tails+150,994,944 supplemental. Continued control consumes29,356,535 and candidate29,355,665,for58,712,200 continuation interactions total. Overall training-data cost is287,570,416; evaluation adds118,685,284. Generic preflight and uninstrumented inference are additional.

Each arm per history has262,144 main and2,359,296 supplemental interactions,plus up to1,048,576 tail interactions;8192 actor and8192 critic Adam steps;2,097,152 sample visits per network. The experiment is matched between its two continuation arms,not an efficiency comparison against ordinary PPO.

Retain the constructive gains for the previously weak histories and the contrary failures. The next question is why this fixed mixture helps some recovery policies while degrading already-good recovery and pole control in others; these results do not justify increasing the mixture or extending training without another bounded test.

Evidence bundle: `rustrobotics-ppo-outward-mix-verified-evidence.zip`, containing all nine unchanged raw archives with historical inputs nested,exact executed sources/native runner,numerical tables,independent verifier/tests and replay instructions. A small analysis-only package and readable report are supplied separately. No duplicate learning run,production lambda change,PR38/master/deployment modification,merge or fallback installation occurred during this verification. #33/#35 remain open.