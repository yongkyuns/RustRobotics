# Seed 203: fixed-state recovery action-credit diagnosis

September 18, 2026. Learning issues: #35 / #33. Existing integration fix: PR #37.

## Decision

**Completed. Wrong-sign GAE action credit persists in all three tested seed-203 policies, including both extra-critic-fitting candidates.** The predeclared final-critic panels contain three confirmed opposite-sign cells for baseline, two for extra-frozen, and three for extra-refresh. All eight cells were already wrong with the pre-extra-phase critic and remain wrong afterward. This is a conditional action-ranking diagnosis, not a production correction or proof of the complete causal history of seed203's training regression.

The preceding extra-critic-fitting study still supplies a promising learning component. However, better target-fitting loss cannot be treated as correct action credit. No production defaults changed, no diagnostic controller was installed, and no PR was merged.

## Protocol and execution

[Protocol before new outcomes](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5736087948). Production baseline: `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`. Actual executed audit commit: **`527f44b87aef1f973180ab4f1202beec5584690c`**, branch `audit/ppo-seed203-credit-20260918`. [Run35394710493](https://github.com/yongkyuns/RustRobotics/actions/runs/35394710493) completed all four jobs successfully: preflight and three measurements. No failed measurement, favorable-result rerun, discarded state, or adaptive sample increase. Runner queue delays are not attributed to a code or learning failure.

Inputs are the final seed203 actors and pre/post-final-extra-phase critics from critic-cadence run35391982059. For each actor, use each stochastic evaluation episode's FIRST observation at ticks25,50,...,475 satisfying abs(noisy x)>=0.5 and noisy x*noisy velocity>0. Sort episode keys, then choose16 evenly spaced keys. Eligible distinct episode counts are55/50/51. Selection reads no rewards, endings, critic errors or new probe outcomes. Restore the exact preceding recorded physical state and preserve the archived noisy initial observation.

At each of48 states, run256 independent paired continuations. First latent action is mu(observation)+sigma*z-0.02 versus mu(observation)+sigma*z+0.02, with the frozen sigma approximately0.1. After that action, both branches follow the SAME original actor with common temporal action innovations and paired native environment randomness. The policy function is the same, not necessarily subsequent forces, since observations diverge. The actual unchanged Rust PendulumEnv::step_with_rng executes every physical transition; only diagnostic state restoration and a new1500-step continuation cap are added.

Main runtime: Rust1.98.1/Burn0.20.1 and baseline Cargo.lock, Python3.11.16/Torch2.8.0+cpu/NumPy2.2.6. Torch evaluates the frozen portable networks; independent inference and archived-value comparisons check the layout. No actor/critic parameter updates, new measurement training episodes, supplied controller, reward/noise change, or privileged-state learning input is introduced. Standard regression tests separately exercise existing training routines.

## Primary results

Positive always means PLUS first action minus MINUS first action. Actual benefit is the finite-horizon sum of actual rewards with gamma=float32(0.99), without critic bootstrap. Estimated benefit is lambda=float32(0.95) GAE on the same independent continuations using the final critic. True terminals remove future value; truncations use final-observation value in the estimator.

Approximate two-sided Student intervals use256 paired draws conditional on each fixed state. Bonferroni correction covers96 primary quantities,48 states times two estimates, with99% family level. Opposite-sign cells require BOTH adjusted intervals to exclude zero with opposite signs; agreement requires both to exclude zero on the same side. Otherwise the result is unresolved. Pointwise99% intervals and raw mean signs are also retained.

| Frozen actor / final critic | Confirmed opposite | Confirmed same sign | Unresolved | States |
|---|---:|---:|---:|---:|
| Baseline |3|9|4|16|
| Extra-frozen |2|12|2|16|
| Extra-refresh |3|12|1|16|

Raw opposite-mean counts before significance requirements are5/3/3. The eight confirmed cells also have opposite-sign estimates with the pre-extra critic. Within each arm, pre/post scoring holds actor, states and outcomes fixed. None of those eight rankings is repaired by the final extra phase.

**Do not interpret3/2/3 as a matched-state treatment effect.** Each actor supplies its own visited-state panel; these are three actors from one exposed training seed, not48 independent trained agents. Sign agreement does not establish accurate magnitude, calibrated values, or performance on untested states.

### Every confirmed opposite cell

Indices are zero-based within each actor's fixed panel. Complete adjusted intervals, physical states, observations and source keys remain in the bundle.

| Actor | State / source tick | Actual return change | Final GAE change | Pre-extra GAE change |
|---|---|---:|---:|---:|
| Baseline |0 /150|+0.013832|-0.003268|-0.003268|
| Baseline |1 /50|-0.002409|+0.002009|+0.002009|
| Baseline |5 /75|-0.012654|+0.005238|+0.005238|
| Extra-frozen |0 /125|+0.009887|-0.007847|-0.008096|
| Extra-frozen |1 /50|+0.014607|-0.006088|-0.005949|
| Extra-refresh |1 /50|+0.102659|-0.043702|-0.044244|
| Extra-refresh |6 /50|+0.105175|-0.032127|-0.032521|
| Extra-refresh |12 /125|-0.016897|+0.003047|+0.002918|

## Concrete future-value reversal

Extra-refresh state1, source key14066184195547004931, has physical x approximately-0.743m, velocity-2.074m/s, pole angle0.0297rad and angular velocity-0.421rad/s. The cart is moving outward. The actual first observation remains noisy.

| Quantity, plus minus minus | Mean | Adjusted interval |
|---|---:|---|
| Actual discounted return change |**+0.102659**|**[+0.085545,+0.119773]**|
| Final-critic GAE change |**-0.043702**|**[-0.047612,-0.039791]**|

The GAE decomposition is **+0.024054 lambda-weighted direct reward -0.067755 future-critic contribution = -0.043702 estimated credit**. The initial-state value baseline is identical in the pair and cancels. This wrong ranking is not merely an offset in that initial baseline: future-value terms reverse an already positive reward contribution while independently measured gamma-discounted benefit is positive. Extra-refresh state6 and extra-frozen state1 show the same direct-reward reversal pattern.

Not every wrong cell has that pattern. Extra-refresh state12 has actual change **-0.016897 [-0.017814,-0.015980]**, but GAE **+0.003047 [+0.002168,+0.003925]**. Its lambda-weighted reward contribution is already+0.002382; the future-critic contribution+0.000665 fails to correct the misleading short-weighted signal. Both patterns are retained.

None of the eight opposite-sign cells changes its aggregate completion count between the two first-action branches. These demonstrate incorrect RETURN ranking, not that the small perturbation itself explains the rail-failure count or original training regression. This is not a replay of an actual optimizer parameter direction or its sampled training batch.

## Separate numerical-runtime corroboration

While primary jobs were queued, [one extra-refresh local replay was declared before its outcomes](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5736312735). It uses the exact compiled native library, frozen inputs, seeds,256 paired draws and1500-step cap, but Python3.13.5/Torch2.10.0+cpu/NumPy2.3.5. The primary pinned jobs were not replaced or restarted.

All16 primary classifications reproduce:3 opposite,12 agree,1 unresolved. Every branch duration, ending, first latent action and environment seed matches. Returns are not bitwise identical: maximum absolute per-branch difference6.74021e-5, GAE difference1.57720e-5, first-force difference9.53674e-7. Their numerical/platform cause is not uniquely attributed, and this does not establish portable bitwise equivalence.

This replay consumes6,006,975 additional physical interactions. It repeats the same draws, is not independent statistical evidence, and is never pooled into primary intervals.

## Verification and limitations

Native preflight passes68 unit tests, including three new direct-native/ABI, boundary and identical-input controls; strictClippy, release build, formatting/source restoration;7 balancing and9 ordinary learning controls. One heavy integration test in each suite is ignored and is not claimed executed. Eight Python diagnostic tests pass. Identical-action controls produce exactly equal trajectories and zero contrasts with2,880 physical interactions.

All four current ZIP digests and502 current manifest-listed entries verify:469 in the build,11 per measurement. Original input archive digests and376 original members also verify. Independent analysis reselects all48 states and validates shapes/counts/endings, paired initial values and first-action shifts. Lambda-one TD sums telescope to actual returns after adding the initial value and removing the explicit time-limit tail; maximum reported error below6.3e-13.

Full traces for predetermined replicate0 of every state/branch contain75,376 primary native transitions. Independent reconstruction checks initial states/noisy observations, continuity/noise bounds, coupled Euler dynamics, bounded force innovations, actual reward relation, actor draws, critic inference, returns, GAE, lambda-one sums and TD. Maximum reward error1.524e-7; independent float64 critic inference error2.375e-5; trace-return reconstruction error below4.2e-13. Local corroboration traces also pass.

Other255 replications per state retain aggregate quantities, lengths/endings, seeds, first actions/forces and tails, not full trajectories. Offline verification does not pretend to resimulate aggregate-only branches. Twelve independent analyzer tests pass. Analysis uses Python3.13.5/NumPy2.3.5/SciPy1.17.0 and never loads the native library.

Main returns are finite-horizon. Largest recorded discounted critic tail at1500 steps is2.881e-5; this is not a proved bound on unobserved future rewards. A fixed physical state plus noisy observation also differs from the observation-only critic's conditioning over all possible hidden states. No Markov-sufficiency, global stability, or unique training-cause claim follows.

## Costs and durable evidence

Primary interactions: baseline8,097,277; extra-frozen5,133,518; extra-refresh6,006,975; **total19,237,770 across24,576 branch trajectories**. Separate local corroboration adds6,006,975; identical-action controls2,880. Measured continuations plus those controls total25,247,625. Generic native regression/test interaction is additional and not fully aggregate-instrumented. No policies are trained for the measurement and no speed/efficiency claim is made.

Self-contained conversation bundle **`rustrobotics-ppo-seed203-credit-evidence.zip`**, **49,675,964 bytes**, SHA256 **`bdde399069978c249325a08fec260cc68def4a135fca173a8f2c5069fd8d351f`**. It contains all four unchanged current raw archives, nested original inputs, frozen panels, exact prepared sources/library/versions, all outcomes, the separate local replay and executed launcher, full report, plot, analyzers/tests and offline replay instructions. No native library is invoked by replay.py.

Fresh unpack verified23 outer hashes, all raw/member hashes and376 nested original members. **summary.json, states.csv, corroboration.json and control-reconstruction.json regenerated byte-for-byte**, and all12 analyzer tests passed. Retrieve the bundle by name rather than assuming a previous sandbox path.

| Current raw artifact | ID | SHA256 |
|---|---:|---|
| Build |10567690558|0a84a33693b80af1bbee9850c27975a8f86dfacfdb4376983676069af43024d4|
| Baseline |10567927369|5c15767fe0da48c2f64381a7c41697abba64df734028c3521636de8cd922ef39|
| Extra-frozen |10568276378|8fb54785ca9c985b02baff735002101d14b9153331f37e3b12ff4d841dbbc8ae|
| Extra-refresh |10567539320|e839ddb937504b6d09521a9feb07df8ab74a112cf6d972acdcde402bf3769cfd|

## Next correction boundary

Extra value fitting at preserved actor cadence does not remove biased recovery-action credit in these frozen seed203 policies. A further candidate must improve independently checked return/action-credit estimates, not merely fit its own bootstrapped targets more closely. Test it against the fixed witnesses and ordinary from-scratch learning without feeding diagnostic states or outcomes into training. Do not simply repeat the already-rejected lambda-one or larger-batch sweeps.

Master, supported defaults and protected MuJoCo/control paths remain unchanged. PR37 is a separate unmerged integration fix. #35/#33 remain open; no new held-out training cohort was consumed and no sustained-balancing acceptance is claimed.
