# Reflection consistency: real policy probes measured; native evaluation pending

September 19 Toronto / September20 UTC, 2026. Issues #35/#33. This is NOT a completed controller-performance experiment.

## Current status

The registered frozen-policy diagnostic is submitted as run35486221478 at source `d11b9fbf99edd0d221c4dd179a421ca46466b6eb`, branch `audit/ppo-reflection-consistency-20260919`. The latest check during this report still shows native preflight job106012983913 queued, without executed steps. No new Rust compilation, native test pass, episode completion, or improved controller is claimed.

Registration: issue35 comment5747262659. Comment5747271701 corrects an arithmetic count before execution: the specified grid is5x3x5x3=225 points, not675. Coordinate sets and evaluation rules were not changed.

## A different, bounded question

The source-normalization experiment failed to improve the matched global recipe. This check asks whether the frozen policy treats mirrored left/right observations inconsistently and whether removing only that inconsistency helps or hurts.

All four observation coordinates are negated for reflection: position, velocity, angle and angular velocity. For the noiseless symmetric plant, the expected contract is f(-s,-u)=-f(s,u), equal quadratic reward, and identical absolute-limit endings. Independent float64 equation tests pass; the actual Rust PendulumEnv reflection checks are included in the queued preflight and have NOT yet run.

Freeze all eight final global-normalization actors at checkpoint512 from run35482824488, source a799c6d3. No critic or weights are updated. Three predetermined action mappings are compared:

- Original latent mean mu(o).
- Reflected latent mean -mu(-o).
- Odd projection [mu(o)-mu(-o)]/2.

Apply the original Gaussian latent innovation and a single20*tanh transform afterward. No fitted mixing coefficient, reference controller, teacher, reward change or privileged observation is introduced. The odd projection requires two network evaluations; no speed claim is made. Averaging nonlinear policies is not guaranteed safe. This is a frozen diagnostic, not a replacement trained controller or solution to from-scratch learning.

## Actual preliminary measurement: fixed observation probes

The225 observations are the Cartesian product of x=[-1.2,-.6,0,.6,1.2]m, v=[-.8,0,.8]m/s, theta=[-.25,-.125,0,.125,.25]rad, omega=[-.5,0,.5]rad/s. All points are retained for all eight actors. They are not equally probable task states, nor presumed recoverable states.

Independent float64 inference uses the original float32 actor weights. Let u(o)=20*tanh(mu(o)). The force antisymmetry defect is abs(u(o)+u(-o)); a reflection-consistent mean policy would have zero defect. Results are derived from actual saved actors, NOT synthetic test fixtures or simulator trajectories.

| Seed | Mean force defect over225 observations, N | Maximum defect, N | Force at exactly centred upright rest, N |
|---|---:|---:|---:|
|41001|2.943000|19.618466|-0.719953|
|41002|1.986056|10.763725|-0.235023|
|41003|1.542818|8.126046|-1.981744|
|41004|1.391270|5.643411|-1.495883|
|41005|1.582345|8.674275|+0.420355|
|41006|1.736070|6.061031|+0.057141|
|41007|1.719933|12.518703|-1.992700|
|41008|1.602768|3.830780|-0.849008|

An illustrative maximum-defect pair for41001 is o=(-1.2,+.8,+.25,-.5) and its full negation. The actor commands approximately-14.177191N and-5.441275N: both leftward rather than opposite directions. This pair was identified from the complete fixed-grid output, not used as a new acceptance criterion. It does not identify which command is better or prove that a single changed action recovers a failed episode.

These measurements establish representation asymmetry. They do NOT establish that asymmetry caused the rail failures, that an optimal policy must have the same chosen action everywhere, or that symmetrizing the output improves stability. The nonzero command at zero observation is a further measured bias, not a stability analysis.

## Pending native comparison

Use the exact existing64 five-minute deterministic nominal,64 five-minute stochastic nominal and64 sixty-second outward trials per history. All three mappings share initial draws, observation/actuation noise, disturbances and policy innovation keys. The original actor's192 episode aggregates per seed must exactly reproduce its immutable preceding records before interpretation. The environment remains the unchanged nonlinear RK4 plant; failure ends the episode immediately.

There are at most4608 evaluation episodes across all eight actors and three mappings, zero main training transitions. Full traces are retained for replication0 of each seed/panel/mode; other trials retain outcomes. Generic preflight/control training is separately additional. Stochastic comparison uses shared noise, NOT a claim that those unchanged noise draws themselves produce exactly reflected paths.

The six primary descriptive contrasts are odd-minus-original completion and discounted return in three panels, paired over eight history means, using two-sided99% Student intervals with Bonferroni correction over six contrasts. Reflected-policy comparisons and initial-side splits are controls/descriptive outputs, not extra trained agents. The current cases/histories have been exposed previously; neither favourable intervals nor zero observed failures would be fresh/hardware qualification. No data-dependent switch, gain or policy selection is allowed.

## Offline checks completed, and limits

Twenty-five Python synthetic/property tests pass, covering reflection algebra, network layout/finite weights, exact grid construction, paired-history denominators, interval families, missing/duplicate/invalid episodes, source integrity and corrupted archives. These tests do not compile the new Rust endpoint or establish a controller success rate.

All eight original source ZIP hashes match the published GitHub artifact digests, and selected actor bytes match their retained manifests. The preliminary probe CSV and summary regenerate byte-for-byte from the supplied frozen weights. Independent probe inference uses float64 rather than claiming identical arbitrary Burn/float32 reduction order. The pending native probe/evaluation outputs will be checked independently against these formulas.

The packaged offline verifier for the future evaluation requires all nine current raw artifacts and validates their manifests, build bindings, original episode replay, fixed denominators/pairing, and every exported action/physics/reward trajectory. It is prepared and unit-tested, not yet tested on outputs of a run that has not executed. Missing artifacts must remain incomplete; no synthetic survival results are substituted.

No production default, trained weights, PR38, master, deployment or fallback controller changed. Global normalization remains in the existing recipe. The experiment's native and controller-performance conclusions remain pending.
