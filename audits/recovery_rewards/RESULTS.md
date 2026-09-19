# Reward-bearing recovery coverage: a passed screening condition, not robust control

September 19, 2026. Issues #35/#33. Corrected nonlinear production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The bounded experiment is complete. Adding reward-labelled recovery observations passes the predeclared necessary continuation screen: all four union objectives improve, no candidate-versus-incoming effect is confidently harmful, and seed203 has confidently beneficial effects on both evaluation panels.** Six other candidate-versus-incoming comparisons are inconclusive, including three negative point estimates. Absence of resolved harm is NOT noninferiority or a safe-update guarantee.

Recovery sampling is not uniformly superior to equal-size near-upright sampling: it is confidently better on seed201's two panels and seed204's outward panel, but worse on seed203's outward panel. Completion totals change little. This supports a repeated-update learning experiment, not a production default or reliability claim.

This study executes ONE conditional update from each of four exposed lambda1 training histories. It does not train a new candidate from scratch, test repeated use, or run the five-minute robustness screen. Production lambda remains0.95; PR38, master, deployed assets and protected MuJoCo/control files remain unchanged. No merge occurred.

## Protocol and provenance

[Pre-execution protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5742279244). Branch `audit/ppo-recovery-rewards-20260919`. Actual executed head **a69d78c82dd1f194a2c3311dec907d3d178c87b7**, [run35446349591](https://github.com/yongkyuns/RustRobotics/actions/runs/35446349591). Repaired preflight and all four measurement jobs completed successfully. Each full measurement executed once; none was discarded or rerun for a favorable score.

Each seed201–204 replays the existing ordinary lambda1 history through update8191, then captures the actual512-row batch for update8192 ending at4,194,304 interactions. The incoming actor/critic, all original observations/actions/rewards, four sampled tails, mean-four corrected targets, unregularized actor/critic endpoint and every original optimizer transaction reproduce the preceding recovery-anchor evidence byte-for-byte.

The live modules, parameter identities, actor/critic Adam histories and minibatch RNG cursor are internally cloned and restored. They are not reconstructed by loading only weights. The unchanged path is also restored and replayed after both candidate updates; separate controls verify agreement on a subsequent identical expanded-batch optimization.

## Four fixed policies and the intervention

The policies are incoming `old`; the historical512-row mean-four corrected update `mean4`; that corrected batch joined with512 new near-upright reward-labelled rows `near-union`; and that batch joined with512 new ordinary-reset/recovery reward-labelled rows `recovery-union`.

Both supplemental banks use eight frozen-old-policy simulator streams of576 transitions. Select the FIRST64 observations/actions per stream, unconditionally, to produce512 training rows. The other512 transitions provide future rewards but are not actor/critic fitting rows. Ordinary detached lambda1 returns preserve true-terminal/reset boundaries and bootstrap surviving cutoffs. A selected row has513–576 sampled rewards ahead unless a true terminal closes its path sooner. No success, reward, critic-error or failure-based selection is used.

The near bank starts at the original cutoff physical state and noisy observation with the external collection clock reset. The recovery bank starts from the original task reset distribution. Both use identical independent random-domain construction and collection budgets. Only the start-state distribution differs between these banks; reset sampling consumes different random draws, so their subsequent physical paths are not asserted identical. After a real failure, ordinary reset behavior applies.

Physical states initialize the actual simulator and are archived for diagnostics. Networks receive only the existing noisy observations. Samples carry the old policy's actual stochastic actions, log probabilities and rewards. There is no old-action imitation penalty, supplied controller, privileged-state input, fitted diagnostic action label, evaluator-return label or score-based update selection. The preceding KL regularizer remains inactive throughout this experiment.

Raw advantages are normalized ONCE over each1024-row union. Both union arms use four epochs and minibatch256:16 actor and16 critic Adam steps, exactly the same step count as the original512/batch128 update. They use twice the gradient-sample visits. The two union paths share their restored incoming optimizer histories and all epoch permutations. The ordinary production optimize loop remains intact apart from existing test observers.

**The strongest controlled comparison is recovery-union versus near-union.** Comparing either union with the historical small batch additionally changes sample count, normalization population, minibatch size and sample-visits. Equal Adam steps do not imply equal compute or isolate state coverage against that small-batch baseline.

Architecture2x64ReLU, Adam epsilon1e-5, learning rate0.0003, gamma0.99, lambda1 for these diagnostic histories, reward, noise, exploration, force transformation and corrected nonlinear plant remain fixed. Longer sampled reward support attenuates the final bootstrap; it does not create exact value labels, eliminate correlated samples, or prove correct advantage signs throughout the batch.

## Independent full-policy results

For every seed, evaluate all four policies on256 fresh paired ordinary noisy resets and256 fresh paired outward-start episodes. The horizon is2048 ticks,20.48 seconds, or true failure with no automatic reset masking. Outward starts have displacement0.8–1.2m, same-sign outward velocity0.4–0.8m/s, angle within±0.15rad and angular velocity within±0.3rad/s. Those stress outcomes never enter fitting.

Training-bank, prior-study and current evaluation domains are distinct. All policies share each current evaluation initial condition and random-innovation key. The48 predeclared paired Student/Bonferroni99% contrasts are four seeds times two panels times all six pairwise policy differences. Intervals quantify evaluation variation for fixed policies, not a population of independently trained agents; the Student approximation is not an exact distribution-free guarantee.

### Recovery-union versus incoming policy

A negative effect means lower discounted return than before the update.

| Seed | Ordinary-reset effect [adjusted interval] | Outward-recovery effect [adjusted interval] |
|---|---|---|
|201|+0.000981 [-0.069331,+0.071293]|-0.051420 [-0.184317,+0.081476]|
|202|-0.071840 [-0.175302,+0.031621]|-0.074533 [-0.280906,+0.131840]|
|203|+0.149339 [+0.007896,+0.290781]|+0.076680 [+0.019814,+0.133547]|
|204|+0.066621 [-0.106647,+0.239889]|+0.026508 [-0.020786,+0.073802]|

Two beneficial effects, six inconclusive effects, zero confidently harmful effects. Three candidate point estimates are negative. Do not rename inconclusive outcomes as successful preservation or universal improvement.

On this same NEW panel, the unaugmented mean-four update has four confidently harmful effects: seed201 reset/recovery, seed202 reset, and seed204 reset/recovery actually total FIVE. Its remaining effects are inconclusive. The specific effects are201 reset-0.258165/recovery-0.345953;202 reset-0.161368/recovery-0.261653;203 reset+0.024836/recovery+0.018293;204 reset-0.060071/recovery-0.026216. Seed202's recovery upper bound is+0.000275 and remains inconclusive. This new evaluation panel is not pooled with prior panels or substituted for historical results.

### Matched near-upright data control

Recovery-union minus near-union has resolved gains on seed201 resets,+0.291533 [+0.142453,+0.440612], seed201 outward recovery,+0.326121 [+0.092607,+0.559635], and seed204 outward recovery,+0.107630 [+0.029306,+0.185953]. It has a resolved regression on seed203 outward recovery,-0.298031 [-0.490183,-0.105879]. Other between-union differences are inconclusive.

The near-union update itself improves seed203 on both panels but remains confidently harmful on seed201's two panels,seed202 resets,and seed204's two panels. Thus simply adding equal-sized fresh near-upright data is not interchangeable with the recovery mixture. Conversely, the large seed203 near-union improvement demonstrates that recovery-specific data are not necessary for every beneficial update.

Recovery-union is also resolved better than unaugmented mean-four on seed201 resets/recovery and seed203 recovery. Other comparisons against mean-four are inconclusive. All48 contrasts are retained in `results/effects.csv`, rather than presenting only these favorable cases.

### Completion counts include every failure

| Panel, pooled across four fixed histories | Incoming | Mean-four512 | Near-union1024 | Recovery-union1024 |
|---|---:|---:|---:|---:|
|Original noisy resets /1024|726|720|728|727|
|Outward-recovery starts /1024|512|496|501|514|

For seed203, recovery-union nominal completions stay193/256 while outward completions rise144→148/256. These gains in discounted return do not translate into near-perfect survival. Across all seeds, the primary candidate improves completion counts only marginally and still fails many trials. These are not1024 independently trained controllers and not sustained five-minute tests.

## What the new trajectories contribute

Every near-bank selected sample has513–576 future rewards and no terminal in any near stream. Recovery streams include FIVE actual terminal events, distributed3/1/1/0 across seeds201–204. Of2048 selected recovery rows,320 have a shorter terminal-closed return, not a falsely zeroed live-cutoff bootstrap. Minimum remaining path lengths per seed are52,100,45,513. No selected row with a short unfinished path is admitted.

Maximum selected noisy angle magnitudes in the recovery banks are0.2533,0.2135,0.2474,0.2145rad, versus0.0190,0.0123,0.0087,0.0130rad in the near banks. This directly broadens actor-fitting coverage rather than merely penalizing changes at those observations. The more extreme outward evaluation start distribution is not explicitly used for fitting.

The recovery raw advantages are also much more variable than those of the near bank: standard deviations3.314,2.046,4.888,7.672 versus0.0854,0.0505,0.3528,0.0958. A different normalization scale and relative sample influence are part of this mixture; the result cannot isolate geometric coverage from its reward/advantage distribution. These observations are descriptive, not an after-the-fact change to the acceptance rule.

### The optimizer makes nonzero progress

| Seed | Near-union own surrogate gain | Recovery-union own surrogate gain | Recovery-union gain on original corrected batch |
|---|---:|---:|---:|
|201|+0.004500264|+0.002510872|+0.000495793|
|202|+0.004335641|+0.004442763|+0.003685048|
|203|+0.002003517|+0.005030009|-0.000610812|
|204|+0.000063966|+0.003784860|+0.000241380|

All recovery-union objectives improve, meeting the declared nonzero-progress condition. However, seed203 gives up objective value on the original corrected batch while improving its union objective and independently measured returns. The original single-batch surrogate is not a universal policy-improvement metric. No objective score is used to select a candidate after execution.

## Verification and retained failures

The first preparation at `cbafe915d4fdcd95fb2c67b4c801f6f529811a9d`,run35446243603,stopped before any measurement because strict Clippy rejected a needless indexed range loop in the diagnostic support assertion. The corrected preparation changes it to an iterator without changing its semantics or any experimental settings. The original failed79-member artifact is retained unchanged. No full measurement was rerun.

Repaired preflight passes **83 native unit/audit tests**,seven ordinary balancing controls,nine ordinary learning/evaluator controls,Clippy,formatting,compiled-runner discovery and protected-source restoration. Four audit endpoints remain ignored during preflight; measurements explicitly invoke only the new reward-coverage endpoint. Two separately ignored historical heavy integration tests are not counted as executed.

Native controls verify supplemental collection reproducibility/live-state and RNG isolation,original-first union ordering/single normalization,required reward support and internal optimizer restoration through a subsequent identical expanded-batch update. Retained earlier controls cover true-terminal bootstrap removal and actor/critic optimizer history. The active optimizer is real Rust/Burn, not a Python replacement.

Independent offline analysis verifies all five successful ZIP digests and84 build+840 measurement payload members,the failed79-member artifact,and724 directly used nested recovery-anchor members. All232 compared historical files match exactly,including original targets/tails/weights and all original per-minibatch records. Checks cover8192 evaluation outcomes,2048 original rollout rows,36864 full supplemental transitions,4096 selected supplemental rows,two union target sets per seed,and192 recorded actor/critic transactions. Both unions use the same permutations and all epochs cover each row once.

All stored sequential-f32 GAE returns/raw advantages and normalized advantages reconstruct bit-for-bit, including true-terminal paths. Actor/critic losses independently reconstruct within3.56e-8/9.44e-6; updated batch means within1.91e-7. Full supplemental trajectory checks have maximum dynamics error2.43e-7,reward1.44e-7,value4.14e-5 and log-density1.29e-6. The32 predetermined first-replication evaluation traces contain27603 transitions; their episode totals reconstruct within4.13e-13. Other evaluation replications retain aggregate outcomes,not every step.

All **29 independent analyzer tests** pass,including real-record mutations,corrupt/unlisted/missing evidence,global versus blockwise normalization,terminal support and paired statistics. Two initial mathematical test oracles failed: one incorrectly demanded a near-exact unit standard deviation after sequential float32 normalization; the other used a differently associated algebraic return expression. The test oracles were corrected to explicitly reproduce the intended f32 operation ordering,without changing measured data,the analyzer's estimator code,or acceptance tolerances. Both failed and corrected test logs are retained. No native measurement was repeated as a result.

Offline checks do not reconstruct every unexported Adam moment,gradient,RNG draw or evaluation trajectory. Persistence is covered by native anchor/sham/next-transaction tests and historical exact binding,not inferred from bare weight files. Native execution uses Rust1.98.1/Burn0.20.1 and the production lockfile. Offline analysis uses Python3.13.5/NumPy2.3.5/SciPy1.17.0 and never runs the archived native executable. Full production platform/browser workflows are not rerun for this audit-only change.

## Cost and next boundary

Main costs:16,777,216 replayed history interactions;8192 mean-four tail interactions;36,864 supplemental-bank interactions;10,736,648 evaluation interactions. Each network performs524224 prefix Adam steps and256 target steps,including the sham replay. Target sample-visits are49152 per network;the unions have twice the per-update sample-visits of the512-row baseline. Generic preflight/control interactions and extra inference are additional and not completely aggregate-instrumented. No equal-compute or efficiency claim is made.

The necessary continuation criterion PASSES,but it is intentionally weaker than learning or safety acceptance. The result supports testing a fixed repeated-update reward-bearing recovery-data recipe. It does not establish universal state-coverage causality,calibrated action credit everywhere,absence of forgetting,noninferiority on the six inconclusive comparisons,or robustness of a new trained controller. A fresh held-out cohort and sustained nominal/recovery tests remain separate work.

Evidence bundle: `rustrobotics-ppo-recovery-rewards-evidence.zip`. Includes all unchanged successful and failed current artifacts,historical inputs nested inside,exact prepared source/native runner,all supplemental trajectories/targets/weights,outcomes,traces,independent analyzer/tests,all numerical reports,provenance and offline replay instructions. Production defaults/PR38/master/deployment remain unchanged and unmerged.
