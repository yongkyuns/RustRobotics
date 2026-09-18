# Longer continuous PPO training and robustness screen

September 18, 2026. Issues #35 / #33. Corrected production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The experiment is complete. Longer training alone did not produce a reliably robust controller.** Performance improved at 4,194,304 transitions, then regressed overall at 16,777,216. None of the four final policies passed the predeclared necessary robustness screen. No production default, checkpoint selection rule, learner code, branch merge or deployment was changed.

This rejects the claim that this unchanged training recipe reliably becomes usable just by running longer through the tested budgets. It is not proof that all future longer training, other algorithms, or corrected recipes must fail.

## Frozen protocol and actual execution

[Protocol recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5737374611). Audit branch `audit/ppo-long-training-20260918`, executed commit **63a866cf820673a4d23de085dc3645f7d2cabe01**, [run **35405924363**](https://github.com/yongkyuns/RustRobotics/actions/runs/35405924363). All five jobs completed: preflight and four measurements. Successful execution is not successful controller qualification.

Four exposed development seeds201–204 each started from random initialization and remained in ONE persistent native session through checkpoints0/1,048,576/4,194,304/16,777,216. No optimizer, environment state, partial episode or training random stream was reset between checkpoints. Readouts never reload weights. Historical weight-only files were used for comparison, not resumption. There was no score-based selection, dropped seed, or favorable training rerun.

Training used the unchanged archived nonlinear native library from artifact10571541845, SHA256 of ZIP `d6a470655467d059e14440c645c6ac73aa0040b26b1bebd9228f302538d440a7`. Its Rust source3ba003a8858a9850e82f3ec30958df24a401884b is identical to the frozen PR38 head; the intervening change is a browser test assertion. All training settings are unchanged: one environment,512-step rollout,128 minibatch,four epochs,default2x64ReLU networks,Adam epsilon1e-5,learning rate0.0003,gamma0.99,lambda0.95,fixed latent exploration scale0.1. No extra critic fitting, controller, reward modification, clipping change or physical randomization enters training.

Each final session performs32,768 policy refreshes and524,288 actor plus524,288 critic Adam transactions. Main training totals **67,108,864 simulator interactions**, not16million shared across four policies.

## Evaluations

At every checkpoint the unchanged historical short panels contain32 noisy deterministic-policy10-second and64 noisy stochastic-policy15-second episodes per training seed. Torch2.8.0+cpu,SB32.9.0,NumPy2.2.6,Gymnasium1.3.0 and Python3.11 are pinned for this read-only evaluation holder; optimization is native Rust/Burn.

At1M/4M/16M, a separate Rust executable calls the actual portable `PolicySnapshot.act` and public corrected nonlinear environment. Each seed/checkpoint has224 robustness episodes:

-32 nominal deterministic and32 stochastic uninterrupted300-second episodes. Their60-second survival counts are nested prefixes, not additional independent episodes.
-32 outward-recovery60-second episodes: eight fixed states, four noise repeats each. Position and outward velocity are plus/minus0.8m and0.8m/s; angle plus/minus0.15rad; angular velocity plus/minus0.3rad/s.
-Six one-at-a-time physical errors: plus/minus10percent length,cart mass,pole mass;16 deterministic60-second episodes each.
-16 doubled-noise/disturbance-amplitude60-second episodes and16 deterministic20ms action-delay60-second episodes, with zero initially queued commands.

Evaluation changes are explicit and never fed to training. The first failure ends the trial; auto-reset cannot conceal a failed episode. Commands,actual applied force,initial/final state,maximum excursions,saturation and last-ten-second centring are retained. Original failure bounds remain2.4m cart position and0.6rad angle. Deterministic describes policy action selection, not a noise-free plant.

The predeclared necessary screen requires zero failures for EACH final policy on the nominal deterministic300-second panel and every deterministic stress panel. This is a finite development screen, not hardware safety or an estimate of an arbitrarily small failure probability. Even a pass would require independent training/evaluation confirmation.

The native long panels use new keys and portable Rust inference; the historical short panels use different keys and their original inference path. Do not infer long-versus-short survival probabilities by treating them as matched episodes.

## All fixed checkpoints

| Training steps per policy | Short deterministic10s /128 | Short stochastic15s /256 | Nominal deterministic300s /128 | Nominal stochastic300s /128 | Outward recovery60s /128 |
|---:|---:|---:|---:|---:|---:|
|1,048,576|90|184|97|94|72|
|4,194,304|122|228|114|114|96|
|16,777,216|102|208|107|104|64|

In these panels all nominal episodes surviving60 seconds also reached300 seconds; early failures still count. This observation does not establish indefinite survival.

| Training steps | Short deterministic mean return | Short stochastic discounted mean |
|---:|---:|---:|
|1,048,576|682.833755|78.696973|
|4,194,304|883.099435|82.712588|
|16,777,216|692.087972|83.486257|

The stochastic discounted score increases from4M to16M while both short and long completion counts fall. Reward alone is not the robustness acceptance criterion. The4M checkpoint was predetermined and is reported, not selected as a qualified deployment checkpoint.

## Every final policy, including regressions

| Seed |10s deterministic /32|15s stochastic /64|300s deterministic /32|300s stochastic /32|Recovery /32|Full deterministic screen|
|---|---:|---:|---:|---:|---:|---|
|201|25|56|28|27|16|FAIL|
|202|16|32|17|17|8|FAIL|
|203|32|63|32|32|24|FAIL|
|204|29|57|30|28|16|FAIL|

Seed202 nominal five-minute completions are25/32 at1M,29/32 at4M,and17/32 at16M. Its recovery completions are16/32,24/32,and8/32. This directly contradicts monotonic robustness improvement on this trained trajectory.

Seed203 improves to32/32 nominal five-minute completions in both modes, but still fails8/32 outward-recovery trials. A favorable nominal seed is not evidence that the whole training procedure or that policy's recovery envelope is qualified.

## Final stress results

| Scenario | Completed | Episodes | Failure endings |
|---|---:|---:|---|
|Nominal deterministic300s|107|128|21 position|
|Nominal stochastic300s|104|128|24 position|
|Outward recovery60s|64|128|64 position|
|Length minus10percent|55|64|9 position|
|Length plus10percent|54|64|10 position|
|Cart mass minus10percent|55|64|9 position|
|Cart mass plus10percent|55|64|9 position|
|Pole mass minus10percent|56|64|8 position|
|Pole mass plus10percent|55|64|9 position|
|Doubled noise|56|64|8 position|
|20ms action delay|55|64|9 position|

Every failure in these final robustness panels is a position-limit failure. This is a measured behavioral result, not proof that a particular reward term, critic error or update caused it. The remaining cart-containment problem is not removed by longer training.

## Verification, reproduction and limits

Preflight passes strict Clippy, eight new evaluator tests,68 native trainer tests,seven ordinary balancing controls and nine ordinary learning/evaluator controls. Two separately ignored heavy historical tests were not invoked. The actual native-runtime environment matches the archived simulator bit-for-bit for256 controlled transitions including seven-step timeouts/resets. Grouped/split updates and a full separate-process robustness readout preserve subsequent training parameters exactly.

All original initialization and1M actor/critic checkpoints reproduce byte-for-byte for all four seeds. Short evaluation outcomes also reproduce exactly for seeds202–204. Seed201 has floating-field differences at0/1M, with all episode lengths/endings unchanged; maximum1M return discrepancy0.0020894222 and discounted discrepancy0.0000175445. These are retained, not rerun away; the numerical/platform cause is not uniquely attributed.

Independent offline analysis verifies all four measurement ZIP hashes and124 original payload-member hashes,32768 logged updates per seed,16 actor/critic checkpoint pairs,1536 short and2688 robustness outcomes,scenario recipes,panel identities,ending semantics and costs. Fixed first-episode trace samples independently reconstruct nonlinear transitions within3.305e-7 maximum absolute error,rewards within1.814e-7,and deterministic actions within3.692e-6. Sparse traces do not verify every unexported step,reward total,random draw or optimizer transaction. Full optimizer moments are not exported as exact-resume checkpoints. Seventeen positive/negative analyzer controls pass.

Build artifact10571679369 has101 checked payload members. An earlier Clippy failure at7c55fc1/run35405695902 stopped before any main training; its87-member artifact10572345914 remains retained. The only correction was equivalent fixed-size byte decoding using the lint-preferred API. No test conditions or acceptance thresholds changed.

A separate local replay of the eight-update preflight policy retains identical224 episode durations/endings but small floating differences; it consumes20544 additional evaluation steps. A separately labelled local seed2011M baseline screen consumes2217910 steps. Neither is pooled into the main results or a claim of cross-host bitwise reproduction.

Main costs:67,108,864 training interactions;1,336,566 short-evaluation and28,386,966 robustness-evaluation interactions;2,097,152 actor plus2,097,152 critic Adam transactions;268,435,456 gradient-sample visits per network. Generic preflight/test work is extra and not fully aggregate-instrumented. No efficiency or all-inclusive cost claim is made.

Four exposed training seeds do not support high-confidence population reliability estimates. No hardware,actuator identification,hardware timing qualification or fresh training confirmation cohort was tested. No confidence interval treats thousands of episodes as independently trained agents.

## Retained evidence and disposition

The conversation bundle `rustrobotics-ppo-long-training-evidence.zip` contains unchanged main/build/failed-preflight archives,the prior nonlinear results nested inside them,exact executed sources and native binaries,all saved weights and outcomes,independent analyzer/tests,summaries and offline replay instructions. Offline replay does not execute the native library or restart training.

Production source and defaults remain unchanged by this study. PR38's environment correction is separate from this rejected longer-training qualification. Do not promote16M training as a robust-controller setting; do not silently promote4M as a selected solution either. The next work must address the observed update regressions and recovery/centering limitations with a separately tested correction, not assume additional training time will repair them.
