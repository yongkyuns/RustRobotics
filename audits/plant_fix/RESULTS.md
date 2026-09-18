# Shared nonlinear plant correction and learning retest

September 18, 2026. Production fix: PR #38, `fix/ppo-shared-nonlinear-plant`, head `4739f370558b9443708c920ac30614f86e3c07bb`. The branch includes the unmerged PR #37 prerequisite at `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`. Master and deployed assets have not been changed by a merge or deployment.

## Result

The confirmed train/live environment mismatch has been corrected in source and passes the dedicated native/WASM contract tests. The fixed-budget learning retest is complete. On the SAME nonlinear evaluation task, newly nonlinear-trained policies achieve **90/128 deterministic 10-second completions and 184/256 stochastic 15-second completions**, versus **50/128 and 98/256** for the retained linear-trained policies. Three seeds improve; seed203 regresses. This is an integration correction with observed learning gains, not a claim that all PPO failures or sustained-balancing requirements are solved.

## Source correction

- `rust_robotics_algo::cart_pole` owns the single nonlinear point-mass derivative and held-force RK4 integrator. The existing linear model remains for LQR/MPC prediction, not PPO physical stepping. State order, SI units, left-positive angle convention and force sign are explicit.
- `PpoTrainerConfig.plant` carries serializable pole length and both masses to stream zero and every pooled training environment. Missing physical values in legacy JSON default to historical parameter values; they do not reactivate linear dynamics.
- Live PPO delegates clipping, actuation/observation noise, disturbances, reset sampling and terminal/timeout handling to the same `PendulumEnv`. The GUI clock only schedules fixed steps.
- Plant or noise changes invalidate the incompatible trainer and stop its retained policy from driving the changed environment. Explicit restart creates the matching learner. A replacement browser worker must publish its matching snapshot before old weights can execute. Policy activation and explicit restart reset the episode; ordinary snapshot publication does not.
- DOM actions synchronize global noise before trainer creation. Stopped-policy execution still observes the same failure/time-limit reset contract. Classical LQR/MPC caches are cleared when entering the PPO stepping path.

No PPO losses, optimizer equations, neural architecture, default learning rate/epsilon/GAE/exploration, extra-critic passes, or reward weights were retuned. No MuJoCo or user-owned robot-control files were modified. Temporary preparation workflows are on audit branches, not the production PR.

## Dedicated verification

The exact measured Rust commit is `3ba003a8858a9850e82f3ec30958df24a401884b`, a direct ancestor of the PR head. The only subsequent PR change is one browser-test assertion treating an absent optional error field as no error; Rust source is identical. Run **35402364550** uses pinned Rust1.98.1 and the committed lockfile.

Passed: strict algo/train/simulator Clippy and formatting; **108 algorithm/physics tests**, **68 trainer unit tests**, **7 balancing controls**, **9 ordinary learning/evaluator controls**, and **45 live-simulator tests**. The separately ignored heavy historical learning tests were not invoked by those commands. The temporary native evaluation ABI adds four passing tests (72 trainer/ABI tests total, not an additional72 independent production tests). Both default and legacy WASM configurations compile; the legacy configuration still emits warnings, so this is not a warning-free claim for that feature set.

Independent physics checks cover implicit Lagrange equations and power balance, upright linearization, equilibrium/force sign, RK4 refinement and invalid physical parameters. The key integration regression runs the actual seeded live stepping wrapper against direct `PendulumEnv` for300 steps at EACH of three physical configurations. Every state, noisy observation, RNG cursor and episode boundary matches exactly, including true terminals and timeouts. Other tests cover nondefault pooled physics, activation versus hot publication, stale settings, copied-state observation invalidation and pending replacement snapshots.

The separate final-head PR workflows are recorded in the PR verification comment. A dedicated run passing is not a substitute for any still-pending platform/browser check.

## Learning retest

The implementation/retest protocol was recorded in #35 before execution (comment5736765786). The four existing development seeds201–204 each trained from random initialization for exactly **1,048,576 physical transitions**, with checkpoints0/65,536/262,144/1,048,576. No full-budget training run was retried, dropped or selected for a favorable outcome. Initial actor/critic bytes match the same-seed historical random initialization.

The existing ordinary Rust learner uses1x512 rollout, batch128,4epochs,2048 policy updates and32768 actor plus32768 critic Adam steps per run. No extra critic training was included. Python3.11, Torch2.8.0+cpu, NumPy2.2.6, Gymnasium1.3.0 and SB32.9.0 are pinned for inference/evaluation; SB3 performs no main-study optimization.

At every checkpoint, both current weights and the historical linear-trained weights are scored through the SAME corrected nonlinear Rust environment and the original common evaluation panels:32 noisy deterministic-policy1000-step episodes and64 noisy stochastic-policy1500-step episodes. Deterministic refers to policy sampling, not a noiseless plant. Both arms' initial evaluations are exactly equal within each seed process. Old linear-on-linear results are not substituted for this same-target comparison.

### Final checkpoint, all four seeds pooled descriptively

| Policy training source | Deterministic10s complete /128 | Deterministic mean return | Stochastic15s complete /256 | Stochastic discounted mean |
|---|---:|---:|---:|---:|
| Historical linear-trained, transferred to nonlinear target |50|490.024|98|77.347|
| Corrected nonlinear training and nonlinear target |90|682.834|184|78.697|

### Every final seed

| Seed | Deterministic completions old → new /32 | Deterministic mean old → new | Stochastic completions old → new /64 | Discounted stochastic mean old → new |
|---|---|---|---|---|
|201|7 →22|451.831 →704.253|14 →46|83.952 →84.928|
|202|4 →23|274.277 →692.081|6 →48|70.870 →80.608|
|203|21 →17|653.405 →549.823|39 →35|79.218 →70.298|
|204|18 →28|580.584 →785.178|39 →55|75.349 →78.954|

Seed203 remains an explicit counterexample to uniform improvement. All episodes and contrary results are retained. Across the final stochastic panels, angle-only endings decrease146→29 but position-only endings increase11→43; remaining coupled cart/pole control failures are not solved by environment consistency alone.

At65,536 transitions deterministic mean return is old120.398/new124.297; at262,144 it is old166.834/new275.203. At262,144 deterministic completions are0→17 and stochastic completions1→23. These are fixed checkpoints, not best-checkpoint selection.

Descriptive paired-seed99% Student intervals use n4,df3: deterministic mean difference+192.810, interval[-442.935,+828.554]; stochastic discounted difference+1.350, interval[-21.338,+24.038]. These broad intervals cross zero. Episode pooling is not a sample of128/256 independently trained agents, and the exposed development seeds are not held-out acceptance. The observed increase does not uniquely attribute every historical learning defect to physics mismatch; historical frozen witnesses were valid observations on their old task.

## Evidence and costs

Main study: **4,194,304 training interactions;1,105,508 evaluation interactions;3,072 evaluation episodes;131,072 actor and131,072 critic Adam steps;16,777,216 gradient-sample visits per network**. Preflight/test interaction and construction overhead are additional, not fully aggregate-instrumented. Historical training is reused evidence, not new training charged to this run. No speed or equal-compute performance claim is made.

Independent offline analysis checks the fixed recipe, all outcomes and panel keys, unique episodes, terminal labels versus recorded final states, timeout lengths, all checkpoint budgets, finite4545-float actor/critic snapshots, original initialization equality and evaluation-cost accounting. All32 old/new actor/critic checkpoint pairs are retained across four seeds and four checkpoints. **Thirteen positive/negative analyzer controls pass.** Reported returns are recomputed from exported episode outcomes; the study does not export every reward step or independently replay full Adam histories. The offline verifier does not execute the native library or re-simulate the task.

Early development failures are retained: migration of linear-dynamics RNG and negative-episode fixtures; a preparation-script formatting anchor; a zero-dt analytic reward fixture; the actual missing MPC-cache cleanup; and an optional-null versus absent-field browser assertion. The reward fixture retains its original exact score assertions, and learning thresholds were not loosened. All preceding audit attempts stopped before full-budget measurements. The initial browser run includes Playwright's automatic retry, not a retried training outcome.

Raw learning artifacts in run35402364550:201/10571585086;202/10571517950;203/10571497957;204/10571237825. Corrected build10571541845 and runtime10570699249. Archive digests and exact source bindings are retained in the evidence package. The package will include unchanged raw ZIPs, historical weights, failure logs, source, analysis/tests and replay instructions. Do not infer current deployed behavior from an unmerged PR or promise portable bitwise replay across different numerical platforms.
