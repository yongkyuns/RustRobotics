# Ongoing critic fitting with unchanged actor-update cadence

September 18, 2026. Learning issues: #35 / #33. Existing integration fix: PR #37.

## Decision

**The experiment is complete. Extra critic fitting at the original actor cadence is a promising component, but neither tested variant is a qualified reliable-balancing fix.** Final noisy deterministic completions increase from 50/128 to 90/128 with frozen targets and 85/128 with refreshed targets; stochastic completions increase from 98/256 to 188/256 and 187/256. Both variants still regress on seed203's deterministic and discounted stochastic returns. Frozen-target fitting also regresses on seed204's discounted stochastic return.

No supported default, actor/critic architecture, reward, reset/noise contract, or production branch was changed. No merge or new held-out qualification was performed. These results neither erase the earlier harmful-update witnesses nor establish that every failure was caused by critic undertraining.

## Protocol, source and execution

[Pre-execution protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5735757799).

Production baseline: `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`. Audit branch: `audit/ppo-critic-cadence-20260918`. Actual executed commit: **`45da2503e2586230e6ba337a85438f5a50011ca9`**. [Run **35391982059**](https://github.com/yongkyuns/RustRobotics/actions/runs/35391982059) completed all **13 jobs** successfully: preflight plus all twelve measurements. No training job failed, was retried for a better score, or was discarded. Successful execution is not policy-quality acceptance.

All three arms use exposed development seeds201–204, identical random initial actor/critic weights per seed, and exactly1,048,576 actual Rust-simulator transitions per run. Retained checkpoints are0/65,536/262,144/1,048,576, not selected best checkpoints.

### The isolated intervention

Every update first runs the existing one-environment 512-step collection and ordinary four-epoch PPO optimizer, with minibatch128. The current actor's stored advantages and old likelihoods remain frozen. The actor and critic are separate existing2x64ReLU networks; Adam epsilon1e-5, learning rate3e-4, gamma0.99/lambda0.95, objective, exploration and task settings are unchanged.

- **Baseline:** no extra phase.
- **Extra-frozen:** after ordinary optimization, fit only the critic for12 more epochs to the original rollout's fixed return targets.
- **Extra-refresh:** the same12 extra critic epochs, but recompute detached lambda-return targets before each extra epoch from current critic predictions and the already recorded noisy observations, rewards and path boundaries. True terminals zero the bootstrap; timeouts and live-buffer cutoffs use the final-before-reset observation.

Extra fitting retains the existing critic Adam state and uses a separate persistent fixed-domain shuffle RNG. It cannot update the actor or consume the original environment/action/actor-minibatch RNG streams. First-update actor equality, ordinary-baseline replay and grouped/split execution are tested. Original actor updates use the production optimizer unchanged, apart from observation hooks. The added critic kernel uses the same weighted loss and actual Burn optimizer, not a Python replacement.

This is **ongoing post-update value fitting**, not startup critic pretraining. It introduces no extra simulator data, supplied controller, privileged-state training input, fitted diagnostic coefficient, objective-based rejection or artificial ReLU derivative. Refreshed targets still use the old behavior rollout: they are not independent Monte Carlo labels or unbiased evaluation of the newly updated actor.

### Cadence and compute per run

| Quantity | Baseline | Either extra-fitting arm |
|---|---:|---:|
| Simulator transitions |1,048,576|1,048,576|
| Actor/policy updates |2,048|2,048|
| Actor Adam steps |32,768|32,768|
| Actor gradient-sample visits |4,194,304|4,194,304|
| Critic Adam steps |32,768|131,072|
| Critic gradient-sample visits |4,194,304|16,777,216|

The original actor cadence is preserved, unlike the previous4096-transition batch study. The extra arms use **four times the critic optimization work**, plus target/inference overhead. Equal interaction budgets are not equal compute, and this is not a speed/efficiency benchmark.

Execution versions: Rust1.98.1/Burn0.20.1 with committed Cargo.lock; Python3.11/Torch2.8.0+cpu/NumPy2.2.6/Gymnasium1.3.0/SB32.9.0. SB3 is an inference/evaluation holder here; all measured training runs use Rust/Burn. Actual sources, library and versions are archived.

## Final checkpoint results

The unchanged C3/C4 evaluator uses32 noisy deterministic-policy episodes capped at1000 steps and64 noisy stochastic-policy episodes capped at1500, per training seed/checkpoint. Deterministic describes policy action selection, not a noiseless environment. Stochastic discounted return uses gamma0.99. All panels are already exposed, and128/256 episode counts are not128/256 independently trained agents.

| Arm | Deterministic10s completions /128 | Deterministic mean return | Stochastic15s completions /256 | Stochastic discounted mean |
|---|---:|---:|---:|---:|
| Baseline |50|489.949|98|77.331|
| Extra-frozen |90|676.984|188|79.545|
| Extra-refresh |85|683.804|187|81.858|

### Every final seed

Each entry is deterministic mean / stochastic discounted mean; completion counts are shown separately.

| Seed | Baseline | Extra-frozen | Extra-refresh |
|---|---|---|---|
|201|450.151 /84.003|711.194 /84.622|791.726 /87.816|
|202|273.431 /70.833|730.040 /82.641|524.158 /80.653|
|203|656.259 /79.296|633.449 /77.218|626.549 /77.528|
|204|579.955 /75.193|633.256 /73.697|792.783 /81.435|

Deterministic completions per32 episodes: baseline7/4/21/18; frozen25/26/20/19; refreshed25/15/19/26. Stochastic completions per64: baseline14/6/39/39; frozen56/55/39/38; refreshed54/41/40/52. Seed order is201–204.

Both variants improve final deterministic return on3/4 seeds. Frozen improves discounted stochastic return on2/4; refreshed improves it on3/4. Refreshed is not uniformly better than frozen: deterministic return improves on only2/4 seeds. None reliably completes every horizon.

### Predeclared paired final effects

Effects compare each candidate with baseline. Pointwise99% Student intervals use four paired training-seed differences, df3, not pooled episode counts or simultaneous/equivalence guarantees.

| Candidate / metric | Mean effect |99% interval|
|---|---:|---|
|Frozen deterministic return|+187.035|[-444.005,+818.076]|
|Frozen stochastic discounted|+2.213|[-16.772,+21.199]|
|Refreshed deterministic return|+193.855|[-269.129,+656.839]|
|Refreshed stochastic discounted|+4.527|[-9.690,+18.744]|

All intervals cross zero. This is imprecise evidence from four exposed seeds, not proof of no effect. The large observed completion gains justify retaining the component, but not claiming reliable expected improvement or promoting a supported default.

### Earlier checkpoints and reversals

| Arm | Deterministic means at65,536 /262,144 /1,048,576 | Stochastic discounted means |
|---|---|---|
|Baseline|121.225 /167.622 /489.949|64.143 /70.296 /77.331|
|Frozen|166.834 /569.958 /676.984|71.298 /78.088 /79.545|
|Refreshed|172.312 /455.122 /683.804|69.286 /78.501 /81.858|

Frozen improves both early scores on all four seeds; at262,144 its seed203 discounted score already regresses. Refreshed has an early seed202 stochastic regression despite later improvement. All per-seed checkpoint scores and failure endings remain in the raw evidence and `results/checkpoints.csv`.

## Fitting targets is not the same as predicting independent returns

On frozen targets, the extra phase reduces its training-target MSE in8,101/8,192 updates. Nevertheless, later policy quality is not uniformly improved. Refreshing targets changes the target being fitted: against the obsolete original targets its error often increases, so that number cannot fairly serve as its current-target optimization score. Within each individual refreshed-target epoch, current-target MSE decreases in88,872/98,304 epochs; frozen-target epochs decrease it in83,642/98,304. No observed loss value accepts, rejects or selects an update.

The independent check compares the critic **immediately before and after the final extra phase**, evaluated on exactly the same subsequent frozen-actor trajectories. The actor is identical across this pair. Recorded rewards determine finite-horizon discounted reward-to-go; no critic bootstrap supplies the test label. Select noisy observations every25 ticks through475 while alive; survivors retain at least1000 subsequent recorded steps. Outward states satisfy abs(noisy cart position)>=0.5 and position*velocity>0.

| Seed | Frozen-target outward RMSE, pre → post | Refreshed-target outward RMSE, pre → post |
|---|---|---|
|201|7.391 →7.341|5.328 →5.285|
|202|4.636 →4.610|4.601 →3.850|
|203|10.403 →10.368|6.292 →6.409|
|204|15.275 →15.250|8.752 →8.785|

Frozen fitting makes small descriptive improvements on all four final own-policy panels. Refreshed fitting improves two and worsens two, including seed203. These are comparisons with noisy realized returns, not exact conditional values or validated action rankings. Whole trajectories contain correlated samples; no independent-sample significance claim is made for these RMSE changes.

**Do not compare columns as matched-state causal effects:** different trained actors visit different states and generate different future returns. Only each within-cell pre/post comparison holds the actor and observations fixed. Baseline pre/post critics are identical by construction. Pre-extra snapshots are retained at all selected checkpoints, but complete paired trajectory-error analysis was implemented only at the final checkpoint, not at every earlier checkpoint. This limits temporal attribution relative to the protocol's broader intended checkpoint check.

Extra fitting also does not eliminate coupled cart/pole failures. For seed203 stochastic evaluation, position-only failures increase from1 at baseline to11 with frozen targets and17 with refreshed targets, while angle failures decrease. This is descriptive behavior, not proof that the critic update caused a particular erroneous action.

## Verification and limitations

Preflight passed strict Clippy, **74 native unit/ABI/audit tests**,7 balancing controls and9 ordinary learning controls, build/format checks, source restoration, original reference controls and the new evaluation-wrapper/first-actor/grouping controls. The two separately ignored heavy integration tests were not run. No measurement retry or favorable checkpoint selection occurred.

Independent offline analysis verifies all13 current ZIP digests and1,255 manifest-listed members, the80 members of the nested preceding native-source archive,4,608 unique evaluation outcomes and48 actor/critic regular checkpoint pairs. It checks the prescribed budgets/versions/source bindings,2,048 update records per run,196,608 extra-epoch records in total, and24,576 selected rollout rows with all corresponding minibatch permutations and stored target arrays.

Original selected GAE returns and normalized advantages reproduce bit-for-bit in float32. Frozen extra targets remain bit-identical. Refreshed targets reconstructed using independent float64 network inference from saved float32 weights agree within maximum2.289e-5 absolute error, inside the unchanged offline comparison tolerance. This reconstruction is not exact Burn arithmetic, and full optimizer moments/gradients were not exported or independently replayed. The actual run's persistence and isolation are covered by its controls, not by weight files pretending to be exact-resume checkpoints.

Final recorded rewards reproduce episode undiscounted and discounted totals within6e-13. Offline reconstruction does not re-simulate the plant or recover unexported commanded forces. Independent critic inference checks use fixed-index samples across the trajectories; maximum discrepancy is2.493e-5. **Fifteen analyzer controls pass**, including corrupted/missing evidence, duplicated records, invalid panels/endings, GAE boundaries, normalization and training-seed statistics.

All original baseline actor/critic checkpoints AND evaluation records reproduce C4 exactly for all four seeds. Initial equal-policy evaluations differ slightly across three new-arm processes despite identical weights and panel keys: frozen203, frozen204 and refreshed203. The largest initial return difference is2.084e-5, discounted difference4.601e-6; durations and endings match. These differences are retained without reruns; the platform/numerical cause is not uniquely established. No cross-host bitwise guarantee follows.

Main costs: **12,582,912 training interactions;2,075,976 evaluation interactions;393,216 actor Adam steps;1,179,648 critic Adam steps;50,331,648 actor gradient-sample visits;150,994,944 critic sample visits.** Generic preflight/test experience is additional and not all aggregate-instrumented; no all-inclusive cost or speed claim is made.

## Evidence and next boundary

Self-contained conversation bundle: **`rustrobotics-ppo-critic-cadence-evidence.zip`**. It contains all13 unchanged current raw archives, four unchanged C4 baseline archives, exact prepared sources/library/versions, all outcomes/checkpoints, independent analyzer and15 controls, full summaries, artifact digest maps and offline replay instructions. Offline analysis requires Python/NumPy/SciPy but never executes the archived native library or restarts training. Analysis used Python3.13.5/NumPy2.3.5/SciPy1.17.0, distinct from the pinned training environment.

Keep master and supported defaults unchanged. This experiment supplies a promising critic-only component at preserved actor cadence, not held-out sustained-balancing qualification. The next narrow question is why the seed203 recovery/action-credit failure remains under both variants, rather than assuming additional fitting or lower training MSE proves correct action credit. Fresh fixed-state counterfactual validation and subsequently predeclared held-out training remain separate work; neither was performed here.
