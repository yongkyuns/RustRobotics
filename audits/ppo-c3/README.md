# C3 — independent PPO cross-check and confirmed coordinator defect

**Completed investigation, not an implementation-clear result or sustained-balancing qualification.**

Date: 2026-09-09. Parent cases: [#35](https://github.com/yongkyuns/RustRobotics/issues/35), [#33](https://github.com/yongkyuns/RustRobotics/issues/33). New integration issue: [#36](https://github.com/yongkyuns/RustRobotics/issues/36).

## Bottom line

Regressions and incomplete balancing are not unique to Rust/Burn: real Stable-Baselines3 PPO exhibits them on the actual Rust task. However, this does **not** clear our implementation or recipe. Recipe-matched SB3 performs better on final deterministic evaluation for all four exposed seeds, while stochastic discounted-return differences are mixed. Default SB3 learns much faster initially, then regresses. None of these runs qualifies reliable sustained balancing.

A separate production integration defect is confirmed in the **native simulator coordinator**: even one replica reloads its own weights after each tick, resetting both Adam optimizers. An isolated no-self-reload guard restores exact persistent-session equivalence in the tested call groupings. This cannot explain the earlier standalone-session witnesses, which bypass that coordinator. Multi-replica parameter averaging is also not ordinary vectorized PPO.

## Fixed experiment and actual implementation boundary

Production baseline: `d724c517c27b182145d4054a66d3937563381dfd`.

Main executed commit: **`a29060c3180cd603347fffafde98f0709054c9c2`**. [Run 34362298428](https://github.com/yongkyuns/RustRobotics/actions/runs/34362298428) completed all 17 jobs successfully. The [protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5603187361) was posted before execution. No main training job was dropped, retried or given an adjusted budget to improve results.

Four arms used exposed development seeds 201–204, all from random initialization. All initialization and 65,536 / 262,144 / 1,048,576 checkpoints were retained. Every final run used exactly **1,048,576 actual environment transitions**, with no best-checkpoint selection.

| Setting | Rust / matched SB3 | Matched vector SB3 | SB3 defaults |
|---|---|---|---|
| Environments × rollout length | 1 × 512 | 8 × 64 | 1 × 2048 |
| Actor/critic hidden layers | 64 + 64 ReLU | Same | 64 + 64 Tanh |
| Initialization | Identical Rust random weights | Same | SB3 orthogonal |
| Exploration/action law | Fixed latent std .1, `20*tanh(z)` | Same | Trainable Gaussian, clipped normalized force |
| Minibatch / epochs | 128 / 4 | Same | 64 / 10 |
| Adam epsilon | 1e-8 | Same | 1e-5 |
| Gradient norm clipping | None | None | .5 |
| Advantage normalization | Whole rollout | Same | Per minibatch |

All arms retain lr .0003, gamma .99, lambda .95, PPO clip .2, value coefficient .5 and entropy coefficient zero. No reward or observation normalization was added to the default SB3 arm.

Pinned external implementation: **SB3 2.9.0**, Torch 2.8.0+cpu, NumPy 2.2.6, Gymnasium 1.3.0, Python 3.11. Rust 1.98.1 uses the original Cargo.lock/Burn .20.1. Exact installed library sources and version manifests are archived.

The matched arms call unmodified SB3 `PPO.train` and `collect_rollouts`. A documented RolloutBuffer subclass calls SB3 GAE, then the Rust-style sequential-f32 whole-rollout normalization; per-minibatch normalization is disabled. Only random initial actor/critic weights are imported and exploration std is frozen. A latent-action adapter applies the bounded transform in Rust; its fixed-action Jacobian cancels in policy likelihood ratios and entropy weight is zero. Guarded finite latent bounds cannot silently clip an action. The stock arm instead uses SB3 defaults with normalized force [-1,1] mapped to [-20,20]N.

**The default arm is a recipe comparison, not a one-variable intervention.** Network initialization/activation, exploration, action parameterization, clipping, minibatch grouping and optimizer work all differ. CleanRL continuous-action PPO and implementation-detail literature were reviewed, **not executed**.

All arms use the actual current Rust `PendulumEnv` and linear plant through a checked native ABI, not a Python dynamics reimplementation. No supplied controller, critic pretraining, privileged-state input, new reward, changed timestep, modified termination or diagnostic oracle trains the actor. ABI controls compare direct Rust noisy trajectories, resets, boundaries, invalid-action atomicity, normalized-force transforms and training readout invariance.

Every saved native actor **and critic** checkpoint matches the original C1 baseline bytes exactly, for all four seeds. Both matched arms import identical native initialization. Online action/shuffle RNGs and floating reductions differ between Rust and SB3; this is **not bit-for-bit online collector/optimizer equivalence**.

## Paired evaluation and results

Each checkpoint receives 32 deterministic-policy noisy episodes capped at 1,000 steps and 64 stochastic-policy noisy episodes capped at 1,500 steps. New per-episode Rust reset/noise streams are independent from training and shared across arm comparisons. Gaussian innovations are also shared for policy comparisons. Evaluation preserves training RNGs and model state; no learning decision uses its scores.

A common Torch evaluator imports native portable snapshots with native-action inference parity checked at each native checkpoint. Action transforms and plant stepping remain Rust. Rewards accumulate in f64, stochastic returns use the f32 gamma .99 and no value bootstrap. New evaluation panels explain differing native completion totals from earlier reports despite identical native policy bytes.

| Arm | 10s complete at 65,536 /128 | At 262,144 /128 | At 1,048,576 /128 | Final deterministic mean return | Final stochastic discounted mean | Final 15s stochastic complete /256 |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged Rust/Burn | 0 | 0 | 50 | 489.949 | 77.331 | 98 |
| Recipe-matched SB3, one env | 0 | 11 | 88 | 673.407 | 78.135 | 142 |
| Recipe-matched SB3, eight envs | 0 | 0 | 92 | 665.293 | 80.418 | 164 |
| SB3 default recipe | 97 | 63 | 67 | 501.755 | 79.104 | 124 |

All four final training-seed results are retained, including failures:

| Arm | Seed | Deterministic complete /32 | Mean return | Stochastic complete /64 | Discounted mean |
|---|---:|---:|---:|---:|---:|
| Rust |201|7|450.151|14|84.003|
| Rust |202|4|273.431|6|70.833|
| Rust |203|21|656.259|39|79.296|
| Rust |204|18|579.955|39|75.193|
| Matched |201|19|627.430|28|81.714|
| Matched |202|19|546.573|24|79.122|
| Matched |203|23|706.794|43|71.727|
| Matched |204|27|812.829|47|79.977|
| Vector |201|20|666.542|39|85.785|
| Vector |202|24|662.996|50|80.183|
| Vector |203|31|832.998|52|79.460|
| Vector |204|17|498.636|23|76.243|
| Defaults |201|17|505.358|33|84.960|
| Defaults |202|26|704.951|52|80.900|
| Defaults |203|23|540.875|39|75.624|
| Defaults |204|1|255.834|0|74.932|

### Interpretation, without clearing the implementation

- Matched SB3 has better final deterministic return on all four seeds. Its mean paired advantage is +183.458, but the descriptive pointwise 99% training-seed Student interval is broad: [-99.648,+466.563]. Stochastic discounted return improves on two seeds and worsens on two; mean difference +.804, interval [-19.954,+21.561]. This is not a statistical equivalence test or proof of an implementation bug.
- SB3 defaults achieve 97/128 early completions versus Rust's 0/128, but later regress to 63 and finish at67. Seed202 goes23/32 →7/32 →26/32. Final default failures are all61 position-limit failures; seed204 finishes with only1/32 deterministic and0/64 stochastic completions. Reference failure is real, not hidden by best-checkpoint selection.
- Eight shared-policy environments are not a uniform repair: final seed204 deterministic return regresses relative to matched one-env SB3. These 8×64 fragments also differ in bootstrap boundaries from1×512 despite equal transition and optimizer budgets.
- Four already-exposed initialization seeds per arm are not new held-out acceptance; pooled128/256 episode counts are not independently trained agents. Native runs are exact training replays, not newly independent trials.

## Confirmed native coordinator defect — #36

Successful supplemental [run34363717769](https://github.com/yongkyuns/RustRobotics/actions/runs/34363717769), executed **`4c6b123da42bc9d9186aece5c79cd71dcb3599f3`**, compiles the exact coordinator/native implementation bodies with only test-only seeded constructors/readouts. Original blobs are verified before preparing the test.

`PpoTrainerCoordinator::tick()` averages and reloads shared state even with one executor; native executors always accept it. `load_shared_state()` reconstructs both networks and creates fresh Adam optimizers. Four one-update coordinator ticks exactly match an explicit self-reload-after-each-update control, not persistent training. Max actor-parameter differences from a persistent session are0 / .0030646324 / .006617504 / .008494914 after updates1–4.

One `tick(4)` matches persistent four-update training, while four `tick(1)` calls do not. In the isolated candidate, skipping averaging/reloading when there is only one executor restores **exact actor and critic equivalence for both call groupings**:

```rust
if self.executors.len() == 1 {
    self.refresh_summary();
    return;
}
```

The minimal patch is preserved in the evidence bundle and tracked in [#36](https://github.com/yongkyuns/RustRobotics/issues/36); it is **not applied to production or merged**. Permanent status/metrics, coordinator, native and browser qualification belong in the scoped fix PR.

This cannot explain #32/C1/C2 or the standalone Rust arm: those sessions bypass the coordinator and retain Adam state. Web readiness/busy scheduling differs; native every-tick behavior is not asserted for browsers without execution.

### Multi-replica averaging is a different algorithm

The coordinator initializes separate policies and averages their parameters rather than pooling trajectories collected under one policy. An executed counterexample uses the actual averaging function: two functionally identical ReLU actors with permuted hidden units both command15.231884N at one observation, but the averaged actor commands0N. This is a synthetic function-preservation test, not learned multi-replica performance; the actors do not enter training. Common initialization alone would not make independently trained weight averaging equivalent to shared-policy vectorized PPO.

## Source comparison and physical-time setup

SB3 explicitly says clipping alone cannot prevent large policy updates and offers an optional target-KL limit. Its default policy/optimizer setup differs materially from ours. The inspected CleanRL continuous-action implementation uses orthogonal/Tanh networks, learned Gaussian exploration, gradient clipping, observation/reward normalization and learning-rate annealing. Those source facts are not evidence that any individual feature is our root cause.

With dt .01 and gamma .99, the discount time constant is about .995s, and a reward ten seconds away has relative weight about4.32e-5. The explicit gamma*lambda residual trace time constant is about.163s. These are not hard limits on foresight: the critic carries information beyond the residual trace. SB3's strong early result under the same gamma also prevents treating short discount alone as an explanation of every failure; earlier gamma-only experiments remain relevant.

References: [SB3 2.9.0 PPO](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html), [collector](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/on_policy_algorithm.html), [policy](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/policies.html), [RL tips](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/rl_tips.html), [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/), [implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), [Engstrom et al.](https://arxiv.org/abs/2005.12729). Inspected CleanRL source blob: `b454521b6e4a08d4a2f5ecec405132776f5e79eb`; actual installed SB3 PPO source matches blob `33017cb3367ff7c035a7638d084628e78c92394f`.

## Validation, costs and evidence

Passed:59 Rust unit tests,7 balancing +9 learning integration controls, strict Clippy, release build, format/diff checks;9 Python ABI/reference checks;7 coordinator tests;10 independent offline validator controls. All17 main jobs and the repaired coordinator supplement passed. Execution success is not learning success.

Independent verification covers **19 new raw ZIPs and390 manifest-listed entries**, plus the4 original C1 baseline archives, all6,144 unique evaluation outcomes and128 actor/critic snapshot networks. Initial bytes/evaluations match across native/matched arms; native checkpoints match all prior bytes. Sources, native binary, SB3 code, physical ending labels, bounded reward-sum invariants, training episode counts and exact environment/optimizer budgets are checked. The complete summary reproduces **byte-for-byte after a fresh package unpack**.

Main measurement costs: **16,777,216 training transitions**,2,714,038 evaluation transitions,1,048,576 actor minibatch passes with corresponding critic work,92,274,688 gradient-sample visits. Per run, Rust/matched use32,768 minibatch passes and four visits per transition; defaults use163,840 passes and ten visits. Thus defaults get **5× optimizer passes and2.5× sample visits**, not equal compute. Timings are retained but not treated as a speed benchmark. Unit/control costs are excluded from these totals.

The initial coordinator harness compilation failure is retained: added constructor struct-update syntax attempted to move fields from a type implementing Drop. Field assignment after `Self::default()` repaired this setup-only error without changing production logic or criteria. No main training run was retried. The initial Gym checker uses the normalized-force contract; separate controls cover the very wide latent-action adapter. Exact committed versus prepared sources are retained.

Self-contained bundle: **`ppo-c3-results.zip`**, **6,102,645 bytes**, SHA256 **`01a413009cccfa147ab0565d9d6bfc9b564d8e8315f15977ef8dfabb5dd8db51`**. It contains all19 unchanged new raw archives including the failed setup,4 original baseline archives, exact compiled sources/native library/version provenance, independent analyzers/tests, every checkpoint/failure, full report and minimal guard patch. Artifact IDs/digests are in `artifacts.json` and `prior/artifacts.json`. Retrieve this named bundle from saved conversation files, not an assumed old sandbox path.

Offline replay from the unpacked bundle requires only Python/NumPy/SciPy:

```sh
python replay.py
```

It does not require network, Rust, Torch or SB3, and does not execute the native library. It also does not independently re-simulate every evaluation trajectory or reconstruct unexported full reward traces. New training execution requires the pinned source/toolchain/dependencies and original artifacts; arbitrary cross-platform bitwise replay is not promised.

## Correction decision and remaining scope

First fix #36's concrete single-replica lifecycle defect with permanent coordinator tests, without silently changing intentional external weight-transfer semantics. Treat multi-replica rollout pooling/design as a separate slice. These defects are not an explanation for all standalone learning failures.

For the standalone learner, the reference comparison leaves a real performance discrepancy but not a uniquely attributed numerical defect. A narrow next check is to force identical actual rollout tensors, actions and minibatch order through the real SB3 and Rust collector/update boundaries across consecutive updates, extending earlier arithmetic-only comparisons. Any recipe correction must then be trained normally from scratch, tested for durability and frozen before a new #33 held-out protocol.

**Master/defaults and protected MuJoCo/control files are unchanged. #33, #35 and #36 remain open. No production merge, merge-ready correction PR or solved-balancing claim.** Heavy frozen-v1 learning, full final-head platform/audit/numerical checks and rebuilt WASM/browser qualification were not rerun for this diagnostic branch.
