# C4 — epsilon-only from-scratch learning comparison

September 18, 2026. Parent cases: #35 / #33. Separate integration correction: PR #37 / #36.

## Decision

**Experiment complete; epsilon alone is not a qualified correction. No supported learner default changed, no new production trainer was added, and no merge was performed.**

Smaller epsilon improves final deterministic completion counts in both implementations, but results remain inconsistent across seeds and objectives. In Rust it worsens seed201's deterministic return and two seeds' stochastic discounted returns. In SB3 it increases pooled deterministic performance but worsens discounted stochastic return on all four seeds. No arm establishes reliable sustained balancing.

## Protocol and execution

[Protocol posted before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5734427307), at 18:27:43 UTC. Completed execution: **`b1100474d512f45856967c9c3b78be0fc6ee9e1b`**, [run **35381533937**](https://github.com/yongkyuns/RustRobotics/actions/runs/35381533937). All17 jobs completed successfully: one preflight and16 measurements. Successful measurement execution is not a learning-quality pass. No actual full-budget study run was dropped or retried to improve its score.

Source baseline: `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`, existing draft PR37, based on master `d724c517c27b182145d4054a66d3937563381dfd`. Audit branch: `audit/ppo-epsilon-20260918`. Committed production files, supported defaults, master and protected MuJoCo/control files remain unchanged by this study.

Four arms, each with exposed development seeds201–204: Rust/Burn epsilon1e-5; Rust/Burn epsilon1e-8; actual SB3 epsilon1e-5; actual SB3 epsilon1e-8. Every seed starts with identical random actor/critic weights across arms. One environment; rollout512; batch128; four epochs; two64-unit ReLU layers; fixed latent std0.1 with force20*tanh(z); gamma0.99/lambda0.95/clip0.2/lr0.0003/vf0.5/entropy0. No gradient clipping, extra normalization, annealing, target-KL stop, supplied controller, critic pretraining or artificial gate derivative.

Both implementations call the actual Rust PendulumEnv/plant. Rust selects epsilon only at untrained-session construction and uses ordinary persistent train_updates thereafter. SB3 2.9.0 uses the actual unmodified collector, GAE and PPO.train with C3's documented whole-rollout float32 normalization and latent-action environment adapter. Runtime native Adam configuration provides epsilon/betas. Epsilon is the only changed recipe setting within an implementation; random draws and floating reductions across implementations are not forced to match.

Each run uses exactly1,048,576 physical training transitions,2048 updates and32768 paired actor/critic minibatch passes. Checkpoints0/65,536/262,144/1,048,576 are all retained, with no best-checkpoint selection. Versions: Rust1.98.1/Burn0.20.1/current committed lock; Python3.11.16/Torch2.8.0+cpu/NumPy2.2.6/Gymnasium1.3.0/SB32.9.0. Actual sources and versions are archived.

Evaluation reuses C3's explicit independent-from-training, now-exposed panels:32 noisy deterministic-policy episodes capped at1000 steps and64 noisy stochastic-policy episodes capped at1500, per checkpoint. Panel keys/noise/action innovations are shared across arms and checkpoints. Evaluation preserves training/model RNG state and does not select updates. Deterministic policy evaluation still has noisy observations, action noise and disturbances. Stochastic discounted return uses gamma0.99, no critic bootstrap.

## Final checkpoint

| Implementation | Epsilon | Noisy10s deterministic completions /128 | Deterministic mean return | Discounted stochastic mean | Noisy15s stochastic completions /256 |
|---|---:|---:|---:|---:|---:|
| Rust/Burn current default | 1e-5 | 50 | 489.949 | 77.331 | 98 |
| Rust/Burn intervention | 1e-8 | 74 | 620.293 | 80.054 | 150 |
| SB3 corrected epsilon match | 1e-5 | 68 | 601.525 | 81.121 | 123 |
| SB3 original C3 epsilon | 1e-8 | 88 | 673.407 | 78.135 | 142 |

These are descriptive results for four exposed training seeds, not128/256 independent trained agents. The return columns use different discount/horizon definitions.

### All final seed effects, including regressions

Values are **epsilon1e-5 → epsilon1e-8**.

| Implementation | Seed | Deterministic mean return | 10s completions /32 | Discounted stochastic mean |
|---|---:|---:|---:|---:|
| Rust |201|450.151 →358.309|7 →5|84.003 →81.466|
| Rust |202|273.431 →551.569|4 →17|70.833 →78.947|
| Rust |203|656.259 →746.192|21 →25|79.296 →78.696|
| Rust |204|579.955 →825.104|18 →27|75.193 →81.106|
| SB3 |201|609.489 →627.430|19 →19|83.758 →81.714|
| SB3 |202|606.102 →546.573|19 →19|80.614 →79.122|
| SB3 |203|623.821 →706.794|17 →23|79.232 →71.727|
| SB3 |204|566.686 →812.829|13 →27|80.880 →79.977|

Rust seed201 is a direct counterexample to a uniformly beneficial epsilon correction. In SB3, smaller epsilon worsens stochastic discounted return on every seed despite higher pooled completion counts. These are measured objective/seed discrepancies, not proof of a specific underlying critic or policy mechanism.

### Predeclared paired effects, e8 minus e5

| Implementation / metric | Mean effect | Pointwise99% training-seed interval | Improved / regressed seeds |
|---|---:|---:|---:|
| Rust deterministic mean |+130.344|[-364.189,+624.878]|3 /1|
| Rust stochastic discounted mean |+2.723|[-12.165,+17.610]|2 /2|
| SB3 deterministic mean |+71.882|[-307.659,+451.423]|3 /1|
| SB3 stochastic discounted mean |−2.986|[-11.890,+5.917]|0 /4|

Intervals are descriptive Student intervals across four paired training-seed differences, df3; not pooled-episode intervals or familywise-adjusted simultaneous intervals. All cross zero. This is not evidence of no effect or equivalence; it is insufficient evidence for a reliable expected improvement or supported-default change.

### Earlier fixed checkpoints retained

| Arm | Deterministic mean:65,536 /262,144 /1,048,576 | 10s completions /128 | Stochastic discounted means |
|---|---|---|---|
| rust-e5 |121.225 /167.622 /489.949|0 /0 /50|64.143 /70.296 /77.331|
| rust-e8 |121.351 /204.632 /620.293|0 /1 /74|66.068 /71.457 /80.054|
| sb3-e5 |117.368 /215.820 /601.525|0 /10 /68|64.049 /71.749 /81.121|
| sb3-e8 |131.835 /265.383 /673.407|0 /11 /88|66.210 /73.626 /78.135|

At65,536 smaller epsilon regresses Rust seed201 on both co-primary metrics. At262,144 it regresses Rust seed203 deterministic return and seeds201/203 stochastic discounted return. SB3 has early/middle stochastic regressions on seed203/202, respectively. Full per-seed values and failures are in the retained summary/CSV.

## Correcting the earlier C3 interpretation

All original saved actor/critic checkpoints for the unchanged Rust-e5 and original SB3-e8 arms reproduce exactly, for all four seeds and all four checkpoints. Original evidence was not replaced.

The old mismatched-epsilon comparison had an SB3 deterministic advantage on all four seeds, mean+183.458. At matched1e-5, SB3's deterministic return is better on two seeds and worse on two; mean difference+111.576, pointwise99% interval[-387.159,+610.311]. At matched1e-8, the mean SB3-minus-Rust difference is+53.113 but only seed201 favors SB3; the other three favor Rust. All cross-implementation effects and intervals remain in the summary.

Epsilon was a real confounder, but correcting it does not establish implementation equivalence or repair sustained balancing. The fixed-tensor arithmetic/gate diagnostics and this from-scratch experiment answer distinct questions.

## Verification and limits

Passed: strict Clippy,73 native tests,7 balancing-diagnostic controls,9 learning-integration controls, compilation, formatting, protected-source restoration, original C3 reference controls and new epsilon/evaluation-isolation/Adam-history controls. Explicit1e-5 construction reproduces ordinary training and subsequent rollout exactly. Initial networks/first rollouts are unchanged by epsilon; grouped/split calls retain optimizer history. Ignored heavy tests are not represented as executed by these small integration controls.

Independent offline verification:17 complete main raw archives/657 member hashes;8 historical training archives/168 members; original C3 native archive/36 members;6,144 unique episode outcomes;64 actor/critic checkpoint pairs; exact optimizer/recipe/source identity, initial weights, portable layout, panel keys, ending labels and cost accounting. Fourteen independent analyzer tests pass. The offline verifier does not independently re-simulate unexported physical reward traces.

Initial policy evaluation is not byte-identical in two new-arm processes despite identical weights: Rust-e8/seed203 differs in73 initial episode records; SB3-e5/seed204 differs in62. Maximum initial undiscounted-return difference2.084e-5 and discounted-return difference4.601e-6; episode lengths and ending labels are identical in every case. These small differences are retained; their platform cause is not uniquely established, and numerical feedback may amplify small differences during training. No portable bitwise or hardware-matched performance claim is made.

Main costs:16,777,216 training interactions;2,144,990 evaluation interactions;6,144 evaluation episodes;524,288 paired actor/critic minibatch passes;67,108,864 gradient-sample visits per network. All main arms have equal training/optimizer budgets. Timings are retained, not used as speed comparisons.

Controls are additional: two completed preflights repeated the same controls. Newly added epsilon-specific native tests account for27,648 transitions per pass; explicit Python two-update checks account for6,144 per pass. Generic native/Gym/ABI tests and control evaluations also consume experience; they did not all have a complete aggregate counter. Main-budget totals are not misrepresented as including every test-suite interaction, and no efficiency claim is made.

### Preserved setup failures

Run35380740707 at086cd02611e0f34e3a0c4e92efe6c04f68ff05ce stopped at Clippy because an added ABI followed a retained test module. Run35381070680 at52fd0d3b00f247ce7f507382864ea2a43b6ccfa5 passed preflight, then all16 main jobs stopped before training because upload-artifact omitted a hidden workflow file listed in the manifest. Only audit layout/packaging changed; settings/seeds/budgets/evaluation/analysis did not.

Both unsuccessful native ZIPs still contain that known missing manifest member. Their raw archive and present-member hashes verify, but they are explicitly marked incomplete and were not repaired. All sixteen empty pre-training outcome IDs are mapped to six unchanged byte-identical raw ZIP groups. No full study run began in either failed setup; no training outcome was retried for a better score.

## Durable evidence

Self-contained conversation bundle: **`rustrobotics-ppo-epsilon-results.zip`**. It includes all unchanged main raw archives, nested original C3 evidence/checkpoints, exact executed sources/versions, all outcomes and weights, both failed setup archives/empty outputs, complete artifact ID/digest maps, detailed report, independent analysis/tests and `replay.py`. Retrieve it by name, not an assumed old runtime path. Main artifact IDs and ZIP hashes are in `artifacts.json`; known unsuccessful-package omissions are in `setup-artifacts.json`.

Offline replay requires only Python/NumPy/SciPy and does not execute the native shared library or start training. Analysis versions: Python3.13.5, NumPy2.3.5, SciPy1.17.0, separate from the pinned training environment. Full reproduction instructions are included.

## Production status

PR37's exact headd3f59f9 has now passed all required workflow groups, including the previously pending Windows tests; [verification](https://github.com/yongkyuns/RustRobotics/pull/37#issuecomment-5734661164). It remains draft/open/unmerged. That lifecycle/pooling correction is separate from long-run learning quality.

**No epsilon-only default change is promoted.** Remaining action-credit/value-estimation/representative-data problems in #35 require a bounded candidate and fresh held-out #33 qualification. This experiment does not establish that any one mechanism explains every failure. No new held-out seeds were consumed; #35/#33 remain unresolved.