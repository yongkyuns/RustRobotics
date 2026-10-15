# Independent CleanRL comparison: strong native balancing, real but non-universal regressions

October 15, 2026. Completes before-outcome issue #35 [protocol6201479730](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6201479730). Twelve new from-scratch runs completed once. No production change, merge, deployment, held-out qualification or issue closure.

## Main result

An independently implemented PPO learner can learn the actual noisy nonlinear Rust task well. With the native task's external timeouts correctly bootstrapped, CleanRL finishes with **256/256 difficult deterministic completions and251/256 difficult stochastic completions**. A literal port treating those timeouts as terminal failures finishes at **68/256 and42/256**, respectively.

Standard Gymnasium CartPole also reaches a perfect deterministic endpoint, but one run temporarily deteriorates from **64/64 to39/64** completed episodes and later recovers to64/64. Regression can occur outside our implementation, but collapse is not an unavoidable result on this task. The correctly adapted native runs have no substantial regression under the predeclared checkpoint rule, although two individual recovery cases are lost late.

This identifies an important **CleanRL-to-native timeout contract**. It does **not** retroactively diagnose the preceding SB3/Rust failures: the earlier SB3 reference already used timeout bootstrapping, and its recipe differs from CleanRL's. This is a useful independent baseline, not a recipe-matched implementation comparison or production reliability qualification.

## Final evaluation

Each count is out of256:64 fixed episodes for each of four separately trained seeds201–204. Gym uses its standard500-step/10-second cap. Native evaluation uses first physical failure or1500ticks/15seconds. Different horizons and rewards are not pooled into one score.

|Task and handling|Ordinary deterministic|Ordinary stochastic|Difficult deterministic|Difficult stochastic|
|---|---:|---:|---:|---:|
|Standard Gymnasium CartPole|256/256|228/256|Not tested|Not tested|
|Native, literal upstream timeout handling|121/256|115/256|68/256|42/256|
|Native, continuing-value timeout bootstrap|**255/256**|**255/256**|**256/256**|**251/256**|

Standard final mean return is500.000 deterministic and477.359375 stochastic. Native-bootstrap final mean physical returns are869.282927/868.995485 ordinary det/stoch and811.191732/800.022283 difficult det/stoch. Literal counterparts are430.000445/365.832444 and194.351208/106.219279. Definitive reductions use sequential float64 sums of physical rewards, not progress previews' float32 sums.

All native commands remain−20N/+20N; the unchanged nonterminal effort penalty is0.4 per step, and true failure gives−10. No reward, noise, disturbance or physical boundary was removed. Completion does not establish low-effort or hardware-safe control.

All final native seed-level counts, each /64:

|Seed|Literal ordinary det/stoch|Bootstrap ordinary det/stoch|Literal difficult det/stoch|Bootstrap difficult det/stoch|
|---:|---:|---:|---:|---:|
|201|58/58|63/63|38/30|64/62|
|202|0/0|64/64|0/0|64/62|
|203|63/57|64/64|30/12|64/63|
|204|0/0|64/64|0/0|64/64|

Every seed improves in each final native comparison. Four exposed development seeds do not establish population-wide reliability; evaluation episodes are not additional independently trained policies.

## Timeout distinction

Native training resets after256ticks,2.56seconds, even without physical failure. That is an external collection cutoff, not the end of the continuing task. Upstream CleanRL combines `terminated OR truncated` and zeros future value at both. The literal native arm intentionally preserves that mismatch. The bootstrap arm changes only this target before unchanged upstream GAE:

```python
if truncated and not terminated:
    training_reward += gamma * critic(final_observation)
```

GAE still stops at the reset and must not leak into the next episode. True physical failures never bootstrap; reset observations are never substituted for final observations. The simulator's physical reward remains unchanged: this is value bookkeeping, not reward shaping.

All other native conditions, initial tensors and budgets are paired. Actual timeout corrections per native-bootstrap run are1356/1452/1366/1556 for201–204. Every timeout-only training episode is accounted for. Short preflight learning did not itself reach a timeout; explicit target/boundary fixtures cover that condition before main execution.

The result isolates this correction **between the native CleanRL arms**, not the root cause of previous correctly bootstrapped SB3 regressions. CleanRL's successful recipe changes many other factors relative to that old reference; no individual hyperparameter is identified as the reason.

## All fixed checkpoint curves

All denominators256. No best-checkpoint substitution.

|Interactions|Gym det|Gym stoch|Literal difficult det|Literal difficult stoch|Bootstrap difficult det|Bootstrap difficult stoch|
|---:|---:|---:|---:|---:|---:|---:|
|0|0|0|0|0|0|0|
|32768|119|19|0|0|0|0|
|65536|163|22|0|0|0|0|
|131072|198|74|0|0|55|44|
|262144|256|247|188|146|253|247|
|393216|231|214|94|54|256|252|
|499712|256|228|68|42|256|251|

Native-bootstrap is not fast at the early checkpoints: most difficult recoveries appear between131072 and262144.

### Standard CartPole regression, with severity kept explicit

Gym seed202:

|Interactions|Deterministic completed/64|Deterministic mean steps|Stochastic completed/64|Stochastic mean steps|
|---:|---:|---:|---:|---:|
|262144|64|500.0000|59|479.4375|
|393216|39|464.4375|38|409.921875|
|499712|64|500.0000|51|452.40625|

At the middle checkpoint25 deterministic cases hit the cart boundary. Mean survival remains high: this is a substantial completion-rate regression, **not near-zero reward or immediate loss of balancing**. The deterministic policy recovers all cases. An early regression also appears in Gym seed204.

Literal-native seed204 drops from62/64 ordinary completions in both modes at262144 to0/64 at the final checkpoint. Seed201 difficult deterministic goes64→35→38. These are much more severe and should not be labeled unavoidable PPO instability when the target correction greatly improves the same learner.

### Native-bootstrap retention is strong, not perfect

Predeclared descriptive substantial regression: a later fixed-checkpoint completion drop≥20percentage points after reaching≥90%. No native-bootstrap run triggers it. It is not a deployment acceptance criterion and does not inspect between-checkpoint policies.

Between393216 and499712, two native-bootstrap cases are lost, both seed201: ordinary deterministic key4401142644 and difficult stochastic key4401243101. Both failed at262144, recovered at393216, then failed again. All smaller gained/lost keys are retained. Ten flagged substantial checkpoint pairs across the other arms overlap; they are not ten independent collapse events.

## Independent learner and recipe

CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, `cleanrl/ppo.py` blob `edde76baef51f05a8654160dfb19a4fdb34d14d3`, is retained byte-for-byte, SHA256 `d8dfbb7ac0b21e747b77d6e6673e566db4d6cdacc83bcebcd19cf87ad0e459ae`. Args/Agent/layer initialization, GAE and the entire PPO optimizer loop are preserved and checked. Eight anchored integration edits add fixed arguments, compatibility, logging/evaluation hooks and the explicit target adapter. Exact source/diff/generator are in the delivered evidence bundle.

Default recipe:4env×128steps,512-row rollout,minibatch128,four epochs,gamma.99,lambda.95,entropy.01,value coefficient.5,value clipping.2,global norm.5,Adam epsilon1e-5,independent64×64 tanh actor/critic,original orthogonal initialization. LR **2.5e-4 with upstream linear annealing**, not previous SB3's3e-4 constant LR. No normalization, reward-unit rescaling, divided critic head, teacher, curriculum, symmetry, pretraining, safety override, rollback or extra optimization.

Requested500000 is floored by upstream to **499712 actual interactions**,976 complete rollouts and15616 Adam transactions per run. All checkpoints preserve Adam history. Final LR≈2.5614754e-7 remains positive but tiny; this does **not** test indefinite training at a substantial nonzero fixed rate.

Gymnasium1.3 defaults to NextStep autoreset, whereas this collector assumes SameStep. The adapter explicitly chooses SameStep, verifying one physical transition per lane/tick, and adapts only final-info logging layout. Optional Tyro/TensorBoard are replaced by fixed dataclass arguments/JSON. Evaluation uses64 fixed inference lanes and independent uniforms; policy, Adam, training RNG and live observations remain unchanged. All four initial full actor/critic states match across the three arms.

Native arms call the actual frozen Rust nonlinear library, not a Python physics reimplementation. SB3 is imported only by inherited source/archive-verification code; **it never trains these agents**. Old SB3 results are context, not a matched-recipe causal comparison.

## Verification

All12 main processes finish once, exit0, no retry or omission. **12 pinned-runtime controls and18 independent NumPy reader/genuine-fixture/corruption tests pass.** Gym and native with/without-evaluation controls each reproduce all51 policy/Adam/live-observation/RNG arrays and training ledgers.800 paired native command steps verify dynamics/noise/reset/ending parity (1600native calls).

Full reader:280groups,17920trajectories,9914817 evaluation transitions. It checks all original keys/initial conditions, first physical endings, zero padding, native float32 reward bits, command/force mapping, observation-noise bounds,84 checkpoint/source bindings, named Adam clocks and all976 updates/run with exact LR schedule. Independent Gym one-step dynamics maximum error4.440892098500626e-16.

9914815 commands match independent float64 policy calculation exactly. Two untrained native seed203 actions differ from float64 argmax because actual float32 probabilities are both0.5. They are the same initial ordinary key/tick in the two native arms. An independent pinned Torch network reconstructs the saved64-lane input and reproduces both actual actions. Casting independent probabilities `[0.4999999983002842, 0.5000000016997158]` to float32 also yields `[0.5,0.5]`, whose first-index tie break is the recorded left action. All concrete logits, arrays and bindings are retained.

The initial strict float64 assertion failure is retained. The reader reports both mismatches and requires the separate exact-runtime recheck; it does not silently waive mismatches or enlarge a global tolerance. No training data, policy or main run changed. VERIFIED.json records the remaining recheck requirement; INFERENCE_RECHECK.json completes it. Offline replay validates the retained receipts and exact tie proof, not a new Torch execution.

Fresh extraction of the full archive verifies **1085 payload hashes**,18 tests,all84 checkpoints,both51-array controls and all raw-path reductions, regenerating **six reports byte-for-byte**. Fresh compact extraction verifies808 hashes,18 tests,all saved case/retention/checkpoint reductions and the tie fixtures; it explicitly omits277/280 main raw groups. Offline replay executes no Torch, native code, optimizer or pickle. It does not independently reintegrate native dynamics/private noise. These checkpoint exports are not full native-simulator resumes.

## Provenance and costs

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/Gymnasium1.3.0 on AMD EPYC9V74; loader also checks SB3 2.9.0 source. Runtime ZIP c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11payloads verified, not redistributed. Prior discrete compact72f02a3fc40de473064d3967f2f5294bed3f461b26825853c5a295b24cb7c242 and270payloads/old offline tests verify before use; old tests are not counted as new.

Native ZIP58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0 and142payloads verify. Plant PR38 head95c670b9f4618a11dc5439b026caf385a39718c8; diagnostic bridge da6623ae4b5878ae5079bf05c18c3d327570bc16. Executed learner SHA256fdd7423d2b82d25725707f467a15b207fcb89a6b37c2882e5a9ae3aea1826dc5; harness f42c91e91f5b2eab8667793c28973d599c436a835292d5d45b7d301f0913a3d9. Both remain unchanged since before main outcomes.

Main training5996544 interactions/187392 Adam transactions; main evaluation9914817 transitions. Controls add32768 training+45813 evaluation+1600 mapping transitions and1024 Adam transactions. **Total15991542 physical transitions and188416 Adam transactions.** Inference rechecks add zero training/simulation. No main run retried/dropped.

## Delivered artifacts

- `ppo_cleanrl_comparison_evidence_20261015.zip`:370870048bytes; SHA256 `34389e5c801a7b7298781582e7a1fedbd64492ff3b2d36d60816eff878a9207a`.
- `ppo_cleanrl_comparison_report_code_20261015.zip`:16735893bytes; SHA256 `5eff6dede10983a88189f7b13fbb4ea2a89f9994eb50c42b92ee68f16e83d592`.
- Standalone Markdown report and detached fresh-extraction verification logs accompany the conversation deliverables.

Both bundles include complete executed source, upstream license/provenance/diff, original native input archive, all84 checkpoints, all case summaries and verification. Full adds every main raw path; compact retains three genuine fixtures. Runtime installations/fonts and duplicate native extractions are excluded. README distinguishes offline verification from optional expensive new training; results are create-only.

Only this report is committed on this audit branch at publication. The complete executable source and numerical records are delivered in the archives, not falsely claimed all committed or CI-integrated.

No production Rust/Burn categorical implementation, continuous-action correction, browser feature, master/default change, protected-source edit, PR merge, deployment or issue closure. We now have a strong independent native-task baseline; the next implementation comparison should preserve its exact successful recipe and episode-boundary semantics before changing individual factors.
