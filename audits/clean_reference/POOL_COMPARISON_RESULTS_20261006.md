# Ordinary multi-environment PPO does not prevent the collapse

RustRobotics · October 6, 2026 · Completed development experiment

## Decision

**Reject 8 environments × 256 steps as the proposed clean fix for this tested recipe.** All four ORIGINAL paired jobs finished successfully on attempt 1, but eight environments produced worse endpoint completion than one environment on every training seed, in both deterministic and stochastic evaluation. The registered developmental reliability screen fails every per-seed condition and its positive-mean stochastic discounted-return condition. This is not a production candidate or held-out qualification.

This rejects this particular collection change, not all multi-environment PPO and not the broader coverage/generalization hypothesis. Neither arm is a reliably sustained-balancing solution. Four exposed training seeds do not establish population-wide superiority or inferiority.

## Frozen experiment and provenance

Protocol was registered before results in issue #35/comment 6023805256. Original Actions run: **37519334101**; executed source: **77f8624d419666ffdadbc00d7c73f0c5bb3aadd9**. Original preflight plus all four original pairs were retrieved; no training was relaunched, retried, extended, replaced, or selected during this continuation.

One environment × 2,048 steps versus eight independent environments × 256 steps. Each arm received exactly 2,048 transitions per update, 512 rollout updates, and 1,048,576 main training interactions. Each seed's two arms ran sequentially on the same CPU/job from exactly equal initial actor/critic tensors. Seeds 201–204 are exposed development seeds. Different seeds' workers need not share CPU models; no across-CPU historical training equality is claimed.

Both use the original native noisy nonlinear PendulumEnv and stock SB3 2.9.0 PPO: separate tanh64×64 actor/critic, learned Gaussian standard deviation, Adam epsilon 1e-5, learning rate 3e-4, gamma .99, lambda .95, minibatch 64, ten epochs, minibatch advantage normalization, clip .2, and gradient norm clipping .5. Reward, observation/actuator noise, physical limits, reset distribution and 5,000-step episode cap are unchanged. The read-only observer delegates to the original training method; no curriculum, oracle critic, extra critic fitting, imitation, LQR, rollback or checkpoint selection enters training.

Each checkpoint (0 / 65,536 / 262,144 / 1,048,576 interactions) retains 32 deterministic ten-second cases and 64 stochastic fifteen-second cases per seed. Fixed evaluation keys 920000+training_seed are shared across arms/checkpoints and separate from training. Failure stops the episode; completion means reaching the evaluation horizon without angle/position failure. These are ordinary reset evaluations, NOT the older separate outward-start or five-minute qualification panels.

## Endpoint: every original training seed

| Seed | Deterministic 1-env /32 | Deterministic 8-env /32 | Stochastic 1-env /64 | Stochastic 8-env /64 | Stochastic discounted return, 1-env → 8-env |
|---|---:|---:|---:|---:|---:|
| 201 | 20 | 2 | 33 | 3 | 74.216455 → 68.285400 |
| 202 | 14 | 10 | 38 | 6 | 78.437317 → 79.211108 |
| 203 | 11 | 7 | 20 | 8 | 70.726358 → 77.120814 |
| 204 | 25 | 8 | 46 | 0 | 80.605466 → 79.011310 |
| Pooled descriptive counts | 70/128 | 27/128 | 137/256 | 17/256 | 75.996399 → 75.907158 |

The eight-environment endpoint completes only **6.64%** of stochastic cases, versus **53.52%** for one environment. Deterministic completion falls from **54.69% to 21.09%**. These counts summarize four trained agents in each arm, not 256 or 128 independent training runs.

The fixed screen requires at least 31/32 deterministic and 61/64 stochastic completions in EVERY eight-environment seed, no per-seed completion loss versus one environment, and a positive mean paired stochastic discounted-return difference. All four seeds miss both absolute and retention requirements; the mean discounted effect is -0.089241. **Screen: FAIL.** No threshold was changed.

## Continued training still erases earlier success

| Main interactions | Deterministic 1-env /128 | Deterministic 8-env /128 | Stochastic 1-env /256 | Stochastic 8-env /256 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 65,536 | 116 | 88 | 228 | 139 |
| 262,144 | 109 | 107 | 215 | 190 |
| 1,048,576 | 70 | 27 | 137 | 17 |

Eight environments reached 107/128 deterministic and 190/256 stochastic completions at 262,144 interactions, then fell to 27/128 and 17/256. Every eight-environment seed loses completion from that middle checkpoint to the endpoint. The earlier checkpoint is not uniformly better: seed 204's eight-environment policy is initially weak, improves by 262,144, then regresses. Every prescribed checkpoint and casewise gain/loss is retained; no best-checkpoint stopping rule is proposed.

Descriptive endpoint 99% Student intervals use FOUR paired training-seed means, df=3, and are not multiplicity-adjusted. Stochastic discounted-return effect: -0.089241, interval [-15.092413, +14.913931]. Stochastic completion-rate effect: -0.468750, interval [-1.105423, +0.167923]. The unbounded t approximation can extend outside the possible rate-difference range; it is not a probability bound. The intervals are very wide and crossing zero does not establish equivalence. The clear failure of the fixed absolute screen does not depend on a significant population superiority test.

## What the retained diagnostics add

### More streams did not guarantee fresh recovery examples

Eight environments still had **273 / 220 / 175 / 185** updates with no episode-start observation for seeds 201 / 202 / 203 / 204. That is **853 of 2,048 updates (41.65%)**, compared with 772/2,048 in the one-environment arm. The collector continues each live environment across rollout cutoffs; it does not reset eight environments at each update. These counts reconcile independently against the complete per-stream episode-end ledger.

A reset-free batch can still contain recovery-like states, and a batch containing a reset can still have poor coverage. These are continuity measurements, not a binary label for sufficient coverage. Per-stream and pooled observation moments show genuine stream separation, but independent streams do not imply that every update contains enough representative recovery experience. Pooled variances were independently reconstructed from within-stream variances plus variance of stream means.

This comparison also changes cutoff frequency (2,048 versus 256 steps per stream), batching and random-draw assignment. It therefore does not uniquely separate state coverage from GAE/bootstrap effects or stochastic training-path differences.

### Local position feedback deteriorated even with eight environments

A new **post-result, exploratory, read-only** analysis evaluates all saved networks on the origin and eight symmetric axis probes. No simulator, native dynamics, training update or evaluator-selected fitting data is used. Define commanded mean force u(o)=20*clip(mu(o),-1,1); the following is its position derivative at zero observation, in N/m:

| Seed, eight environments | At 262,144 interactions | At 1,048,576 interactions |
|---|---:|---:|
| 201 | +2.265065 | -0.812738 |
| 202 | +2.305022 | -1.905433 |
| 203 | +3.358986 | +0.567207 |
| 204 | +2.744336 | -1.082124 |

Three reverse sign; the remaining derivative falls about 83%. This is concrete evidence that adding streams did not preserve this aspect of the feedback law. The origin need not be an equilibrium because the network may command nonzero force there. These local derivatives are not a global stability certificate, do not identify the responsible intervening PPO update, and do not prove that all collapse is caused by this derivative.

All 288 probes (4 seeds × 2 arms × 4 checkpoints × 9 probes) were checked against independent Torch autograd, central finite differences and float32 forward evaluation. Maximum derivative disagreement: 4.27e-14 versus autograd and 3.55e-7 versus finite differences; maximum float32 force difference: 2.78e-6 N. The exploratory probe analysis does not alter the registered screen or primary statistics.

### Discounted return can conceal the loss of survival

At the endpoint, stochastic discounted return is nearly unchanged on average (75.996399 → 75.907158), while completion falls 137/256 → 17/256 and mean duration falls **8.963242 → 5.243984 seconds**. Seeds 202 and 203 even improve discounted return while losing completion.

Eight-environment stochastic failures include **157 position-limit and 82 angle-limit endings**. This is not solely a cart-rail failure. The gamma=.99 objective discounts per .01-second tick: the future-reward weight is about .00657 after five seconds and .0000432 after ten seconds. This calculation illustrates why improved early behavior can coexist with a worse long-survival metric; it does NOT prove that discounting uniquely caused the learning regression, and is not a proposal to restart the previously tested gamma sweeps.

## Verification, costs and limits

The full offline reader verifies all five original artifact digests and **842 manifest-listed payloads**, with the original native input additionally hash-anchored and its extracted source/library copies compared exactly. It checks both initial networks within each pair, every saved actor/critic checkpoint, final policy/Adam identities and step counters, all 4,096 coverage/update records, the complete per-stream episode boundary ledger and **3,072 evaluation records**. An additional standard-library reducer independently reproduces all 64 panel counts/means and endpoint paired means/standard errors; maximum mean disagreement is 1.43e-14.

Tests: 26 primary reader/statistical/archive tests, eight supplementary feedback/moment tests, and nine actual-record mutation tests on EACH of four original pairs: **70 successful test executions, 43 distinct test methods**. The actual-record tests include missing data, altered budgets/weights/Adam counters/boundaries, duplicate updates and nonfinite weights. Setup's original six hosted control groups remain retained. A fresh unpack replay must verify package hashes and reproduce all four retained numerical/report outputs byte-for-byte; its receipt is supplied with the completed package.

The main experiment used **8,388,608 training interactions** and **2,250,565 checkpoint-evaluation transitions**. Each arm/seed used 163,840 Adam steps and 10,485,760 fitting-sample visits per network. Hosted preflight adds 24,576 short-control interactions; the prior separate local preflight adds another 24,576. Generic setup/inference overhead is additional. Equal fitting budgets are not a claim of equal wall-clock runtime. This continuation added **zero training and zero native/simulator transitions**.

Per-episode reward traces, every intermediate actor and full historical gradients/Adam moments are not exported. The verifier checks retained sums, identities, budgets and physical endings; it does not regenerate every unrecorded reward, random draw, gradient or optimizer update. Saved final Adam state is not an exact resume at an earlier checkpoint. The historical single-update conditional-credit result is not automatically attributed to these new eight-environment failures.

No production default, reward, noise, native model, learned deployment, protected MuJoCo/control file, master branch or production PR was changed or merged by this continuation. This is independent SB3 evidence, not an implementation or qualification of Rust multi-environment PPO. No new held-out training cohort, five-minute sustained-balancing screen, or production browser/platform matrix was executed. #33/#35 remain open.

## Next bounded question

Do not increase environment count blindly or replace the negative seeds. The next useful diagnostic is to localize an actual harmful **eight-environment** update and inspect its own training support and action credit: did recovery states remain absent, or did a batch containing recovery data still damage the controller? The changed cutoff/bootstrap exposure must remain explicit. That attribution requires an independently verified exact replay with warm optimizer state or newly captured actual update data; the four saved checkpoints alone cannot answer it. No new training or sweep for that question was launched in this continuation.

## Evidence inventory

| Original artifact | GitHub ID | SHA256 |
|---|---:|---|
| Preflight | 11437969566 | eed51d09fa54573842a7b7ca84af281d7e6baee9b95451547e58013c87f39b2a |
| Pair 201 | 11441435840 | 22ea3b48e60b8199131fd5379e859017cc7b5212d58bbb5c06cb5b52166ce86c |
| Pair 202 | 11440462006 | 8fc20aa2c3774fcb367282beea7a2c288db7b01029b214d74df8bb20d1e7cdc3 |
| Pair 203 | 11440262513 | 209c59490ac776b277bdea3fb3e28c632c2246305dbd56278dba6fb0ea960317 |
| Pair 204 | 11441077356 | 2329c8192a5a9d16da6d3fdec19f4e7b3e09655e74961cd84f934fd3032723cf |

The companion `ppo-pool-complete-evidence-20261006.zip` retains those unchanged archives, exact training/workflow sources, primary and supplementary offline readers/tests, setup receipts, every primary result and supplementary probe, validation logs and `replay.py`. See its README for offline reproduction. No old conversation sandbox path is required to recover the source artifacts from the identifiers above.
