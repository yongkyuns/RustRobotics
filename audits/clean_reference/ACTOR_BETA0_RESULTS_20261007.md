# Actor beta1=0: small endpoint gains, but learning is still unstable

October 7, 2026. Completed LOCAL four-seed development experiment, not a production correction.

## Decision

**The fixed endpoint screen FAILS. The additional late-retention screen also FAILS.** Removing actor first-moment carry from the beginning of training modestly improves pooled final completion, but one seed misses the endpoint target and another demonstrates substantial learn-then-degrade behavior before recovering. This does not justify a beta sweep or a failure-timed optimizer reset.

Protocol registered before outcomes: issue35/comment6034283959. Executed driver: **d94d499773f73494a8bfc78b68694ef694bc0d5c**, `audits/clean_reference/actor_beta0_compare.py`; Git blob2193d87a4139a487ddf224616d83797bc41c2135 matches the local source exactly. Preflight/execution binding: comment6034423945.

## One fixed configuration change, normal training throughout

| Setting | Reference | Candidate |
|---|---|---|
| Actor Adam betas | (.9,.999) | (0.,.999) |
| Critic Adam betas | (.9,.999) | (.9,.999) |

The two standard torch.optim.Adam parameter groups are installed once before learning. Beta1=0 removes actor first-moment carry while keeping second-moment adaptation and bias correction. It is not RMSprop, not default SB3 settings, and not a witness-timed clearing of exp_avg.

Both actor and critic train normally from random initialization. There is NO fixed/replayed critic trajectory in this experiment. PPO.train, collection and GAE remain stock SB3. All other settings are unchanged: one environment x2048 samples/update, minibatch64, ten epochs, joint gradient-norm clipping.5, lr3e-4, Adam eps1e-5, gamma.999/lambda1, tanh64x64 separate actor/critic, learned Gaussian std, ordinary resets, original native nonlinear physics/reward/noise and physical limits. Training cap256 remains an external value-bootstrapped truncation; it does not shorten evaluation.

No supplied controller, oracle critic, critic prefit, curriculum, KL guard, rollback, gradient surgery, reward change or selected checkpoint enters learning. The changed actor naturally changes later data and critic fitting; those changes are part of the normal from-scratch comparison.

## Complete endpoint results

Exposed development training seeds201–204. Each arm/seed receives exactly1,048,576 training interactions,512 updates,163,840 Adam transactions per parameter. Both newly paired arms use the same fresh evaluation domain11960000+training_seed:32 deterministic10s and64 stochastic15s ordinary noisy cases. The legacy evaluation score uses the original f32-promoted gamma.99, distinct from training gamma.999.

| Seed | Reference deterministic /32 | Candidate deterministic /32 | Reference stochastic /64 | Candidate stochastic /64 | Paired stochastic discounted effect |
|---:|---:|---:|---:|---:|---:|
|201|31|31|61|58|+0.953930|
|202|32|32|64|64|-0.257106|
|203|28|32|60|64|+0.552220|
|204|30|31|62|64|+0.368356|
|Pooled descriptive counts|121/128|126/128|247/256|250/256|+0.404350|

Pooled endpoint gains are five deterministic and three stochastic completions. Across the four trained-seed differences, mean discounted effect is+0.404350 with descriptive standard error0.252112. Four exposed training seeds do not establish population-wide superiority; pooled episodes are not independent trained policies.

**Seed201 blocks the primary screen:**58/64 stochastic completions is below61/64 and below its paired reference's61/64. Its deterministic count31/32 meets that absolute target. Seeds202/203/204 meet the per-seed endpoint conditions. The positive mean discounted-return condition passes. No threshold was changed.

Candidate endpoint failures are all position-limit endings: two deterministic failures (one each in201/204) and six stochastic failures (all201). This does not mean every earlier failure was exclusively position-related.

## Early learning and continued-training histories

All seven checkpoints were fixed before outcomes and retained. Each column pools four policies descriptively.

| Training interactions | Reference deterministic /128 | Candidate deterministic /128 | Reference stochastic /256 | Candidate stochastic /256 |
|---:|---:|---:|---:|---:|
|0|0|0|0|0|
|65536|12|10|5|5|
|131072|102|96|180|186|
|262144|128|111|255|198|
|524288|121|124|244|248|
|786432|126|124|251|249|
|1048576|121|126|247|250|

The final improvement does not imply uniformly better acquisition or retention. At262144 interactions, the candidate has198/256 stochastic completions versus the reference's255/256.

The separately registered late-retention screen compares each candidate endpoint count against its own respective counts at262144/524288/786432. Seed204 fails it: deterministic completion falls32/32 at786432 to31/32 at the endpoint, one same-case loss and zero gains. Other candidate seeds pass this count-based late screen. Count retention is not a claim of perfect same-case retention or monotonicity after every update.

## New counterexample: substantial degradation without actor first-moment carry

Candidate seed204's full history is especially informative:

| Interactions | Deterministic /32 | Stochastic /64 |
|---:|---:|---:|
|65536|8|4|
|131072|32|64|
|262144|28|34|
|524288|31|62|
|786432|32|64|
|1048576|31|64|

**Between131072 and262144, it loses four deterministic and30 stochastic completions, with ZERO gains on the same fixed cases.** It subsequently recovers. The30 stochastic failures comprise29 position-limit endings and one angle-limit ending.

Throughout this run actor beta1 is zero. Thus actor first-moment carry is not necessary for every observed learn-then-degrade event in this task. This does not invalidate the earlier local seven-step rescue: clearing an inherited direction at one selected point and eliminating actor momentum from initialization are different interventions. It does not identify the cause of this new interval or establish that critic momentum is the cause.

During that stochastic decline, mean duration falls15.000000 to12.312344 seconds while the legacy gamma.99 discounted evaluation score rises67.701460 to70.878102. This is NOT proof that the gamma.999 training objective improved; that separate objective was not independently measured on these evaluation trajectories. Survival and a short-discount score remain distinct metrics.

No individual damaging update has been localized in the new seed204 interval. Critic/bootstrap error, fitting-objective behavior, finite-step optimization and state coverage have not been causally separated there. **This no-actor-momentum witness is the next useful diagnostic target, rather than another beta/lr sweep.**

## Runtime, identity and validation

All eight main trainings completed once locally in four independent paired processes. Reference and candidate run sequentially inside each same-host process, with identical initial actor/critic tensors and initial evaluation records. Runtime: Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0 on AMD EPYC9V74. No hosted duplicate experiment, outcome-based retry, seed substitution or favorable historical control was used.

This is a NEW paired cohort, not an exact replay of prior Intel-host policies. The earlier cross-CPU historical replay limitation remains unresolved. These are exposed development seeds, not a fresh held-out cohort. Five-minute/outward qualification was not run after the failed screen.

The runtime ZIP7418edd50c11ae67ef761610a15c4a0c6ee70f2d3fa032609d6394eb745b5dd1 and31 payload hashes were checked before installation. The original native ZIP58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0 is included, with142 original manifest payloads verified and all extracted copies compared. Actual pinned PPO/collector/GAE/policy sources match the original archived versions.

**Sixteen live preflight groups pass.** New controls prove grouped-reference equality to untouched stock optimization (policy, named Adam moments/clocks, RNG and visible environment/episode state), identical first2048 pre-update trajectories,640 real candidate Adam steps with exp_avg equal to the current clipped gradient, observer/evaluator noninterference including checkpoint0, and disjoint actor/critic loss graphs. Original timeout-bootstrapping and independent forward-GAE controls remain intact.

The independent NumPy-only reader validates5,376 evaluation records,4,096 update/coverage rows,56 finite float32 network snapshots,56 exported optimizer snapshots, complete training episode boundaries, configuration/source/input identities and native-call counters. Every one of the4,096 updates still has at least eight episode starts, independently reconciled to the episode ledger.

**Thirty reader tests pass:** full-data acceptance plus29 corruption controls for cases/seeds/horizons/endings, configuration changes, budget/order, weights/moments/clocks and hashes. The reader itself imports no Torch, SB3 or target native library and does not unpickle. It does not reconstruct unretained gradients, per-step evaluation reward traces or private native RNG state.

Saved .pt snapshots include model/optimizer/Python-NumPy-Torch RNG and visible observations/episode state, but omit the native environment's private RNG/physics slots. They are NOT standalone exact-resume checkpoints. Their bytes are hash-covered; named optimizer arrays are checked without pickle.

## Costs and durable evidence

Main training:8,388,608 native transitions. Main checkpoint evaluation:5,208,031 calls. Preflight:40,964 training transitions and77,521 evaluation calls. Total new native calls:13,715,124. Earlier investigations' costs are separate. No wall-clock efficiency or lower sample-budget claim is made.

Evidence package **ppo-actor-beta0-evidence-20261007.zip**:22,946,485 bytes, SHA256 **3451603fa1b1d1218c26057b3567d64b18fcf1bd4f391d74616b9450a67cdbee**. A FRESH extraction verifies1,053 package payloads,872 measurement-manifest payload hashes,151 preflight-manifest payload hashes and the original native input; reproduces VERIFIED.json BYTE-FOR-BYTE; and passes all30 reader tests. It retains all paired measurements/checkpoints/optimizer arrays, exact executed sources, independent reader/tests, logs, native input, runtime receipts and detailed full report. Third-party runtime installers are excluded.

Offline replay: `PYTHONDONTWRITEBYTECODE=1 python code/REPLAY.py .` after extraction, with Python and NumPy. This runs neither simulator nor optimizer. Human report: `ppo-actor-beta0-results-20261007.md`; detached receipt: `ppo-actor-beta0-verification-20261007.json`.

**Fixed experiment COMPLETE; endpoint and late-retention screens FAILED; training correction remains UNQUALIFIED. No production default, reward/noise/reset law, protected source, production PR or deployment was changed. No merge. Issues33/35 remain open.**
