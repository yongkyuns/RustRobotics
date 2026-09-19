# Repeated reward-bearing recovery learning: substantial gains, incomplete robustness

September 19, 2026. Issues #35/#33. Corrected nonlinear production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The fixed repeated-application screen passes.** After 512 consecutive updates, the recovery-data candidate improves both final co-primary scores over all three continued controls on every seed. Five-minute nominal completions reach **126/128 deterministic** and **124/128 stochastic**; sixty-second outward recovery reaches **110/128**. Every candidate survivor in those long panels also stays centred/upright throughout the final ten seconds.

This is not complete robustness: two deterministic nominal, four stochastic nominal, and eighteen long outward-recovery trials still fail. All final candidate failures are cart-position-limit failures. The short outward panel also has 51 failures out of 256. No candidate passes every evaluation case. These are four previously exposed, already-trained histories, not a fresh from-scratch or held-out cohort. No production default, PR38, master or deployed asset changes, merge, or hardware qualification occurred.

## Protocol and execution

[Protocol recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5742639165). Audit branch: `audit/ppo-repeated-recovery-20260919`. Successful executed commit: **4c1b5b3dd5ab13dd461ffd199247fa8608fde99c**. [Run35449208508](https://github.com/yongkyuns/RustRobotics/actions/runs/35449208508) completes all five jobs successfully: native preflight and four seed jobs. Each seed job executes all four continuation arms. No completed measurement is dropped or retried for a favorable result.

Each seed201–204 replays its ordinary lambda1 history through update8191, using4,193,792 main interactions, once. Four complete live sessions are then internally cloned from that ancestor: actor/critic modules and parameter identities, both Adam histories, environment and episode clock, current observation, all original RNGs, metrics, recent returns and unfinished episode return. This is not weight-only checkpoint loading. The original ancestor is fingerprinted and remains unchanged by its descendants.

The four arms each execute512 successive PPO updates and preserve all predefined checkpoints: incoming0, then1/32/128/512. Each adds262,144 main-stream interactions and8,192 actor plus8,192 critic Adam steps. Its final primary counter is4,455,936, which does NOT include the additional lookahead/supplement interactions described below. All arms retain the same actor-update cadence, not the same total simulator or computational budget.

The first mean-four and both union actor/critic endpoints reproduce the preceding one-update experiment byte-for-byte for every seed, as do the incoming networks. Thus the repeated study starts from the actual tested intervention, not a newly convenient learner state.

## Four fixed continuation arms

- **Ordinary:** existing lambda1 learner,512 main fitting rows,minibatch128,four epochs.
- **Mean-four only:** same fitting rows and optimizer steps, replacing the last unfinished-path bootstrap with the mean of four sampled512-step old-policy futures.
- **Near-union:** mean-four plus512 fresh near-state fitting rows, using1,024-row batches,minibatch256,four epochs.
- **Recovery-union:** same union recipe, except supplemental streams start at ordinary task resets rather than the main stream's post-collection state.

Each supplemental bank uses eight576-step frozen-old-policy streams. The first64 observations/actions per stream supply fitting rows; the remaining512 transitions supply future rewards only. Shorter selected paths are permitted only when a true terminal closes them. Failures are retained and subsequent resets use ordinary semantics. Raw advantages are normalized once over the union. Every update uses the actual existing Rust/Burn optimize loop. The prior KL-preservation intervention is inactive.

Network architecture2x64ReLU, learning rate0.0003, Adam epsilon1e-5, gamma0.99, lambda1 for these experimental histories, rewards, noise, stochastic exploration and nonlinear RK4 plant remain unchanged. Supported production lambda is still0.95. There is no supplied controller, startup critic pretraining, extra critic epochs, privileged network input, evaluator-return label, score-based update rejection or selected best checkpoint.

Supplemental domains vary by the fixed identifier `seed + 65536*(local_update-1)`. The first identifier matches the previous study for historical binding; later banks use new domains. Data generation and evaluation cannot consume the live trainer's RNG streams. Networks only receive the ordinary noisy observations.

Repeated operation explicitly handles boundaries. At a final live timeout, sampled futures start from the recorded pre-reset physical endpoint and noisy final observation, not the already-reset live episode. A final true-terminal target remains terminal; unused cutoff-tail sampling is skipped. Near supplemental streams use the current post-collection live state, while recovery streams use task resets. Native tests verify the timeout and isolation paths. No generated future or preserved physical state is claimed to be hardware-model-free experience.

**Near-union versus recovery-union is the strongest sampling comparison.** Both have the same prescribed supplemental collection size, fitting count and optimizer schedule. Comparing a union against a small-batch arm additionally changes normalization population, sample count and gradient-sample visits. The different reward/advantage distributions also change normalized sample influence, so the experiment does not uniquely isolate geometric coverage from all statistical effects.

## Evaluation panels

At every checkpoint, each policy receives64 paired trials per seed on noisy ordinary resets with deterministic actions, noisy ordinary resets with stochastic actions, and stochastic outward starts, each capped at2,048ticks/20.48seconds. At incoming0 and final512, additional panels contain32 trials per seed: nominal deterministic and stochastic30,000ticks/300seconds, plus stochastic outward6,000ticks/60seconds.

Outward starts place the cart0.8–1.2m off-centre, moving farther outward at0.4–0.8m/s, with angle within±0.15rad and angular velocity±0.3rad/s. These stress outcomes do not train or select the policy. A true failure ends a trial immediately; no automatic reset masks it. Deterministic refers to policy action selection, not a noiseless environment.

All arms/checkpoints share the prescribed keys within a panel. Distinct panels use different initial-condition/randomness keys; long trials are not extensions of specific short trials. Compare arms within a row or checkpoints within a fixed panel. Incoming evaluation is performed once per seed rather than counted as four independently trained policies.

Centred/upright success requires survival plus physical cart displacement at most0.5m and pole angle at most0.1rad throughout the final1,000ticks/10seconds. Full traces cover predetermined replication0 of each panel at incoming/final; the other trials retain aggregate outcomes.

## Final checkpoint results

### Short panels,256 trials per column

| Policy | Deterministic completions | Deterministic mean return | Stochastic completions | Stochastic discounted mean | Outward stochastic completions |
|---|---:|---:|---:|---:|---:|
|Incoming, before continuation|176|1409.386|178|80.160|108|
|Ordinary lambda1 continuation|166|1345.387|160|79.267|99|
|Mean-four only|169|1370.038|163|78.776|115|
|Matched near-union|199|1586.452|207|81.544|136|
|**Recovery-union**|**252**|**1984.608**|**251**|**85.697**|**205**|

These are final predetermined checkpoints, not selected best scores. Recovery-union improves both co-primary final scores over ordinary, mean-four and matched near-union on all four histories. Pooled counts are not256 independently trained agents.

### Sustained survival,128 trials per column

| Policy | Five-minute deterministic nominal | Five-minute stochastic nominal | Sixty-second stochastic outward |
|---|---:|---:|---:|
|Incoming|88|93|68|
|Ordinary continuation|77|83|62|
|Mean-four only|84|93|59|
|Matched near-union|100|109|81|
|**Recovery-union**|**126**|**124**|**110**|

All126,124 and110 candidate long-panel survivors also satisfy the final-ten-second centring/upright condition. This is stronger than merely surviving off-centre. It does not mean they stayed inside those tighter bounds for the entire episode or that their full attraction basin is certified.

### Every final candidate seed

| Seed | Short deterministic /64 | Short stochastic /64 | Short outward /64 | Five-minute deterministic /32 | Five-minute stochastic /32 | Sixty-second outward /32 |
|---|---:|---:|---:|---:|---:|---:|
|201|64|63|46|32|29|24|
|202|62|63|59|31|32|30|
|203|62|62|43|32|31|24|
|204|64|63|57|31|32|32|

Every failure in these final candidate panels is a cart-position-limit failure. No candidate survives all panels perfectly. In particular, long outward recovery still fails8/32 for seeds201 and203. Rail containment/recovery remains a material operating limitation despite strong nominal improvement.

## Learning progression and uncertainty

Candidate short deterministic completions progress176→179→194→227→252 at checkpoints0/1/32/128/512. Short stochastic completions progress178→179→196→223→251. Outward completions progress108→106→120→153→205: the initial small regression is retained, not removed.

The final candidate deterministic mean exceeds matched near-union on each seed by425.386,286.157,235.180 and645.902. Stochastic discounted mean improves by4.2172,4.4475,3.3168 and4.6320. The predefined paired-history99% Student intervals use n=4 histories,df3:

| Candidate minus control | Deterministic mean effect [99% interval] | Stochastic discounted effect [99% interval] |
|---|---|---|
|Matched near-union|+398.156 [-138.298,+934.611]|+4.153 [+2.451,+5.856]|
|Ordinary continuation|+639.221 [+158.213,+1120.229]|+6.431 [+1.861,+11.000]|
|Mean-four only|+614.570 [+313.072,+916.069]|+6.921 [+2.541,+11.301]|

These are pointwise descriptive intervals, not simultaneous or distribution-free safety guarantees. The deterministic effect against the strongest matched control remains imprecise despite all four positive observed differences. Fresh evaluation draws do not turn exposed training histories into a new held-out training cohort.

Earlier per-seed reversals remain visible: at update1 candidate seed203 is below the matched near control on both co-primary metrics; at update32 seed201 has lower deterministic return than near-union even though its stochastic score is higher. By update128 all four candidates lead that control on both scores. The checkpoint sequence supports accumulation on these histories, not a guarantee that every intermediate update is beneficial or that training indefinitely will preserve the gain.

The necessary advancement screen PASSES: positive mean co-primary effects versus matched near-union, no final seed regression on both scores, and no loss of pooled long completion counts relative to incoming. It is a continuation screen only. It is not absence-of-harm certification, held-out reliability, or permission for an unattended hardware deployment.

## Verification and retained failure

First preflight at02d92bae8143bd4db25d154fe5edac351c8b32fa/run35448962727 stopped before measurement because strict Clippy rejected a redundant clone of a Copy device handle. The preparation uses the equivalent copy; no algorithm, seed, criterion or budget changed. The failed83-member artifact remains unchanged. No full training run was retried to improve its result.

Repaired preflight passes **88 native unit/audit tests**, seven ordinary balancing controls, nine ordinary learning/evaluator controls, strict Clippy, formatting, compiled-runner discovery and protected-source restoration. Five heavy audit endpoints remain ignored during preflight; seed measurements explicitly invoke only the new repeated-learning endpoint. Two separately ignored historical heavy integration tests are not counted as executed.

New native tests cover complete-session cloning including environment clock and warm Adam, subsequent grouped/split agreement, original-loop parity for the ordinary arm, unique repeat sampling identifiers, evaluation and sampling isolation, and final-timeout pre-reset endpoint handling. Existing bootstrap/terminal/global-normalization and optimizer-anchor tests remain in place.

Independent offline verification checks all five successful ZIP digests,88 build members and3,288 measurement members; the83 failed-preflight members; and840 directly used nested historical recovery-reward members. Incoming actor/critic and first mean-four/near-union/recovery-union endpoints match historical bytes on every seed. It verifies **14,976 evaluation records**, **8,192 continuation-update records**, and **68 regular actor/critic checkpoint pairs**, with every outcome's planned identity, random-domain key, horizon and ending label validated.

Detailed numerical reconstruction covers the first,128th and512th update of every arm:24,576 primary rows,110,592 supplemental rows for return/support reconstruction, and768 recorded optimizer transactions with exact epoch coverage. Original and union GAE/raw advantages/normalization reproduce bit-for-bit with sequential float32 arithmetic. Maximum independent actor loss error is3.91e-8, value loss1.28e-5, and post-step mean3.81e-7. These are selected updates, not exports of all131,072 continued optimizer transactions.

All120 predetermined full evaluation traces, containing1,049,254 steps, are independently checked for nonlinear dynamics, action transformation, rewards, endings, return totals, force RMS and final-ten-second centring. Maximum errors are1.23e-7 dynamics,2.30e-7 actor inference,2.02e-6 command,1.41e-7 reward and9.81e-13 return total. The remaining evaluation episodes retain aggregates rather than complete trajectories. Supplemental GAE is reconstructed, but every supplemental physical transition is not independently re-simulated by this analyzer.

**27 independent analyzer tests pass**, including evidence corruption/missing/unlisted members, invalid panel identity/domain/length/ending, numerical decoding, normalization, weights and training-history statistics. No measured field or tolerance was altered. Full Adam moments, gradients and unexported RNG sequences are not independently reconstructed; native complete-state/next-step controls establish those preservation claims rather than bare weight snapshots.

Native execution uses Rust1.98.1/Burn0.20.1 and the pinned production lock. Offline Python/NumPy/SciPy versions are separately recorded. Offline replay never executes the archived native binary, contacts GitHub, or starts training. The full production native/browser matrix was not rerun for this audit-only branch.

## Compute and disposition

Shared original-history replay uses16,775,168 interactions. Across all sixteen continuation branches:4,194,304 additional main interactions,12,456,168 cutoff-tail interactions,18,874,368 supplemental interactions, and49,833,625 evaluation interactions. Continued actor and critic each perform131,072 Adam steps and25,165,824 gradient-sample visits. Generic preflight and additional inference work are not completely aggregate-instrumented and are additional.

Each union arm per seed consumes262,144 main interactions and2,359,296 supplemental interactions, plus up to1,048,576 sampled-tail interactions. Each original small-batch arm has the same8,192 actor steps, but fewer samples and much less data work. Tail lengths vary with failure; equal update cadence is not equal sampling or compute. This result does not establish sample-efficiency superiority.

Retain the exact recipe as a promising repeated-learning candidate. The next qualification boundary is a clean from-scratch implementation with fresh predeclared training/evaluation cohorts and sustained recovery acceptance, not a larger claim based on this warm-start continuation alone. A freeze/transfer into actual hardware is not justified by these remaining rail failures.

Evidence bundle: `rustrobotics-ppo-repeated-recovery-evidence.zip`. It contains all unchanged successful and failed raw archives, historical inputs nested inside, actual prepared sources/native runner, all retained targets/weights/outcomes/traces, independent analyzer/tests, numerical reports, hashes and offline replay instructions. Production lambda remains0.95; PR38/master/deployment remain unchanged and unmerged.
