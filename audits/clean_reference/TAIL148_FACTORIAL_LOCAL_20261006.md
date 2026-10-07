# Update148 tail: clipping × inherited momentum

October6,2026, America/Toronto. Diagnostic evidence, not a qualified PPO correction.

## Decision

**The separate LOCAL conditional2x2 is complete. The exact HISTORICAL causal comparison remains blocked.** Clearing only inherited actor first moment once prevents the large local tail loss on the registered panels; removing critic-dependent actor clipping also helps, but is less effective. This does not validate periodically clearing Adam or a new from-scratch recipe.

Protocols before respective outcomes: issue35/comments6029110683 (historical2x2),6029163947 (hosted exact-prefix reconstruction),6029222792 (separate local conditional experiment). No historical equality gate was loosened.

## Historical failure and local scope

Original233MB evidence ZIP SHA256f48ccce10782a1c6dcdfb0b751f6d37b3784f4932b8bde82d412bc968bd365bc and all1483 payload hashes were checked. Preserved runtime ZIP SHA2567418edd50c11ae67ef761610a15c4a0c6ee70f2d3fa032609d6394eb745b5dd1 and all31 payload hashes were checked; runtime is Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0.

This container is AMD EPYC9V74; the previous accepted local replay used Intel Xeon Platinum8573C. Local strict replay failed the historical post313 tensor gate after313 original optimizer transactions, before interventions or new evaluation. CPU dependence is a lead, not an isolated cause. Historical outgoing Adam also differs.

Hosted run37559254921 at7ae60f2cbf603826a421236171c17c9bc99cbed6, attempts1–3, received six AMD workers. Each stopped before dependencies/training at the CPU gate. All six original failure archives are retained; no more host retries were requested. The committed exact-prefix driver0941aa1227adaf129f335874505d7ff1f1a6626e did not execute past that gate.

The separately registered local experiment restores the original incoming148 actor/critic/Adam/RNG and fixed actual2048-row batch, then executes unchanged stock PPO on the available host to obtain a LOCAL313 state. Local313 differs from historical313 in4716 parameter elements; local320 differs in4950. Maximum absolute parameter discrepancy is9.5367431640625e-7 at each. Small discrepancies are not accepted as exact identity.

A separate transport check evaluates the ORIGINAL saved313/320 policies on their original exposed panels. All768 numerical records per policy differ, but every episode length and ending agrees. Maximum absolute undiscounted-return discrepancies are0.0076749921 and0.0064060520. These records are not pooled into the new local panels.

## Fixed interventions and state controls

All cells start from exact clones of LOCAL313, then execute only original minibatches314–320. Training gamma.999/lambda1, original observations/actions/old log probabilities/advantages/targets, normalization, epochs and rewards remain unchanged. No training rollout is collected.

| Cell | Actor clipping | Actor first moment before314 |
|---|---|---|
|A|Joint actor+critic norm|Retain|
|B|Joint actor+critic norm|Clear once|
|C|Actor-only norm,max_norm=.5|Retain|
|D|Actor-only norm,max_norm=.5|Clear once|

Clearing zeroes ONLY actor exp_avg once. Weights, critic moments, actor second moments, clocks, betas, learning rate and epsilon remain identical initially. Subsequent momentum accumulation is ordinary Adam, not a permanently momentum-free optimizer.

Actor and critic parameters/gradients are disjoint in this MLP. Critic raw gradients must match the LOCAL stock tail, then its recorded scaled gradients keep every critic weight and Adam state exactly on that local original path. A/B actor coefficients use their actual current joint norm; C/D use actor norm. This deliberately controlled critic path is diagnostic, not a proposed deployable trainer.

A's unchanged-tail sham matches LOCAL stock policy AND full Adam state bit-for-bit after EVERY step. All cells' critic paths match exactly during execution. Independent exported-state checks confirm identical clocks/hyperparameters/second moments, only intended initial actor first-moment differences, and identical complete initial/final critic states.

## Complete fresh local panels

Eight new domains6960204+100000*k,k0..7; original32 deterministic10s and64 stochastic15s inference shapes/noise rules per domain. Common evaluation gamma.99 remains distinct from training gamma.999. All cells and cases retained; these are conditional policy comparisons, not independent trained seeds.

| Policy | Deterministic /256 | Stochastic /512 | Mean seconds,det/stoch | Mean stochastic discounted return |
|---|---:|---:|---:|---:|
|Common LOCAL313|248|486|9.889/14.590|74.987628|
|A joint/keep|140|194|8.123/10.714|74.296708|
|B joint/clear|252|499|9.913/14.752|75.388852|
|C actor/keep|226|454|9.450/14.119|75.088402|
|D actor/clear|237|481|9.720/14.536|75.530796|

A loses108 deterministic and292 stochastic completions relative to the common start, with zero gains. B instead gains4 and13, with zero losses against that start. Against A, B gains112/305 deterministic/stochastic completions, C86/260 and D97/287, with zero lost completions in each comparison.

Contrary evidence: C still loses22 deterministic completions and has38 stochastic losses versus6 gains compared with313. Combining both interventions is not the best survival result, despite D having the largest mean discounted score. The descriptive interaction D-C-B+A is-101 deterministic and-278 stochastic completions; ceilings and one conditional witness prevent treating this as a population factorial effect.

A has116 deterministic and318 stochastic position failures. B has4 deterministic position failures and11 position plus2 angle stochastic failures. C has30/58 position failures and D19/31. Physical-bound checks verify these labels.

## Optimizer mechanism and remaining confounds

Whole-batch actor-loss diagnostic uses negative clipped surrogate and whole-batch advantage normalization, lower better; it is not the identical objective of every normalized minibatch. Common313 loss is+0.000182899.

| Cell | Local actor-loss ascent steps /7 | Final whole-batch actor loss | Mean actor step norm |
|---|---:|---:|---:|
|A|5|+.015187507|.016331|
|B|1|-.001193899|.006541|
|C|0|+.005333048|.034127|
|D|0|+.005359746|.031778|

In A, clipping retains roughly2.1–11.0% of raw actor gradient; in C,26.2–81.0%. Keeping the same incoming momentum but removing the critic's actor-gradient suppression changes the fresh-versus-carried balance. Independent reconstruction checks actual Adam displacements, clipping, second moments/bias corrections and carried/fresh contributions within float32 rounding. No Adam arithmetic bug is established.

Clearing inherited momentum ALSO changes step magnitude: B's first step norm is.003976 versus A's.028807, with mean norms.006541 versus.016331. Direction correction is not isolated from effective step-size reduction. C improves despite approximately twice A's mean step norm, but still degrades versus313. Zero local actor-loss ascent does not prevent global loss increase or poorer control. The two interventions are not interchangeable or additively beneficial.

No critic-calibration or value-conditioning correction was applied. Holding its tail path fixed does not repair its earlier saturation/error or identify its influence over previous updates.

## Verification and limits

NumPy-only independent reader checks3840 primary case records,28 intervention transactions,28 post-step policies, eight exported complete optimizer states, fixed critic gradients/weights, actor clipping, norms, moment recurrence and Adam reconstruction. Live sham additionally checked complete policy/Adam after each step. Twenty actual-record tests pass: original acceptance plus corruption rejections for cases/domains/checkpoints, endings/nonfinite scores, ordering/clocks/clipping, gradients/weights/moments and critic paths. Ten separate Torch/NumPy fingerprint tests pass. Offline replay runs neither simulator nor optimizer and does not unpickle.

The initial reader assumed hosted rather than local trace schema and failed. Its source/log are retained; correcting schema interpretation did not alter data or numerical thresholds.

A monolithic evaluation command timed out after transport and start/A panels, during B. Completed results were retained; B/C/D were completed at fixed whole-panel boundaries, with unchanged policies/keys/shapes. A repeated313 panel matches all96 records exactly in a separate process. Interrupted extra B work was not fully recorded. This was an infrastructure interruption, not favorable-outcome selection; do not claim complete per-step retention or exact total cost.

Known completed native evaluation:4723622 primary+1755501 transport+122773 split identity=6601896 transitions. Additional interrupted B work is unknown within0–1024000. Training-environment transitions:0. Fixed-batch optimizer transactions:661=313 failed historical+320 local stock+28 factorial. Six hosted CPU failures add no training/optimizer work. Original prior experiment costs are separate.

## Evidence and next bounded question

Focused conversation package `ppo-tail148-factorial-evidence-20261006.zip`:11357192bytes, SHA256 **b97ace9a969f16af240a6976f43c9577be603c27efb22a2b3cd5e4fd88fcfdc5**. Fresh extraction verifies168 package payloads,15 original selected inputs against the retained original manifest, six unchanged hosted failure ZIPs/24 inner members, reproduces LOCAL_RESULTS_VERIFIED.json BYTE-FOR-BYTE, and passes all30 tests (20 NumPy reader tests plus10 separate Torch fingerprint tests). It includes full report, all local measurements/states, exact executed sources plus documented header/path-only portable variants, failures and replay instructions. It excludes the full prior233MB ZIP and third-party runtime installers; it cannot recompute that old outer ZIP hash from only selected inputs.

Reader command: `PYTHONDONTWRITEBYTECODE=1 python code/REPLAY.py .`. Detached receipt: `ppo-tail148-factorial-verification-20261006.json`; full report: `ppo-tail148-factorial-results-20261006.md`.

The narrow next control is matched step magnitude, to distinguish inherited-moment direction from the smaller steps caused by clearing it. Any practical separate-clipping/optimizer-conditioning recipe needs a new fixed from-scratch comparison and fresh held-out qualification. A selected seven-step rescue does not establish a reliable trainer.

**Historical causal experiment BLOCKED; separate local conditional experiment COMPLETE; training correction UNQUALIFIED. Master, defaults, reward/noise/reset laws, protected sources, deployments and production PRs remain unchanged. No merge. #33/#35 remain open.**
