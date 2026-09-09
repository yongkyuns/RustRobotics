# PPO C2 — actual action-credit failures and optimizer effects

**Targeted investigation complete; no production fix qualified.** Coverage-only, lambda-one reward-to-go, removing the current critic baseline, and a fixed sixteen-Adam-step limit are not established corrections. [#35](https://github.com/yongkyuns/RustRobotics/issues/35) and [#33](https://github.com/yongkyuns/RustRobotics/issues/33) remain open. No supported-default change or merge-ready correction PR is claimed.

Date: 2026-09-09. Master baseline: `d724c517c27b182145d4054a66d3937563381dfd`. Executed experiment: **`4ad89a1ec47979e83f3b9eb15c8653d0f662625a`**. [Actions run 34356464551](https://github.com/yongkyuns/RustRobotics/actions/runs/34356464551) completed all five jobs and cleanup successfully. [Protocol recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5602445741). No seed/job was retried to obtain a favorable result.

## What changed and what was replayed

This closes C1's missing evidence: seed201/update516 is now analyzed on **its own captured batch**, not merely the original 512-step panel. At all four original witness checkpoints, unchanged prefixes were replayed from random initialization with real Adam history. Old actor/critic and unchanged next actor bytes match the original attribution artifacts. Joint seed201 next actor **and critic** match C1 exactly. Weight-only snapshot loading was not treated as exact resume.

Test-only preparation in disposable checkouts observes the existing collector and optimizer. It retains raw rewards, physical states separately from noisy observations, next observations, boundaries, bootstraps, raw/normalized advantages, targets, every minibatch permutation, and every intermediate actor. No loss/Adam reimplementation, supplied controller, one-time critic pretraining, new public trainer, privileged-state training or reward/noise changes were introduced. Production/defaults and protected MuJoCo/control files are unchanged.

Coverage-only uses lambda .95 and at least 16 episode endings. Batch sizes for seeds201/202/203/204 are 5,079 / 3,096 / 4,894 / 2,410, with 160 / 100 / 156 / 76 normal Adam steps. All sixteen endings in each are true terminals; none is an external timeout. The first trace can start part-way through the live baseline episode.

The optimizer controls are (a) the first16 ordinary Adam steps, which use the first2,048 gradient samples rather than all data four times, and (b) minibatch size ceil(N/4), four epochs: all data visited four times in16 Adam steps. The latter changes minibatch grouping/noise as well as count; it is not a pure step-count or equal-compute intervention.

Seed201 additionally retains C1 intermediate actors1/4/16/40/160, 1%/10% of final displacement, and L2-matched small moves along the initial full-batch ascent gradient. A same-batch control replaces normalized `(target - current value)` actor credit with normalized targets, preserving critic targets/updates, actual optimizer history and minibatch order. No diagnostic oracle forms that intervention.

Evaluation uses256 paired fresh reset episodes per actor and32 fixed evenly spaced own-batch states with32 paired continuations each. Joint seed201 also uses the original panel. First-action comparisons follow the old policy after changing only the first action; full-policy comparisons use the candidate throughout. Actual noisy task/rewards, gamma.99, horizon1500, no critic bootstrap in independently measured return. Diagnostic data neither train nor select actors.

## 1. C1's remaining action-credit failure is not future critic bootstrapping

The actual joint seed201 batch has **5,079 transitions, sixteen true-terminal endings and zero timeouts**. With lambda1, future-value/terminal-bootstrap terms cancel. Raw GAE, targets and normalization reproduce bit-for-bit independently. The original lambda-.95 future-critic attribution does not transfer to this different update.

On the joint update's own fixed state panel:

| Frozen actor | First-action discounted-return effect, pointwise99% interval | Reset full-policy effect, pointwise99% interval |
|---|---|---|
| Final joint update | -0.034570 [-0.039135, -0.030006] | -0.696546 [-1.060893, -0.332199] |
| 1% of final displacement | -0.000358 [-0.000409, -0.000308] | -0.004242 [-0.007910, -0.000574] |
| Initial full-batch ascent direction, L2 matched to1% | -0.001023 [-0.001574, -0.000472] | -0.034985 [-0.074013, +0.004043] |
| Same-batch current-baseline removal | -0.014147 [-0.016573, -0.011722] | -0.307983 [-0.532930, -0.083035] |

Even the small initial-gradient move is harmful on the prescribed own-state first-action panel; its reset interval at that size is inconclusive. This is finite-panel/finite-step evidence, not an infinitesimal or all-state policy-gradient claim.

### Recorded-action decomposition on the same own-batch panel

Project independently resampled recorded-action returns onto the sampled action scores in the actual final update direction. These are directional-score quantities, not the same units as the return effects above:

| Quantity | Raw score projection / conditional99% interval |
|---|---|
| Recorded raw advantage score | +0.211753 |
| Independently measured recorded-action credit | **-0.046137 [-0.056657, -0.035617]** |
| State-conditional mean return minus recorded critic, projected on sampled scores | **+0.317755 [+0.267816, +0.367693]** |
| Resampled continuation with original baseline retained | +0.271617 [+0.222936, +0.320299] |
| Original continuation draw minus its resampled mean | -0.059865 [-0.108546, -0.011183] |

The identity is reconciled: **-.046137 + .317755 - .059865 = +.211753**. Future-noise resampling alone leaves the preserved-baseline score positive. Whole-rollout normalization does not remove it: recorded normalized panel score+.026014, independent action-credit score-.005513, resampled baseline-and-centering-preserved score+.033167.

A state-only baseline still has zero action-score contribution in expectation over fresh actions. The finding is its substantial projection onto this particular finite correlated sample. Physical-state/noisy-observation conditional means are not identical to the observation-only critic's conditioning, so the residual is not proved to be solely a fitting defect removable by this feed-forward critic.

On the **full5,079-transition batch**, the direct reward-to-go projection is already +.078742 along the harmful final direction; current `-V` contributes -.027058 and centering -.000251 before scaling. Do not claim that deleting the current critic fixes a negative full-batch reward signal whose sign it reversed. The full recorded batch and fixed-panel conditional estimands differ. Baseline removal mitigates harm but does not repair it. Its full-policy effect on the original panel is positive(+.055705), while its own-panel and reset effects are negative; selecting only that old panel would falsely promote it.

## 2. Repeated optimization also worsens the actual clipped objective

Initial full-batch normalized gradient dot final displacement is+.006146 (cosine .225445), but later minibatch passes make the **whole-batch clipped surrogate worse**. This is the quantity to maximize; minimized policy loss has the opposite sign. It is not `last_policy_loss`, which stores the last minibatch's pre-step loss.

| Adam step | Whole-batch clipped surrogate | Unclipped surrogate | Mean latent-Gaussian KL |
|---|---:|---:|---:|
|0|+0.000000134|+0.000000134|0|
|16|+0.000259474|+0.000477065|0.001372784|
|40, first epoch|-0.000130748|+0.001879873|0.003020572|
|80|-0.002389940|+0.004533440|0.010165504|
|160|-0.009175100|+0.005336812|0.015174581|

Independent PyTorch-f32 and sequential-f32/direct-density calculations corroborate the final clipped value at approximately-.009175104. It is not a reconstruction-rounding artifact. About14.45% of recorded likelihood ratios lie outside[.8,1.2]. Correct loss/Adam arithmetic does not imply monotonic improvement of the full-batch objective.

The supplementary paired final-minus-first-epoch contrast is **-1.706742 [-1.831337,-1.582146]** on own-panel full-policy return and **-.694890 [-1.016319,-.373461]** from resets. Continuing these epochs worsens this particular actor. The first-epoch reset effect itself is inconclusive, not a selected winning checkpoint or proposed one-epoch recipe. A whole-batch objective guard would address this surrogate deterioration, not certify true improvement when initial credit/coverage are unreliable.

## 3. Coverage-only fails two of the four original witnesses

All entries compare with the old policy, not with an already harmful original update. Coverage-final intervals apply an approximate99% family-of-four Bonferroni correction to the predeclared primary reset tests.

|Seed/update|Original update mean|Coverage-final mean and corrected interval|First16 normal steps, mean|All-data16 steps, mean|
|---|---:|---|---:|---:|
|201/516|-1.937643|**-.639234 [-1.085710,-.192757]**|-.042676|+.011441|
|202/537|-.203054|**+.264401 [+.202608,+.326195]**|+.105851|+.016339|
|203/514|-.347448|**+.441028 [+.264667,+.617390]**|+.132302|+.203586|
|204/526|-.197676|**-.067194 [-.110828,-.023561]**|-.120334|-.093154|

Coverage helps202/203 but harms201/204. Seed204 remains harmful under both16-step controls: first16 pointwise99% interval[-.167334,-.073334], all-data16[-.135021,-.051286]. All-data16 mitigates201 reset harm to an inconclusive effect, but its own-panel first-action effect remains negative. Neither more coverage alone nor a fixed optimizer-count rule is a complete correction.

The prior C1 from-scratch result is retained unchanged: coverage-only's93/128 versus baseline57/128 ten-second completions is a development clue, with seed regressions and unequal whole-update overshoots—not held-out qualification. These new frozen-update results do not replace that distinct experiment.

## Verification, costs and limits

Passed: **68 Rust unit tests**, strict Clippy, **7 balancing +9 learning integration controls**, release compilation, formatting/diff checks; **10 independent Python controls**. Original and C1 checkpoint bytes, all14 captured batches, every minibatch/finite actor, and actual training dynamics/reward/boundary accounting were checked. GAE/targets/normalization match bit-for-bit. Actual Burn initial gradients match independent PyTorch-f64 component calculations; largest observed absolute component error is approximately5.27e-6. Both packaged summaries reproduce byte-for-byte after a fresh unpack.

Counts: **43,164 captured training transitions**, **89,600 complete unique diagnostic outcome records**, **3,779,228 training transitions including fourteen original-prefix replays**, **117,692 actor Adam steps including prefixes**, **14,255,380 additional diagnostic transitions**. Unit controls excluded. Prefix repetitions are not independent training runs. No new full from-scratch candidate or held-out acceptance cohort was consumed.

Pointwise intervals are conditional simulation uncertainty, not training-seed uncertainty or all-state guarantees. Fixed-panel uncertainty combines within-state draws. Only the four primary coverage-reset tests receive the stated family correction. Aggregate continuation outcomes are retained, not every continuation's entire reward trace; offline validation did not resimulate all Monte Carlo trajectories. No global/nonlinear/hardware stability claim follows.

### Packaging defect recovered explicitly

The C2 uploader omitted `include-hidden-files:true`, so each raw artifact lacks exactly `sources/.github/workflows/ppo-c2.yml`, although its original manifest lists it. The initial manifest check detected this. The file was recovered from the exact executed commit and verified against original SHA256 **`5b0ab881cd631a127e26907f2bf139067a854af4aef5f2d326939462304a4725`** and Git blob **`1005847efc328ab59496012a37560a2001fc260d`**. Raw ZIPs and measurements are unchanged; no training rerun. The bundle retains this source separately and restores it explicitly.

There are1,258 original manifest-listed files directly in the raw ZIPs plus five exact-hash restored entries: **1,263 verified original manifest entries**. Do not silently describe the original uploads as complete.

## Durable evidence and next correction boundary

Self-contained conversation bundle **`ppo-c2-results.zip`**, SHA256 **`ecc41dc122be0160541998713975d6c37e09e288146d47bc7a8acc8e48b20f1c`** (42,354,098 bytes). It includes the five unchanged raw C2 archives, complete unchanged prior C1/original attribution evidence, actual prepared compiled source, supplemental recovered workflow, independent analyzers/controls, all summaries, full REPORT.md, provenance and replay logs. Retrieve by name from saved conversation files rather than assuming an earlier sandbox path. Artifact IDs/published digests are in `artifacts.json`; the raw run is34356464551.

With Python/NumPy/SciPy/PyTorch, from the unpacked bundle:

```sh
cd ppo-c2-results
python replay.py
```

No network or Rust compiler is needed for offline replay. Do not use Python `-O`. New Rust execution requires the pinned toolchain, disposable checkout of the executed revision, original evidence and exact preparation/workflow; committed source alone is not the prepared executable source. A different numerical environment is not promised bitwise replay.

**Correction decision:** reject the simple C1/C2 changes as sufficient fixes. The candidate must address finite-sample action-credit reliability and degradation during repeated optimization, while checking the reset distribution. Ongoing return/baseline calibration with independent episode groups and whole-batch update monitoring is a mechanism-driven direction, not yet a validated recipe or permission to use diagnostic oracles for training. An actual corrective implementation, from-scratch comparison, frozen new-held-out protocol and full final-head qualification remain outstanding.

Master/defaults and protected paths remain unchanged. Heavy frozen v1 learning, full platform/numerical/audit checks and rebuilt WASM/browser qualification were not run on this experimental branch. **No production merge, merge-ready correction PR or sustained-balancing claim.**
