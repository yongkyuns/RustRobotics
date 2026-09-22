# Learned PPO baseline: whole-trajectory generalization pilot

**22 September 2026. Diagnostic only; no actor update or production change.**

The same 4→64→64→1 ReLU critic architecture can learn a substantially better
baseline from independent, ordinary-policy trajectories. On the exposed
seed41006/update4140 actor batch, that learned baseline changes the initial actor
ascent gradient from opposing paired action credit to agreeing with it. The
original batch's Monte Carlo baselines, returns and credit labels were not used
to fit either independent-data arm.

This is a frozen-policy, fresh-optimizer regression pilot, not a demonstration of
continual critic tracking, live PPO/Adam update safety, or sustained balancing.

## Fixed protocol and data separation

The protocol was posted in issue35 **before any critic fitting**:
https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5780498957

Source data are the prior frozen independent-futures run35735238503, source
`c49c5cd31625b2f7cf327e41e1a128a4d0b9c5bf`, and its embedded original paired-credit
archive. Only the N=1 layout is used, avoiding reused streams across layouts.
Replicates0–23 provide12288 fitting rows; replicates24–31 provide4096 holdout rows.
Each contains phases0–511 from one independently seeded nominal-reset stream.
Entire replicate trajectories are separated; no neighboring-row train/test split.
The first64 phases are reported separately as the early transient, not relabeled
as an independent outward-recovery policy evaluation.

Fitting labels are `frozen_V(observation) + exported_raw_lambda1_advantage`:
ordinary sampled long-return targets with the existing far critic bootstrap.
They are not independently branched value-oracle labels at the fitting states.
All native artifacts and337 nested payload hashes are verified before fitting.
Frozen critic reconstruction differs from the original recorded witness values by
at most6.6924e-5; this is a compatibility measurement on the original witness, not
a rigorous numerical bound on all other states. Independent NumPy/PyTorch
inference on all17408 train/holdout/witness rows passes the1e-4 check.

The original1024-row harmful-update batch is an additional, already exposed
recovery-containing diagnostic panel. Its eight-draw Monte Carlo baseline labels
and action-credit estimates are used only after fitting the independent arms.
Those labels retain the old far-cutoff bootstrap and are not exact values. All
fitting finishes before witness or holdout metrics are computed, so there is no
early-stopping or checkpoint-selection feedback path.

## Arms and budget

All arms begin at the exact incoming pre4140 critic parameter bytes, with the
same4→64→64→1 ReLU network. They use fresh PyTorch float32 Adam state, not the live
Burn optimizer: learning rate.0003, betas(.9,.999), epsilon1e-5, unscaled MSE,
minibatch256, no clipping/regularization/input or output normalization.
The fixed sampler seed is20260922. Sampling is with replacement.

- **Independent-uniform:** draw uniformly from the12288 independent fitting rows.
- **Independent-early-balanced:** exactly the same rows/targets, but each minibatch
  draws128 rows from phases0–63 and128 from phases64–511. The64-row boundary is
  inherited from the original recovery prefixes, not chosen from new results.
- **Original-union-refit:** an explicit in-sample control fitted to the original
  1024 observations and recorded returns. Its original-witness results are NOT
  evidence of generalization. It never uses Monte Carlo baseline labels as targets.

Each arm executes4096 critic optimizer transactions, or1048576 row visits.
Checkpoints0,16,64,256,1024,4096, all losses, all sampled indices, all checkpoint
weights and predictions are retained. The primary comparison is the final4096
checkpoint regardless of earlier results. No candidate or checkpoint was selected
by evaluation. Equal optimizer work does not mean equal unique-data counts.

**This is256 times the original16-transaction per-update critic budget**, applied
to a fixed policy with a fresh optimizer. It is not a proposed4096-step mandatory
pretraining stage. New simulator transitions and actor updates both equal zero.
The archived lambda1 train/holdout streams contribute36864/12288
simulator transitions respectively; only512 rows from each1536-step stream enter
fitting or holdout scoring. The source study also recollected the same paths for
lambda.95 controls; its full cross-layout collection cost remains3047424 executed
transitions, not just these selected lambda1 subsets. The existing paired-oracle sampling cost is inherited
from the previous diagnostic and is not hidden as zero-cost training data.

## Final results

Gradient alignment is cosine against the paired conditional-action-credit ascent
gradient on the original1024 observations/actions. Actor parameters, sampled
actions, original returns and global normalization remain fixed; only the value
baseline changes. These are INITIAL gradients, not16-minibatch PPO/Adam changes.

| Baseline | Independent holdout RMSE | Original witness value RMSE | Credit cosine |
|---|---:|---:|---:|
| Incoming critic | 2.72441 | 2.62237 | -0.50303 |
| Independent-uniform,4096 | 0.53640 | 0.70485 | +0.57075 |
| Independent-early-balanced,4096 | 0.54442 | 0.73002 | +0.79163 |
| Original-union-refit,4096 | 1.30874 | 0.32329 (in-sample states) | +0.05264 |

The original-union control's fitting RMSE is0.27911 against its recorded return
labels. Its lower original-panel error does not imply a better actor-gradient
direction. Independent-early-balanced also has slightly worse aggregate witness
RMSE than independent-uniform, but better credit cosine. This supports measuring
score-weighted error effects rather than treating critic RMSE as a safety metric.

### Disjoint-draw credit references and trajectory holdouts

| Independent arm | Credit draws0–3 cosine | Credit draws4–7 cosine | Heldout early RMSE | Heldout late RMSE |
|---|---:|---:|---:|---:|
| Incoming critic | -0.49792 | -0.50773 | 4.24072 | 2.43180 |
| Uniform,4096 | +0.57757 | +0.56435 | 1.04708 | 0.41498 |
| Early-balanced,4096 | +0.79692 | +0.78664 | 0.90029 | 0.47217 |

The draw halves are independent continuation replicates on the SAME exposed
states/actions, not new state coverage or training seeds. All9 original witness
groups and all8 independent holdout trajectories have lower whole-group RMSE than
the incoming critic for both independent arms. Their remaining errors are not
small enough to claim perfect calibration: the worst final original recovery-group
RMSE is1.5914(uniform) and1.9355(early-balanced).

Both independent arms pass the registered final local screen: original-witness
value RMSE below incoming, positive cosine against both four-draw credit references,
and lower pooled early and late holdout RMSE. No confidence-qualified population
claim follows from8 heldout trajectories or one exposed incoming actor.

## The16-step warning and nonmonotonic results

| Independent arm | Witness RMSE at16 | Credit cosine at16 | Witness RMSE at4096 | Credit cosine at4096 |
|---|---:|---:|---:|---:|
| Uniform | 1.59155 | -0.79027 | 0.70485 | +0.57075 |
| Early-balanced | 1.41181 | -0.84797 | 0.73002 | +0.79163 |

After16 fresh critic updates, RMSE is much lower than incoming2.62237, but the
resulting recomputed advantage gradient is MORE opposed to measured action credit
than the incoming cosine-0.50303. This is a counterfactual using recomputed
advantages; it is not a claim that the original live PPO loop recomputes them
mid-update or that its Adam history was reproduced.

All retained credit cosines, in checkpoint order0,16,64,256,1024,4096:

- Uniform: -.50303,-.79027,+.18195,+.85040,+.69166,+.57075.
- Early-balanced: -.50303,-.84797,-.41297,-.01867,+.84874,+.79163.
- Original-union-refit: -.50303,-.78505,+.36838,+.63458,+.71936,+.05264.

The original-union control's holdout RMSE worsens from.86731 at256 to1.30874 at4096
while fitting error keeps decreasing. This is overfitting relative to earlier
checkpoints, not a claim that its final holdout error exceeds the incoming critic.
The final checkpoint is retained; none of the better earlier checkpoints replaces
it. More epochs or a lower aggregate fitting loss is not by itself the correction.

## Interpretation and next boundary

This strengthens the explanation that the incoming baseline is underfit and/or
poorly generalized across trajectory regions: the unchanged critic architecture
can learn a baseline useful on a distinct trajectory panel. It does not isolate
coverage from optimizer history, label quality, or slow tracking of a changing
policy. Frozen-policy fitting removes that last source of nonstationarity.
Early-phase reweighting trades aggregate error against action-credit alignment;
this is not yet permission to alter the training distribution or default weights.

The next bounded check is the full native actor update with the original live
Adam moments/counters and exact fitting/minibatch history, replacing only the
baseline with these fixed learned candidates. Preserve the original critic target
and transaction controls and use independent before/after evaluation. The pilot
does not execute that check and makes no new policy-completion claim. A favorable
full-step result would still not qualify repeated from-scratch sustained balancing.

## Tests, repeatability, and reproduction

**26 new pilot tests plus26 inherited baseline-analysis tests pass.** Controls
cover exact initial network bytes/layout, independent NumPy/Torch inference and
MSE directional derivatives, fresh optimizer configuration, whole-trajectory split,
phase-balanced sampler, no evaluation-target access from the fitting API, strict
artifact hashes, invalid inputs, every saved sampler/budget, and all18 checkpoint
metric/prediction recomputations. The inherited actual-witness autograd check
matches NumPy gradients to2.00e-15.

A second complete execution, with the same fixed protocol and no outcome-based
changes, reproduces all18 critic checkpoints,6 sampling/loss payloads, report and
provenance byte-for-byte:26 checked files. This is a reproducibility rerun, not an
additional independent training result. No new native/browser/platform CI result
is claimed. Artifact-dependent tests skip explicitly when inputs are absent.

From this directory, with Python3.10+, NumPy and PyTorch installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py \
  /path/independent-futures-result.zip /path/independent-futures-build.zip --out results
CRITIC_RESULT=/path/independent-futures-result.zip \
CRITIC_BUILD=/path/independent-futures-build.zip CRITIC_OUTPUT=$PWD/results \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -v test_study
```

The sibling `../baseline_attribution/analyze.py` is retained at parent commit
`f1da5f1b801d13f7a9db3daa205779a3a0298f14` and is imported rather than duplicated.
The exact execution used Python3.13.5,NumPy2.3.5,PyTorch2.10.0+cpu.

Required archives from run35735238503:

| Artifact | ID | SHA256 |
|---|---:|---|
| independent-futures-result.zip | 10698150266 | `795f94bdfbd7246a0ad99ac0d0c9b9085c364abad948b2532dab3b4c48b09b3d` |
| independent-futures-build.zip | 10697715468 | `cceee910794ac637cb2e7ce4c01e976fb95635fb045fd383229683444913c1b7` |

The result embeds paired artifact10638379259, SHA256
`65e071b04fa23f550a8be4cc306c05ccfe7d76eec0dfc6e7e1bab21cb3698ece`.
The conversation bundle retains these original archives, source, tests, all final
and intermediate results, sampling indices, losses, logs and repeat receipt.
Artifact retention is finite; do not assume a prior sandbox path exists later.
