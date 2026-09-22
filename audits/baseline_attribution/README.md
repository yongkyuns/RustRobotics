# Frozen PPO baseline attribution — 22 September 2026

## Result

**For the exposed seed 41006/update 4140, replacing only the incoming value baseline repairs the frozen actor-gradient direction. Replacing only the realized return does not.** This strengthens the baseline-error explanation substantially, but is not a production correction or a full PPO/Adam counterfactual.

The completed independent-futures study was retrieved from Actions run `35735238503`, source `c49c5cd31625b2f7cf327e41e1a128a4d0b9c5bf`. This analysis independently reproduces every published statistic from all 81,920 selected rows, then performs a new post-hoc baseline/return decomposition on the original 1,024-row harmful-update batch and its 8,192 previously recorded paired continuations. No new simulator transition, actor update, critic fitting, training run, or policy evaluation was performed.

## 1. Same-data baseline-only isolation

Keep incoming actor, original observations, recorded latent actions, and old likelihoods fixed. Let `R` be the original recorded target, `V_old` the recorded incoming critic, and `Q_mc`, `V_mc` the averages of the eight paired continuation estimates. The Q branch takes the recorded first action. The V branch independently samples its first action from the old policy; both follow the old policy thereafter. They retain their original common-random-number coupling and far-cutoff bootstrap.

The exact finite-data decomposition is:

```text
R - V_old = (Q_mc - V_mc) + (R - Q_mc) + (V_mc - V_old)
             action credit    return residual    baseline error
```

Each intervention uses the unchanged union normalization. Gradients below are **ascent score gradients at the incoming actor**, not the final 16-minibatch Adam movement. Alignment means cosine against the paired conditional-action-credit gradient on these same states/actions.

| Advantage used | All-eight-draw alignment | First four correction / last four reference | Last four correction / first four reference |
|---|---:|---:|---:|
| Original `R - V_old` | -0.503033 | -0.507725 | -0.497920 |
| Baseline only: `R - V_mc` | **+0.954288** | **+0.936534** | **+0.969625** |
| Return only: `Q_mc - V_old` | -0.590457 | -0.603505 | -0.576613 |
| Both: `Q_mc - V_mc` | +1.000000 by construction | +0.999774 | +0.999774 |

The split checks use disjoint continuation draws for correction and reference, but retain the same exposed states/actions. They are not independent training seeds, new held-out state coverage, or confidence intervals. The all-eight "both" result is an identity, not validation. All variants and both split directions are retained.

### Which component dominates?

For additive gradient accounting only, each component is centered and divided by the **original** advantage standard deviation, so component gradients sum to the original gradient. The intervention table instead separately normalizes each variant.

| Component | Scalar RMS | Gradient norm on original scale | Cosine against credit direction |
|---|---:|---:|---:|
| Conditional action credit | 0.017501 | 0.161858 | +1.000000 |
| Realized-return residual | 0.173179 | 0.279258 | +0.888082 |
| Value-baseline error | **2.622377** | **2.285301** | **-0.635625** |

Baseline error is about 150 times the action-credit RMS; its gradient contribution is about 14.1 times the action-credit gradient contribution. Its finite-batch direction opposes the measured credit direction. The return residual happens to align positively on this witness, so merely removing stochastic return variability does not resolve the harmful direction.

The original union has one 512-row primary trajectory plus eight 64-row recovery prefixes. Observation/action identity binds these labels to their actual source batches. **79.60% of the baseline-error variance is between these trajectory groups.** The trajectory-mean baseline component has norm 2.51909 and cosine -0.79070 against credit. This is a measured trajectory-offset effect, not simply a large aggregate critic loss. It does not establish why the critic became miscalibrated; sparse coverage versus tracking/optimization remains to be separated.

Projected onto the recorded full actor displacement, the three initial-gradient components contribute -0.00060631 (credit), -0.00071819 (return residual), and +0.01488051 (baseline), summing to +0.01355601. These are directional derivatives at the incoming actor, not the nonlinear performance change or a decomposition of Adam itself.

### Monte Carlo limitations

Per-row baseline-estimate standard-error RMS is 0.047443; paired-credit standard-error RMS is 0.005427. All 16,384 continuation branches reach the 1,024-step cutoff without true termination. Their old-critic bootstrap contribution is retained: baseline cutoff-term RMS is 1.15854. The paired Q-minus-V cutoff difference has only 2.90e-8 RMS, but this does not make the absolute value estimates exact. The recorded target and continuation estimator can also differ in horizon/cutoff handling; therefore `R-Q_mc` is called a return residual rather than purely stochastic noise.

State cloning is diagnostic only. The actor continues to use its original noisy observations; no privileged physical state is installed in a controller. No simulator-derived baseline is proposed as a supported runtime default.

## 2. Frozen multi-environment result finalized

The existing run uses 32 replicates per layout, 512 selected fitting rows, phase offsets 0–511, and 1,536 simulated steps per stream under one frozen actor/critic. Lambda 0.95 and 1 use identical observations/actions/old likelihoods according to the native checks. Increasing streams changes independent futures while keeping fitting-row count fixed; support simulation grows with the number of streams. The original run records 3,047,424 transitions across both lambdas and zero training updates.

| Streams | Signal/RMS deviation, lambda 1 | Signal/RMS deviation, lambda 0.95 | Trajectory-mean gradient norm, lambda 1 | Trajectory-mean gradient norm, lambda 0.95 |
|---:|---:|---:|---:|---:|
| 1 | 0.2311 | 0.6358 | approximately 0 | approximately 0 |
| 2 | 0.1558 | 0.5254 | 0.7097 | 0.2691 |
| 4 | 0.1664 | 0.4021 | 1.0635 | 0.4441 |
| 8 | 0.1277 | 0.2374 | 1.5746 | 0.7269 |
| 16 | 0.1438 | 0.1868 | 1.7430 | 1.5026 |

Here "signal" is the norm of the empirical mean gradient; the inherited field `mean_gradient_norm` means **norm(mean gradient)**, not mean(norm gradient). These descriptive estimates have finite-replicate noise and are not precision-qualified signal-to-noise parameters.

This study does **not** support increasing environment count alone as a fix at this fitting budget. Between-trajectory offsets survive global normalization. The zero trajectory-mean component with one physical trajectory is an algebraic consequence of centering, not evidence that the critic is calibrated. It does not follow that parallel collection with a larger fitting batch or better recovery coverage cannot help.

These new streams start from nominal resets and are not the original recovery-union state panel. Cosines against the historical recovery-union paired-credit gradient therefore do not independently measure estimator correctness on the new state distribution. The source study's better local lambda-0.95 statistics also do not overturn the rejected lambda-0.95 learning comparison in issue #35.

## 3. Verification and reproduction

Required original Actions archives from run `35735238503`:

| Artifact | ID | SHA256 |
|---|---:|---|
| independent-futures-result.zip | 10698150266 | `795f94bdfbd7246a0ad99ac0d0c9b9085c364abad948b2532dab3b4c48b09b3d` |
| independent-futures-build.zip | 10697715468 | `cceee910794ac637cb2e7ce4c01e976fb95635fb045fd383229683444913c1b7` |

The result ZIP embeds paired artifact `10638379259`, SHA256 `65e071b04fa23f550a8be4cc306c05ccfe7d76eec0dfc6e7e1bab21cb3698ece`. Artifact retention is finite. The conversation evidence bundle retains both original outer ZIPs, source, tests, logs, all 1,024 row-attribution records and summaries; do not assume its old sandbox path exists in a future session.

Run from this directory with Python 3.10+ and NumPy. PyTorch is required only for the optional independent autograd witness test:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze.py \
  /path/independent-futures-result.zip /path/independent-futures-build.zip --out results
BASELINE_RESULT_ZIP=/path/independent-futures-result.zip \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -v test_analyze
```

Local execution verified all **10 result + 16 build + 311 paired payload hashes**. All published frozen statistics reproduced to maximum absolute difference **6.47e-13**. Actual 1,024-row actor gradients for original, baseline-only and paired advantages matched independent PyTorch autograd with maximum absolute error **2.00e-15**. The additive gradient decomposition closes to **1.45e-15**. **26 tests passed**, including synthetic directional finite differences, ascent sign, row permutation, group decomposition, normalization, invalid input/provenance, paired-record identity and actual-witness autograd. Without `BASELINE_RESULT_ZIP`, the autograd witness is explicitly skipped, not silently counted as executed.

These are local Python analysis tests and a reproduction of completed native artifacts. No new Rust, browser, platform or production CI qualification is claimed.

## 4. Next bounded mechanism test

The immediate question is whether an ordinary learned critic, using independent ongoing-policy trajectory data, can reproduce this baseline repair without a Monte Carlo lookup at each actor fitting state. Measure critic calibration on held-out whole trajectories, not random neighboring rows or its own bootstrapped training loss. Then replay the exact live actor/Adam history with only the baseline changed, preserving original critic targets, observations, actions and minibatch order, and evaluate independently. Better initial gradient alignment is not sufficient to approve the full step.

This should be an ongoing critic-tracking intervention, not a supplied stabilizing actor, a mandatory separate pretraining stage, trajectory-mean advantage deletion, another reward/gamma sweep, or a default change justified by this exposed witness. Equal-data controls and additional compute must be reported. Repeated from-scratch and held-out sustained-balancing qualification remain open.
