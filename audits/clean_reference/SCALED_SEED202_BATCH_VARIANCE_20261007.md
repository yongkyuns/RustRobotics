# Scaled update223 batch variance and reuse — October 7, 2026

Development diagnostic for RustRobotics issue #35. Protocol comments: 6038154165 and seed binding 6038187824. This continues the reproduced scaled-seed202 update223 witness from 6038103626. No production/default change.

## Question

The selected scaled update223 is numerically well-conditioned but the one realized 2,048-row rollout gives positive PPO credit to an actor displacement that is harmful under independent same-state continuation. Is the residual damage mainly caused by repeatedly reusing one noisy batch for ten epochs, or by finite-sample rollout variance itself?

## Fixed arms

All arms start from the same reproduced incoming update222 policy and warm Adam state.

1. **historical-reuse** — actual 2,048-row update223 batch, 10 epochs: 20,480 sample presentations, 320 minibatch transactions.
2. **one-epoch** — same actual batch, 1 epoch: 2,048 presentations, 32 transactions. Not budget-matched; measures whether reuse is necessary.
3. **fresh-pooled** — ten independent 2,048-row rollouts collected from the frozen incoming policy before any optimization, pooled and optimized once: 20,480 presentations, 320 transactions.

Fresh rollout seeds were fixed before outcomes as `15223202 + 100000*k`, k=0..9. Confirmation evaluation reuses the eight fixed domains `14220202 + 100000*k`, k=0..7.

The historical-reuse arm reproduces the locally reconstructed outgoing update223 policy **exactly**.

## Fresh-rollout credit variability

Directional credit is the incoming-policy likelihood-score projection onto the historical harmful update222→223 actor displacement.

Historical selected rollout:
- physicalized raw-GAE direction: **+0.073314**
- normalized-GAE direction: **+0.001761**

Ten independent fresh rollouts, raw directions:

`+0.07980, +0.48805, +0.05608, -0.02241, +0.05817, +0.05038, +0.25675, +0.04443, -0.01806, -0.04762`

Thus **7/10 are positive and 3/10 negative**. Their mean is +0.09456, but the batch-level two-sided 99% Student interval is approximately **[-0.0716, +0.2607]**. The normalized-direction mean is +0.000788 with approximate 99% interval **[-0.000847, +0.002422]**. Therefore ten rollout realizations are still insufficient to establish the expected sign.

The pooled 20,480-row descriptive direction is:
- raw: **+0.094557**
- normalized: **+0.001103**

Rows within rollouts are correlated, so the pooled row count is not treated as 20,480 independent samples.

## Optimizer displacement

| Arm | Adam transactions | Actor displacement norm | Cosine to historical harmful displacement |
|---|---:|---:|---:|
| historical-reuse | 320 | 0.14044 | 1.000 |
| one-epoch | 32 | 0.05163 | 0.317 |
| fresh-pooled | 320 | 0.09807 | **-0.074** |

Despite positive descriptive credit along the historical displacement in the fresh data, optimizing the pooled dataset produces an almost orthogonal/slightly opposite actor displacement. Nonlinear clipping, minibatch normalization, warm Adam state and joint actor/critic gradient clipping mean the final update direction is not determined by that one scalar probe.

## Fixed eight-domain evaluation

Relative to the incoming update222 policy:

| Arm | Deterministic | Stochastic | Paired stochastic lost/gained | Mean stochastic discounted change | Mean stochastic step change |
|---|---:|---:|---:|---:|---:|
| incoming update222 | 239/256 | 488/512 | — | — | — |
| historical-reuse | 239/256 | **478/512** | **10 / 0** | -0.07306 | -9.55 |
| one-epoch | 238/256 | **470/512** | **18 / 0** | -0.10493 | -17.23 |
| fresh-pooled | **240/256** | **485/512** | **4 / 1** | **+0.18754** | +0.38 |

The one-epoch arm is *more* damaging than the full historical ten-epoch update on these fixed panels. Therefore **repeated ten-epoch reuse is not necessary for this witness**, and the later epochs partly recover performance rather than monotonically amplifying the first-epoch damage.

The budget-matched fresh-pooled arm greatly reduces the completion damage and improves mean discounted return, but does not eliminate the reliability loss: four previously successful stochastic cases fail, while one previously failed case succeeds.

## Interpretation

This experiment rejects the simple hypothesis that the residual scaled collapse is caused specifically by ten-fold reuse of one batch.

It supports a narrower statement: **more independent rollout data substantially stabilizes the update**, because the matched 20,480-fresh-sample arm is much safer than 20,480 presentations of the single historical batch. But the ten independent rollout credits remain highly variable and do not yet establish whether the expected PPO credit for the historical harmful direction is positive, negative or near zero.

It also exposes a reliability/objective tension: the fresh-pooled arm improves mean discounted reward while losing a net three stochastic completions. Average return improvement is therefore not sufficient to guarantee near-perfect survival reliability.

## Preserved execution detail

The first evaluation reducer stopped after measurements were already fixed because it paired evaluator records by output order. The evaluator emits early failures before timeouts, so order differs across policies. No training, rollout or evaluation outcome was rerun for favorability. The reducer was corrected to pair by immutable `(mode, episode, seed)` identity and reused already-created files. The historical policy replay and all ten fresh rollouts were unchanged.

New diagnostic costs:
- fresh rollout interactions: **20,480**
- new candidate evaluation native transitions: **1,955,196**
- optimizer-only transactions: **672** (320 historical replay + 32 one-epoch + 320 fresh-pooled)

Existing incoming/historical confirmation panels were reused, not rerun.

## Next causal question

Estimate the **batch-level expectation and variance** of the historical displacement credit with a larger fixed set of independent 2,048-row rollouts, without any optimization. This distinguishes a genuinely positive expected PPO direction from a near-zero expectation dominated by finite-batch noise. Do not launch another hyperparameter sweep before that sign is resolved.

The original value-scale archive byte-identity limitation from 6038103626 remains unchanged. No production/default/master/deployment change or merge.
