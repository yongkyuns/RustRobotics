# Scaled update223 expected-credit estimate — October 7, 2026

Development diagnostic for RustRobotics issue #35. Fixed protocol: comment 6038304583. This is a read-only rollout/credit estimate after the scaled seed202 update223 attribution and batch-variance test. No optimizer transaction or policy update occurs in the primary experiment.

## Question

Does the historically harmful update222→223 actor displacement have a reliably positive expected PPO first-order credit under the incoming update222 on-policy training distribution, or is the observed direction dominated by finite 2,048-row rollout noise?

## Fixed design

- 64 new independent rollouts.
- 2,048 transitions per rollout.
- Frozen incoming scaled update222 actor and critic.
- Seeds/domains: `16223202 + 100000*k`, k=0…63.
- gamma=.999, lambda=1, value scale S=1/(1-gamma), 256-step training timeout, unchanged plant/reward/noise/reset law.
- Zero optimizer transactions.

Primary statistic: first-order clipped-surrogate direction at the incoming policy along the fixed historical update222→223 actor displacement. Since ratio=1 at the incoming policy, clipping is inactive. Each rollout uses one fixed permutation `RandomState(26223202+k)`, split into 32 minibatches of 64, with SB3-style within-minibatch advantage normalization. The 32 minibatch directional means are averaged to one independent rollout estimate.

Primary inference was fixed in advance: mean over 64 rollouts with a two-sided 99% Student interval across rollouts. No extension after seeing the interval.

## Primary result

**Minibatch-normalized first-order PPO direction:**

- n = 64 independent rollouts
- mean = **+0.00022156**
- SD = **0.00181076**
- SE = **0.00022634**
- preregistered 99% interval = **[-0.00037964, +0.00082277]**
- range = **[-0.00570745, +0.00426666]**
- signs = **37 positive / 27 negative**

The interval includes zero, so under the fixed interpretation the expected sign remains **unresolved**. The mean is small relative to rollout-to-rollout variability.

Secondary directions agree with that conclusion:

| Statistic | Mean | 99% interval |
|---|---:|---:|
| Physicalized raw-GAE direction | +0.01173 | [-0.05751, +0.08097] |
| Whole-rollout normalized direction | +0.000332 | [-0.000246, +0.000910] |

The earlier selected historical rollout's whole-rollout normalized direction, +0.001761, lies well inside the new 64-rollout distribution (about the 83rd empirical percentile), rather than being an extraordinary outlier.

## What this means

The residual destructive update is **not supported by a strong, stable expected gradient signal** in its historical direction. Individual 2,048-row rollouts regularly disagree on sign, and the fixed 64-rollout cohort cannot distinguish the expected direction from zero at the preregistered 99% level.

This strengthens the finite-sample-gradient explanation from the preceding matched test:

- one historical 2,048-row batch can produce a damaging update;
- one epoch is already sufficient to damage control, so ten-epoch reuse is not necessary;
- replacing repeated presentations of one batch with 20,480 independent samples greatly reduces the reliability loss;
- and a much larger independent cohort shows that the first-order credit along the harmful direction has low signal-to-noise ratio.

This does **not** prove that all residual PPO erosion is finite-batch noise, nor that the exact expected gradient is zero. It establishes that, for this exposed witness and this fixed actor direction, a 2,048-transition update is operating in a regime where batch-to-batch uncertainty is large compared with the estimated mean direction.

## Correlations retained

The primary directional estimate has only weak sample correlations with simple batch summaries:

- episode-start count: -0.087
- physicalized advantage mean: +0.049
- physicalized advantage SD: -0.026

So the sign variation is not explained by a simple “more resets / larger advantages” scalar in this cohort.

## Preserved execution details

Two setup/reducer failures are retained rather than hidden:

1. The initial script imported SciPy, which is intentionally absent from the preserved runtime. It stopped before collecting any rollout. The fixed df=63 99% Student critical value was then supplied explicitly.
2. The next invocation collected rollout k=0, then stopped while constructing its summary because `domain` was supplied twice to a Python `dict`. That already-saved rollout was reused byte-for-byte after correcting the summary code; it was not recollected.

A foreground execution-window limit later stopped after rollout k=51. The same fixed run resumed from the 52 saved rollout files and collected only k=52…63. No rollout was discarded, replaced, retried for its sign, or added after the fixed 64.

Total new native training-style interactions: **131,072**. Optimizer transactions: **0**.

## Next candidate

The mechanism evidence now supports testing a larger **unique rollout batch per PPO update**, not fewer epochs or another critic-scale change. A clean candidate is value-scaled PPO with `n_steps=8192` instead of 2048 while keeping the 256-step episode timeout, batch64, 10 epochs, learning rate, clipping, optimizer, total training interactions and total gradient-sample presentations fixed. Over 1,048,576 interactions this also keeps the total Adam minibatch transaction count equal: fewer policy updates, four times more independent rollout data per update.

That should be tested from random initialization against the existing 2,048-step scaled recipe, with all fixed checkpoints retained. Passing would still require held-out training qualification before a default change.

No production/default/master/deployment change or merge.
