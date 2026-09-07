# Localization Tutorial

Localization is the problem of maintaining a belief about robot state when motion and sensing are
uncertain. This chapter uses a particle filter because the individual hypotheses are visible: you
can watch uncertainty spread, concentrate, collapse, and recover instead of treating the estimator
as a black box.

```{raw} html
<div class="sim-embed-card">
  <iframe
    class="sim-embed-frame"
    data-sim-mode="localization"
    data-sim-path="?mode=localization&embed=focused&ui=20260907a"
    title="Rust Robotics localization simulator"
    loading="lazy"
  ></iframe>
</div>
```

## Learning goals

By the end of this chapter, you should be able to:

- explain why localization maintains a distribution rather than one exact pose
- describe predict, weight, normalize, estimate, and resample steps
- distinguish simulated sensor noise from noise assumed by the filter
- explain effective sample size and particle degeneracy
- recognize the repository's practical recovery heuristics as additions to a textbook particle filter
- compare estimate error, particle spread, and computation rather than relying on visual smoothness

## The concrete estimator used here

The reusable particle filter estimates the planar state

$$
\mathbf{x} =
\begin{bmatrix}
p_x & p_y & \psi & v
\end{bmatrix}^T,
$$

where `p_x` and `p_y` are position, `\psi` is heading, and `v` is speed. Motion input contains
forward velocity and yaw rate. Landmark observations contain **range to a known landmark
position**.

The current algorithm crate uses **100 particles**. That is an implementation choice for this demo,
not a property of particle filters in general.

The target posterior is

$$
p(x_t \mid z_{1:t},u_{1:t}),
$$

approximated by weighted samples

$$
p(x_t \mid z_{1:t},u_{1:t}) \approx
\sum_i w_t^{(i)}\,\delta(x_t-x_t^{(i)}).
$$

## One update, step by step

### 1. Predict

Each particle is propagated through the motion model with perturbed control input. This spreads the
belief according to the assumed motion uncertainty.

### 2. Weight

For each landmark range observation, the predicted range from a particle to that landmark is
compared with the measured range. Particles that explain the measurement well receive more weight.

### 3. Normalize or recover

Weights are normalized when their sum is usable. The repository also contains explicit recovery
logic when weights collapse: particles can be regenerated around observations and weights reset.

### 4. Estimate

The displayed estimate is formed from the weighted particle cloud. The current implementation also
applies adaptive smoothing, especially while the cloud is widely spread or recovering.

### 5. Resample

Effective sample size is computed as

$$
N_{\text{eff}} = \frac{1}{\sum_i w_i^2}.
$$

When it falls below half the particle count, the implementation performs low-variance resampling.

```{admonition} This is a practical demo filter, not a minimal textbook filter
:class: note-shell

The implementation mixes a small uniform component into the likelihood, detects severe weight
collapse, can reset particles around observations, and smooths the final estimate adaptively.
Those features make recovery easier to demonstrate, but they also mean you should not attribute
every behavior you see to the basic particle-filter equations alone.
```

## Two different kinds of noise

A frequent source of confusion in estimator demos is using one slider called "noise" for two very
different ideas.

**Simulation noise** changes the measurements or motion presented to the estimator. It represents
what the world/sensors actually do.

**Filter noise parameters** describe what the estimator believes about uncertainty. They determine
how widely particles propagate and how sharply measurements are scored.

A well-tuned estimator does not necessarily assume the smallest possible noise. It should model the
uncertainty of the data it actually receives.

## Experiment 1: motion uncertainty

**Question:** What happens when the filter believes motion is less predictable?

1. Start with the default localization setup.
2. Let the vehicle move long enough to establish a compact cloud.
3. Increase the filter's motion uncertainty while leaving the simulated path and measurement setup
   otherwise unchanged.
4. Compare particle spread, estimate lag, and recovery after turns.

**Prediction:** the cloud should spread more during prediction. That can improve robustness to
motion-model mismatch, but it also makes the posterior less concentrated and costs more useful
particle density around the best hypotheses.

## Experiment 2: sensor-model mismatch

**Question:** Is trusting measurements more always better?

1. Establish a baseline with moderate measurement uncertainty.
2. Make the filter's assumed observation noise smaller without improving the simulated sensor.
3. Repeat with a larger assumed observation noise.
4. Compare the estimate error and cloud behavior, especially after an unusually bad measurement.

**Prediction:** an overconfident likelihood can collapse the cloud around misleading observations;
an overly broad likelihood can fail to extract enough information from good observations.

## Experiment 3: degeneracy and resampling

**Question:** Why not simply keep multiplying weights forever?

1. Watch the particle weights/cloud through several informative updates.
2. Look for a point where only a small fraction of particles remain plausible.
3. Observe the cloud immediately after resampling.
4. Compare concentration before and diversity after resampling.

**What this teaches:** resampling redirects computation toward useful hypotheses, but repeated
resampling can also remove diversity. It solves degeneracy by creating a different tradeoff.

## Experiment 4: recovery is not the same as ordinary tracking

**Question:** What additional mechanisms help once the belief is badly wrong?

1. Create a high-uncertainty or temporarily unobservable situation.
2. Allow the estimate and true state to separate.
3. Restore informative observations.
4. Watch the transition back toward a concentrated belief.

Interpret this experiment with the implementation notes above: recovery can include likelihood
mixing, explicit particle reset, and output smoothing. A recovery success therefore demonstrates
the whole estimator policy, not only resampling.

## Complexity

Let `N` be particle count and `M` the number of observations processed in one update.

- prediction: `O(N)`
- measurement weighting: approximately `O(NM)`
- estimate and effective-sample-size calculations: `O(N)`
- low-variance resampling: `O(N)`
- particle storage: `O(N)`

The current fixed-size implementation uses `N = 100`. A more complete performance experiment would
make particle count configurable and measure both estimation quality and update time under the same
seeded scenario.

## What to measure

When comparing settings, prefer:

- position error to the simulated truth
- heading error
- particle spread or covariance proxy
- effective sample size before resampling
- recovery time after loss of useful observations
- update cost when particle count or observation count changes

A visually smooth estimate is not automatically a better estimate. Smoothing can hide uncertainty
or delay correction.

## Common mistakes

- interpreting the weighted mean as the whole belief
- confusing simulation noise with estimator noise assumptions
- assuming a sharper likelihood is always better
- treating recovery heuristics as part of the canonical particle-filter derivation
- judging one randomized run as a deterministic benchmark
- ignoring how particle count trades compute for approximation quality

## Where to go next

The planning tutorial removes probabilistic state estimation and asks a different question: given a
representation of free space, how much search is needed to produce a path, and what does
"better path" actually mean?
