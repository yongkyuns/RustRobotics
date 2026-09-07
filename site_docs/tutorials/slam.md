# SLAM Tutorial

SLAM—simultaneous localization and mapping—estimates robot trajectory and map together. The hard
part is not simply that both are unknown. It is that errors in one become evidence for the other:
pose error distorts the map, and a distorted map changes later pose estimates.

```{raw} html
<div class="sim-embed-card">
  <iframe
    class="sim-embed-frame"
    data-sim-mode="slam"
    data-sim-path="?mode=slam&embed=focused&ui=20260907a"
    title="Rust Robotics SLAM simulator"
    loading="lazy"
  ></iframe>
</div>
```

## Learning goals

By the end of this chapter, you should be able to:

- explain why pose and map uncertainty cannot be treated independently
- distinguish recursive EKF-SLAM from graph-based nonlinear least squares
- explain why loop closure can revise an entire trajectory
- understand how information matrices weight residuals
- distinguish sparse problem structure from a genuinely sparse numerical solve
- identify data association and outlier handling as correctness-critical parts of SLAM

## The joint problem

SLAM estimates

$$
p(x_{1:t},m\mid z_{1:t},u_{1:t}).
$$

The map is useful because it constrains pose; pose is useful because it determines where map
observations belong. That feedback is the defining structure of SLAM.

## EKF-SLAM: one joint Gaussian

EKF-SLAM stores robot pose and landmarks in one joint state with one joint covariance. Prediction
propagates motion uncertainty; observations update the coupled state using the Kalman gain

$$
K = PH^T(HPH^T+R)^{-1}.
$$

The important part is not only the pose block or each landmark variance. Pose-landmark and
landmark-landmark cross-correlation encode how information about one part of the state should alter
another.

As landmark count grows, that dense joint covariance becomes an important scaling concern. It is
also why covariance symmetry and positive-semidefiniteness are useful numerical invariants for CI.

## Graph SLAM: optimize factors globally

Graph SLAM treats poses and landmarks as variables and measurements as factors. A common objective
has the form

$$
\min_x \sum_i r_i(x)^T\Omega_i r_i(x),
$$

where `r_i` is a residual and `\Omega_i` is its information matrix.

At each iteration, residuals are linearized, a Jacobian is assembled, and a linear system is solved
for an update. This formulation naturally supports long-range constraints such as loop closures.

### Information weighting matters

If a positive-definite information matrix has Cholesky factorization

$$
\Omega = LL^T,
$$

then a valid square-root least-squares residual is

$$
L^T r,
$$

because

$$
\|L^Tr\|^2 = r^TLL^Tr = r^T\Omega r.
$$

The current polishing pass adds a regression with a **correlated, non-diagonal** information matrix.
That matters because an incorrect Cholesky orientation can go unnoticed when every information
matrix is diagonal.

## Sparse structure versus sparse solving

Graph SLAM is structurally sparse because each factor touches only a few variables. The repository's
`SparseSlamSolver` preserves sparsity while building the Jacobian and forming `J^T J`.

However, the current implementation then materializes the regularized normal matrix as a dense
matrix and solves it using dense LU. It is therefore best described as **sparse assembly with a
dense reference solve**, not as a complete sparse-Cholesky backend.

```{admonition} Do not confuse an algorithm family with this implementation's current backend
:class: note-shell

Large graph-SLAM systems benefit from sparse factorization and ordering strategies. Those are real
advantages of the formulation, but the current Rust implementation has not yet carried sparsity
through the final factorization. Benchmarks should describe the code that actually runs, not the
backend we intend to add later.
```

## Why loop closure is powerful

Suppose odometry accumulates a small heading error on every step. Locally, each neighboring pair of
poses may still look plausible, but after a long loop the estimated trajectory can miss its start
position substantially.

A credible loop closure adds a long-range constraint between states that were previously connected
only through many local measurements. Global optimization can then redistribute error across the
whole trajectory instead of applying one cosmetic correction at the end.

The hard part is deciding whether the proposed loop closure is actually valid. A false long-range
constraint can damage the whole solution. Candidate generation, observation quality, covariance,
robust weighting, and statistical gating therefore deserve as much attention as the optimizer.

## Experiment 1: watch local error become global map error

**Question:** How does pose drift corrupt map geometry?

1. Run a scenario without relying on a strong loop closure.
2. Watch the estimated trajectory and landmark positions together.
3. Identify where small pose errors begin to create visible map distortion.
4. Compare local consistency over a short segment with global consistency over the whole route.

**What this teaches:** a map is not independent ground truth inside SLAM; it is estimated through the
same uncertain trajectory.

## Experiment 2: inspect a correction event

**Question:** What information makes the optimizer revise old states?

1. Allow drift to accumulate.
2. Watch for repeated observations or a revisit that can create a long-range constraint.
3. Compare the trajectory immediately before and after the correction.
4. Look for changes throughout the old path, not only at the newest pose.

**Prediction:** a useful loop closure can move many previously estimated poses and landmarks because
the optimizer is seeking global consistency.

## Experiment 3: robust handling of a bad constraint

**Question:** Why can one false loop closure be worse than gradual drift?

1. Compare a clean scenario with one containing an inconsistent measurement or association when the
   controls permit it.
2. Observe whether the estimator down-weights or rejects the bad constraint.
3. Compare the map before and after robust handling.

Interpret the result carefully: robust kernels reduce the influence of large residuals, while
statistical gating and data association decide whether a constraint should participate at all.
They solve related but different problems.

## Experiment 4: scaling is an implementation question

**Question:** Where does computation actually grow?

When using the current code, separate three stages conceptually:

1. sparse factor/Jacobian construction
2. sparse formation of the normal matrix
3. conversion to a dense normal matrix and dense LU solve

Increasing pose/landmark count makes the third stage increasingly important. A future true sparse
backend should be benchmarked against this dense reference on deterministic fixtures, checking both
final objective agreement and scaling behavior.

## EKF-SLAM versus graph SLAM

| Property | EKF-SLAM | Graph SLAM |
| --- | --- | --- |
| Main representation | joint mean/covariance | variables + factors |
| Update style | recursive filter | iterative optimization |
| Correlation | explicit dense covariance | implicit through factor graph/system |
| Loop closure | possible but can be awkward at scale | natural long-range factor |
| Current repo scaling concern | dense covariance | dense final solve after sparse assembly |

## What to measure

For meaningful comparisons, record more than the final picture:

- pose error against simulated truth
- landmark error
- objective before and after optimization
- accepted/rejected constraint count
- loop-closure residuals and robust weights
- covariance symmetry/PSD for EKF-SLAM
- Jacobian/normal-matrix nonzero count
- solve time as graph size grows

## Common mistakes

- treating localization and mapping as separable inside SLAM
- ignoring cross-correlation in EKF-SLAM
- assuming every revisit is a valid loop closure
- calling sparse assembly a sparse factorization
- testing only diagonal covariance/information matrices
- judging correctness from final map appearance without checking the objective and residuals

## Where to go next

The robot-runtime chapter shifts from estimation algorithms to system integration: observation
construction, policy/controller execution, action decoding, timing, and runtime portability.
