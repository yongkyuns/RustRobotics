# Numerical verification

The `Numerical invariants` workflow runs algorithm-only Rust tests without
browser or MuJoCo requirements. Normal workspace tests also discover the Rust
oracles. Existing tolerances and fixtures are retained.

## Independent grid-planner reference

`rust_robotics_algo/tests/grid_planner_oracles.rs` checks A* and Dijkstra against
an f64 Floyd-Warshall reference with separately owned occupancy and world
coordinates. It covers every 3x3 occupancy mask and ordered endpoint pair
(512 x 81 = 41,472 planner comparisons), 24 fixed-seed 8x7 maps with eight
endpoint pairs at three resolutions (576 more comparisons), known open-grid
costs, diagonal-only corner passages, and out-of-bounds inputs. Each comparison
executes both planners. Success, endpoints, finite coordinates, in-bounds cell
centers, obstacle avoidance, adjacent segments, and optimal path cost are checked.
Equal-cost routes need not have identical cell sequences.

Diagonal moves deliberately preserve the documented destination-only collision
rule; these tests do not introduce footprint-aware collision avoidance. Resolutions
0.25, 1 and 2 keep fixture centers exactly representable. The relative/mixed 1e-5
cost budget follows the existing f32 ranking tolerance rather than current outputs.

## Seeded EKF-SLAM consistency regressions

`rust_robotics_algo/tests/ekf_consistency.rs` uses five ensembles of 1,024 trials,
12 prediction steps per trial, and four Rust tests. Three ensembles cover
prediction-only straight, turning and reverse motion; the other two exercise
public single-observation and batch updates with two uncertain, well-separated
landmarks. An additional sampler test pins the integer stream and checks 32,768
normal samples for reproducibility, finite values, mean and second moment.

Every trial samples a proper seven-dimensional joint Gaussian prior, including
shared translation/rotation and robot-landmark cross-covariances. Ground truth
uses independent f64 composite Simpson integration of continuous velocity and
additive world-pose process noise. The noise law is independently evaluated from
the documented production model, not inferred from the filter's output covariance.
Observations use independent range/bearing geometry and correlated Gaussian noise
(correlation 0.55), without clipping or recycling draws. SplitMix64 and Box-Muller
make the seeded generator independent of the rand crate; only the integer stream
is expected to be bit-identical across platforms.

At one terminal timestep, robot and joint normalized estimation error squared
(NEES) compare actual errors with reported covariance; normalized innovation
squared (NIS) uses the pre-update innovation, an independently finite-differenced
f64 observation Jacobian and full correlated noise matrix. Cholesky solves avoid
explicit covariance inverses. Finiteness, symmetry and positive definiteness are
checked. Statistics pool trials, **not correlated timesteps**. No failed or gated
trial is removed: NIS is evaluated before the final public update and its gates.
Production association, gating and covariance floors are not disabled; assertions
verify that this fixture stays above the floors and does not grow the map.

### Bounds and limits

For an ideal calibrated Gaussian model with N independent d-dimensional errors,
summed NEES/NIS is chi-square with nu=N*d degrees of freedom. We use conservative
Laurent-Massart bounds, not fitted reference outputs: for t=ln(2*16/1e-6), the
mean must lie in `[max(0,nu-2*sqrt(nu*t))/N, (nu+2*sqrt(nu*t)+2*t)/N]`.
A union bound budgets up to 16 comparisons (11 are currently used). The sampler
mean uses the corresponding two-sided normal Chernoff bound.

These are **regression envelopes, not a formal 1e-6 false-alarm guarantee for a
nonlinear, gated EKF**. This controlled low-nonlinearity fixture does not certify
arbitrary operating regimes, non-Gaussian/bias models, default simulator tuning,
long-horizon SLAM observability, landmark birth, ambiguous association, full
error-distribution shape or real sensor calibration. Passing average NEES/NIS is
necessary evidence here, not sufficient proof of general estimator consistency.
No production noise settings or filter algorithms are retuned by these tests.

Distribution and concentration background:

- [NIST: chi-square distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm).
- Laurent and Massart, *Adaptive estimation of a quadratic functional by model
  selection*, Annals of Statistics 28(5), 2000, Lemma 1 (chi-square tails).
- [Moshksar, Gaussian quadratic concentration, equation (4)](https://arxiv.org/html/2412.03774v1).
- [Chen et al., limits of NEES/NIS-only tuning tests](https://arxiv.org/abs/2306.07225).

## Reproducible mutation qualification

Run from a checkout with committed changes and no tracked modifications:

```sh
python3 -m unittest discover -s scripts -p test_numerical_mutations.py -v
python3 scripts/check_numerical_mutations.py --output /tmp/rustrobotics-mutation-evidence
```

Use a new output directory each time. Requires Python 3.10+, Git, Cargo, and the
locked Rust dependencies. The script tests committed HEAD in a disposable Git
worktree and shares Cargo's configured target directory. It never edits the
caller's source files, commits, pushes, or changes branches.

The bounded set targets A* greedy scoring, both planners' diagonal costs, obstacle
checking, shared Grid coordinates, covariance symmetry and PSD, dense Graph SLAM
whitening, and the EKF motion Jacobian. Four further mutations under/overreport
process noise and underreport measurement noise in single/batch updates. These
must fail a statistical NEES/NIS assertion, not a floor or map-size assertion.
Every source edit is restored in `finally`. All 28 numerical tests must pass
before mutation and again after restoration. Each of the 13 mutations must
execute its named test and fail the intended assertion; compile errors,
zero/ignored tests, unrelated panics, incomplete summaries, timeouts, and
survivors fail qualification. Source-anchor changes require explicit harness review.
Commands have a 180-second timeout; the CI job also has a 20-minute timeout.

The workflow preserves logs and structured results with the tested commit,
platform/toolchain, command lines, and production-source SHA256 hashes. Successful
libtest output is retained so ensemble statistics can be inspected in baseline
and restored logs. These are bounded mutation witnesses, not an exhaustive mutation
score or statistical consistency certification. Broader estimator-consistency
coverage remains in #4.