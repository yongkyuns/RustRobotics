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
Two RNG-seam mutations reject an ignored particle-filter seed and omitted EKF
observation noise. Every source edit is restored in `finally`. All 36 numerical
tests must pass before mutation and again after restoration. Each of the 15
mutations must execute its named test and fail the intended assertion; compile
errors, zero/ignored tests, unrelated panics, incomplete summaries, timeouts, and
survivors fail qualification. Source-anchor changes require explicit harness review.
Commands have a 180-second timeout; the CI job also has a 20-minute timeout.

The workflow preserves logs and structured results with the tested commit,
platform/toolchain, command lines, and production-source SHA256 hashes. Successful
libtest output is retained so ensemble statistics can be inspected in baseline
and restored logs. These are bounded mutation witnesses, not an exhaustive mutation
score or statistical consistency certification. Broader estimator-consistency
coverage beyond this initial acceptance baseline is not claimed.

## Issue #4 acceptance ledger and seed inventory

This ledger reconciles the original bounded CI-verification issue; it is not a
claim that every robotics algorithm or operating regime is formally verified.

| Original requirement | Retained verification | Perturbation evidence |
| --- | --- | --- |
| SLAM/EKF analytic Jacobians | `graph_slam_numerical_tests.rs` and `ekf_slam_numerical_tests.rs`: independent f64 finite differences, normal matrices and gradients | PRs #24/#25 qualified dense/sparse/observation sign changes; the permanent harness retains the EKF motion-sign witness |
| Covariance symmetry and PSD | `numerical_invariants.rs::ekf_slam_covariance_remains_symmetric_and_psd` and particle-filter rollout checks | Permanent asymmetry and negative-variance mutations |
| A* optimal costs versus Dijkstra | Deterministic comparison in `numerical_invariants.rs`, plus both planners versus independent Floyd-Warshall in `grid_planner_oracles.rs` | Greedy scoring, diagonal-cost, obstacle and coordinate mutations |
| Dense/sparse Graph SLAM solutions and objectives | Existing integration fixture plus correlated factor/normal-equation checks and a closed-form conflicting-measurement optimum | Correlated-whitening regression and PR #24 red/green qualification |
| Practical seeded estimator consistency | `ekf_consistency.rs`: five independent-trial ensembles, terminal NEES and pre-update NIS | Under/overreported process noise and single/batch measurement-noise mutations |
| Deterministic randomized planner/filter regressions | Per-test RNG ownership below, including hidden synthetic sensor, recovery and resampling draws | Full particle/weight/covariance replay; ignoring the PF seed and omitting EKF observation noise are rejected |

All numerical groups run in the algorithm-only workflow, without browser/MuJoCo.
Existing Jacobian, path-cost, covariance and ensemble tolerances were retained;
their rationale is documented at the assertions and above. The original legacy
EKF accuracy thresholds and trial counts are unchanged. These older scenario
thresholds are engineering regression bounds, not chi-square consistency tests.

The seed audit inspected the 33 tracked algorithm Rust files at `b238433`, including
integration tests, and traced callers of the three RNG-using source modules:

- `path_planning/rrt.rs`: all four unit tests already set `seed: Some(42)`.
  Production `seed: None` still intentionally draws from entropy.
- `slam/ekf_slam.rs`: 13 existing tests now own `StdRng` instances with literal
  seeds `0xe4f00000` through `0xe4f00010` (gaps correspond to four deterministic
  tests). Map generation, odometry perturbations, synthetic observations and the
  complete-cycle helper receive that same explicit RNG. Each test prints its seed;
  tests do not share global/thread-local seeded state. Trial counts are retained.
- `localization/particle_filter.rs`: replace two assertion-free demonstrations
  with five seeded tests (`0x50460001` through `0x50460007`). The old 50-second
  rollout now checks every particle, weight and covariance and compares complete
  same-seed histories. Recovery and resampling have separately forced fixtures.
- Existing numerical integration/Graph SLAM/EKF algebra fixtures are deterministic.
  Grid ensembles retain xorshift32 and fixed seeds; consistency ensembles retain
  their explicit SplitMix64/Box-Muller streams and predeclared bounds.

Three short EKF RNG-seam tests use `0xe4f10001` through `0xe4f10003`: preserve the
existing uniform noise formula/draw order, consume no draws when noise is disabled,
and replay a complete step against the explicit prediction/observation/update
sequence. The simulator's two noise-generator tests also have per-test seeds
`0x50150001` and `0x50150002`; their sample counts and distribution bounds are retained.

RNG injection helpers are private. Existing public API signatures and entropy-backed
interactive behavior are preserved, as are noise distributions/scales, draw order,
filter equations, and recovery/resampling rules. This is not a particle-filter
statistical-calibration change and adds no new public seeded API. Replay equality
is required on the same executable/platform; `StdRng` plus floating-point math is
not promised to be bit-identical across platforms or dependency updates. The
lockfile remains unchanged. No speedup or benchmark claim is made.

The permanent harness now checks 36 numerical tests before and after 15 mutations.
The 17 legacy EKF scenario tests additionally run in the algorithm-only CI lane;
normal workspace tests exercise the same seeded paths on all native platforms.
This inventory is an audited snapshot, not a static proof that future code cannot
introduce another hidden random source. New stochastic tests must receive explicit
caller-owned RNGs (including their transitive synthetic-data helpers).

Follow-on statistical work (landmark birth, broader priors, large heading uncertainty,
association ambiguity, biased/mismatched sensors and long-horizon observability) is
outside #4's practical first consistency baseline. Branch protection is separate
administrative work in #18; a workflow alone cannot enforce repository merge policy.
