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
whitening, and the EKF motion Jacobian. Every source edit is restored in `finally`.
All 24 numerical tests must pass before mutation and again after restoration.
Each mutation must execute its named test and fail the intended assertion; compile
errors, zero/ignored tests, unrelated panics, incomplete summaries, timeouts, and
survivors fail qualification. Source-anchor changes require explicit harness review.
Commands have a 180-second timeout; the CI job also has a 20-minute timeout.

The workflow preserves logs and structured results with the tested commit,
platform/toolchain, command lines, and production-source SHA256 hashes. These are
bounded mutation witnesses, not an exhaustive mutation score or a statistical
consistency certification. Broader seeded estimator-consistency work remains in #4.
