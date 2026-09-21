# Fixed-size math migration

This is the first implementation stage, not complete nalgebra removal.

## Migrated

The shared fixed-size matrix/vector aliases, vehicle and pendulum models,
prepared LQR, fixed-horizon MPC assembly, and particle-filter storage and math
use stack-algebra directly. Training and simulator callers use the same types.
Clarabel remains the MPC optimizer; Burn and ORT remain the learning/inference
backends. There is no backend-trait hierarchy or nalgebra emulation layer.

The dependency is pinned to stack-algebra commit
`3356761a917b308f05c3edf7158aad202d1567a1`, corresponding to upstream PR #62.
That revision retains small nonzero singular directions and supports an explicit
absolute pseudoinverse cutoff. The LQR epsilon continues to mean an absolute
singular-value cutoff and a coefficient-wise Riccati convergence tolerance.
The existing preparation/control split, iteration budget, cost weights, noise,
resampling, and optimizer failure behavior are not intentionally changed.

## Source-level API changes

`Mat<R, C, T>`, `Vector<N, T>`, and the named fixed aliases now denote
stack-algebra types. Use indexing, `from_columns`, `from_rows`, and `from_fn`
rather than nalgebra-specific constructors or views.

The robotics `vector![a, b, c]` deliberately remains a column vector. Import it
from rust_robotics_algo, not stack_algebra (whose comma literal denotes a row).
Likewise, one-argument `zeros!(n)` and `ones!(n)` retain their 1-by-n shape, and
`diag![x]` supports scalar costs. Macro paths are crate-qualified so callers do
not need a direct dependency on the implementation crate for expansion.

The old allocator/dimension-trait block construction helpers (`Block`,
`Horizontal`, `Vertical`, `Diagonal`, `allocate_block_output`, and block/stack/
Kronecker macros) are removed. MPC writes its known block layout directly;
independent tests compare every entry with the previous Kronecker formulation.
The unused general closed-loop eigenvalue calculation is removed from LQR.

## Explicitly deferred

Dynamic EKF-SLAM state/covariance, batch observations, graph linear systems,
sparse assembly, and marginalization still use nalgebra. The transitional
`nalgebra` reexport serves those public interfaces and independent test oracles.
SLAM visualization retains its nalgebra pose and covariance types at the boundary.

No new landmark, observation, graph, or factor-fill limit is imposed. Complete
replacement requires runtime-active matrix operations and factorizations, or a
separately reviewed bounded-state design with explicit overflow behavior. Merely
substituting MatrixBuf does not implement these operations. Do not call this
stage heap-free, no_std, faster, or a completed dependency removal.

## Qualification

Run `cargo test --locked -p rust_robotics_algo --lib --tests` and
`cargo test --locked -p rust_robotics_train --lib --tests`, then native/WASM
workspace checks. The stack_algebra_migration integration suite compares motion,
covariance, and prepared LQR against independent nalgebra/scalar references.
The scaled two-input LQR fixture detects loss of the small singular direction.
Existing seeded filter and training invariants remain enabled, with no weakened
acceptance thresholds. Native/platform and real-browser CI remain required
before merging. Audit preparation scripts/workflows are not production files.
