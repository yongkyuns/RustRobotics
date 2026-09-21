from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    assert old in text, (path, old)
    p.write_text(text.replace(old, new))


replace('rust_robotics_algo/src/localization/particle_filter.rs',
        'cov += pw[i] * (dx * dx.transpose());',
        'cov += (dx * dx.transpose()) * pw[i];')
replace('rust_robotics_algo/src/localization/particle_filter_seeded_tests.rs',
        'weights /= weights.sum();', 'weights /= weights.iter().sum::<f32>();')
replace('rust_robotics_algo/src/localization/particle_filter_seeded_tests.rs',
        'px.column_iter()', '(0..NP).map(|i| px.column(i))')
replace('rust_robotics_algo/src/localization/particle_filter_seeded_tests.rs',
        'a.column_iter()', '(0..NP).map(|i| a.column(i))')
replace('rust_robotics_algo/src/control/vehicle/mod.rs',
        '''        // Check dimensions
        assert_eq!(Ad.nrows(), NX_LAT);
        assert_eq!(Ad.ncols(), NX_LAT);
        assert_eq!(Bd.nrows(), NX_LAT);
        assert_eq!(Bd.ncols(), NU_LAT);''',
        '''        // Shape is checked at compile time by the fixed-size backend.
        let _: &Mat<NX_LAT, NX_LAT> = &Ad;
        let _: &Mat<NX_LAT, NU_LAT> = &Bd;''')
replace('rust_robotics_train/src/ppo_seed_tests.rs',
        'Vector4::from_column_slice(&x)', 'Vector4::from_columns([x])')
replace('rust_robotics_train/tests/ppo_balancing_diagnostics.rs',
        'let mut maximum_absolute_state = env.state().map(f32::abs);',
        'let mut maximum_absolute_state: [f32; 4] = std::array::from_fn(|i| env.state()[i].abs());')
replace('rust_robotics_sim/src/simulator/localization.rs',
        '''self.px
                    .column_iter()''',
        '''self.px
                    .as_slice()
                    .chunks_exact(4)''')
replace('rust_robotics_algo/src/lib.rs',
        '''    pub use nalgebra;
    // Transitional reexport for the runtime-sized SLAM API only.
    pub use stack_algebra;''',
        '''    // Transitional reexport for the runtime-sized SLAM API only.
    pub use nalgebra;
    pub use stack_algebra;''')

Path('STACK_ALGEBRA_MIGRATION.md').write_text('''# Fixed-size math migration

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
''')
