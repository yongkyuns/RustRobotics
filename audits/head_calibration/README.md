# Native output-layer calibration kernel

Registered protocol: RustRobotics issue35 comment5794572410, before execution.
This is a numerical implementation qualification, not an online-learning or
policy-performance experiment. No production crate imports this code.

`head.rs` accepts borrowed immutable float32 final-hidden features, targets and
incoming output coefficients. It standardizes features using population moments,
solves a 65-variable ridge correction in float64 (ridge0.001, unpenalized bias,
feature-scale floor1e-8), folds the correction into an owned float32 output head,
and checks finite values, the normal-equation residual and objective decrease.
The fixed coefficient is not an API tuning knob. No optimizer objects, actor,
physical state, holdout label or evaluation metric enter this fitting API.

## Ownership for eventual online use

An actor-baseline sidecar should own an immutable snapshot of the encoder used
to generate its fit features, together with this returned head. It must be fitted
from trajectories independent of the actor batch under that same behavior policy,
used without recomputing advantages mid-PPO-update, and discarded/rebuilt when
its feature or policy snapshot changes. The ordinary critic and all of its Adam
moments keep their existing ownership. Do not write these solved coefficients
into the Adam-managed network and continue using stale moment state.

This sidecar approach deliberately separates two questions: whether current-state
baseline calibration improves actor credit, and whether the ordinary critic's
representation/targets keep tracking as policies change. It does not automatically
repair far-cutoff bootstraps or establish that the sidecar remains calibrated.
The changing-policy driver is not implemented or tested by this kernel package.

## Executed qualification contract

The workflow compiles a formatted copy in a standalone no-dependency crate,
runs strict Clippy and nine native unit tests, then compares both previously fixed
4-trajectory/24-trajectory fits against independent augmented least-squares/SVD.
Original and formatted sources are retained; production sources are not changed.
Hidden features come from the prior float32 Torch computation on hash-verified
source artifacts. This isolates the native FIT from backend feature-rounding
changes; it is not a claim of complete Burn/Torch inference parity.

`verify.py` uses existing `audits/critic_generalization/study.py` to load only the
N=1 independent-future fitting trajectories. Fitting witnesses/holdouts do not
enter either fit. The original frozen arrays, all Rust outputs, SVD arrays,
repeat results, input hashes and toolchain are saved. Numerical thresholds are
fixed in the protocol: equation residual1e-9, standardized delta difference1e-7,
folded float32 head difference1e-5, identical-feature scalar-f32 prediction
difference1e-4. Passing these tests does not certify policy improvement.

Each fit makes one solve and zero Adam updates. No new simulation or actor
optimization is performed. The separate full-network online-critic pilot is
neither modified nor replaced by this qualification.

Run `cargo test --manifest-path audits/head_calibration/Cargo.toml` for native
synthetic controls. Actual-system verification also requires Python3.13.5,
NumPy2.3.5, Torch2.10.0+cpu, and independent-futures result/build artifacts
10698150266/10697715468. The workflow retains their exact hashes. Result status
must be checked on the actual workflow head, not inferred from these commands.
