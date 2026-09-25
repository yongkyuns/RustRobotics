//! Model Predictive Control for the inverted pendulum.
//!
//! This module solves a finite-horizon quadratic program at each control step.
//! Compared with LQR, the main difference is that MPC explicitly optimizes over
//! a sequence of future states and controls:
//!
//! - decision variables: all predicted states `x(0..N)` and controls `u(0..N-1)`
//! - objective: sum of stage costs plus a terminal cost
//! - constraints: system dynamics and actuator bounds
//!
//! The quadratic program has the standard form:
//!
//! `min 1/2 z^T P z + q^T z`
//!
//! subject to:
//!
//! `A z = b` for dynamics
//! `G z <= h` for control bounds
//!
//! where `z` stacks all future states and controls. The terminal cost uses the
//! infinite-horizon LQR Riccati solution so the finite-horizon optimization
//! inherits a stabilizing tail cost.
use super::*;
use crate::prelude::*;

/// Prediction horizon length.
const N: usize = 12;

/// Total number of decision variables: (N+1) states + N controls
const N_VARS: usize = (N + 1) * NX + N * NU;
/// Total number of equality constraints (dynamics)
const N_EQ: usize = (N + 1) * NX;
/// Number of inequality constraints (control bounds only: upper + lower for each control)
const N_INEQ: usize = 2 * N * NU;

/// Error returned when the MPC quadratic-program solver does not produce a usable solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpcSolveError {
    status: String,
}

impl MpcSolveError {
    fn from_status(status: impl std::fmt::Debug) -> Self {
        Self {
            status: format!("{status:?}"),
        }
    }

    /// Returns the underlying solver status as a diagnostic string.
    pub fn status(&self) -> &str {
        &self.status
    }
}

impl std::fmt::Display for MpcSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MPC solver did not produce a usable solution: {}",
            self.status
        )
    }
}

impl std::error::Error for MpcSolveError {}

/// MPC problem data that is constant for a fixed model and timestep.
///
/// Preparing the problem once avoids repeating the terminal Riccati solve,
/// horizon matrix assembly, and dense-to-CSC conversion on every control tick.
/// A fresh Clarabel solver is still constructed for each solve; this type does
/// not cache solver factorization or warm-start the optimization.
pub struct PreparedMpc {
    p: clarabel::algebra::CscMatrix<f64>,
    q: Vec<f64>,
    a: clarabel::algebra::CscMatrix<f64>,
    b_template: Vec<f64>,
}

impl PreparedMpc {
    /// Builds the static quadratic-program data for a fixed model and timestep.
    pub fn new(model: Model, dt: f32) -> Self {
        use crate::control::LQR;
        use clarabel::algebra::*;

        let (Ad, Bd) = model.model(dt);
        let umin = -50.0_f64;
        let umax = 50.0_f64;
        let Q = model.Q;
        let R = model.R;
        let QN = model.solve_DARE(Ad, Bd);

        let (P_mat, A_mat) = horizon_matrices(&Ad, &Bd, &Q, &QN, &R);

        // The pendulum MPC reference is the zero state, so q is identically zero.
        let q = vec![0.0; N_VARS];

        let mut p_row_indices = Vec::new();
        let mut p_col_ptrs = vec![0];
        let mut p_values = Vec::new();
        for j in 0..N_VARS {
            for i in 0..=j {
                let val = P_mat[(i, j)] as f64;
                if val.abs() > 1e-12 {
                    p_row_indices.push(i);
                    p_values.push(val);
                }
            }
            p_col_ptrs.push(p_row_indices.len());
        }
        let p = CscMatrix::new(N_VARS, N_VARS, p_col_ptrs, p_row_indices, p_values);

        let n_constraints = N_EQ + N_INEQ;
        let mut a_row_indices = Vec::new();
        let mut a_col_ptrs = vec![0];
        let mut a_values = Vec::new();
        for j in 0..N_VARS {
            for i in 0..n_constraints {
                let val = A_mat[(i, j)] as f64;
                if val.abs() > 1e-12 {
                    a_row_indices.push(i);
                    a_values.push(val);
                }
            }
            a_col_ptrs.push(a_row_indices.len());
        }
        let a = CscMatrix::new(n_constraints, N_VARS, a_col_ptrs, a_row_indices, a_values);

        let mut b_template = vec![0.0; n_constraints];
        for k in 0..N {
            b_template[N_EQ + k] = umax;
            b_template[N_EQ + N * NU + k] = -umin;
        }

        Self {
            p,
            q,
            a,
            b_template,
        }
    }

    /// Solves the prepared MPC problem for the current measured state.
    pub fn try_control(&self, x: Vector4) -> Result<f32, MpcSolveError> {
        use clarabel::solver::*;

        let mut b = self.b_template.clone();
        for i in 0..NX {
            b[i] = -(x[i] as f64);
        }

        let cones = [ZeroConeT(N_EQ), NonnegativeConeT(N_INEQ)];
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .build()
            .unwrap();
        let mut solver = DefaultSolver::new(&self.p, &self.q, &self.a, &b, &cones, settings);
        solver.solve();

        let u_idx = (N + 1) * NX;
        match solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                Ok(solver.solution.x[u_idx] as f32)
            }
            ref status => Err(MpcSolveError::from_status(status)),
        }
    }
}

/// Assemble the existing state-first, control-last horizon layout explicitly.
/// Dynamics rows are -x[k] + A*x[k-1] + B*u[k-1] = 0; the first
/// state row is -x[0] = -x_measured. Clarabel conversion stays unchanged.
fn horizon_matrices(
    ad: &AMat,
    bd: &BMat,
    q: &QMat,
    terminal_q: &QMat,
    r: &RMat,
) -> (Mat<N_VARS, N_VARS>, Mat<{ N_EQ + N_INEQ }, N_VARS>) {
    let mut cost = Mat::zeros();
    let mut constraints = Mat::zeros();
    let control_start = (N + 1) * NX;
    for step in 0..=N {
        let weight = if step == N { terminal_q } else { q };
        for row in 0..NX {
            constraints[(step * NX + row, step * NX + row)] = -1.0;
            for column in 0..NX {
                cost[(step * NX + row, step * NX + column)] = weight[(row, column)];
                if step > 0 {
                    constraints[(step * NX + row, (step - 1) * NX + column)] = ad[(row, column)];
                }
            }
            if step > 0 {
                for input in 0..NU {
                    constraints[(step * NX + row, control_start + (step - 1) * NU + input)] =
                        bd[(row, input)];
                }
            }
        }
    }
    for step in 0..N {
        for input in 0..NU {
            let column = control_start + step * NU + input;
            for other in 0..NU {
                cost[(column, control_start + step * NU + other)] = r[(input, other)];
            }
            constraints[(N_EQ + step * NU + input, column)] = 1.0;
            constraints[(N_EQ + N * NU + step * NU + input, column)] = -1.0;
        }
    }
    (cost, constraints)
}

/// MPC control using quadratic programming with the Clarabel solver.
///
/// This one-shot API prepares the fixed QP structure and solves it once. Runtime
/// callers with a fixed model and timestep should reuse [`PreparedMpc`] instead.
pub fn try_mpc_control(x: Vector4, model: Model, dt: f32) -> Result<f32, MpcSolveError> {
    PreparedMpc::new(model, dt).try_control(x)
}

/// Compatibility helper that preserves the historical zero-force fallback.
///
/// New runtime code should prefer [`try_mpc_control`] so solver failure cannot be
/// mistaken for a valid zero-force optimum.
pub fn mpc_control(x: Vector4, model: Model, dt: f32) -> f32 {
    try_mpc_control(x, model, dt).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn explicit_horizon_matches_legacy_kronecker_layout() {
        use nalgebra::{DMatrix, Matrix4, Vector4 as NaVector4};
        let ad: AMat =
            matrix![1.0, 2.0, 0.0, 0.0; 0.0, 1.0, 3.0, 0.0; 0.0, 0.0, 1.0, 4.0; 5.0, 0.0, 0.0, 1.0];
        let bd: BMat = vector![0.1, 0.2, 0.3, 0.4];
        let q: QMat = matrix![2.0, 0.5, 0.0, 0.0; 0.5, 3.0, 0.0, 0.0; 0.0, 0.0, 4.0, 0.25; 0.0, 0.0, 0.25, 5.0];
        let terminal_q = q * 2.0;
        let r: RMat = diag![0.7];
        let (cost, constraints) = horizon_matrices(&ad, &bd, &q, &terminal_q, &r);
        let na_ad = Matrix4::from_column_slice(ad.as_slice());
        let na_bd = NaVector4::from_column_slice(bd.as_slice());
        let na_q = Matrix4::from_column_slice(q.as_slice());
        let mut expected_cost = DMatrix::<f32>::zeros(N_VARS, N_VARS);
        expected_cost
            .slice_mut((0, 0), (N * NX, N * NX))
            .copy_from(&DMatrix::<f32>::identity(N, N).kronecker(&na_q));
        expected_cost
            .slice_mut((N * NX, N * NX), (NX, NX))
            .copy_from(&(na_q * 2.0));
        expected_cost
            .slice_mut(((N + 1) * NX, (N + 1) * NX), (N * NU, N * NU))
            .copy_from(&(DMatrix::<f32>::identity(N * NU, N * NU) * 0.7));
        let shift =
            DMatrix::<f32>::from_fn(
                N + 1,
                N + 1,
                |row, column| if row == column + 1 { 1.0 } else { 0.0 },
            );
        let ax = DMatrix::<f32>::identity(N + 1, N + 1).kronecker(&(-Matrix4::identity()))
            + shift.kronecker(&na_ad);
        let control_shift =
            DMatrix::<f32>::from_fn(
                N + 1,
                N,
                |row, column| if row == column + 1 { 1.0 } else { 0.0 },
            );
        let bu = control_shift.kronecker(&na_bd);
        let mut expected_constraints = DMatrix::<f32>::zeros(N_EQ + N_INEQ, N_VARS);
        expected_constraints
            .slice_mut((0, 0), (N_EQ, (N + 1) * NX))
            .copy_from(&ax);
        expected_constraints
            .slice_mut((0, (N + 1) * NX), (N_EQ, N * NU))
            .copy_from(&bu);
        expected_constraints
            .slice_mut((N_EQ, (N + 1) * NX), (N * NU, N * NU))
            .copy_from(&DMatrix::<f32>::identity(N * NU, N * NU));
        expected_constraints
            .slice_mut((N_EQ + N * NU, (N + 1) * NX), (N * NU, N * NU))
            .copy_from(&(-DMatrix::<f32>::identity(N * NU, N * NU)));
        assert_eq!(cost.as_slice(), expected_cost.as_slice());
        assert_eq!(constraints.as_slice(), expected_constraints.as_slice());
    }

    #[test]
    fn prepared_api_matches_one_shot_api_across_states() {
        let model = Model::default();
        let dt = 0.1;
        let prepared = PreparedMpc::new(model, dt);
        let states = [
            vector![0.0, 0.1, 0.1, 0.0],
            vector![0.5, -0.2, 0.05, 0.1],
            vector![-0.5, 0.3, -0.1, -0.2],
        ];

        for x in states {
            let expected = try_mpc_control(x, model, dt).expect("one-shot MPC solve");
            let actual = prepared.try_control(x).expect("prepared MPC solve");
            assert!(
                (actual - expected).abs() < 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn fallible_api_reports_a_valid_solution() {
        let x = vector![0.0, 0.1, 0.1, 0.0];
        let result = try_mpc_control(x, Model::default(), 0.1);
        let u = result.expect("nominal MPC solve should succeed");
        assert!(u.is_finite());
    }

    #[test]
    fn compatibility_api_matches_fallible_api_on_success() {
        let x = vector![0.0, 0.1, 0.1, 0.0];
        let model = Model::default();
        let expected = try_mpc_control(x, model, 0.1).expect("nominal solve");
        let actual = mpc_control(x, model, 0.1);
        assert!((actual - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mpc_control() {
        let x = vector![0., 0.1, 0.1, 0.];
        let dt = 0.1;
        let model = Model::default();

        // Warm-up run
        let _u = mpc_control(x, model, dt);

        // Benchmark multiple iterations
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = mpc_control(x, model, dt);
        }

        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("\n=== MPC Performance Benchmark ===");
        println!("Iterations: {}", iterations);
        println!("Total time: {:?}", elapsed);
        println!("Average time per call: {:.2} µs", avg_time_us);
        println!("Frequency: {:.2} Hz", 1_000_000.0 / avg_time_us);

        // Verify the control output is reasonable
        let u = mpc_control(x, model, dt);
        println!("Control output: {:.6}", u);
        assert!(u.is_finite(), "Control output should be finite");
    }

    #[test]
    fn test_mpc_control_at_reference() {
        // Test when state is at reference (should need minimal control)
        let x = vector![1., 0., 0., 0.]; // At reference position
        let dt = 0.1;
        let model = Model::default();

        let u = mpc_control(x, model, dt);
        println!("Control at reference state: {:.6}", u);
        assert!(u.is_finite());
    }

    #[test]
    fn test_mpc_control_various_states() {
        let dt = 0.1;
        let model = Model::default();

        let test_states = [
            vector![0., 0., 0., 0.],     // Origin
            vector![0.5, 0., 0., 0.],    // Halfway to reference
            vector![-1., 0., 0.2, 0.],   // Negative position, tilted
            vector![2., 0.5, -0.1, 0.1], // Beyond reference
        ];

        println!("\n=== MPC Control for Various States ===");
        for (i, x) in test_states.iter().enumerate() {
            let u = mpc_control(*x, model, dt);
            println!("State {}: {:?} -> u = {:.6}", i, x.as_slice(), u);
            assert!(u.is_finite());
        }
    }

    #[test]
    fn test_mpc_closed_loop_simulation() {
        use crate::control::StateSpace;

        let dt = 0.05; // 50ms time step (20 Hz)
        let model = Model::default();
        let (Ad, Bd) = model.model(dt);

        // Start with tilted pendulum
        let mut x = vector![0., 0., 0.2, 0.]; // 0.2 rad tilt (~11 degrees)

        println!("\n=== MPC Closed-Loop Simulation ===");
        println!(
            "Initial state: pos={:.3}, vel={:.3}, angle={:.3}, omega={:.3}",
            x[0], x[1], x[2], x[3]
        );

        let mut stable = true;
        for step in 0..100 {
            // Compute control
            let u = mpc_control(x, model, dt);

            // Apply dynamics: x_next = Ad * x + Bd * u
            x = Ad * x + Bd * u;

            // Print every 10 steps
            if step % 10 == 0 {
                println!(
                    "Step {:3}: pos={:7.3}, vel={:7.3}, angle={:7.4}, omega={:7.3}, u={:7.3}",
                    step, x[0], x[1], x[2], x[3], u
                );
            }

            // Check if unstable (angle > 1 rad or position > 10m)
            if x[2].abs() > 1.0 || x[0].abs() > 10.0 {
                println!("UNSTABLE at step {}", step);
                stable = false;
                break;
            }
        }

        println!(
            "Final state: pos={:.3}, vel={:.3}, angle={:.4}, omega={:.3}",
            x[0], x[1], x[2], x[3]
        );

        // Check that the angle has been stabilized (should be close to 0)
        assert!(stable, "MPC should stabilize the pendulum");
        assert!(
            x[2].abs() < 0.1,
            "Angle should be stabilized near 0, got {}",
            x[2]
        );
    }

    #[test]
    fn test_lqr_closed_loop_simulation() {
        use crate::control::{StateSpace, LQR};

        let dt = 0.05;
        let model = Model::default();
        let (Ad, Bd) = model.model(dt);

        let mut x = vector![0., 0., 0.2, 0.];

        println!("\n=== LQR Closed-Loop Simulation (for comparison) ===");
        println!(
            "Initial state: pos={:.3}, vel={:.3}, angle={:.3}, omega={:.3}",
            x[0], x[1], x[2], x[3]
        );

        for step in 0..100 {
            let u_vec = model.control(x, dt);
            let u = u_vec[0];
            x = Ad * x + Bd * u;

            if step % 10 == 0 {
                println!(
                    "Step {:3}: pos={:7.3}, vel={:7.3}, angle={:7.4}, omega={:7.3}, u={:7.3}",
                    step, x[0], x[1], x[2], x[3], u
                );
            }
        }

        println!(
            "Final state: pos={:.3}, vel={:.3}, angle={:.4}, omega={:.3}",
            x[0], x[1], x[2], x[3]
        );
    }
}

#[cfg(test)]
mod prepared_mpc_regression_tests {
    use super::*;

    // Reference outputs captured from the pre-refactor implementation at 7086dc7b6d3e18681b0ec9b034358a6cd69986b6.

    fn reference_models() -> [Model; 2] {
        let tuned = Model {
            l_bar: 1.5,
            m_cart: 1.2,
            m_ball: 0.8,
            Q: diag![1.0, 1.0, 10.0, 1.0],
            R: diag![0.1],
            ..Model::default()
        };
        [Model::default(), tuned]
    }

    fn reference_states() -> [Vector4; 5] {
        [
            vector![0.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.1, 0.1, 0.0],
            vector![0.5, -0.2, 0.05, 0.1],
            vector![-0.5, 0.3, -0.1, -0.2],
            vector![2.0, 1.0, 0.6, 0.3],
        ]
    }

    #[test]
    fn prepared_mpc_preserves_pre_refactor_outputs() {
        let expected: [f32; 30] = [
            0.0, -8.597041, -9.025621, 17.5912, -50.0, 0.0, -8.867897, -10.036228, 19.373695,
            -50.0, 0.0, -7.3404503, -8.015412, 15.528098, -50.0, 0.0, -5.964992, -5.471871,
            11.17137, -38.25106, 0.0, -6.5874786, -5.974267, 12.624699, -39.630215, 0.0,
            -6.0539446, -5.490645, 11.508687, -37.035965,
        ];
        let mut index = 0;
        for model in reference_models() {
            for dt in [0.01, 0.05, 0.1] {
                let prepared = PreparedMpc::new(model, dt);
                for state in reference_states() {
                    let actual = prepared
                        .try_control(state)
                        .expect("prepared MPC reference solve");
                    assert!(actual.is_finite());
                    assert!(actual.abs() <= 50.0001, "actuator bound: {actual}");
                    assert!(
                        (actual - expected[index]).abs() < 1e-4,
                        "case {index}: actual={actual}, expected={}",
                        expected[index]
                    );
                    index += 1;
                }
            }
        }
        assert_eq!(index, expected.len());
    }
}
