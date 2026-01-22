use super::*;
use crate::prelude::*;

const N: usize = 12; // Prediction horizon

/// Total number of decision variables: (N+1) states + N controls
const N_VARS: usize = (N + 1) * NX + N * NU;
/// Total number of constraints: equality (dynamics) + inequality (bounds)
const N_EQ: usize = (N + 1) * NX;
const N_INEQ: usize = N_VARS;
const N_CONSTRAINTS: usize = N_EQ + N_INEQ;

/// MPC control using quadratic programming (requires "osqp" feature)
///
/// This function computes the optimal control input for an inverted pendulum
/// using Model Predictive Control with a quadratic cost function.
///
/// # Arguments
/// * `x` - Current state vector [position, velocity, angle, angular_velocity]
/// * `model` - System model parameters
/// * `dt` - Time step
///
/// # Returns
/// The optimal control input (force on cart)
#[cfg(feature = "osqp")]
pub fn mpc_control(x: Vector4, model: Model, dt: f32) -> f32 {
    use crate::control::LQR;
    use osqp::{CscMatrix, Problem, Settings};

    let (Ad, Bd) = model.model(dt);

    // Control input bounds (force in Newtons)
    let umin = -50.0_f64;
    let umax = 50.0_f64;

    // State bounds (using infinity for unconstrained states)
    use std::f64::INFINITY;
    let xmin: [f64; NX] = [-INFINITY, -INFINITY, -INFINITY, -INFINITY];
    let xmax: [f64; NX] = [INFINITY, INFINITY, INFINITY, INFINITY];

    // Stage cost weights - add position regulation to prevent drift
    // Base on model's Q but ensure position is penalized
    let Q = diag![1.0_f32, 1., 10., 1.]; // [pos, vel, angle, angular_vel]
    let R = model.R; // Use model's R matrix

    // Create a modified model with our Q for computing terminal cost
    let mut mpc_model = model;
    mpc_model.Q = Q;

    // Terminal cost: Use LQR cost-to-go (DARE solution) for stability
    // This ensures the MPC has a stabilizing terminal cost
    let QN = mpc_model.solve_DARE(Ad, Bd);

    // Initial state (from input) and reference state
    let x0_vec: [f64; NX] = [x[0] as f64, x[1] as f64, x[2] as f64, x[3] as f64];
    let xr: [f64; NX] = [0.0, 0.0, 0.0, 0.0]; // Reference: upright at origin

    // Build QP matrices
    // Decision variables: z = [x(0), x(1), ..., x(N), u(0), ..., u(N-1)]

    // Quadratic cost matrix P (block diagonal)
    let P_mat = block_diag!(kron!(eye!(N), Q), QN, kron!(eye!(N), R));

    // Linear cost vector q
    let q_part1 = kron!(ones!(N, 1), -dot!(Q, vector![xr[0] as f32, xr[1] as f32, xr[2] as f32, xr[3] as f32]));
    let q_part2 = -dot!(QN, vector![xr[0] as f32, xr[1] as f32, xr[2] as f32, xr[3] as f32]);
    let q_part3 = zeros!({ N * NU }, 1);
    let q_mat = vstack!(q_part1, q_part2, q_part3);

    // Dynamics constraints: x(k+1) = Ad * x(k) + Bd * u(k)
    // Rewritten as: -I * x(k+1) + Ad * x(k) + Bd * u(k) = 0
    let Ax = kron!(eye!(N + 1), -eye!(NX)) + kron!(eye!({ N + 1 }, -1), Ad);
    let Bu = kron!(vstack!(zeros!(1, N), eye!(N)), Bd);
    let Aeq = hstack!(Ax, Bu);

    // Inequality constraints (bounds on states and inputs)
    let Aineq = eye!((N + 1) * NX + N * NU);

    // Stack equality and inequality constraints
    let A_mat = vstack!(Aeq, Aineq);

    // Convert matrices to f64 arrays for OSQP
    let mut P_data: [[f64; N_VARS]; N_VARS] = [[0.0; N_VARS]; N_VARS];
    let mut A_data: [[f64; N_VARS]; N_CONSTRAINTS] = [[0.0; N_VARS]; N_CONSTRAINTS];
    let mut q_data: [f64; N_VARS] = [0.0; N_VARS];
    let mut l_data: [f64; N_CONSTRAINTS] = [0.0; N_CONSTRAINTS];
    let mut u_data: [f64; N_CONSTRAINTS] = [0.0; N_CONSTRAINTS];

    // Copy P matrix (symmetric, we use full matrix then extract upper tri)
    for i in 0..N_VARS {
        for j in 0..N_VARS {
            P_data[i][j] = P_mat[(i, j)] as f64;
        }
    }

    // Copy A matrix
    for i in 0..N_CONSTRAINTS {
        for j in 0..N_VARS {
            A_data[i][j] = A_mat[(i, j)] as f64;
        }
    }

    // Copy q vector
    for i in 0..N_VARS {
        q_data[i] = q_mat[(i, 0)] as f64;
    }

    // Build constraint bounds
    // Equality constraints: leq = ueq = [-x0, 0, 0, ...]
    for i in 0..NX {
        l_data[i] = -x0_vec[i];
        u_data[i] = -x0_vec[i];
    }
    for i in NX..N_EQ {
        l_data[i] = 0.0;
        u_data[i] = 0.0;
    }

    // Inequality constraints: bounds on states and inputs
    for k in 0..=N {
        for i in 0..NX {
            let idx = N_EQ + k * NX + i;
            l_data[idx] = xmin[i];
            u_data[idx] = xmax[i];
        }
    }
    for k in 0..N {
        let idx = N_EQ + (N + 1) * NX + k * NU;
        l_data[idx] = umin;
        u_data[idx] = umax;
    }

    // Create OSQP problem
    let P_csc = CscMatrix::from(&P_data).into_upper_tri();
    let A_csc = CscMatrix::from(&A_data);

    let settings = Settings::default().verbose(false);

    let mut prob = Problem::new(P_csc, &q_data, A_csc, &l_data, &u_data, &settings)
        .expect("Failed to setup OSQP problem");

    // Solve
    let result = prob.solve();

    // Extract first control input u(0)
    // Decision variables are ordered: [x(0)..x(N), u(0)..u(N-1)]
    // u(0) starts at index (N+1)*NX
    let u_idx = (N + 1) * NX;

    match result.x() {
        Some(solution) => solution[u_idx] as f32,
        None => {
            // Solver failed - return 0 as safe fallback
            // This can happen if the problem is infeasible
            0.0
        }
    }
}

/// Fallback MPC control when OSQP is not available
#[cfg(not(feature = "osqp"))]
pub fn mpc_control(_x: Vector4, _model: Model, _dt: f32) -> f32 {
    // Without OSQP, we cannot solve the QP problem
    // Return zero control input
    0.0
}

#[cfg(test)]
#[cfg(feature = "osqp")]
mod tests {
    use super::*;
    use std::time::Instant;

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
        println!("Initial state: pos={:.3}, vel={:.3}, angle={:.3}, omega={:.3}",
                 x[0], x[1], x[2], x[3]);

        let mut stable = true;
        for step in 0..100 {
            // Compute control
            let u = mpc_control(x, model, dt);

            // Apply dynamics: x_next = Ad * x + Bd * u
            x = Ad * x + Bd * u;

            // Print every 10 steps
            if step % 10 == 0 {
                println!("Step {:3}: pos={:7.3}, vel={:7.3}, angle={:7.4}, omega={:7.3}, u={:7.3}",
                         step, x[0], x[1], x[2], x[3], u);
            }

            // Check if unstable (angle > 1 rad or position > 10m)
            if x[2].abs() > 1.0 || x[0].abs() > 10.0 {
                println!("UNSTABLE at step {}", step);
                stable = false;
                break;
            }
        }

        println!("Final state: pos={:.3}, vel={:.3}, angle={:.4}, omega={:.3}",
                 x[0], x[1], x[2], x[3]);

        // Check that the angle has been stabilized (should be close to 0)
        assert!(stable, "MPC should stabilize the pendulum");
        assert!(x[2].abs() < 0.1, "Angle should be stabilized near 0, got {}", x[2]);
    }

    #[test]
    fn test_lqr_closed_loop_simulation() {
        use crate::control::{StateSpace, LQR};

        let dt = 0.05;
        let model = Model::default();
        let (Ad, Bd) = model.model(dt);

        let mut x = vector![0., 0., 0.2, 0.];

        println!("\n=== LQR Closed-Loop Simulation (for comparison) ===");
        println!("Initial state: pos={:.3}, vel={:.3}, angle={:.3}, omega={:.3}",
                 x[0], x[1], x[2], x[3]);

        for step in 0..100 {
            let u_vec = model.control(x, dt);
            let u = u_vec[0];
            x = Ad * x + Bd * u;

            if step % 10 == 0 {
                println!("Step {:3}: pos={:7.3}, vel={:7.3}, angle={:7.4}, omega={:7.3}, u={:7.3}",
                         step, x[0], x[1], x[2], x[3], u);
            }
        }

        println!("Final state: pos={:.3}, vel={:.3}, angle={:.4}, omega={:.3}",
                 x[0], x[1], x[2], x[3]);
    }
}
