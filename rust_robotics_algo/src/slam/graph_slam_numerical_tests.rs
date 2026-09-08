//! Numerical oracles for both Graph SLAM assemblers (issue #4).
//!
//! Expected costs use the full information matrix directly, never the production
//! whitening code. Expected normal equations differentiate a separate f64
//! measurement model. Fixtures stay away from angle cuts and zero range.

use super::*;

fn fixture(fix_first_pose: bool) -> GraphSlam {
    let mut graph = GraphSlam::new();
    graph.max_poses = 0;
    graph.fix_first_pose = fix_first_pose;
    graph.config.enable_robust_kernel = false;
    graph.config.enable_outlier_rejection = false;
    graph.config.enable_loop_closure = false;
    graph.config.enable_relocalization = false;
    for pose in [
        Pose2D::new(0.4, -0.5, 0.3),
        Pose2D::new(1.7, 0.2, -0.2),
        Pose2D::new(2.6, 1.6, 0.5),
    ] {
        graph.add_pose(pose);
    }
    for landmark in [Landmark2D::new(2.5, -1.2), Landmark2D::new(-0.4, 2.6)] {
        graph.add_landmark(landmark);
    }
    let information = Matrix3::new(4.0, 1.0, 0.6, 1.0, 2.0, -0.4, 0.6, -0.4, 1.5);
    assert!(information.cholesky().is_some());
    for (index, (from_idx, to_idx)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
        graph.odometry_constraints.push(OdometryConstraint {
            from_idx,
            to_idx,
            measurement: Vector3::new(0.8 + index as f32 * 0.2, -0.3, 0.1),
            information,
            is_outlier: false,
            robust_weight: [1.0, 0.36, 0.0][index],
        });
    }
    let information = Matrix2::new(3.0, 0.8, 0.8, 1.4);
    assert!(information.cholesky().is_some());
    for pose_idx in 0..3 {
        for landmark_idx in 0..2 {
            graph.observation_constraints.push(ObservationConstraint {
                pose_idx,
                landmark_idx,
                measurement: Vector2::new(2.0 + pose_idx as f32 * 0.3, 0.2),
                information,
                is_outlier: false,
                robust_weight: [1.0, 0.36, 0.0][(pose_idx + landmark_idx) % 3],
            });
        }
    }
    graph
}

fn system(graph: &GraphSlam, sparse: bool) -> (DMatrix<f64>, DVector<f64>) {
    if !sparse {
        return graph.build_linear_system();
    }
    let (csr, residuals) = SparseSlamSolver::new().build_sparse_system(
        &graph.poses,
        &graph.landmarks,
        &graph.odometry_constraints,
        &graph.observation_constraints,
        graph.fix_first_pose,
    );
    let mut jacobian = DMatrix::zeros(csr.rows(), csr.cols());
    for (row, entries) in csr.outer_iterator().enumerate() {
        for (col, value) in entries.iter() {
            jacobian[(row, col)] = *value;
        }
    }
    (jacobian, residuals)
}

fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

// This oracle deliberately does not call the production residual, indexing, or
// analytic-Jacobian helpers. Only the public factor data and state layout enter.
fn raw_residuals(graph: &GraphSlam, state: &DVector<f64>) -> DVector<f64> {
    let first_free = usize::from(graph.fix_first_pose);
    let landmark_offset = 3 * (graph.poses.len() - first_free);
    let pose = |index: usize| -> Vector3<f64> {
        if graph.fix_first_pose && index == 0 {
            let p = graph.poses[0];
            Vector3::new(f64::from(p.x), f64::from(p.y), f64::from(p.theta))
        } else {
            let offset = 3 * (index - first_free);
            Vector3::new(state[offset], state[offset + 1], state[offset + 2])
        }
    };
    let mut values = Vec::new();
    for factor in &graph.odometry_constraints {
        let from = pose(factor.from_idx);
        let to = pose(factor.to_idx);
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let (s, c) = from.z.sin_cos();
        values.push(f64::from(factor.measurement[0]) - c * dx - s * dy);
        values.push(f64::from(factor.measurement[1]) + s * dx - c * dy);
        values.push(wrap(f64::from(factor.measurement[2]) - wrap(to.z - from.z)));
    }
    for factor in &graph.observation_constraints {
        let p = pose(factor.pose_idx);
        let offset = landmark_offset + 2 * factor.landmark_idx;
        let dx = state[offset] - p.x;
        let dy = state[offset + 1] - p.y;
        values.push(f64::from(factor.measurement[0]) - dx.hypot(dy));
        values.push(wrap(f64::from(factor.measurement[1]) - wrap(dy.atan2(dx) - p.z)));
    }
    DVector::from_vec(values)
}

fn information_matrix(graph: &GraphSlam) -> DMatrix<f64> {
    let count = graph.odometry_constraints.len() * 3 + graph.observation_constraints.len() * 2;
    let mut omega = DMatrix::zeros(count, count);
    let mut offset = 0;
    for factor in &graph.odometry_constraints {
        for row in 0..3 {
            for col in 0..3 {
                omega[(offset + row, offset + col)] =
                    f64::from(factor.information[(row, col)]) * factor.robust_weight;
            }
        }
        offset += 3;
    }
    for factor in &graph.observation_constraints {
        for row in 0..2 {
            for col in 0..2 {
                omega[(offset + row, offset + col)] =
                    f64::from(factor.information[(row, col)]) * factor.robust_weight;
            }
        }
        offset += 2;
    }
    omega
}

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(actual.is_finite() && expected.is_finite(), "{context}: non-finite value");
    let bound = tolerance * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "{context}: actual={actual}, expected={expected}, bound={bound}"
    );
}

fn check_factor_metrics(sparse: bool, odometry: bool) {
    for fixed in [false, true] {
        let graph = fixture(fixed);
        let (_, whitened) = system(&graph, sparse);
        let raw = raw_residuals(&graph, &graph.pack_state());
        let omega = information_matrix(&graph);
        let (offset, dimension, count) = if odometry {
            (0, 3, graph.odometry_constraints.len())
        } else {
            (3 * graph.odometry_constraints.len(), 2, graph.observation_constraints.len())
        };
        for index in 0..count {
            let begin = offset + dimension * index;
            let error = raw.rows(begin, dimension);
            let block = omega.view((begin, begin), (dimension, dimension));
            let expected = (error.transpose() * block * error)[0];
            // Geometry and whitening use f32 in production. 2e-6 relative/absolute
            // slack covers their rounding, not a changed covariance metric.
            close(whitened.rows(begin, dimension).norm_squared(), expected, 2e-6, "factor metric");
        }
    }
}

#[test]
fn dense_correlated_odometry_metric() {
    check_factor_metrics(false, true);
}

#[test]
fn dense_correlated_observation_metric() {
    check_factor_metrics(false, false);
}

#[test]
fn sparse_correlated_odometry_metric() {
    check_factor_metrics(true, true);
}

#[test]
fn sparse_correlated_observation_metric() {
    check_factor_metrics(true, false);
}

fn check_normal_equations(sparse: bool) {
    for fixed in [false, true] {
        let graph = fixture(fixed);
        let state = graph.pack_state();
        let raw = raw_residuals(&graph, &state);
        let omega = information_matrix(&graph);
        let (jacobian, whitened) = system(&graph, sparse);
        let mut numeric = DMatrix::zeros(raw.len(), state.len());
        // Central differences of an independent f64 model: h=1e-5 gives O(h^2)
        // truncation and O(epsilon_f64/h) cancellation, both well below the
        // production f32 rounding budget. No fixture crosses an angle wrap.
        let h = 1e-5;
        for col in 0..state.len() {
            let mut plus = state.clone();
            let mut minus = state.clone();
            plus[col] += h;
            minus[col] -= h;
            let derivative = (raw_residuals(&graph, &plus) - raw_residuals(&graph, &minus)) / (2.0 * h);
            numeric.set_column(col, &derivative);
        }
        let expected_hessian = numeric.transpose() * &omega * &numeric;
        let expected_gradient = numeric.transpose() * &omega * raw;
        let actual_hessian = jacobian.transpose() * &jacobian;
        let actual_gradient = jacobian.transpose() * whitened;
        assert_eq!(actual_hessian.shape(), expected_hessian.shape());
        for row in 0..state.len() {
            close(actual_gradient[row], expected_gradient[row], 2e-5, "finite-difference gradient");
            for col in 0..state.len() {
                close(actual_hessian[(row, col)], expected_hessian[(row, col)], 2e-5, "finite-difference normal matrix");
            }
        }
    }
}

#[test]
fn dense_correlated_normal_equations_match_finite_differences() {
    check_normal_equations(false);
}

#[test]
fn sparse_correlated_normal_equations_match_finite_differences() {
    check_normal_equations(true);
}

#[test]
fn dense_sparse_correlated_linear_systems_agree() {
    for fixed in [false, true] {
        let graph = fixture(fixed);
        let (dense_j, dense_r) = system(&graph, false);
        let (sparse_j, sparse_r) = system(&graph, true);
        assert_eq!(dense_j.shape(), sparse_j.shape());
        assert_eq!(dense_r.len(), sparse_r.len());
        for row in 0..dense_r.len() {
            close(dense_r[row], sparse_r[row], 2e-6, "backend residual");
            for col in 0..dense_j.ncols() {
                close(dense_j[(row, col)], sparse_j[(row, col)], 2e-6, "backend Jacobian");
            }
        }
    }
}

#[test]
fn correlated_odometry_optimization_reaches_closed_form_solution() {
    let measurements = [Vector3::new(1.4_f32, 0.2, 0.12), Vector3::new(0.7_f32, 1.1, -0.15)];
    let information = [
        Matrix3::new(9.0_f32, 2.0, 1.0, 2.0, 4.0, 0.2, 1.0, 0.2, 2.0),
        Matrix3::new(1.0_f32, -0.6, 0.1, -0.6, 3.0, 0.5, 0.1, 0.5, 4.0),
    ];
    // Two disagreeing measurements of one free pose make the weighting affect
    // the optimum. A zero-residual graph could hide a wrong whitening matrix.
    let a = information[0].map(f64::from);
    let b = information[1].map(f64::from);
    let za = measurements[0].map(f64::from);
    let zb = measurements[1].map(f64::from);
    let optimum = (a + b).lu().solve(&(a * za + b * zb)).unwrap();
    let expected_cost = ((za - optimum).transpose() * a * (za - optimum))[0]
        + ((zb - optimum).transpose() * b * (zb - optimum))[0];
    for sparse in [false, true] {
        let mut graph = GraphSlam::new();
        graph.fix_first_pose = true;
        graph.config.use_sparse_solver = sparse;
        graph.config.enable_robust_kernel = false;
        graph.config.enable_outlier_rejection = false;
        graph.config.enable_loop_closure = false;
        graph.config.enable_relocalization = false;
        graph.config.max_iterations = 100;
        graph.config.convergence_threshold = 1e-10;
        graph.add_pose(Pose2D::origin());
        graph.add_pose(Pose2D::new(-0.8, -0.4, 0.5));
        for index in 0..2 {
            graph.odometry_constraints.push(OdometryConstraint {
                from_idx: 0,
                to_idx: 1,
                measurement: measurements[index],
                information: information[index],
                is_outlier: false,
                robust_weight: 1.0,
            });
        }
        let result = graph.optimize();
        assert!(result.final_error < result.initial_error);
        let actual = graph.poses[1].to_vector().map(f64::from);
        // Sub-millimetre/radian state slack permits f32 objective stagnation near
        // the optimum; the incorrect metric produces much larger state errors.
        for axis in 0..3 {
            close(actual[axis], optimum[axis], 5e-4, "closed-form optimum");
        }
        close(result.final_error, expected_cost, 2e-5, "closed-form cost");
    }
}
