use nalgebra::{DMatrix, Matrix2, Matrix3, SymmetricEigen, Vector2, Vector3};
use rust_robotics_algo::path_planning::{AStarPlanner, DijkstraPlanner, Grid};
use rust_robotics_algo::slam::{
    motion_model, predict as ekf_predict, update as ekf_update, EkfSlamConfig, EkfSlamState,
    GraphSlam, Landmark2D, Observation as EkfObservation, Pose2D,
};

fn path_cost(path: &[(f32, f32)]) -> f32 {
    path.windows(2)
        .map(|segment| {
            let dx = segment[1].0 - segment[0].0;
            let dy = segment[1].1 - segment[0].1;
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

fn assert_astar_matches_dijkstra(
    grid: &Grid,
    start_cell: (usize, usize),
    goal_cell: (usize, usize),
) {
    let start = grid.grid_to_world(start_cell.0, start_cell.1);
    let goal = grid.grid_to_world(goal_cell.0, goal_cell.1);

    let astar = AStarPlanner::new(grid).plan(start, goal);
    let dijkstra = DijkstraPlanner::new(grid).plan(start, goal);

    assert_eq!(
        astar.success, dijkstra.success,
        "A* and Dijkstra must agree on reachability"
    );
    assert!(astar.success, "fixture must contain a valid path");

    let astar_cost = path_cost(&astar.path);
    let dijkstra_cost = path_cost(&dijkstra.path);

    // Both planners use the same 8-connected graph with edge costs 1 and sqrt(2).
    // Paths may differ when several optima exist, so compare total cost rather than
    // exact cell sequence. The 1e-5 relative tolerance only covers f32 summation
    // order; a wrong movement cost or inadmissible heuristic changes cost far more.
    let tolerance = 1e-5 * dijkstra_cost.max(1.0);
    assert!(
        (astar_cost - dijkstra_cost).abs() <= tolerance,
        "A* cost {astar_cost} differs from Dijkstra optimum {dijkstra_cost} (tol={tolerance})"
    );
}

#[test]
fn astar_matches_dijkstra_optimal_cost_on_deterministic_maps() {
    let open = Grid::new(20, 20, 1.0);
    assert_astar_matches_dijkstra(&open, (1, 2), (17, 15));

    let mut wall_with_gap = Grid::new(20, 20, 1.0);
    for y in 0..20 {
        if y != 11 {
            wall_with_gap.set_obstacle(9, y);
        }
    }
    assert_astar_matches_dijkstra(&wall_with_gap, (2, 3), (17, 16));

    let mut staggered = Grid::new(24, 18, 1.0);
    for x in 4..20 {
        if x != 7 && x != 16 {
            staggered.set_obstacle(x, 6);
        }
        if x != 11 && x != 18 {
            staggered.set_obstacle(x, 12);
        }
    }
    assert_astar_matches_dijkstra(&staggered, (1, 1), (22, 16));
}

#[test]
fn astar_and_dijkstra_agree_when_goal_is_unreachable() {
    let mut grid = Grid::new(16, 16, 1.0);
    for y in 0..16 {
        grid.set_obstacle(8, y);
    }

    let start = grid.grid_to_world(2, 8);
    let goal = grid.grid_to_world(13, 8);
    let astar = AStarPlanner::new(&grid).plan(start, goal);
    let dijkstra = DijkstraPlanner::new(&grid).plan(start, goal);

    assert!(!astar.success);
    assert!(!dijkstra.success);
    assert!(astar.path.is_empty());
    assert!(dijkstra.path.is_empty());
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn relative_pose(from: Pose2D, to: Pose2D) -> Vector3<f32> {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let c = from.theta.cos();
    let s = from.theta.sin();
    Vector3::new(
        c * dx + s * dy,
        -s * dx + c * dy,
        normalize_angle(to.theta - from.theta),
    )
}

fn observation(pose: Pose2D, landmark: Landmark2D) -> Vector2<f32> {
    let dx = landmark.x - pose.x;
    let dy = landmark.y - pose.y;
    Vector2::new(
        (dx * dx + dy * dy).sqrt(),
        normalize_angle(dy.atan2(dx) - pose.theta),
    )
}

fn graph_fixture(use_sparse_solver: bool) -> GraphSlam {
    let true_poses = [
        Pose2D::new(0.0, 0.0, 0.0),
        Pose2D::new(1.0, 0.1, 0.05),
        Pose2D::new(2.0, 0.45, 0.16),
        Pose2D::new(2.85, 1.05, 0.34),
    ];
    let true_landmarks = [
        Landmark2D::new(0.5, 2.5),
        Landmark2D::new(2.4, 2.8),
        Landmark2D::new(4.0, 0.6),
    ];

    let mut graph = GraphSlam::new();
    graph.max_poses = 0;
    graph.config.use_sparse_solver = use_sparse_solver;
    graph.config.enable_robust_kernel = false;
    graph.config.enable_outlier_rejection = false;
    graph.config.enable_loop_closure = false;
    graph.config.enable_relocalization = false;
    graph.config.max_iterations = 80;
    graph.config.convergence_threshold = 1e-9;

    // Keep the fixed first pose exact to remove gauge freedom; perturb every
    // other state so both backends must solve the same nonlinear problem.
    for pose in [
        true_poses[0],
        Pose2D::new(1.18, -0.08, 0.12),
        Pose2D::new(1.75, 0.72, 0.02),
        Pose2D::new(3.20, 0.70, 0.50),
    ] {
        graph.add_pose(pose);
    }
    for landmark in [
        Landmark2D::new(0.8, 2.2),
        Landmark2D::new(2.0, 3.15),
        Landmark2D::new(4.25, 0.25),
    ] {
        graph.add_landmark(landmark);
    }

    let odom_cov = Matrix3::from_diagonal(&Vector3::new(0.04, 0.04, 0.01));
    for i in 0..(true_poses.len() - 1) {
        graph.add_odometry(
            i,
            i + 1,
            relative_pose(true_poses[i], true_poses[i + 1]),
            &odom_cov,
        );
    }

    let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.05, 0.01));
    for (pose_idx, pose) in true_poses.iter().copied().enumerate() {
        for (landmark_idx, landmark) in true_landmarks.iter().copied().enumerate() {
            let z = observation(pose, landmark);
            graph.add_observation(pose_idx, landmark_idx, z[0], z[1], &obs_cov);
        }
    }

    graph
}

#[test]
fn dense_and_sparse_graph_slam_are_numerically_equivalent() {
    let mut dense = graph_fixture(false);
    let mut sparse = graph_fixture(true);

    let dense_result = dense.optimize();
    let sparse_result = sparse.optimize();

    assert!(dense_result.final_error < dense_result.initial_error);
    assert!(sparse_result.final_error < sparse_result.initial_error);

    // The two solvers assemble the same least-squares problem but use different
    // matrix representations/factorizations. A 1e-4 relative objective tolerance
    // and millimetre/milliradian-scale state tolerances allow normal f64/f32
    // roundoff while still catching Jacobian, indexing, or weighting divergence.
    let objective_scale = dense_result
        .final_error
        .abs()
        .max(sparse_result.final_error.abs())
        .max(1.0);
    let objective_tolerance = 1e-4 * objective_scale;
    assert!(
        (dense_result.final_error - sparse_result.final_error).abs() <= objective_tolerance,
        "dense objective {} differs from sparse objective {} (tol={})",
        dense_result.final_error,
        sparse_result.final_error,
        objective_tolerance
    );

    assert_eq!(dense.poses.len(), sparse.poses.len());
    for (index, (dense_pose, sparse_pose)) in dense.poses.iter().zip(&sparse.poses).enumerate() {
        assert!(
            (dense_pose.x - sparse_pose.x).abs() <= 5e-3,
            "pose {index} x differs: {} vs {}",
            dense_pose.x,
            sparse_pose.x
        );
        assert!(
            (dense_pose.y - sparse_pose.y).abs() <= 5e-3,
            "pose {index} y differs: {} vs {}",
            dense_pose.y,
            sparse_pose.y
        );
        assert!(
            normalize_angle(dense_pose.theta - sparse_pose.theta).abs() <= 5e-3,
            "pose {index} theta differs: {} vs {}",
            dense_pose.theta,
            sparse_pose.theta
        );
    }

    assert_eq!(dense.landmarks.len(), sparse.landmarks.len());
    for (index, (dense_lm, sparse_lm)) in dense.landmarks.iter().zip(&sparse.landmarks).enumerate()
    {
        assert!(
            (dense_lm.x - sparse_lm.x).abs() <= 5e-3,
            "landmark {index} x differs: {} vs {}",
            dense_lm.x,
            sparse_lm.x
        );
        assert!(
            (dense_lm.y - sparse_lm.y).abs() <= 5e-3,
            "landmark {index} y differs: {} vs {}",
            dense_lm.y,
            sparse_lm.y
        );
    }
}

fn ekf_observations(
    pose: &Vector3<f32>,
    landmarks: &[Vector2<f32>],
) -> Vec<(usize, EkfObservation)> {
    landmarks
        .iter()
        .enumerate()
        .map(|(index, landmark)| {
            let dx = landmark.x - pose.x;
            let dy = landmark.y - pose.y;
            (
                index,
                EkfObservation {
                    range: (dx * dx + dy * dy).sqrt(),
                    bearing: normalize_angle(dy.atan2(dx) - pose.z),
                },
            )
        })
        .collect()
}

fn assert_covariance_symmetric_psd(sigma: &DMatrix<f32>, context: &str) {
    assert_eq!(sigma.nrows(), sigma.ncols());

    let scale = sigma
        .iter()
        .fold(1.0_f32, |acc, value| acc.max(value.abs()));
    let symmetry_tolerance = 1e-5 * scale;
    let mut max_asymmetry = 0.0_f32;
    for row in 0..sigma.nrows() {
        for col in 0..sigma.ncols() {
            max_asymmetry = max_asymmetry.max((sigma[(row, col)] - sigma[(col, row)]).abs());
        }
    }
    assert!(
        max_asymmetry <= symmetry_tolerance,
        "{context}: covariance asymmetry {max_asymmetry} exceeds {symmetry_tolerance}"
    );

    // SymmetricEigen assumes a symmetric input, so explicitly symmetrize within
    // the already-checked roundoff tolerance before inspecting eigenvalues.
    let symmetric = (sigma + sigma.transpose()) * 0.5;
    let eigenvalues = SymmetricEigen::new(symmetric).eigenvalues;
    let min_eigenvalue = eigenvalues.iter().copied().fold(f32::INFINITY, f32::min);
    let psd_tolerance = 1e-5 * scale;
    assert!(
        min_eigenvalue >= -psd_tolerance,
        "{context}: covariance min eigenvalue {min_eigenvalue} is below -{psd_tolerance}"
    );
}

#[test]
fn ekf_slam_covariance_remains_symmetric_and_psd() {
    let config = EkfSlamConfig::default();
    let mut state = EkfSlamState::new();
    let landmarks = [
        Vector2::new(4.5, 2.0),
        Vector2::new(3.0, -3.5),
        Vector2::new(-2.5, 3.5),
    ];
    let mut truth = Vector3::new(0.0, 0.0, 0.0);

    // The first update augments the joint state with landmarks. Subsequent
    // predict/update cycles exercise both cross-covariance propagation and
    // measurement corrections. Noise-free synthetic observations make this
    // deterministic; the invariant concerns covariance algebra, not estimation
    // accuracy under a particular random draw.
    let initial_observations = ekf_observations(&truth, &landmarks);
    ekf_update(&mut state, &config, &initial_observations);
    assert_eq!(state.n_landmarks, landmarks.len());
    assert_covariance_symmetric_psd(&state.sigma, "after landmark augmentation");

    for step in 0..30 {
        let v = 0.8 + 0.1 * ((step % 3) as f32);
        let w = if step < 10 {
            0.12
        } else if step < 20 {
            -0.08
        } else {
            0.04
        };
        let dt = 0.1;

        truth = motion_model(&truth, v, w, dt);
        ekf_predict(&mut state, &config, v, w, dt);
        assert_covariance_symmetric_psd(&state.sigma, &format!("after predict step {step}"));

        let observations = ekf_observations(&truth, &landmarks);
        ekf_update(&mut state, &config, &observations);
        assert_covariance_symmetric_psd(&state.sigma, &format!("after update step {step}"));
    }
}
