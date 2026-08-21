from pathlib import Path

path = Path("rust_robotics_algo/src/slam/graph_slam.rs")
text = path.read_text()

# 1) Handle graphs that already satisfy the stopping criterion before iterating.
init_old = """        let initial_error = self.total_error();
        let mut current_error = initial_error;
        let mut lambda = self.config.initial_lambda;
        let mut iterations = 0;
"""
init_new = """        let initial_error = self.total_error();
        if initial_error <= self.config.convergence_threshold {
            return OptimizationResult {
                iterations: 0,
                initial_error,
                final_error: initial_error,
                converged: true,
                outliers_detected: self.outlier_count(),
            };
        }

        let mut current_error = initial_error;
        let mut lambda = self.config.initial_lambda;
        let mut iterations = 0;
        let mut converged = false;
"""
if text.count(init_old) != 2:
    raise SystemExit(f"expected 2 optimizer init blocks, found {text.count(init_old)}")
text = text.replace(init_old, init_new)

# 2) Measure improvement against the previous accepted objective, not the value
#    after it has already been overwritten. Rejected steps never imply convergence.
accept_old = """            if new_error < current_error {
                current_error = new_error;
                lambda *= self.config.lambda_down;

                let improvement = (current_error - new_error).abs() / (current_error + 1e-10);
                if improvement < self.config.convergence_threshold {
                    break;
                }
            } else {
"""
accept_new = """            if new_error < current_error {
                let previous_error = current_error;
                current_error = new_error;
                lambda *= self.config.lambda_down;

                let improvement =
                    (previous_error - current_error).abs() / previous_error.max(1e-10);
                if current_error <= self.config.convergence_threshold
                    || improvement < self.config.convergence_threshold
                {
                    converged = true;
                    break;
                }
            } else {
"""
if text.count(accept_old) != 2:
    raise SystemExit(f"expected 2 accepted-step blocks, found {text.count(accept_old)}")
text = text.replace(accept_old, accept_new)

# 3) OptimizationResult::converged means an actual stopping criterion fired.
result_old = "            converged: current_error < initial_error || initial_error < 1e-6,\n"
if text.count(result_old) != 2:
    raise SystemExit(f"expected 2 old convergence result expressions, found {text.count(result_old)}")
text = text.replace(result_old, "            converged,\n")

# 4) Add deterministic regressions that fail under the old one-accepted-step behavior.
test_anchor = """    #[test]
    fn test_loop_closure() {
"""
if text.count(test_anchor) != 1:
    raise SystemExit(f"expected one loop-closure test anchor, found {text.count(test_anchor)}")

tests = r'''    fn nonlinear_loop_graph(use_sparse_solver: bool, max_iterations: usize) -> GraphSlam {
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;
        graph.config.enable_outlier_rejection = false;
        graph.config.use_sparse_solver = use_sparse_solver;
        graph.config.max_iterations = max_iterations;
        graph.config.convergence_threshold = 1e-12;

        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(10.0, 0.5, PI / 2.0));
        graph.add_pose(Pose2D::new(10.5, 10.0, PI));
        graph.add_pose(Pose2D::new(0.5, 10.5, -PI / 2.0));
        graph.add_pose(Pose2D::new(0.8, 0.8, 0.1));

        let cov = Matrix3::from_diagonal(&Vector3::new(0.5, 0.5, 0.05));
        graph.add_odometry(0, 1, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(1, 2, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(2, 3, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(3, 4, Vector3::new(10.0, 0.0, PI / 2.0), &cov);

        let loop_cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(4, 0, Vector3::new(0.0, 0.0, 0.0), &loop_cov);
        graph
    }

    #[test]
    fn test_single_substantial_step_is_not_convergence() {
        for use_sparse in [false, true] {
            let mut graph = nonlinear_loop_graph(use_sparse, 1);
            let result = graph.optimize();

            assert!(
                result.final_error < result.initial_error,
                "first LM step should improve the nonlinear fixture (sparse={use_sparse})"
            );
            assert!(
                !result.converged,
                "objective improvement alone is not convergence (sparse={use_sparse})"
            );
        }
    }

    #[test]
    fn test_optimizer_continues_beyond_first_accepted_step() {
        for use_sparse in [false, true] {
            let mut one_step = nonlinear_loop_graph(use_sparse, 1);
            let one_step_result = one_step.optimize();
            assert!(one_step_result.final_error < one_step_result.initial_error);

            let mut full = nonlinear_loop_graph(use_sparse, 50);
            let full_result = full.optimize();

            assert!(
                full_result.iterations > 1,
                "nonlinear fixture should require more than one iteration (sparse={use_sparse})"
            );
            assert!(
                full_result.final_error < one_step_result.final_error,
                "continuing optimization must improve beyond the first accepted state (sparse={use_sparse})"
            );
        }
    }

'''
text = text.replace(test_anchor, tests + test_anchor, 1)

path.write_text(text)
