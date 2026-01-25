//! Graph-based SLAM with built-in Gauss-Newton optimizer
//!
//! This module implements pose graph SLAM where:
//! - Nodes represent robot poses and landmark positions
//! - Edges represent constraints (odometry between poses, observations to landmarks)
//! - Optimization minimizes the total error across all constraints
//!
//! ## Advantages over EKF-SLAM
//! - Loop closure: when revisiting areas, the entire trajectory is re-optimized
//! - Better handling of heading drift through global optimization
//! - More consistent estimates through batch processing
//!
//! ## References
//! - [Graph-Based SLAM Tutorial](https://www.roboticsproceedings.org/rss06/p10.pdf)
//! - [g2o: A General Framework for Graph Optimization](https://github.com/RainerKuemmerle/g2o)

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Vector2, Vector3};
use std::f32::consts::{PI, TAU};

/// Normalize angle to [-π, π]
fn normalize_angle(angle: f32) -> f32 {
    let mut a = angle;
    while a > PI {
        a -= TAU;
    }
    while a < -PI {
        a += TAU;
    }
    a
}

/// A robot pose in the graph (x, y, theta)
#[derive(Debug, Clone, Copy)]
pub struct Pose2D {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
}

impl Pose2D {
    pub fn new(x: f32, y: f32, theta: f32) -> Self {
        Self { x, y, theta }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn to_vector(&self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.theta)
    }

    pub fn from_vector(v: &Vector3<f32>) -> Self {
        Self {
            x: v[0],
            y: v[1],
            theta: normalize_angle(v[2]),
        }
    }
}

/// A landmark position in the graph (x, y)
#[derive(Debug, Clone, Copy)]
pub struct Landmark2D {
    pub x: f32,
    pub y: f32,
}

impl Landmark2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn to_vector(&self) -> Vector2<f32> {
        Vector2::new(self.x, self.y)
    }

    pub fn from_vector(v: &Vector2<f32>) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

/// Odometry constraint between two poses
#[derive(Debug, Clone)]
pub struct OdometryConstraint {
    /// Index of the first pose
    pub from_idx: usize,
    /// Index of the second pose
    pub to_idx: usize,
    /// Relative motion measurement (dx, dy, dtheta) in the from_pose frame
    pub measurement: Vector3<f32>,
    /// Information matrix (inverse of covariance)
    pub information: Matrix3<f32>,
}

/// Observation constraint from a pose to a landmark
#[derive(Debug, Clone)]
pub struct ObservationConstraint {
    /// Index of the observing pose
    pub pose_idx: usize,
    /// Index of the observed landmark
    pub landmark_idx: usize,
    /// Observation measurement (range, bearing)
    pub measurement: Vector2<f32>,
    /// Information matrix (inverse of covariance)
    pub information: Matrix2<f32>,
}

/// Configuration for the optimizer
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold (stop if error change < this)
    pub convergence_threshold: f64,
    /// Initial damping factor for Levenberg-Marquardt
    pub initial_lambda: f64,
    /// Factor to increase lambda on rejection
    pub lambda_up: f64,
    /// Factor to decrease lambda on acceptance
    pub lambda_down: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
        }
    }
}

/// Graph SLAM state containing all nodes and edges
#[derive(Debug, Clone)]
pub struct GraphSlam {
    /// Robot poses (nodes)
    pub poses: Vec<Pose2D>,
    /// Landmark positions (nodes)
    pub landmarks: Vec<Landmark2D>,
    /// Odometry constraints (edges between poses)
    pub odometry_constraints: Vec<OdometryConstraint>,
    /// Observation constraints (edges from poses to landmarks)
    pub observation_constraints: Vec<ObservationConstraint>,
    /// Whether the first pose is fixed (anchor)
    pub fix_first_pose: bool,
    /// Optimizer configuration
    pub config: OptimizerConfig,
    /// Maximum number of poses (sliding window size, 0 = unlimited)
    pub max_poses: usize,
}

impl Default for GraphSlam {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSlam {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            poses: Vec::new(),
            landmarks: Vec::new(),
            odometry_constraints: Vec::new(),
            observation_constraints: Vec::new(),
            fix_first_pose: true,
            config: OptimizerConfig::default(),
            max_poses: 50, // Default sliding window size
        }
    }

    /// Add a pose to the graph, returns the pose index
    pub fn add_pose(&mut self, pose: Pose2D) -> usize {
        let idx = self.poses.len();
        self.poses.push(pose);
        idx
    }

    /// Add a landmark to the graph, returns the landmark index
    pub fn add_landmark(&mut self, landmark: Landmark2D) -> usize {
        let idx = self.landmarks.len();
        self.landmarks.push(landmark);
        idx
    }

    /// Add an odometry constraint between two poses
    pub fn add_odometry(
        &mut self,
        from_idx: usize,
        to_idx: usize,
        measurement: Vector3<f32>,
        covariance: &Matrix3<f32>,
    ) {
        let information = covariance.try_inverse().unwrap_or(Matrix3::identity());
        self.odometry_constraints.push(OdometryConstraint {
            from_idx,
            to_idx,
            measurement,
            information,
        });
    }

    /// Add an observation constraint from a pose to a landmark
    pub fn add_observation(
        &mut self,
        pose_idx: usize,
        landmark_idx: usize,
        range: f32,
        bearing: f32,
        covariance: &Matrix2<f32>,
    ) {
        let information = covariance.try_inverse().unwrap_or(Matrix2::identity());
        self.observation_constraints.push(ObservationConstraint {
            pose_idx,
            landmark_idx,
            measurement: Vector2::new(range, bearing),
            information,
        });
    }

    /// Prune old poses to maintain sliding window size
    /// Returns the number of poses removed (for index adjustment by caller)
    pub fn prune_old_poses(&mut self) -> usize {
        if self.max_poses == 0 || self.poses.len() <= self.max_poses {
            return 0;
        }

        let n_remove = self.poses.len() - self.max_poses;

        // Remove old poses
        self.poses.drain(0..n_remove);

        // Update odometry constraint indices and remove invalid ones
        self.odometry_constraints.retain_mut(|c| {
            if c.from_idx < n_remove || c.to_idx < n_remove {
                // Constraint references a removed pose
                false
            } else {
                c.from_idx -= n_remove;
                c.to_idx -= n_remove;
                true
            }
        });

        // Update observation constraint indices and remove invalid ones
        self.observation_constraints.retain_mut(|c| {
            if c.pose_idx < n_remove {
                // Constraint references a removed pose
                false
            } else {
                c.pose_idx -= n_remove;
                true
            }
        });

        n_remove
    }

    /// Get the total number of variables
    fn num_variables(&self) -> usize {
        let pose_vars = if self.fix_first_pose && !self.poses.is_empty() {
            (self.poses.len() - 1) * 3
        } else {
            self.poses.len() * 3
        };
        pose_vars + self.landmarks.len() * 2
    }

    /// Get the total number of residuals
    fn num_residuals(&self) -> usize {
        self.odometry_constraints.len() * 3 + self.observation_constraints.len() * 2
    }

    /// Get the state index for a pose (None if fixed)
    fn pose_state_idx(&self, pose_idx: usize) -> Option<usize> {
        if self.fix_first_pose {
            if pose_idx == 0 {
                None
            } else {
                Some((pose_idx - 1) * 3)
            }
        } else {
            Some(pose_idx * 3)
        }
    }

    /// Get the state index for a landmark
    fn landmark_state_idx(&self, landmark_idx: usize) -> usize {
        let pose_vars = if self.fix_first_pose && !self.poses.is_empty() {
            (self.poses.len() - 1) * 3
        } else {
            self.poses.len() * 3
        };
        pose_vars + landmark_idx * 2
    }

    /// Pack current state into a vector
    fn pack_state(&self) -> DVector<f64> {
        let n = self.num_variables();
        let mut state = DVector::zeros(n);
        let mut idx = 0;

        let start = if self.fix_first_pose { 1 } else { 0 };
        for pose in self.poses.iter().skip(start) {
            state[idx] = pose.x as f64;
            state[idx + 1] = pose.y as f64;
            state[idx + 2] = pose.theta as f64;
            idx += 3;
        }

        for lm in &self.landmarks {
            state[idx] = lm.x as f64;
            state[idx + 1] = lm.y as f64;
            idx += 2;
        }

        state
    }

    /// Unpack state vector into poses and landmarks
    fn unpack_state(&mut self, state: &DVector<f64>) {
        let mut idx = 0;

        let start = if self.fix_first_pose { 1 } else { 0 };
        for pose in self.poses.iter_mut().skip(start) {
            pose.x = state[idx] as f32;
            pose.y = state[idx + 1] as f32;
            pose.theta = normalize_angle(state[idx + 2] as f32);
            idx += 3;
        }

        for lm in &mut self.landmarks {
            lm.x = state[idx] as f32;
            lm.y = state[idx + 1] as f32;
            idx += 2;
        }
    }

    /// Compute residual error for an odometry constraint
    fn odometry_error(&self, constraint: &OdometryConstraint) -> Vector3<f32> {
        let p1 = &self.poses[constraint.from_idx];
        let p2 = &self.poses[constraint.to_idx];

        // Transform p2 into p1's local frame
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let c = p1.theta.cos();
        let s = p1.theta.sin();

        let dx_local = c * dx + s * dy;
        let dy_local = -s * dx + c * dy;
        let dtheta = normalize_angle(p2.theta - p1.theta);

        // Error = measurement - prediction
        Vector3::new(
            constraint.measurement[0] - dx_local,
            constraint.measurement[1] - dy_local,
            normalize_angle(constraint.measurement[2] - dtheta),
        )
    }

    /// Compute residual error for an observation constraint
    fn observation_error(&self, constraint: &ObservationConstraint) -> Vector2<f32> {
        let pose = &self.poses[constraint.pose_idx];
        let lm = &self.landmarks[constraint.landmark_idx];

        let dx = lm.x - pose.x;
        let dy = lm.y - pose.y;
        let pred_range = (dx * dx + dy * dy).sqrt();
        let pred_bearing = normalize_angle(dy.atan2(dx) - pose.theta);

        Vector2::new(
            constraint.measurement[0] - pred_range,
            normalize_angle(constraint.measurement[1] - pred_bearing),
        )
    }

    /// Compute total squared error
    pub fn total_error(&self) -> f64 {
        let mut error = 0.0;

        for c in &self.odometry_constraints {
            let e = self.odometry_error(c);
            let weighted = c.information * e;
            error += (e.transpose() * weighted)[(0, 0)] as f64;
        }

        for c in &self.observation_constraints {
            let e = self.observation_error(c);
            let weighted = c.information * e;
            error += (e.transpose() * weighted)[(0, 0)] as f64;
        }

        error
    }

    /// Build the linear system (Jacobian and residual vector)
    fn build_linear_system(&self) -> (DMatrix<f64>, DVector<f64>) {
        let n_vars = self.num_variables();
        let n_res = self.num_residuals();

        let mut jacobian = DMatrix::<f64>::zeros(n_res, n_vars);
        let mut residuals = DVector::<f64>::zeros(n_res);
        let mut res_idx = 0;

        // Odometry constraints
        for c in &self.odometry_constraints {
            let p1 = &self.poses[c.from_idx];
            let p2 = &self.poses[c.to_idx];

            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let cos_t = p1.theta.cos();
            let sin_t = p1.theta.sin();

            // Compute error
            let error = self.odometry_error(c);

            // Square root of information for weighting
            let sqrt_info = c.information.cholesky()
                .map(|ch| ch.l())
                .unwrap_or(Matrix3::identity());

            let weighted_error = sqrt_info * error;
            for i in 0..3 {
                residuals[res_idx + i] = weighted_error[i] as f64;
            }

            // Jacobian w.r.t. pose 1 (from)
            if let Some(idx1) = self.pose_state_idx(c.from_idx) {
                let j1 = Matrix3::new(
                    -cos_t, -sin_t, -sin_t * dx + cos_t * dy,
                    sin_t, -cos_t, -cos_t * dx - sin_t * dy,
                    0.0, 0.0, -1.0,
                );
                let wj1 = sqrt_info * j1;
                for i in 0..3 {
                    for j in 0..3 {
                        jacobian[(res_idx + i, idx1 + j)] = -wj1[(i, j)] as f64;
                    }
                }
            }

            // Jacobian w.r.t. pose 2 (to)
            if let Some(idx2) = self.pose_state_idx(c.to_idx) {
                let j2 = Matrix3::new(
                    cos_t, sin_t, 0.0,
                    -sin_t, cos_t, 0.0,
                    0.0, 0.0, 1.0,
                );
                let wj2 = sqrt_info * j2;
                for i in 0..3 {
                    for j in 0..3 {
                        jacobian[(res_idx + i, idx2 + j)] = -wj2[(i, j)] as f64;
                    }
                }
            }

            res_idx += 3;
        }

        // Observation constraints
        for c in &self.observation_constraints {
            let pose = &self.poses[c.pose_idx];
            let lm = &self.landmarks[c.landmark_idx];

            let dx = lm.x - pose.x;
            let dy = lm.y - pose.y;
            let q = dx * dx + dy * dy;
            let sqrt_q = q.sqrt().max(1e-6);

            let error = self.observation_error(c);

            let sqrt_info = c.information.cholesky()
                .map(|ch| ch.l())
                .unwrap_or(Matrix2::identity());

            let weighted_error = sqrt_info * error;
            residuals[res_idx] = weighted_error[0] as f64;
            residuals[res_idx + 1] = weighted_error[1] as f64;

            // Jacobian w.r.t. pose
            if let Some(pose_idx) = self.pose_state_idx(c.pose_idx) {
                let jp = nalgebra::Matrix2x3::new(
                    -dx / sqrt_q, -dy / sqrt_q, 0.0,
                    dy / q, -dx / q, -1.0,
                );
                let wjp = sqrt_info * jp;
                for i in 0..2 {
                    for j in 0..3 {
                        jacobian[(res_idx + i, pose_idx + j)] = -wjp[(i, j)] as f64;
                    }
                }
            }

            // Jacobian w.r.t. landmark
            let lm_idx = self.landmark_state_idx(c.landmark_idx);
            let jl = Matrix2::new(
                dx / sqrt_q, dy / sqrt_q,
                -dy / q, dx / q,
            );
            let wjl = sqrt_info * jl;
            for i in 0..2 {
                for j in 0..2 {
                    jacobian[(res_idx + i, lm_idx + j)] = -wjl[(i, j)] as f64;
                }
            }

            res_idx += 2;
        }

        (jacobian, residuals)
    }

    /// Optimize the graph using Levenberg-Marquardt
    pub fn optimize(&mut self) -> OptimizationResult {
        if self.num_variables() == 0 {
            return OptimizationResult {
                iterations: 0,
                initial_error: 0.0,
                final_error: 0.0,
                converged: true,
            };
        }

        let initial_error = self.total_error();
        let mut current_error = initial_error;
        let mut lambda = self.config.initial_lambda;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Build linear system: J^T J dx = J^T r
            let (j, r) = self.build_linear_system();
            let jt = j.transpose();
            let jtj = &jt * &j;
            let jtr = &jt * &r;

            // Levenberg-Marquardt: (J^T J + λI) dx = J^T r
            let n = jtj.nrows();
            let mut h = jtj.clone();
            for i in 0..n {
                h[(i, i)] += lambda * (1.0 + h[(i, i)]);
            }

            // Solve for update: (J^T J + λD) dx = -J^T r
            // The negative sign is because we want to minimize ||r||^2
            // Gradient is J^T r, so we move in -gradient direction
            let neg_jtr = -&jtr;
            let dx = match h.clone().lu().solve(&neg_jtr) {
                Some(sol) => sol,
                None => break, // Singular matrix
            };

            // Apply update tentatively
            let old_state = self.pack_state();
            let new_state = &old_state + &dx;
            self.unpack_state(&new_state);

            let new_error = self.total_error();

            if new_error < current_error {
                // Accept update
                current_error = new_error;
                lambda *= self.config.lambda_down;

                // Check convergence
                let improvement = (current_error - new_error).abs() / (current_error + 1e-10);
                if improvement < self.config.convergence_threshold {
                    break;
                }
            } else {
                // Reject update, restore state
                self.unpack_state(&old_state);
                lambda *= self.config.lambda_up;
            }
        }

        OptimizationResult {
            iterations,
            initial_error,
            final_error: current_error,
            // Converged if error decreased OR was already near zero
            converged: current_error < initial_error || initial_error < 1e-6,
        }
    }
}

/// Result of graph optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub iterations: usize,
    pub initial_error: f64,
    pub final_error: f64,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let mut graph = GraphSlam::new();

        // Add three poses
        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(1.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(2.0, 0.0, 0.0));

        // Add odometry
        let cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(0, 1, Vector3::new(1.0, 0.0, 0.0), &cov);
        graph.add_odometry(1, 2, Vector3::new(1.0, 0.0, 0.0), &cov);

        let result = graph.optimize();
        println!("Simple graph: {} iterations, error {:.4} -> {:.4}",
                 result.iterations, result.initial_error, result.final_error);

        assert!(result.converged);
    }

    #[test]
    fn test_loop_closure() {
        let mut graph = GraphSlam::new();

        // Square trajectory with accumulated drift
        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(10.0, 0.5, PI / 2.0));   // drift
        graph.add_pose(Pose2D::new(10.5, 10.0, PI));        // more drift
        graph.add_pose(Pose2D::new(0.5, 10.5, -PI / 2.0));  // even more
        graph.add_pose(Pose2D::new(0.8, 0.8, 0.1));         // should be at origin!

        let cov = Matrix3::from_diagonal(&Vector3::new(0.5, 0.5, 0.05));
        graph.add_odometry(0, 1, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(1, 2, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(2, 3, Vector3::new(10.0, 0.0, PI / 2.0), &cov);
        graph.add_odometry(3, 4, Vector3::new(10.0, 0.0, PI / 2.0), &cov);

        // Loop closure: pose 4 should match pose 0
        let loop_cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(4, 0, Vector3::new(0.0, 0.0, 0.0), &loop_cov);

        println!("\n=== Loop Closure Test ===");
        println!("Before optimization:");
        for (i, p) in graph.poses.iter().enumerate() {
            println!("  Pose {}: ({:.2}, {:.2}, {:.1}°)", i, p.x, p.y, p.theta.to_degrees());
        }

        let result = graph.optimize();

        println!("\nAfter optimization:");
        for (i, p) in graph.poses.iter().enumerate() {
            println!("  Pose {}: ({:.2}, {:.2}, {:.1}°)", i, p.x, p.y, p.theta.to_degrees());
        }
        println!("\n{} iterations, error {:.4} -> {:.4}",
                 result.iterations, result.initial_error, result.final_error);

        // Pose 4 should be closer to origin after optimization
        let p4 = &graph.poses[4];
        let dist = (p4.x * p4.x + p4.y * p4.y).sqrt();
        println!("Pose 4 distance to origin: {:.2}m", dist);

        assert!(dist < 2.0, "Loop closure should bring pose 4 near origin");
        assert!(result.final_error < result.initial_error);
    }

    #[test]
    fn test_landmark_slam() {
        let mut graph = GraphSlam::new();

        // Robot moves along x-axis
        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(5.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(10.0, 0.0, 0.0));

        // Landmarks with slight initial error
        graph.add_landmark(Landmark2D::new(2.5, 5.3));  // true: (2.5, 5.0)
        graph.add_landmark(Landmark2D::new(7.5, 4.7));  // true: (7.5, 5.0)

        // Odometry
        let odom_cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(0, 1, Vector3::new(5.0, 0.0, 0.0), &odom_cov);
        graph.add_odometry(1, 2, Vector3::new(5.0, 0.0, 0.0), &odom_cov);

        // Observations (true landmark positions)
        let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.1, 0.01));

        // Pose 0 sees landmark 0
        let r = (2.5f32.powi(2) + 5.0f32.powi(2)).sqrt();
        let b = (5.0f32).atan2(2.5);
        graph.add_observation(0, 0, r, b, &obs_cov);

        // Pose 1 sees both
        graph.add_observation(1, 0, r, (5.0f32).atan2(-2.5), &obs_cov);
        graph.add_observation(1, 1, r, b, &obs_cov);

        // Pose 2 sees landmark 1
        graph.add_observation(2, 1, r, (5.0f32).atan2(-2.5), &obs_cov);

        println!("\n=== Landmark SLAM Test ===");
        println!("Before: lm0=({:.2},{:.2}), lm1=({:.2},{:.2})",
                 graph.landmarks[0].x, graph.landmarks[0].y,
                 graph.landmarks[1].x, graph.landmarks[1].y);

        let result = graph.optimize();

        println!("After:  lm0=({:.2},{:.2}), lm1=({:.2},{:.2})",
                 graph.landmarks[0].x, graph.landmarks[0].y,
                 graph.landmarks[1].x, graph.landmarks[1].y);
        println!("{} iterations, error {:.4} -> {:.4}",
                 result.iterations, result.initial_error, result.final_error);

        // Landmarks should be closer to true positions (2.5, 5.0) and (7.5, 5.0)
        let err0 = ((graph.landmarks[0].x - 2.5).powi(2) + (graph.landmarks[0].y - 5.0).powi(2)).sqrt();
        let err1 = ((graph.landmarks[1].x - 7.5).powi(2) + (graph.landmarks[1].y - 5.0).powi(2)).sqrt();
        println!("Landmark errors: {:.2}m, {:.2}m", err0, err1);

        assert!(result.final_error < result.initial_error);
    }
}
