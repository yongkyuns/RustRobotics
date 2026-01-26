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
//! ## Features
//! - Robust kernels (Huber, Cauchy, Tukey) for outlier resistance
//! - Chi-squared test for outlier detection and flagging
//! - Iteratively Reweighted Least Squares (IRLS) for robust optimization
//!
//! ## References
//! - [Graph-Based SLAM Tutorial](https://www.roboticsproceedings.org/rss06/p10.pdf)
//! - [g2o: A General Framework for Graph Optimization](https://github.com/RainerKuemmerle/g2o)

use crate::slam::loop_closure::{LoopClosure, LoopClosureConfig, LoopClosureDetector};
use crate::slam::marginalization::PriorConstraint;
use crate::slam::robust_kernels::{
    chi_squared, compute_mad, CauchyKernel, HuberKernel, RobustKernel, RobustKernelType,
    TrivialKernel, TukeyKernel,
};
use crate::slam::sparse_solver::SparseSlamSolver;
use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Vector2, Vector3};
use std::f32::consts::{PI, TAU};
use std::io::Write;

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
    /// Whether this constraint is flagged as an outlier
    pub is_outlier: bool,
    /// Robust weight from the last optimization (1.0 = full weight)
    pub robust_weight: f64,
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
    /// Whether this constraint is flagged as an outlier
    pub is_outlier: bool,
    /// Robust weight from the last optimization (1.0 = full weight)
    pub robust_weight: f64,
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
    /// Enable robust kernel for outlier resistance
    pub enable_robust_kernel: bool,
    /// Type of robust kernel to use
    pub robust_kernel_type: RobustKernelType,
    /// Custom robust kernel scale (None = auto-scale using MAD)
    pub robust_kernel_scale: Option<f64>,
    /// Enable chi-squared outlier detection
    pub enable_outlier_rejection: bool,
    /// Chi-squared confidence level for outlier detection (0.95 or 0.99)
    pub outlier_chi2_confidence: f64,
    /// Use sparse solver for better performance with large graphs
    pub use_sparse_solver: bool,
    /// Enable marginalization when pruning old poses (preserves information)
    pub enable_marginalization: bool,
    /// Enable automatic loop closure detection
    pub enable_loop_closure: bool,
    /// Proximity threshold for loop closure candidates (meters)
    pub loop_closure_proximity: f32,
    /// Minimum pose separation for loop closure candidates
    pub loop_closure_min_separation: usize,
    /// Enable re-localization when returning from blind navigation with extreme residuals
    pub enable_relocalization: bool,
    /// Minimum landmarks required for re-localization (must have high observation count)
    pub relocalization_min_landmarks: usize,
    /// Minimum observation count for a landmark to be considered "confident"
    pub relocalization_min_obs_count: usize,
    /// Mahalanobis threshold to trigger re-localization (very high = extreme mismatch)
    pub relocalization_mahal_threshold: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            enable_robust_kernel: true,
            robust_kernel_type: RobustKernelType::Huber,
            robust_kernel_scale: None,
            enable_outlier_rejection: true,
            outlier_chi2_confidence: 0.95,
            use_sparse_solver: true,
            enable_marginalization: false,
            enable_loop_closure: false,
            loop_closure_proximity: 2.0,
            loop_closure_min_separation: 10,
            enable_relocalization: true,
            relocalization_min_landmarks: 3,
            relocalization_min_obs_count: 20,
            relocalization_mahal_threshold: 100.0,
        }
    }
}

/// Diagnostic data for debugging SLAM issues
#[derive(Debug, Clone, Default)]
pub struct SlamDiagnostics {
    /// Current pose index
    pub current_pose_idx: usize,
    /// Number of poses in graph
    pub num_poses: usize,
    /// Number of landmarks
    pub num_landmarks: usize,
    /// Number of odometry constraints
    pub num_odom_constraints: usize,
    /// Number of observation constraints
    pub num_obs_constraints: usize,
    /// Poses since last observation (blind navigation counter)
    pub poses_since_observation: usize,
    /// Per-landmark info: (landmark_idx, last_obs_pose, total_observations)
    pub landmark_info: Vec<(usize, Option<usize>, usize)>,
    /// Current observation residuals: (landmark_idx, range_residual, bearing_residual, mahalanobis_sq)
    pub observation_residuals: Vec<(usize, f32, f32, f64)>,
    /// Landmark position errors if true positions known: (landmark_idx, error)
    pub landmark_errors: Vec<(usize, f32)>,
    /// Optimization result
    pub last_opt_iterations: usize,
    pub last_opt_initial_error: f64,
    pub last_opt_final_error: f64,
    /// Outlier counts (odom, obs)
    pub outlier_counts: (usize, usize),
    /// Loop closures detected this step
    pub loop_closures_detected: usize,
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
    /// Prior constraints from marginalization
    pub prior_constraints: Vec<PriorConstraint>,
    /// Whether the first pose is fixed (anchor)
    pub fix_first_pose: bool,
    /// Optimizer configuration
    pub config: OptimizerConfig,
    /// Maximum number of poses (sliding window size, 0 = unlimited)
    pub max_poses: usize,
    /// Poses since last observation (for blind navigation detection)
    pub poses_since_observation: usize,
    /// Last pose that observed each landmark
    pub landmark_last_obs_pose: Vec<Option<usize>>,
    /// Number of observations per landmark
    pub landmark_obs_count: Vec<usize>,
    /// Enable diagnostic logging
    pub enable_diagnostics: bool,
    /// Diagnostic data (updated each optimization)
    pub diagnostics: SlamDiagnostics,
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
            prior_constraints: Vec::new(),
            fix_first_pose: true,
            config: OptimizerConfig::default(),
            max_poses: 50,
            poses_since_observation: 0,
            landmark_last_obs_pose: Vec::new(),
            landmark_obs_count: Vec::new(),
            enable_diagnostics: false,
            diagnostics: SlamDiagnostics::default(),
        }
    }

    /// Add a pose to the graph, returns the pose index
    pub fn add_pose(&mut self, pose: Pose2D) -> usize {
        let idx = self.poses.len();
        self.poses.push(pose);
        self.poses_since_observation += 1;
        idx
    }

    /// Add a landmark to the graph, returns the landmark index
    pub fn add_landmark(&mut self, landmark: Landmark2D) -> usize {
        let idx = self.landmarks.len();
        self.landmarks.push(landmark);
        self.landmark_last_obs_pose.push(None);
        self.landmark_obs_count.push(0);
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
            is_outlier: false,
            robust_weight: 1.0,
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

        // Track observation statistics
        if landmark_idx < self.landmark_last_obs_pose.len() {
            self.landmark_last_obs_pose[landmark_idx] = Some(pose_idx);
            self.landmark_obs_count[landmark_idx] += 1;
        }

        // Reset blind navigation counter when we see any landmark
        self.poses_since_observation = 0;

        self.observation_constraints.push(ObservationConstraint {
            pose_idx,
            landmark_idx,
            measurement: Vector2::new(range, bearing),
            information,
            is_outlier: false,
            robust_weight: 1.0,
        });
    }

    /// Prune old poses to maintain sliding window size
    pub fn prune_old_poses(&mut self) -> usize {
        if self.max_poses == 0 || self.poses.len() <= self.max_poses {
            return 0;
        }

        let n_remove = self.poses.len() - self.max_poses;

        // Remove old poses
        self.poses.drain(0..n_remove);

        // Update odometry constraint indices
        self.odometry_constraints.retain_mut(|c| {
            if c.from_idx < n_remove || c.to_idx < n_remove {
                false
            } else {
                c.from_idx -= n_remove;
                c.to_idx -= n_remove;
                true
            }
        });

        // Update observation constraint indices
        self.observation_constraints.retain_mut(|c| {
            if c.pose_idx < n_remove {
                false
            } else {
                c.pose_idx -= n_remove;
                true
            }
        });

        // Update landmark last observation poses
        for last_obs in self.landmark_last_obs_pose.iter_mut() {
            if let Some(pose_idx) = last_obs {
                if *pose_idx < n_remove {
                    *last_obs = None;
                } else {
                    *pose_idx -= n_remove;
                }
            }
        }

        n_remove
    }

    /// Get number of poses since last observation (blind navigation duration)
    pub fn get_blind_duration(&self) -> usize {
        self.poses_since_observation
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

        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let c = p1.theta.cos();
        let s = p1.theta.sin();

        let dx_local = c * dx + s * dy;
        let dy_local = -s * dx + c * dy;
        let dtheta = normalize_angle(p2.theta - p1.theta);

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

    /// Collect all residual magnitudes for MAD computation
    fn collect_residuals(&self) -> Vec<f64> {
        let mut residuals = Vec::with_capacity(
            self.odometry_constraints.len() * 3 + self.observation_constraints.len() * 2,
        );

        for c in &self.odometry_constraints {
            let e = self.odometry_error(c);
            for i in 0..3 {
                residuals.push(e[i] as f64);
            }
        }

        for c in &self.observation_constraints {
            let e = self.observation_error(c);
            for i in 0..2 {
                residuals.push(e[i] as f64);
            }
        }

        residuals
    }

    /// Helper: Compute Mahalanobis sq for odometry
    fn odometry_mahalanobis_sq(&self, constraint: &OdometryConstraint) -> f64 {
        let e = self.odometry_error(constraint);
        let weighted = constraint.information * e;
        (e.transpose() * weighted)[(0, 0)] as f64
    }

    /// Helper: Compute Mahalanobis sq for observation
    fn observation_mahalanobis_sq(&self, constraint: &ObservationConstraint) -> f64 {
        let e = self.observation_error(constraint);
        let weighted = constraint.information * e;
        (e.transpose() * weighted)[(0, 0)] as f64
    }

    /// Update outlier flags and robust weights
    pub fn update_robust_weights(&mut self) {
        if !self.config.enable_robust_kernel && !self.config.enable_outlier_rejection {
            return;
        }

        let scale = self.config.robust_kernel_scale.unwrap_or_else(|| {
            let residuals = self.collect_residuals();
            let mad = compute_mad(&residuals);
            1.4826 * mad
        });

        let kernel: Box<dyn RobustKernel> = match self.config.robust_kernel_type {
            RobustKernelType::None => Box::new(TrivialKernel),
            RobustKernelType::Huber => Box::new(HuberKernel::new(1.345 * scale)),
            RobustKernelType::Cauchy => Box::new(CauchyKernel::new(2.3849 * scale)),
            RobustKernelType::Tukey => Box::new(TukeyKernel::new(4.685 * scale)),
        };

        let enable_robust = self.config.enable_robust_kernel;
        let enable_outlier = self.config.enable_outlier_rejection;
        let chi2_conf = self.config.outlier_chi2_confidence;

        // Odometry constraints
        let odom_mahal: Vec<f64> = self
            .odometry_constraints
            .iter()
            .map(|c| self.odometry_mahalanobis_sq(c))
            .collect();

        for (c, mahal_sq) in self.odometry_constraints.iter_mut().zip(odom_mahal.iter()) {
            if enable_outlier {
                c.is_outlier = chi_squared::is_outlier(*mahal_sq, 3, chi2_conf);
            }
            if enable_robust && !c.is_outlier {
                c.robust_weight = kernel.weight(mahal_sq.sqrt());
            } else if c.is_outlier {
                c.robust_weight = 0.0;
            } else {
                c.robust_weight = 1.0;
            }
        }

        // Observation constraints
        let obs_mahal: Vec<f64> = self
            .observation_constraints
            .iter()
            .map(|c| self.observation_mahalanobis_sq(c))
            .collect();

        for (c, mahal_sq) in self.observation_constraints.iter_mut().zip(obs_mahal.iter()) {
            if enable_outlier {
                c.is_outlier = chi_squared::is_outlier(*mahal_sq, 2, chi2_conf);
            }
            if enable_robust && !c.is_outlier {
                c.robust_weight = kernel.weight(mahal_sq.sqrt());
            } else if c.is_outlier {
                c.robust_weight = 0.0;
            } else {
                c.robust_weight = 1.0;
            }
        }
    }

    /// Get the number of detected outliers
    pub fn outlier_count(&self) -> (usize, usize) {
        let odom = self.odometry_constraints.iter().filter(|c| c.is_outlier).count();
        let obs = self.observation_constraints.iter().filter(|c| c.is_outlier).count();
        (odom, obs)
    }

    /// Detect loop closures for the current pose
    pub fn detect_loop_closures(&self) -> Vec<LoopClosure> {
        if !self.config.enable_loop_closure || self.poses.is_empty() {
            return Vec::new();
        }

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = self.config.loop_closure_proximity;
        config.min_temporal_separation = self.config.loop_closure_min_separation;

        let detector = LoopClosureDetector::with_config(config);
        let current_idx = self.poses.len() - 1;

        detector.detect(
            &self.poses,
            &self.landmarks,
            &self.observation_constraints,
            current_idx,
        )
    }

    /// Add a loop closure constraint
    pub fn add_loop_closure(&mut self, closure: &LoopClosure) {
        let covariance = Matrix3::from_diagonal(&Vector3::new(0.2, 0.2, 0.05));
        self.add_odometry(
            closure.from_pose_idx,
            closure.to_pose_idx,
            closure.transform,
            &covariance,
        );
    }

    /// Compute pose from landmark observations using iterative least-squares.
    /// Returns None if fewer than min_landmarks are available or optimization fails.
    ///
    /// This is used for re-localization when the current pose estimate is way off.
    /// Given range-bearing observations to landmarks with known positions, we find
    /// the pose that best explains all observations.
    pub fn compute_pose_from_observations(
        &self,
        observations: &[(usize, f32, f32)], // (landmark_idx, range, bearing)
        initial_guess: Option<Pose2D>,
        max_iterations: usize,
    ) -> Option<Pose2D> {
        if observations.len() < 2 {
            return None;
        }

        // Filter to only use landmarks that exist
        let valid_obs: Vec<_> = observations
            .iter()
            .filter(|(lm_idx, _, _)| *lm_idx < self.landmarks.len())
            .cloned()
            .collect();

        if valid_obs.len() < 2 {
            return None;
        }

        // Initial guess: use provided or compute from first two observations
        let mut pose = initial_guess.unwrap_or_else(|| {
            // Simple triangulation from first observation
            let (lm_idx, range, bearing) = valid_obs[0];
            let lm = &self.landmarks[lm_idx];
            // Assume robot is at distance `range` from landmark in direction opposite to bearing
            // This is a rough guess; the iterative solver will refine it
            Pose2D::new(
                lm.x - range * bearing.cos(),
                lm.y - range * bearing.sin(),
                0.0,
            )
        });

        // Gauss-Newton iterations
        for _ in 0..max_iterations {
            let mut h_sum = Matrix3::<f64>::zeros();
            let mut b_sum = Vector3::<f64>::zeros();
            let mut _total_error = 0.0;

            for &(lm_idx, range_obs, bearing_obs) in &valid_obs {
                let lm = &self.landmarks[lm_idx];

                // Predicted observation from current pose estimate
                let dx = lm.x - pose.x;
                let dy = lm.y - pose.y;
                let range_pred = (dx * dx + dy * dy).sqrt();
                let bearing_pred = dy.atan2(dx) - pose.theta;

                if range_pred < 1e-6 {
                    continue; // Avoid singularity
                }

                // Residual (observation - prediction)
                let dr = range_obs - range_pred;
                let db = normalize_angle(bearing_obs - normalize_angle(bearing_pred));

                _total_error += (dr * dr + db * db) as f64;

                // Jacobian of observation function h(pose) = [range, bearing] w.r.t. pose [x, y, theta]
                // range = sqrt((lm.x - x)^2 + (lm.y - y)^2)
                // bearing = atan2(lm.y - y, lm.x - x) - theta
                //
                // d(range)/dx = -(lm.x - x) / range = -dx / range
                // d(range)/dy = -(lm.y - y) / range = -dy / range
                // d(range)/dtheta = 0
                //
                // d(bearing)/dx = (lm.y - y) / range^2 = dy / range^2
                // d(bearing)/dy = -(lm.x - x) / range^2 = -dx / range^2
                // d(bearing)/dtheta = -1

                let r2 = range_pred * range_pred;
                let j = nalgebra::Matrix2x3::<f64>::new(
                    (-dx / range_pred) as f64,
                    (-dy / range_pred) as f64,
                    0.0,
                    (dy / r2) as f64,
                    (-dx / r2) as f64,
                    -1.0,
                );

                let residual = Vector2::<f64>::new(dr as f64, db as f64);

                // Accumulate normal equations: H += J^T * J, b += J^T * residual
                h_sum += j.transpose() * j;
                b_sum += j.transpose() * residual;
            }

            // Solve H * delta = b
            let h_inv = match h_sum.try_inverse() {
                Some(inv) => inv,
                None => return Some(pose), // Return current estimate if singular
            };

            let delta = h_inv * b_sum;

            // Update pose
            pose.x += delta[0] as f32;
            pose.y += delta[1] as f32;
            pose.theta = normalize_angle(pose.theta + delta[2] as f32);

            // Check convergence
            if delta.norm() < 1e-6 {
                break;
            }
        }

        Some(pose)
    }

    /// Attempt re-localization if conditions are met.
    /// Returns true if re-localization was performed.
    ///
    /// Re-localization is triggered when:
    /// 1. We have observations at the current pose
    /// 2. The observation residuals are extremely high (beyond threshold)
    /// 3. We have at least `min_landmarks` with high observation counts
    pub fn attempt_relocalization(&mut self) -> bool {
        if !self.config.enable_relocalization || self.poses.is_empty() {
            return false;
        }

        let current_pose_idx = self.poses.len() - 1;

        // Collect current observations (at the latest pose)
        let current_obs: Vec<_> = self
            .observation_constraints
            .iter()
            .filter(|c| c.pose_idx == current_pose_idx)
            .collect();

        if current_obs.is_empty() {
            return false;
        }

        // Check if observations have extreme residuals
        let mut high_residual_count = 0;
        let mut total_mahal = 0.0;
        for c in &current_obs {
            let mahal_sq = self.observation_mahalanobis_sq(c);
            total_mahal += mahal_sq;
            if mahal_sq > self.config.relocalization_mahal_threshold {
                high_residual_count += 1;
            }
        }

        let avg_mahal = total_mahal / current_obs.len() as f64;

        // Only proceed if majority of observations have extreme residuals
        if high_residual_count < current_obs.len() / 2 {
            return false;
        }

        if avg_mahal < self.config.relocalization_mahal_threshold {
            return false;
        }

        // Find confident landmarks (high observation count)
        let confident_obs: Vec<_> = current_obs
            .iter()
            .filter(|c| {
                c.landmark_idx < self.landmark_obs_count.len()
                    && self.landmark_obs_count[c.landmark_idx]
                        >= self.config.relocalization_min_obs_count
            })
            .map(|c| (c.landmark_idx, c.measurement[0], c.measurement[1]))
            .collect();

        if confident_obs.len() < self.config.relocalization_min_landmarks {
            return false;
        }

        // Compute pose from confident landmark observations
        let current_pose = self.poses[current_pose_idx];
        let computed_pose = match self.compute_pose_from_observations(
            &confident_obs,
            Some(current_pose), // Use current as initial guess
            20,
        ) {
            Some(p) => p,
            None => return false,
        };

        // Verify the computed pose actually improves residuals
        // by checking if observations make more sense from computed pose
        let mut old_error = 0.0;
        let mut new_error = 0.0;

        for &(lm_idx, range_obs, bearing_obs) in &confident_obs {
            let lm = &self.landmarks[lm_idx];

            // Error from current pose
            let dx_old = lm.x - current_pose.x;
            let dy_old = lm.y - current_pose.y;
            let range_old = (dx_old * dx_old + dy_old * dy_old).sqrt();
            let bearing_old = normalize_angle(dy_old.atan2(dx_old) - current_pose.theta);
            old_error += (range_obs - range_old).powi(2) + normalize_angle(bearing_obs - bearing_old).powi(2);

            // Error from computed pose
            let dx_new = lm.x - computed_pose.x;
            let dy_new = lm.y - computed_pose.y;
            let range_new = (dx_new * dx_new + dy_new * dy_new).sqrt();
            let bearing_new = normalize_angle(dy_new.atan2(dx_new) - computed_pose.theta);
            new_error += (range_obs - range_new).powi(2) + normalize_angle(bearing_obs - bearing_new).powi(2);
        }

        // Only apply if the computed pose significantly reduces error
        if new_error >= old_error * 0.5 {
            return false; // Not enough improvement
        }

        // Apply the pose correction
        // Compute delta from current to computed pose
        let delta_x = computed_pose.x - current_pose.x;
        let delta_y = computed_pose.y - current_pose.y;
        let delta_theta = normalize_angle(computed_pose.theta - current_pose.theta);

        // Apply correction to all poses in the sliding window
        // This shifts the entire trajectory to match the new pose
        for pose in &mut self.poses {
            pose.x += delta_x;
            pose.y += delta_y;
            pose.theta = normalize_angle(pose.theta + delta_theta);
        }

        // Log the re-localization event
        if self.enable_diagnostics {
            eprintln!(
                "RELOCALIZATION: Applied correction dx={:.2}, dy={:.2}, dtheta={:.1}° (old_err={:.2}, new_err={:.2})",
                delta_x, delta_y, delta_theta.to_degrees(), old_error, new_error
            );
        }

        true
    }

    /// Update diagnostics data
    pub fn update_diagnostics(&mut self) {
        if !self.enable_diagnostics {
            return;
        }

        self.diagnostics.current_pose_idx = self.poses.len().saturating_sub(1);
        self.diagnostics.num_poses = self.poses.len();
        self.diagnostics.num_landmarks = self.landmarks.len();
        self.diagnostics.num_odom_constraints = self.odometry_constraints.len();
        self.diagnostics.num_obs_constraints = self.observation_constraints.len();
        self.diagnostics.poses_since_observation = self.poses_since_observation;
        self.diagnostics.outlier_counts = self.outlier_count();

        // Landmark info
        self.diagnostics.landmark_info.clear();
        for (idx, (last_obs, count)) in self.landmark_last_obs_pose.iter()
            .zip(self.landmark_obs_count.iter())
            .enumerate()
        {
            self.diagnostics.landmark_info.push((idx, *last_obs, *count));
        }

        // Observation residuals for current pose
        self.diagnostics.observation_residuals.clear();
        let current_pose_idx = self.poses.len().saturating_sub(1);
        for c in &self.observation_constraints {
            if c.pose_idx == current_pose_idx {
                let e = self.observation_error(c);
                let mahal_sq = self.observation_mahalanobis_sq(c);
                self.diagnostics.observation_residuals.push((
                    c.landmark_idx,
                    e[0],
                    e[1],
                    mahal_sq,
                ));
            }
        }
    }

    /// Write diagnostics to file
    pub fn write_diagnostics_to_file(&self, file: &mut std::fs::File) -> std::io::Result<()> {
        let d = &self.diagnostics;
        writeln!(file, "=== SLAM DIAGNOSTICS ===")?;
        writeln!(file, "Pose: {} / {} poses", d.current_pose_idx, d.num_poses)?;
        writeln!(file, "Landmarks: {}", d.num_landmarks)?;
        writeln!(file, "Constraints: {} odom, {} obs", d.num_odom_constraints, d.num_obs_constraints)?;
        writeln!(file, "Blind duration: {} poses", d.poses_since_observation)?;
        writeln!(file, "Outliers: {} odom, {} obs", d.outlier_counts.0, d.outlier_counts.1)?;
        writeln!(file, "Loop closures: {}", d.loop_closures_detected)?;
        writeln!(file, "Optimization: {} iters, error {:.4} -> {:.4}",
            d.last_opt_iterations, d.last_opt_initial_error, d.last_opt_final_error)?;

        writeln!(file, "\n--- Landmark Info ---")?;
        for (idx, last_obs, count) in &d.landmark_info {
            let obs_str = last_obs.map(|p| format!("pose {}", p)).unwrap_or("None".to_string());
            writeln!(file, "  LM {}: last_obs={}, total_obs={}", idx, obs_str, count)?;
        }

        writeln!(file, "\n--- Current Observation Residuals ---")?;
        for (lm_idx, range_res, bearing_res, mahal_sq) in &d.observation_residuals {
            writeln!(file, "  LM {}: range_res={:.4}, bearing_res={:.4}, mahal_sq={:.4}",
                lm_idx, range_res, bearing_res, mahal_sq)?;
        }

        if !d.landmark_errors.is_empty() {
            writeln!(file, "\n--- Landmark Position Errors ---")?;
            for (idx, err) in &d.landmark_errors {
                writeln!(file, "  LM {}: error={:.4}m", idx, err)?;
            }
        }

        writeln!(file, "\n--- Pose Positions ---")?;
        for (i, pose) in self.poses.iter().enumerate() {
            writeln!(file, "  Pose {}: ({:.2}, {:.2}, {:.1}°)",
                i, pose.x, pose.y, pose.theta.to_degrees())?;
        }

        writeln!(file, "\n--- Landmark Positions ---")?;
        for (i, lm) in self.landmarks.iter().enumerate() {
            writeln!(file, "  LM {}: ({:.2}, {:.2})", i, lm.x, lm.y)?;
        }

        writeln!(file, "========================\n")?;
        Ok(())
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

            let error = self.odometry_error(c);

            let sqrt_info = c.information.cholesky()
                .map(|ch| ch.l())
                .unwrap_or(Matrix3::identity());

            let sqrt_robust = (c.robust_weight as f32).sqrt();
            let weighted_error = sqrt_info * error * sqrt_robust;
            for i in 0..3 {
                residuals[res_idx + i] = weighted_error[i] as f64;
            }

            if let Some(idx1) = self.pose_state_idx(c.from_idx) {
                let j1 = Matrix3::new(
                    -cos_t, -sin_t, -sin_t * dx + cos_t * dy,
                    sin_t, -cos_t, -cos_t * dx - sin_t * dy,
                    0.0, 0.0, -1.0,
                );
                let wj1 = sqrt_info * j1 * sqrt_robust;
                for i in 0..3 {
                    for j in 0..3 {
                        jacobian[(res_idx + i, idx1 + j)] = -wj1[(i, j)] as f64;
                    }
                }
            }

            if let Some(idx2) = self.pose_state_idx(c.to_idx) {
                let j2 = Matrix3::new(
                    cos_t, sin_t, 0.0,
                    -sin_t, cos_t, 0.0,
                    0.0, 0.0, 1.0,
                );
                let wj2 = sqrt_info * j2 * sqrt_robust;
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

            let sqrt_robust = (c.robust_weight as f32).sqrt();
            let weighted_error = sqrt_info * error * sqrt_robust;
            residuals[res_idx] = weighted_error[0] as f64;
            residuals[res_idx + 1] = weighted_error[1] as f64;

            if let Some(pose_idx) = self.pose_state_idx(c.pose_idx) {
                let jp = nalgebra::Matrix2x3::new(
                    -dx / sqrt_q, -dy / sqrt_q, 0.0,
                    dy / q, -dx / q, -1.0,
                );
                let wjp = sqrt_info * jp * sqrt_robust;
                for i in 0..2 {
                    for j in 0..3 {
                        jacobian[(res_idx + i, pose_idx + j)] = -wjp[(i, j)] as f64;
                    }
                }
            }

            let lm_idx = self.landmark_state_idx(c.landmark_idx);
            let jl = Matrix2::new(
                dx / sqrt_q, dy / sqrt_q,
                -dy / q, dx / q,
            );
            let wjl = sqrt_info * jl * sqrt_robust;
            for i in 0..2 {
                for j in 0..2 {
                    jacobian[(res_idx + i, lm_idx + j)] = -wjl[(i, j)] as f64;
                }
            }

            res_idx += 2;
        }

        (jacobian, residuals)
    }

    /// Optimize the graph using Levenberg-Marquardt with IRLS
    pub fn optimize(&mut self) -> OptimizationResult {
        if self.num_variables() == 0 {
            return OptimizationResult {
                iterations: 0,
                initial_error: 0.0,
                final_error: 0.0,
                converged: true,
                outliers_detected: (0, 0),
            };
        }

        // Attempt re-localization before optimization if conditions are met
        // This corrects severe pose drift when returning from blind navigation
        self.attempt_relocalization();

        let result = if self.config.use_sparse_solver {
            self.optimize_sparse()
        } else {
            self.optimize_dense()
        };

        // Update diagnostics
        self.diagnostics.last_opt_iterations = result.iterations;
        self.diagnostics.last_opt_initial_error = result.initial_error;
        self.diagnostics.last_opt_final_error = result.final_error;
        self.update_diagnostics();

        result
    }

    /// Dense optimization
    fn optimize_dense(&mut self) -> OptimizationResult {
        let initial_error = self.total_error();
        let mut current_error = initial_error;
        let mut lambda = self.config.initial_lambda;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            self.update_robust_weights();

            let (j, r) = self.build_linear_system();
            let jt = j.transpose();
            let jtj = &jt * &j;
            let jtr = &jt * &r;

            let n = jtj.nrows();
            let mut h = jtj.clone();
            for i in 0..n {
                h[(i, i)] += lambda * (1.0 + h[(i, i)]);
            }

            let neg_jtr = -&jtr;
            let dx = match h.clone().lu().solve(&neg_jtr) {
                Some(sol) => sol,
                None => break,
            };

            let old_state = self.pack_state();
            let new_state = &old_state + &dx;
            self.unpack_state(&new_state);

            let new_error = self.total_error();

            if new_error < current_error {
                current_error = new_error;
                lambda *= self.config.lambda_down;

                let improvement = (current_error - new_error).abs() / (current_error + 1e-10);
                if improvement < self.config.convergence_threshold {
                    break;
                }
            } else {
                self.unpack_state(&old_state);
                lambda *= self.config.lambda_up;
            }
        }

        OptimizationResult {
            iterations,
            initial_error,
            final_error: current_error,
            converged: current_error < initial_error || initial_error < 1e-6,
            outliers_detected: self.outlier_count(),
        }
    }

    /// Sparse optimization
    fn optimize_sparse(&mut self) -> OptimizationResult {
        let initial_error = self.total_error();
        let mut current_error = initial_error;
        let mut lambda = self.config.initial_lambda;
        let mut iterations = 0;

        let sparse_solver = SparseSlamSolver::new();

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            self.update_robust_weights();

            let (j_sparse, r) = sparse_solver.build_sparse_system(
                &self.poses,
                &self.landmarks,
                &self.odometry_constraints,
                &self.observation_constraints,
                self.fix_first_pose,
            );

            let dx = match sparse_solver.solve(&j_sparse, &r, lambda) {
                Some(sol) => sol,
                None => break,
            };

            let old_state = self.pack_state();
            let new_state = &old_state + &dx;
            self.unpack_state(&new_state);

            let new_error = self.total_error();

            if new_error < current_error {
                current_error = new_error;
                lambda *= self.config.lambda_down;

                let improvement = (current_error - new_error).abs() / (current_error + 1e-10);
                if improvement < self.config.convergence_threshold {
                    break;
                }
            } else {
                self.unpack_state(&old_state);
                lambda *= self.config.lambda_up;
            }
        }

        OptimizationResult {
            iterations,
            initial_error,
            final_error: current_error,
            converged: current_error < initial_error || initial_error < 1e-6,
            outliers_detected: self.outlier_count(),
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
    pub outliers_detected: (usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;
        graph.config.use_sparse_solver = false;

        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(1.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(2.0, 0.0, 0.0));

        let cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(0, 1, Vector3::new(1.0, 0.0, 0.0), &cov);
        graph.add_odometry(1, 2, Vector3::new(1.0, 0.0, 0.0), &cov);

        let result = graph.optimize();
        assert!(result.converged);
    }

    #[test]
    fn test_loop_closure() {
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;
        graph.config.use_sparse_solver = false;

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

        let result = graph.optimize();

        let p4 = &graph.poses[4];
        let dist = (p4.x * p4.x + p4.y * p4.y).sqrt();
        assert!(dist < 2.0, "Loop closure should bring pose 4 near origin");
        assert!(result.final_error < result.initial_error);
    }
}
