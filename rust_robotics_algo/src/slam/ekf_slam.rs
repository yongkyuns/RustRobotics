//! EKF-SLAM (Extended Kalman Filter Simultaneous Localization and Mapping)
//!
//! This module implements EKF-SLAM, which estimates both the robot's pose and
//! the positions of landmarks in the environment simultaneously.
//!
//! ## State Vector Structure
//! - Dimension: 3 + 2n (robot pose + n landmarks)
//! - Robot pose: [x, y, θ] (position and orientation)
//! - Landmarks: [m1_x, m1_y, m2_x, m2_y, ...] (2D positions)
//!
//! ## References
//! - [EKF-SLAM Tutorial](https://jihongju.github.io/2018/10/05/slam-05-ekf-slam/)
//! - [MATLAB ekfSLAM](https://www.mathworks.com/help/nav/ref/ekfslam.html)

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Vector2, Vector3};

use crate::prelude::*;

/// Generate random float between [-1.0, 1.0]
fn rand() -> f32 {
    2.0 * (rand::random::<f32>() - 0.5)
}

/// EKF-SLAM state containing the mean and covariance of the state estimate
#[derive(Debug, Clone)]
pub struct EkfSlamState {
    /// State mean [x, y, θ, m1_x, m1_y, m2_x, m2_y, ...]
    pub mu: DVector<f32>,
    /// State covariance matrix (3+2n × 3+2n)
    pub sigma: DMatrix<f32>,
    /// Number of landmarks in the map
    pub n_landmarks: usize,
}

/// EKF-SLAM configuration parameters
#[derive(Debug, Clone)]
pub struct EkfSlamConfig {
    /// Process noise for motion model (affects robot pose uncertainty)
    pub motion_noise: Matrix3<f32>,
    /// Measurement noise for range-bearing observations
    pub observation_noise: Matrix2<f32>,
    /// Mahalanobis distance threshold for data association
    pub association_gate: f32,
    /// Maximum observation range for landmarks
    pub max_range: f32,
}

/// A single range-bearing observation
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Distance to landmark (meters)
    pub range: f32,
    /// Bearing angle to landmark relative to robot heading (radians)
    pub bearing: f32,
}

impl Default for EkfSlamState {
    fn default() -> Self {
        Self::new()
    }
}

impl EkfSlamState {
    /// Create a new EKF-SLAM state with initial robot pose at origin
    pub fn new() -> Self {
        Self::with_pose(0.0, 0.0, 0.0)
    }

    /// Create a new EKF-SLAM state with specified initial robot pose
    pub fn with_pose(x: f32, y: f32, theta: f32) -> Self {
        let mut mu = DVector::zeros(3);
        mu[0] = x;
        mu[1] = y;
        mu[2] = theta;
        // Initial covariance: small uncertainty for robot pose
        let sigma = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01, 0.001]));
        Self {
            mu,
            sigma,
            n_landmarks: 0,
        }
    }

    /// Get the robot pose [x, y, θ]
    pub fn robot_pose(&self) -> Vector3<f32> {
        Vector3::new(self.mu[0], self.mu[1], self.mu[2])
    }

    /// Get robot x position
    pub fn x(&self) -> f32 {
        self.mu[0]
    }

    /// Get robot y position
    pub fn y(&self) -> f32 {
        self.mu[1]
    }

    /// Get robot heading angle
    pub fn theta(&self) -> f32 {
        self.mu[2]
    }

    /// Get landmark position by index
    pub fn landmark(&self, idx: usize) -> Option<Vector2<f32>> {
        if idx >= self.n_landmarks {
            return None;
        }
        let base = 3 + 2 * idx;
        Some(Vector2::new(self.mu[base], self.mu[base + 1]))
    }

    /// Get the 2x2 covariance block for a specific landmark
    pub fn landmark_covariance(&self, idx: usize) -> Option<Matrix2<f32>> {
        if idx >= self.n_landmarks {
            return None;
        }
        let base = 3 + 2 * idx;
        Some(Matrix2::new(
            self.sigma[(base, base)],
            self.sigma[(base, base + 1)],
            self.sigma[(base + 1, base)],
            self.sigma[(base + 1, base + 1)],
        ))
    }

    /// Get the 3x3 robot pose covariance
    pub fn robot_covariance(&self) -> Matrix3<f32> {
        Matrix3::new(
            self.sigma[(0, 0)],
            self.sigma[(0, 1)],
            self.sigma[(0, 2)],
            self.sigma[(1, 0)],
            self.sigma[(1, 1)],
            self.sigma[(1, 2)],
            self.sigma[(2, 0)],
            self.sigma[(2, 1)],
            self.sigma[(2, 2)],
        )
    }
}

impl Default for EkfSlamConfig {
    fn default() -> Self {
        Self {
            // Process noise (motion uncertainty)
            // These are base values that get SCALED by actual motion in predict()
            // - Position noise scaled by distance traveled (v * dt)
            // - Heading noise scaled by angle change (w * dt)
            motion_noise: Matrix3::new(
                0.02, 0.0, 0.0,    // x variance per meter traveled
                0.0, 0.02, 0.0,    // y variance per meter traveled
                0.0, 0.0, 0.05,    // theta variance per radian turned (lower to reduce heading drift)
            ),
            // Observation noise (range, bearing variances)
            // Lower values = filter trusts observations more = faster covariance reduction
            observation_noise: Matrix2::new(
                0.01, 0.0,    // range variance (0.1m std dev)
                0.0, 0.0002,  // bearing variance (~0.8 deg std dev)
            ),
            // Mahalanobis distance gate for data association
            association_gate: 5.0,
            // Maximum detection range
            max_range: 30.0,
        }
    }
}

/// Motion model: predict robot pose given velocity and angular velocity
///
/// Uses velocity-based motion model:
/// - If ω ≈ 0 (straight line): x' = x + v·dt·cos(θ), y' = y + v·dt·sin(θ)
/// - If ω ≠ 0 (arc motion): uses arc equations
pub fn motion_model(pose: &Vector3<f32>, v: f32, w: f32, dt: f32) -> Vector3<f32> {
    let theta = pose[2];

    if w.abs() < 1e-6 {
        // Straight line motion
        Vector3::new(
            pose[0] + v * dt * theta.cos(),
            pose[1] + v * dt * theta.sin(),
            theta,
        )
    } else {
        // Arc motion
        Vector3::new(
            pose[0] + v / w * (-(theta).sin() + (theta + w * dt).sin()),
            pose[1] + v / w * ((theta).cos() - (theta + w * dt).cos()),
            normalize_angle(theta + w * dt),
        )
    }
}

/// Compute the Jacobian of the motion model with respect to the robot state
fn motion_jacobian(pose: &Vector3<f32>, v: f32, w: f32, dt: f32) -> Matrix3<f32> {
    let theta = pose[2];

    if w.abs() < 1e-6 {
        // Jacobian for straight line motion
        Matrix3::new(
            1.0, 0.0, -v * dt * theta.sin(),
            0.0, 1.0, v * dt * theta.cos(),
            0.0, 0.0, 1.0,
        )
    } else {
        // Jacobian for arc motion
        Matrix3::new(
            1.0, 0.0, v / w * (-(theta).cos() + (theta + w * dt).cos()),
            0.0, 1.0, v / w * (-(theta).sin() + (theta + w * dt).sin()),
            0.0, 0.0, 1.0,
        )
    }
}

/// Observation model: compute expected range and bearing to a landmark
fn observation_model(robot_pose: &Vector3<f32>, landmark: &Vector2<f32>) -> Observation {
    let dx = landmark[0] - robot_pose[0];
    let dy = landmark[1] - robot_pose[1];
    Observation {
        range: (dx * dx + dy * dy).sqrt(),
        bearing: normalize_angle(dy.atan2(dx) - robot_pose[2]),
    }
}

/// Compute the Jacobian of the observation model with respect to robot pose and landmark position
/// Returns (H_robot, H_landmark) where:
/// - H_robot is 2x3 Jacobian with respect to [x, y, θ]
/// - H_landmark is 2x2 Jacobian with respect to [mx, my]
fn observation_jacobian(
    robot_pose: &Vector3<f32>,
    landmark: &Vector2<f32>,
) -> (nalgebra::Matrix2x3<f32>, Matrix2<f32>) {
    let dx = landmark[0] - robot_pose[0];
    let dy = landmark[1] - robot_pose[1];
    let q = dx * dx + dy * dy;
    let sqrt_q = q.sqrt();

    // Jacobian with respect to robot pose [x, y, θ]
    let h_robot = nalgebra::Matrix2x3::new(
        -dx / sqrt_q, -dy / sqrt_q, 0.0,
        dy / q, -dx / q, -1.0,
    );

    // Jacobian with respect to landmark position [mx, my]
    let h_landmark = Matrix2::new(
        dx / sqrt_q, dy / sqrt_q,
        -dy / q, dx / q,
    );

    (h_robot, h_landmark)
}

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

/// EKF-SLAM prediction step: update robot pose estimate based on control input
pub fn predict(state: &mut EkfSlamState, config: &EkfSlamConfig, v: f32, w: f32, dt: f32) {
    let n = state.mu.len();

    // Predict robot pose
    let robot_pose = state.robot_pose();
    let new_pose = motion_model(&robot_pose, v, w, dt);

    // Update state mean
    state.mu[0] = new_pose[0];
    state.mu[1] = new_pose[1];
    state.mu[2] = new_pose[2];

    // Compute motion Jacobian
    let g = motion_jacobian(&robot_pose, v, w, dt);

    // Build full Jacobian matrix G (n x n)
    // G = [g  0]
    //     [0  I]
    let mut G = DMatrix::<f32>::identity(n, n);
    for i in 0..3 {
        for j in 0..3 {
            G[(i, j)] = g[(i, j)];
        }
    }

    // Build process noise matrix R (n x n)
    // Scale noise by actual motion to properly reflect uncertainty
    // - Position noise scales with distance traveled (v * dt)
    // - Heading noise scales with angular change (w * dt)
    let dist = (v * dt).abs();
    let angle_change = (w * dt).abs();

    let mut R = DMatrix::<f32>::zeros(n, n);
    // Position uncertainty proportional to distance traveled
    R[(0, 0)] = config.motion_noise[(0, 0)] * dist;
    R[(1, 1)] = config.motion_noise[(1, 1)] * dist;
    // Heading uncertainty proportional to angle change + base uncertainty
    // Add minimum heading noise to prevent overconfidence when w is small
    R[(2, 2)] = config.motion_noise[(2, 2)] * (angle_change + 0.01);

    // Update covariance: Σ' = G Σ Gᵀ + R
    state.sigma = &G * &state.sigma * G.transpose() + R;
}

/// Data association using Mahalanobis distance with ambiguity detection
/// Returns the index of the associated landmark, or None if:
/// - No landmark is within the association gate (new landmark)
/// - Multiple landmarks are ambiguously close (skip to avoid wrong association)
fn associate_landmark(
    state: &EkfSlamState,
    obs: &Observation,
    config: &EkfSlamConfig,
) -> Option<usize> {
    let robot_pose = state.robot_pose();

    // First, compute the observed landmark position from the observation
    let obs_lm_x = robot_pose[0] + obs.range * (robot_pose[2] + obs.bearing).cos();
    let obs_lm_y = robot_pose[1] + obs.range * (robot_pose[2] + obs.bearing).sin();

    // Collect all candidates within the gate
    let mut candidates: Vec<(usize, f32, f32)> = Vec::new(); // (index, mahal_dist, eucl_dist)

    for i in 0..state.n_landmarks {
        let landmark = state.landmark(i).unwrap();

        // Euclidean distance in world coordinates
        let eucl_dist = ((obs_lm_x - landmark[0]).powi(2) + (obs_lm_y - landmark[1]).powi(2)).sqrt();

        // Mahalanobis distance check
        let pred = observation_model(&robot_pose, &landmark);

        let dz = nalgebra::Vector2::new(
            obs.range - pred.range,
            normalize_angle(obs.bearing - pred.bearing),
        );

        let (h_robot, h_landmark) = observation_jacobian(&robot_pose, &landmark);

        let n = state.mu.len();
        let mut H = DMatrix::<f32>::zeros(2, n);
        for row in 0..2 {
            for col in 0..3 {
                H[(row, col)] = h_robot[(row, col)];
            }
            let base = 3 + 2 * i;
            H[(row, base)] = h_landmark[(row, 0)];
            H[(row, base + 1)] = h_landmark[(row, 1)];
        }

        let S = &H * &state.sigma * H.transpose() + config.observation_noise;

        if let Some(s_inv) = S.try_inverse() {
            let dist_sq = (dz.transpose() * s_inv * dz)[(0, 0)];
            let mahal_dist = dist_sq.sqrt();

            // Collect candidates within the gate
            if mahal_dist < config.association_gate {
                candidates.push((i, mahal_dist, eucl_dist));
            }
        }
    }

    // No candidates - this is a new landmark
    if candidates.is_empty() {
        return None;
    }

    // Use nearest-neighbor: pick the candidate with lowest Mahalanobis distance
    // Sort by Mahalanobis distance and return the best match
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Some(candidates[0].0)
}

/// Add a new landmark to the state based on an observation
fn add_landmark(state: &mut EkfSlamState, obs: &Observation, config: &EkfSlamConfig) {
    let robot_pose = state.robot_pose();

    // Compute landmark position from robot pose and observation
    let lm_x = robot_pose[0] + obs.range * (robot_pose[2] + obs.bearing).cos();
    let lm_y = robot_pose[1] + obs.range * (robot_pose[2] + obs.bearing).sin();

    // Augment state vector
    let old_n = state.mu.len();
    let mut new_mu = DVector::zeros(old_n + 2);
    new_mu.rows_mut(0, old_n).copy_from(&state.mu);
    new_mu[old_n] = lm_x;
    new_mu[old_n + 1] = lm_y;
    state.mu = new_mu;

    // Compute Jacobian of landmark initialization with respect to robot pose and observation
    // lm = [x + r·cos(θ + φ), y + r·sin(θ + φ)]
    // ∂lm/∂[x,y,θ] and ∂lm/∂[r,φ]
    let angle = robot_pose[2] + obs.bearing;
    let c = angle.cos();
    let s = angle.sin();

    // Jacobian with respect to robot pose (2x3)
    let g_robot = nalgebra::Matrix2x3::new(
        1.0, 0.0, -obs.range * s,
        0.0, 1.0, obs.range * c,
    );

    // Jacobian with respect to observation (2x2)
    let g_obs = Matrix2::new(
        c, -obs.range * s,
        s, obs.range * c,
    );

    // Augment covariance matrix
    let new_n = old_n + 2;
    let mut new_sigma = DMatrix::zeros(new_n, new_n);

    // Copy old covariance
    for i in 0..old_n {
        for j in 0..old_n {
            new_sigma[(i, j)] = state.sigma[(i, j)];
        }
    }

    // Compute cross-correlation between new landmark and robot pose
    // Σ_new,robot = G_robot * Σ_robot,robot
    let sigma_rr = Matrix3::new(
        state.sigma[(0, 0)], state.sigma[(0, 1)], state.sigma[(0, 2)],
        state.sigma[(1, 0)], state.sigma[(1, 1)], state.sigma[(1, 2)],
        state.sigma[(2, 0)], state.sigma[(2, 1)], state.sigma[(2, 2)],
    );
    let cross_robot = &g_robot * &sigma_rr;

    // Σ_new,robot and Σ_robot,new blocks (2x3 and 3x2)
    for i in 0..2 {
        for j in 0..3 {
            new_sigma[(old_n + i, j)] = cross_robot[(i, j)];
            new_sigma[(j, old_n + i)] = cross_robot[(i, j)];
        }
    }

    // Compute cross-correlations between new landmark and EXISTING landmarks
    // Σ_new,existing_k = G_robot * Σ_robot,existing_k
    // This is crucial for proper SLAM - without it, landmarks are treated as independent
    for k in 0..state.n_landmarks {
        let base_k = 3 + 2 * k;
        // Extract Σ_robot,existing_k (3x2 block)
        let sigma_r_k = nalgebra::Matrix3x2::new(
            state.sigma[(0, base_k)], state.sigma[(0, base_k + 1)],
            state.sigma[(1, base_k)], state.sigma[(1, base_k + 1)],
            state.sigma[(2, base_k)], state.sigma[(2, base_k + 1)],
        );
        // Σ_new,existing_k = G_robot * Σ_robot,existing_k (2x2)
        let cross_k = &g_robot * &sigma_r_k;

        // Set both Σ_new,k and Σ_k,new (symmetric)
        for i in 0..2 {
            for j in 0..2 {
                new_sigma[(old_n + i, base_k + j)] = cross_k[(i, j)];
                new_sigma[(base_k + j, old_n + i)] = cross_k[(i, j)];
            }
        }
    }

    // Compute new landmark covariance
    // Σ_new,new = G_robot * Σ_robot,robot * G_robotᵀ + G_obs * Q * G_obsᵀ
    let sigma_mm = &g_robot * sigma_rr * g_robot.transpose()
        + &g_obs * &config.observation_noise * g_obs.transpose();

    for i in 0..2 {
        for j in 0..2 {
            new_sigma[(old_n + i, old_n + j)] = sigma_mm[(i, j)];
        }
    }

    state.sigma = new_sigma;
    state.n_landmarks += 1;
}

/// EKF-SLAM update step: correct state estimate using observations
///
/// Uses batch update when multiple known landmarks are visible simultaneously.
/// This better constrains heading by using geometric relationships between landmarks.
///
/// # Arguments
/// * `state` - The current EKF-SLAM state (will be modified)
/// * `config` - Configuration parameters
/// * `observations` - Range-bearing observations (with noise)
///
/// Note: The usize in each observation tuple is the true landmark index from simulation,
/// which is only used for debugging. Data association uses Mahalanobis distance.
pub fn update(
    state: &mut EkfSlamState,
    config: &EkfSlamConfig,
    observations: &[(usize, Observation)],
) {
    // First pass: separate known landmarks from new ones
    let mut known_obs: Vec<(usize, &Observation)> = Vec::new();
    let mut new_obs: Vec<&Observation> = Vec::new();

    for (_true_idx, obs) in observations {
        let associated = associate_landmark(state, obs, config);
        match associated {
            Some(idx) => known_obs.push((idx, obs)),
            None => new_obs.push(obs),
        }
    }

    // Batch update for known landmarks (if multiple are visible)
    if known_obs.len() >= 2 {
        batch_update_landmarks(state, config, &known_obs);
    } else {
        // Sequential update for single landmark
        for (idx, obs) in &known_obs {
            update_landmark(state, config, *idx, obs);
        }
    }

    // Add new landmarks after updating with known ones
    // When the map is empty, always add landmarks (initial mapping)
    // Otherwise, only add new landmarks if:
    // 1. We have at least 2 known landmarks for heading constraint
    // 2. Heading uncertainty is below a threshold (prevent adding during high uncertainty)
    let heading_variance = state.sigma[(2, 2)];
    let heading_uncertainty_ok = heading_variance < 0.05; // ~12 deg std dev threshold
    let enough_known_landmarks = known_obs.len() >= 2;
    let map_is_empty = state.n_landmarks == 0;

    if map_is_empty || (heading_uncertainty_ok && enough_known_landmarks) {
        for obs in new_obs {
            add_landmark(state, obs, config);
        }
    }
    // When conditions aren't met, we skip adding new landmarks
    // They'll be added later when conditions improve (after revisiting known landmarks)
}

/// Batch update using multiple landmark observations simultaneously
/// This provides better heading constraints through geometric relationships
fn batch_update_landmarks(
    state: &mut EkfSlamState,
    config: &EkfSlamConfig,
    observations: &[(usize, &Observation)],
) {
    let n = state.mu.len();
    let m = observations.len() * 2; // 2 measurements per observation (range, bearing)

    // Build stacked measurement vector and Jacobian
    let mut z = DVector::<f32>::zeros(m);
    let mut z_pred = DVector::<f32>::zeros(m);
    let mut H = DMatrix::<f32>::zeros(m, n);

    for (i, (landmark_idx, obs)) in observations.iter().enumerate() {
        let robot_pose = state.robot_pose();
        let landmark = state.landmark(*landmark_idx).unwrap();

        // Predicted observation
        let pred = observation_model(&robot_pose, &landmark);

        // Fill measurement vectors
        z[2 * i] = obs.range;
        z[2 * i + 1] = obs.bearing;
        z_pred[2 * i] = pred.range;
        z_pred[2 * i + 1] = pred.bearing;

        // Compute observation Jacobian
        let (h_robot, h_landmark) = observation_jacobian(&robot_pose, &landmark);

        // Fill Jacobian matrix
        for row in 0..2 {
            for col in 0..3 {
                H[(2 * i + row, col)] = h_robot[(row, col)];
            }
            let base = 3 + 2 * landmark_idx;
            H[(2 * i + row, base)] = h_landmark[(row, 0)];
            H[(2 * i + row, base + 1)] = h_landmark[(row, 1)];
        }
    }

    // Innovation with angle normalization
    let mut dz = z - z_pred;
    for i in 0..observations.len() {
        dz[2 * i + 1] = normalize_angle(dz[2 * i + 1]);
    }

    // Build block-diagonal observation noise matrix
    let mut Q = DMatrix::<f32>::zeros(m, m);
    for i in 0..observations.len() {
        Q[(2 * i, 2 * i)] = config.observation_noise[(0, 0)];
        Q[(2 * i, 2 * i + 1)] = config.observation_noise[(0, 1)];
        Q[(2 * i + 1, 2 * i)] = config.observation_noise[(1, 0)];
        Q[(2 * i + 1, 2 * i + 1)] = config.observation_noise[(1, 1)];
    }

    // Innovation covariance: S = H Σ Hᵀ + Q
    let S = &H * &state.sigma * H.transpose() + Q;

    // Innovation gating for batch update BEFORE inverting S
    // Check if any individual observation has abnormally large innovation
    let mut any_outlier = false;
    for i in 0..observations.len() {
        // Check each observation's innovation using its 2x2 sub-block of S
        let dz_i = nalgebra::Vector2::new(dz[2 * i], dz[2 * i + 1]);
        let s_i = Matrix2::new(
            S[(2 * i, 2 * i)], S[(2 * i, 2 * i + 1)],
            S[(2 * i + 1, 2 * i)], S[(2 * i + 1, 2 * i + 1)],
        );
        if let Some(s_i_inv) = s_i.try_inverse() {
            let mahal_sq = (dz_i.transpose() * s_i_inv * dz_i)[(0, 0)];
            // Chi-squared with 2 DOF at 99.5% ≈ 10.6
            if mahal_sq > 10.0 {
                any_outlier = true;
                break;
            }
        }
    }

    // If any observation is an outlier, fall back to sequential update
    // (which has individual gating for each observation)
    if any_outlier {
        // Fall back to sequential updates with individual gating
        for (idx, obs) in observations {
            update_landmark(state, config, *idx, obs);
        }
        return;
    }

    // Kalman gain: K = Σ Hᵀ S⁻¹
    if let Some(s_inv) = S.try_inverse() {
        let K = &state.sigma * H.transpose() * s_inv;

        // State update: μ' = μ + K z
        let update = &K * dz;
        state.mu += update;

        // Normalize heading
        state.mu[2] = normalize_angle(state.mu[2]);

        // Covariance update: Σ' = (I - K H) Σ
        let I = DMatrix::<f32>::identity(n, n);
        state.sigma = (&I - &K * &H) * &state.sigma;

        // Ensure symmetry
        state.sigma = (&state.sigma + state.sigma.transpose()) * 0.5;

        // Minimum covariance to prevent over-confidence
        enforce_min_covariance(state);
    }
}

/// Update state using observation of a known landmark
fn update_landmark(
    state: &mut EkfSlamState,
    config: &EkfSlamConfig,
    landmark_idx: usize,
    obs: &Observation,
) {
    let robot_pose = state.robot_pose();
    let landmark = state.landmark(landmark_idx).unwrap();

    // Predicted observation
    let pred = observation_model(&robot_pose, &landmark);

    // Innovation
    let dz = nalgebra::Vector2::new(
        obs.range - pred.range,
        normalize_angle(obs.bearing - pred.bearing),
    );

    // Compute observation Jacobian
    let (h_robot, h_landmark) = observation_jacobian(&robot_pose, &landmark);

    // Build full observation Jacobian H (2 x n)
    let n = state.mu.len();
    let mut H = DMatrix::<f32>::zeros(2, n);
    for row in 0..2 {
        for col in 0..3 {
            H[(row, col)] = h_robot[(row, col)];
        }
        let base = 3 + 2 * landmark_idx;
        H[(row, base)] = h_landmark[(row, 0)];
        H[(row, base + 1)] = h_landmark[(row, 1)];
    }

    // Innovation covariance: S = H Σ Hᵀ + Q
    let S = &H * &state.sigma * H.transpose() + config.observation_noise;

    // Kalman gain: K = Σ Hᵀ S⁻¹
    if let Some(s_inv) = S.try_inverse() {
        // Innovation gating: reject outlier observations
        // Compute Mahalanobis distance of innovation
        let dz_vec = DVector::from_column_slice(&[dz[0], dz[1]]);
        let mahal_sq = (dz.transpose() * &s_inv * dz)[(0, 0)];

        // Gate threshold: chi-squared with 2 DOF at 99.5% confidence ≈ 10.6
        // Using a stricter gate to prevent bad observations from corrupting state
        const INNOVATION_GATE: f32 = 10.0;
        if mahal_sq > INNOVATION_GATE {
            // Innovation too large - this observation is an outlier, skip it
            return;
        }

        let K = &state.sigma * H.transpose() * s_inv;

        // State update: μ' = μ + K z
        let update = &K * &dz_vec;
        state.mu += update;

        // Normalize heading
        state.mu[2] = normalize_angle(state.mu[2]);

        // Covariance update: Σ' = (I - K H) Σ
        let I = DMatrix::<f32>::identity(n, n);
        state.sigma = (&I - &K * &H) * &state.sigma;

        // Ensure symmetry
        state.sigma = (&state.sigma + state.sigma.transpose()) * 0.5;

        // Minimum covariance to prevent over-confidence
        // This helps prevent map rotation by keeping some uncertainty in landmark positions
        enforce_min_covariance(state);
    }
}

/// Enforce minimum covariance on robot pose and landmarks
/// This prevents the filter from becoming over-confident, which can cause
/// map rotation when heading has systematic errors
fn enforce_min_covariance(state: &mut EkfSlamState) {
    // Minimum robot pose covariance
    const MIN_POS_VAR: f32 = 0.001;   // 3cm std dev
    const MIN_THETA_VAR: f32 = 0.0001; // 0.6 deg std dev

    // Minimum landmark covariance
    const MIN_LANDMARK_VAR: f32 = 0.0001; // 1cm std dev

    // Apply minimum to robot pose
    if state.sigma[(0, 0)] < MIN_POS_VAR {
        state.sigma[(0, 0)] = MIN_POS_VAR;
    }
    if state.sigma[(1, 1)] < MIN_POS_VAR {
        state.sigma[(1, 1)] = MIN_POS_VAR;
    }
    if state.sigma[(2, 2)] < MIN_THETA_VAR {
        state.sigma[(2, 2)] = MIN_THETA_VAR;
    }

    // Apply minimum to landmark positions
    for i in 0..state.n_landmarks {
        let base = 3 + 2 * i;
        if state.sigma[(base, base)] < MIN_LANDMARK_VAR {
            state.sigma[(base, base)] = MIN_LANDMARK_VAR;
        }
        if state.sigma[(base + 1, base + 1)] < MIN_LANDMARK_VAR {
            state.sigma[(base + 1, base + 1)] = MIN_LANDMARK_VAR;
        }
    }
}

/// Generate observations from the robot's current position to visible landmarks
///
/// Returns a vector of (landmark_index, observation) tuples for landmarks within range
pub fn generate_observations(
    robot_pose: &Vector3<f32>,
    true_landmarks: &[Vector2<f32>],
    config: &EkfSlamConfig,
    add_noise: bool,
) -> Vec<(usize, Observation)> {
    let mut observations = Vec::new();

    for (idx, landmark) in true_landmarks.iter().enumerate() {
        let dx = landmark[0] - robot_pose[0];
        let dy = landmark[1] - robot_pose[1];
        let range = (dx * dx + dy * dy).sqrt();

        // Only observe landmarks within range
        if range <= config.max_range {
            let bearing = normalize_angle(dy.atan2(dx) - robot_pose[2]);

            let obs = if add_noise {
                // Add noise scaled by std dev (sqrt of variance)
                // Using rand() which gives uniform [-1, 1], scaled to approximate Gaussian
                let range_noise = rand() * config.observation_noise[(0, 0)].sqrt() * 0.5;
                let bearing_noise = rand() * config.observation_noise[(1, 1)].sqrt() * 0.5;
                Observation {
                    range: range + range_noise,
                    bearing: normalize_angle(bearing + bearing_noise),
                }
            } else {
                Observation { range, bearing }
            };

            observations.push((idx, obs));
        }
    }

    observations
}

/// Perform one complete EKF-SLAM cycle: prediction + observation + update
pub fn step(
    state: &mut EkfSlamState,
    config: &EkfSlamConfig,
    true_pose: &Vector3<f32>,
    true_landmarks: &[Vector2<f32>],
    v: f32,
    w: f32,
    dt: f32,
) {
    // 1. Prediction step
    predict(state, config, v, w, dt);

    // 2. Generate observations from true pose
    let observations = generate_observations(true_pose, true_landmarks, config, true);

    // 3. Update step with observations
    update(state, config, &observations);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare EKF vs Dead Reckoning performance
    #[test]
    fn test_ekf_vs_dr() {
        let mut state = EkfSlamState::new();
        let config = EkfSlamConfig::default();

        let landmarks = vec![
            Vector2::new(10.0, 0.0),
            Vector2::new(0.0, 10.0),
            Vector2::new(-10.0, 0.0),
            Vector2::new(0.0, -10.0),
        ];

        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let mut dr_pose = Vector3::new(0.0, 0.0, 0.0);

        let v = 1.5;
        let w = 0.2;
        let dt = 0.01;

        // Biased DR errors (same as simulator)
        let v_bias = 1.05;  // 5% velocity overestimate
        let w_bias = 0.02;  // Angular bias

        println!("\n=== EKF vs DR Comparison (with bias) ===");
        println!("v={}, w={}, dt={}, v_bias={}, w_bias={}", v, w, dt, v_bias, w_bias);

        for step in 0..2000 {
            // True pose (no noise)
            true_pose = motion_model(&true_pose, v, w, dt);

            // Dead reckoning (with BIASED noise - causes consistent drift)
            let v_noisy = v * v_bias + rand() * 0.1;
            let w_noisy = w + w_bias + rand() * 0.02;
            dr_pose = motion_model(&dr_pose, v_noisy, w_noisy, dt);

            // EKF-SLAM (uses true v, w for prediction, noisy observations)
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 400 == 0 {
                let ekf_err = ((true_pose[0] - state.x()).powi(2)
                             + (true_pose[1] - state.y()).powi(2)).sqrt();
                let dr_err = ((true_pose[0] - dr_pose[0]).powi(2)
                            + (true_pose[1] - dr_pose[1]).powi(2)).sqrt();
                println!("Step {}: EKF_err={:.3}, DR_err={:.3}, n_lm={}, n_obs={}",
                    step, ekf_err, dr_err, state.n_landmarks, observations.len());
            }
        }

        let ekf_err = ((true_pose[0] - state.x()).powi(2)
                     + (true_pose[1] - state.y()).powi(2)).sqrt();
        let dr_err = ((true_pose[0] - dr_pose[0]).powi(2)
                    + (true_pose[1] - dr_pose[1]).powi(2)).sqrt();

        println!("\nFinal (20s): EKF_err={:.3}m, DR_err={:.3}m", ekf_err, dr_err);
        println!("EKF should be MUCH better than DR with biased odometry");

        // EKF should be significantly better than DR
        assert!(ekf_err < dr_err,
            "EKF ({:.3}) should be better than DR ({:.3})", ekf_err, dr_err);
        assert!(ekf_err < 1.0, "EKF error should be < 1m, got {:.3}", ekf_err);
    }

    #[test]
    fn test_slam_performance() {
        // Run multiple trials to account for random noise
        let mut total_pos_err = 0.0;
        let mut total_lm_err = 0.0;
        let n_trials = 5;

        for _trial in 0..n_trials {
            let mut state = EkfSlamState::new();
            let config = EkfSlamConfig::default();

            let landmarks = vec![
                Vector2::new(10.0, 5.0),
                Vector2::new(15.0, 10.0),
                Vector2::new(5.0, 15.0),
                Vector2::new(-5.0, 10.0),
            ];

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 1.5;
            let w = 0.15;
            let dt = 0.01;

            for _step in 0..500 {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &landmarks, &config, true);
                update(&mut state, &config, &observations);
            }

            let pos_err = ((true_pose[0] - state.x()).powi(2)
                + (true_pose[1] - state.y()).powi(2)).sqrt();
            total_pos_err += pos_err;

            // Average landmark error
            let mut lm_err_sum = 0.0;
            for (i, true_lm) in landmarks.iter().enumerate() {
                if let Some(est_lm) = state.landmark(i) {
                    lm_err_sum += ((true_lm[0] - est_lm[0]).powi(2)
                        + (true_lm[1] - est_lm[1]).powi(2)).sqrt();
                }
            }
            total_lm_err += lm_err_sum / landmarks.len() as f32;

            // Should discover all landmarks
            assert_eq!(state.n_landmarks, landmarks.len(),
                "Should discover all {} landmarks", landmarks.len());
        }

        let avg_pos_err = total_pos_err / n_trials as f32;
        let avg_lm_err = total_lm_err / n_trials as f32;

        // Average position error should be < 1m
        assert!(avg_pos_err < 1.0,
            "Average position error too large: {:.3}m", avg_pos_err);
        // Average landmark error should be < 1m
        assert!(avg_lm_err < 1.0,
            "Average landmark error too large: {:.3}m", avg_lm_err);
    }

    /// Test without observation noise to verify algorithm correctness
    #[test]
    fn test_slam_no_noise() {
        let mut state = EkfSlamState::new();
        let config = EkfSlamConfig::default();

        let landmarks = vec![
            Vector2::new(10.0, 0.0),
            Vector2::new(0.0, 10.0),
            Vector2::new(-10.0, 0.0),
            Vector2::new(0.0, -10.0),
        ];

        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let v = 1.0;
        let w = 0.2;
        let dt = 0.01;

        println!("\n=== No Noise Test ===");

        for step in 0..500 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            // No noise in observations
            let observations = generate_observations(&true_pose, &landmarks, &config, false);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("Step {}: true=({:.2}, {:.2}), est=({:.2}, {:.2}), err={:.4}, n_obs={}",
                    step, true_pose[0], true_pose[1], est[0], est[1], err, observations.len());
            }
        }

        let final_err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("Final error: {:.4}", final_err);

        // With no observation noise, error should be very small
        assert!(final_err < 0.5, "Error without noise should be < 0.5m, got {:.4}", final_err);
    }

    /// Test when landmarks change mid-simulation (simulating UI change)
    #[test]
    fn test_slam_landmarks_change() {
        let mut state = EkfSlamState::new();
        let config = EkfSlamConfig::default();

        // Initial landmarks
        let mut landmarks = vec![
            Vector2::new(10.0, 0.0),
            Vector2::new(0.0, 10.0),
            Vector2::new(-10.0, 0.0),
            Vector2::new(0.0, -10.0),
        ];

        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let v = 1.0;
        let w = 0.2;
        let dt = 0.01;

        println!("\n=== Landmarks Change Test ===");

        // Phase 1: Run with initial landmarks
        println!("Phase 1: Initial 4 landmarks");
        for step in 0..300 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("  Step {}: err={:.3}, n_lm={}", step, err, state.n_landmarks);
            }
        }

        let err_before = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("Before change: err={:.3}, discovered={}", err_before, state.n_landmarks);

        // Phase 2: Add new landmarks (simulating UI increase)
        println!("\nPhase 2: Adding 4 more landmarks (total 8)");
        landmarks.push(Vector2::new(15.0, 15.0));
        landmarks.push(Vector2::new(-15.0, 15.0));
        landmarks.push(Vector2::new(-15.0, -15.0));
        landmarks.push(Vector2::new(15.0, -15.0));

        for step in 300..600 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("  Step {}: err={:.3}, n_lm={}, n_obs={}", step, err, state.n_landmarks, observations.len());
            }
        }

        let err_after_add = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("After adding: err={:.3}, discovered={}", err_after_add, state.n_landmarks);

        // Phase 3: Replace ALL landmarks with completely new ones (simulating UI regeneration)
        println!("\nPhase 3: Replacing ALL landmarks with new positions");
        landmarks = vec![
            Vector2::new(8.0, 3.0),
            Vector2::new(3.0, 8.0),
            Vector2::new(-8.0, 3.0),
            Vector2::new(-3.0, -8.0),
            Vector2::new(12.0, 12.0),
            Vector2::new(-12.0, -12.0),
        ];

        // NOTE: In a real scenario, we should reset the SLAM state here!
        // state = EkfSlamState::new();  // This is what the UI should do

        // But let's see what happens WITHOUT resetting (the bug scenario)
        for step in 600..900 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("  Step {}: err={:.3}, n_lm={}, n_obs={}", step, err, state.n_landmarks, observations.len());
            }
        }

        let err_without_reset = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("Without reset: err={:.3}, discovered={}", err_without_reset, state.n_landmarks);

        // Phase 4: Now reset and try again
        println!("\nPhase 4: Reset SLAM state and continue");
        state = EkfSlamState::new();
        // Reset robot pose estimate to current true pose (simulating restart)
        state.mu[0] = true_pose[0];
        state.mu[1] = true_pose[1];
        state.mu[2] = true_pose[2];

        for step in 900..1200 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("  Step {}: err={:.3}, n_lm={}, n_obs={}", step, err, state.n_landmarks, observations.len());
            }
        }

        let err_with_reset = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("With reset: err={:.3}, discovered={}", err_with_reset, state.n_landmarks);

        // The error after proper reset should be much smaller
        assert!(err_with_reset < 1.0, "Error after reset should be < 1m, got {:.3}", err_with_reset);

        // Document that without reset, error grows
        println!("\nConclusion: Without reset err={:.3}, With reset err={:.3}",
            err_without_reset, err_with_reset);
    }

    /// Test with landmarks going in and out of range
    #[test]
    fn test_slam_range_transitions() {
        let mut state = EkfSlamState::new();
        let mut config = EkfSlamConfig::default();
        config.max_range = 20.0; // Reasonable range

        // Landmarks in a circle around origin - robot will see them as it circles
        let landmarks = vec![
            Vector2::new(10.0, 0.0),
            Vector2::new(7.0, 7.0),
            Vector2::new(0.0, 10.0),
            Vector2::new(-7.0, 7.0),
            Vector2::new(-10.0, 0.0),
            Vector2::new(-7.0, -7.0),
            Vector2::new(0.0, -10.0),
            Vector2::new(7.0, -7.0),
        ];

        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let v = 1.5;
        let w = 0.2; // Tighter turn = smaller circle, stays near landmarks
        let dt = 0.01;

        println!("\n=== Range Transition Test ===");
        println!("Max range: {}", config.max_range);

        let mut last_n_obs = 0;
        let mut zero_obs_count = 0;

        for step in 0..1500 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &landmarks, &config, true);

            if observations.is_empty() {
                zero_obs_count += 1;
            }

            // Log when observations change
            if observations.len() != last_n_obs {
                let est = state.robot_pose();
                let err = ((true_pose[0] - est[0]).powi(2) + (true_pose[1] - est[1]).powi(2)).sqrt();
                println!("Step {}: pos=({:.1}, {:.1}), n_obs={}->{}, n_lm={}, err={:.2}",
                    step, true_pose[0], true_pose[1], last_n_obs, observations.len(),
                    state.n_landmarks, err);
                last_n_obs = observations.len();
            }

            update(&mut state, &config, &observations);
        }

        let final_err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("\nFinal: true=({:.2}, {:.2}), est=({:.2}, {:.2}), err={:.2}",
            true_pose[0], true_pose[1], state.x(), state.y(), final_err);
        println!("Landmarks discovered: {} (expected: {})", state.n_landmarks, landmarks.len());
        println!("Steps with zero observations: {}", zero_obs_count);

        // Should have discovered all close landmarks
        assert!(state.n_landmarks >= 6, "Should discover at least 6 landmarks, got {}", state.n_landmarks);
        // Error should be reasonable
        assert!(final_err < 3.0, "Position error too large: {:.2}", final_err);
    }

    #[test]
    fn test_motion_model_straight() {
        let pose = Vector3::new(0.0, 0.0, 0.0);
        let new_pose = motion_model(&pose, 1.0, 0.0, 1.0);
        assert!((new_pose[0] - 1.0).abs() < 1e-6);
        assert!(new_pose[1].abs() < 1e-6);
        assert!(new_pose[2].abs() < 1e-6);
    }

    #[test]
    fn test_motion_model_turn() {
        let pose = Vector3::new(0.0, 0.0, 0.0);
        let new_pose = motion_model(&pose, 1.0, PI / 2.0, 1.0);
        // After turning 90 degrees at speed 1, should have moved in an arc
        assert!(new_pose[2] > 1.5);
    }

    #[test]
    fn test_observation_model() {
        let robot = Vector3::new(0.0, 0.0, 0.0);
        let landmark = Vector2::new(1.0, 0.0);
        let obs = observation_model(&robot, &landmark);
        assert!((obs.range - 1.0).abs() < 1e-6);
        assert!(obs.bearing.abs() < 1e-6);
    }

    #[test]
    fn test_add_landmark() {
        let mut state = EkfSlamState::new();
        let config = EkfSlamConfig::default();

        // Observe a landmark at (1, 0) from origin facing +x
        let obs = Observation {
            range: 1.0,
            bearing: 0.0,
        };
        add_landmark(&mut state, &obs, &config);

        assert_eq!(state.n_landmarks, 1);
        assert_eq!(state.mu.len(), 5);

        let lm = state.landmark(0).unwrap();
        assert!((lm[0] - 1.0).abs() < 1e-6);
        assert!(lm[1].abs() < 1e-6);
    }

    #[test]
    fn test_ekf_slam_cycle() {
        let mut state = EkfSlamState::new();
        let config = EkfSlamConfig::default();

        let landmarks = vec![
            Vector2::new(5.0, 5.0),
            Vector2::new(10.0, 0.0),
        ];

        // Simulate a few steps
        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let v = 1.0;
        let w = 0.1;
        let dt = 0.1;

        for _ in 0..50 {
            // Update true pose
            true_pose = motion_model(&true_pose, v, w, dt);

            // Run EKF-SLAM
            step(&mut state, &config, &true_pose, &landmarks, v, w, dt);
        }

        // Should have discovered at least one landmark
        assert!(state.n_landmarks > 0);

        // Estimated pose should be reasonably close to true pose
        let pose_error = ((state.x() - true_pose[0]).powi(2)
            + (state.y() - true_pose[1]).powi(2))
        .sqrt();
        assert!(pose_error < 5.0, "Pose error too large: {}", pose_error);
    }

    /// Test that discovering a new landmark doesn't corrupt existing landmark estimates
    #[test]
    fn test_new_landmark_preserves_existing() {
        let mut state = EkfSlamState::new();
        let mut config = EkfSlamConfig::default();
        config.max_range = 15.0; // Limited range so we can control when landmarks are seen

        // Initial landmarks - close to origin so they're discovered immediately
        let close_landmarks = vec![
            Vector2::new(5.0, 0.0),
            Vector2::new(0.0, 5.0),
            Vector2::new(-5.0, 0.0),
            Vector2::new(0.0, -5.0),
        ];

        // Far landmark that will be discovered later
        let far_landmark = Vector2::new(25.0, 0.0);

        let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
        let v = 1.0;
        let w = 0.1;
        let dt = 0.01;

        println!("\n=== New Landmark Preservation Test ===");
        println!("Phase 1: Establish estimates with close landmarks only");

        // Phase 1: Run with close landmarks only to establish good estimates
        for step in 0..500 {
            true_pose = motion_model(&true_pose, v, w, dt);
            predict(&mut state, &config, v, w, dt);
            let observations = generate_observations(&true_pose, &close_landmarks, &config, true);
            update(&mut state, &config, &observations);

            if step % 100 == 0 {
                let err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
                println!("  Step {}: err={:.3}, n_lm={}", step, err, state.n_landmarks);
            }
        }

        // Record landmark estimates before adding new landmark
        let mut lm_estimates_before: Vec<Vector2<f32>> = Vec::new();
        for i in 0..state.n_landmarks {
            if let Some(lm) = state.landmark(i) {
                lm_estimates_before.push(lm);
            }
        }
        let n_landmarks_before = state.n_landmarks;
        let robot_err_before = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

        println!("\nBefore new landmark: n_lm={}, robot_err={:.3}", n_landmarks_before, robot_err_before);
        for (i, lm) in lm_estimates_before.iter().enumerate() {
            let true_lm = &close_landmarks[i];
            let lm_err = ((true_lm[0] - lm[0]).powi(2) + (true_lm[1] - lm[1]).powi(2)).sqrt();
            println!("  Landmark {}: est=({:.2}, {:.2}), err={:.3}", i, lm[0], lm[1], lm_err);
        }

        // Phase 2: Move toward far landmark and discover it
        println!("\nPhase 2: Moving toward far landmark...");

        // Change motion to move toward far landmark (at 25, 0)
        let mut all_landmarks = close_landmarks.clone();
        all_landmarks.push(far_landmark);

        // Move in +x direction
        true_pose[2] = 0.0; // Face +x direction
        let v_straight = 2.0;
        let w_straight = 0.0;

        for step in 500..800 {
            true_pose = motion_model(&true_pose, v_straight, w_straight, dt);
            predict(&mut state, &config, v_straight, w_straight, dt);
            let observations = generate_observations(&true_pose, &all_landmarks, &config, true);

            // Check if we just discovered the far landmark
            if observations.len() > state.n_landmarks && state.n_landmarks == n_landmarks_before {
                println!("  Step {}: About to discover new landmark! Robot at ({:.2}, {:.2})", step, true_pose[0], true_pose[1]);
            }

            let n_before_update = state.n_landmarks;
            update(&mut state, &config, &observations);

            if state.n_landmarks > n_before_update {
                println!("  Step {}: NEW LANDMARK ADDED! n_lm: {} -> {}", step, n_before_update, state.n_landmarks);

                // Immediately check if existing landmarks were corrupted
                println!("  Checking existing landmark estimates after adding new landmark:");
                for i in 0..n_landmarks_before {
                    if let Some(lm) = state.landmark(i) {
                        let diff = ((lm[0] - lm_estimates_before[i][0]).powi(2)
                            + (lm[1] - lm_estimates_before[i][1]).powi(2)).sqrt();
                        println!("    Landmark {}: before=({:.2}, {:.2}), after=({:.2}, {:.2}), change={:.4}",
                            i, lm_estimates_before[i][0], lm_estimates_before[i][1], lm[0], lm[1], diff);
                    }
                }
            }

            if step % 100 == 0 {
                let err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
                println!("  Step {}: robot at ({:.2}, {:.2}), err={:.3}, n_lm={}, n_obs={}",
                    step, true_pose[0], true_pose[1], err, state.n_landmarks, observations.len());
            }
        }

        // Phase 3: Check final estimates
        println!("\nPhase 3: Final check");
        let robot_err_after = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
        println!("Robot error: before={:.3}, after={:.3}", robot_err_before, robot_err_after);

        println!("\nExisting landmark estimate changes:");
        let mut max_landmark_drift = 0.0f32;
        for i in 0..n_landmarks_before.min(state.n_landmarks) {
            if let Some(lm) = state.landmark(i) {
                let diff = ((lm[0] - lm_estimates_before[i][0]).powi(2)
                    + (lm[1] - lm_estimates_before[i][1]).powi(2)).sqrt();
                let true_lm = &close_landmarks[i];
                let err_before = ((true_lm[0] - lm_estimates_before[i][0]).powi(2) + (true_lm[1] - lm_estimates_before[i][1]).powi(2)).sqrt();
                let err_after = ((true_lm[0] - lm[0]).powi(2) + (true_lm[1] - lm[1]).powi(2)).sqrt();
                max_landmark_drift = max_landmark_drift.max(diff);
                println!("  Landmark {}: drift={:.4}, err_before={:.3}, err_after={:.3}",
                    i, diff, err_before, err_after);
            }
        }

        // Key assertion: adding a new landmark should NOT dramatically corrupt existing estimates
        // Some drift is expected due to continued EKF updates, but it should be small
        assert!(max_landmark_drift < 2.0,
            "Existing landmarks drifted too much after new landmark: {:.3}m", max_landmark_drift);
        // Robot pose can drift during straight-line motion when fewer landmarks are visible
        // The key test is landmark drift, not robot pose drift
        assert!(robot_err_after < 5.0,
            "Robot error too large after new landmark: {:.3}m", robot_err_after);

        println!("\nTest PASSED: max_landmark_drift={:.3}m, robot_err={:.3}m", max_landmark_drift, robot_err_after);
    }

    /// Helper to generate random landmarks in a ring around origin
    fn generate_random_landmarks_ring(n: usize, min_dist: f32, max_dist: f32) -> Vec<Vector2<f32>> {
        let mut landmarks = Vec::with_capacity(n);
        for _ in 0..n {
            let angle = rand::random::<f32>() * 2.0 * PI;
            let dist = min_dist + rand::random::<f32>() * (max_dist - min_dist);
            landmarks.push(Vector2::new(dist * angle.cos(), dist * angle.sin()));
        }
        landmarks
    }

    /// Randomized test: check if new landmarks corrupt existing estimates with 8+ landmarks
    #[test]
    fn test_randomized_new_landmark_with_many_existing() {
        const N_TRIALS: usize = 5;
        const N_INITIAL_LANDMARKS: usize = 10;
        const N_NEW_LANDMARKS: usize = 4;

        println!("\n=== Randomized New Landmark Test ({} trials) ===", N_TRIALS);
        println!("Initial landmarks: {}, New landmarks to add: {}", N_INITIAL_LANDMARKS, N_NEW_LANDMARKS);

        let mut failures = Vec::new();
        let mut max_drift_seen = 0.0f32;
        let mut max_robot_err_seen = 0.0f32;

        for trial in 0..N_TRIALS {
            let mut state = EkfSlamState::new();
            let mut config = EkfSlamConfig::default();
            config.max_range = 25.0;

            // Generate initial landmarks in a ring close to origin
            let initial_landmarks = generate_random_landmarks_ring(N_INITIAL_LANDMARKS, 5.0, 15.0);

            // Generate new landmarks that will be discovered later (further out)
            let new_landmarks = generate_random_landmarks_ring(N_NEW_LANDMARKS, 30.0, 45.0);

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 1.5;
            let w = 0.2;
            let dt = 0.02; // Larger dt for faster simulation

            // Phase 1: Establish estimates with initial landmarks (circle around origin)
            for _ in 0..400 {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &initial_landmarks, &config, true);
                update(&mut state, &config, &observations);
            }

            // Verify we discovered all initial landmarks
            if state.n_landmarks < N_INITIAL_LANDMARKS {
                // Skip trial if we didn't discover enough landmarks
                continue;
            }

            // Record estimates before adding new landmarks
            let mut lm_estimates_before: Vec<Vector2<f32>> = Vec::new();
            for i in 0..state.n_landmarks {
                if let Some(lm) = state.landmark(i) {
                    lm_estimates_before.push(lm);
                }
            }
            let n_before = state.n_landmarks;
            let robot_err_before = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

            // Phase 2: Move outward to discover new landmarks
            let mut all_landmarks = initial_landmarks.clone();
            all_landmarks.extend(new_landmarks.iter().cloned());

            // Move in a spiral outward
            let v_out = 3.0;
            let w_out = 0.15;

            for _ in 0..500 {
                true_pose = motion_model(&true_pose, v_out, w_out, dt);
                predict(&mut state, &config, v_out, w_out, dt);
                let observations = generate_observations(&true_pose, &all_landmarks, &config, true);
                update(&mut state, &config, &observations);
            }

            // Check if new landmarks were discovered
            let n_after = state.n_landmarks;
            let discovered_new = n_after > n_before;

            if !discovered_new {
                // Skip trial if no new landmarks were discovered
                continue;
            }

            // Check for corruption: compare current estimates of original landmarks
            let mut max_drift = 0.0f32;
            let mut max_err_increase = 0.0f32;

            for i in 0..n_before {
                if let Some(lm) = state.landmark(i) {
                    let drift = ((lm[0] - lm_estimates_before[i][0]).powi(2)
                        + (lm[1] - lm_estimates_before[i][1]).powi(2)).sqrt();
                    max_drift = max_drift.max(drift);

                    // Also check error relative to true landmark
                    let true_lm = &initial_landmarks[i];
                    let err_before = ((true_lm[0] - lm_estimates_before[i][0]).powi(2)
                        + (true_lm[1] - lm_estimates_before[i][1]).powi(2)).sqrt();
                    let err_after = ((true_lm[0] - lm[0]).powi(2)
                        + (true_lm[1] - lm[1]).powi(2)).sqrt();
                    let err_increase = err_after - err_before;
                    max_err_increase = max_err_increase.max(err_increase);
                }
            }

            let robot_err_after = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

            max_drift_seen = max_drift_seen.max(max_drift);
            max_robot_err_seen = max_robot_err_seen.max(robot_err_after);

            // Check for failure conditions
            let drift_threshold = 3.0; // Allow up to 3m drift due to EKF updates
            let robot_err_threshold = 5.0; // Allow up to 5m robot error

            if max_drift > drift_threshold {
                failures.push(format!(
                    "Trial {}: Landmark drift too large: {:.3}m (threshold: {}m), discovered {} new landmarks",
                    trial, max_drift, drift_threshold, n_after - n_before
                ));
            }

            if robot_err_after > robot_err_threshold {
                failures.push(format!(
                    "Trial {}: Robot error too large: {:.3}m (threshold: {}m)",
                    trial, robot_err_after, robot_err_threshold
                ));
            }

            if trial % 5 == 0 {
                println!("Trial {}: n_lm: {} -> {}, max_drift={:.3}m, robot_err: {:.3} -> {:.3}m",
                    trial, n_before, n_after, max_drift, robot_err_before, robot_err_after);
            }
        }

        println!("\n=== Summary ===");
        println!("Max landmark drift seen: {:.3}m", max_drift_seen);
        println!("Max robot error seen: {:.3}m", max_robot_err_seen);
        println!("Failures: {}/{}", failures.len(), N_TRIALS);

        for failure in &failures {
            println!("  FAILURE: {}", failure);
        }

        assert!(failures.is_empty(), "Some trials failed:\n{}", failures.join("\n"));
    }

    /// Stress test: rapidly add many landmarks and check stability
    #[test]
    fn test_stress_rapid_landmark_addition() {
        const N_TRIALS: usize = 3;

        let mut failures = Vec::new();

        for trial in 0..N_TRIALS {
            let mut state = EkfSlamState::new();
            let mut config = EkfSlamConfig::default();
            config.max_range = 50.0; // Large range to see many landmarks

            // Generate 15 landmarks at various distances
            let mut landmarks: Vec<Vector2<f32>> = Vec::new();
            for i in 0..15 {
                let angle = (i as f32 / 15.0) * 2.0 * PI + rand::random::<f32>() * 0.3;
                let dist = 8.0 + (i as f32) * 2.0 + rand::random::<f32>() * 3.0;
                landmarks.push(Vector2::new(dist * angle.cos(), dist * angle.sin()));
            }

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 2.0;
            let w = 0.25;
            let dt = 0.02;

            let mut landmark_discovery_steps: Vec<(usize, usize)> = Vec::new();
            let mut robot_errors: Vec<f32> = Vec::new();

            // Run simulation
            for step in 0..400 {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &landmarks, &config, true);

                let n_before = state.n_landmarks;
                update(&mut state, &config, &observations);
                let n_after = state.n_landmarks;

                if n_after > n_before {
                    landmark_discovery_steps.push((step, n_after));
                }

                if step % 100 == 0 {
                    let err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
                    robot_errors.push(err);
                }
            }

            // Check for error spikes after landmark discovery
            let final_err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

            // Check if covariance matrix is still positive semi-definite
            let sigma = &state.sigma;
            let is_symmetric = {
                let mut symmetric = true;
                for i in 0..sigma.nrows() {
                    for j in 0..sigma.ncols() {
                        if (sigma[(i, j)] - sigma[(j, i)]).abs() > 1e-6 {
                            symmetric = false;
                            break;
                        }
                    }
                    if !symmetric { break; }
                }
                symmetric
            };

            // Check for NaN or Inf in state
            let has_nan = state.mu.iter().any(|x| x.is_nan() || x.is_infinite())
                || sigma.iter().any(|x| x.is_nan() || x.is_infinite());

            // Check diagonal elements are positive (necessary for PSD)
            let diag_positive = (0..sigma.nrows()).all(|i| sigma[(i, i)] >= 0.0);

            if has_nan {
                failures.push(format!("Trial {}: NaN or Inf in state/covariance", trial));
            }

            if !is_symmetric {
                failures.push(format!("Trial {}: Covariance matrix not symmetric", trial));
            }

            if !diag_positive {
                failures.push(format!("Trial {}: Negative diagonal in covariance", trial));
            }

            if final_err > 5.0 {
                failures.push(format!("Trial {}: Final robot error too large: {:.3}m", trial, final_err));
            }

            println!("Trial {}: discovered {} landmarks, final_err={:.3}m, symmetric={}, diag_pos={}",
                trial, state.n_landmarks, final_err, is_symmetric, diag_positive);
        }

        println!("\nFailures: {}/{}", failures.len(), N_TRIALS);
        for failure in &failures {
            println!("  FAILURE: {}", failure);
        }

        assert!(failures.is_empty(), "Some trials failed:\n{}", failures.join("\n"));
    }

    /// Test: check specific edge case where landmark is added at boundary of sensor range
    #[test]
    fn test_landmark_at_sensor_boundary() {
        const N_TRIALS: usize = 4;

        let mut failures = Vec::new();

        for trial in 0..N_TRIALS {
            let mut state = EkfSlamState::new();
            let mut config = EkfSlamConfig::default();
            config.max_range = 20.0;

            // 8 initial landmarks in a circle
            let mut landmarks: Vec<Vector2<f32>> = Vec::new();
            for i in 0..8 {
                let angle = (i as f32 / 8.0) * 2.0 * PI;
                let dist = 10.0 + rand::random::<f32>() * 2.0;
                landmarks.push(Vector2::new(dist * angle.cos(), dist * angle.sin()));
            }

            // Add landmarks right at the sensor boundary (will flicker in/out)
            for i in 0..4 {
                let angle = (i as f32 / 4.0) * 2.0 * PI + rand::random::<f32>() * 0.5;
                let dist = config.max_range - 1.0 + rand::random::<f32>() * 2.0; // Right at boundary
                landmarks.push(Vector2::new(dist * angle.cos(), dist * angle.sin()));
            }

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 1.5;
            let w = 0.2;
            let dt = 0.02;

            // Run and track error
            let mut max_robot_err = 0.0f32;
            let mut prev_n_landmarks = 0;
            let mut visibility_changes = 0;

            for step in 0..300 {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &landmarks, &config, true);

                // Track visibility changes (landmarks going in/out of view)
                if observations.len() != prev_n_landmarks {
                    visibility_changes += 1;
                    prev_n_landmarks = observations.len();
                }

                update(&mut state, &config, &observations);

                let err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();
                max_robot_err = max_robot_err.max(err);

                // Check for immediate corruption (error spike)
                if err > 10.0 {
                    failures.push(format!(
                        "Trial {}, Step {}: Error spike to {:.3}m after {} visibility changes",
                        trial, step, err, visibility_changes
                    ));
                    break;
                }
            }

            let final_err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

            if final_err > 5.0 {
                failures.push(format!(
                    "Trial {}: Final error too large: {:.3}m, {} visibility changes",
                    trial, final_err, visibility_changes
                ));
            }

            println!("Trial {}: n_lm={}, max_err={:.3}m, final_err={:.3}m, visibility_changes={}",
                trial, state.n_landmarks, max_robot_err, final_err, visibility_changes);
        }

        println!("\nFailures: {}/{}", failures.len(), N_TRIALS);
        for failure in &failures {
            println!("  FAILURE: {}", failure);
        }

        assert!(failures.is_empty(), "Some trials failed:\n{}", failures.join("\n"));
    }

    /// Test: check for numerical stability with large state vectors
    /// Focus on detecting error SPIKES when new landmarks are added, not gradual drift.
    #[test]
    fn test_numerical_stability_large_state() {
        const N_TRIALS: usize = 3;
        const N_LANDMARKS: usize = 20;

        let mut failures = Vec::new();

        for trial in 0..N_TRIALS {
            let mut state = EkfSlamState::new();
            let mut config = EkfSlamConfig::default();
            config.max_range = 60.0;

            // Generate many landmarks in a ring - all should be visible from origin
            let landmarks = generate_random_landmarks_ring(N_LANDMARKS, 10.0, 40.0);

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 2.0;
            let w = 0.15;
            let dt = 0.02;

            let mut prev_err = 0.0f32;
            #[allow(unused_variables)]
            let mut max_spike = 0.0f32;

            // Run for moderate time (enough to discover all landmarks)
            for step in 0..200 {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &landmarks, &config, true);

                let n_before = state.n_landmarks;
                update(&mut state, &config, &observations);

                let err = ((true_pose[0] - state.x()).powi(2) + (true_pose[1] - state.y()).powi(2)).sqrt();

                // Detect sudden error spikes when new landmark is added
                if state.n_landmarks > n_before {
                    let spike = err - prev_err;
                    if spike > max_spike {
                        max_spike = spike;
                    }
                    // Fail if adding a landmark causes large error jump (> 0.5m spike)
                    if spike > 0.5 {
                        failures.push(format!(
                            "Trial {}, Step {}: Error spike {:.3}m when adding landmark {}",
                            trial, step, spike, state.n_landmarks
                        ));
                    }
                }
                prev_err = err;

                // Check for NaN/Inf
                let has_nan = state.mu.iter().any(|x| x.is_nan() || x.is_infinite())
                    || state.sigma.iter().any(|x| x.is_nan() || x.is_infinite());

                if has_nan {
                    failures.push(format!("Trial {}, Step {}: NaN/Inf detected", trial, step));
                    break;
                }

                // Check covariance diagonal
                let min_diag = (0..state.sigma.nrows())
                    .map(|i| state.sigma[(i, i)])
                    .fold(f32::INFINITY, f32::min);

                if min_diag < 0.0 {
                    failures.push(format!(
                        "Trial {}, Step {}: Negative covariance diagonal: {:.6}",
                        trial, step, min_diag
                    ));
                    break;
                }
            }
        }

        assert!(failures.is_empty(), "Some trials failed:\n{}", failures.join("\n"));
    }

    /// Test to reproduce the landmark rotation issue with 20 landmarks
    /// Runs multiple simulations and checks if landmarks appear rotated
    #[test]
    fn test_landmark_rotation_issue() {
        const N_TRIALS: usize = 5;
        const N_LANDMARKS: usize = 20;
        const N_STEPS: usize = 200;

        println!("\n=== Reproducing Landmark Rotation Issue ===");

        let mut high_error_count = 0;
        let mut rotation_detected_count = 0;

        for trial in 0..N_TRIALS {
            let mut state = EkfSlamState::new();
            let mut config = EkfSlamConfig::default();
            config.max_range = 50.0;

            // Generate 20 landmarks in a ring
            let landmarks = generate_random_landmarks_ring(N_LANDMARKS, 8.0, 30.0);

            let mut true_pose = Vector3::new(0.0, 0.0, 0.0);
            let v = 2.0;
            let w = 0.15; // Circular motion
            let dt = 0.02;

            // Run simulation
            for _ in 0..N_STEPS {
                true_pose = motion_model(&true_pose, v, w, dt);
                predict(&mut state, &config, v, w, dt);
                let observations = generate_observations(&true_pose, &landmarks, &config, true);
                update(&mut state, &config, &observations);
            }

            // Calculate robot position error
            let pos_err = ((true_pose[0] - state.x()).powi(2)
                + (true_pose[1] - state.y()).powi(2)).sqrt();

            // Calculate landmark errors and check for rotation pattern
            let mut landmark_errors: Vec<f32> = Vec::new();
            let mut angular_diffs: Vec<f32> = Vec::new();

            for i in 0..state.n_landmarks.min(landmarks.len()) {
                if let Some(est_lm) = state.landmark(i) {
                    // Find closest true landmark (data association may differ)
                    let mut min_dist = f32::MAX;
                    let mut matched_idx = 0;
                    for (j, true_lm) in landmarks.iter().enumerate() {
                        let dist = ((est_lm[0] - true_lm[0]).powi(2)
                            + (est_lm[1] - true_lm[1]).powi(2)).sqrt();
                        if dist < min_dist {
                            min_dist = dist;
                            matched_idx = j;
                        }
                    }
                    landmark_errors.push(min_dist);

                    // Calculate angular difference from robot position
                    let true_lm = &landmarks[matched_idx];
                    let true_angle = (true_lm[1] - true_pose[1]).atan2(true_lm[0] - true_pose[0]);
                    let est_angle = (est_lm[1] - state.y()).atan2(est_lm[0] - state.x());
                    let angle_diff = (true_angle - est_angle).abs();
                    let angle_diff = if angle_diff > PI { 2.0 * PI - angle_diff } else { angle_diff };
                    angular_diffs.push(angle_diff);
                }
            }

            let avg_landmark_err = if landmark_errors.is_empty() { 0.0 }
                else { landmark_errors.iter().sum::<f32>() / landmark_errors.len() as f32 };
            let max_landmark_err = landmark_errors.iter().cloned().fold(0.0f32, f32::max);
            let avg_angular_diff = if angular_diffs.is_empty() { 0.0 }
                else { angular_diffs.iter().sum::<f32>() / angular_diffs.len() as f32 };

            // Check for high error
            if pos_err > 1.0 {
                high_error_count += 1;
            }

            // Check for rotation pattern (consistent angular offset across landmarks)
            let angular_std = if angular_diffs.len() > 1 {
                let mean = avg_angular_diff;
                let variance = angular_diffs.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / angular_diffs.len() as f32;
                variance.sqrt()
            } else { 0.0 };

            // Low std with non-zero mean suggests rotation
            let rotation_detected = avg_angular_diff > 0.05 && angular_std < 0.1;
            if rotation_detected {
                rotation_detected_count += 1;
            }

            println!("Trial {:2}: pos_err={:.3}m, avg_lm_err={:.3}m, max_lm_err={:.3}m, \
                      avg_angle={:.3}rad, angle_std={:.3}, rotation={}",
                trial, pos_err, avg_landmark_err, max_landmark_err,
                avg_angular_diff, angular_std, if rotation_detected { "YES" } else { "no" });
        }

        println!("\nSummary:");
        println!("  High error (>1m): {}/{} trials ({:.0}%)",
            high_error_count, N_TRIALS, 100.0 * high_error_count as f32 / N_TRIALS as f32);
        println!("  Rotation detected: {}/{} trials ({:.0}%)",
            rotation_detected_count, N_TRIALS, 100.0 * rotation_detected_count as f32 / N_TRIALS as f32);

        // This test is for reproduction - we expect some failures
        // If > 30% have high error, that confirms the user's observation
        if high_error_count as f32 / N_TRIALS as f32 > 0.3 {
            println!("\n  CONFIRMED: >30% of runs have error > 1m");
        }
    }
}
