//! Loop closure detection for Graph SLAM
//!
//! Loop closure is essential for correcting accumulated drift when the robot
//! revisits a previously mapped area. This module provides:
//!
//! 1. **Proximity search**: Find candidate poses within a spatial threshold
//! 2. **Landmark matching**: Find common landmarks between current and candidate poses
//! 3. **Transform estimation**: Compute relative transform from matched landmarks
//! 4. **Validation**: Chi-squared test to reject false positives
//!
//! ## Algorithm Overview
//!
//! 1. For the current pose, find all poses within `proximity_threshold` distance
//! 2. Filter out recent poses (within `min_temporal_separation`)
//! 3. For each candidate, find common observed landmarks
//! 4. If enough common landmarks (≥3), estimate relative transform using SVD
//! 5. Validate the closure using chi-squared test
//! 6. Add validated closures as odometry constraints
//!
//! ## References
//! - [Real-Time Loop Closure in 2D LIDAR SLAM](https://research.google/pubs/pub45466/)
//! - [slam_toolbox Loop Closure](https://github.com/SteveMacenski/slam_toolbox)

use nalgebra::{Matrix3, Vector2, Vector3};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

/// A detected loop closure between two poses
#[derive(Debug, Clone)]
pub struct LoopClosure {
    /// Index of the candidate (older) pose
    pub from_pose_idx: usize,
    /// Index of the current (newer) pose
    pub to_pose_idx: usize,
    /// Relative transform (dx, dy, dtheta) from candidate to current in candidate's frame
    pub transform: Vector3<f32>,
    /// Confidence score (higher is better)
    pub confidence: f32,
    /// Number of matched landmarks
    pub num_matches: usize,
    /// Mean matched-observation Mahalanobis error (lower is better fit)
    pub mahalanobis_sq: f32,
}

/// Configuration for loop closure detection
#[derive(Debug, Clone)]
pub struct LoopClosureConfig {
    /// Maximum distance to consider a candidate pose (meters)
    pub proximity_threshold: f32,
    /// Minimum number of poses between current and candidate
    pub min_temporal_separation: usize,
    /// Minimum number of common landmarks for matching
    pub min_common_landmarks: usize,
    /// Chi-squared confidence level for validation (0.95 or 0.99)
    pub chi2_confidence: f64,
    /// Maximum mean matched-observation Mahalanobis error for acceptance
    pub max_mahalanobis_sq: f32,
    /// Covariance for loop closure constraints
    pub closure_covariance: Matrix3<f32>,
    /// Enable landmark-based loop closure (works even with pose drift)
    pub enable_landmark_based: bool,
    /// Minimum gap in pose indices since last observation of a landmark
    /// to trigger landmark-based loop closure
    pub landmark_observation_gap: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            proximity_threshold: 2.0,
            min_temporal_separation: 10,
            min_common_landmarks: 3,
            chi2_confidence: 0.95,
            max_mahalanobis_sq: 11.345, // Chi-squared 3 DOF at 99%
            closure_covariance: Matrix3::from_diagonal(&Vector3::new(0.2, 0.2, 0.05)),
            enable_landmark_based: true,
            landmark_observation_gap: 15,
        }
    }
}

/// Loop-closure detector for graph SLAM.
///
/// The detector combines simple spatial heuristics with landmark-consistency
/// checks so that revisiting a previously mapped region can add a corrective
/// long-range constraint to the graph.
pub struct LoopClosureDetector {
    config: LoopClosureConfig,
}

impl Default for LoopClosureDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopClosureDetector {
    /// Creates a detector with default thresholds.
    pub fn new() -> Self {
        Self {
            config: LoopClosureConfig::default(),
        }
    }

    /// Creates a detector with a caller-provided configuration.
    pub fn with_config(config: LoopClosureConfig) -> Self {
        Self { config }
    }

    /// Detect loop closures for the current pose
    ///
    /// Returns a list of validated loop closures that can be added to the graph.
    pub fn detect(
        &self,
        poses: &[super::Pose2D],
        _landmarks: &[super::Landmark2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        let mut closures = Vec::new();

        // Try proximity-based detection
        closures.extend(self.detect_proximity_based(
            poses,
            observation_constraints,
            current_pose_idx,
        ));

        // Try landmark-based detection (works even with pose drift)
        if self.config.enable_landmark_based {
            closures.extend(self.detect_landmark_based(
                poses,
                observation_constraints,
                current_pose_idx,
            ));
        }

        // Remove duplicates (same from/to pose pair)
        closures.sort_by(|a, b| {
            a.from_pose_idx
                .cmp(&b.from_pose_idx)
                .then(a.to_pose_idx.cmp(&b.to_pose_idx))
        });
        closures
            .dedup_by(|a, b| a.from_pose_idx == b.from_pose_idx && a.to_pose_idx == b.to_pose_idx);

        // Sort by confidence (highest first)
        closures.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        closures
    }

    /// Proximity-based loop closure detection (original algorithm)
    fn detect_proximity_based(
        &self,
        poses: &[super::Pose2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        if current_pose_idx < self.config.min_temporal_separation {
            return Vec::new();
        }

        let current_pose = &poses[current_pose_idx];
        let mut closures = Vec::new();

        // Build pose-to-landmarks map
        let pose_landmarks = self.build_pose_landmarks_map(observation_constraints);

        // Get landmarks visible from current pose
        let current_landmarks: HashSet<usize> = pose_landmarks
            .get(&current_pose_idx)
            .cloned()
            .unwrap_or_default();

        if current_landmarks.len() < self.config.min_common_landmarks {
            return Vec::new();
        }

        // Find candidate poses
        let max_candidate_idx =
            current_pose_idx.saturating_sub(self.config.min_temporal_separation);

        for (candidate_idx, candidate_pose) in poses.iter().enumerate().take(max_candidate_idx + 1)
        {
            // Proximity check
            let dx = current_pose.x - candidate_pose.x;
            let dy = current_pose.y - candidate_pose.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > self.config.proximity_threshold {
                continue;
            }

            // Find common landmarks
            let candidate_landmarks: HashSet<usize> = pose_landmarks
                .get(&candidate_idx)
                .cloned()
                .unwrap_or_default();

            let common: Vec<usize> = current_landmarks
                .intersection(&candidate_landmarks)
                .cloned()
                .collect();

            if common.len() < self.config.min_common_landmarks {
                continue;
            }

            // Estimate an independent relative transform directly from the
            // two poses' matched range/bearing observations.
            if let Some((transform, confidence, mahalanobis_sq)) = self.estimate_transform(
                &common,
                observation_constraints,
                candidate_idx,
                current_pose_idx,
            ) {
                if mahalanobis_sq <= self.config.max_mahalanobis_sq {
                    closures.push(LoopClosure {
                        from_pose_idx: candidate_idx,
                        to_pose_idx: current_pose_idx,
                        transform,
                        confidence,
                        num_matches: common.len(),
                        mahalanobis_sq,
                    });
                }
            }
        }

        closures
    }

    /// Landmark-based loop closure detection
    ///
    /// This detects when we see landmarks that we haven't seen for a while,
    /// which indicates returning to a previously visited area. This works
    /// even when pose estimates have drifted significantly.
    fn detect_landmark_based(
        &self,
        _poses: &[super::Pose2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        let mut closures = Vec::new();

        if current_pose_idx < self.config.landmark_observation_gap {
            return Vec::new();
        }

        // Build landmark-to-poses map (which poses observed each landmark)
        let mut landmark_poses: HashMap<usize, Vec<usize>> = HashMap::new();
        for c in observation_constraints {
            landmark_poses
                .entry(c.landmark_idx)
                .or_default()
                .push(c.pose_idx);
        }

        // Get landmarks visible from current pose
        let current_landmarks: HashSet<usize> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == current_pose_idx)
            .map(|c| c.landmark_idx)
            .collect();

        if current_landmarks.is_empty() {
            return Vec::new();
        }

        // For each landmark we see now, check if there's a gap in observations
        // This indicates we left the area and came back
        for &lm_idx in &current_landmarks {
            let Some(observing_poses) = landmark_poses.get(&lm_idx) else {
                continue;
            };

            // Find the most recent pose (before current) that saw this landmark
            let mut last_observation_pose = None;
            for &pose_idx in observing_poses.iter().rev() {
                if pose_idx < current_pose_idx {
                    last_observation_pose = Some(pose_idx);
                    break;
                }
            }

            let Some(last_pose_idx) = last_observation_pose else {
                continue; // First time seeing this landmark
            };

            // Check if there's a significant gap (we were "blind" for a while)
            let gap = current_pose_idx - last_pose_idx;
            if gap < self.config.landmark_observation_gap {
                continue; // Not enough gap to be considered a "return"
            }

            // We have a landmark-based loop closure candidate!
            // Find all landmarks we currently see that were also seen from last_pose_idx
            let last_pose_landmarks: HashSet<usize> = observation_constraints
                .iter()
                .filter(|c| c.pose_idx == last_pose_idx)
                .map(|c| c.landmark_idx)
                .collect();

            let common: Vec<usize> = current_landmarks
                .intersection(&last_pose_landmarks)
                .cloned()
                .collect();

            if common.len() < self.config.min_common_landmarks {
                continue;
            }

            // Landmark-based detection is deliberately independent of the
            // drifted graph poses, so it uses the same observation-consistency
            // gate rather than a relaxed pose-error threshold.
            if let Some((transform, confidence, mahalanobis_sq)) = self.estimate_transform(
                &common,
                observation_constraints,
                last_pose_idx,
                current_pose_idx,
            ) {
                if mahalanobis_sq <= self.config.max_mahalanobis_sq {
                    closures.push(LoopClosure {
                        from_pose_idx: last_pose_idx,
                        to_pose_idx: current_pose_idx,
                        transform,
                        confidence: confidence * 0.8,
                        num_matches: common.len(),
                        mahalanobis_sq,
                    });
                }
            }
        }

        closures
    }

    /// Build a map from pose index to set of observed landmark indices
    fn build_pose_landmarks_map(
        &self,
        observation_constraints: &[super::ObservationConstraint],
    ) -> HashMap<usize, HashSet<usize>> {
        let mut map: HashMap<usize, HashSet<usize>> = HashMap::new();

        for c in observation_constraints {
            map.entry(c.pose_idx).or_default().insert(c.landmark_idx);
        }

        map
    }

    /// Estimate candidate-to-current relative pose directly from matched observations.
    ///
    /// For a shared landmark, let `p_candidate` and `p_current` be its Cartesian
    /// coordinates reconstructed from range/bearing measurements in each robot
    /// frame. The relative robot transform satisfies:
    ///
    /// `p_candidate = R(candidate_to_current) * p_current + t`.
    ///
    /// Aligning current-frame points onto candidate-frame points therefore gives
    /// the loop-closure measurement without consulting the graph pose estimates
    /// or the globally estimated landmark positions.
    fn estimate_transform(
        &self,
        common_landmarks: &[usize],
        observation_constraints: &[super::ObservationConstraint],
        candidate_idx: usize,
        current_idx: usize,
    ) -> Option<(Vector3<f32>, f32, f32)> {
        if common_landmarks.len() < 2 {
            return None;
        }

        let candidate_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == candidate_idx && common_landmarks.contains(&c.landmark_idx))
            .map(|c| (c.landmark_idx, c))
            .collect();
        let current_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == current_idx && common_landmarks.contains(&c.landmark_idx))
            .map(|c| (c.landmark_idx, c))
            .collect();

        let mut points_candidate = Vec::with_capacity(common_landmarks.len());
        let mut points_current = Vec::with_capacity(common_landmarks.len());
        let mut matched_observations = Vec::with_capacity(common_landmarks.len());

        for &lm_idx in common_landmarks {
            let candidate = *candidate_obs.get(&lm_idx)?;
            let current = *current_obs.get(&lm_idx)?;
            points_candidate.push(Self::observation_point(candidate)?);
            points_current.push(Self::observation_point(current)?);
            matched_observations.push((candidate, current));
        }

        // Source=current, target=candidate. This directly estimates the robot
        // transform from candidate pose to current pose in the candidate frame.
        let (rotation, translation) =
            self.compute_rigid_transform(&points_current, &points_candidate)?;
        let residual = self.compute_alignment_residual(
            &points_current,
            &points_candidate,
            rotation,
            &translation,
        );
        let mahalanobis_sq =
            self.compute_alignment_mahalanobis(&matched_observations, rotation, &translation)?;
        let confidence = 1.0 / (1.0 + residual);

        Some((
            Vector3::new(translation.x, translation.y, normalize_angle(rotation)),
            confidence,
            mahalanobis_sq,
        ))
    }

    /// Compute rigid transform (rotation angle, translation) using SVD
    fn compute_rigid_transform(
        &self,
        source: &[Vector2<f32>],
        target: &[Vector2<f32>],
    ) -> Option<(f32, Vector2<f32>)> {
        if source.len() < 2 || source.len() != target.len() {
            return None;
        }

        let n = source.len() as f32;

        // Compute centroids
        let mut src_centroid = Vector2::zeros();
        let mut tgt_centroid = Vector2::zeros();
        for (s, t) in source.iter().zip(target.iter()) {
            src_centroid += s;
            tgt_centroid += t;
        }
        src_centroid /= n;
        tgt_centroid /= n;

        // Center the point sets
        let src_centered: Vec<Vector2<f32>> = source.iter().map(|p| p - src_centroid).collect();
        let tgt_centered: Vec<Vector2<f32>> = target.iter().map(|p| p - tgt_centroid).collect();

        // Compute cross-covariance matrix H
        let mut h = nalgebra::Matrix2::zeros();
        for (s, t) in src_centered.iter().zip(tgt_centered.iter()) {
            h += s * t.transpose();
        }

        // SVD
        let svd = h.svd(true, true);
        let u = svd.u?;
        let v_t = svd.v_t?;

        // Rotation matrix R = V * U^T
        let r = v_t.transpose() * u.transpose();

        // Handle reflection
        let det = r.determinant();
        let r = if det < 0.0 {
            let mut v_t_corrected = v_t;
            v_t_corrected[(1, 0)] *= -1.0;
            v_t_corrected[(1, 1)] *= -1.0;
            v_t_corrected.transpose() * u.transpose()
        } else {
            r
        };

        // Extract rotation angle
        let rotation = r[(1, 0)].atan2(r[(0, 0)]);

        // Compute translation
        let translation = tgt_centroid - r * src_centroid;

        Some((rotation, translation))
    }

    /// Compute alignment residual (RMS error after transform)
    fn compute_alignment_residual(
        &self,
        source: &[Vector2<f32>],
        target: &[Vector2<f32>],
        rotation: f32,
        translation: &Vector2<f32>,
    ) -> f32 {
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        let mut sum_sq = 0.0;
        for (s, t) in source.iter().zip(target.iter()) {
            let transformed = Vector2::new(
                cos_r * s.x - sin_r * s.y + translation.x,
                sin_r * s.x + cos_r * s.y + translation.y,
            );
            let diff = transformed - t;
            sum_sq += diff.dot(&diff);
        }

        (sum_sq / source.len() as f32).sqrt()
    }

    fn observation_point(c: &super::ObservationConstraint) -> Option<Vector2<f32>> {
        let range = c.measurement[0];
        let bearing = c.measurement[1];
        if !range.is_finite() || !bearing.is_finite() || range <= 0.0 {
            return None;
        }
        Some(Vector2::new(range * bearing.cos(), range * bearing.sin()))
    }

    fn observation_cartesian_covariance(
        c: &super::ObservationConstraint,
    ) -> Option<nalgebra::Matrix2<f32>> {
        let range = c.measurement[0];
        let bearing = c.measurement[1];
        if !range.is_finite() || !bearing.is_finite() || range <= 0.0 {
            return None;
        }
        let polar_covariance = c.information.try_inverse()?;
        let jacobian = nalgebra::Matrix2::new(
            bearing.cos(),
            -range * bearing.sin(),
            bearing.sin(),
            range * bearing.cos(),
        );
        Some(jacobian * polar_covariance * jacobian.transpose())
    }

    /// Mean covariance-normalized point-alignment error for matched observations.
    ///
    /// This is independent of the graph pose estimates. Each residual covariance
    /// combines the candidate observation covariance with the current observation
    /// covariance rotated into the candidate frame.
    fn compute_alignment_mahalanobis(
        &self,
        matches: &[(&super::ObservationConstraint, &super::ObservationConstraint)],
        rotation: f32,
        translation: &Vector2<f32>,
    ) -> Option<f32> {
        if matches.is_empty() {
            return None;
        }

        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let rotation_matrix = nalgebra::Matrix2::new(cos_r, -sin_r, sin_r, cos_r);
        let mut total = 0.0;

        for &(candidate, current) in matches {
            let candidate_point = Self::observation_point(candidate)?;
            let current_point = Self::observation_point(current)?;
            let predicted_candidate = rotation_matrix * current_point + *translation;
            let error = predicted_candidate - candidate_point;

            let candidate_covariance = Self::observation_cartesian_covariance(candidate)?;
            let current_covariance = Self::observation_cartesian_covariance(current)?;
            let residual_covariance = candidate_covariance
                + rotation_matrix * current_covariance * rotation_matrix.transpose();
            let information = residual_covariance.try_inverse()?;
            total += error.dot(&(information * error));
        }

        Some(total / matches.len() as f32)
    }

    /// Validate loop closures using chain consistency
    ///
    /// If A-B and B-C closures exist, verify A-C consistency
    pub fn validate_chain_consistency(
        &self,
        closures: &[LoopClosure],
        _poses: &[super::Pose2D],
    ) -> Vec<LoopClosure> {
        // Simple validation: just return closures that pass individual tests
        // Full chain validation would check transitivity
        closures.to_vec()
    }
}

/// Normalize angle to [-π, π]
fn normalize_angle(angle: f32) -> f32 {
    let mut a = angle;
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Helper to add loop closure constraints to a graph
pub fn add_loop_closure_to_graph(
    graph: &mut super::GraphSlam,
    closure: &LoopClosure,
    covariance: &Matrix3<f32>,
) {
    graph.add_odometry(
        closure.from_pose_idx,
        closure.to_pose_idx,
        closure.transform,
        covariance,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slam::{GraphSlam, Landmark2D, Pose2D};
    use nalgebra::{Matrix2, Vector2};

    #[test]
    fn test_loop_closure_detection_square() {
        // Robot drives in a square and returns to start
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;
        graph.config.use_sparse_solver = false;

        // Square trajectory: start -> right -> up -> left -> back to start area
        let poses = vec![
            Pose2D::new(0.0, 0.0, 0.0),
            Pose2D::new(5.0, 0.0, PI / 2.0),
            Pose2D::new(5.0, 5.0, PI),
            Pose2D::new(0.0, 5.0, -PI / 2.0),
            Pose2D::new(0.2, 0.3, 0.1), // Back near start with some drift
        ];

        for pose in &poses {
            graph.add_pose(*pose);
        }

        // Add 4 landmarks at corners of a smaller square
        let landmarks = vec![
            Landmark2D::new(2.0, 2.0),
            Landmark2D::new(3.0, 2.0),
            Landmark2D::new(3.0, 3.0),
            Landmark2D::new(2.0, 3.0),
        ];

        for lm in &landmarks {
            graph.add_landmark(*lm);
        }

        // Add observations from pose 0 and pose 4 to all landmarks
        // (They're both near the center area)
        let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.1, 0.01));
        for (lm_idx, lm) in landmarks.iter().enumerate().take(4) {
            // Pose 0 observations
            let dx = lm.x - poses[0].x;
            let dy = lm.y - poses[0].y;
            let r = (dx * dx + dy * dy).sqrt();
            let b = dy.atan2(dx) - poses[0].theta;
            graph.add_observation(0, lm_idx, r, normalize_angle(b), &obs_cov);

            // Pose 4 observations (same landmarks)
            let dx = lm.x - poses[4].x;
            let dy = lm.y - poses[4].y;
            let r = (dx * dx + dy * dy).sqrt();
            let b = dy.atan2(dx) - poses[4].theta;
            graph.add_observation(4, lm_idx, r, normalize_angle(b), &obs_cov);
        }

        // Detect loop closures
        let config = LoopClosureConfig {
            proximity_threshold: 1.0, // Pose 4 is ~0.36m from pose 0
            min_temporal_separation: 3,
            min_common_landmarks: 3,
            ..Default::default()
        };

        let detector = LoopClosureDetector::with_config(config);
        let closures = detector.detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4, // Current pose
        );

        println!("\n=== Loop Closure Detection Test ===");
        println!("Number of poses: {}", graph.poses.len());
        println!("Number of landmarks: {}", graph.landmarks.len());
        println!("Closures detected: {}", closures.len());

        for (i, c) in closures.iter().enumerate() {
            println!(
                "  Closure {}: pose {} -> pose {}",
                i, c.from_pose_idx, c.to_pose_idx
            );
            println!(
                "    Transform: ({:.3}, {:.3}, {:.3}°)",
                c.transform.x,
                c.transform.y,
                c.transform.z.to_degrees()
            );
            println!(
                "    Matches: {}, Confidence: {:.3}, Mahalanobis: {:.3}",
                c.num_matches, c.confidence, c.mahalanobis_sq
            );
        }

        assert!(
            !closures.is_empty(),
            "Should detect loop closure between pose 0 and 4"
        );
        assert_eq!(closures[0].from_pose_idx, 0);
        assert_eq!(closures[0].to_pose_idx, 4);
    }

    fn add_observations_from_pose(
        graph: &mut GraphSlam,
        pose_idx: usize,
        ground_truth_pose: Pose2D,
        landmarks: &[Landmark2D],
        covariance: &Matrix2<f32>,
    ) {
        for (lm_idx, landmark) in landmarks.iter().enumerate() {
            let dx = landmark.x - ground_truth_pose.x;
            let dy = landmark.y - ground_truth_pose.y;
            let range = (dx * dx + dy * dy).sqrt();
            let bearing = normalize_angle(dy.atan2(dx) - ground_truth_pose.theta);
            graph.add_observation(pose_idx, lm_idx, range, bearing, covariance);
        }
    }

    fn relative_pose(from: Pose2D, to: Pose2D) -> Vector3<f32> {
        let dx_global = to.x - from.x;
        let dy_global = to.y - from.y;
        let cos_t = from.theta.cos();
        let sin_t = from.theta.sin();
        Vector3::new(
            cos_t * dx_global + sin_t * dy_global,
            -sin_t * dx_global + cos_t * dy_global,
            normalize_angle(to.theta - from.theta),
        )
    }

    #[test]
    fn test_landmark_loop_closure_recovers_ground_truth_despite_pose_drift() {
        let mut graph = GraphSlam::new();

        // Deliberately wrong graph estimates. Pose 4 is nowhere near pose 0, so
        // only landmark-based detection can find the revisit.
        for pose in [
            Pose2D::new(-3.0, 2.0, -0.5),
            Pose2D::new(2.0, 1.0, 0.1),
            Pose2D::new(4.0, 0.0, 0.5),
            Pose2D::new(6.0, -2.0, 1.0),
            Pose2D::new(8.0, -5.0, 1.5),
        ] {
            graph.add_pose(pose);
        }

        let landmarks = vec![
            Landmark2D::new(3.0, 1.0),
            Landmark2D::new(2.0, -2.0),
            Landmark2D::new(-1.0, 2.0),
            Landmark2D::new(4.0, 3.0),
        ];
        for landmark in &landmarks {
            graph.add_landmark(*landmark);
        }

        let ground_truth_candidate = Pose2D::new(0.5, -0.5, 0.4);
        let ground_truth_current = Pose2D::new(1.2, 0.3, 0.8);
        let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.01, 0.001));
        add_observations_from_pose(&mut graph, 0, ground_truth_candidate, &landmarks, &obs_cov);
        add_observations_from_pose(&mut graph, 4, ground_truth_current, &landmarks, &obs_cov);

        let config = LoopClosureConfig {
            proximity_threshold: 0.1,
            min_temporal_separation: 3,
            landmark_observation_gap: 3,
            min_common_landmarks: 3,
            ..Default::default()
        };
        let detector = LoopClosureDetector::with_config(config);

        let closures = detector.detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4,
        );
        let closure = closures
            .iter()
            .find(|closure| closure.from_pose_idx == 0 && closure.to_pose_idx == 4)
            .expect("landmark observations should recover the revisit despite pose drift");

        let expected = relative_pose(ground_truth_candidate, ground_truth_current);
        assert!((closure.transform.x - expected.x).abs() < 1e-3);
        assert!((closure.transform.y - expected.y).abs() < 1e-3);
        assert!(normalize_angle(closure.transform.z - expected.z).abs() < 1e-3);

        let drifted_estimate = relative_pose(graph.poses[0], graph.poses[4]);
        assert!(
            (closure.transform - drifted_estimate).norm() > 1.0,
            "closure measurement must be independent of the drifted pose estimates"
        );
    }

    #[test]
    fn test_inconsistent_landmark_geometry_is_rejected() {
        let mut graph = GraphSlam::new();
        for pose in [
            Pose2D::new(0.0, 0.0, 0.0),
            Pose2D::new(2.0, 0.0, 0.0),
            Pose2D::new(4.0, 0.0, 0.0),
            Pose2D::new(6.0, 0.0, 0.0),
            Pose2D::new(8.0, 0.0, 0.0),
        ] {
            graph.add_pose(pose);
        }
        for landmark in [
            Landmark2D::new(2.0, 0.0),
            Landmark2D::new(0.0, 2.0),
            Landmark2D::new(-2.0, 0.0),
            Landmark2D::new(0.0, -2.0),
        ] {
            graph.add_landmark(landmark);
        }

        let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.001, 0.0001));
        let candidate_points: [Vector2<f32>; 4] = [
            Vector2::new(2.0, 0.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(-2.0, 0.0),
            Vector2::new(0.0, -2.0),
        ];
        let current_points: [Vector2<f32>; 4] = [
            Vector2::new(2.0, 0.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(-1.0, -1.0),
        ];
        for (lm_idx, point) in candidate_points.iter().enumerate() {
            graph.add_observation(0, lm_idx, point.norm(), point.y.atan2(point.x), &obs_cov);
        }
        for (lm_idx, point) in current_points.iter().enumerate() {
            graph.add_observation(4, lm_idx, point.norm(), point.y.atan2(point.x), &obs_cov);
        }

        let config = LoopClosureConfig {
            proximity_threshold: 0.1,
            landmark_observation_gap: 3,
            min_common_landmarks: 3,
            ..Default::default()
        };
        let detector = LoopClosureDetector::with_config(config);

        let closures = detector.detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4,
        );
        assert!(
            closures.is_empty(),
            "non-rigidly inconsistent shared observations must not create a closure"
        );
    }

    #[test]
    fn test_rigid_transform_estimation() {
        let detector = LoopClosureDetector::new();

        // Simple translation test
        let source = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        ];
        let target = vec![
            Vector2::new(1.0, 1.0),
            Vector2::new(2.0, 1.0),
            Vector2::new(2.0, 2.0),
        ];

        let result = detector.compute_rigid_transform(&source, &target);
        assert!(result.is_some());

        let (rotation, translation) = result.unwrap();
        println!(
            "Rotation: {:.3}°, Translation: ({:.3}, {:.3})",
            rotation.to_degrees(),
            translation.x,
            translation.y
        );

        assert!(rotation.abs() < 0.1, "Rotation should be near zero");
        assert!(
            (translation.x - 1.0).abs() < 0.1,
            "Translation X should be ~1"
        );
        assert!(
            (translation.y - 1.0).abs() < 0.1,
            "Translation Y should be ~1"
        );
    }

    #[test]
    fn test_rigid_transform_with_rotation() {
        let detector = LoopClosureDetector::new();

        // 90 degree rotation
        let source = vec![
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(2.0, 1.0),
        ];
        let target = vec![
            Vector2::new(0.0, 1.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(-1.0, 2.0),
        ];

        let result = detector.compute_rigid_transform(&source, &target);
        assert!(result.is_some());

        let (rotation, _translation) = result.unwrap();
        println!("Rotation: {:.3}° (expected ~90°)", rotation.to_degrees());

        assert!(
            (rotation - PI / 2.0).abs() < 0.1,
            "Rotation should be ~90 degrees"
        );
    }
}
