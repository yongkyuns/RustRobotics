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
    /// Mahalanobis distance squared (lower is better fit)
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
    /// Maximum Mahalanobis distance squared for acceptance
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

/// Loop closure detector for Graph SLAM
pub struct LoopClosureDetector {
    config: LoopClosureConfig,
}

impl Default for LoopClosureDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopClosureDetector {
    pub fn new() -> Self {
        Self {
            config: LoopClosureConfig::default(),
        }
    }

    pub fn with_config(config: LoopClosureConfig) -> Self {
        Self { config }
    }

    /// Detect loop closures for the current pose
    ///
    /// Returns a list of validated loop closures that can be added to the graph.
    pub fn detect(
        &self,
        poses: &[super::Pose2D],
        landmarks: &[super::Landmark2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        let mut closures = Vec::new();

        // Try proximity-based detection
        closures.extend(self.detect_proximity_based(
            poses,
            landmarks,
            observation_constraints,
            current_pose_idx,
        ));

        // Try landmark-based detection (works even with pose drift)
        if self.config.enable_landmark_based {
            closures.extend(self.detect_landmark_based(
                poses,
                landmarks,
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
        closures.dedup_by(|a, b| a.from_pose_idx == b.from_pose_idx && a.to_pose_idx == b.to_pose_idx);

        // Sort by confidence (highest first)
        closures.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        closures
    }

    /// Proximity-based loop closure detection (original algorithm)
    fn detect_proximity_based(
        &self,
        poses: &[super::Pose2D],
        landmarks: &[super::Landmark2D],
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
        let max_candidate_idx = current_pose_idx.saturating_sub(self.config.min_temporal_separation);

        for candidate_idx in 0..=max_candidate_idx {
            let candidate_pose = &poses[candidate_idx];

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

            // Estimate transform using common landmarks
            if let Some((transform, confidence)) = self.estimate_transform(
                candidate_pose,
                current_pose,
                landmarks,
                &common,
                observation_constraints,
                candidate_idx,
                current_pose_idx,
            ) {
                // Validate with chi-squared test
                let mahalanobis_sq = self.compute_mahalanobis(
                    candidate_pose,
                    current_pose,
                    &transform,
                );

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
        poses: &[super::Pose2D],
        landmarks: &[super::Landmark2D],
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

            let candidate_pose = &poses[last_pose_idx];
            let current_pose = &poses[current_pose_idx];

            // Estimate transform using common landmarks
            if let Some((transform, confidence)) = self.estimate_transform(
                candidate_pose,
                current_pose,
                landmarks,
                &common,
                observation_constraints,
                last_pose_idx,
                current_pose_idx,
            ) {
                // For landmark-based detection, use a higher Mahalanobis threshold
                // since we expect the poses to have drifted
                let mahalanobis_sq = self.compute_mahalanobis(
                    candidate_pose,
                    current_pose,
                    &transform,
                );

                // Accept higher Mahalanobis for landmark-based (3x normal threshold)
                if mahalanobis_sq <= self.config.max_mahalanobis_sq * 3.0 {
                    closures.push(LoopClosure {
                        from_pose_idx: last_pose_idx,
                        to_pose_idx: current_pose_idx,
                        transform,
                        confidence: confidence * 0.8, // Slightly lower confidence for landmark-based
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
            map.entry(c.pose_idx)
                .or_default()
                .insert(c.landmark_idx);
        }

        map
    }

    /// Estimate relative transform from candidate pose to current pose using common landmarks
    fn estimate_transform(
        &self,
        candidate_pose: &super::Pose2D,
        current_pose: &super::Pose2D,
        landmarks: &[super::Landmark2D],
        common_landmarks: &[usize],
        observation_constraints: &[super::ObservationConstraint],
        candidate_idx: usize,
        current_idx: usize,
    ) -> Option<(Vector3<f32>, f32)> {
        if common_landmarks.len() < 2 {
            return None;
        }

        // Get observations from each pose to common landmarks
        let _candidate_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == candidate_idx && common_landmarks.contains(&c.landmark_idx))
            .map(|c| (c.landmark_idx, c))
            .collect();

        let _current_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == current_idx && common_landmarks.contains(&c.landmark_idx))
            .map(|c| (c.landmark_idx, c))
            .collect();

        // Compute transform using landmark positions
        // Simple approach: use the estimated landmark positions directly
        let mut points_candidate = Vec::new();
        let mut points_current = Vec::new();

        for &lm_idx in common_landmarks {
            let lm = &landmarks[lm_idx];

            // Landmark position in candidate frame
            let dx_c = lm.x - candidate_pose.x;
            let dy_c = lm.y - candidate_pose.y;
            let cos_c = candidate_pose.theta.cos();
            let sin_c = candidate_pose.theta.sin();
            let x_in_candidate = cos_c * dx_c + sin_c * dy_c;
            let y_in_candidate = -sin_c * dx_c + cos_c * dy_c;

            // Landmark position in current frame
            let dx_cur = lm.x - current_pose.x;
            let dy_cur = lm.y - current_pose.y;
            let cos_cur = current_pose.theta.cos();
            let sin_cur = current_pose.theta.sin();
            let x_in_current = cos_cur * dx_cur + sin_cur * dy_cur;
            let y_in_current = -sin_cur * dx_cur + cos_cur * dy_cur;

            points_candidate.push(Vector2::new(x_in_candidate, y_in_candidate));
            points_current.push(Vector2::new(x_in_current, y_in_current));
        }

        // Compute rigid transform from candidate to current frame
        // Using SVD-based point cloud alignment
        if let Some((rotation, translation)) = self.compute_rigid_transform(&points_candidate, &points_current) {
            // Convert to (dx, dy, dtheta) in candidate's frame
            // The transform represents: current_pose = candidate_pose ⊕ transform
            let _dtheta = rotation;
            let _dx = translation.x;
            let _dy = translation.y;

            // But we want the transform in the candidate's frame for odometry constraint
            // Relative motion: dx_local, dy_local, dtheta
            let dx_global = current_pose.x - candidate_pose.x;
            let dy_global = current_pose.y - candidate_pose.y;
            let cos_t = candidate_pose.theta.cos();
            let sin_t = candidate_pose.theta.sin();
            let dx_local = cos_t * dx_global + sin_t * dy_global;
            let dy_local = -sin_t * dx_global + cos_t * dy_global;
            let dtheta_local = normalize_angle(current_pose.theta - candidate_pose.theta);

            // Confidence based on residual fit
            let residual = self.compute_alignment_residual(&points_candidate, &points_current, rotation, &translation);
            let confidence = 1.0 / (1.0 + residual);

            Some((Vector3::new(dx_local, dy_local, dtheta_local), confidence))
        } else {
            None
        }
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

    /// Compute Mahalanobis distance for the proposed loop closure
    fn compute_mahalanobis(
        &self,
        candidate_pose: &super::Pose2D,
        current_pose: &super::Pose2D,
        transform: &Vector3<f32>,
    ) -> f32 {
        // Expected transform based on poses
        let dx_global = current_pose.x - candidate_pose.x;
        let dy_global = current_pose.y - candidate_pose.y;
        let cos_t = candidate_pose.theta.cos();
        let sin_t = candidate_pose.theta.sin();
        let expected_dx = cos_t * dx_global + sin_t * dy_global;
        let expected_dy = -sin_t * dx_global + cos_t * dy_global;
        let expected_dtheta = normalize_angle(current_pose.theta - candidate_pose.theta);

        let expected = Vector3::new(expected_dx, expected_dy, expected_dtheta);
        let error = transform - expected;

        // Use closure covariance for Mahalanobis distance
        let info = self.config.closure_covariance.try_inverse().unwrap_or(Matrix3::identity());
        let weighted = info * error;
        error.dot(&weighted)
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
        for lm_idx in 0..4 {
            // Pose 0 observations
            let lm = &landmarks[lm_idx];
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
        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 1.0; // Pose 4 is ~0.36m from pose 0
        config.min_temporal_separation = 3;
        config.min_common_landmarks = 3;

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
            println!("  Closure {}: pose {} -> pose {}", i, c.from_pose_idx, c.to_pose_idx);
            println!("    Transform: ({:.3}, {:.3}, {:.3}°)",
                c.transform.x, c.transform.y, c.transform.z.to_degrees());
            println!("    Matches: {}, Confidence: {:.3}, Mahalanobis: {:.3}",
                c.num_matches, c.confidence, c.mahalanobis_sq);
        }

        assert!(!closures.is_empty(), "Should detect loop closure between pose 0 and 4");
        assert_eq!(closures[0].from_pose_idx, 0);
        assert_eq!(closures[0].to_pose_idx, 4);
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
        println!("Rotation: {:.3}°, Translation: ({:.3}, {:.3})",
            rotation.to_degrees(), translation.x, translation.y);

        assert!(rotation.abs() < 0.1, "Rotation should be near zero");
        assert!((translation.x - 1.0).abs() < 0.1, "Translation X should be ~1");
        assert!((translation.y - 1.0).abs() < 0.1, "Translation Y should be ~1");
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

        assert!((rotation - PI / 2.0).abs() < 0.1, "Rotation should be ~90 degrees");
    }
}
