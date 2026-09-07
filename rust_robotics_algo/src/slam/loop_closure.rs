//! Loop closure detection for Graph SLAM
//!
//! Loop closures are estimated directly from matched range/bearing observations so
//! that the measurement remains independent of drifted graph poses and landmark
//! estimates. Rejected/down-weighted observations are excluded, the rigid fit is
//! covariance weighted, and the final consistency gate uses the residual degrees
//! of freedom of the fitted 2D rigid transform.

use crate::slam::robust_kernels::chi_squared;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

/// A detected loop closure between two poses.
#[derive(Debug, Clone)]
pub struct LoopClosure {
    /// Index of the candidate (older) pose.
    pub from_pose_idx: usize,
    /// Index of the current (newer) pose.
    pub to_pose_idx: usize,
    /// Relative transform `(dx, dy, dtheta)` from candidate to current in the
    /// candidate frame.
    pub transform: Vector3<f32>,
    /// Confidence score (higher is better).
    pub confidence: f32,
    /// Number of matched usable landmarks.
    pub num_matches: usize,
    /// Total matched-observation chi-squared statistic after fitting the 2D rigid
    /// transform. Lower is better; acceptance accounts for `2N - 3` residual DOF.
    pub mahalanobis_sq: f32,
}

/// Configuration for loop closure detection.
#[derive(Debug, Clone)]
pub struct LoopClosureConfig {
    /// Maximum distance to consider a candidate pose (meters).
    pub proximity_threshold: f32,
    /// Minimum number of poses between current and candidate.
    pub min_temporal_separation: usize,
    /// Minimum number of common usable landmarks for matching.
    pub min_common_landmarks: usize,
    /// Chi-squared confidence level for observation-consistency validation.
    pub chi2_confidence: f64,
    /// Additional upper bound on reduced chi-squared (`chi2 / DOF`). This is kept
    /// for API compatibility; the DOF-aware chi-squared gate is the primary test.
    pub max_mahalanobis_sq: f32,
    /// Covariance for loop closure constraints added to the graph.
    pub closure_covariance: Matrix3<f32>,
    /// Enable landmark-based loop closure (works even with pose drift).
    pub enable_landmark_based: bool,
    /// Minimum gap in pose indices since last observation of a landmark to trigger
    /// landmark-based loop closure.
    pub landmark_observation_gap: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            proximity_threshold: 2.0,
            min_temporal_separation: 10,
            min_common_landmarks: 3,
            chi2_confidence: 0.95,
            // Deliberately loose as a secondary reduced-statistic safety cap. The
            // primary gate is chi_squared::is_outlier with the fitted residual DOF.
            max_mahalanobis_sq: 11.345,
            closure_covariance: Matrix3::from_diagonal(&Vector3::new(0.2, 0.2, 0.05)),
            enable_landmark_based: true,
            landmark_observation_gap: 15,
        }
    }
}

/// Loop-closure detector for graph SLAM.
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

    /// Creates a detector with caller-provided configuration.
    pub fn with_config(config: LoopClosureConfig) -> Self {
        Self { config }
    }

    /// Detects validated loop closures for `current_pose_idx`.
    pub fn detect(
        &self,
        poses: &[super::Pose2D],
        _landmarks: &[super::Landmark2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        let mut closures =
            self.detect_proximity_based(poses, observation_constraints, current_pose_idx);

        if self.config.enable_landmark_based {
            closures.extend(self.detect_landmark_based(observation_constraints, current_pose_idx));
        }

        closures.sort_by(|a, b| {
            a.from_pose_idx
                .cmp(&b.from_pose_idx)
                .then(a.to_pose_idx.cmp(&b.to_pose_idx))
        });
        closures
            .dedup_by(|a, b| a.from_pose_idx == b.from_pose_idx && a.to_pose_idx == b.to_pose_idx);
        closures.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        closures
    }

    fn detect_proximity_based(
        &self,
        poses: &[super::Pose2D],
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        if current_pose_idx >= poses.len() || current_pose_idx < self.config.min_temporal_separation
        {
            return Vec::new();
        }

        let current_pose = &poses[current_pose_idx];
        let pose_landmarks = self.build_pose_landmarks_map(observation_constraints);
        let current_landmarks = pose_landmarks
            .get(&current_pose_idx)
            .cloned()
            .unwrap_or_default();
        if current_landmarks.len() < self.config.min_common_landmarks {
            return Vec::new();
        }

        let max_candidate_idx =
            current_pose_idx.saturating_sub(self.config.min_temporal_separation);
        let mut closures = Vec::new();

        for candidate_idx in 0..=max_candidate_idx.min(poses.len().saturating_sub(1)) {
            let candidate_pose = &poses[candidate_idx];
            let dx = current_pose.x - candidate_pose.x;
            let dy = current_pose.y - candidate_pose.y;
            if (dx * dx + dy * dy).sqrt() > self.config.proximity_threshold {
                continue;
            }

            let candidate_landmarks = pose_landmarks
                .get(&candidate_idx)
                .cloned()
                .unwrap_or_default();
            let common: Vec<usize> = current_landmarks
                .intersection(&candidate_landmarks)
                .copied()
                .collect();
            if common.len() < self.config.min_common_landmarks {
                continue;
            }

            if let Some((transform, confidence, chi2, num_matches)) = self.estimate_transform(
                &common,
                observation_constraints,
                candidate_idx,
                current_pose_idx,
            ) {
                if self.alignment_is_acceptable(chi2, num_matches) {
                    closures.push(LoopClosure {
                        from_pose_idx: candidate_idx,
                        to_pose_idx: current_pose_idx,
                        transform,
                        confidence,
                        num_matches,
                        mahalanobis_sq: chi2,
                    });
                }
            }
        }

        closures
    }

    fn detect_landmark_based(
        &self,
        observation_constraints: &[super::ObservationConstraint],
        current_pose_idx: usize,
    ) -> Vec<LoopClosure> {
        if current_pose_idx < self.config.landmark_observation_gap {
            return Vec::new();
        }

        let mut landmark_poses: HashMap<usize, Vec<usize>> = HashMap::new();
        for c in observation_constraints
            .iter()
            .filter(|c| Self::observation_is_usable(c))
        {
            landmark_poses
                .entry(c.landmark_idx)
                .or_default()
                .push(c.pose_idx);
        }
        for observing_poses in landmark_poses.values_mut() {
            observing_poses.sort_unstable();
            observing_poses.dedup();
        }

        let current_landmarks: HashSet<usize> = observation_constraints
            .iter()
            .filter(|c| c.pose_idx == current_pose_idx && Self::observation_is_usable(c))
            .map(|c| c.landmark_idx)
            .collect();
        if current_landmarks.len() < self.config.min_common_landmarks {
            return Vec::new();
        }

        let mut closures = Vec::new();
        for &lm_idx in &current_landmarks {
            let Some(observing_poses) = landmark_poses.get(&lm_idx) else {
                continue;
            };
            let Some(last_pose_idx) = observing_poses
                .iter()
                .rev()
                .copied()
                .find(|pose_idx| *pose_idx < current_pose_idx)
            else {
                continue;
            };
            if current_pose_idx - last_pose_idx < self.config.landmark_observation_gap {
                continue;
            }

            let last_pose_landmarks: HashSet<usize> = observation_constraints
                .iter()
                .filter(|c| c.pose_idx == last_pose_idx && Self::observation_is_usable(c))
                .map(|c| c.landmark_idx)
                .collect();
            let common: Vec<usize> = current_landmarks
                .intersection(&last_pose_landmarks)
                .copied()
                .collect();
            if common.len() < self.config.min_common_landmarks {
                continue;
            }

            if let Some((transform, confidence, chi2, num_matches)) = self.estimate_transform(
                &common,
                observation_constraints,
                last_pose_idx,
                current_pose_idx,
            ) {
                if self.alignment_is_acceptable(chi2, num_matches) {
                    closures.push(LoopClosure {
                        from_pose_idx: last_pose_idx,
                        to_pose_idx: current_pose_idx,
                        transform,
                        confidence: confidence * 0.8,
                        num_matches,
                        mahalanobis_sq: chi2,
                    });
                }
            }
        }

        closures
    }

    fn build_pose_landmarks_map(
        &self,
        observation_constraints: &[super::ObservationConstraint],
    ) -> HashMap<usize, HashSet<usize>> {
        let mut map = HashMap::new();
        for c in observation_constraints
            .iter()
            .filter(|c| Self::observation_is_usable(c))
        {
            map.entry(c.pose_idx)
                .or_insert_with(HashSet::new)
                .insert(c.landmark_idx);
        }
        map
    }

    fn observation_is_usable(c: &super::ObservationConstraint) -> bool {
        !c.is_outlier
            && c.robust_weight.is_finite()
            && c.robust_weight > 0.0
            && c.measurement[0].is_finite()
            && c.measurement[1].is_finite()
            && c.measurement[0] > 0.0
    }

    /// Estimates candidate-to-current relative pose directly from matched observations.
    fn estimate_transform(
        &self,
        common_landmarks: &[usize],
        observation_constraints: &[super::ObservationConstraint],
        candidate_idx: usize,
        current_idx: usize,
    ) -> Option<(Vector3<f32>, f32, f32, usize)> {
        let common: HashSet<usize> = common_landmarks.iter().copied().collect();
        let candidate_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| {
                c.pose_idx == candidate_idx
                    && common.contains(&c.landmark_idx)
                    && Self::observation_is_usable(c)
            })
            .map(|c| (c.landmark_idx, c))
            .collect();
        let current_obs: HashMap<usize, &super::ObservationConstraint> = observation_constraints
            .iter()
            .filter(|c| {
                c.pose_idx == current_idx
                    && common.contains(&c.landmark_idx)
                    && Self::observation_is_usable(c)
            })
            .map(|c| (c.landmark_idx, c))
            .collect();

        let mut points_candidate = Vec::with_capacity(common_landmarks.len());
        let mut points_current = Vec::with_capacity(common_landmarks.len());
        let mut weights = Vec::with_capacity(common_landmarks.len());
        let mut matched_observations = Vec::with_capacity(common_landmarks.len());

        for &lm_idx in common_landmarks {
            let (Some(candidate), Some(current)) =
                (candidate_obs.get(&lm_idx), current_obs.get(&lm_idx))
            else {
                continue;
            };
            let candidate = *candidate;
            let current = *current;
            let weight = Self::observation_pair_weight(candidate, current)?;
            points_candidate.push(Self::observation_point(candidate)?);
            points_current.push(Self::observation_point(current)?);
            weights.push(weight);
            matched_observations.push((candidate, current));
        }

        let num_matches = matched_observations.len();
        if num_matches < self.config.min_common_landmarks.max(2) {
            return None;
        }

        // Source=current, target=candidate. Aligning current-frame landmark points
        // onto candidate-frame points yields the candidate->current robot transform.
        let (rotation, translation) =
            self.compute_weighted_rigid_transform(&points_current, &points_candidate, &weights)?;
        let residual = self.compute_weighted_alignment_residual(
            &points_current,
            &points_candidate,
            &weights,
            rotation,
            &translation,
        )?;
        let chi2 =
            self.compute_alignment_mahalanobis(&matched_observations, rotation, &translation)?;
        let confidence = 1.0 / (1.0 + residual);

        Some((
            Vector3::new(translation.x, translation.y, normalize_angle(rotation)),
            confidence,
            chi2,
            num_matches,
        ))
    }

    fn observation_point(c: &super::ObservationConstraint) -> Option<Vector2<f32>> {
        if !Self::observation_is_usable(c) {
            return None;
        }
        let range = c.measurement[0];
        let bearing = c.measurement[1];
        Some(Vector2::new(range * bearing.cos(), range * bearing.sin()))
    }

    fn observation_cartesian_covariance(c: &super::ObservationConstraint) -> Option<Matrix2<f32>> {
        if !Self::observation_is_usable(c) {
            return None;
        }
        let range = c.measurement[0];
        let bearing = c.measurement[1];
        let polar_covariance = c.information.try_inverse()?;
        let jacobian = Matrix2::new(
            bearing.cos(),
            -range * bearing.sin(),
            bearing.sin(),
            range * bearing.cos(),
        );
        let covariance = jacobian * polar_covariance * jacobian.transpose();
        covariance
            .iter()
            .all(|v| v.is_finite())
            .then_some(covariance)
    }

    /// Scalar precision used by weighted Procrustes. Trace is rotation invariant,
    /// so it gives a stable combined Cartesian variance before the rotation is known.
    fn observation_pair_weight(
        candidate: &super::ObservationConstraint,
        current: &super::ObservationConstraint,
    ) -> Option<f32> {
        let candidate_covariance = Self::observation_cartesian_covariance(candidate)?;
        let current_covariance = Self::observation_cartesian_covariance(current)?;
        let combined_variance = candidate_covariance.trace() + current_covariance.trace();
        let robust_weight = candidate.robust_weight.min(current.robust_weight) as f32;
        if !combined_variance.is_finite()
            || combined_variance <= f32::EPSILON
            || !robust_weight.is_finite()
            || robust_weight <= 0.0
        {
            return None;
        }
        Some(robust_weight / combined_variance)
    }

    fn compute_weighted_rigid_transform(
        &self,
        source: &[Vector2<f32>],
        target: &[Vector2<f32>],
        weights: &[f32],
    ) -> Option<(f32, Vector2<f32>)> {
        if source.len() < 2 || source.len() != target.len() || source.len() != weights.len() {
            return None;
        }

        let total_weight: f32 = weights.iter().copied().sum();
        if !total_weight.is_finite() || total_weight <= f32::EPSILON {
            return None;
        }

        let mut source_centroid = Vector2::zeros();
        let mut target_centroid = Vector2::zeros();
        for ((source_point, target_point), weight) in source
            .iter()
            .zip(target.iter())
            .zip(weights.iter().copied())
        {
            if !weight.is_finite() || weight <= 0.0 {
                return None;
            }
            source_centroid += source_point * weight;
            target_centroid += target_point * weight;
        }
        source_centroid /= total_weight;
        target_centroid /= total_weight;

        let mut h = Matrix2::zeros();
        for ((source_point, target_point), weight) in source
            .iter()
            .zip(target.iter())
            .zip(weights.iter().copied())
        {
            let source_centered = source_point - source_centroid;
            let target_centered = target_point - target_centroid;
            h += weight * source_centered * target_centered.transpose();
        }

        let svd = h.svd(true, true);
        let u = svd.u?;
        let v_t = svd.v_t?;
        let mut r = v_t.transpose() * u.transpose();
        if r.determinant() < 0.0 {
            let mut v_t_corrected = v_t;
            v_t_corrected[(1, 0)] *= -1.0;
            v_t_corrected[(1, 1)] *= -1.0;
            r = v_t_corrected.transpose() * u.transpose();
        }

        let rotation = r[(1, 0)].atan2(r[(0, 0)]);
        let translation = target_centroid - r * source_centroid;
        if rotation.is_finite() && translation.iter().all(|v| v.is_finite()) {
            Some((rotation, translation))
        } else {
            None
        }
    }

    fn compute_weighted_alignment_residual(
        &self,
        source: &[Vector2<f32>],
        target: &[Vector2<f32>],
        weights: &[f32],
        rotation: f32,
        translation: &Vector2<f32>,
    ) -> Option<f32> {
        if source.is_empty() || source.len() != target.len() || source.len() != weights.len() {
            return None;
        }
        let rotation_matrix = Matrix2::new(
            rotation.cos(),
            -rotation.sin(),
            rotation.sin(),
            rotation.cos(),
        );
        let total_weight: f32 = weights.iter().copied().sum();
        if !total_weight.is_finite() || total_weight <= f32::EPSILON {
            return None;
        }
        let weighted_sum = source
            .iter()
            .zip(target.iter())
            .zip(weights.iter().copied())
            .map(|((source_point, target_point), weight)| {
                let error = rotation_matrix * source_point + *translation - target_point;
                weight * error.norm_squared()
            })
            .sum::<f32>();
        Some((weighted_sum / total_weight).sqrt())
    }

    fn compute_alignment_mahalanobis(
        &self,
        matches: &[(&super::ObservationConstraint, &super::ObservationConstraint)],
        rotation: f32,
        translation: &Vector2<f32>,
    ) -> Option<f32> {
        if matches.len() < 2 {
            return None;
        }

        let rotation_matrix = Matrix2::new(
            rotation.cos(),
            -rotation.sin(),
            rotation.sin(),
            rotation.cos(),
        );
        let mut total = 0.0;
        for &(candidate, current) in matches {
            let candidate_point = Self::observation_point(candidate)?;
            let current_point = Self::observation_point(current)?;
            let error = rotation_matrix * current_point + *translation - candidate_point;
            let candidate_covariance = Self::observation_cartesian_covariance(candidate)?;
            let current_covariance = Self::observation_cartesian_covariance(current)?;
            let residual_covariance = candidate_covariance
                + rotation_matrix * current_covariance * rotation_matrix.transpose();
            let information = residual_covariance.try_inverse()?;
            let contribution = error.dot(&(information * error));
            if !contribution.is_finite() || contribution < 0.0 {
                return None;
            }
            total += contribution;
        }
        total.is_finite().then_some(total)
    }

    fn alignment_degrees_of_freedom(num_matches: usize) -> usize {
        // Each match contributes a 2D residual; fitting SE(2) consumes 3 parameters.
        num_matches.saturating_mul(2).saturating_sub(3).max(1)
    }

    fn alignment_is_acceptable(&self, chi2: f32, num_matches: usize) -> bool {
        if !chi2.is_finite() || num_matches < self.config.min_common_landmarks {
            return false;
        }
        let dof = Self::alignment_degrees_of_freedom(num_matches);
        let reduced_chi2 = chi2 / dof as f32;
        reduced_chi2 <= self.config.max_mahalanobis_sq
            && !chi_squared::is_outlier(chi2 as f64, dof, self.config.chi2_confidence)
    }

    /// Validates loop closures using chain consistency. Individual closures have
    /// already passed observation consistency; transitive chain checks are a future
    /// extension.
    pub fn validate_chain_consistency(
        &self,
        closures: &[LoopClosure],
        _poses: &[super::Pose2D],
    ) -> Vec<LoopClosure> {
        closures.to_vec()
    }
}

fn normalize_angle(angle: f32) -> f32 {
    let mut angle = angle;
    while angle > PI {
        angle -= 2.0 * PI;
    }
    while angle < -PI {
        angle += 2.0 * PI;
    }
    angle
}

/// Adds a detected loop closure to the graph as an odometry-style constraint.
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

    fn add_observations_from_pose(
        graph: &mut GraphSlam,
        pose_idx: usize,
        ground_truth_pose: Pose2D,
        landmarks: &[Landmark2D],
        covariance: &Matrix2<f32>,
    ) {
        for (landmark_idx, landmark) in landmarks.iter().enumerate() {
            let dx = landmark.x - ground_truth_pose.x;
            let dy = landmark.y - ground_truth_pose.y;
            let range = (dx * dx + dy * dy).sqrt();
            let bearing = normalize_angle(dy.atan2(dx) - ground_truth_pose.theta);
            graph.add_observation(pose_idx, landmark_idx, range, bearing, covariance);
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

    fn five_pose_graph() -> GraphSlam {
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
        graph
    }

    #[test]
    fn test_loop_closure_detection_square() {
        let mut graph = GraphSlam::new();
        let poses = vec![
            Pose2D::new(0.0, 0.0, 0.0),
            Pose2D::new(5.0, 0.0, PI / 2.0),
            Pose2D::new(5.0, 5.0, PI),
            Pose2D::new(0.0, 5.0, -PI / 2.0),
            Pose2D::new(0.2, 0.3, 0.1),
        ];
        for pose in &poses {
            graph.add_pose(*pose);
        }
        let landmarks = vec![
            Landmark2D::new(2.0, 2.0),
            Landmark2D::new(3.0, 2.0),
            Landmark2D::new(3.0, 3.0),
            Landmark2D::new(2.0, 3.0),
        ];
        for landmark in &landmarks {
            graph.add_landmark(*landmark);
        }
        let covariance = Matrix2::from_diagonal(&Vector2::new(0.1, 0.01));
        add_observations_from_pose(&mut graph, 0, poses[0], &landmarks, &covariance);
        add_observations_from_pose(&mut graph, 4, poses[4], &landmarks, &covariance);

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 1.0;
        config.min_temporal_separation = 3;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
        let closures = LoopClosureDetector::with_config(config).detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4,
        );
        assert!(closures
            .iter()
            .any(|closure| closure.from_pose_idx == 0 && closure.to_pose_idx == 4));
    }

    #[test]
    fn test_landmark_loop_closure_recovers_ground_truth_despite_pose_drift() {
        let mut graph = GraphSlam::new();
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
        let covariance = Matrix2::from_diagonal(&Vector2::new(0.01, 0.001));
        add_observations_from_pose(
            &mut graph,
            0,
            ground_truth_candidate,
            &landmarks,
            &covariance,
        );
        add_observations_from_pose(&mut graph, 4, ground_truth_current, &landmarks, &covariance);

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 0.1;
        config.min_temporal_separation = 3;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
        let closures = LoopClosureDetector::with_config(config).detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4,
        );
        let closure = closures
            .iter()
            .find(|closure| closure.from_pose_idx == 0 && closure.to_pose_idx == 4)
            .expect("observation geometry should recover the revisit despite pose drift");
        let expected = relative_pose(ground_truth_candidate, ground_truth_current);
        assert!((closure.transform.x - expected.x).abs() < 1e-3);
        assert!((closure.transform.y - expected.y).abs() < 1e-3);
        assert!(normalize_angle(closure.transform.z - expected.z).abs() < 1e-3);
        assert!((closure.transform - relative_pose(graph.poses[0], graph.poses[4])).norm() > 1.0);
    }

    #[test]
    fn test_inconsistent_landmark_geometry_is_rejected() {
        let mut graph = five_pose_graph();
        for landmark in [
            Landmark2D::new(2.0, 0.0),
            Landmark2D::new(0.0, 2.0),
            Landmark2D::new(-2.0, 0.0),
            Landmark2D::new(0.0, -2.0),
        ] {
            graph.add_landmark(landmark);
        }
        let covariance = Matrix2::from_diagonal(&Vector2::new(0.001, 0.0001));
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
        for (landmark_idx, point) in candidate_points.iter().enumerate() {
            graph.add_observation(
                0,
                landmark_idx,
                point.norm(),
                point.y.atan2(point.x),
                &covariance,
            );
        }
        for (landmark_idx, point) in current_points.iter().enumerate() {
            graph.add_observation(
                4,
                landmark_idx,
                point.norm(),
                point.y.atan2(point.x),
                &covariance,
            );
        }
        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 0.1;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
        assert!(LoopClosureDetector::with_config(config)
            .detect(
                &graph.poses,
                &graph.landmarks,
                &graph.observation_constraints,
                4,
            )
            .is_empty());
    }

    #[test]
    fn rejected_observations_do_not_participate_in_loop_closure() {
        let mut graph = five_pose_graph();
        let landmarks = vec![
            Landmark2D::new(2.0, 1.0),
            Landmark2D::new(2.0, -1.0),
            Landmark2D::new(-2.0, 1.0),
        ];
        for landmark in &landmarks {
            graph.add_landmark(*landmark);
        }
        let covariance = Matrix2::from_diagonal(&Vector2::new(0.01, 0.001));
        add_observations_from_pose(&mut graph, 0, Pose2D::origin(), &landmarks, &covariance);
        add_observations_from_pose(&mut graph, 4, Pose2D::origin(), &landmarks, &covariance);
        graph
            .observation_constraints
            .iter_mut()
            .find(|constraint| constraint.pose_idx == 4 && constraint.landmark_idx == 2)
            .expect("fixture observation")
            .is_outlier = true;

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 20.0;
        config.min_temporal_separation = 3;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
        let closures = LoopClosureDetector::with_config(config).detect(
            &graph.poses,
            &graph.landmarks,
            &graph.observation_constraints,
            4,
        );
        assert!(
            closures.is_empty(),
            "flagged observations must not satisfy the common-landmark minimum"
        );
    }

    #[test]
    fn covariance_weighted_fit_resists_noisy_landmark() {
        let mut graph = five_pose_graph();
        let landmarks = vec![
            Landmark2D::new(3.0, 0.0),
            Landmark2D::new(0.0, 3.0),
            Landmark2D::new(-3.0, 0.0),
            Landmark2D::new(0.0, -3.0),
        ];
        for landmark in &landmarks {
            graph.add_landmark(*landmark);
        }
        let precise = Matrix2::from_diagonal(&Vector2::new(1e-4, 1e-5));
        let noisy = Matrix2::from_diagonal(&Vector2::new(100.0, 10.0));
        let candidate = Pose2D::origin();
        let current = Pose2D::new(1.0, 0.0, 0.0);
        add_observations_from_pose(&mut graph, 0, candidate, &landmarks, &precise);
        for (landmark_idx, landmark) in landmarks.iter().enumerate() {
            if landmark_idx == 3 {
                let corrupted_point: Vector2<f32> = Vector2::new(15.0, 12.0);
                graph.add_observation(
                    4,
                    landmark_idx,
                    corrupted_point.norm(),
                    corrupted_point.y.atan2(corrupted_point.x),
                    &noisy,
                );
            } else {
                let dx = landmark.x - current.x;
                let dy = landmark.y - current.y;
                graph.add_observation(
                    4,
                    landmark_idx,
                    (dx * dx + dy * dy).sqrt(),
                    dy.atan2(dx),
                    &precise,
                );
            }
        }

        let detector = LoopClosureDetector::new();
        let common = vec![0, 1, 2, 3];
        let (transform, _, _, _) = detector
            .estimate_transform(&common, &graph.observation_constraints, 0, 4)
            .expect("weighted fit should remain solvable");
        let expected = relative_pose(candidate, current);
        assert!((transform.x - expected.x).abs() < 0.05);
        assert!((transform.y - expected.y).abs() < 0.05);
        assert!(normalize_angle(transform.z - expected.z).abs() < 0.02);
    }

    #[test]
    fn chi_square_gate_uses_fitted_residual_degrees_of_freedom() {
        let mut config = LoopClosureConfig::default();
        config.chi2_confidence = 0.99;
        let detector = LoopClosureDetector::with_config(config);
        assert_eq!(LoopClosureDetector::alignment_degrees_of_freedom(3), 3);
        assert!(
            !detector.alignment_is_acceptable(20.0, 3),
            "three matches leave 3 residual DOF, so chi2=20 must exceed the 99% gate"
        );
        assert!(detector.alignment_is_acceptable(1.0, 3));
    }

    #[test]
    fn test_rigid_transform_estimation() {
        let detector = LoopClosureDetector::new();
        let source = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        ];
        let target = vec![
            Vector2::new(1.0, 1.0),
            Vector2::<f32>::new(2.0, 1.0),
            Vector2::new(2.0, 2.0),
        ];
        let weights = vec![1.0; source.len()];
        let (rotation, translation) = detector
            .compute_weighted_rigid_transform(&source, &target, &weights)
            .expect("rigid transform");
        assert!(rotation.abs() < 0.1);
        assert!((translation.x - 1.0).abs() < 0.1);
        assert!((translation.y - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rigid_transform_with_rotation() {
        let detector = LoopClosureDetector::new();
        let source = vec![
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::<f32>::new(2.0, 1.0),
        ];
        let target = vec![
            Vector2::new(0.0, 1.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(-1.0, 2.0),
        ];
        let weights = vec![1.0; source.len()];
        let (rotation, _) = detector
            .compute_weighted_rigid_transform(&source, &target, &weights)
            .expect("rigid transform");
        assert!((rotation - PI / 2.0).abs() < 0.1);
    }
}
