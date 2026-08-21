from pathlib import Path

path = Path("rust_robotics_algo/src/slam/loop_closure.rs")
text = path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label} anchor changed: expected 1 match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    """        landmarks: &[super::Landmark2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n        let mut closures = Vec::new();\n\n        // Try proximity-based detection\n""",
    """        _landmarks: &[super::Landmark2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n        let mut closures = Vec::new();\n\n        // Try proximity-based detection\n""",
    "public detect signature",
)

replace_once(
    """        closures.extend(self.detect_proximity_based(\n            poses,\n            landmarks,\n            observation_constraints,\n            current_pose_idx,\n        ));\n""",
    """        closures.extend(self.detect_proximity_based(\n            poses,\n            observation_constraints,\n            current_pose_idx,\n        ));\n""",
    "proximity call",
)
replace_once(
    """            closures.extend(self.detect_landmark_based(\n                poses,\n                landmarks,\n                observation_constraints,\n                current_pose_idx,\n            ));\n""",
    """            closures.extend(self.detect_landmark_based(\n                poses,\n                observation_constraints,\n                current_pose_idx,\n            ));\n""",
    "landmark call",
)

replace_once(
    """    fn detect_proximity_based(\n        &self,\n        poses: &[super::Pose2D],\n        landmarks: &[super::Landmark2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n""",
    """    fn detect_proximity_based(\n        &self,\n        poses: &[super::Pose2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n""",
    "proximity signature",
)
replace_once(
    """    fn detect_landmark_based(\n        &self,\n        poses: &[super::Pose2D],\n        landmarks: &[super::Landmark2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n""",
    """    fn detect_landmark_based(\n        &self,\n        _poses: &[super::Pose2D],\n        observation_constraints: &[super::ObservationConstraint],\n        current_pose_idx: usize,\n    ) -> Vec<LoopClosure> {\n""",
    "landmark signature",
)

proximity_old = """            // Estimate transform using common landmarks\n            if let Some((transform, confidence)) = self.estimate_transform(\n                candidate_pose,\n                current_pose,\n                landmarks,\n                &common,\n                observation_constraints,\n                candidate_idx,\n                current_pose_idx,\n            ) {\n                // Validate with chi-squared test\n                let mahalanobis_sq =\n                    self.compute_mahalanobis(candidate_pose, current_pose, &transform);\n\n                if mahalanobis_sq <= self.config.max_mahalanobis_sq {\n                    closures.push(LoopClosure {\n                        from_pose_idx: candidate_idx,\n                        to_pose_idx: current_pose_idx,\n                        transform,\n                        confidence,\n                        num_matches: common.len(),\n                        mahalanobis_sq,\n                    });\n                }\n            }\n"""
proximity_new = """            // Estimate an independent relative transform directly from the\n            // two poses' matched range/bearing observations.\n            if let Some((transform, confidence, mahalanobis_sq)) = self.estimate_transform(\n                &common,\n                observation_constraints,\n                candidate_idx,\n                current_pose_idx,\n            ) {\n                if mahalanobis_sq <= self.config.max_mahalanobis_sq {\n                    closures.push(LoopClosure {\n                        from_pose_idx: candidate_idx,\n                        to_pose_idx: current_pose_idx,\n                        transform,\n                        confidence,\n                        num_matches: common.len(),\n                        mahalanobis_sq,\n                    });\n                }\n            }\n"""
replace_once(proximity_old, proximity_new, "proximity closure block")

landmark_old = """            let candidate_pose = &poses[last_pose_idx];\n            let current_pose = &poses[current_pose_idx];\n\n            // Estimate transform using common landmarks\n            if let Some((transform, confidence)) = self.estimate_transform(\n                candidate_pose,\n                current_pose,\n                landmarks,\n                &common,\n                observation_constraints,\n                last_pose_idx,\n                current_pose_idx,\n            ) {\n                // For landmark-based detection, use a higher Mahalanobis threshold\n                // since we expect the poses to have drifted\n                let mahalanobis_sq =\n                    self.compute_mahalanobis(candidate_pose, current_pose, &transform);\n\n                // Accept higher Mahalanobis for landmark-based (3x normal threshold)\n                if mahalanobis_sq <= self.config.max_mahalanobis_sq * 3.0 {\n                    closures.push(LoopClosure {\n                        from_pose_idx: last_pose_idx,\n                        to_pose_idx: current_pose_idx,\n                        transform,\n                        confidence: confidence * 0.8, // Slightly lower confidence for landmark-based\n                        num_matches: common.len(),\n                        mahalanobis_sq,\n                    });\n                }\n            }\n"""
landmark_new = """            // Landmark-based detection is deliberately independent of the\n            // drifted graph poses, so it uses the same observation-consistency\n            // gate rather than a relaxed pose-error threshold.\n            if let Some((transform, confidence, mahalanobis_sq)) = self.estimate_transform(\n                &common,\n                observation_constraints,\n                last_pose_idx,\n                current_pose_idx,\n            ) {\n                if mahalanobis_sq <= self.config.max_mahalanobis_sq {\n                    closures.push(LoopClosure {\n                        from_pose_idx: last_pose_idx,\n                        to_pose_idx: current_pose_idx,\n                        transform,\n                        confidence: confidence * 0.8,\n                        num_matches: common.len(),\n                        mahalanobis_sq,\n                    });\n                }\n            }\n"""
replace_once(landmark_old, landmark_new, "landmark closure block")

start = text.index("    /// Estimate relative transform from candidate pose to current pose using common landmarks\n")
end = text.index("    /// Compute rigid transform (rotation angle, translation) using SVD\n", start)
new_estimator = r'''    /// Estimate candidate-to-current relative pose directly from matched observations.
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

        let candidate_obs: HashMap<usize, &super::ObservationConstraint> =
            observation_constraints
                .iter()
                .filter(|c| {
                    c.pose_idx == candidate_idx && common_landmarks.contains(&c.landmark_idx)
                })
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
        let mahalanobis_sq = self.compute_alignment_mahalanobis(
            &matched_observations,
            rotation,
            &translation,
        )?;
        let confidence = 1.0 / (1.0 + residual);

        Some((
            Vector3::new(
                translation.x,
                translation.y,
                normalize_angle(rotation),
            ),
            confidence,
            mahalanobis_sq,
        ))
    }

'''
text = text[:start] + new_estimator + text[end:]

mahal_start = text.index("    /// Compute Mahalanobis distance for the proposed loop closure\n")
mahal_end = text.index("    /// Validate loop closures using chain consistency\n", mahal_start)
new_validation = r'''    fn observation_point(c: &super::ObservationConstraint) -> Option<Vector2<f32>> {
        let range = c.measurement[0];
        let bearing = c.measurement[1];
        if !range.is_finite() || !bearing.is_finite() || range <= 0.0 {
            return None;
        }
        Some(Vector2::new(
            range * bearing.cos(),
            range * bearing.sin(),
        ))
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

'''
text = text[:mahal_start] + new_validation + text[mahal_end:]

text = text.replace(
    "    /// Mahalanobis distance squared (lower is better fit)\n",
    "    /// Mean matched-observation Mahalanobis error (lower is better fit)\n",
    1,
)
text = text.replace(
    "    /// Maximum Mahalanobis distance squared for acceptance\n",
    "    /// Maximum mean matched-observation Mahalanobis error for acceptance\n",
    1,
)

test_anchor = """    #[test]\n    fn test_rigid_transform_estimation() {\n"""
if text.count(test_anchor) != 1:
    raise SystemExit(f"rigid-transform test anchor changed: {text.count(test_anchor)} matches")
new_tests = r'''    fn add_observations_from_pose(
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
        add_observations_from_pose(
            &mut graph,
            0,
            ground_truth_candidate,
            &landmarks,
            &obs_cov,
        );
        add_observations_from_pose(
            &mut graph,
            4,
            ground_truth_current,
            &landmarks,
            &obs_cov,
        );

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 0.1;
        config.min_temporal_separation = 3;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
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
            graph.add_observation(
                0,
                lm_idx,
                point.norm(),
                point.y.atan2(point.x),
                &obs_cov,
            );
        }
        for (lm_idx, point) in current_points.iter().enumerate() {
            graph.add_observation(
                4,
                lm_idx,
                point.norm(),
                point.y.atan2(point.x),
                &obs_cov,
            );
        }

        let mut config = LoopClosureConfig::default();
        config.proximity_threshold = 0.1;
        config.landmark_observation_gap = 3;
        config.min_common_landmarks = 3;
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

'''
text = text.replace(test_anchor, new_tests + test_anchor, 1)

path.write_text(text)
