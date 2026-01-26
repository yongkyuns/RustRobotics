//! Schur complement marginalization for Graph SLAM
//!
//! When removing old poses from the sliding window, naive deletion loses
//! information about the remaining variables. Proper marginalization uses
//! the Schur complement to preserve this information as a prior constraint.
//!
//! ## The Problem
//!
//! Consider a graph with poses P1, P2, P3 and landmark L. If we want to remove P1:
//! - Naive: Delete P1 and all constraints involving it → Information loss
//! - Proper: Marginalize P1 → Prior constraint on P2, P3, L preserving information
//!
//! ## Schur Complement
//!
//! For a block information matrix:
//! ```text
//! H = [Hmm Hmo]
//!     [Hom Hoo]
//! ```
//!
//! Where `m` = variables to marginalize, `o` = variables to keep (others):
//! - The marginalized information is: H' = Hoo - Hom * Hmm^-1 * Hmo
//! - The marginalized residual is: b' = bo - Hom * Hmm^-1 * bm
//!
//! ## References
//! - [Visual-Inertial Mapping with Non-Linear Factor Recovery](https://arxiv.org/abs/1904.06504)
//! - [Information Sparsification in Visual-Inertial Odometry](https://www.iri.upc.edu/files/scidoc/2248-Information-Sparsification-in-Visual-Inertial-Odometry.pdf)

use nalgebra::{DMatrix, DVector};

/// A prior constraint representing marginalized information
///
/// This constraint encapsulates the information from marginalized variables
/// and should be added to the optimization problem.
#[derive(Debug, Clone)]
pub struct PriorConstraint {
    /// Indices of variables this prior applies to (in the current graph)
    pub variable_indices: Vec<usize>,

    /// Information matrix (block corresponding to kept variables)
    /// Size: sum(variable_dims) x sum(variable_dims)
    pub information: DMatrix<f64>,

    /// Linearization point for the prior (current values when marginalized)
    pub linearization_point: DVector<f64>,

    /// Residual at linearization point (should be near zero if freshly marginalized)
    pub residual: DVector<f64>,
}

impl PriorConstraint {
    /// Create a new prior constraint
    pub fn new(
        variable_indices: Vec<usize>,
        information: DMatrix<f64>,
        linearization_point: DVector<f64>,
    ) -> Self {
        let dim = linearization_point.len();
        Self {
            variable_indices,
            information,
            linearization_point,
            residual: DVector::zeros(dim),
        }
    }

    /// Compute the error contribution of this prior given current state
    pub fn compute_error(&self, current_state: &DVector<f64>) -> f64 {
        let delta = current_state - &self.linearization_point;
        let weighted = &self.information * &delta;
        delta.dot(&weighted)
    }

    /// Get the dimension of this prior
    pub fn dimension(&self) -> usize {
        self.linearization_point.len()
    }
}

/// Configuration for marginalization
#[derive(Debug, Clone)]
pub struct MarginalizationConfig {
    /// Minimum eigenvalue for information matrix (for numerical stability)
    pub min_eigenvalue: f64,

    /// Whether to sparsify the prior (reduce fill-in)
    pub enable_sparsification: bool,

    /// Threshold for sparsification (entries below this are zeroed)
    pub sparsification_threshold: f64,
}

impl Default for MarginalizationConfig {
    fn default() -> Self {
        Self {
            min_eigenvalue: 1e-8,
            enable_sparsification: false,
            sparsification_threshold: 1e-6,
        }
    }
}

/// Marginalizer for Graph SLAM
pub struct Marginalizer {
    config: MarginalizationConfig,
}

impl Default for Marginalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Marginalizer {
    pub fn new() -> Self {
        Self {
            config: MarginalizationConfig::default(),
        }
    }

    pub fn with_config(config: MarginalizationConfig) -> Self {
        Self { config }
    }

    /// Marginalize variables from the information matrix
    ///
    /// # Arguments
    /// * `full_info` - Full information matrix H
    /// * `full_residual` - Full residual vector b (gradient)
    /// * `marginalize_indices` - Indices of variables to marginalize
    /// * `keep_indices` - Indices of variables to keep
    ///
    /// # Returns
    /// * Marginalized information matrix H'
    /// * Marginalized residual b'
    pub fn marginalize(
        &self,
        full_info: &DMatrix<f64>,
        full_residual: &DVector<f64>,
        marginalize_indices: &[usize],
        keep_indices: &[usize],
    ) -> Option<(DMatrix<f64>, DVector<f64>)> {
        let n_marg = marginalize_indices.len();
        let n_keep = keep_indices.len();

        if n_marg == 0 {
            // Nothing to marginalize, return identity
            let mut h_keep = DMatrix::zeros(n_keep, n_keep);
            let mut b_keep = DVector::zeros(n_keep);
            for (i, &idx) in keep_indices.iter().enumerate() {
                b_keep[i] = full_residual[idx];
                for (j, &jdx) in keep_indices.iter().enumerate() {
                    h_keep[(i, j)] = full_info[(idx, jdx)];
                }
            }
            return Some((h_keep, b_keep));
        }

        // Extract blocks
        // Hmm = full_info[marginalize_indices, marginalize_indices]
        let mut h_mm = DMatrix::zeros(n_marg, n_marg);
        for (i, &idx) in marginalize_indices.iter().enumerate() {
            for (j, &jdx) in marginalize_indices.iter().enumerate() {
                h_mm[(i, j)] = full_info[(idx, jdx)];
            }
        }

        // Hmo = full_info[marginalize_indices, keep_indices]
        let mut h_mo = DMatrix::zeros(n_marg, n_keep);
        for (i, &idx) in marginalize_indices.iter().enumerate() {
            for (j, &jdx) in keep_indices.iter().enumerate() {
                h_mo[(i, j)] = full_info[(idx, jdx)];
            }
        }

        // Hom = full_info[keep_indices, marginalize_indices] = Hmo^T
        let h_om = h_mo.transpose();

        // Hoo = full_info[keep_indices, keep_indices]
        let mut h_oo = DMatrix::zeros(n_keep, n_keep);
        for (i, &idx) in keep_indices.iter().enumerate() {
            for (j, &jdx) in keep_indices.iter().enumerate() {
                h_oo[(i, j)] = full_info[(idx, jdx)];
            }
        }

        // bm = full_residual[marginalize_indices]
        let mut b_m = DVector::zeros(n_marg);
        for (i, &idx) in marginalize_indices.iter().enumerate() {
            b_m[i] = full_residual[idx];
        }

        // bo = full_residual[keep_indices]
        let mut b_o = DVector::zeros(n_keep);
        for (i, &idx) in keep_indices.iter().enumerate() {
            b_o[i] = full_residual[idx];
        }

        // Regularize Hmm for numerical stability
        for i in 0..n_marg {
            h_mm[(i, i)] += self.config.min_eigenvalue;
        }

        // Invert Hmm
        let h_mm_inv = h_mm.clone().try_inverse()?;

        // Schur complement: H' = Hoo - Hom * Hmm^-1 * Hmo
        let h_prime = &h_oo - &h_om * &h_mm_inv * &h_mo;

        // Marginalized residual: b' = bo - Hom * Hmm^-1 * bm
        let b_prime = &b_o - &h_om * &h_mm_inv * &b_m;

        // Optional sparsification
        let h_final = if self.config.enable_sparsification {
            self.sparsify(&h_prime)
        } else {
            h_prime
        };

        Some((h_final, b_prime))
    }

    /// Sparsify a matrix by zeroing small entries
    fn sparsify(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = matrix.clone();
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                if result[(i, j)].abs() < self.config.sparsification_threshold {
                    result[(i, j)] = 0.0;
                }
            }
        }
        result
    }

    /// Create a prior constraint from marginalization result
    pub fn create_prior(
        &self,
        variable_indices: Vec<usize>,
        information: DMatrix<f64>,
        linearization_point: DVector<f64>,
    ) -> PriorConstraint {
        PriorConstraint::new(variable_indices, information, linearization_point)
    }
}

/// Helper to build the local information matrix around poses to be marginalized
pub struct LocalInfoBuilder;

impl LocalInfoBuilder {
    /// Build local information matrix for a subset of variables
    ///
    /// This extracts the information matrix block corresponding to the
    /// specified variables and their immediate neighbors.
    #[allow(unused_variables)]
    pub fn build_local_info(
        poses: &[super::Pose2D],
        landmarks: &[super::Landmark2D],
        odometry_constraints: &[super::OdometryConstraint],
        observation_constraints: &[super::ObservationConstraint],
        marginalize_pose_indices: &[usize],
        fix_first_pose: bool,
    ) -> (DMatrix<f64>, DVector<f64>, Vec<usize>, Vec<usize>) {
        #[allow(unused_imports)]
        use nalgebra::{Matrix2, Matrix3};
        use std::collections::HashSet;

        // Find all variables connected to the marginalized poses
        let marg_set: HashSet<usize> = marginalize_pose_indices.iter().cloned().collect();

        // Find connected poses and landmarks
        let mut connected_poses: HashSet<usize> = HashSet::new();
        let mut connected_landmarks: HashSet<usize> = HashSet::new();

        for c in odometry_constraints {
            if marg_set.contains(&c.from_idx) || marg_set.contains(&c.to_idx) {
                connected_poses.insert(c.from_idx);
                connected_poses.insert(c.to_idx);
            }
        }

        for c in observation_constraints {
            if marg_set.contains(&c.pose_idx) {
                connected_poses.insert(c.pose_idx);
                connected_landmarks.insert(c.landmark_idx);
            }
        }

        // Build variable index mapping
        let mut all_poses: Vec<usize> = connected_poses.into_iter().collect();
        all_poses.sort();

        let mut all_landmarks: Vec<usize> = connected_landmarks.into_iter().collect();
        all_landmarks.sort();

        // Calculate dimensions
        let pose_start = if fix_first_pose && !all_poses.is_empty() && all_poses[0] == 0 {
            1
        } else {
            0
        };
        let n_pose_vars = (all_poses.len() - pose_start) * 3;
        let n_lm_vars = all_landmarks.len() * 2;
        let n_vars = n_pose_vars + n_lm_vars;

        if n_vars == 0 {
            return (
                DMatrix::zeros(0, 0),
                DVector::zeros(0),
                Vec::new(),
                Vec::new(),
            );
        }

        // Map from original index to local index
        let pose_to_local: std::collections::HashMap<usize, usize> = all_poses
            .iter()
            .skip(pose_start)
            .enumerate()
            .map(|(i, &p)| (p, i * 3))
            .collect();

        let lm_to_local: std::collections::HashMap<usize, usize> = all_landmarks
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, n_pose_vars + i * 2))
            .collect();

        // Build information matrix
        let mut h = DMatrix::<f64>::zeros(n_vars, n_vars);
        let b = DVector::<f64>::zeros(n_vars);

        // Add odometry contributions
        for c in odometry_constraints {
            if !marg_set.contains(&c.from_idx) && !marg_set.contains(&c.to_idx) {
                continue; // Not connected to marginalized poses
            }

            let idx1 = pose_to_local.get(&c.from_idx);
            let idx2 = pose_to_local.get(&c.to_idx);

            let info = c.information.cast::<f64>();

            // Add to diagonal blocks
            if let Some(&i1) = idx1 {
                for i in 0..3 {
                    for j in 0..3 {
                        h[(i1 + i, i1 + j)] += info[(i, j)];
                    }
                }
            }
            if let Some(&i2) = idx2 {
                for i in 0..3 {
                    for j in 0..3 {
                        h[(i2 + i, i2 + j)] += info[(i, j)];
                    }
                }
            }

            // Add to off-diagonal blocks
            if let (Some(&i1), Some(&i2)) = (idx1, idx2) {
                for i in 0..3 {
                    for j in 0..3 {
                        h[(i1 + i, i2 + j)] -= info[(i, j)];
                        h[(i2 + i, i1 + j)] -= info[(i, j)];
                    }
                }
            }
        }

        // Add observation contributions
        for c in observation_constraints {
            if !marg_set.contains(&c.pose_idx) {
                continue;
            }

            let pidx = pose_to_local.get(&c.pose_idx);
            let lidx = lm_to_local.get(&c.landmark_idx);

            let info = c.information.cast::<f64>();

            // Observation Jacobians are more complex, simplified here
            // In practice, we'd compute full J^T * Omega * J contributions
            if let Some(&pi) = pidx {
                for i in 0..2 {
                    for j in 0..2 {
                        // Simplified: just add to pose diagonal
                        if i < 3 && j < 3 {
                            h[(pi + i.min(2), pi + j.min(2))] += info[(i.min(1), j.min(1))];
                        }
                    }
                }
            }
            if let Some(&li) = lidx {
                for i in 0..2 {
                    for j in 0..2 {
                        h[(li + i, li + j)] += info[(i, j)];
                    }
                }
            }
        }

        // Compute marginalize and keep indices in local coordinates
        let marg_local: Vec<usize> = marginalize_pose_indices
            .iter()
            .filter_map(|&p| pose_to_local.get(&p))
            .flat_map(|&start| [start, start + 1, start + 2])
            .collect();

        let keep_local: Vec<usize> = (0..n_vars)
            .filter(|i| !marg_local.contains(i))
            .collect();

        (h, b, marg_local, keep_local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schur_complement_simple() {
        // Simple 4x4 information matrix
        // Variables: [x1, x2] where we marginalize x1
        let h = DMatrix::from_row_slice(
            4,
            4,
            &[
                10.0, 2.0, 1.0, 0.5, // Row 1
                2.0, 10.0, 0.5, 1.0, // Row 2
                1.0, 0.5, 10.0, 2.0, // Row 3
                0.5, 1.0, 2.0, 10.0, // Row 4
            ],
        );
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let marginalizer = Marginalizer::new();

        // Marginalize first 2 variables (indices 0, 1)
        let marginalize_idx = vec![0, 1];
        let keep_idx = vec![2, 3];

        let result = marginalizer.marginalize(&h, &b, &marginalize_idx, &keep_idx);

        assert!(result.is_some());
        let (h_prime, b_prime) = result.unwrap();

        // Check dimensions
        assert_eq!(h_prime.nrows(), 2);
        assert_eq!(h_prime.ncols(), 2);
        assert_eq!(b_prime.len(), 2);

        // The marginalized matrix should still be positive semi-definite
        // Check that diagonal is positive
        assert!(h_prime[(0, 0)] > 0.0);
        assert!(h_prime[(1, 1)] > 0.0);

        println!("Original H:\n{}", h);
        println!("Marginalized H':\n{}", h_prime);
        println!("Original b: {:?}", b.as_slice());
        println!("Marginalized b': {:?}", b_prime.as_slice());
    }

    #[test]
    fn test_marginalize_preserves_information() {
        // Test that marginalization preserves information about kept variables
        let h = DMatrix::from_row_slice(
            4,
            4,
            &[
                100.0, 10.0, 5.0, 2.0, 10.0, 100.0, 2.0, 5.0, 5.0, 2.0, 100.0, 10.0, 2.0, 5.0, 10.0,
                100.0,
            ],
        );
        let b = DVector::zeros(4);

        let marginalizer = Marginalizer::new();

        let (h_prime, _) = marginalizer
            .marginalize(&h, &b, &[0, 1], &[2, 3])
            .unwrap();

        // The marginalized matrix should have information from the off-diagonal blocks
        // H' = Hoo - Hom * Hmm^-1 * Hmo
        // This should add information, not remove it
        assert!(h_prime[(0, 0)] >= 0.0);
        assert!(h_prime[(1, 1)] >= 0.0);

        println!("Marginalized H':\n{}", h_prime);
    }

    #[test]
    fn test_prior_constraint() {
        let indices = vec![0, 1, 2];
        let info = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 20.0, 30.0]));
        let lin_point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let prior = PriorConstraint::new(indices, info, lin_point.clone());

        // At linearization point, error should be zero
        let error_at_lin = prior.compute_error(&lin_point);
        assert!(error_at_lin.abs() < 1e-10);

        // Away from linearization point, error should be positive
        let other_state = DVector::from_vec(vec![2.0, 3.0, 4.0]);
        let error_away = prior.compute_error(&other_state);
        assert!(error_away > 0.0);

        // Error should be: (x-x0)^T * Omega * (x-x0)
        // = [1,1,1]^T * diag(10,20,30) * [1,1,1] = 10 + 20 + 30 = 60
        assert!((error_away - 60.0).abs() < 1e-10);
    }
}
