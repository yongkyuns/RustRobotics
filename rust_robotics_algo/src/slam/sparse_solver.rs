//! Sparse linear algebra for Graph SLAM optimization.
//!
//! This module keeps Jacobian construction and normal-matrix assembly sparse.
//! The current solve step deliberately materializes the regularized normal
//! matrix as a dense matrix and uses dense LU. That is useful as a reference
//! implementation, but it is not yet a fully sparse factorization path.
//!
//! For a graph with `n` poses and `m` landmarks, the Jacobian and normal matrix
//! are structurally sparse because each factor touches only a small number of
//! variables. A future sparse backend can exploit that structure through the
//! factorization step as well.
//!
//! ## References
//! - [Efficient Sparse Pose Adjustment](https://www.cs.jhu.edu/~misha/Fall07/konolige10.pdf)
//! - [g2o: A General Framework for Graph Optimization](https://github.com/RainerKuemmerle/g2o)

use nalgebra::{DVector, Matrix2, Matrix3, Vector2, Vector3};
use sprs::{CsMat, CsVec, TriMat};
use std::f32::consts::PI;

/// Normalize angle to [-π, π].
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

/// Sparse triplet for building CSR/CSC matrices.
#[derive(Debug, Clone)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

/// Configuration for the normal-equation solve.
#[derive(Debug, Clone)]
pub struct SparseSolverConfig {
    /// Additional diagonal regularization applied before the dense reference solve.
    pub regularization: f64,
}

impl Default for SparseSolverConfig {
    fn default() -> Self {
        Self {
            regularization: 1e-8,
        }
    }
}

/// Sparse linear-system builder for graph SLAM.
///
/// Jacobians and `J^T J` are assembled sparsely. The final regularized normal
/// matrix is currently converted to dense form and solved with LU.
pub struct SparseSlamSolver {
    config: SparseSolverConfig,
}

impl Default for SparseSlamSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSlamSolver {
    /// Creates a solver with default settings.
    pub fn new() -> Self {
        Self {
            config: SparseSolverConfig::default(),
        }
    }

    /// Creates a solver with caller-provided configuration.
    pub fn with_config(config: SparseSolverConfig) -> Self {
        Self { config }
    }

    /// Build sparse Jacobian and residual vector from constraints.
    ///
    /// Returns `(J, r)`, with `J` in CSR format. Information matrices weight
    /// residuals and Jacobians using a square-root factor `W` satisfying
    /// `W^T W = information`.
    pub fn build_sparse_system(
        &self,
        poses: &[super::Pose2D],
        landmarks: &[super::Landmark2D],
        odometry_constraints: &[super::OdometryConstraint],
        observation_constraints: &[super::ObservationConstraint],
        fix_first_pose: bool,
    ) -> (CsMat<f64>, DVector<f64>) {
        let n_poses = poses.len();
        let n_landmarks = landmarks.len();
        let pose_start = if fix_first_pose && n_poses > 0 { 1 } else { 0 };
        let n_pose_vars = (n_poses - pose_start) * 3;
        let n_vars = n_pose_vars + n_landmarks * 2;
        let n_residuals = odometry_constraints.len() * 3 + observation_constraints.len() * 2;

        if n_vars == 0 || n_residuals == 0 {
            return (CsMat::empty(sprs::CSR, 0), DVector::zeros(0));
        }

        let nnz_estimate = odometry_constraints.len() * 18 + observation_constraints.len() * 10;
        let mut triplets = Vec::with_capacity(nnz_estimate);
        let mut residuals = DVector::zeros(n_residuals);
        let mut res_idx = 0;

        let pose_state_idx = |pose_idx: usize| -> Option<usize> {
            if fix_first_pose {
                if pose_idx == 0 {
                    None
                } else {
                    Some((pose_idx - 1) * 3)
                }
            } else {
                Some(pose_idx * 3)
            }
        };
        let landmark_state_idx = |landmark_idx: usize| -> usize { n_pose_vars + landmark_idx * 2 };

        for c in odometry_constraints {
            let p1 = &poses[c.from_idx];
            let p2 = &poses[c.to_idx];

            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let cos_t = p1.theta.cos();
            let sin_t = p1.theta.sin();

            let dx_local = cos_t * dx + sin_t * dy;
            let dy_local = -sin_t * dx + cos_t * dy;
            let dtheta = normalize_angle(p2.theta - p1.theta);

            let error = Vector3::new(
                c.measurement[0] - dx_local,
                c.measurement[1] - dy_local,
                normalize_angle(c.measurement[2] - dtheta),
            );

            // nalgebra returns L with information = L L^T. Least-squares
            // whitening needs W^T W = information, so W = L^T.
            let sqrt_info = c
                .information
                .cholesky()
                .map(|ch| ch.l().transpose())
                .unwrap_or(Matrix3::identity());

            let sqrt_robust = (c.robust_weight as f32).sqrt();
            let weighted_error = sqrt_info * error * sqrt_robust;

            for i in 0..3 {
                residuals[res_idx + i] = weighted_error[i] as f64;
            }

            if let Some(idx1) = pose_state_idx(c.from_idx) {
                let j1 = Matrix3::new(
                    -cos_t,
                    -sin_t,
                    -sin_t * dx + cos_t * dy,
                    sin_t,
                    -cos_t,
                    -cos_t * dx - sin_t * dy,
                    0.0,
                    0.0,
                    -1.0,
                );
                let wj1 = sqrt_info * j1 * sqrt_robust;
                for i in 0..3 {
                    for j in 0..3 {
                        let val = -wj1[(i, j)] as f64;
                        if val.abs() > 1e-12 {
                            triplets.push(Triplet {
                                row: res_idx + i,
                                col: idx1 + j,
                                value: val,
                            });
                        }
                    }
                }
            }

            if let Some(idx2) = pose_state_idx(c.to_idx) {
                let j2 = Matrix3::new(cos_t, sin_t, 0.0, -sin_t, cos_t, 0.0, 0.0, 0.0, 1.0);
                let wj2 = sqrt_info * j2 * sqrt_robust;
                for i in 0..3 {
                    for j in 0..3 {
                        let val = -wj2[(i, j)] as f64;
                        if val.abs() > 1e-12 {
                            triplets.push(Triplet {
                                row: res_idx + i,
                                col: idx2 + j,
                                value: val,
                            });
                        }
                    }
                }
            }

            res_idx += 3;
        }

        for c in observation_constraints {
            let pose = &poses[c.pose_idx];
            let lm = &landmarks[c.landmark_idx];

            let dx = lm.x - pose.x;
            let dy = lm.y - pose.y;
            let q = dx * dx + dy * dy;
            let sqrt_q = q.sqrt().max(1e-6);

            let pred_range = sqrt_q;
            let pred_bearing = normalize_angle(dy.atan2(dx) - pose.theta);
            let error = Vector2::new(
                c.measurement[0] - pred_range,
                normalize_angle(c.measurement[1] - pred_bearing),
            );

            let sqrt_info = c
                .information
                .cholesky()
                .map(|ch| ch.l().transpose())
                .unwrap_or(Matrix2::identity());

            let sqrt_robust = (c.robust_weight as f32).sqrt();
            let weighted_error = sqrt_info * error * sqrt_robust;

            residuals[res_idx] = weighted_error[0] as f64;
            residuals[res_idx + 1] = weighted_error[1] as f64;

            if let Some(pidx) = pose_state_idx(c.pose_idx) {
                let jp = nalgebra::Matrix2x3::new(
                    -dx / sqrt_q,
                    -dy / sqrt_q,
                    0.0,
                    dy / q,
                    -dx / q,
                    -1.0,
                );
                let wjp = sqrt_info * jp * sqrt_robust;
                for i in 0..2 {
                    for j in 0..3 {
                        let val = -wjp[(i, j)] as f64;
                        if val.abs() > 1e-12 {
                            triplets.push(Triplet {
                                row: res_idx + i,
                                col: pidx + j,
                                value: val,
                            });
                        }
                    }
                }
            }

            let lm_idx = landmark_state_idx(c.landmark_idx);
            let jl = Matrix2::new(dx / sqrt_q, dy / sqrt_q, -dy / q, dx / q);
            let wjl = sqrt_info * jl * sqrt_robust;
            for i in 0..2 {
                for j in 0..2 {
                    let val = -wjl[(i, j)] as f64;
                    if val.abs() > 1e-12 {
                        triplets.push(Triplet {
                            row: res_idx + i,
                            col: lm_idx + j,
                            value: val,
                        });
                    }
                }
            }

            res_idx += 2;
        }

        let jacobian = triplets_to_csr(&triplets, n_residuals, n_vars);
        (jacobian, residuals)
    }

    /// Solve `(J^T J + damping) dx = -J^T r`.
    ///
    /// Sparse products are retained through construction of `J^T J`, then the
    /// regularized normal matrix is converted to dense form and solved by LU.
    /// Returns `None` if the dense reference system is singular.
    pub fn solve(
        &self,
        jacobian: &CsMat<f64>,
        residuals: &DVector<f64>,
        lambda: f64,
    ) -> Option<DVector<f64>> {
        let n_vars = jacobian.cols();
        if n_vars == 0 {
            return Some(DVector::zeros(0));
        }

        let jt = jacobian.transpose_view();
        let jtj = &jt * jacobian;

        let r_sparse = dense_to_sparse_vec(residuals);
        let jtr_sparse = &jt * &r_sparse;
        let mut jtr = DVector::zeros(n_vars);
        for (idx, &val) in jtr_sparse.iter() {
            jtr[idx] = val;
        }

        let mut h_diag = DVector::zeros(n_vars);
        for (val, (row, col)) in jtj.iter() {
            if row == col {
                h_diag[row] = *val;
            }
        }

        let mut h = nalgebra::DMatrix::zeros(n_vars, n_vars);
        for (val, (row, col)) in jtj.iter() {
            h[(row, col)] = *val;
        }
        for i in 0..n_vars {
            h[(i, i)] += lambda * (1.0 + h_diag[i]) + self.config.regularization;
        }

        h.lu().solve(&(-&jtr))
    }

    /// Compute statistics about the sparse Jacobian structure.
    pub fn sparsity_stats(jacobian: &CsMat<f64>) -> SparsityStats {
        let rows = jacobian.rows();
        let cols = jacobian.cols();
        let nnz = jacobian.nnz();
        let total = rows * cols;
        let density = if total > 0 {
            nnz as f64 / total as f64
        } else {
            0.0
        };

        SparsityStats {
            rows,
            cols,
            nnz,
            density,
        }
    }
}

/// Statistics about sparse matrix structure.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub density: f64,
}

impl std::fmt::Display for SparsityStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} matrix, {} non-zeros ({:.2}% dense)",
            self.rows,
            self.cols,
            self.nnz,
            self.density * 100.0
        )
    }
}

fn triplets_to_csr(triplets: &[Triplet], rows: usize, cols: usize) -> CsMat<f64> {
    if triplets.is_empty() {
        return CsMat::empty(sprs::CSR, cols);
    }

    let mut tri_mat = TriMat::new((rows, cols));
    for t in triplets {
        tri_mat.add_triplet(t.row, t.col, t.value);
    }
    tri_mat.to_csr()
}

fn dense_to_sparse_vec(v: &DVector<f64>) -> CsVec<f64> {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (i, &val) in v.iter().enumerate() {
        if val.abs() > 1e-12 {
            indices.push(i);
            values.push(val);
        }
    }

    CsVec::new(v.len(), indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slam::{GraphSlam, Landmark2D, Pose2D};
    use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

    #[test]
    fn test_sparse_vs_dense_simple() {
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;

        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(1.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(2.0, 0.0, 0.0));

        let cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(0, 1, Vector3::new(1.0, 0.0, 0.0), &cov);
        graph.add_odometry(1, 2, Vector3::new(1.0, 0.0, 0.0), &cov);

        let solver = SparseSlamSolver::new();
        let (j_sparse, r_sparse) = solver.build_sparse_system(
            &graph.poses,
            &graph.landmarks,
            &graph.odometry_constraints,
            &graph.observation_constraints,
            graph.fix_first_pose,
        );

        assert_eq!(r_sparse.len(), 6);
        assert_eq!(j_sparse.cols(), 6);
        assert_eq!(j_sparse.rows(), 6);

        let stats = SparseSlamSolver::sparsity_stats(&j_sparse);
        assert!(stats.nnz < stats.rows * stats.cols);
    }

    #[test]
    fn correlated_information_uses_the_weighted_least_squares_metric() {
        let solver = SparseSlamSolver::new();
        let poses = [Pose2D::new(0.0, 0.0, 0.0), Pose2D::new(1.0, 0.0, 0.0)];
        let information = Matrix3::new(2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0);
        let error = Vector3::new(1.0, -0.5, 0.25);
        let constraints = [crate::slam::OdometryConstraint {
            from_idx: 0,
            to_idx: 1,
            measurement: Vector3::new(2.0, -0.5, 0.25),
            information,
            is_outlier: false,
            robust_weight: 1.0,
        }];

        let (_, residuals) = solver.build_sparse_system(&poses, &[], &constraints, &[], true);
        let sparse_cost = residuals.dot(&residuals) as f32;
        let expected_cost = (error.transpose() * information * error)[0];

        assert!(
            (sparse_cost - expected_cost).abs() < 1e-5,
            "whitened cost {sparse_cost} must equal r^T Ω r {expected_cost}"
        );
    }

    #[test]
    fn test_sparse_solver_convergence() {
        let mut graph = GraphSlam::new();
        graph.config.enable_robust_kernel = false;

        graph.add_pose(Pose2D::new(0.0, 0.0, 0.0));
        graph.add_pose(Pose2D::new(5.0, 0.1, 0.0));
        graph.add_pose(Pose2D::new(10.0, 0.2, 0.0));
        graph.add_landmark(Landmark2D::new(2.5, 5.1));

        let odom_cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
        graph.add_odometry(0, 1, Vector3::new(5.0, 0.0, 0.0), &odom_cov);
        graph.add_odometry(1, 2, Vector3::new(5.0, 0.0, 0.0), &odom_cov);

        let obs_cov = Matrix2::from_diagonal(&Vector2::new(0.1, 0.01));
        let r0 = (2.5f32.powi(2) + 5.0f32.powi(2)).sqrt();
        let b0 = 5.0f32.atan2(2.5);
        graph.add_observation(0, 0, r0, b0, &obs_cov);

        let r1 = (2.5f32.powi(2) + 5.0f32.powi(2)).sqrt();
        let b1 = 5.0f32.atan2(-2.5);
        graph.add_observation(1, 0, r1, b1, &obs_cov);

        let r2 = (7.5f32.powi(2) + 5.0f32.powi(2)).sqrt();
        let b2 = 5.0f32.atan2(-7.5);
        graph.add_observation(2, 0, r2, b2, &obs_cov);

        let solver = SparseSlamSolver::new();
        let mut lambda = 1e-3;
        let mut current_error = graph.total_error();

        for _ in 0..10 {
            let (j, r) = solver.build_sparse_system(
                &graph.poses,
                &graph.landmarks,
                &graph.odometry_constraints,
                &graph.observation_constraints,
                graph.fix_first_pose,
            );

            if let Some(dx) = solver.solve(&j, &r, lambda) {
                let mut idx = 0;
                for pose in graph.poses.iter_mut().skip(1) {
                    pose.x += dx[idx] as f32;
                    pose.y += dx[idx + 1] as f32;
                    pose.theta += dx[idx + 2] as f32;
                    idx += 3;
                }
                for lm in &mut graph.landmarks {
                    lm.x += dx[idx] as f32;
                    lm.y += dx[idx + 1] as f32;
                    idx += 2;
                }

                let new_error = graph.total_error();
                if new_error < current_error {
                    current_error = new_error;
                    lambda *= 0.1;
                } else {
                    lambda *= 10.0;
                }
            }
        }

        assert!(current_error < 1.0, "sparse assembly + dense solve should converge");
    }

    #[test]
    fn test_sparsity_scales() {
        let solver = SparseSlamSolver::new();

        for n_poses in [5, 10, 20, 50] {
            let mut graph = GraphSlam::new();
            for i in 0..n_poses {
                graph.add_pose(Pose2D::new(i as f32, 0.0, 0.0));
            }

            let cov = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01));
            for i in 0..(n_poses - 1) {
                graph.add_odometry(i, i + 1, Vector3::new(1.0, 0.0, 0.0), &cov);
            }

            let (j, _) = solver.build_sparse_system(
                &graph.poses,
                &graph.landmarks,
                &graph.odometry_constraints,
                &graph.observation_constraints,
                graph.fix_first_pose,
            );

            let stats = SparseSlamSolver::sparsity_stats(&j);
            assert!(stats.density < 0.5, "graph with {n_poses} poses should be sparse");
        }
    }
}
