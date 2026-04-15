//! Simultaneous localization and mapping algorithms.
//!
//! This module exposes two major SLAM families used by the simulator:
//!
//! - EKF-SLAM, which maintains one joint Gaussian over pose and landmarks
//! - graph-based SLAM, which optimizes a sparse global constraint graph
//!
//! The supporting submodules provide loop-closure detection, robust kernels,
//! sparse linear algebra, and marginalization helpers.
pub mod ekf_slam;
pub mod graph_slam;
pub mod loop_closure;
pub mod marginalization;
pub mod robust_kernels;
pub mod sparse_solver;

pub use ekf_slam::*;
pub use graph_slam::*;
pub use loop_closure::*;
pub use marginalization::*;
pub use robust_kernels::*;
pub use sparse_solver::*;
