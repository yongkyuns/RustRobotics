//! Classical control models and supporting dynamics.
//!
//! This module groups together control-oriented components that are reusable
//! across demos:
//!
//! - inverted-pendulum models and controllers
//! - generic discrete-time LQR support
//! - vehicle dynamics used by localization and control demos
pub mod inverted_pendulum;
pub mod lqr;
pub mod vehicle;

use crate::prelude::*;

pub use lqr::LQR;

/// Trait for providing a discrete-time state-space model
///
/// Implementors expose the linearized or exact discrete-time system matrices
/// used by downstream controllers such as LQR and MPC:
///
/// `x_(k+1) = A(dt) x_k + B(dt) u_k`
pub trait StateSpace<const N: usize, const M: usize, S = f32> {
    fn model(&self, dt: S) -> (Mat<N, N, S>, Mat<N, M, S>);
}
