pub mod lqr;
pub mod inverted_pendulum;
pub mod vehicle;

use crate::prelude::*;

pub use lqr::LQR;

/// Trait for providing a discrete-time state-space model
pub trait StateSpace<const N: usize, const M: usize, S = f32> {
    fn model(&self, dt: S) -> (Mat<N, N, S>, Mat<N, M, S>);
}
