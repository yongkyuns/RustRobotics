//! Pendulum-specific binding of the generic LQR trait.
//!
//! The pendulum `Model` already knows how to produce its discrete-time
//! state-space matrices. This file simply tells the generic LQR machinery which
//! `Q`, `R`, convergence tolerance, and iteration limit belong to that model.
use super::*;

impl LQR<NX, NU> for Model {
    /// Returns the pendulum state's quadratic penalty matrix.
    fn Q(&self) -> QMat {
        self.Q
    }
    /// Returns the pendulum control-effort penalty matrix.
    fn R(&self) -> RMat {
        self.R
    }
    /// Returns the pseudo-inverse / Riccati convergence tolerance.
    fn epsilon(&self) -> f32 {
        self.eps
    }
    /// Returns the maximum number of DARE iterations.
    fn max_iter(&self) -> u32 {
        self.max_iter
    }
}
