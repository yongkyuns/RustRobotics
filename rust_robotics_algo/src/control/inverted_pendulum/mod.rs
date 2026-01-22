mod lqr;
mod mpc;
mod pid;

pub use lqr::*;
pub use mpc::*;
pub use pid::*;

pub use crate::control::{StateSpace, LQR};
use crate::prelude::*;

/// Gravity [m/s^2]
pub const g: f32 = 9.81;

/// Number of states
pub const NX: usize = 4;
/// Number of control input
pub const NU: usize = 1;

/// Convenience type for denoting system matrix A
pub type AMat = Mat<NX, NX>;
/// Convenience type for denoting input matrix B
pub type BMat = Mat<NX, NU>;
/// Convenience type for denoting Q matrix
pub type QMat = AMat;
/// Convenience type for denoting R matrix
pub type RMat = Mat<NU, NU>;

/// Define model parameters and LQR-related parameters.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Model {
    /// Length of bar [m]
    pub l_bar: f32,
    /// Mass of cart [kg]
    pub m_cart: f32,
    /// Mass of ball [kg]
    pub m_ball: f32,
    /// Q matrix
    pub Q: QMat,
    /// R matrix
    pub R: RMat,
    /// Tolerance for computing matrix pseudo-inverse
    pub eps: f32,
    /// Maximum number of iteration for solving
    /// Discrete Algebraic Ricatti Equation
    pub max_iter: u32,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            l_bar: 2.0,
            m_cart: 1.0,
            m_ball: 1.0,
            eps: 0.01,
            max_iter: 150,
            Q: diag![0., 1., 1., 0.],
            R: diag![0.01],
        }
    }
}

impl StateSpace<NX, NU> for Model {
    fn model(&self, dt: f32) -> (AMat, BMat) {
        let Self {
            l_bar,
            m_cart: m_c,
            m_ball: m_b,
            ..
        } = *self;

        let A = matrix![0., 1.,    0., 0.;
						0., 0., m_b*g / m_c, 0.;
						0., 0., 0., 1.;
						0., 0., g*(m_c+m_b)/(l_bar*m_c), 0.];

        let B = vector![0., 1. / m_c, 0., 1. / (l_bar * m_c)];

        (eye!(NX) + A * dt, B * dt)
    }
}
