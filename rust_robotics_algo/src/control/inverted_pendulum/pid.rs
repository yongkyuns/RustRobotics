//! Simple PID controller used as a baseline for the pendulum demo.
//!
//! Unlike LQR and MPC, which rely on a model of the cart-pole dynamics, this
//! controller acts directly on an error signal supplied by the caller. In the
//! simulator that error is typically the rod-angle error relative to upright.
//!
//! The control law is:
//!
//! `u = P * e + I * integral(e dt) + D * de/dt`
//!
//! This makes PID easy to tune and reason about, but it does not explicitly
//! capture cross-coupling between cart position and rod angle the way the
//! model-based controllers do.
/// Simple PID controller
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PID {
    /// P Gain. `Default = 1.0`
    pub P: f32,
    /// I Gain. `Default = 0.0`
    pub I: f32,
    /// D Gain. `Default = 0.0`
    pub D: f32,
    /// Integral error
    err_int: f32,
    /// Error from previous sample time
    err_prev: f32,
}

impl Default for PID {
    fn default() -> Self {
        Self {
            P: 1.0,
            I: 0.0,
            D: 0.0,
            err_int: 0.0,
            err_prev: 0.0,
        }
    }
}

impl PID {
    /// Creates a controller with default gains.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a controller with caller-provided `P`, `I`, and `D` gains.
    pub fn with_gains(P: f32, I: f32, D: f32) -> Self {
        let mut pid = Self::default();
        pid.P = P;
        pid.I = I;
        pid.D = D;
        pid
    }
    /// Clears the accumulated integral and previous-error state.
    pub fn reset_state(&mut self) {
        self.err_int = 0.0;
        self.err_prev = 0.0;
    }
    /// Computes the PID control action for the current error sample.
    ///
    /// The derivative term is approximated with a backward finite difference:
    ///
    /// `de/dt ~= (e_k - e_(k-1)) / dt`
    pub fn control(&mut self, err: f32, dt: f32) -> f32 {
        self.err_int += err * dt;
        let u = self.P * err + self.I * self.err_int + self.D * (err - self.err_prev) / dt;
        self.err_prev = err;
        u
    }
}
