//! Nonlinear point-mass cart-pole plant shared by training and live simulation.
//! The linear `inverted_pendulum::Model` remains a controller prediction model.
use crate::{
    inverted_pendulum::{g, Model},
    vector, Vector4,
};
use serde::{Deserialize, Serialize};

/// Physical parameters only: controller costs are deliberately not plant state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CartPoleParameters {
    pub length_m: f32,
    pub cart_mass_kg: f32,
    pub pole_mass_kg: f32,
}

impl Default for CartPoleParameters {
    fn default() -> Self {
        Self::from(Model::default())
    }
}

impl From<Model> for CartPoleParameters {
    fn from(model: Model) -> Self {
        Self {
            length_m: model.l_bar,
            cart_mass_kg: model.m_cart,
            pole_mass_kg: model.m_ball,
        }
    }
}

impl CartPoleParameters {
    /// Reject invalid configuration before it enters either runtime.
    pub fn validate(self) {
        assert!(
            [self.length_m, self.cart_mass_kg, self.pole_mass_kg]
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0),
            "cart-pole length and masses must be finite and positive"
        );
    }

    /// Compatibility with the existing classical prediction model.
    pub fn model(self) -> Model {
        self.validate();
        Model {
            l_bar: self.length_m,
            m_cart: self.cart_mass_kg,
            m_ball: self.pole_mass_kg,
            ..Model::default()
        }
    }

    /// State is [x, x_dot, theta, theta_dot], SI units. Positive theta tilts LEFT:
    /// bob position = (x - length*sin(theta), length*cos(theta)). Force is +right.
    pub fn derivative(self, state: Vector4, force_n: f32) -> Vector4 {
        let theta = state[2];
        let omega = state[3];
        let sin = theta.sin();
        let cos = theta.cos();
        let denominator = self.cart_mass_kg + self.pole_mass_kg * sin * sin;
        let acceleration = (force_n
            + self.pole_mass_kg * sin * (g * cos - self.length_m * omega * omega))
            / denominator;
        vector!(
            state[1],
            acceleration,
            omega,
            (g * sin + cos * acceleration) / self.length_m,
        )
    }

    /// One RK4 step with force held constant throughout the integration interval.
    pub fn step(self, state: Vector4, force_n: f32, dt: f32) -> Vector4 {
        self.validate();
        assert!(
            dt.is_finite() && dt > 0.0,
            "cart-pole dt must be finite and positive"
        );
        let k1 = self.derivative(state, force_n);
        let k2 = self.derivative(state + k1 * (0.5 * dt), force_n);
        let k3 = self.derivative(state + k2 * (0.5 * dt), force_n);
        let k4 = self.derivative(state + k3 * dt, force_n);
        state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::StateSpace;

    #[test]
    fn upright_equilibrium_and_force_sign() {
        let p = CartPoleParameters::default();
        assert_eq!(p.step(Vector4::zeros(), 0.0, 0.01), Vector4::zeros());
        let d = p.derivative(Vector4::zeros(), 2.0);
        assert_eq!(d[1], 2.0 / p.cart_mass_kg);
        assert_eq!(d[3], d[1] / p.length_m);
    }

    #[test]
    fn small_signal_matches_classical_linearization() {
        let p = CartPoleParameters {
            length_m: 1.3,
            cart_mass_kg: 0.8,
            pole_mass_kg: 0.4,
        };
        let (a, b) = p.model().model(1.0);
        for i in 0..4 {
            let mut x = Vector4::zeros();
            x[i] = 1e-4;
            let numeric = (p.derivative(x, 0.0) - p.derivative(-x, 0.0)) / 2e-4;
            let mut expected = Vector4::from_fn(|row, _| a[(row, i)]);
            expected[i] -= 1.0;
            assert!((numeric - expected).norm() < 1e-4);
        }
        assert!((p.derivative(Vector4::zeros(), 1.0) - b).norm() < 1e-6);
    }

    #[test]
    fn nonlinear_dynamics_satisfy_lagrange_equations_and_power_balance() {
        let p = CartPoleParameters {
            length_m: 1.7,
            cart_mass_kg: 0.7,
            pole_mass_kg: 0.3,
        };
        // Independent implicit equations from T-U, rather than the explicit implementation.
        for angle in [-1.2_f32, -0.4, 0.0, 0.7, 2.5] {
            let x = vector!(0.3, -0.6, angle, 1.1);
            let force = 2.3;
            let d = p.derivative(x, force);
            let (m, big_m, l) = (p.pole_mass_kg, p.cart_mass_kg, p.length_m);
            let (s, c) = (angle.sin(), angle.cos());
            let cart_residual =
                (big_m + m) * d[1] - m * l * c * d[3] + m * l * s * x[3] * x[3] - force;
            let pole_residual = l * d[3] - c * d[1] - g * s;
            assert!(cart_residual.abs() < 3e-6 && pole_residual.abs() < 3e-6);
            let de = (big_m + m) * x[1] * d[1] - m * l * c * (d[1] * x[3] + x[1] * d[3])
                + m * l * s * x[1] * x[3] * x[3]
                + m * l * l * x[3] * d[3]
                - m * g * l * s * x[3];
            assert!((de - force * x[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn rk4_refines_toward_independent_small_steps() {
        let p = CartPoleParameters::default();
        let start = vector!(0.2, -0.1, 0.6, 0.7);
        let integrate = |steps: usize| {
            let mut x = start;
            for _ in 0..steps {
                x = p.step(x, 1.3, 0.2 / steps as f32);
            }
            x
        };
        let reference = integrate(256);
        let coarse = (integrate(1) - reference).norm();
        let fine = (integrate(2) - reference).norm();
        assert!(fine < coarse / 5.0, "coarse={coarse}, fine={fine}");
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn invalid_physical_configuration_is_rejected() {
        CartPoleParameters {
            length_m: 0.0,
            ..Default::default()
        }
        .model();
    }
}
