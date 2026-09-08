//! Fixed-scale Gaussian exploration followed by a bounded action transform.
//!
//! The MLP predicts a dimensionless latent mean. `action_std` retains force
//! units: z ~ N(mean, (action_std / limit)^2), action = limit * tanh(z).
//! Store z rather than inverting a rounded/saturated f32 action during replay.
use burn::tensor::{backend::Backend, Tensor};
use rand::{distributions::Open01, Rng};

#[derive(Clone, Copy)]
pub(super) struct SquashedGaussian {
    std: f32,
    action_limit: f32,
}

pub(super) struct PolicySample {
    pub latent: f32,
    pub action: f32,
    pub log_prob: f32,
}

impl SquashedGaussian {
    pub fn new(action_std: f32, action_limit: f32) -> Self {
        assert!(
            action_std.is_finite() && action_std > 0.0,
            "action_std must be finite and positive"
        );
        assert!(
            action_limit.is_finite() && action_limit > 0.0,
            "action_limit must be finite and positive"
        );
        let std = action_std / action_limit;
        assert!(
            std.is_finite() && std > 0.0,
            "normalized action standard deviation must be finite and positive"
        );
        Self { std, action_limit }
    }

    pub fn sample<R: Rng + ?Sized>(&self, mean: f32, rng: &mut R) -> PolicySample {
        let latent = mean + self.std * standard_normal(rng);
        PolicySample {
            latent,
            action: self.action_limit * latent.tanh(),
            log_prob: self.log_prob(mean, latent),
        }
    }

    pub fn log_prob(&self, mean: f32, latent: f32) -> f32 {
        let standardized = (latent - mean) / self.std;
        -0.5 * standardized * standardized
            - self.std.ln()
            - 0.5 * (2.0 * std::f32::consts::PI).ln()
            - self.action_limit.ln()
            - log_tanh_jacobian(latent)
    }

    pub fn log_prob_tensor<B: Backend>(
        &self,
        means: Tensor<B, 2>,
        latents: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let gaussian = (latents.clone() - means)
            .div_scalar(self.std)
            .square()
            .mul_scalar(-0.5)
            .sub_scalar(self.std.ln() + 0.5 * (2.0 * std::f32::consts::PI).ln());
        gaussian.sub_scalar(self.action_limit.ln()) - log_tanh_jacobian_tensor(latents)
    }

    /// Reparameterized Monte Carlo entropy of the *current bounded* policy.
    /// The caller supplies fresh standard-normal noise, not old rollout actions.
    /// Gaussian entropy is analytic; only the expected log-Jacobian is sampled.
    pub fn entropy<B: Backend>(&self, means: Tensor<B, 2>, noise: Tensor<B, 2>) -> Tensor<B, 1> {
        let latents = means + noise.mul_scalar(self.std);
        log_tanh_jacobian_tensor(latents)
            .add_scalar(
                self.std.ln()
                    + self.action_limit.ln()
                    + 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln()),
            )
            .mean()
    }
}

// log(1 - tanh(z)^2) = 2 * (ln(2) - |z| - log1p(exp(-2|z|))).
// This stays finite when f32 tanh rounds to +/-1; no inverse tanh or epsilon
// inside the density is needed. The even form also avoids exp overflow.
fn log_tanh_jacobian(latent: f32) -> f32 {
    let magnitude = latent.abs();
    2.0 * (std::f32::consts::LN_2 - magnitude - (-2.0 * magnitude).exp().ln_1p())
}

fn log_tanh_jacobian_tensor<B: Backend>(latents: Tensor<B, 2>) -> Tensor<B, 2> {
    let magnitude = latents.abs();
    let correction = magnitude
        .clone()
        .mul_scalar(-2.0)
        .exp()
        .add_scalar(1.0)
        .log();
    (magnitude.neg().add_scalar(std::f32::consts::LN_2) - correction).mul_scalar(2.0)
}

pub(super) fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1: f32 = rng.sample(Open01);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
