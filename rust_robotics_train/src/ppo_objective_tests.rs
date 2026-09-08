//! Independent f64 density/objective references and real optimizer regressions.
//! Seeds and mixed f32 error budgets are declared before qualification.
use super::*;
use crate::backend::TrainBackend;
use burn::tensor::Tensor;
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_core::{LinearSnapshot, ValueSnapshot};

fn close(actual: f32, expected: f64, tolerance: f64) {
    assert!(actual.is_finite() && expected.is_finite(), "non-finite PPO result");
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance * expected.abs().max(1.0),
        "PPO oracle mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

// Direct change of variables, independent of the stable production identity.
fn reference_log_prob(mean: f64, latent: f64, action_std: f64, limit: f64) -> f64 {
    let sigma = action_std / limit;
    let action = limit * latent.tanh();
    -0.5 * ((latent - mean) / sigma).powi(2)
        - (sigma * (2.0 * std::f64::consts::PI).sqrt()).ln()
        - (limit * (1.0 - (action / limit).powi(2))).ln()
}

fn column(values: &[f32]) -> Tensor<AutodiffBackend, 2> {
    scalar_tensor(&Default::default(), values)
}

#[test]
fn samples_preserve_transform_and_seed() {
    let distribution = SquashedGaussian::new(2.0, 3.0);
    let mut a = StdRng::seed_from_u64(0x5050_0001);
    let mut b = StdRng::seed_from_u64(0x5050_0001);
    for mean in [-2.0, 0.0, 2.0] {
        for _ in 0..256 {
            let x = distribution.sample(mean, &mut a);
            let y = distribution.sample(mean, &mut b);
            assert_eq!(x.latent, y.latent, "seeded latent replay");
            assert_eq!(x.action, y.action, "seeded action replay");
            assert_eq!(x.action, 3.0 * x.latent.tanh(), "action must be the latent transform");
            assert!(x.action.is_finite() && x.action.abs() <= 3.0);
            assert!(x.log_prob.is_finite());
        }
    }
}

#[test]
fn sampled_cdf_matches_normal_quantiles() {
    let distribution = SquashedGaussian::new(2.0, 4.0);
    let mut rng = StdRng::seed_from_u64(0x5050_0002);
    let mean = 0.4_f32;
    let thresholds = [-1.0_f32, 0.0, 1.0].map(|z| 4.0 * (mean + 0.5 * z).tanh());
    let expected = [0.158655253931457, 0.5, 0.841344746068543];
    let mut counts = [0; 3];
    const N: usize = 16_384;
    for _ in 0..N {
        let sample = distribution.sample(mean, &mut rng);
        for i in 0..3 {
            counts[i] += usize::from(sample.action <= thresholds[i]);
        }
    }
    // Hoeffding + union bound for three Bernoulli proportions, budget 1e-6.
    let bound = ((6.0_f64 / 1e-6).ln() / (2.0 * N as f64)).sqrt();
    for i in 0..3 {
        assert!((counts[i] as f64 / N as f64 - expected[i]).abs() < bound);
    }
}

#[test]
fn scalar_density_matches_independent_change_of_variables() {
    for limit in [0.5_f32, 3.0, 20.0] {
        let action_std = 0.7 * limit;
        let distribution = SquashedGaussian::new(action_std, limit);
        for mean in [-2.0_f32, 0.0, 2.0] {
            for offset in [-1.0_f32, 0.0, 0.7] {
                let latent = mean + offset;
                close(
                    distribution.log_prob(mean, latent),
                    reference_log_prob(mean.into(), latent.into(), action_std.into(), limit.into()),
                    2e-5,
                );
            }
        }
    }
}

#[test]
fn bounded_density_integrates_to_one() {
    // Midpoint quadrature directly in action coordinates. These smooth fixtures
    // have negligible boundary density. The 2e-4 budget covers discretization
    // plus evaluating each integrand in f32, not a fitted baseline output.
    for limit in [0.5_f32, 20.0] {
        let distribution = SquashedGaussian::new(0.4 * limit, limit);
        for mean in [-0.6_f32, 0.6] {
            for n in [32_768, 65_536] {
                let dx = 2.0 * f64::from(limit) / f64::from(n);
                let integral: f64 = (0..n).map(|i| {
                    let action = -f64::from(limit) + (f64::from(i) + 0.5) * dx;
                    let latent = (action / f64::from(limit)).atanh() as f32;
                    f64::from(distribution.log_prob(mean, latent)).exp() * dx
                }).sum();
                assert!((integral - 1.0).abs() < 2e-4, "density normalization: {integral}");
            }
        }
    }
}

#[test]
fn tensor_density_and_score_match_independent_reference() {
    let means = [-1.5, 0.0, 0.7, 2.0];
    let latents = [-2.0, 0.2, 1.1, 2.7];
    let distribution = SquashedGaussian::new(2.0, 3.0);
    let mu = column(&means).require_grad();
    let log_probs = distribution.log_prob_tensor(mu.clone(), column(&latents));
    let data = log_probs.to_data().to_vec::<f32>().unwrap();
    let gradients = log_probs.sum().backward();
    let scores = mu.grad(&gradients).unwrap().to_data().to_vec::<f32>().unwrap();
    for i in 0..means.len() {
        close(data[i], reference_log_prob(means[i].into(), latents[i].into(), 2.0, 3.0), 2e-5);
        close(scores[i], (f64::from(latents[i]) - f64::from(means[i])) / (2.0_f64 / 3.0).powi(2), 2e-5);
    }
}

#[test]
fn saturated_latents_have_finite_consistent_replay() {
    let latents = [-1000.0_f32, -20.0, -10.0, 0.0, 10.0, 20.0, 1000.0];
    let distribution = SquashedGaussian::new(2.0, 20.0);
    let tensor = distribution.log_prob_tensor(column(&latents), column(&latents));
    let tensor_values = tensor.to_data().to_vec::<f32>().unwrap();
    for (i, latent) in latents.into_iter().enumerate() {
        let scalar = distribution.log_prob(latent, latent);
        close(tensor_values[i], scalar.into(), 2e-6);
        // Same mean/latent must give unit ratio, including actions rounded to bounds.
        close((tensor_values[i] - scalar).exp(), 1.0, 2e-5);
    }
}

#[test]
fn bounded_entropy_gradient_matches_finite_differences() {
    let means = [-2.0_f32, -0.2, 0.0, 0.3, 2.0];
    let noise = [-1.5_f32, -0.4, 0.0, 0.7, 1.6];
    let mu = column(&means).require_grad();
    let distribution = SquashedGaussian::new(2.0, 3.0);
    let entropy = distribution.entropy(mu.clone(), column(&noise));
    let gradients = entropy.clone().backward();
    let actual = mu.grad(&gradients).unwrap().to_data().to_vec::<f32>().unwrap();
    let reference = |ms: &[f64]| -> f64 {
        let sigma = 2.0 / 3.0;
        let gaussian = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * sigma * sigma).ln();
        gaussian + 3.0_f64.ln() + ms.iter().zip(noise).map(|(m, e)| {
            let z = m + sigma * f64::from(e);
            (1.0 - z.tanh().powi(2)).ln()
        }).sum::<f64>() / ms.len() as f64
    };
    let ms = means.map(f64::from);
    close(tensor_scalar(&entropy), reference(&ms), 2e-5);
    // h balances f64 cancellation/truncation well below the f32 gradient budget.
    let h = 1e-5;
    for i in 0..ms.len() {
        let mut plus = ms;
        let mut minus = ms;
        plus[i] += h;
        minus[i] -= h;
        close(actual[i], (reference(&plus) - reference(&minus)) / (2.0 * h), 5e-5);
    }
}

#[test]
fn clipping_values_and_gradients_cover_both_advantage_signs() {
    let logs = [0.5_f32.ln(), 1.0_f32.ln(), 1.5_f32.ln(), 0.5_f32.ln(), 1.0_f32.ln(), 1.5_f32.ln()];
    let adv = [1.0_f32, 1.0, 1.0, -1.0, -1.0, -1.0];
    for epsilon in [0.1_f32, 0.2, 0.4] {
        let new_logs = column(&logs).require_grad();
        let loss = clipped_surrogate(new_logs.clone(), column(&[0.0; 6]), column(&adv), epsilon);
        let gradients = loss.clone().backward();
        let actual = new_logs.grad(&gradients).unwrap().to_data().to_vec::<f32>().unwrap();
        let reference = |input: &[f64]| -> f64 {
            -input.iter().zip(adv).map(|(log, a)| {
                let r = log.exp();
                (r * f64::from(a)).min(r.clamp(1.0 - f64::from(epsilon), 1.0 + f64::from(epsilon)) * f64::from(a))
            }).sum::<f64>() / 6.0
        };
        let inputs = logs.map(f64::from);
        close(tensor_scalar(&loss), reference(&inputs), 2e-6);
        for i in 0..6 {
            let mut plus = inputs;
            let mut minus = inputs;
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            close(actual[i], (reference(&plus) - reference(&minus)) / 2e-5, 2e-5);
        }
    }
}

#[test]
fn value_coefficient_scales_loss_and_gradient() {
    let predictions = [0.5_f32, -0.2, 1.0];
    let targets = [2.0_f32, 0.3, -0.4];
    for coefficient in [0.0_f32, 0.25, 1.0, 2.0] {
        let values = column(&predictions).require_grad();
        let mse = (values.clone() - column(&targets)).square().mean();
        let loss = weighted_value_loss(mse, coefficient);
        let gradients = loss.clone().backward();
        let actual = values.grad(&gradients).unwrap().to_data().to_vec::<f32>().unwrap();
        let expected = predictions.iter().zip(targets).map(|(p, t)| {
            (f64::from(*p) - f64::from(t)).powi(2)
        }).sum::<f64>() * f64::from(coefficient) / 3.0;
        close(tensor_scalar(&loss), expected, 2e-6);
        for i in 0..3 {
            close(actual[i], 2.0 * f64::from(coefficient) * (f64::from(predictions[i]) - f64::from(targets[i])) / 3.0, 2e-6);
        }
    }
}

fn constant_layers(output: f32) -> (LinearSnapshot, LinearSnapshot, LinearSnapshot) {
    (
        LinearSnapshot { in_dim: 4, out_dim: 1, weight: vec![0.0; 4], bias: vec![1.0] },
        LinearSnapshot { in_dim: 1, out_dim: 1, weight: vec![1.0], bias: vec![0.0] },
        LinearSnapshot { in_dim: 1, out_dim: 1, weight: vec![0.0], bias: vec![output] },
    )
}

fn session(mean: f32, entropy: f32) -> PpoTrainerSession {
    let config = PpoTrainerConfig {
        hidden_dim: 1,
        ppo: PpoConfig { rollout_steps: 8, mini_batch_size: 8, epochs_per_update: 1, entropy_coef: entropy, ..PpoConfig::default() },
        ..PpoTrainerConfig::default()
    };
    let mut session = PpoTrainerSession::new(config);
    let (input, hidden, output) = constant_layers(mean);
    let policy = PolicySnapshot { input, hidden, output, action_limit: 20.0, action_std: 2.0 };
    let (input, hidden, output) = constant_layers(0.5);
    session.load_shared_state(&PpoSharedState { policy, value: ValueSnapshot { input, hidden, output } });
    session
}

fn rollout(mean: f32, advantage: f32) -> RolloutBatch {
    let distribution = SquashedGaussian::new(2.0, 20.0);
    let latents = vec![mean + 0.1; 8];
    RolloutBatch {
        observations: vec![[0.0; 4]; 8],
        old_log_probs: latents.iter().map(|z| distribution.log_prob(mean, *z)).collect(),
        latent_actions: latents,
        returns: vec![2.0; 8],
        advantages: vec![advantage; 8],
    }
}

#[test]
fn entropy_coefficient_changes_actual_actor_without_advantages() {
    let mut disabled = session(2.0, 0.0);
    let mut enabled = session(2.0, 0.1);
    let before = disabled.snapshot();
    let batch = rollout(2.0, 0.0);
    disabled.optimize_with_rng(&batch, &mut StdRng::seed_from_u64(0x5050_0003));
    enabled.optimize_with_rng(&batch, &mut StdRng::seed_from_u64(0x5050_0003));
    assert_eq!(disabled.snapshot(), before, "zero advantages and entropy leave fresh actor unchanged");
    assert_ne!(enabled.snapshot(), before, "entropy coefficient must affect actor parameters");
    assert!(enabled.snapshot().act([0.0; 4]) < before.act([0.0; 4]), "bounded entropy must move a saturated mean toward the center");
    // Metrics intentionally remain the unregularized surrogate and raw MSE.
    assert_eq!(enabled.metrics.last_policy_loss, 0.0);
    close(enabled.metrics.last_value_loss, 2.25, 1e-6);
}

#[test]
fn zero_value_coefficient_freezes_existing_adam_momentum() {
    let mut session = session(0.4, 0.0);
    let batch = rollout(0.4, 0.0);
    let before = session.shared_state().value;
    let mut rng = StdRng::seed_from_u64(0x5050_0004);
    session.optimize_with_rng(&batch, &mut rng);
    let trained = session.shared_state().value;
    assert_ne!(trained, before, "positive value coefficient must train critic");
    session.config.ppo.value_loss_coef = 0.0;
    for _ in 0..3 {
        session.optimize_with_rng(&batch, &mut rng);
        assert_eq!(session.shared_state().value, trained, "zero coefficient must also freeze Adam momentum");
    }
}

#[test]
fn policy_optimizer_follows_latent_action_score() {
    let mut session = session(0.4, 0.0);
    let before = session.policy_latent_mean([0.0; 4]);
    session.optimize_with_rng(&rollout(before, 1.0), &mut StdRng::seed_from_u64(0x5050_0005));
    assert!(session.policy_latent_mean([0.0; 4]) > before, "positive advantage should increase sampled latent likelihood");
}

#[test]
fn collected_rollout_replays_bounded_log_probabilities() {
    for mean in [0.4, 10.0] {
        let mut session = session(mean, 0.0);
        let batch = session.collect_rollout();
        let observations = obs_tensor::<TrainBackend>(&session.device, &batch.observations);
        let means = session.actor.valid().latent_mean(observations);
        let latents = scalar_tensor::<TrainBackend>(&session.device, &batch.latent_actions);
        let distribution = SquashedGaussian::new(2.0, 20.0);
        let replay = distribution.log_prob_tensor(means, latents).to_data().to_vec::<f32>().unwrap();
        for (new, old) in replay.iter().zip(batch.old_log_probs) {
            close((*new - old).exp(), 1.0, 2e-5);
        }
    }
}

#[test]
fn deterministic_snapshot_inference_is_unchanged() {
    for mean in [-10.0_f32, -2.0, 0.0, 0.4, 2.0, 10.0] {
        let session = session(mean, 0.0);
        for obs in [[0.0; 4], [0.1, -0.2, 0.3, 0.4]] {
            let tensor = obs_tensor::<TrainBackend>(&session.device, &[obs]);
            let expected = session.snapshot().act(obs);
            close(tensor_scalar(&session.actor.valid().forward(tensor)), expected.into(), 2e-6);
            close(expected, 20.0 * f64::from(mean).tanh(), 2e-6);
        }
    }
}

#[test]
fn shared_weights_adopt_saved_exploration_scale() {
    let mut session = session(0.4, 0.0);
    let mut saved = session.shared_state();
    saved.policy.action_std = 0.25;
    session.load_shared_state(&saved);
    assert_eq!(session.config.action_std, 0.25);
    assert_eq!(session.snapshot(), saved.policy);
    assert_eq!(session.shared_state().value, saved.value);
}

#[test]
#[should_panic(expected = "shared policy action limit must match the environment")]
fn shared_weights_reject_incompatible_action_limit() {
    let mut session = session(0.4, 0.0);
    let mut saved = session.shared_state();
    saved.policy.action_limit = 40.0;
    session.load_shared_state(&saved);
}

#[test]
fn invalid_distribution_parameters_are_rejected() {
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| SquashedGaussian::new(invalid, 20.0)).is_err());
        assert!(std::panic::catch_unwind(|| SquashedGaussian::new(2.0, invalid)).is_err());
    }
}
