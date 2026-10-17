//! Isolated native Rust implementation of the successful CleanRL *recipe*.
//!
//! This is an experimental control for PPO qualification, not an additional
//! production training mode. The existing Gaussian PPO and runtime snapshots
//! remain unchanged until this control passes independent held-out evaluation.
//!
//! The observed successful reference used four environments, a 512-row pool,
//! a two-logit categorical policy, two tanh hidden layers, clipped critic loss,
//! one joint Adam optimizer, minibatch advantage normalization, .01 entropy
//! bonus and an annealed 2.5e-4 learning rate. Physical rewards are unchanged.
//! Importantly, a 256-step native timeout bootstraps V(final observation),
//! while an actual physical failure never bootstraps or leaks into the reset.
use crate::{
    backend::{AutodiffBackend, AutodiffDevice},
    env::{PendulumEnv, PendulumEnvConfig},
    model::obs_tensor,
};
use burn::{
    module::{AutodiffModule, Module, Param},
    nn::Linear,
    optim::{adaptor::OptimizerAdaptor, AdamConfig, GradientsParams, Optimizer},
    tensor::{activation, backend::Backend, Tensor, TensorData},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rust_robotics_algo::cart_pole::CartPoleParameters;

const INPUTS: usize = 4;
const ACTIONS: usize = 2;

/// Action distribution is the SINGLE experimental ablation variable. Shared
/// PPO rollout, advantage, critic and optimizer code remains the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceActionDistribution {
    Categorical,
    /// Two actor outputs mean and log standard deviation for a scalar
    /// Gaussian latent, followed by bounded force = F*tanh(latent).
    SquashedGaussian,
}

/// Fixed, explicit experiment parameters. The real native plant/reward/noise
/// model is always used; these are not standard-Gym CartPole dynamics.
#[derive(Debug, Clone)]
pub struct CleanRlReferenceConfig {
    pub plant: CartPoleParameters,
    pub environment: PendulumEnvConfig,
    pub seed: u64,
    pub action_distribution: ReferenceActionDistribution,
    /// Experimental bounded output-force gain used in BOTH training and
    /// evaluation. Categorical must retain the 1.0 default.
    pub continuous_force_gain: f32,
    pub environments: usize,
    pub steps_per_environment: usize,
    pub minibatch_size: usize,
    pub epochs: usize,
    pub total_updates: usize,
    pub learning_rate: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_epsilon: f32,
    pub entropy_coefficient: f32,
    pub value_coefficient: f32,
    pub maximum_gradient_norm: f32,
}

impl Default for CleanRlReferenceConfig {
    fn default() -> Self {
        let environment = PendulumEnvConfig {
            max_steps: 256, // An external collection cutoff, not failure.
            ..PendulumEnvConfig::default()
        };
        Self {
            plant: CartPoleParameters::default(),
            environment,
            seed: 201,
            action_distribution: ReferenceActionDistribution::Categorical,
            continuous_force_gain: 1.0,
            environments: 4,
            steps_per_environment: 128,
            minibatch_size: 128,
            epochs: 4,
            total_updates: 976, // floor(500000 / (4 * 128))
            learning_rate: 2.5e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coefficient: 0.01,
            value_coefficient: 0.5,
            maximum_gradient_norm: 0.5,
        }
    }
}

impl CleanRlReferenceConfig {
    fn validate(&self) {
        self.plant.validate();
        assert!(self.environments > 0 && self.steps_per_environment > 0);
        let pool = self
            .environments
            .checked_mul(self.steps_per_environment)
            .expect("PPO pool size overflows");
        assert!(self.minibatch_size > 1 && pool.is_multiple_of(self.minibatch_size));
        assert!(self.epochs > 0 && self.total_updates > 0);
        assert!(self.learning_rate.is_finite() && self.learning_rate > 0.0);
        assert!(self.maximum_gradient_norm.is_finite() && self.maximum_gradient_norm > 0.0);
        assert!(self.gamma.is_finite() && (0.0..=1.0).contains(&self.gamma));
        assert!(self.gae_lambda.is_finite() && (0.0..=1.0).contains(&self.gae_lambda));
        assert!(self.clip_epsilon.is_finite() && self.clip_epsilon > 0.0);
        assert!(self.entropy_coefficient.is_finite() && self.entropy_coefficient >= 0.0);
        assert!(self.value_coefficient.is_finite() && self.value_coefficient >= 0.0);
        assert!(self.environment.max_force.is_finite() && self.environment.max_force > 0.0);
        assert!(self.continuous_force_gain.is_finite() && self.continuous_force_gain > 0.0);
        assert!(
            self.action_distribution == ReferenceActionDistribution::SquashedGaussian
                || self.continuous_force_gain == 1.0,
            "categorical reference must retain default force gain"
        );
    }

    /// Mirrors upstream's per-iteration annealing; the final rate is positive.
    pub fn rate_for_update(&self, index: usize) -> f64 {
        assert!(index < self.total_updates, "training budget exhausted");
        self.learning_rate * (1.0 - index as f64 / self.total_updates as f64)
    }
}

#[derive(Module, Debug)]
struct TanhMlp<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> TanhMlp<B> {
    fn new<R: Rng + ?Sized>(
        device: &B::Device,
        output_dim: usize,
        output_gain: f32,
        rng: &mut R,
    ) -> Self {
        Self {
            input: orthogonal_linear(device, INPUTS, 64, 2.0_f32.sqrt(), rng),
            hidden: orthogonal_linear(device, 64, 64, 2.0_f32.sqrt(), rng),
            output: orthogonal_linear(device, 64, output_dim, output_gain, rng),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let a = activation::tanh(self.input.forward(input));
        let b = activation::tanh(self.hidden.forward(a));
        self.output.forward(b)
    }
}

#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    actor: TanhMlp<B>,
    critic: TanhMlp<B>,
}

impl<B: Backend> ActorCritic<B> {
    fn new<R: Rng + ?Sized>(device: &B::Device, rng: &mut R) -> Self {
        Self {
            actor: TanhMlp::new(device, ACTIONS, 0.01, rng),
            critic: TanhMlp::new(device, 1, 1.0, rng),
        }
    }
}

// PyTorch stores [output, input]; Burn stores [input, output]. Generate
// orthonormal columns on the *tall* matrix and transpose as required.
// This reproduces the orthogonal-variance/gain contract, not PyTorch's
// byte-identical seeded QR matrices or RNG order.
fn orthogonal_linear<B: Backend, R: Rng + ?Sized>(
    device: &B::Device,
    inputs: usize,
    outputs: usize,
    gain: f32,
    rng: &mut R,
) -> Linear<B> {
    let height = inputs.max(outputs);
    let width = inputs.min(outputs);
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(width);
    for _ in 0..width {
        let mut basis = None;
        for _ in 0..32 {
            let mut v = (0..height).map(|_| normal(rng)).collect::<Vec<f64>>();
            // Two modified Gram-Schmidt passes control loss of orthogonality.
            for _ in 0..2 {
                for prev in &columns {
                    let projection: f64 = v.iter().zip(prev).map(|(x, y)| x * y).sum();
                    for (x, y) in v.iter_mut().zip(prev) {
                        *x -= projection * y;
                    }
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for x in &mut v {
                    *x /= norm;
                }
                basis = Some(v);
                break;
            }
        }
        columns.push(basis.expect("unable to generate orthogonal weights"));
    }
    // Row-major Burn weights: [input, output]. Columns contain the
    // orthonormal basis of the tall orientation.
    let weight: Vec<f32> = if outputs >= inputs {
        columns
            .iter()
            .flat_map(|column| column.iter().map(|&element| (gain as f64 * element) as f32))
            .collect()
    } else {
        (0..inputs)
            .flat_map(|input| {
                columns
                    .iter()
                    .map(move |column| (gain as f64 * column[input]) as f32)
            })
            .collect()
    };
    Linear {
        weight: Param::from_data(TensorData::new(weight, [inputs, outputs]), device),
        bias: Some(Param::from_data(
            TensorData::new(vec![0.0_f32; outputs], [outputs]),
            device,
        )),
    }
}

fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u = (1.0 - rng.gen::<f64>()).max(f64::MIN_POSITIVE);
    let v = rng.gen::<f64>();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Stable probabilities for both categories. Do not reconstruct the rare
/// category as `1 - p`: cancellation changes recorded action likelihoods.
fn categorical_probabilities(logits: [f32; 2]) -> [f32; 2] {
    assert!(logits.iter().all(|x| x.is_finite()));
    let high = logits[0].max(logits[1]);
    let exps = [(logits[0] - high).exp(), (logits[1] - high).exp()];
    let sum = exps[0] + exps[1];
    [exps[0] / sum, exps[1] / sum]
}

/// Numerically stable probability of the left category.
pub fn left_probability(logits: [f32; 2]) -> f32 {
    categorical_probabilities(logits)[0]
}

fn deterministic_action(logits: [f32; 2]) -> usize {
    let probabilities = categorical_probabilities(logits);
    usize::from(probabilities[1] > probabilities[0])
}

fn choose_action<R: Rng + ?Sized>(logits: [f32; 2], rng: &mut R) -> usize {
    let probability = left_probability(logits);
    usize::from(rng.gen::<f32>() >= probability)
}

/// The Gaussian is a density in the PRE-tanh latent domain. PPO uses the
/// exact same stored latent z on collection and replay. The tanh Jacobian
/// cancels from the importance ratio (but not from true action entropy).
const NORMAL_LOG_NORM: f32 = 0.918_938_5; // ln(sqrt(2*pi))
const GAUSSIAN_ENTROPY_BASE: f32 = 1.418_938_5; // 0.5*ln(2*pi*e)
const MIN_LOG_STD: f32 = -2.0;
const MAX_LOG_STD: f32 = 1.0;

fn continuous_force(latent: f32, max_force: f32) -> f32 {
    let force = max_force * latent.tanh();
    assert!(force.is_finite() && force.abs() <= max_force);
    force
}

/// Train-time force map. The policy importance ratio is taken over the
/// stored Gaussian latent, including where this clipped map is many-to-one.
/// This is not a physical-force-density or exact squashed-entropy claim.
fn continuous_force_with_gain(latent: f32, max_force: f32, gain: f32) -> f32 {
    assert!(gain.is_finite() && gain > 0.0);
    (continuous_force(latent, max_force) * gain).clamp(-max_force, max_force)
}

fn choose_policy_force<R: Rng + ?Sized>(
    parameters: [f32; 2],
    distribution: ReferenceActionDistribution,
    mode: ReferenceEvaluationMode,
    max_force: f32,
    continuous_gain: f32,
    rng: &mut R,
) -> (usize, f32, f32) {
    match distribution {
        ReferenceActionDistribution::Categorical => {
            let action = match mode {
                ReferenceEvaluationMode::Deterministic => deterministic_action(parameters),
                ReferenceEvaluationMode::Stochastic => choose_action(parameters, rng),
            };
            (action, 0.0, force_for_action(action, max_force))
        }
        ReferenceActionDistribution::SquashedGaussian => {
            assert!(parameters.iter().all(|x| x.is_finite()));
            let mean = parameters[0];
            let log_std = parameters[1].clamp(MIN_LOG_STD, MAX_LOG_STD);
            let latent = match mode {
                ReferenceEvaluationMode::Deterministic => mean,
                ReferenceEvaluationMode::Stochastic => mean + log_std.exp() * normal(rng) as f32,
            };
            // For diagnostics only, "force_switches" counts sign crossings,
            // not full-scale +/-20 N reversals as it does for categorical.
            (
                usize::from(latent > 0.0),
                latent,
                continuous_force_with_gain(latent, max_force, continuous_gain),
            )
        }
    }
}

pub fn force_for_action(action: usize, max_force: f32) -> f32 {
    match action {
        0 => -max_force,
        1 => max_force,
        _ => panic!("categorical cart-force action must be 0 or 1"),
    }
}

struct Lane {
    env: PendulumEnv,
    observation: [f32; INPUTS],
    environment_rng: StdRng,
    action_rng: StdRng,
}

#[derive(Clone)]
struct Sample {
    observation: [f32; INPUTS],
    action: usize,
    /// Pre-tanh continuous latent; zero and unused for categorical samples.
    action_latent: f32,
    old_log_prob: f32,
    old_value: f32,
    raw_reward: f32,
    done: bool,
    truncated: bool,
    /// V(final pre-reset observation) ONLY for artificial timeouts.
    final_value: f32,
}

fn gae_targets(
    rows: &[Sample],
    final_values: &[f32],
    environments: usize,
    steps: usize,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(rows.len(), environments * steps);
    assert_eq!(final_values.len(), environments);
    let mut advantages = vec![0.0; rows.len()];
    let mut returns = vec![0.0; rows.len()];
    for (lane, &final_value) in final_values.iter().enumerate() {
        let mut next_advantage = 0.0_f32;
        for t in (0..steps).rev() {
            let index = t * environments + lane;
            let row = &rows[index];
            let next_value = if row.done {
                0.0
            } else if t + 1 == steps {
                final_value
            } else {
                rows[index + environments].old_value
            };
            let corrected_reward = row.raw_reward
                + if row.truncated {
                    gamma * row.final_value
                } else {
                    0.0
                };
            let mask = if row.done { 0.0 } else { 1.0 };
            let delta = corrected_reward + gamma * mask * next_value - row.old_value;
            let advantage = delta + gamma * lambda * mask * next_advantage;
            advantages[index] = advantage;
            returns[index] = advantage + row.old_value;
            next_advantage = advantage;
        }
    }
    (returns, advantages)
}

fn tensor_scalars<B: Backend>(device: &B::Device, values: &[f32]) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len(), 1]), device)
}

fn masks<B: Backend>(device: &B::Device, actions: &[usize]) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(actions.len() * ACTIONS);
    for &action in actions {
        assert!(action < ACTIONS, "only left/right categorical actions");
        values.extend_from_slice(if action == 0 {
            &[1.0_f32, 0.0]
        } else {
            &[0.0, 1.0]
        });
    }
    Tensor::from_data(TensorData::new(values, [actions.len(), ACTIONS]), device)
}

fn mini_batch_advantages(src: &[f32]) -> Vec<f32> {
    let mean = src.iter().copied().sum::<f32>() / src.len() as f32;
    let variance = src.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (src.len() - 1) as f32; // torch.std: ddof=1
    src.iter()
        .map(|x| (x - mean) / (variance.sqrt() + 1e-8))
        .collect()
}

/// Elementwise maximum with PyTorch's half-gradient on exact ties.
/// Burn 0.20.1 max_pair uses a strict comparison and routes ties to self.
/// At a clipped-value boundary those have equal values but different gradients.
fn torch_maximum<B: Backend, const D: usize>(
    left: Tensor<B, D>,
    right: Tensor<B, D>,
) -> Tensor<B, D> {
    let weight = left.clone().greater(right.clone()).float()
        + left.clone().equal(right.clone()).float().mul_scalar(0.5);
    let other_weight = weight.clone().neg().add_scalar(1.0);
    left * weight + right * other_weight
}

/// Return selected-action log likelihood [N,1] and mean entropy [1].
///
/// For tanh-Gaussian: the Jacobian cancels in the PPO importance ratio when
/// the identical PRE-tanh action z is replayed. The entropy bonus is the
/// UNSQUASHED Gaussian's differential entropy, not exact tanh-action entropy.
/// It is a deliberate cross-distribution semantic difference and MUST NOT
/// be interpreted as an identical Shannon-entropy regularizer.
fn action_log_prob_and_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    device: &B::Device,
    actions: &[usize],
    latents: &[f32],
    distribution: ReferenceActionDistribution,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    match distribution {
        ReferenceActionDistribution::Categorical => {
            let log_probs = activation::log_softmax(logits, 1);
            let selected = (log_probs.clone() * masks(device, actions)).sum_dim(1);
            let entropy = (log_probs.clone().exp() * log_probs)
                .sum_dim(1)
                .mean()
                .mul_scalar(-1.0);
            (selected, entropy)
        }
        ReferenceActionDistribution::SquashedGaussian => {
            let n = latents.len();
            assert_eq!(actions.len(), n);
            let mean_mask = Tensor::<B, 2>::from_data(
                TensorData::new([1.0_f32, 0.0].repeat(n), [n, ACTIONS]),
                device,
            );
            let log_std_mask = Tensor::<B, 2>::from_data(
                TensorData::new([0.0_f32, 1.0].repeat(n), [n, ACTIONS]),
                device,
            );
            let mean = (logits.clone() * mean_mask).sum_dim(1);
            let log_std = (logits * log_std_mask)
                .sum_dim(1)
                .clamp(MIN_LOG_STD, MAX_LOG_STD);
            let centered = tensor_scalars(device, latents) - mean;
            let standardized = centered * log_std.clone().neg().exp();
            let log_prob = standardized.square().mul_scalar(-0.5) - log_std.clone();
            let log_prob = log_prob.sub_scalar(NORMAL_LOG_NORM);
            let entropy = log_std.add_scalar(GAUSSIAN_ENTROPY_BASE).mean();
            (log_prob, entropy)
        }
    }
}

/// Same clipped PPO surrogate, critic clip and GAE targets in both action
/// distributions; only log density and entropy of the action law change.
#[allow(clippy::too_many_arguments)] // Explicit operand contract shared with the numerical fixture.
fn ppo_objective_terms<B: Backend>(
    logits: Tensor<B, 2>,
    values: Tensor<B, 2>,
    device: &B::Device,
    actions: &[usize],
    action_latents: &[f32],
    distribution: ReferenceActionDistribution,
    old_log_probs: &[f32],
    old_values: &[f32],
    targets: &[f32],
    normalized_advantages: &[f32],
    clip_epsilon: f32,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let (new_log_prob, entropy) =
        action_log_prob_and_entropy(logits, device, actions, action_latents, distribution);
    let ratio = (new_log_prob - tensor_scalars(device, old_log_probs)).exp();
    let neg_advantages = tensor_scalars(device, normalized_advantages).mul_scalar(-1.0);
    let policy_loss = torch_maximum(
        ratio.clone() * neg_advantages.clone(),
        ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * neg_advantages,
    )
    .mean();

    let previous_values = tensor_scalars(device, old_values);
    let return_targets = tensor_scalars(device, targets);
    let unclipped = (values.clone() - return_targets.clone()).square();
    let clipped_predictions =
        previous_values.clone() + (values - previous_values).clamp(-clip_epsilon, clip_epsilon);
    let clipped = (clipped_predictions - return_targets).square();
    let value_loss = torch_maximum(unclipped, clipped).mean().mul_scalar(0.5);

    (policy_loss, value_loss, entropy)
}

/// Clip one global L2 norm across ALL actor+critic weights and biases.
/// Burn's built-in GradientClippingConfig::Norm clips individual parameter
/// tensors, which is not equivalent to CleanRL/PyTorch's clip_grad_norm_.
fn clip_global_gradients(
    model: &ActorCritic<AutodiffBackend>,
    mut gradients: GradientsParams,
    maximum_norm: f32,
) -> GradientsParams {
    use crate::backend::TrainBackend;

    let layers = [
        &model.actor.input,
        &model.actor.hidden,
        &model.actor.output,
        &model.critic.input,
        &model.critic.hidden,
        &model.critic.output,
    ];
    let mut squared_norm = 0.0_f64;
    for linear in layers {
        let weight = gradients
            .get::<TrainBackend, 2>(linear.weight.id)
            .expect("missing PPO weight gradient")
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let bias = gradients
            .get::<TrainBackend, 1>(linear.bias.as_ref().unwrap().id)
            .expect("missing PPO bias gradient")
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        squared_norm += weight
            .into_iter()
            .chain(bias)
            .map(|v| {
                let v = f64::from(v);
                v * v
            })
            .sum::<f64>();
    }
    assert!(squared_norm.is_finite(), "nonfinite PPO global gradient");
    let coefficient = (f64::from(maximum_norm) / (squared_norm.sqrt() + 1e-6)).min(1.0) as f32;
    for linear in layers {
        let weight_id = linear.weight.id;
        let bias_id = linear.bias.as_ref().unwrap().id;
        let weight = gradients
            .remove::<TrainBackend, 2>(weight_id)
            .expect("missing PPO weight gradient");
        gradients.register::<TrainBackend, 2>(weight_id, weight.mul_scalar(coefficient));
        let bias = gradients
            .remove::<TrainBackend, 1>(bias_id)
            .expect("missing PPO bias gradient");
        gradients.register::<TrainBackend, 1>(bias_id, bias.mul_scalar(coefficient));
    }
    gradients
}

/// Read-only evaluation modes. Evaluation RNG streams do not affect training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceEvaluationMode {
    Deterministic,
    Stochastic,
}

/// A single native Rust trajectory on a fixed, externally supplied key.
/// Results at different policy checkpoints must use the same keys and mode.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceEvaluationEpisode {
    pub key: u64,
    pub initial_state: [f32; INPUTS],
    pub initial_value: f32,
    pub steps: usize,
    pub completed: bool,
    pub terminated: bool,
    pub raw_return: f64,
    pub min_cart_clearance_m: f32,
    pub force_switches: usize,
}

/// Frozen, evaluation-only actuator variations. Neither optimizer updates nor
/// the original nominal evaluator depend on these transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceActuation {
    /// Original immediate +/-20 N command (before plant disturbances).
    Ideal,
    /// Apply the previous tick's policy command, initially 0 N.
    DelayOneTick,
    /// Apply one command for two consecutive 10-ms ticks.
    HoldTwoTicks,
    /// Limit each 10-ms tick to a 5 N change in actuator command.
    SlewFiveNewtonsPerTick,
    /// Evaluation-only corrective-force interventions, clipped to max force.
    BoostOneAndHalf,
    BoostTwo,
    BangBangSign,
}

impl ReferenceActuation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ideal => "ideal",
            Self::DelayOneTick => "delay_one_tick",
            Self::HoldTwoTicks => "hold_two_ticks",
            Self::SlewFiveNewtonsPerTick => "slew_five_n_per_tick",
            Self::BoostOneAndHalf => "boost_1p5",
            Self::BoostTwo => "boost_2p0",
            Self::BangBangSign => "bang_bang_sign",
        }
    }

    fn output_force(
        self,
        desired: f32,
        previous_output: f32,
        previous_desired: f32,
        tick: usize,
        max_force: f32,
    ) -> f32 {
        assert!(max_force.is_finite() && max_force > 0.0);
        match self {
            Self::Ideal => desired,
            Self::DelayOneTick => previous_desired,
            Self::HoldTwoTicks => {
                if tick.is_multiple_of(2) {
                    desired
                } else {
                    previous_output
                }
            }
            Self::SlewFiveNewtonsPerTick => {
                previous_output + (desired - previous_output).clamp(-5.0, 5.0)
            }
            Self::BoostOneAndHalf => (1.5 * desired).clamp(-max_force, max_force),
            Self::BoostTwo => (2.0 * desired).clamp(-max_force, max_force),
            Self::BangBangSign => {
                if desired > 0.0 {
                    max_force
                } else if desired < 0.0 {
                    -max_force
                } else {
                    0.0
                }
            }
        }
    }
}

/// Evaluation-only transition trace. The policy receives the noisy observation
/// and NEVER receives privileged true state or the realized disturbance force.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceEvaluationStep {
    pub key: u64,
    pub tick: usize,
    pub observation_before: [f32; INPUTS],
    pub true_state_before: [f32; INPUTS],
    pub desired_force_n: f32,
    pub actuator_force_n: f32,
    pub applied_force_n: f32,
    pub reward: f32,
    pub observation_after: [f32; INPUTS],
    pub true_state_after: [f32; INPUTS],
    pub rail_breached: bool,
    pub angle_breached: bool,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CleanRlReferenceMetrics {
    pub updates: usize,
    pub environment_steps: usize,
    pub episodes: usize,
    pub last_policy_loss: f32,
    pub last_value_loss: f32,
    pub last_entropy: f32,
    pub last_learning_rate: f64,
}

/// One learner with four independent physical rollout streams and a JOINT
/// optimizer. There is no replica averaging or implicit optimizer reset.
pub struct CleanRlReferenceTrainer {
    config: CleanRlReferenceConfig,
    device: AutodiffDevice,
    model: ActorCritic<AutodiffBackend>,
    optimizer: OptimizerAdaptor<burn::optim::Adam, ActorCritic<AutodiffBackend>, AutodiffBackend>,
    lanes: Vec<Lane>,
    optimizer_rng: StdRng,
    metrics: CleanRlReferenceMetrics,
}

impl CleanRlReferenceTrainer {
    pub fn new(config: CleanRlReferenceConfig) -> Self {
        config.validate();
        let device = Default::default();
        let mut root = StdRng::seed_from_u64(config.seed);
        let mut initialization_rng = StdRng::seed_from_u64(root.gen());
        let model = ActorCritic::new(&device, &mut initialization_rng); // identical 2-output architecture
        let mut lanes = Vec::with_capacity(config.environments);
        for _ in 0..config.environments {
            let mut environment_rng = StdRng::seed_from_u64(root.gen());
            let action_rng = StdRng::seed_from_u64(root.gen());
            let env = PendulumEnv::new_with_rng(
                config.plant.model(),
                config.environment,
                &mut environment_rng,
            );
            let observation = env.observation_with_rng(&mut environment_rng);
            lanes.push(Lane {
                env,
                observation,
                environment_rng,
                action_rng,
            });
        }
        let optimizer_rng = StdRng::seed_from_u64(root.gen());
        // Apply an explicit joint global norm to the full actor+critic gradient
        // before Adam. Burn's built-in Norm clips EACH tensor independently.
        let optimizer = AdamConfig::new().with_epsilon(1e-5).init();
        Self {
            config,
            device,
            model,
            optimizer,
            lanes,
            optimizer_rng,
            metrics: CleanRlReferenceMetrics::default(),
        }
    }

    pub fn metrics(&self) -> &CleanRlReferenceMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &CleanRlReferenceConfig {
        &self.config
    }

    /// Deterministic inference only; does not touch RNG, optimizer or live env.
    pub fn act_deterministic(&self, observation: [f32; INPUTS]) -> f32 {
        let valid = self.model.valid();
        let input = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(&self.device, &[observation]);
        let logits = valid
            .actor
            .forward(input)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        match self.config.action_distribution {
            ReferenceActionDistribution::Categorical => force_for_action(
                deterministic_action([logits[0], logits[1]]),
                self.config.environment.max_force,
            ),
            ReferenceActionDistribution::SquashedGaussian => continuous_force_with_gain(
                logits[0],
                self.config.environment.max_force,
                self.config.continuous_force_gain,
            ),
        }
    }

    /// Evaluate a frozen checkpoint on a caller-owned panel of reset keys.
    ///
    /// Uses the real noisy nonlinear native environment. Every frame evaluates
    /// ALL lanes in one batch, including completed ones, avoiding changes to
    /// the inference shape as episodes finish. Completed lanes are never
    /// physically stepped again. No training RNG, optimizer or live environment
    /// state is touched.
    pub fn evaluate_seeded(
        &self,
        keys: &[u64],
        horizon: usize,
        mode: ReferenceEvaluationMode,
    ) -> Vec<ReferenceEvaluationEpisode> {
        self.evaluate_seeded_at_gain(keys, horizon, mode, self.config.continuous_force_gain)
    }

    /// Read-only post-training evaluation with an ABSOLUTE Gaussian force gain,
    /// independent of the gain used during collection and policy updates.
    /// The native plant, physical reward, reset seeds, batch shape and RNG
    /// streams are identical to evaluate_seeded. No optimizer, model, or live
    /// training lane is modified. Categorical policies may only use gain=1.
    pub fn evaluate_seeded_at_gain(
        &self,
        keys: &[u64],
        horizon: usize,
        mode: ReferenceEvaluationMode,
        evaluation_gain: f32,
    ) -> Vec<ReferenceEvaluationEpisode> {
        assert!(!keys.is_empty() && horizon > 0);
        assert!(evaluation_gain.is_finite() && evaluation_gain > 0.0);
        assert!(
            self.config.action_distribution == ReferenceActionDistribution::SquashedGaussian
                || evaluation_gain == 1.0,
            "categorical evaluation must retain force gain 1"
        );
        struct EvalLane {
            key: u64,
            env: PendulumEnv,
            observation: [f32; INPUTS],
            environment_rng: StdRng,
            action_rng: StdRng,
            initial_state: [f32; INPUTS],
            initial_value: f32,
            steps: usize,
            done: bool,
            completed: bool,
            terminated: bool,
            raw_return: f64,
            min_cart_clearance_m: f32,
            force_switches: usize,
            previous_action: Option<usize>,
        }

        let mut env_config = self.config.environment;
        env_config.max_steps = horizon; // Same physical boundaries; new evaluation cap.
        let mut lanes = Vec::with_capacity(keys.len());
        for &key in keys {
            let mut root = StdRng::seed_from_u64(key);
            let mut environment_rng = StdRng::seed_from_u64(root.gen());
            let action_rng = StdRng::seed_from_u64(root.gen());
            let env = PendulumEnv::new_with_rng(
                self.config.plant.model(),
                env_config,
                &mut environment_rng,
            );
            let observation = env.observation_with_rng(&mut environment_rng);
            let state = env.state();
            lanes.push(EvalLane {
                key,
                env,
                observation,
                environment_rng,
                action_rng,
                initial_state: [state[0], state[1], state[2], state[3]],
                initial_value: self.value(observation),
                steps: 0,
                done: false,
                completed: false,
                terminated: false,
                raw_return: 0.0,
                min_cart_clearance_m: env_config.max_position_m - state[0].abs(),
                force_switches: 0,
                previous_action: None,
            });
        }

        let actor = self.model.valid().actor;
        for _ in 0..horizon {
            let observations = lanes
                .iter()
                .map(|lane| lane.observation)
                .collect::<Vec<_>>();
            let tensor = obs_tensor::<
                <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
            >(&self.device, &observations);
            let logits = actor.forward(tensor).to_data().to_vec::<f32>().unwrap();
            for (i, lane) in lanes.iter_mut().enumerate() {
                if lane.done {
                    continue;
                }
                let pair = [logits[i * ACTIONS], logits[i * ACTIONS + 1]];
                let (action, _latent, force) = choose_policy_force(
                    pair,
                    self.config.action_distribution,
                    mode,
                    env_config.max_force,
                    evaluation_gain,
                    &mut lane.action_rng,
                );
                if let Some(previous) = lane.previous_action {
                    lane.force_switches += usize::from(previous != action);
                }
                lane.previous_action = Some(action);
                let step = lane.env.step_with_rng(force, &mut lane.environment_rng);
                lane.steps += 1;
                lane.raw_return += f64::from(step.reward);
                lane.observation = step.observation;
                lane.min_cart_clearance_m = lane
                    .min_cart_clearance_m
                    .min(env_config.max_position_m - lane.env.state()[0].abs());
                if step.done {
                    lane.done = true;
                    lane.completed = step.truncated;
                    lane.terminated = step.terminated();
                }
            }
            if lanes.iter().all(|lane| lane.done) {
                break;
            }
        }
        lanes
            .into_iter()
            .map(|lane| ReferenceEvaluationEpisode {
                key: lane.key,
                initial_state: lane.initial_state,
                initial_value: lane.initial_value,
                steps: lane.steps,
                completed: lane.completed,
                terminated: lane.terminated,
                raw_return: lane.raw_return,
                min_cart_clearance_m: lane.min_cart_clearance_m,
                force_switches: lane.force_switches,
            })
            .collect()
    }

    /// Audit-only evaluation: identical seeded 64-lane inference contract,
    /// with an explicit actuator transform and optional per-step witness trace.
    /// Does not modify any optimizer, training environment or RNG state.
    pub fn evaluate_seeded_with_actuator_trace(
        &self,
        keys: &[u64],
        horizon: usize,
        mode: ReferenceEvaluationMode,
        actuator: ReferenceActuation,
        trace_key: Option<u64>,
    ) -> (
        Vec<ReferenceEvaluationEpisode>,
        Vec<ReferenceEvaluationStep>,
    ) {
        assert!(!keys.is_empty() && horizon > 0);
        let mut traces = Vec::new();
        struct EvalLane {
            key: u64,
            env: PendulumEnv,
            observation: [f32; INPUTS],
            environment_rng: StdRng,
            action_rng: StdRng,
            initial_state: [f32; INPUTS],
            initial_value: f32,
            steps: usize,
            done: bool,
            completed: bool,
            terminated: bool,
            raw_return: f64,
            min_cart_clearance_m: f32,
            force_switches: usize,
            previous_action: Option<usize>,
            output_force_n: f32,
            queued_desired_force_n: f32,
        }

        let mut env_config = self.config.environment;
        env_config.max_steps = horizon; // Same physical boundaries; new evaluation cap.
        let mut lanes = Vec::with_capacity(keys.len());
        for &key in keys {
            let mut root = StdRng::seed_from_u64(key);
            let mut environment_rng = StdRng::seed_from_u64(root.gen());
            let action_rng = StdRng::seed_from_u64(root.gen());
            let env = PendulumEnv::new_with_rng(
                self.config.plant.model(),
                env_config,
                &mut environment_rng,
            );
            let observation = env.observation_with_rng(&mut environment_rng);
            let state = env.state();
            lanes.push(EvalLane {
                key,
                env,
                observation,
                environment_rng,
                action_rng,
                initial_state: [state[0], state[1], state[2], state[3]],
                initial_value: self.value(observation),
                steps: 0,
                done: false,
                completed: false,
                terminated: false,
                raw_return: 0.0,
                min_cart_clearance_m: env_config.max_position_m - state[0].abs(),
                force_switches: 0,
                previous_action: None,
                output_force_n: 0.0,
                queued_desired_force_n: 0.0,
            });
        }

        let actor = self.model.valid().actor;
        for _ in 0..horizon {
            let observations = lanes
                .iter()
                .map(|lane| lane.observation)
                .collect::<Vec<_>>();
            let tensor = obs_tensor::<
                <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
            >(&self.device, &observations);
            let logits = actor.forward(tensor).to_data().to_vec::<f32>().unwrap();
            for (i, lane) in lanes.iter_mut().enumerate() {
                if lane.done {
                    continue;
                }
                let pair = [logits[i * ACTIONS], logits[i * ACTIONS + 1]];
                let (action, _latent, force) = choose_policy_force(
                    pair,
                    self.config.action_distribution,
                    mode,
                    env_config.max_force,
                    self.config.continuous_force_gain,
                    &mut lane.action_rng,
                );
                if let Some(previous) = lane.previous_action {
                    lane.force_switches += usize::from(previous != action);
                }
                lane.previous_action = Some(action);
                let desired_force_n = force;
                let observation_before = lane.observation;
                let before = lane.env.state();
                let actuator_force_n = actuator.output_force(
                    desired_force_n,
                    lane.output_force_n,
                    lane.queued_desired_force_n,
                    lane.steps,
                    env_config.max_force,
                );
                lane.output_force_n = actuator_force_n;
                lane.queued_desired_force_n = desired_force_n;
                let step = lane
                    .env
                    .step_with_rng(actuator_force_n, &mut lane.environment_rng);
                let after = lane.env.state();
                lane.steps += 1;
                if trace_key == Some(lane.key) {
                    traces.push(ReferenceEvaluationStep {
                        key: lane.key,
                        tick: lane.steps,
                        observation_before,
                        true_state_before: [before[0], before[1], before[2], before[3]],
                        desired_force_n,
                        actuator_force_n,
                        applied_force_n: lane.env.last_applied_force(),
                        reward: step.reward,
                        observation_after: step.observation,
                        true_state_after: [after[0], after[1], after[2], after[3]],
                        rail_breached: after[0].abs() > env_config.max_position_m,
                        angle_breached: after[2].abs() > env_config.max_angle_rad,
                        terminated: step.terminated(),
                        truncated: step.truncated,
                    });
                }
                lane.raw_return += f64::from(step.reward);
                lane.observation = step.observation;
                lane.min_cart_clearance_m = lane
                    .min_cart_clearance_m
                    .min(env_config.max_position_m - lane.env.state()[0].abs());
                if step.done {
                    lane.done = true;
                    lane.completed = step.truncated;
                    lane.terminated = step.terminated();
                }
            }
            if lanes.iter().all(|lane| lane.done) {
                break;
            }
        }
        let summaries = lanes
            .into_iter()
            .map(|lane| ReferenceEvaluationEpisode {
                key: lane.key,
                initial_state: lane.initial_state,
                initial_value: lane.initial_value,
                steps: lane.steps,
                completed: lane.completed,
                terminated: lane.terminated,
                raw_return: lane.raw_return,
                min_cart_clearance_m: lane.min_cart_clearance_m,
                force_switches: lane.force_switches,
            })
            .collect::<Vec<_>>();
        (summaries, traces)
    }

    fn value(&self, observation: [f32; INPUTS]) -> f32 {
        let valid = self.model.valid();
        let input = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(&self.device, &[observation]);
        valid
            .critic
            .forward(input)
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0]
    }

    pub fn train_updates(&mut self, updates: usize) {
        assert!(
            self.metrics.updates + updates <= self.config.total_updates,
            "CleanRL reference total training budget exceeded"
        );
        for _ in 0..updates {
            let (rows, returns, advantages) = self.collect();
            self.optimize(&rows, &returns, &advantages);
            self.metrics.updates += 1;
        }
    }

    fn collect(&mut self) -> (Vec<Sample>, Vec<f32>, Vec<f32>) {
        let lanes = self.lanes.len();
        let steps = self.config.steps_per_environment;
        let valid = self.model.valid(); // Frozen actor and critic throughout collection.
        let mut rows = Vec::with_capacity(lanes * steps);
        for _ in 0..steps {
            let observations = self.lanes.iter().map(|l| l.observation).collect::<Vec<_>>();
            let inputs = obs_tensor::<
                <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
            >(&self.device, &observations);
            let policy_logits = valid.actor.forward(inputs.clone());
            let logits = policy_logits.clone().to_data().to_vec::<f32>().unwrap();
            // Sample actions first; use the exact same tensor likelihood
            // arithmetic in collection and replay for both distributions.
            let sampled = self
                .lanes
                .iter_mut()
                .enumerate()
                .map(|(i, lane)| {
                    choose_policy_force(
                        [logits[2 * i], logits[2 * i + 1]],
                        self.config.action_distribution,
                        ReferenceEvaluationMode::Stochastic,
                        self.config.environment.max_force,
                        self.config.continuous_force_gain,
                        &mut lane.action_rng,
                    )
                })
                .collect::<Vec<_>>();
            let actions = sampled.iter().map(|x| x.0).collect::<Vec<_>>();
            let latents = sampled.iter().map(|x| x.1).collect::<Vec<_>>();
            let (log_prob_tensor, _) = action_log_prob_and_entropy(
                policy_logits,
                &self.device,
                &actions,
                &latents,
                self.config.action_distribution,
            );
            let log_probs = log_prob_tensor.to_data().to_vec::<f32>().unwrap();
            let values = valid
                .critic
                .forward(inputs)
                .to_data()
                .to_vec::<f32>()
                .unwrap();
            for (i, lane) in self.lanes.iter_mut().enumerate() {
                let (action, action_latent, force) = sampled[i];
                let log_prob = log_probs[i];
                let step = lane.env.step_with_rng(force, &mut lane.environment_rng);
                let final_value = if step.truncated {
                    let input = obs_tensor::<
                        <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
                    >(&self.device, &[step.observation]);
                    valid
                        .critic
                        .forward(input)
                        .to_data()
                        .to_vec::<f32>()
                        .unwrap()[0]
                } else {
                    0.0
                };
                rows.push(Sample {
                    observation: lane.observation,
                    action,
                    action_latent,
                    old_log_prob: log_prob,
                    old_value: values[i],
                    raw_reward: step.reward,
                    done: step.done,
                    truncated: step.truncated,
                    final_value,
                });
                self.metrics.environment_steps += 1;
                lane.observation = if step.done {
                    self.metrics.episodes += 1;
                    lane.env.reset_with_rng(&mut lane.environment_rng)
                } else {
                    step.observation
                };
            }
        }
        let observations = self.lanes.iter().map(|l| l.observation).collect::<Vec<_>>();
        let values = valid
            .critic
            .forward(obs_tensor::<
                <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
            >(&self.device, &observations))
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let (returns, advantages) = gae_targets(
            &rows,
            &values,
            lanes,
            steps,
            self.config.gamma,
            self.config.gae_lambda,
        );
        (rows, returns, advantages)
    }

    fn optimize(&mut self, rows: &[Sample], returns: &[f32], advantages: &[f32]) {
        let mut permutation = (0..rows.len()).collect::<Vec<_>>();
        let learning_rate = self.config.rate_for_update(self.metrics.updates);
        for _ in 0..self.config.epochs {
            permutation.shuffle(&mut self.optimizer_rng);
            for batch in permutation.chunks(self.config.minibatch_size) {
                let observations = batch
                    .iter()
                    .map(|&i| rows[i].observation)
                    .collect::<Vec<_>>();
                let actions = batch.iter().map(|&i| rows[i].action).collect::<Vec<_>>();
                let latents = batch
                    .iter()
                    .map(|&i| rows[i].action_latent)
                    .collect::<Vec<_>>();
                let old_log_probs = batch
                    .iter()
                    .map(|&i| rows[i].old_log_prob)
                    .collect::<Vec<_>>();
                let old_values = batch.iter().map(|&i| rows[i].old_value).collect::<Vec<_>>();
                let targets = batch.iter().map(|&i| returns[i]).collect::<Vec<_>>();
                let batch_adv = batch.iter().map(|&i| advantages[i]).collect::<Vec<_>>();
                let normalized_adv = mini_batch_advantages(&batch_adv);

                let input = obs_tensor::<AutodiffBackend>(&self.device, &observations);
                let logits = self.model.actor.forward(input.clone());
                let values = self.model.critic.forward(input);
                let (policy_loss, value_loss, entropy) = ppo_objective_terms(
                    logits,
                    values,
                    &self.device,
                    &actions,
                    &latents,
                    self.config.action_distribution,
                    &old_log_probs,
                    &old_values,
                    &targets,
                    &normalized_adv,
                    self.config.clip_epsilon,
                );
                self.metrics.last_policy_loss =
                    policy_loss.clone().to_data().to_vec::<f32>().unwrap()[0];
                self.metrics.last_value_loss =
                    value_loss.clone().to_data().to_vec::<f32>().unwrap()[0];
                self.metrics.last_entropy = entropy.clone().to_data().to_vec::<f32>().unwrap()[0];
                let loss = policy_loss + value_loss.mul_scalar(self.config.value_coefficient)
                    - entropy.mul_scalar(self.config.entropy_coefficient);
                let grads = GradientsParams::from_grads(loss.backward(), &self.model);
                let grads =
                    clip_global_gradients(&self.model, grads, self.config.maximum_gradient_norm);
                self.model = self
                    .optimizer
                    .step(learning_rate, self.model.clone(), grads);
            }
        }
        self.metrics.last_learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(value: f32, reward: f32, done: bool, truncated: bool, final_value: f32) -> Sample {
        Sample {
            observation: [0.0; INPUTS],
            action: 0,
            action_latent: 0.0,
            old_log_prob: 0.0,
            old_value: value,
            raw_reward: reward,
            done,
            truncated,
            final_value,
        }
    }

    #[test]
    fn categorical_probability_force_contract() {
        assert!((left_probability([0.0, 0.0]) - 0.5).abs() < 1e-6);
        assert!(left_probability([50.0, -50.0]) > 0.999);
        assert_eq!(force_for_action(0, 20.0), -20.0);
        assert_eq!(force_for_action(1, 20.0), 20.0);
        assert!((left_probability([1.0, 2.0]) + left_probability([2.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "categorical cart-force")]
    fn invalid_category_is_rejected() {
        force_for_action(2, 20.0);
    }

    #[test]
    fn bootstrap_uses_final_state_not_reset_and_never_leaks() {
        // A two-step array containing a timeout/reset boundary. Old values=2.
        // Terminal path gets gamma*V(final)=.9*8, never V(reset)=1000.
        let rows = [
            example(2.0, 1.0, true, true, 8.0),
            example(2.0, 2.0, true, false, 999.0),
        ];
        let (ret, adv) = gae_targets(&rows, &[1000.0], 1, 2, 0.9, 0.95);
        assert!((ret[0] - 8.2).abs() < 1e-5);
        assert!((adv[0] - 6.2).abs() < 1e-5);
        assert!((ret[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn unfinished_rollout_bootstraps_but_does_not_terminate_episode() {
        let rows = [example(2.0, 1.0, false, false, 0.0)];
        let (ret, _) = gae_targets(&rows, &[3.0], 1, 1, 0.9, 0.95);
        assert!((ret[0] - 3.7).abs() < 1e-5);
    }

    #[test]
    fn minibatch_advantage_normalization_is_sample_std() {
        let result = mini_batch_advantages(&[1.0, 3.0, 5.0, 7.0]);
        let expected = [-1.161895, -0.387298, 0.387298, 1.161895];
        for (a, b) in result.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn fixed_annealing_clock_does_not_restart_between_updates() {
        let c = CleanRlReferenceConfig::default();
        assert_eq!(c.total_updates, 976);
        assert_eq!(c.rate_for_update(0), 2.5e-4);
        assert!((c.rate_for_update(975) - 2.5e-4 / 976.0).abs() < 1e-12);
    }

    #[test]
    fn orthogonal_initialization_has_correct_input_output_scale() {
        let mut rng = StdRng::seed_from_u64(77);
        let model = orthogonal_linear::<crate::backend::TrainBackend, _>(
            &Default::default(),
            4,
            64,
            2.0_f32.sqrt(),
            &mut rng,
        );
        let data = model.weight.val().to_data().to_vec::<f32>().unwrap();
        for a in 0..4 {
            for b in 0..4 {
                let dot: f32 = (0..64).map(|j| data[a * 64 + j] * data[b * 64 + j]).sum();
                let expected = if a == b { 2.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn global_gradient_clip_scales_actor_and_critic_together() {
        use crate::backend::TrainBackend;
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(13091);
        let model = ActorCritic::<AutodiffBackend>::new(&device, &mut rng);
        let mut gradients = GradientsParams::new();
        let layers = [
            &model.actor.input,
            &model.actor.hidden,
            &model.actor.output,
            &model.critic.input,
            &model.critic.hidden,
            &model.critic.output,
        ];
        let mut total_squared = 0.0_f64;
        for (index, layer) in layers.iter().enumerate() {
            let gain = (index + 1) as f32;
            let shape: [usize; 2] = layer.weight.shape().dims();
            let weights = shape[0] * shape[1];
            let [biases]: [usize; 1] = layer.bias.as_ref().unwrap().shape().dims();
            total_squared += f64::from(gain * gain) * (weights + biases) as f64;
            let weight = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(vec![gain; weights], shape),
                &device,
            );
            let bias = Tensor::<TrainBackend, 1>::from_data(
                TensorData::new(vec![gain; biases], [biases]),
                &device,
            );
            gradients.register::<TrainBackend, 2>(layer.weight.id, weight);
            gradients.register::<TrainBackend, 1>(layer.bias.as_ref().unwrap().id, bias);
        }
        assert_eq!(gradients.len(), 12);
        let maximum = 0.5_f32;
        let coefficient = (f64::from(maximum) / (total_squared.sqrt() + 1e-6)).min(1.0);
        let clipped = clip_global_gradients(&model, gradients, maximum);
        assert_eq!(clipped.len(), 12);
        let mut new_squared = 0.0_f64;
        for (index, layer) in layers.iter().enumerate() {
            let expected = ((index + 1) as f64 * coefficient) as f32;
            let w = clipped
                .get::<TrainBackend, 2>(layer.weight.id)
                .unwrap()
                .to_data()
                .to_vec::<f32>()
                .unwrap();
            let b = clipped
                .get::<TrainBackend, 1>(layer.bias.as_ref().unwrap().id)
                .unwrap()
                .to_data()
                .to_vec::<f32>()
                .unwrap();
            for x in w.into_iter().chain(b) {
                assert!((x - expected).abs() < 1e-6, "joint clip was not uniform");
                new_squared += f64::from(x) * f64::from(x);
            }
        }
        assert!((new_squared.sqrt() - f64::from(maximum)).abs() < 1e-5);
    }

    /// Pinned float32 Torch 2.10 reference computed using CleanRL's
    /// Categorical logits, clipped ratio, value clipping and entropy formula.
    #[test]
    fn clipped_ppo_objective_matches_independent_pytorch_fixture() {
        let device = Default::default();
        let logits = Tensor::<crate::backend::TrainBackend, 2>::from_data(
            TensorData::new(vec![0.25, -0.8, -0.3, 0.4, 1.2, -0.6, -0.8, 0.4], [4, 2]),
            &device,
        );
        let values = Tensor::<crate::backend::TrainBackend, 2>::from_data(
            TensorData::new(vec![0.9, -0.3, -0.5, 1.8], [4, 1]),
            &device,
        );
        let advantages = mini_batch_advantages(&[-2.0, -0.4, 0.7, 1.2]);
        let (pg, vf, entropy) = ppo_objective_terms(
            logits,
            values,
            &device,
            &[0, 1, 1, 0],
            &[0.0; 4],
            ReferenceActionDistribution::Categorical,
            &[-0.8, -0.1, -0.7, -1.4],
            &[0.3, -0.5, 0.1, 2.0],
            &[2.0, -1.0, 0.8, 0.0],
            &advantages,
            0.2,
        );
        let scalar = |value: Tensor<crate::backend::TrainBackend, 1>| {
            value.to_data().to_vec::<f32>().unwrap()[0]
        };
        let pg = scalar(pg);
        let vf = scalar(vf);
        let entropy = scalar(entropy);
        assert!((pg - 0.32307282).abs() < 1e-5, "clipped actor surrogate");
        assert!((vf - 0.95874995).abs() < 1e-5, "clipped critic target");
        assert!((entropy - 0.53926545).abs() < 1e-5, "categorical entropy");
        assert!((pg + 0.5 * vf - 0.01 * entropy - 0.797_055_1).abs() < 1e-5);
    }

    #[test]
    fn true_failure_and_timeout_do_not_share_bootstrap_rules() {
        let rows = [
            example(2.0, -10.0, true, false, 999.0),
            example(2.0, 1.0, true, true, 3.0),
            example(2.0, 2.0, false, false, 999.0),
        ];
        // First physical failure has no tail value; second timeout uses the
        // actual final value 3.0 and not the following 2.0 old value.
        let (ret, _) = gae_targets(&rows, &[4.0], 1, 3, 0.9, 0.95);
        assert!((ret[0] - (-10.0)).abs() < 1e-6);
        assert!((ret[1] - 3.7).abs() < 1e-5);
        assert!((ret[2] - 5.6).abs() < 1e-5);
    }

    #[test]
    fn native_evaluation_is_reproducible_and_does_not_mutate_training() {
        let config = CleanRlReferenceConfig::default();
        let trainer = CleanRlReferenceTrainer::new(config);
        let before_action = trainer.act_deterministic([0.0; INPUTS]);
        let keys = [55_000_001_u64, 55_000_002, 55_000_003];
        for mode in [
            ReferenceEvaluationMode::Deterministic,
            ReferenceEvaluationMode::Stochastic,
        ] {
            let original = trainer.evaluate_seeded(&keys, 24, mode);
            let repeat = trainer.evaluate_seeded(&keys, 24, mode);
            assert_eq!(original, repeat, "evaluation changed training or RNG");
            assert_eq!(original.len(), keys.len());
            for (row, &key) in original.iter().zip(&keys) {
                assert_eq!(row.key, key);
                assert!(row.steps > 0 && row.steps <= 24);
                assert_eq!(row.completed, !row.terminated);
                assert!(row.raw_return.is_finite());
                assert!(row.initial_value.is_finite());
                assert!(row.min_cart_clearance_m.is_finite());
                assert!(row.force_switches < row.steps);
            }
        }
        assert_eq!(trainer.metrics().environment_steps, 0);
        assert_eq!(trainer.metrics().updates, 0);
        assert_eq!(trainer.act_deterministic([0.0; INPUTS]), before_action);
    }

    #[test]
    fn evaluation_actuator_transform_has_frozen_step_semantics() {
        let ideal = ReferenceActuation::Ideal;
        assert_eq!(ideal.output_force(20.0, -20.0, -20.0, 1, 20.0), 20.0);
        let delay = ReferenceActuation::DelayOneTick;
        assert_eq!(delay.output_force(20.0, 0.0, 0.0, 0, 20.0), 0.0);
        assert_eq!(delay.output_force(-20.0, 20.0, 20.0, 1, 20.0), 20.0);
        let hold = ReferenceActuation::HoldTwoTicks;
        assert_eq!(hold.output_force(20.0, 0.0, 0.0, 0, 20.0), 20.0);
        assert_eq!(hold.output_force(-20.0, 20.0, 20.0, 1, 20.0), 20.0);
        assert_eq!(hold.output_force(-20.0, 20.0, 20.0, 2, 20.0), -20.0);
        let slew = ReferenceActuation::SlewFiveNewtonsPerTick;
        assert_eq!(slew.output_force(20.0, 0.0, 0.0, 0, 20.0), 5.0);
        assert_eq!(slew.output_force(-20.0, 5.0, 20.0, 1, 20.0), 0.0);
        let boost = ReferenceActuation::BoostOneAndHalf;
        assert_eq!(boost.output_force(12.0, 0.0, 0.0, 0, 20.0), 18.0);
        assert_eq!(boost.output_force(-16.0, 0.0, 0.0, 0, 20.0), -20.0);
        let double = ReferenceActuation::BoostTwo;
        assert_eq!(double.output_force(7.0, 0.0, 0.0, 0, 20.0), 14.0);
        assert_eq!(double.output_force(-12.0, 0.0, 0.0, 0, 20.0), -20.0);
        let sign = ReferenceActuation::BangBangSign;
        assert_eq!(sign.output_force(0.01, 0.0, 0.0, 0, 20.0), 20.0);
        assert_eq!(sign.output_force(-0.01, 0.0, 0.0, 0, 20.0), -20.0);
        assert_eq!(sign.output_force(0.0, 0.0, 0.0, 0, 20.0), 0.0);
    }

    #[test]
    fn ideal_instrumented_evaluation_matches_original_episode_exactly() {
        let config = CleanRlReferenceConfig::default();
        let trainer = CleanRlReferenceTrainer::new(config.clone());
        let keys = [4_100_100_100_u64, 4_100_100_101, 4_100_100_102];
        for mode in [
            ReferenceEvaluationMode::Deterministic,
            ReferenceEvaluationMode::Stochastic,
        ] {
            let legacy = trainer.evaluate_seeded(&keys, 24, mode);
            let (audited, trace) = trainer.evaluate_seeded_with_actuator_trace(
                &keys,
                24,
                mode,
                ReferenceActuation::Ideal,
                Some(keys[0]),
            );
            assert_eq!(legacy, audited);
            assert_eq!(trace.len(), audited[0].steps);
            for (index, step) in trace.iter().enumerate() {
                assert_eq!(step.key, keys[0]);
                assert_eq!(step.tick, index + 1);
                assert_eq!(step.desired_force_n, step.actuator_force_n);
                assert!(step.applied_force_n.is_finite());
                assert!(step.true_state_after.iter().all(|v| v.is_finite()));
                assert_eq!(
                    step.rail_breached,
                    step.true_state_after[0].abs() > config.environment.max_position_m
                );
                assert_eq!(
                    step.angle_breached,
                    step.true_state_after[2].abs() > config.environment.max_angle_rad
                );
            }
        }
        assert_eq!(trainer.metrics().updates, 0);
    }

    #[test]
    fn squashed_gaussian_log_density_and_bounds_match_scalar_reference() {
        use crate::backend::TrainBackend;
        let device = Default::default();
        let logits = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(vec![0.2_f32, -0.5, -0.3, 0.4], [2, 2]),
            &device,
        );
        let z = [0.7_f32, -1.1];
        let (log_probs, entropy) = action_log_prob_and_entropy(
            logits,
            &device,
            &[0, 1],
            &z,
            ReferenceActionDistribution::SquashedGaussian,
        );
        let measured = log_probs.to_data().to_vec::<f32>().unwrap();
        for (i, &(mu, log_std)) in [(0.2_f32, -0.5_f32), (-0.3_f32, 0.4_f32)]
            .iter()
            .enumerate()
        {
            let expected = -0.5 * ((z[i] - mu) / log_std.exp()).powi(2) - log_std - NORMAL_LOG_NORM;
            assert!((measured[i] - expected).abs() < 1e-5);
        }
        let expected_entropy = GAUSSIAN_ENTROPY_BASE + (-0.5 + 0.4) / 2.0;
        assert!((entropy.to_data().to_vec::<f32>().unwrap()[0] - expected_entropy).abs() < 1e-5);
        for latent in [-100.0_f32, -2.0, 0.0, 0.5, 1.0, 100.0] {
            let force = continuous_force(latent, 20.0);
            assert!((-20.0..=20.0).contains(&force));
        }
        let ratio = measured
            .iter()
            .zip(z)
            .enumerate()
            .map(|(i, (&old_log_prob, latent))| {
                let mu = if i == 0 { 0.2_f32 } else { -0.3 };
                let log_std = if i == 0 { -0.5_f32 } else { 0.4 };
                let replay =
                    -0.5 * ((latent - mu) / log_std.exp()).powi(2) - log_std - NORMAL_LOG_NORM;
                (replay - old_log_prob).exp()
            })
            .collect::<Vec<_>>();
        for r in ratio {
            assert!((r - 1.0).abs() < 1e-5, "unchanged Gaussian policy ratio");
        }
    }

    #[test]
    fn squashed_gaussian_two_update_native_learning_smoke() {
        let mut config = CleanRlReferenceConfig {
            action_distribution: ReferenceActionDistribution::SquashedGaussian,
            environments: 2,
            steps_per_environment: 4,
            minibatch_size: 4,
            epochs: 1,
            total_updates: 2,
            ..Default::default()
        };
        config.environment.max_steps = 3;
        let mut trainer = CleanRlReferenceTrainer::new(config);
        for obs in [[0.0; 4], [0.2, 0.1, -0.2, -0.1]] {
            let a = trainer.act_deterministic(obs);
            assert!(a.is_finite() && a.abs() <= 20.0);
        }
        trainer.train_updates(2);
        assert_eq!(trainer.metrics().updates, 2);
        assert_eq!(trainer.metrics().environment_steps, 16);
        assert!(trainer.metrics().last_policy_loss.is_finite());
        assert!(trainer.metrics().last_value_loss.is_finite());
        assert!(trainer.metrics().last_entropy.is_finite());
        let rows = trainer.evaluate_seeded(
            &[4_000_100_000_u64, 4_100_100_100],
            24,
            ReferenceEvaluationMode::Stochastic,
        );
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.raw_return.is_finite()));
    }

    #[test]
    fn bounded_continuous_force_gain_matches_nominal_map_at_one() {
        for latent in [-3.0_f32, -0.8, -0.1, 0.0, 0.1, 0.8, 3.0] {
            let force = continuous_force(latent, 20.0);
            assert_eq!(continuous_force_with_gain(latent, 20.0, 1.0), force);
            assert_eq!(
                continuous_force_with_gain(latent, 20.0, 1.5),
                (force * 1.5).clamp(-20.0, 20.0)
            );
            assert_eq!(
                continuous_force_with_gain(latent, 20.0, 2.0),
                (force * 2.0).clamp(-20.0, 20.0)
            );
        }
    }

    #[test]
    fn train_and_evaluate_with_gained_gaussian_is_bounded_and_finite() {
        let mut config = CleanRlReferenceConfig {
            action_distribution: ReferenceActionDistribution::SquashedGaussian,
            continuous_force_gain: 2.0,
            environments: 2,
            steps_per_environment: 4,
            minibatch_size: 4,
            epochs: 1,
            total_updates: 2,
            ..Default::default()
        };
        config.environment.max_steps = 3;
        let mut trainer = CleanRlReferenceTrainer::new(config);
        let before = trainer.act_deterministic([0.0; INPUTS]);
        assert!(before.is_finite() && (-20.0..=20.0).contains(&before));
        trainer.train_updates(2);
        assert_eq!(trainer.metrics().updates, 2);
        assert_eq!(trainer.metrics().environment_steps, 16);
        let rows = trainer.evaluate_seeded(
            &[4_000_100_000_u64, 4_100_100_100],
            24,
            ReferenceEvaluationMode::Stochastic,
        );
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.raw_return.is_finite()));
    }

    #[test]
    fn absolute_evaluation_gain_preserves_frozen_nominal_outcomes() {
        let keys = [4_000_100_000_u64, 4_100_100_100, 4_100_100_015];
        for gain in [1.0_f32, 1.5, 2.0] {
            let config = CleanRlReferenceConfig {
                action_distribution: ReferenceActionDistribution::SquashedGaussian,
                continuous_force_gain: gain,
                ..Default::default()
            };
            let trainer = CleanRlReferenceTrainer::new(config);
            for mode in [
                ReferenceEvaluationMode::Deterministic,
                ReferenceEvaluationMode::Stochastic,
            ] {
                let nominal = trainer.evaluate_seeded(&keys, 24, mode);
                let equivalent = trainer.evaluate_seeded_at_gain(&keys, 24, mode, gain);
                assert_eq!(nominal, equivalent, "same gain changed evaluator");
                let counterfactual = trainer.evaluate_seeded_at_gain(&keys, 24, mode, 2.0);
                assert_eq!(counterfactual.len(), keys.len());
                assert!(counterfactual.iter().all(|row| row.raw_return.is_finite()));
            }
            assert_eq!(trainer.metrics().updates, 0);
            assert_eq!(trainer.metrics().environment_steps, 0);
        }
    }

    #[test]
    fn small_native_training_session_preserves_real_physics_and_finite_policy() {
        let mut c = CleanRlReferenceConfig {
            environments: 2,
            steps_per_environment: 4,
            minibatch_size: 4,
            epochs: 1,
            total_updates: 2,
            ..Default::default()
        };
        c.environment.max_steps = 3; // Exercise timeouts during actual training.
        let mut trainer = CleanRlReferenceTrainer::new(c);
        assert_eq!(trainer.metrics().environment_steps, 0);
        let action = trainer.act_deterministic([0.0; 4]);
        assert!(action == 20.0 || action == -20.0);
        trainer.train_updates(2);
        assert_eq!(trainer.metrics().updates, 2);
        assert_eq!(trainer.metrics().environment_steps, 16);
        assert!(trainer.metrics().episodes > 0);
        assert!(trainer.metrics().last_policy_loss.is_finite());
        assert!(trainer.metrics().last_value_loss.is_finite());
    }
}

#[cfg(test)]
#[path = "cleanrl_reference_parity_tests.rs"]
mod parity_tests;
