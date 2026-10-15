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
    optim::{AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Tensor, TensorData, activation, backend::{AutodiffBackend as _, Backend}},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rust_robotics_algo::cart_pole::CartPoleParameters;

const INPUTS: usize = 4;
const ACTIONS: usize = 2;

/// Fixed, explicit experiment parameters. The real native plant/reward/noise
/// model is always used; these are not standard-Gym CartPole dynamics.
#[derive(Debug, Clone)]
pub struct CleanRlReferenceConfig {
    pub plant: CartPoleParameters,
    pub environment: PendulumEnvConfig,
    pub seed: u64,
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
        let mut environment = PendulumEnvConfig::default();
        environment.max_steps = 256; // An external collection cutoff, not failure.
        Self {
            plant: CartPoleParameters::default(),
            environment,
            seed: 201,
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
        let pool = self.environments.checked_mul(self.steps_per_environment)
            .expect("PPO pool size overflows");
        assert!(self.minibatch_size > 1 && pool % self.minibatch_size == 0);
        assert!(self.epochs > 0 && self.total_updates > 0);
        assert!(self.learning_rate.is_finite() && self.learning_rate > 0.0);
        assert!(self.maximum_gradient_norm.is_finite() && self.maximum_gradient_norm > 0.0);
        assert!(self.gamma.is_finite() && (0.0..=1.0).contains(&self.gamma));
        assert!(self.gae_lambda.is_finite() && (0.0..=1.0).contains(&self.gae_lambda));
        assert!(self.clip_epsilon.is_finite() && self.clip_epsilon > 0.0);
        assert!(self.entropy_coefficient.is_finite() && self.entropy_coefficient >= 0.0);
        assert!(self.value_coefficient.is_finite() && self.value_coefficient >= 0.0);
        assert!(self.environment.max_force.is_finite() && self.environment.max_force > 0.0);
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
    fn new<R: Rng + ?Sized>(device: &B::Device, output_dim: usize, output_gain: f32, rng: &mut R) -> Self {
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
    device: &B::Device, inputs: usize, outputs: usize, gain: f32, rng: &mut R,
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
                for x in &mut v { *x /= norm; }
                basis = Some(v);
                break;
            }
        }
        columns.push(basis.expect("unable to generate orthogonal weights"));
    }
    let mut weight = Vec::with_capacity(inputs * outputs);
    for input in 0..inputs {
        for output in 0..outputs {
            let element = if outputs >= inputs {
                columns[input][output]
            } else {
                columns[output][input]
            };
            weight.push((gain as f64 * element) as f32);
        }
    }
    Linear {
        weight: Param::from_data(TensorData::new(weight, [inputs, outputs]), device),
        bias: Some(Param::from_data(TensorData::new(vec![0.0_f32; outputs], [outputs]), device)),
    }
}

fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u = (1.0 - rng.gen::<f64>()).max(f64::MIN_POSITIVE);
    let v = rng.gen::<f64>();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Numerically stable categorical probabilities. A 0.5/0.5 float32 tie
/// deterministically chooses the LEFT category, as in torch.argmax.
pub fn left_probability(logits: [f32; 2]) -> f32 {
    assert!(logits.iter().all(|x| x.is_finite()));
    let high = logits[0].max(logits[1]);
    let left = (logits[0] - high).exp();
    left / (left + (logits[1] - high).exp())
}

fn choose_action<R: Rng + ?Sized>(logits: [f32; 2], rng: &mut R) -> (usize, f32) {
    let prob = left_probability(logits);
    let action = if rng.gen::<f32>() < prob { 0 } else { 1 };
    let probability = if action == 0 { prob } else { 1.0 - prob };
    (action, probability.max(f32::MIN_POSITIVE).ln())
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
    old_log_prob: f32,
    old_value: f32,
    raw_reward: f32,
    done: bool,
    truncated: bool,
    /// V(final pre-reset observation) ONLY for artificial timeouts.
    final_value: f32,
}

fn gae_targets(
    rows: &[Sample], final_values: &[f32], environments: usize, steps: usize,
    gamma: f32, lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(rows.len(), environments * steps);
    assert_eq!(final_values.len(), environments);
    let mut advantages = vec![0.0; rows.len()];
    let mut returns = vec![0.0; rows.len()];
    for lane in 0..environments {
        let mut next_advantage = 0.0_f32;
        for t in (0..steps).rev() {
            let index = t * environments + lane;
            let row = &rows[index];
            let next_value = if row.done {
                0.0
            } else if t + 1 == steps {
                final_values[lane]
            } else {
                rows[index + environments].old_value
            };
            let corrected_reward = row.raw_reward
                + if row.truncated { gamma * row.final_value } else { 0.0 };
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
        values.extend_from_slice(if action == 0 { &[1.0_f32, 0.0] } else { &[0.0, 1.0] });
    }
    Tensor::from_data(TensorData::new(values, [actions.len(), ACTIONS]), device)
}

fn mini_batch_advantages(src: &[f32]) -> Vec<f32> {
    let mean = src.iter().copied().sum::<f32>() / src.len() as f32;
    let variance = src.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / (src.len() - 1) as f32; // torch.std: ddof=1
    src.iter().map(|x| (x - mean) / (variance.sqrt() + 1e-8)).collect()
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
        &model.actor.input, &model.actor.hidden, &model.actor.output,
        &model.critic.input, &model.critic.hidden, &model.critic.output,
    ];
    let mut squared_norm = 0.0_f64;
    for linear in layers {
        let weight = gradients
            .get::<TrainBackend, 2>(linear.weight.id)
            .expect("missing PPO weight gradient")
            .to_data().to_vec::<f32>().unwrap();
        let bias = gradients
            .get::<TrainBackend, 1>(linear.bias.as_ref().unwrap().id)
            .expect("missing PPO bias gradient")
            .to_data().to_vec::<f32>().unwrap();
        squared_norm += weight.into_iter().chain(bias).map(|v| {
            let v = f64::from(v);
            v * v
        }).sum::<f64>();
    }
    assert!(squared_norm.is_finite(), "nonfinite PPO global gradient");
    let coefficient = (f64::from(maximum_norm) / (squared_norm.sqrt() + 1e-6))
        .min(1.0) as f32;
    for linear in layers {
        let weight_id = linear.weight.id;
        let bias_id = linear.bias.as_ref().unwrap().id;
        let weight = gradients.remove::<TrainBackend, 2>(weight_id)
            .expect("missing PPO weight gradient");
        gradients.register::<TrainBackend, 2>(weight_id, weight.mul_scalar(coefficient));
        let bias = gradients.remove::<TrainBackend, 1>(bias_id)
            .expect("missing PPO bias gradient");
        gradients.register::<TrainBackend, 1>(bias_id, bias.mul_scalar(coefficient));
    }
    gradients
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
        let model = ActorCritic::new(&device, &mut initialization_rng);
        let mut lanes = Vec::with_capacity(config.environments);
        for _ in 0..config.environments {
            let mut environment_rng = StdRng::seed_from_u64(root.gen());
            let action_rng = StdRng::seed_from_u64(root.gen());
            let env = PendulumEnv::new_with_rng(
                config.plant.model(), config.environment, &mut environment_rng,
            );
            let observation = env.observation_with_rng(&mut environment_rng);
            lanes.push(Lane { env, observation, environment_rng, action_rng });
        }
        let optimizer_rng = StdRng::seed_from_u64(root.gen());
        // Apply an explicit joint global norm to the full actor+critic gradient
        // before Adam. Burn's built-in Norm clips EACH tensor independently.
        let optimizer = AdamConfig::new().with_epsilon(1e-5).init();
        Self {
            config, device, model, optimizer, lanes, optimizer_rng,
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
        let input = obs_tensor::<<AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
            &self.device, &[observation],
        );
        let logits = valid.actor.forward(input).to_data().to_vec::<f32>().unwrap();
        force_for_action(if logits[0] >= logits[1] { 0 } else { 1 }, self.config.environment.max_force)
    }

    fn value(&self, observation: [f32; INPUTS]) -> f32 {
        let valid = self.model.valid();
        let input = obs_tensor::<<AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
            &self.device, &[observation],
        );
        valid.critic.forward(input).to_data().to_vec::<f32>().unwrap()[0]
    }

    pub fn train_updates(&mut self, updates: usize) {
        assert!(self.metrics.updates + updates <= self.config.total_updates,
                "CleanRL reference total training budget exceeded");
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
            let inputs = obs_tensor::<<AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
                &self.device, &observations,
            );
            let logits = valid.actor.forward(inputs.clone()).to_data().to_vec::<f32>().unwrap();
            let values = valid.critic.forward(inputs).to_data().to_vec::<f32>().unwrap();
            for (i, lane) in self.lanes.iter_mut().enumerate() {
                let distribution = [logits[2 * i], logits[2 * i + 1]];
                let (action, log_prob) = choose_action(distribution, &mut lane.action_rng);
                let step = lane.env.step_with_rng(
                    force_for_action(action, self.config.environment.max_force),
                    &mut lane.environment_rng,
                );
                let final_value = if step.truncated {
                    let input = obs_tensor::<<AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
                        &self.device, &[step.observation],
                    );
                    valid.critic.forward(input).to_data().to_vec::<f32>().unwrap()[0]
                } else { 0.0 };
                rows.push(Sample {
                    observation: lane.observation,
                    action, old_log_prob: log_prob, old_value: values[i],
                    raw_reward: step.reward, done: step.done,
                    truncated: step.truncated, final_value,
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
        let values = valid.critic.forward(
            obs_tensor::<<AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
                &self.device, &observations,
            ),
        ).to_data().to_vec::<f32>().unwrap();
        let (returns, advantages) = gae_targets(
            &rows, &values, lanes, steps, self.config.gamma, self.config.gae_lambda,
        );
        (rows, returns, advantages)
    }

    fn optimize(&mut self, rows: &[Sample], returns: &[f32], advantages: &[f32]) {
        let mut permutation = (0..rows.len()).collect::<Vec<_>>();
        let learning_rate = self.config.rate_for_update(self.metrics.updates);
        for _ in 0..self.config.epochs {
            permutation.shuffle(&mut self.optimizer_rng);
            for batch in permutation.chunks(self.config.minibatch_size) {
                let observations = batch.iter().map(|&i| rows[i].observation).collect::<Vec<_>>();
                let actions = batch.iter().map(|&i| rows[i].action).collect::<Vec<_>>();
                let old_log_probs = batch.iter().map(|&i| rows[i].old_log_prob).collect::<Vec<_>>();
                let old_values = batch.iter().map(|&i| rows[i].old_value).collect::<Vec<_>>();
                let targets = batch.iter().map(|&i| returns[i]).collect::<Vec<_>>();
                let batch_adv = batch.iter().map(|&i| advantages[i]).collect::<Vec<_>>();
                let normalized_adv = mini_batch_advantages(&batch_adv);

                let input = obs_tensor::<AutodiffBackend>(&self.device, &observations);
                let log_probs = activation::log_softmax(self.model.actor.forward(input.clone()), 1);
                let new_log_prob = (log_probs.clone() * masks(&self.device, &actions)).sum_dim(1);
                let ratio = (new_log_prob - tensor_scalars(&self.device, &old_log_probs)).exp();
                let adv = tensor_scalars(&self.device, &normalized_adv);
                let neg_adv = adv.mul_scalar(-1.0);
                let policy_loss = (ratio.clone() * neg_adv.clone()).max_pair(
                    ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * neg_adv,
                ).mean();

                let values = self.model.critic.forward(input);
                let previous = tensor_scalars(&self.device, &old_values);
                let targets = tensor_scalars(&self.device, &targets);
                let unclipped = (values.clone() - targets.clone()).square();
                let clipped_prediction = previous.clone() + (values - previous)
                    .clamp(-self.config.clip_epsilon, self.config.clip_epsilon);
                let clipped = (clipped_prediction - targets).square();
                let value_loss = unclipped.max_pair(clipped).mean().mul_scalar(0.5);

                let entropy = (log_probs.clone().exp() * log_probs)
                    .sum_dim(1).mean().mul_scalar(-1.0);
                self.metrics.last_policy_loss = policy_loss.clone().to_data().to_vec::<f32>().unwrap()[0];
                self.metrics.last_value_loss = value_loss.clone().to_data().to_vec::<f32>().unwrap()[0];
                self.metrics.last_entropy = entropy.clone().to_data().to_vec::<f32>().unwrap()[0];
                let loss = policy_loss + value_loss.mul_scalar(self.config.value_coefficient)
                    - entropy.mul_scalar(self.config.entropy_coefficient);
                let grads = GradientsParams::from_grads(loss.backward(), &self.model);
                let grads = clip_global_gradients(&self.model, grads, self.config.maximum_gradient_norm);
                self.model = self.optimizer.step(learning_rate, self.model.clone(), grads);
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
            observation: [0.0; INPUTS], action: 0, old_log_prob: 0.0,
            old_value: value, raw_reward: reward, done, truncated, final_value,
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
        for (a,b) in result.iter().zip(expected) { assert!((a-b).abs() < 1e-5); }
    }

    #[test]
    fn fixed_annealing_clock_does_not_restart_between_updates() {
        let c = CleanRlReferenceConfig::default();
        assert_eq!(c.total_updates, 976);
        assert_eq!(c.rate_for_update(0), 2.5e-4);
        assert!((c.rate_for_update(975) - 2.5e-4/976.0).abs() < 1e-12);
    }

    #[test]
    fn orthogonal_initialization_has_correct_input_output_scale() {
        let mut rng = StdRng::seed_from_u64(77);
        let model = orthogonal_linear::<crate::backend::TrainBackend, _>(
            &Default::default(), 4, 64, 2.0_f32.sqrt(), &mut rng,
        );
        let data = model.weight.val().to_data().to_vec::<f32>().unwrap();
        for a in 0..4 {
            for b in 0..4 {
                let dot: f32 = (0..64).map(|j| data[a*64+j]*data[b*64+j]).sum();
                let expected = if a == b {2.0} else {0.0};
                assert!((dot-expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn small_native_training_session_preserves_real_physics_and_finite_policy() {
        let mut c = CleanRlReferenceConfig {
            environments: 2, steps_per_environment: 4, minibatch_size: 4,
            epochs: 1, total_updates: 2, ..Default::default()
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
