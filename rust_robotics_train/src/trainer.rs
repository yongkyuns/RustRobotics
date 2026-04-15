//! PPO training loop for the pendulum environment.
//!
//! The implementation here is intentionally direct and readable. It follows the
//! standard PPO recipe:
//!
//! 1. roll out the current policy in the environment
//! 2. estimate returns and generalized advantages
//! 3. normalize advantages
//! 4. optimize the clipped surrogate objective for the actor
//! 5. optimize mean-squared error for the critic
//! 6. expose portable snapshots for runtime consumption
use crate::{
    algorithm::PpoConfig,
    backend::{AutodiffBackend, AutodiffDevice},
    env::{PendulumEnv, PendulumEnvConfig},
    model::{
        obs_tensor, policy_network_from_snapshot, scalar_tensor, value_network_from_snapshot,
        PolicyNetwork, ValueNetwork,
    },
};
use burn::{
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, AdamConfig, GradientsParams, Optimizer},
};
use rand::{seq::SliceRandom, Rng};
use rust_robotics_core::{PolicySnapshot, PpoMetrics, PpoSharedState};
use serde::{Deserialize, Serialize};

const OBS_DIM: usize = 4;

/// Configuration for a pendulum PPO training session.
///
/// The fields are split between:
///
/// - environment dynamics (`env`)
/// - PPO optimizer behavior (`ppo`)
/// - model capacity (`hidden_dim`)
/// - action sampling behavior (`action_std`)
/// - whether snapshots should be synchronized back into the simulator after
///   each update (`sync_policy_each_update`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoTrainerConfig {
    pub env: PendulumEnvConfig,
    pub ppo: PpoConfig,
    pub hidden_dim: usize,
    pub action_std: f32,
    pub sync_policy_each_update: bool,
}

impl Default for PpoTrainerConfig {
    fn default() -> Self {
        Self {
            env: PendulumEnvConfig::default(),
            ppo: PpoConfig::default(),
            hidden_dim: 64,
            action_std: 2.0,
            sync_policy_each_update: true,
        }
    }
}

/// A single rollout worth of PPO training data.
///
/// The trainer stores already-computed returns and advantages so optimization
/// can focus purely on minibatch updates.
#[derive(Debug, Clone)]
struct RolloutBatch {
    observations: Vec<[f32; 4]>,
    actions: Vec<f32>,
    old_log_probs: Vec<f32>,
    returns: Vec<f32>,
    advantages: Vec<f32>,
}

/// Stateful PPO trainer session for the inverted pendulum task.
///
/// A session owns:
///
/// - the current environment state
/// - actor / critic networks
/// - optimizer state
/// - running metrics
/// - the latest observation used to continue rollouts across update calls
pub struct PpoTrainerSession {
    config: PpoTrainerConfig,
    device: AutodiffDevice,
    env: PendulumEnv,
    current_observation: [f32; 4],
    actor: PolicyNetwork<AutodiffBackend>,
    critic: ValueNetwork<AutodiffBackend>,
    actor_optimizer:
        OptimizerAdaptor<burn::optim::Adam, PolicyNetwork<AutodiffBackend>, AutodiffBackend>,
    critic_optimizer:
        OptimizerAdaptor<burn::optim::Adam, ValueNetwork<AutodiffBackend>, AutodiffBackend>,
    metrics: PpoMetrics,
    recent_episode_returns: Vec<f32>,
}

impl PpoTrainerSession {
    /// Creates a new trainer session with fresh actor and critic networks.
    pub fn new(config: PpoTrainerConfig) -> Self {
        let device = Default::default();
        let env = PendulumEnv::new(Default::default(), config.env);
        let current_observation = env.observation();
        let actor = PolicyNetwork::new(&device, OBS_DIM, config.hidden_dim, config.env.max_force);
        let critic = ValueNetwork::new(&device, OBS_DIM, config.hidden_dim);
        let actor_optimizer = AdamConfig::new().init();
        let critic_optimizer = AdamConfig::new().init();

        Self {
            config,
            device,
            env,
            current_observation,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            metrics: PpoMetrics::default(),
            recent_episode_returns: Vec::new(),
        }
    }

    /// Returns the immutable training configuration for this session.
    pub fn config(&self) -> &PpoTrainerConfig {
        &self.config
    }

    /// Returns the latest runtime metrics.
    pub fn metrics(&self) -> &PpoMetrics {
        &self.metrics
    }

    /// Exports the current actor as a portable runtime snapshot.
    pub fn snapshot(&self) -> PolicySnapshot {
        self.actor.valid().snapshot(self.config.action_std)
    }

    /// Exports both actor and critic state for replica synchronization.
    pub fn shared_state(&self) -> PpoSharedState {
        PpoSharedState {
            policy: self.snapshot(),
            value: self.critic.valid().snapshot(),
        }
    }

    /// Replaces the actor and critic with externally provided shared state.
    ///
    /// Optimizers are reinitialized because their internal moments are tied to
    /// the old parameter tensors.
    pub fn load_shared_state(&mut self, state: &PpoSharedState) {
        self.actor = policy_network_from_snapshot(&state.policy, &self.device);
        self.critic = value_network_from_snapshot(&state.value, &self.device);
        self.actor_optimizer = AdamConfig::new().init();
        self.critic_optimizer = AdamConfig::new().init();
    }

    /// Runs a requested number of PPO updates.
    ///
    /// Each update collects a fresh rollout, then performs one optimization pass
    /// over that rollout using shuffled minibatches.
    pub fn train_updates(&mut self, num_updates: usize) {
        for _ in 0..num_updates {
            let rollout = self.collect_rollout();
            self.optimize(&rollout);
            self.metrics.total_updates += 1;
        }
    }

    /// Collects a rollout from the current policy and computes GAE targets.
    ///
    /// This is the part of PPO where raw environment interaction is converted
    /// into stable supervised targets for the actor and critic:
    ///
    /// - observations / actions / log-probs are stored for policy replay
    /// - rewards and values feed generalized advantage estimation
    /// - terminal transitions reset the environment and update metrics
    fn collect_rollout(&mut self) -> RolloutBatch {
        let rollout_steps = self.config.ppo.rollout_steps;
        let mut observations = Vec::with_capacity(rollout_steps);
        let mut actions = Vec::with_capacity(rollout_steps);
        let mut old_log_probs = Vec::with_capacity(rollout_steps);
        let mut rewards = Vec::with_capacity(rollout_steps);
        let mut values = Vec::with_capacity(rollout_steps);
        let mut terminals = Vec::with_capacity(rollout_steps);

        let mut episode_return = 0.0;

        for _ in 0..rollout_steps {
            let observation = self.current_observation;
            let mean = self.policy_mean(observation);
            let value = self.value_estimate(observation);
            let action = self.sample_action(mean);
            let log_prob = gaussian_log_prob(action, mean, self.config.action_std);
            let step = self.env.step(action);

            observations.push(observation);
            actions.push(action);
            old_log_probs.push(log_prob);
            rewards.push(step.reward);
            values.push(value);
            terminals.push(step.done);

            episode_return += step.reward;
            self.metrics.total_env_steps += 1;
            self.current_observation = step.observation;

            if step.done {
                self.metrics.total_episodes += 1;
                self.metrics.last_episode_return = episode_return;
                if self.metrics.total_episodes == 1 {
                    self.metrics.best_episode_return = episode_return;
                } else {
                    self.metrics.best_episode_return =
                        self.metrics.best_episode_return.max(episode_return);
                }
                push_recent(&mut self.recent_episode_returns, episode_return, 32);
                self.metrics.mean_episode_return =
                    mean_slice(&self.recent_episode_returns).unwrap_or(episode_return);
                episode_return = 0.0;
                self.current_observation = self.env.reset();
            }
        }

        let bootstrap_value = if terminals.last().copied().unwrap_or(false) {
            0.0
        } else {
            self.value_estimate(self.current_observation)
        };

        let (returns, advantages) = compute_gae(
            &rewards,
            &values,
            &terminals,
            bootstrap_value,
            self.config.ppo.gamma,
            self.config.ppo.gae_lambda,
        );
        let advantages = normalize(&advantages);

        self.metrics.last_mean_advantage = mean_slice(&advantages).unwrap_or(0.0);

        RolloutBatch {
            observations,
            actions,
            old_log_probs,
            returns,
            advantages,
        }
    }

    /// Optimizes actor and critic networks from a prepared rollout batch.
    ///
    /// The actor uses the clipped PPO surrogate:
    ///
    /// `min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)`
    ///
    /// where `r_t` is the ratio of new to old action probability. The critic
    /// uses a plain mean-squared error objective against bootstrapped returns.
    fn optimize(&mut self, rollout: &RolloutBatch) {
        let mut indices = (0..rollout.observations.len()).collect::<Vec<_>>();
        let batch_size = self.config.ppo.mini_batch_size.max(1);
        let mut rng = rand::thread_rng();
        let mut last_policy_loss = 0.0;
        let mut last_value_loss = 0.0;

        for _ in 0..self.config.ppo.epochs_per_update {
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(batch_size) {
                let observations = gather_observations(&rollout.observations, chunk);
                let actions = gather_scalars(&rollout.actions, chunk);
                let old_log_probs = gather_scalars(&rollout.old_log_probs, chunk);
                let returns = gather_scalars(&rollout.returns, chunk);
                let advantages = gather_scalars(&rollout.advantages, chunk);

                let observations = obs_tensor::<AutodiffBackend>(&self.device, &observations);

                let new_means = self.actor.forward(observations.clone());
                let actions = scalar_tensor::<AutodiffBackend>(&self.device, &actions);
                let old_log_probs = scalar_tensor::<AutodiffBackend>(&self.device, &old_log_probs);
                let returns = scalar_tensor::<AutodiffBackend>(&self.device, &returns);
                let advantages = scalar_tensor::<AutodiffBackend>(&self.device, &advantages);

                let new_log_probs =
                    gaussian_log_prob_tensor(new_means, actions.clone(), self.config.action_std);
                let ratios = (new_log_probs - old_log_probs).exp();
                let unclipped = ratios.clone() * advantages.clone();
                let clipped = ratios.clamp(
                    1.0 - self.config.ppo.clip_epsilon,
                    1.0 + self.config.ppo.clip_epsilon,
                ) * advantages;
                let policy_loss = unclipped.min_pair(clipped).mean().mul_scalar(-1.0);
                let policy_loss_scalar = tensor_scalar(&policy_loss);
                let actor_grads = GradientsParams::from_grads(policy_loss.backward(), &self.actor);
                self.actor = self.actor_optimizer.step(
                    self.config.ppo.learning_rate,
                    self.actor.clone(),
                    actor_grads,
                );

                let values = self.critic.forward(observations);
                let value_loss = (values - returns).square().mean();
                let value_loss_scalar = tensor_scalar(&value_loss);
                let critic_grads = GradientsParams::from_grads(value_loss.backward(), &self.critic);
                self.critic = self.critic_optimizer.step(
                    self.config.ppo.learning_rate,
                    self.critic.clone(),
                    critic_grads,
                );

                last_policy_loss = policy_loss_scalar;
                last_value_loss = value_loss_scalar;
            }
        }

        self.metrics.last_policy_loss = last_policy_loss;
        self.metrics.last_value_loss = last_value_loss;
    }

    /// Computes the deterministic mean action predicted by the actor.
    fn policy_mean(&self, observation: [f32; 4]) -> f32 {
        let policy = self.actor.valid();
        let tensor = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(&self.device, &[observation]);
        tensor_scalar(&policy.forward(tensor))
    }

    /// Computes the critic's scalar value estimate for one observation.
    fn value_estimate(&self, observation: [f32; 4]) -> f32 {
        let critic = self.critic.valid();
        let tensor = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(&self.device, &[observation]);
        tensor_scalar(&critic.forward(tensor))
    }

    /// Samples a bounded scalar action from the actor mean and configured
    /// Gaussian exploration noise.
    fn sample_action(&self, mean: f32) -> f32 {
        let std = self.config.action_std.max(1.0e-3);
        (mean + sample_standard_normal() * std)
            .clamp(-self.config.env.max_force, self.config.env.max_force)
    }
}

fn gather_observations(source: &[[f32; 4]], indices: &[usize]) -> Vec<[f32; 4]> {
    indices.iter().map(|index| source[*index]).collect()
}

fn gather_scalars(source: &[f32], indices: &[usize]) -> Vec<f32> {
    indices.iter().map(|index| source[*index]).collect()
}

fn gaussian_log_prob(action: f32, mean: f32, std: f32) -> f32 {
    let var = std * std;
    let diff = action - mean;
    -0.5 * ((diff * diff) / var + (2.0 * std::f32::consts::PI * var).ln())
}

fn gaussian_log_prob_tensor<B: burn::tensor::backend::Backend>(
    mean: burn::tensor::Tensor<B, 2>,
    action: burn::tensor::Tensor<B, 2>,
    std: f32,
) -> burn::tensor::Tensor<B, 2> {
    let var = std * std;
    let diff = action - mean;
    diff.square()
        .div_scalar(var)
        .add_scalar((2.0 * std::f32::consts::PI * var).ln())
        .mul_scalar(-0.5)
}

fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminals: &[bool],
    bootstrap_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut advantages = vec![0.0; rewards.len()];
    let mut returns = vec![0.0; rewards.len()];
    let mut next_value = bootstrap_value;
    let mut gae = 0.0;

    for index in (0..rewards.len()).rev() {
        let non_terminal = if terminals[index] { 0.0 } else { 1.0 };
        let delta = rewards[index] + gamma * next_value * non_terminal - values[index];
        gae = delta + gamma * gae_lambda * non_terminal * gae;
        advantages[index] = gae;
        returns[index] = gae + values[index];
        next_value = values[index];
    }

    (returns, advantages)
}

fn normalize(values: &[f32]) -> Vec<f32> {
    let mean = mean_slice(values).unwrap_or(0.0);
    let variance = if values.is_empty() {
        0.0
    } else {
        values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / values.len() as f32
    };
    let std = variance.sqrt().max(1.0e-6);
    values
        .iter()
        .map(|value| (*value - mean) / std)
        .collect::<Vec<_>>()
}

fn mean_slice(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn push_recent(values: &mut Vec<f32>, value: f32, max_len: usize) {
    values.push(value);
    if values.len() > max_len {
        let overflow = values.len() - max_len;
        values.drain(0..overflow);
    }
}

fn sample_standard_normal() -> f32 {
    let mut rng = rand::thread_rng();
    let u1 = rng.gen_range(f32::EPSILON..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn tensor_scalar<B: burn::tensor::backend::Backend, const D: usize>(
    tensor: &burn::tensor::Tensor<B, D>,
) -> f32 {
    tensor.to_data().to_vec::<f32>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_session_smoke_test() {
        let mut session = PpoTrainerSession::new(PpoTrainerConfig::default());
        session.train_updates(1);

        let metrics = session.metrics();
        assert_eq!(metrics.total_updates, 1);
        assert!(metrics.total_env_steps >= session.config().ppo.rollout_steps);
        assert!(session.snapshot().act([0.0, 0.0, 0.1, 0.0]).is_finite());
    }
}
