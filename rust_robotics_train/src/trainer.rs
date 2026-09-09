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
        obs_tensor, policy_network_from_snapshot, scalar_tensor, value_network_from_snapshot, Mlp,
        PolicyNetwork, ValueNetwork,
    },
};
use burn::{
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, AdamConfig, GradientsParams, Optimizer},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rust_robotics_core::{PolicySnapshot, PpoMetrics, PpoSharedState};
use serde::{Deserialize, Serialize};

#[path = "ppo_distribution.rs"]
mod distribution;
use distribution::{standard_normal, SquashedGaussian};

#[path = "ppo_rollout_pool.rs"]
mod rollout_pool;
use rollout_pool::{RolloutCollector, RolloutCursor, RolloutEnvironment};

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
    /// Pre-squash exploration scale in force units, finite and positive.
    /// The latent Gaussian standard deviation is `action_std / env.max_force`.
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
    latent_actions: Vec<f32>,
    old_log_probs: Vec<f32>,
    returns: Vec<f32>,
    advantages: Vec<f32>,
}

/// Stateful PPO trainer session for the inverted pendulum task.
///
/// A session owns:
///
/// - one or more independent environment/episode/RNG states
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
    // Undiscounted return of the unfinished environment episode.
    episode_return: f32,
    // Separate continuing streams prevent optimizer draw counts from changing
    // collection noise. No per-tick thread_rng or backend-global seed is used.
    environment_rng: StdRng,
    action_rng: StdRng,
    update_rng: StdRng,
    // Stream zero retains the original fields/seeding contract above. Additional
    // streams contain only environment/episode/RNG state, not independent agents.
    additional_environments: Vec<RolloutEnvironment>,
}

impl PpoTrainerSession {
    /// Creates a fresh, entropy-seeded session with privately owned randomness.
    pub fn new(config: PpoTrainerConfig) -> Self {
        Self::new_seeded(config, rand::thread_rng().gen())
    }

    /// Creates a reproducible fresh session, including model initialization,
    /// environment resets/noise, action sampling, minibatches and entropy.
    ///
    /// Equal seeds, configurations and call sequences replay numerical state on
    /// the same backend/build. No cross-platform/version bitwise or checkpoint
    /// guarantee is made. This does not seed unrelated sessions or Burn globally.
    pub fn new_seeded(config: PpoTrainerConfig, seed: u64) -> Self {
        SquashedGaussian::new(config.action_std, config.env.max_force);
        assert!(
            config.ppo.entropy_coef.is_finite() && config.ppo.entropy_coef >= 0.0,
            "entropy_coef must be finite and nonnegative"
        );
        assert!(
            config.ppo.value_loss_coef.is_finite() && config.ppo.value_loss_coef >= 0.0,
            "value_loss_coef must be finite and nonnegative"
        );
        let device = Default::default();
        // Fixed child order: actor initialization, critic initialization,
        // environment, actions, optimizer. Initialization draws do not consume
        // live environment/action streams, including when hidden_dim changes.
        let mut root = StdRng::seed_from_u64(seed);
        let mut child = || StdRng::seed_from_u64(root.gen());
        let mut actor_rng = child();
        let mut critic_rng = child();
        let mut environment_rng = child();
        let action_rng = child();
        let update_rng = child();
        let env = PendulumEnv::new_with_rng(Default::default(), config.env, &mut environment_rng);
        let current_observation = env.observation_with_rng(&mut environment_rng);
        let actor = PolicyNetwork {
            mlp: Mlp::new_with_rng(&device, OBS_DIM, config.hidden_dim, 1, &mut actor_rng),
            action_limit: config.env.max_force,
        };
        let critic = ValueNetwork {
            mlp: Mlp::new_with_rng(&device, OBS_DIM, config.hidden_dim, 1, &mut critic_rng),
        };
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
            episode_return: 0.0,
            environment_rng,
            action_rng,
            update_rng,
            additional_environments: Vec::new(),
        }
    }

    /// Creates one learner with several environment streams and fresh randomness.
    /// `ppo.rollout_steps` is the number of transitions PER environment/update.
    /// Collection is synchronous; this does not promise CPU-worker parallelism.
    pub fn new_with_environments(config: PpoTrainerConfig, environments: usize) -> Self {
        Self::new_seeded_with_environments(config, rand::thread_rng().gen(), environments)
    }

    /// Reproducible shared-policy collection. One environment is exactly the
    /// original seeded session, including initialization and every RNG stream.
    /// Additional streams use a separate, fixed seed domain; adding streams does
    /// not alter the initial models or any existing stream's initial randomness.
    ///
    /// Panics for zero environments or an overflowing rollout size. As with
    /// `new_seeded`, replay is within the same numerical backend/build only.
    pub fn new_seeded_with_environments(
        config: PpoTrainerConfig,
        seed: u64,
        environments: usize,
    ) -> Self {
        assert!(environments > 0, "PPO requires at least one environment");
        assert!(
            config.ppo.rollout_steps.checked_mul(environments).is_some(),
            "PPO pooled rollout size overflows usize"
        );
        let env_config = config.env;
        let mut session = Self::new_seeded(config, seed);
        let mut streams = StdRng::seed_from_u64(seed ^ 0x504f_4f4c_4544_0001);
        for _ in 1..environments {
            session
                .additional_environments
                .push(RolloutEnvironment::new(
                    env_config,
                    streams.gen(),
                    streams.gen(),
                ));
        }
        session
    }

    /// Number of persistent environments sharing this session's actor/critic.
    pub fn environment_count(&self) -> usize {
        1 + self.additional_environments.len()
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

    /// Exports actor and critic weights for an explicit external warm start.
    pub fn shared_state(&self) -> PpoSharedState {
        PpoSharedState {
            policy: self.snapshot(),
            value: self.critic.valid().snapshot(),
        }
    }

    /// Replaces the actor and critic with externally provided shared state.
    ///
    /// Optimizers are reinitialized because their internal moments are tied to
    /// the old parameter tensors. This is a weight-transfer warm start, not an
    /// exact-resume checkpoint: environment state, RNG and metrics are retained.
    /// The saved exploration scale is adopted; the action limit must match the
    /// receiving environment so it cannot silently clip a different policy.
    pub fn load_shared_state(&mut self, state: &PpoSharedState) {
        assert_eq!(
            state.policy.action_limit, self.config.env.max_force,
            "shared policy action limit must match the environment"
        );
        SquashedGaussian::new(state.policy.action_std, state.policy.action_limit);
        self.config.action_std = state.policy.action_std;
        self.actor = policy_network_from_snapshot(&state.policy, &self.device);
        self.critic = value_network_from_snapshot(&state.value, &self.device);
        self.actor_optimizer = AdamConfig::new().init();
        self.critic_optimizer = AdamConfig::new().init();
    }

    /// Runs a requested number of PPO updates.
    ///
    /// Each update collects every environment under one frozen actor/critic,
    /// then optimizes the union for the configured epochs with shuffled minibatches.
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
    /// - each reset ends a GAE trace; only true termination removes bootstrap
    /// - unfinished episode returns survive rollout/update boundaries
    fn collect_rollout(&mut self) -> RolloutBatch {
        // Use a local cursor to avoid aliasing &mut self with its RNG, then
        // retain the advanced cursor. This is a copy, not a reseed or shared RNG.
        let mut rng = self.action_rng.clone();
        let rollout = self.collect_rollout_with_rng(&mut rng);
        self.action_rng = rng;
        rollout
    }

    fn collect_rollout_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> RolloutBatch {
        // All streams see the SAME frozen actor and critic. No optimization is
        // allowed until the complete pool has been collected. Each call below
        // independently closes terminal, timeout and buffer-cutoff GAE traces.
        let mut collector = RolloutCollector {
            config: &self.config,
            device: &self.device,
            actor: &self.actor,
            critic: &self.critic,
            metrics: &mut self.metrics,
            recent_episode_returns: &mut self.recent_episode_returns,
        };
        let mut rollout = collector.collect(
            RolloutCursor {
                env: &mut self.env,
                current_observation: &mut self.current_observation,
                episode_return: &mut self.episode_return,
                environment_rng: &mut self.environment_rng,
            },
            rng,
        );
        for stream in &mut self.additional_environments {
            let RolloutEnvironment {
                env,
                current_observation,
                episode_return,
                environment_rng,
                action_rng,
            } = stream;
            let next = collector.collect(
                RolloutCursor {
                    env,
                    current_observation,
                    episode_return,
                    environment_rng,
                },
                action_rng,
            );
            rollout.observations.extend(next.observations);
            rollout.latent_actions.extend(next.latent_actions);
            rollout.old_log_probs.extend(next.old_log_probs);
            rollout.returns.extend(next.returns);
            rollout.advantages.extend(next.advantages);
        }
        // Normalize ONCE over the union, never separately per stream/episode.
        rollout.advantages = normalize(&rollout.advantages);
        self.metrics.last_mean_advantage = mean_slice(&rollout.advantages).unwrap_or(0.0);
        rollout
    }

    /// Optimizes actor and critic networks from a prepared rollout batch.
    ///
    /// The actor uses the clipped PPO surrogate:
    ///
    /// `min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)`
    ///
    /// where `r_t` is the ratio of new to old action probability. The critic
    /// uses configured MSE weighting. Entropy uses fresh current-policy samples,
    /// not negative log likelihood of actions from the old rollout policy.
    fn optimize(&mut self, rollout: &RolloutBatch) {
        let mut rng = self.update_rng.clone();
        self.optimize_with_rng(rollout, &mut rng);
        self.update_rng = rng;
    }

    fn optimize_with_rng<R: Rng + ?Sized>(&mut self, rollout: &RolloutBatch, rng: &mut R) {
        let distribution = SquashedGaussian::new(self.config.action_std, self.actor.action_limit);
        let mut indices = (0..rollout.observations.len()).collect::<Vec<_>>();
        let batch_size = self.config.ppo.mini_batch_size.max(1);
        let mut last_policy_loss = 0.0;
        let mut last_value_loss = 0.0;

        for _ in 0..self.config.ppo.epochs_per_update {
            indices.shuffle(rng);

            for chunk in indices.chunks(batch_size) {
                let observations = gather_observations(&rollout.observations, chunk);
                let latents = gather_scalars(&rollout.latent_actions, chunk);
                let old_log_probs = gather_scalars(&rollout.old_log_probs, chunk);
                let returns = gather_scalars(&rollout.returns, chunk);
                let advantages = gather_scalars(&rollout.advantages, chunk);

                let observations = obs_tensor::<AutodiffBackend>(&self.device, &observations);

                let new_means = self.actor.latent_mean(observations.clone());
                let latents = scalar_tensor::<AutodiffBackend>(&self.device, &latents);
                let old_log_probs = scalar_tensor::<AutodiffBackend>(&self.device, &old_log_probs);
                let returns = scalar_tensor::<AutodiffBackend>(&self.device, &returns);
                let advantages = scalar_tensor::<AutodiffBackend>(&self.device, &advantages);

                let new_log_probs = distribution.log_prob_tensor(new_means.clone(), latents);
                let policy_loss = clipped_surrogate(
                    new_log_probs,
                    old_log_probs,
                    advantages,
                    self.config.ppo.clip_epsilon,
                );
                let policy_loss_scalar = tensor_scalar(&policy_loss);
                let actor_loss = if self.config.ppo.entropy_coef == 0.0 {
                    policy_loss
                } else {
                    let noise = (0..chunk.len())
                        .map(|_| standard_normal(rng))
                        .collect::<Vec<_>>();
                    let noise = scalar_tensor::<AutodiffBackend>(&self.device, &noise);
                    let entropy = distribution.entropy(new_means, noise);
                    policy_loss - entropy.mul_scalar(self.config.ppo.entropy_coef)
                };
                let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &self.actor);
                self.actor = self.actor_optimizer.step(
                    self.config.ppo.learning_rate,
                    self.actor.clone(),
                    actor_grads,
                );

                let values = self.critic.forward(observations);
                let value_loss = (values - returns).square().mean();
                let value_loss_scalar = tensor_scalar(&value_loss);
                // Skipping Adam is necessary at zero: a zero gradient alone
                // would still move parameters using earlier optimizer moments.
                if self.config.ppo.value_loss_coef > 0.0 {
                    let objective =
                        weighted_value_loss(value_loss, self.config.ppo.value_loss_coef);
                    let critic_grads =
                        GradientsParams::from_grads(objective.backward(), &self.critic);
                    self.critic = self.critic_optimizer.step(
                        self.config.ppo.learning_rate,
                        self.critic.clone(),
                        critic_grads,
                    );
                }

                last_policy_loss = policy_loss_scalar;
                last_value_loss = value_loss_scalar;
            }
        }

        self.metrics.last_policy_loss = last_policy_loss;
        self.metrics.last_value_loss = last_value_loss;
    }

    #[cfg(test)]
    /// Computes the unsquashed Gaussian mean predicted by the actor.
    fn policy_latent_mean(&self, observation: [f32; 4]) -> f32 {
        let policy = self.actor.valid();
        let tensor = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(&self.device, &[observation]);
        tensor_scalar(&policy.latent_mean(tensor))
    }
}

fn gather_observations(source: &[[f32; 4]], indices: &[usize]) -> Vec<[f32; 4]> {
    indices.iter().map(|index| source[*index]).collect()
}

fn gather_scalars(source: &[f32], indices: &[usize]) -> Vec<f32> {
    indices.iter().map(|index| source[*index]).collect()
}

fn clipped_surrogate<B: burn::tensor::backend::Backend>(
    new_log_probs: burn::tensor::Tensor<B, 2>,
    old_log_probs: burn::tensor::Tensor<B, 2>,
    advantages: burn::tensor::Tensor<B, 2>,
    epsilon: f32,
) -> burn::tensor::Tensor<B, 1> {
    let ratios = (new_log_probs - old_log_probs).exp();
    let unclipped = ratios.clone() * advantages.clone();
    let clipped = ratios.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages;
    unclipped.min_pair(clipped).mean().mul_scalar(-1.0)
}

fn weighted_value_loss<B: burn::tensor::backend::Backend>(
    mse: burn::tensor::Tensor<B, 1>,
    coefficient: f32,
) -> burn::tensor::Tensor<B, 1> {
    mse.mul_scalar(coefficient)
}

fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminals: &[bool],
    bootstrap_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        rewards.len(),
        values.len(),
        "GAE rewards/values length mismatch"
    );
    assert_eq!(
        rewards.len(),
        terminals.len(),
        "GAE rewards/terminals length mismatch"
    );
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

#[cfg(test)]
#[path = "ppo_objective_tests.rs"]
mod objective_tests;

#[cfg(test)]
#[path = "ppo_rollout_tests.rs"]
mod rollout_tests;

#[cfg(test)]
#[path = "ppo_seed_tests.rs"]
mod seed_tests;

#[cfg(test)]
#[path = "ppo_pool_tests.rs"]
mod pool_tests;
