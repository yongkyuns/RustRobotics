//! Trajectory-local collection under one immutable actor/critic pair.
//!
//! Additional environments own no models or optimizers. This code is also the
//! single-environment collector: there is no second PPO training implementation.
use super::*;

#[derive(Debug, Clone)]
pub(super) struct RolloutEnvironment {
    pub(super) env: PendulumEnv,
    pub(super) current_observation: [f32; 4],
    pub(super) episode_return: f32,
    pub(super) environment_rng: StdRng,
    pub(super) action_rng: StdRng,
}

impl RolloutEnvironment {
    pub(super) fn new(config: PendulumEnvConfig, environment_seed: u64, action_seed: u64) -> Self {
        let mut environment_rng = StdRng::seed_from_u64(environment_seed);
        let env = PendulumEnv::new_with_rng(Default::default(), config, &mut environment_rng);
        let current_observation = env.observation_with_rng(&mut environment_rng);
        Self {
            env,
            current_observation,
            episode_return: 0.0,
            environment_rng,
            action_rng: StdRng::seed_from_u64(action_seed),
        }
    }
}

pub(super) struct RolloutCursor<'a> {
    pub(super) env: &'a mut PendulumEnv,
    pub(super) current_observation: &'a mut [f32; 4],
    pub(super) episode_return: &'a mut f32,
    pub(super) environment_rng: &'a mut StdRng,
}

pub(super) struct RolloutCollector<'a> {
    pub(super) config: &'a PpoTrainerConfig,
    pub(super) device: &'a AutodiffDevice,
    pub(super) actor: &'a PolicyNetwork<AutodiffBackend>,
    pub(super) critic: &'a ValueNetwork<AutodiffBackend>,
    pub(super) metrics: &'a mut PpoMetrics,
    pub(super) recent_episode_returns: &'a mut Vec<f32>,
}

impl RolloutCollector<'_> {
    pub(super) fn collect<R: Rng + ?Sized>(
        &mut self,
        cursor: RolloutCursor<'_>,
        rng: &mut R,
    ) -> RolloutBatch {
        let rollout_steps = self.config.ppo.rollout_steps;
        let mut observations = Vec::with_capacity(rollout_steps);
        let mut latent_actions = Vec::with_capacity(rollout_steps);
        let mut old_log_probs = Vec::with_capacity(rollout_steps);
        let mut rewards = Vec::with_capacity(rollout_steps);
        let mut values = Vec::with_capacity(rollout_steps);
        let mut terminals = Vec::with_capacity(rollout_steps);
        let mut returns = Vec::with_capacity(rollout_steps);
        let mut advantages = Vec::with_capacity(rollout_steps);
        let mut path_start = 0;
        let distribution = SquashedGaussian::new(self.config.action_std, self.actor.action_limit);

        for index in 0..rollout_steps {
            let observation = *cursor.current_observation;
            let mean = self.policy_latent_mean(observation);
            let value = self.value_estimate(observation);
            let sample = distribution.sample(mean, rng);
            let step = cursor
                .env
                .step_with_rng(sample.action, cursor.environment_rng);

            observations.push(observation);
            latent_actions.push(sample.latent);
            old_log_probs.push(sample.log_prob);
            rewards.push(step.reward);
            values.push(value);
            terminals.push(step.terminated());
            *cursor.episode_return += step.reward;
            self.metrics.total_env_steps += 1;

            // Every reset ends a GAE trace. Only true task termination removes
            // the bootstrap; time limits use the final observation BEFORE reset.
            // A buffer cutoff finishes targets without ending the live episode.
            if step.done || index + 1 == rollout_steps {
                let bootstrap_value = if step.terminated() {
                    0.0
                } else {
                    self.value_estimate(step.observation)
                };
                let (path_returns, path_advantages) = compute_gae(
                    &rewards[path_start..],
                    &values[path_start..],
                    &terminals[path_start..],
                    bootstrap_value,
                    self.config.ppo.gamma,
                    self.config.ppo.gae_lambda,
                );
                returns.extend(path_returns);
                advantages.extend(path_advantages);
                path_start = rewards.len();
            }

            *cursor.current_observation = step.observation;
            if step.done {
                let episode_return = std::mem::take(cursor.episode_return);
                self.metrics.total_episodes += 1;
                self.metrics.last_episode_return = episode_return;
                if self.metrics.total_episodes == 1 {
                    self.metrics.best_episode_return = episode_return;
                } else {
                    self.metrics.best_episode_return =
                        self.metrics.best_episode_return.max(episode_return);
                }
                push_recent(self.recent_episode_returns, episode_return, 32);
                self.metrics.mean_episode_return =
                    mean_slice(self.recent_episode_returns).unwrap_or(episode_return);
                *cursor.current_observation = cursor.env.reset_with_rng(cursor.environment_rng);
            }
        }

        RolloutBatch {
            observations,
            latent_actions,
            old_log_probs,
            returns,
            advantages,
        }
    }

    /// Computes the unsquashed Gaussian mean predicted by the actor.
    fn policy_latent_mean(&self, observation: [f32; 4]) -> f32 {
        let policy = self.actor.valid();
        let tensor = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(self.device, &[observation]);
        tensor_scalar(&policy.latent_mean(tensor))
    }

    /// Computes the critic's scalar value estimate for one observation.
    fn value_estimate(&self, observation: [f32; 4]) -> f32 {
        let critic = self.critic.valid();
        let tensor = obs_tensor::<
            <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend,
        >(self.device, &[observation]);
        tensor_scalar(&critic.forward(tensor))
    }
}
