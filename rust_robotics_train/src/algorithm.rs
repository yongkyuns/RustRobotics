use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmKind {
    Ppo,
    Sac,
    Dqn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    pub rollout_steps: usize,
    pub mini_batch_size: usize,
    pub epochs_per_update: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_epsilon: f32,
    pub value_loss_coef: f32,
    pub entropy_coef: f32,
    pub learning_rate: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            rollout_steps: 512,
            mini_batch_size: 128,
            epochs_per_update: 4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.0,
            learning_rate: 3e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacConfig {
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub gamma: f32,
    pub tau: f32,
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    pub alpha_learning_rate: f64,
    pub target_entropy: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            replay_capacity: 100_000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            actor_learning_rate: 3e-4,
            critic_learning_rate: 3e-4,
            alpha_learning_rate: 3e-4,
            target_entropy: -1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnConfig {
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub gamma: f32,
    pub tau: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay_steps: usize,
    pub learning_rate: f64,
    pub action_bins: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            replay_capacity: 100_000,
            batch_size: 128,
            gamma: 0.99,
            tau: 0.005,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 20_000,
            learning_rate: 1e-3,
            action_bins: 21,
        }
    }
}
