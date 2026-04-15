//! Portable shared data-transfer objects used across the workspace.
//!
//! This crate is intentionally small. Its purpose is to define serializable
//! policy / value snapshots and training metrics that can cross the boundaries
//! between:
//!
//! - `rust_robotics_train`, which owns optimization and model updates
//! - `rust_robotics_sim`, which owns live execution and UI
//! - any future offline evaluation or model-conversion tooling
//!
//! The core design choice is that the simulator should not depend on a live
//! training backend or framework-specific tensor type in order to execute a
//! learned policy. Instead, the training crate exports plain Rust structures,
//! and the simulator consumes those directly.
use serde::{Deserialize, Serialize};

/// Serializable representation of a fully-connected layer.
///
/// The snapshot is framework-agnostic:
///
/// - `in_dim` and `out_dim` define the expected matrix shape
/// - `weight` contains a flattened dense matrix
/// - `bias` contains one bias value per output neuron
///
/// The current flattened indexing convention is:
///
/// `weight[in_idx * out_dim + out_idx]`
///
/// which matches the existing Burn export/import path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSnapshot {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LinearSnapshot {
    /// Runs a forward pass through the dense layer.
    ///
    /// For each output neuron:
    ///
    /// `y[out] = bias[out] + sum_in x[in] * W[in, out]`
    ///
    /// This keeps runtime inference possible in plain Rust and in wasm without
    /// recreating the original training framework module.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_dim];
        for out_idx in 0..self.out_dim {
            let mut sum = self.bias.get(out_idx).copied().unwrap_or(0.0);
            for in_idx in 0..self.in_dim {
                sum += input[in_idx] * self.weight[in_idx * self.out_dim + out_idx];
            }
            output[out_idx] = sum;
        }
        output
    }
}

/// Serializable actor network snapshot for the pendulum PPO policy.
///
/// The current actor architecture is:
///
/// `observation[4] -> linear -> ReLU -> linear -> ReLU -> linear -> tanh`
///
/// The final `tanh` output is scaled by `action_limit`, which makes the policy
/// directly compatible with the bounded pendulum force command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
    pub action_limit: f32,
    pub action_std: f32,
}

impl PolicySnapshot {
    /// Returns the exploration standard deviation used during PPO training.
    ///
    /// The simulator currently uses deterministic inference, but surfacing this
    /// value is still useful for UI, debugging, and future replay tooling.
    pub fn action_std(&self) -> f32 {
        self.action_std
    }

    /// Runs deterministic inference for a single pendulum observation.
    ///
    /// The observation ordering is `[x, x_dot, theta, theta_dot]`.
    ///
    /// This is effectively the mean-action branch of the trained PPO actor:
    ///
    /// 1. affine transform into hidden layer 0
    /// 2. ReLU activation
    /// 3. affine transform into hidden layer 1
    /// 4. ReLU activation
    /// 5. affine transform into the scalar output
    /// 6. `tanh` squash
    /// 7. scale by `action_limit`
    pub fn act(&self, observation: [f32; 4]) -> f32 {
        let hidden0 = relu_vec(self.input.forward(&observation));
        let hidden1 = relu_vec(self.hidden.forward(&hidden0));
        let out = self.output.forward(&hidden1);
        out[0].tanh() * self.action_limit
    }
}

/// Serializable critic network snapshot for the pendulum PPO trainer.
///
/// The structure mirrors [`PolicySnapshot`], but produces a scalar value
/// estimate instead of a bounded control action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueSnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
}

/// Runtime-oriented PPO training metrics exported by the trainer.
///
/// These fields are designed to let the simulator explain "what just happened"
/// during training, rather than to replace a full experiment-tracking system.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PpoMetrics {
    pub total_updates: usize,
    pub total_env_steps: usize,
    pub total_episodes: usize,
    pub last_episode_return: f32,
    pub mean_episode_return: f32,
    pub best_episode_return: f32,
    pub last_policy_loss: f32,
    pub last_value_loss: f32,
    pub last_mean_advantage: f32,
}

/// Bundle of policy and value snapshots exchanged between trainer replicas.
///
/// The simulator's multi-replica coordinator averages these dense tensors and
/// redistributes the merged result. Keeping the actor and critic bundled
/// together avoids accidental shape mismatches between independently updated
/// models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoSharedState {
    pub policy: PolicySnapshot,
    pub value: ValueSnapshot,
}

/// Applies ReLU activation element-wise.
fn relu_vec(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(|value| value.max(0.0)).collect()
}
