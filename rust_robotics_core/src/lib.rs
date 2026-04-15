use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSnapshot {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LinearSnapshot {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
    pub action_limit: f32,
    pub action_std: f32,
}

impl PolicySnapshot {
    pub fn action_std(&self) -> f32 {
        self.action_std
    }

    pub fn act(&self, observation: [f32; 4]) -> f32 {
        let hidden0 = relu_vec(self.input.forward(&observation));
        let hidden1 = relu_vec(self.hidden.forward(&hidden0));
        let out = self.output.forward(&hidden1);
        out[0].tanh() * self.action_limit
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueSnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoSharedState {
    pub policy: PolicySnapshot,
    pub value: ValueSnapshot,
}

fn relu_vec(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(|value| value.max(0.0)).collect()
}
