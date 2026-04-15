use rand::Rng;
use rust_robotics_algo::{
    control::StateSpace,
    inverted_pendulum::Model,
    nalgebra,
    prelude::{vector, Vector4},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PendulumEnvConfig {
    pub dt: f32,
    pub max_force: f32,
    pub reset_position_range_m: f32,
    pub reset_velocity_range_mps: f32,
    pub reset_angle_range_rad: f32,
    pub reset_angular_velocity_range_radps: f32,
    pub max_angle_rad: f32,
    pub max_position_m: f32,
    pub max_steps: usize,
    pub observation_position_noise_m: f32,
    pub observation_velocity_noise_mps: f32,
    pub observation_angle_noise_rad: f32,
    pub observation_angular_velocity_noise_radps: f32,
    pub action_noise_force_n: f32,
    pub disturbance_force_n: f32,
    pub disturbance_probability_per_step: f32,
    pub reward_position_weight: f32,
    pub reward_velocity_weight: f32,
    pub reward_angle_weight: f32,
    pub reward_angular_velocity_weight: f32,
    pub reward_action_weight: f32,
}

impl Default for PendulumEnvConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            max_force: 20.0,
            reset_position_range_m: 0.2,
            reset_velocity_range_mps: 0.4,
            reset_angle_range_rad: 0.25,
            reset_angular_velocity_range_radps: 0.5,
            max_angle_rad: 0.6,
            max_position_m: 2.4,
            max_steps: 5_000,
            observation_position_noise_m: 0.002,
            observation_velocity_noise_mps: 0.01,
            observation_angle_noise_rad: 0.002,
            observation_angular_velocity_noise_radps: 0.01,
            action_noise_force_n: 0.15,
            disturbance_force_n: 1.0,
            disturbance_probability_per_step: 0.005,
            reward_position_weight: 0.2,
            reward_velocity_weight: 0.02,
            reward_angle_weight: 1.0,
            reward_angular_velocity_weight: 0.05,
            reward_action_weight: 0.001,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StepResult {
    pub observation: [f32; 4],
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct PendulumEnv {
    model: Model,
    config: PendulumEnvConfig,
    state: Vector4,
    steps: usize,
}

impl PendulumEnv {
    pub fn new(model: Model, config: PendulumEnvConfig) -> Self {
        let mut env = Self {
            model,
            config,
            state: vector![0.0, 0.0, 0.0, 0.0],
            steps: 0,
        };
        env.reset();
        env
    }

    pub fn model(&self) -> Model {
        self.model
    }

    pub fn config(&self) -> PendulumEnvConfig {
        self.config
    }

    pub fn state(&self) -> Vector4 {
        self.state
    }

    pub fn observation(&self) -> [f32; 4] {
        let mut rng = rand::thread_rng();
        [
            self.state[0] + sample_symmetric(&mut rng, self.config.observation_position_noise_m),
            self.state[1] + sample_symmetric(&mut rng, self.config.observation_velocity_noise_mps),
            self.state[2] + sample_symmetric(&mut rng, self.config.observation_angle_noise_rad),
            self.state[3]
                + sample_symmetric(
                    &mut rng,
                    self.config.observation_angular_velocity_noise_radps,
                ),
        ]
    }

    pub fn reset(&mut self) -> [f32; 4] {
        let mut rng = rand::thread_rng();
        self.state = vector![
            sample_symmetric(&mut rng, self.config.reset_position_range_m),
            sample_symmetric(&mut rng, self.config.reset_velocity_range_mps),
            sample_symmetric(&mut rng, self.config.reset_angle_range_rad),
            sample_symmetric(&mut rng, self.config.reset_angular_velocity_range_radps)
        ];
        self.steps = 0;
        self.observation()
    }

    pub fn step(&mut self, action: f32) -> StepResult {
        let mut rng = rand::thread_rng();
        let clipped_action = action.clamp(-self.config.max_force, self.config.max_force);
        let action_noise = sample_symmetric(&mut rng, self.config.action_noise_force_n);
        let disturbance =
            if rng.gen_bool(self.config.disturbance_probability_per_step.clamp(0.0, 1.0) as f64) {
                sample_symmetric(&mut rng, self.config.disturbance_force_n)
            } else {
                0.0
            };
        let applied_force = clipped_action + action_noise + disturbance;
        let (a, b) = self.model.model(self.config.dt);
        self.state = a * self.state + b * applied_force;
        self.steps += 1;

        let terminated = self.state[2].abs() > self.config.max_angle_rad
            || self.state[0].abs() > self.config.max_position_m;
        let truncated = self.steps >= self.config.max_steps;
        let reward = self.reward(clipped_action, terminated);

        StepResult {
            observation: self.observation(),
            reward,
            done: terminated || truncated,
            truncated,
        }
    }

    fn reward(&self, action: f32, terminated: bool) -> f32 {
        if terminated {
            return -10.0;
        }

        1.0 - self.config.reward_position_weight * self.state[0] * self.state[0]
            - self.config.reward_velocity_weight * self.state[1] * self.state[1]
            - self.config.reward_angle_weight * self.state[2] * self.state[2]
            - self.config.reward_angular_velocity_weight * self.state[3] * self.state[3]
            - self.config.reward_action_weight * action * action
    }
}

fn sample_symmetric(rng: &mut impl Rng, magnitude: f32) -> f32 {
    let magnitude = magnitude.max(0.0);
    if magnitude == 0.0 {
        0.0
    } else {
        rng.gen_range(-magnitude..magnitude)
    }
}

impl Default for PendulumEnv {
    fn default() -> Self {
        Self::new(Model::default(), PendulumEnvConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_returns_finite_observation() {
        let mut env = PendulumEnv::default();
        let obs = env.reset();

        assert!(obs.into_iter().all(f32::is_finite));
    }

    #[test]
    fn step_clamps_large_action() {
        let mut env = PendulumEnv::default();
        env.config.action_noise_force_n = 0.0;
        env.config.disturbance_force_n = 0.0;
        env.config.disturbance_probability_per_step = 0.0;
        env.config.observation_position_noise_m = 0.0;
        env.config.observation_velocity_noise_mps = 0.0;
        env.config.observation_angle_noise_rad = 0.0;
        env.config.observation_angular_velocity_noise_radps = 0.0;
        env.state = vector![0.0, 0.0, 0.1, 0.0];
        env.steps = 0;

        let clamped = env.step(env.config.max_force);
        env.state = vector![0.0, 0.0, 0.1, 0.0];
        env.steps = 0;
        let unclamped = env.step(env.config.max_force * 100.0);

        assert_eq!(clamped.observation, unclamped.observation);
        assert_eq!(clamped.reward, unclamped.reward);
    }
}
