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
    /// External collection time limit, not a finite-horizon task terminal.
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
    /// True after task termination or time-limit truncation; reset is required.
    pub done: bool,
    /// Time-limit-only end. Task failure takes precedence when both coincide.
    pub truncated: bool,
}

impl StepResult {
    /// Task termination for results produced by `PendulumEnv`.
    /// This relies on its time-limit-only `truncated` contract.
    pub fn terminated(&self) -> bool {
        self.done && !self.truncated
    }
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
        Self::new_with_rng(model, config, &mut rand::thread_rng())
    }

    /// Constructs and resets using only the caller's random stream.
    /// Keep using the `_with_rng` methods for a reproducible trajectory.
    pub fn new_with_rng<R: Rng + ?Sized>(
        model: Model,
        config: PendulumEnvConfig,
        rng: &mut R,
    ) -> Self {
        let mut env = Self {
            model,
            config,
            state: vector![0.0, 0.0, 0.0, 0.0],
            steps: 0,
        };
        env.reset_with_rng(rng);
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
        self.observation_with_rng(&mut rand::thread_rng())
    }

    /// Samples observation noise without touching any shared random stream.
    pub fn observation_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> [f32; 4] {
        [
            self.state[0] + sample_symmetric(rng, self.config.observation_position_noise_m),
            self.state[1] + sample_symmetric(rng, self.config.observation_velocity_noise_mps),
            self.state[2] + sample_symmetric(rng, self.config.observation_angle_noise_rad),
            self.state[3]
                + sample_symmetric(rng, self.config.observation_angular_velocity_noise_radps),
        ]
    }

    pub fn reset(&mut self) -> [f32; 4] {
        self.reset_with_rng(&mut rand::thread_rng())
    }

    /// Draws reset state and its observation from the caller's continuing stream.
    /// This does not reseed that stream.
    pub fn reset_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f32; 4] {
        self.state = vector![
            sample_symmetric(rng, self.config.reset_position_range_m),
            sample_symmetric(rng, self.config.reset_velocity_range_mps),
            sample_symmetric(rng, self.config.reset_angle_range_rad),
            sample_symmetric(rng, self.config.reset_angular_velocity_range_radps)
        ];
        self.steps = 0;
        self.observation_with_rng(rng)
    }

    pub fn step(&mut self, action: f32) -> StepResult {
        self.step_with_rng(action, &mut rand::thread_rng())
    }

    /// Draws action noise, disturbances and observation noise from one explicit
    /// stream. Dynamics, rewards and boundary semantics match `step`.
    pub fn step_with_rng<R: Rng + ?Sized>(&mut self, action: f32, rng: &mut R) -> StepResult {
        let clipped_action = action.clamp(-self.config.max_force, self.config.max_force);
        let action_noise = sample_symmetric(rng, self.config.action_noise_force_n);
        let disturbance =
            if rng.gen_bool(self.config.disturbance_probability_per_step.clamp(0.0, 1.0) as f64) {
                sample_symmetric(rng, self.config.disturbance_force_n)
            } else {
                0.0
            };
        let applied_force = clipped_action + action_noise + disturbance;
        let (a, b) = self.model.model(self.config.dt);
        self.state = a * self.state + b * applied_force;
        self.steps += 1;

        let terminated = self.state[2].abs() > self.config.max_angle_rad
            || self.state[0].abs() > self.config.max_position_m;
        let truncated = !terminated && self.steps >= self.config.max_steps;
        let reward = self.reward(clipped_action, terminated);

        StepResult {
            observation: self.observation_with_rng(rng),
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

fn sample_symmetric<R: Rng + ?Sized>(rng: &mut R, magnitude: f32) -> f32 {
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

    fn boundary_env(max_steps: usize) -> PendulumEnv {
        PendulumEnv::new(
            Model::default(),
            PendulumEnvConfig {
                max_steps,
                max_angle_rad: 1.0,
                max_position_m: 1.0,
                reset_position_range_m: 0.0,
                reset_velocity_range_mps: 0.0,
                reset_angle_range_rad: 0.0,
                reset_angular_velocity_range_radps: 0.0,
                observation_position_noise_m: 0.0,
                observation_velocity_noise_mps: 0.0,
                observation_angle_noise_rad: 0.0,
                observation_angular_velocity_noise_radps: 0.0,
                action_noise_force_n: 0.0,
                disturbance_force_n: 0.0,
                disturbance_probability_per_step: 0.0,
                ..PendulumEnvConfig::default()
            },
        )
    }

    #[test]
    fn step_flags_distinguish_continuation_and_timeout() {
        let mut env = boundary_env(3);
        for index in 1..=3 {
            let result = env.step(1.0);
            assert_eq!(result.done, index == 3);
            assert_eq!(result.truncated, index == 3);
            assert!(!result.terminated());
            assert_eq!(
                result.observation,
                env.observation(),
                "step returns pre-reset observation"
            );
        }
    }

    #[test]
    fn state_failure_ends_before_limit() {
        for axis in [0, 2] {
            let mut env = boundary_env(10);
            env.state[axis] = 2.0;
            let result = env.step(0.0);
            assert!(result.done && result.terminated());
            assert!(!result.truncated);
            assert_eq!(result.reward, -10.0);
        }
    }

    #[test]
    fn failure_on_time_limit_is_terminal() {
        for axis in [0, 2] {
            let mut env = boundary_env(1);
            env.state[axis] = 2.0;
            let result = env.step(0.0);
            assert!(result.terminated(), "time-limit failure must terminate");
            assert!(result.done && !result.truncated);
            assert_eq!(result.reward, -10.0);
        }
    }

    #[test]
    fn step_result_literal_retains_existing_fields() {
        let result = StepResult {
            observation: [0.0; 4],
            reward: 1.0,
            done: true,
            truncated: true,
        };
        assert!(!result.terminated());
        assert!(StepResult {
            truncated: false,
            ..result
        }
        .terminated());
        assert!(!StepResult {
            done: false,
            truncated: false,
            ..result
        }
        .terminated());
    }

    #[test]
    fn reset_starts_fresh_time_limit() {
        let mut env = boundary_env(2);
        assert!(!env.step(0.0).done);
        assert!(env.step(0.0).truncated);
        assert_eq!(env.reset(), [0.0; 4]);
        assert!(!env.step(0.0).done);
        assert!(env.step(0.0).truncated);
    }
}
