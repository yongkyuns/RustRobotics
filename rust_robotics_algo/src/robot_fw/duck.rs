//! Open Duck Mini-specific observation and actuation logic.
//!
//! The duck controller is imitation-policy oriented: it builds a compact
//! observation vector from IMU, joint, action-history, contact, and phase
//! features, then decodes policy outputs into bounded joint-position targets.
//!
//! The main execution path is:
//!
//! 1. collect sensors and joint state into a policy observation
//! 2. run the imitation policy
//! 3. store the newest action in a short action history
//! 4. advance the gait phase embedding
//! 5. decode actions into position targets
//! 6. rate-limit those targets to respect motor-velocity limits
use super::{
    Actuation, Command, InferenceBackend, InferenceInput, Observation, PolicyOutput, RawState,
};

/// Shared controller wrapper for the Open Duck Mini policy.
pub struct DuckController {
    default_joint_pos: Vec<f32>,
    action_scale: f32,
    phase_steps: usize,
    max_motor_velocity: f32,
    last_actions: Vec<f32>,
    last_last_actions: Vec<f32>,
    last_last_last_actions: Vec<f32>,
    motor_targets: Vec<f32>,
    prev_motor_targets: Vec<f32>,
    imitation_index: usize,
    imitation_phase: [f32; 2],
    timestep: f32,
    decimation: usize,
}

impl DuckController {
    /// Creates a controller with the provided nominal joint targets and scaling.
    pub fn new(
        default_joint_pos: Vec<f32>,
        action_scale: f32,
        phase_steps: usize,
        timestep: f32,
        decimation: usize,
    ) -> Self {
        let num_dofs = default_joint_pos.len();
        Self {
            default_joint_pos: default_joint_pos.clone(),
            action_scale,
            phase_steps,
            max_motor_velocity: 5.24,
            last_actions: vec![0.0; num_dofs],
            last_last_actions: vec![0.0; num_dofs],
            last_last_last_actions: vec![0.0; num_dofs],
            motor_targets: default_joint_pos.clone(),
            prev_motor_targets: default_joint_pos,
            imitation_index: 0,
            imitation_phase: [1.0, 0.0],
            timestep,
            decimation,
        }
    }

    /// Resets action histories, phase, and motor targets to defaults.
    pub fn reset(&mut self) {
        self.last_actions.fill(0.0);
        self.last_last_actions.fill(0.0);
        self.last_last_last_actions.fill(0.0);
        self.motor_targets.clone_from(&self.default_joint_pos);
        self.prev_motor_targets.clone_from(&self.default_joint_pos);
        self.imitation_index = 0;
        self.imitation_phase = [1.0, 0.0];
    }

    /// Builds the duck policy observation vector from raw runtime state.
    ///
    /// The vector mixes several sources of information:
    ///
    /// - IMU angular velocity and acceleration
    /// - high-level command features
    /// - joint position and velocity relative to defaults
    /// - a short action history
    /// - current motor targets
    /// - contact indicators
    /// - a 2D sinusoidal phase embedding
    ///
    /// This layout is typical for locomotion policies trained from privileged
    /// simulator data and then deployed from compact proprioceptive features.
    pub fn build_observation(&self, raw: &RawState, command: &Command) -> Observation {
        let gyro = self.sensor3(raw, "imu_gyro");
        let mut accel = self.sensor3(raw, "imu_accel");
        accel[0] += 1.3;
        let commands = self.build_command_vector(raw, command);
        let left_foot_pos = self.sensor3(raw, "left_foot_pos");
        let right_foot_pos = self.sensor3(raw, "right_foot_pos");
        let contacts = [
            if left_foot_pos[2] <= 0.03 { 1.0 } else { 0.0 },
            if right_foot_pos[2] <= 0.03 { 1.0 } else { 0.0 },
        ];

        let num_dofs = self.default_joint_pos.len();
        let mut obs = Vec::with_capacity(101);
        obs.extend_from_slice(&gyro);
        obs.extend_from_slice(&accel);
        obs.extend_from_slice(&commands);
        for i in 0..num_dofs {
            obs.push(raw.joint_pos_dyn.get(i).copied().unwrap_or(0.0) - self.default_joint_pos[i]);
        }
        for i in 0..num_dofs {
            obs.push(raw.joint_vel_dyn.get(i).copied().unwrap_or(0.0) * 0.05);
        }
        obs.extend_from_slice(&self.last_actions);
        obs.extend_from_slice(&self.last_last_actions);
        obs.extend_from_slice(&self.last_last_last_actions);
        obs.extend_from_slice(&self.motor_targets);
        obs.extend_from_slice(&contacts);
        obs.extend_from_slice(&self.imitation_phase);
        Observation { values: obs }
    }

    /// Incorporates a policy output into controller-internal action history.
    ///
    /// The duck controller keeps three lagged action vectors so the next
    /// observation carries temporal action context in addition to the current
    /// motor target.
    pub fn integrate_policy_output(&mut self, output: &PolicyOutput) -> Result<(), String> {
        if output.actions.len() < self.default_joint_pos.len() {
            return Err(format!(
                "Policy returned {} actions, expected at least {}",
                output.actions.len(),
                self.default_joint_pos.len()
            ));
        }

        self.last_last_last_actions
            .clone_from(&self.last_last_actions);
        self.last_last_actions.clone_from(&self.last_actions);
        self.last_actions
            .copy_from_slice(&output.actions[..self.default_joint_pos.len()]);
        self.update_phase();
        Ok(())
    }

    /// Runs one full observe-infer-decode step for the duck policy.
    pub fn step(
        &mut self,
        raw: &RawState,
        command: &Command,
        inference: &mut dyn InferenceBackend,
    ) -> Result<Actuation, String> {
        let observation = self.build_observation(raw, command);
        let output = inference.run(InferenceInput::Duck {
            observation: &observation.values,
        })?;
        self.integrate_policy_output(&output)?;
        Ok(self.decode_actuation())
    }

    /// Decodes the latest policy action into bounded joint-position targets.
    ///
    /// The raw action is interpreted as an offset around `default_joint_pos`,
    /// scaled by `action_scale`. The resulting target is then rate-limited by a
    /// motor-velocity bound:
    ///
    /// `|q_target(k) - q_target(k-1)| <= max_motor_velocity * timestep * decimation`
    ///
    /// This keeps the policy from instantaneously commanding unreachable joint
    /// motions.
    pub fn decode_actuation(&mut self) -> Actuation {
        let delta = self.max_motor_velocity * self.timestep * self.decimation as f32;
        for i in 0..self.default_joint_pos.len() {
            self.motor_targets[i] =
                self.default_joint_pos[i] + self.last_actions[i] * self.action_scale;
            self.motor_targets[i] = self.motor_targets[i].clamp(
                self.prev_motor_targets[i] - delta,
                self.prev_motor_targets[i] + delta,
            );
            self.prev_motor_targets[i] = self.motor_targets[i];
        }
        Actuation::JointPositionTargets(self.motor_targets.clone())
    }

    /// Returns the most recent action vector produced by the policy.
    pub fn last_actions(&self) -> &[f32] {
        &self.last_actions
    }

    /// Builds the compact command vector consumed by the duck policy.
    fn build_command_vector(&self, raw: &RawState, command: &Command) -> [f32; 7] {
        let setpoint = command.setpoint_world.unwrap_or([
            raw.base_pos[0] + command.vel_x,
            raw.base_pos[1] + command.vel_y,
            0.0,
        ]);
        let rel_body = super::rotate_vector_by_inverse_quaternion(
            raw.base_quat,
            [
                setpoint[0] - raw.base_pos[0],
                setpoint[1] - raw.base_pos[1],
                0.0,
            ],
        );
        [
            rel_body[0].clamp(-0.15, 0.15),
            rel_body[1].clamp(-0.2, 0.2),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// Advances the sinusoidal phase embedding used by the imitation policy.
    fn update_phase(&mut self) {
        let phase_steps = self.phase_steps.max(1);
        self.imitation_index = (self.imitation_index + 1) % phase_steps;
        let phase = (self.imitation_index as f32 / phase_steps as f32) * std::f32::consts::TAU;
        self.imitation_phase = [phase.cos(), phase.sin()];
    }

    /// Reads a 3-value sensor from the raw sensor map, defaulting missing values
    /// to zero.
    fn sensor3(&self, raw: &RawState, name: &str) -> [f32; 3] {
        let values = raw.sensor_values.get(name);
        [
            values.and_then(|v| v.first()).copied().unwrap_or(0.0),
            values.and_then(|v| v.get(1)).copied().unwrap_or(0.0),
            values.and_then(|v| v.get(2)).copied().unwrap_or(0.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn raw_state() -> RawState {
        let mut sensor_values = BTreeMap::new();
        sensor_values.insert("imu_gyro".to_string(), vec![0.1, 0.2, 0.3]);
        sensor_values.insert("imu_accel".to_string(), vec![0.4, 0.5, 0.6]);
        sensor_values.insert("left_foot_pos".to_string(), vec![0.0, 0.0, 0.02]);
        sensor_values.insert("right_foot_pos".to_string(), vec![0.0, 0.0, 0.05]);

        RawState {
            sim_time_s: 0.0,
            base_pos: [1.0, 2.0, 0.4],
            base_quat: [1.0, 0.0, 0.0, 0.0],
            base_lin_vel: [0.0; 3],
            base_ang_vel: [0.0; 3],
            joint_pos: [0.0; 12],
            joint_vel: [0.0; 12],
            joint_pos_dyn: vec![0.2, -0.1, 0.05, -0.2],
            joint_vel_dyn: vec![1.0, -1.0, 0.5, -0.5],
            sensor_values,
            qpos: vec![],
            qvel: vec![],
        }
    }

    #[test]
    fn duck_observation_has_expected_shape_and_contacts() {
        let controller = DuckController::new(vec![0.1, -0.2, 0.0, 0.3], 0.4, 8, 0.01, 2);
        let command = Command {
            vel_x: 0.5,
            vel_y: -0.5,
            yaw_rate: 0.0,
            setpoint_world: None,
        };

        let obs = controller.build_observation(&raw_state(), &command);

        assert_eq!(obs.values.len(), 41);
        assert_eq!(&obs.values[0..3], &[0.1, 0.2, 0.3]);
        assert!((obs.values[3] - 1.7).abs() < 1e-6);
        assert_eq!(&obs.values[37..39], &[1.0, 0.0]);
        assert_eq!(&obs.values[39..41], &[1.0, 0.0]);
    }

    #[test]
    fn duck_integrate_and_decode_limit_target_velocity() {
        let mut controller = DuckController::new(vec![0.0, 0.0, 0.0, 0.0], 0.4, 4, 0.01, 2);
        let output = PolicyOutput {
            actions: vec![1.0, -1.0, 0.5, -0.5],
            recurrent: Vec::new(),
        };

        controller.integrate_policy_output(&output).unwrap();
        let Actuation::JointPositionTargets(targets) = controller.decode_actuation() else {
            panic!("expected joint position targets");
        };

        let delta = 5.24 * 0.01 * 2.0;
        assert_eq!(controller.last_actions(), &[1.0, -1.0, 0.5, -0.5]);
        assert!((targets[0] - delta).abs() < 1e-6);
        assert!((targets[1] + delta).abs() < 1e-6);
        assert!((targets[2] - delta).abs() < 1e-6);
        assert!((targets[3] + delta).abs() < 1e-6);
    }

    #[test]
    fn duck_phase_advances_after_policy_integration() {
        let mut controller = DuckController::new(vec![0.0, 0.0], 0.4, 4, 0.01, 1);
        let output = PolicyOutput {
            actions: vec![0.2, -0.2],
            recurrent: Vec::new(),
        };

        controller.integrate_policy_output(&output).unwrap();
        let obs = controller.build_observation(&raw_state(), &Command::default());

        assert!(obs.values[27].abs() < 1e-6);
        assert!((obs.values[28] - 1.0).abs() < 1e-6);
    }
}
