use super::{Actuation, Command, Observation, PolicyOutput, RawState};

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

    pub fn reset(&mut self) {
        self.last_actions.fill(0.0);
        self.last_last_actions.fill(0.0);
        self.last_last_last_actions.fill(0.0);
        self.motor_targets.clone_from(&self.default_joint_pos);
        self.prev_motor_targets.clone_from(&self.default_joint_pos);
        self.imitation_index = 0;
        self.imitation_phase = [1.0, 0.0];
    }

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

    pub fn integrate_policy_output(&mut self, output: &PolicyOutput) -> Result<(), String> {
        if output.actions.len() < self.default_joint_pos.len() {
            return Err(format!(
                "Policy returned {} actions, expected at least {}",
                output.actions.len(),
                self.default_joint_pos.len()
            ));
        }

        self.last_last_last_actions.clone_from(&self.last_last_actions);
        self.last_last_actions.clone_from(&self.last_actions);
        self.last_actions.copy_from_slice(&output.actions[..self.default_joint_pos.len()]);
        self.update_phase();
        Ok(())
    }

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

    pub fn last_actions(&self) -> &[f32] {
        &self.last_actions
    }

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

    fn update_phase(&mut self) {
        let phase_steps = self.phase_steps.max(1);
        self.imitation_index = (self.imitation_index + 1) % phase_steps;
        let phase = (self.imitation_index as f32 / phase_steps as f32) * std::f32::consts::TAU;
        self.imitation_phase = [phase.cos(), phase.sin()];
    }

    fn sensor3(&self, raw: &RawState, name: &str) -> [f32; 3] {
        let values = raw.sensor_values.get(name);
        [
            values.and_then(|v| v.first()).copied().unwrap_or(0.0),
            values.and_then(|v| v.get(1)).copied().unwrap_or(0.0),
            values.and_then(|v| v.get(2)).copied().unwrap_or(0.0),
        ]
    }
}
