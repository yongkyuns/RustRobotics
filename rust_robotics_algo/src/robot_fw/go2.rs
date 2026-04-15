use super::{
    oscillator, quaternion_yaw, rotate_vector_by_inverse_quaternion, shift_history_12,
    shift_history_3, Actuation, Command, InferenceBackend, InferenceInput, Observation,
    PolicyOutput, RawState,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Go2CommandMode {
    Velocity,
    Impedance,
}

pub struct Go2Controller {
    command_mode: Go2CommandMode,
    default_jpos: [f32; 12],
    action_scale: [f32; 12],
    kp: [f32; 12],
    kd: [f32; 12],
    last_actions: [f32; 12],
    action_history: [[f32; 12]; 3],
    gravity_history: [[f32; 3]; 3],
    joint_pos_history: [[f32; 12]; 3],
    joint_vel_history: [[f32; 12]; 3],
    adapt_hx: [f32; 128],
    is_init: bool,
}

impl Go2Controller {
    pub fn new(
        command_mode: Go2CommandMode,
        default_jpos: [f32; 12],
        action_scale: [f32; 12],
        kp: [f32; 12],
        kd: [f32; 12],
    ) -> Self {
        Self {
            command_mode,
            default_jpos,
            action_scale,
            kp,
            kd,
            last_actions: [0.0; 12],
            action_history: [[0.0; 12]; 3],
            gravity_history: [[0.0, 0.0, -1.0]; 3],
            joint_pos_history: [[0.0; 12]; 3],
            joint_vel_history: [[0.0; 12]; 3],
            adapt_hx: [0.0; 128],
            is_init: true,
        }
    }

    pub fn reset(&mut self) {
        self.last_actions = [0.0; 12];
        self.action_history = [[0.0; 12]; 3];
        self.gravity_history = [[0.0, 0.0, -1.0]; 3];
        self.joint_pos_history = [[0.0; 12]; 3];
        self.joint_vel_history = [[0.0; 12]; 3];
        self.adapt_hx = [0.0; 128];
        self.is_init = true;
    }

    pub fn update_from_raw_state(&mut self, raw: &RawState) {
        let gravity = rotate_vector_by_inverse_quaternion(raw.base_quat, [0.0, 0.0, -1.0]);
        shift_history_3(&mut self.gravity_history, gravity);
        shift_history_12(&mut self.joint_pos_history, raw.joint_pos);
        shift_history_12(&mut self.joint_vel_history, raw.joint_vel);
    }

    pub fn build_observation(&self) -> Observation {
        let mut out = Vec::with_capacity(117);

        for step in &self.gravity_history {
            out.extend_from_slice(step);
        }
        for step in &self.joint_pos_history {
            out.extend_from_slice(step);
        }
        for step in &self.joint_vel_history {
            out.extend_from_slice(step);
        }
        for joint in 0..12 {
            for step in &self.action_history {
                out.push(step[joint]);
            }
        }

        Observation { values: out }
    }

    pub fn build_command_input(&self, raw: &RawState, command: &Command) -> Observation {
        match self.command_mode {
            Go2CommandMode::Velocity => self.build_velocity_command_input(raw, command),
            Go2CommandMode::Impedance => self.build_impedance_command_input(raw, command),
        }
    }

    fn build_velocity_command_input(&self, raw: &RawState, command: &Command) -> Observation {
        let command_body =
            rotate_vector_by_inverse_quaternion(raw.base_quat, [command.vel_x, command.vel_y, 0.0]);
        let yaw = quaternion_yaw(raw.base_quat);
        let osc = oscillator(raw.sim_time_s);

        Observation {
            values: vec![
                command_body[0],
                command_body[1],
                -yaw,
                command.yaw_rate,
                osc[0],
                osc[1],
                osc[2],
                osc[3],
                osc[4],
                osc[5],
                osc[6],
                osc[7],
                osc[8],
                osc[9],
                osc[10],
                osc[11],
            ],
        }
    }

    fn build_impedance_command_input(&self, raw: &RawState, command: &Command) -> Observation {
        let kp = 24.0f32;
        let kd = 1.8 * kp.sqrt();
        let osc = oscillator(raw.sim_time_s);
        let base_pos = raw.base_pos;
        let setpoint_world = command.setpoint_world.unwrap_or([
            base_pos[0] + command.vel_x * (kd / kp),
            base_pos[1] + command.vel_y * (kd / kp),
            0.0,
        ]);
        let mut setpoint_body = rotate_vector_by_inverse_quaternion(
            raw.base_quat,
            [
                setpoint_world[0] - base_pos[0],
                setpoint_world[1] - base_pos[1],
                setpoint_world[2] - base_pos[2],
            ],
        );
        let norm = (setpoint_body[0] * setpoint_body[0]
            + setpoint_body[1] * setpoint_body[1]
            + setpoint_body[2] * setpoint_body[2])
            .sqrt();
        if norm > 1e-6 {
            let scale = norm.min(2.0) / norm;
            setpoint_body[0] *= scale;
            setpoint_body[1] *= scale;
            setpoint_body[2] *= scale;
        }
        let yaw = quaternion_yaw(raw.base_quat);
        let mass = 1.0f32;

        Observation {
            values: vec![
                setpoint_body[0],
                setpoint_body[1],
                -yaw,
                kp * setpoint_body[0],
                kp * setpoint_body[1],
                kd,
                kd,
                kd,
                kp * -yaw,
                mass,
                (kp * setpoint_body[0]) / mass,
                (kp * setpoint_body[1]) / mass,
                kd / mass,
                kd / mass,
                kd / mass,
                osc[0],
                osc[1],
                osc[2],
                osc[3],
                osc[4],
                osc[5],
                osc[6],
                osc[7],
                osc[8],
                osc[9],
                osc[10],
                osc[11],
            ],
        }
    }

    pub fn is_init(&self) -> bool {
        self.is_init
    }

    pub fn adapt_hx(&self) -> &[f32; 128] {
        &self.adapt_hx
    }

    pub fn integrate_policy_output(&mut self, output: &PolicyOutput) -> Result<(), String> {
        if output.actions.len() < 12 {
            return Err(format!(
                "Policy returned {} actions, expected at least 12",
                output.actions.len()
            ));
        }
        if output.recurrent.len() < 128 {
            return Err(format!(
                "Policy returned {} recurrent values, expected at least 128",
                output.recurrent.len()
            ));
        }

        let mut smoothed = [0.0f32; 12];
        for (i, slot) in smoothed.iter_mut().enumerate() {
            *slot = self.last_actions[i] * 0.2 + output.actions[i] * 0.8;
        }

        self.last_actions = smoothed;
        shift_history_12(&mut self.action_history, self.last_actions);
        self.adapt_hx.copy_from_slice(&output.recurrent[0..128]);
        self.is_init = false;
        Ok(())
    }

    pub fn step(
        &mut self,
        raw: &RawState,
        command: &Command,
        inference: &mut dyn InferenceBackend,
    ) -> Result<Actuation, String> {
        self.update_from_raw_state(raw);
        let observation = self.build_observation();
        let command_input = self.build_command_input(raw, command);
        let output = inference.run(InferenceInput::Go2 {
            policy: &observation.values,
            is_init: self.is_init(),
            adapt_hx: self.adapt_hx(),
            command: &command_input.values,
        })?;
        self.integrate_policy_output(&output)?;
        Ok(self.decode_actuation(raw))
    }

    pub fn decode_actuation(&self, raw: &RawState) -> Actuation {
        let mut torques = [0.0f32; 12];
        for i in 0..12 {
            let target = self.action_scale[i] * self.last_actions[i] + self.default_jpos[i];
            torques[i] =
                self.kp[i] * (target - raw.joint_pos[i]) + self.kd[i] * (0.0 - raw.joint_vel[i]);
        }
        Actuation::JointTorques(torques)
    }

    pub fn last_actions(&self) -> &[f32; 12] {
        &self.last_actions
    }

    pub fn command_mode(&self) -> Go2CommandMode {
        self.command_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn raw_state() -> RawState {
        RawState {
            sim_time_s: 0.25,
            base_pos: [1.0, -2.0, 0.3],
            base_quat: [1.0, 0.0, 0.0, 0.0],
            base_lin_vel: [0.0; 3],
            base_ang_vel: [0.0; 3],
            joint_pos: [0.1; 12],
            joint_vel: [0.2; 12],
            joint_pos_dyn: vec![0.1; 12],
            joint_vel_dyn: vec![0.2; 12],
            sensor_values: BTreeMap::new(),
            qpos: vec![],
            qvel: vec![],
        }
    }

    #[test]
    fn go2_observation_has_expected_shape_and_front_loaded_history() {
        let mut controller = Go2Controller::new(
            Go2CommandMode::Velocity,
            [0.0; 12],
            [1.0; 12],
            [20.0; 12],
            [0.5; 12],
        );

        controller.update_from_raw_state(&raw_state());
        let obs = controller.build_observation();

        assert_eq!(obs.values.len(), 117);
        assert_eq!(&obs.values[0..3], &[0.0, 0.0, -1.0]);
        assert!(obs.values[9..21].iter().all(|v| (*v - 0.1).abs() < 1e-6));
        assert!(obs.values[21..45].iter().all(|v| v.abs() < 1e-6));
        assert!(obs.values[45..57].iter().all(|v| (*v - 0.2).abs() < 1e-6));
        assert!(obs.values[57..81].iter().all(|v| v.abs() < 1e-6));
        assert!(obs.values[81..117].iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn velocity_command_keeps_body_frame_velocity_and_yaw_rate() {
        let controller = Go2Controller::new(
            Go2CommandMode::Velocity,
            [0.0; 12],
            [1.0; 12],
            [20.0; 12],
            [0.5; 12],
        );
        let raw = raw_state();
        let command = Command {
            vel_x: 0.4,
            vel_y: -0.2,
            yaw_rate: 0.7,
            setpoint_world: None,
        };

        let obs = controller.build_command_input(&raw, &command);

        assert_eq!(obs.values.len(), 16);
        assert!((obs.values[0] - 0.4).abs() < 1e-6);
        assert!((obs.values[1] + 0.2).abs() < 1e-6);
        assert!(obs.values[2].abs() < 1e-6);
        assert!((obs.values[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn impedance_command_clamps_setpoint_radius_in_body_frame() {
        let controller = Go2Controller::new(
            Go2CommandMode::Impedance,
            [0.0; 12],
            [1.0; 12],
            [20.0; 12],
            [0.5; 12],
        );
        let raw = raw_state();
        let command = Command {
            vel_x: 0.0,
            vel_y: 0.0,
            yaw_rate: 0.0,
            setpoint_world: Some([10.0, -10.0, 0.3]),
        };

        let obs = controller.build_command_input(&raw, &command);
        let norm = (obs.values[0] * obs.values[0] + obs.values[1] * obs.values[1]).sqrt();

        assert_eq!(obs.values.len(), 27);
        assert!((norm - 2.0).abs() < 1e-4);
    }

    #[test]
    fn integrate_and_decode_produce_smoothed_joint_torques() {
        let mut controller = Go2Controller::new(
            Go2CommandMode::Velocity,
            [0.1; 12],
            [0.5; 12],
            [10.0; 12],
            [1.0; 12],
        );
        let raw = raw_state();
        let output = PolicyOutput {
            actions: vec![1.0; 12],
            recurrent: vec![0.25; 128],
        };

        controller.integrate_policy_output(&output).unwrap();
        let Actuation::JointTorques(torques) = controller.decode_actuation(&raw) else {
            panic!("expected joint torques");
        };

        assert!(!controller.is_init());
        assert!((controller.last_actions()[0] - 0.8).abs() < 1e-6);
        assert!((controller.adapt_hx()[0] - 0.25).abs() < 1e-6);
        assert!((torques[0] - 3.8).abs() < 1e-6);
    }
}
