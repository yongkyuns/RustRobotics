pub mod go2;
pub mod duck;

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct Command {
    pub vel_x: f32,
    pub vel_y: f32,
    pub yaw_rate: f32,
    pub setpoint_world: Option<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct RawState {
    pub sim_time_s: f32,
    pub base_pos: [f32; 3],
    pub base_quat: [f32; 4],
    pub base_lin_vel: [f32; 3],
    pub base_ang_vel: [f32; 3],
    pub joint_pos: [f32; 12],
    pub joint_vel: [f32; 12],
    pub joint_pos_dyn: Vec<f32>,
    pub joint_vel_dyn: Vec<f32>,
    pub sensor_values: BTreeMap<String, Vec<f32>>,
    pub qpos: Vec<f32>,
    pub qvel: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct PolicyOutput {
    pub actions: Vec<f32>,
    pub recurrent: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum Actuation {
    JointTorques([f32; 12]),
    JointPositionTargets(Vec<f32>),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    pub sim_time_s: f32,
    pub last_step_ms: f32,
    pub last_policy_ms: f32,
    pub last_render_ms: f32,
}

pub fn quaternion_yaw(q: [f32; 4]) -> f32 {
    let [w, x, y, z] = q;
    let sinz_cosp = 2.0 * (x * y + z * w);
    let cosz_cosp = w * w + x * x - y * y - z * z;
    sinz_cosp.atan2(cosz_cosp)
}

pub fn rotate_vector_by_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [w, x, y, z] = q;
    let uv = [
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0],
    ];
    let uuv = [
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    ];

    [
        v[0] + 2.0 * (w * uv[0] + uuv[0]),
        v[1] + 2.0 * (w * uv[1] + uuv[1]),
        v[2] + 2.0 * (w * uv[2] + uuv[2]),
    ]
}

pub fn rotate_vector_by_inverse_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    rotate_vector_by_quaternion([q[0], -q[1], -q[2], -q[3]], v)
}

pub fn oscillator(time_s: f32) -> [f32; 12] {
    let omega = 4.0 * std::f32::consts::PI;
    let phases = [
        omega * time_s + std::f32::consts::PI,
        omega * time_s,
        omega * time_s,
        omega * time_s + std::f32::consts::PI,
    ];

    [
        phases[0].sin(),
        phases[1].sin(),
        phases[2].sin(),
        phases[3].sin(),
        phases[0].cos(),
        phases[1].cos(),
        phases[2].cos(),
        phases[3].cos(),
        omega,
        omega,
        omega,
        omega,
    ]
}

pub fn shift_history_3(history: &mut [[f32; 3]; 3], sample: [f32; 3]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}

pub fn shift_history_12(history: &mut [[f32; 12]; 3], sample: [f32; 12]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}
