//! Shared robot-framework/controller contracts used by native, web, and
//! hardware-facing adapters.
//!
//! This module is the semantic boundary between a robot world/runtime and a
//! policy/controller implementation. It defines:
//!
//! - commands sent from UI or high-level behaviors
//! - raw world state extracted from MuJoCo or hardware
//! - observation vectors consumed by policies
//! - policy outputs
//! - actuation commands sent back to the world layer
//!
//! The key design rule is that `robot_fw` does not own MuJoCo model/data or
//! rendering. It owns the translation between "what the world knows" and "what
//! the controller needs".
pub mod duck;
pub mod go2;
pub mod onnx;

use std::collections::BTreeMap;

/// High-level command sent into a robot controller.
#[derive(Clone, Copy, Debug, Default)]
pub struct Command {
    pub vel_x: f32,
    pub vel_y: f32,
    pub yaw_rate: f32,
    pub setpoint_world: Option<[f32; 3]>,
}

/// Raw world state extracted from the robot runtime.
///
/// This struct intentionally keeps both fixed-layout robot data and more dynamic
/// vectors / sensor maps so different robots and backends can share one common
/// handoff type.
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

/// Dense observation vector presented to an inference backend.
#[derive(Clone, Debug)]
pub struct Observation {
    pub values: Vec<f32>,
}

/// Raw policy output from an inference backend.
#[derive(Clone, Debug)]
pub struct PolicyOutput {
    pub actions: Vec<f32>,
    pub recurrent: Vec<f32>,
}

/// Typed input variants for backend inference.
pub enum InferenceInput<'a> {
    Go2 {
        policy: &'a [f32],
        is_init: bool,
        adapt_hx: &'a [f32],
        command: &'a [f32],
    },
    Duck {
        observation: &'a [f32],
    },
}

/// Minimal inference backend trait used by robot controllers.
pub trait InferenceBackend {
    fn run(&mut self, input: InferenceInput<'_>) -> Result<PolicyOutput, String>;
}

/// Low-level actuation command returned by a controller.
#[derive(Clone, Debug)]
pub enum Actuation {
    JointTorques([f32; 12]),
    JointPositionTargets(Vec<f32>),
}

/// Basic runtime telemetry collected around one simulation step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    pub sim_time_s: f32,
    pub last_step_ms: f32,
    pub last_policy_ms: f32,
    pub last_render_ms: f32,
}

/// Extracts yaw from a quaternion.
pub fn quaternion_yaw(q: [f32; 4]) -> f32 {
    let [w, x, y, z] = q;
    let sinz_cosp = 2.0 * (x * y + z * w);
    let cosz_cosp = w * w + x * x - y * y - z * z;
    sinz_cosp.atan2(cosz_cosp)
}

/// Rotates a world-space vector by a quaternion.
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

/// Rotates a vector by the inverse of a quaternion.
pub fn rotate_vector_by_inverse_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    rotate_vector_by_quaternion([q[0], -q[1], -q[2], -q[3]], v)
}

/// Generates the 12-value gait oscillator feature block used by the Go2 policy.
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

/// Shifts a short 3-sample history buffer and inserts the newest sample first.
pub fn shift_history_3(history: &mut [[f32; 3]; 3], sample: [f32; 3]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}

/// Shifts a short 3-sample joint-history buffer and inserts the newest sample first.
pub fn shift_history_12(history: &mut [[f32; 12]; 3], sample: [f32; 12]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}
