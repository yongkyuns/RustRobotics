#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use rust_robotics_algo::mujoco::duck::DuckController;
use rust_robotics_algo::mujoco::go2::{Go2CommandMode, Go2Controller};
use rust_robotics_algo::mujoco::{
    Actuation, Command, PolicyOutput, RawState,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

thread_local! {
    static NEXT_HANDLE: Cell<u32> = const { Cell::new(1) };
    static CONTROLLERS: RefCell<BTreeMap<u32, WebRobotController>> = RefCell::new(BTreeMap::new());
}

enum WebRobotController {
    Go2(Go2Controller),
    Duck(DuckController),
}

#[derive(Deserialize)]
struct WebControllerConfig {
    controller_kind: String,
    command_mode: Option<String>,
    default_joint_pos: Vec<f32>,
    action_scale: f32,
    stiffness: f32,
    damping: f32,
    phase_steps: Option<usize>,
    timestep: f32,
    decimation: usize,
}

#[derive(Deserialize, Default)]
struct WebCommandPayload {
    vel_x: f32,
    vel_y: f32,
    yaw_rate: f32,
    setpoint_world: Option<[f32; 3]>,
}

#[derive(Deserialize)]
struct WebRawStatePayload {
    sim_time_s: f32,
    base_pos: [f32; 3],
    base_quat: [f32; 4],
    base_lin_vel: [f32; 3],
    base_ang_vel: [f32; 3],
    joint_pos: Vec<f32>,
    joint_vel: Vec<f32>,
    joint_pos_dyn: Vec<f32>,
    joint_vel_dyn: Vec<f32>,
    sensor_values: BTreeMap<String, Vec<f32>>,
    qpos: Vec<f32>,
    qvel: Vec<f32>,
}

#[derive(Deserialize)]
struct WebPolicyOutputPayload {
    actions: Vec<f32>,
    #[serde(default)]
    recurrent: Vec<f32>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WebPrepareOutput {
    Go2 {
        policy: Vec<f32>,
        is_init: bool,
        adapt_hx: Vec<f32>,
        command: Vec<f32>,
    },
    Duck {
        observation: Vec<f32>,
    },
}

#[derive(Serialize)]
struct WebFinishOutput {
    actuation_kind: &'static str,
    values: Vec<f32>,
    last_action_preview: Vec<f32>,
}

fn js_error(message: impl Into<String>) -> JsValue {
    JsValue::from_str(&message.into())
}

fn from_js<T: for<'de> Deserialize<'de>>(value: JsValue) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(value).map_err(|err| js_error(err.to_string()))
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(|err| js_error(err.to_string()))
}

fn normalize_fixed_12(values: &[f32], fallback: &[f32]) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    for (idx, slot) in out.iter_mut().enumerate() {
        *slot = values
            .get(idx)
            .copied()
            .or_else(|| fallback.get(idx).copied())
            .unwrap_or(0.0);
    }
    out
}

fn into_raw_state(payload: WebRawStatePayload) -> RawState {
    let joint_pos = normalize_fixed_12(&payload.joint_pos, &payload.joint_pos_dyn);
    let joint_vel = normalize_fixed_12(&payload.joint_vel, &payload.joint_vel_dyn);
    RawState {
        sim_time_s: payload.sim_time_s,
        base_pos: payload.base_pos,
        base_quat: payload.base_quat,
        base_lin_vel: payload.base_lin_vel,
        base_ang_vel: payload.base_ang_vel,
        joint_pos,
        joint_vel,
        joint_pos_dyn: payload.joint_pos_dyn,
        joint_vel_dyn: payload.joint_vel_dyn,
        sensor_values: payload.sensor_values,
        qpos: payload.qpos,
        qvel: payload.qvel,
    }
}

fn into_command(payload: WebCommandPayload) -> Command {
    Command {
        vel_x: payload.vel_x,
        vel_y: payload.vel_y,
        yaw_rate: payload.yaw_rate,
        setpoint_world: payload.setpoint_world,
    }
}

fn command_mode_from_config(config: &WebControllerConfig) -> Go2CommandMode {
    match config.command_mode.as_deref() {
        Some("impedance") => Go2CommandMode::Impedance,
        _ => Go2CommandMode::Velocity,
    }
}

fn with_controller_mut<T>(
    handle: u32,
    f: impl FnOnce(&mut WebRobotController) -> Result<T, JsValue>,
) -> Result<T, JsValue> {
    CONTROLLERS.with(|controllers| {
        let mut controllers = controllers.borrow_mut();
        let controller = controllers
            .get_mut(&handle)
            .ok_or_else(|| js_error(format!("unknown robot controller handle {handle}")))?;
        f(controller)
    })
}

#[wasm_bindgen]
pub fn rust_robotics_fw_create_controller(config: JsValue) -> Result<u32, JsValue> {
    let config: WebControllerConfig = from_js(config)?;
    let controller = if config.controller_kind == "open_duck_mini_walk" {
        WebRobotController::Duck(DuckController::new(
            config.default_joint_pos.clone(),
            config.action_scale,
            config.phase_steps.unwrap_or(0),
            config.timestep,
            config.decimation,
        ))
    } else {
        if config.default_joint_pos.len() != 12 {
            return Err(js_error(format!(
                "expected 12 default joint positions for Go2, found {}",
                config.default_joint_pos.len()
            )));
        }
        let mut default_jpos = [0.0f32; 12];
        default_jpos.copy_from_slice(&config.default_joint_pos);
        WebRobotController::Go2(Go2Controller::new(
            command_mode_from_config(&config),
            default_jpos,
            [config.action_scale; 12],
            [config.stiffness; 12],
            [config.damping; 12],
        ))
    };

    let handle = NEXT_HANDLE.with(|next| {
        let handle = next.get();
        next.set(handle.saturating_add(1).max(1));
        handle
    });
    CONTROLLERS.with(|controllers| {
        controllers.borrow_mut().insert(handle, controller);
    });
    Ok(handle)
}

#[wasm_bindgen]
pub fn rust_robotics_fw_reset_controller(handle: u32) -> Result<(), JsValue> {
    with_controller_mut(handle, |controller| {
        match controller {
            WebRobotController::Go2(controller) => controller.reset(),
            WebRobotController::Duck(controller) => controller.reset(),
        }
        Ok(())
    })
}

#[wasm_bindgen]
pub fn rust_robotics_fw_destroy_controller(handle: u32) -> bool {
    CONTROLLERS.with(|controllers| controllers.borrow_mut().remove(&handle).is_some())
}

#[wasm_bindgen]
pub fn rust_robotics_fw_prepare_step(
    handle: u32,
    raw_state: JsValue,
    command: JsValue,
) -> Result<JsValue, JsValue> {
    let raw_state = into_raw_state(from_js(raw_state)?);
    let command = into_command(from_js(command)?);

    with_controller_mut(handle, |controller| {
        let output = match controller {
            WebRobotController::Go2(controller) => {
                controller.update_from_raw_state(&raw_state);
                let observation = controller.build_observation();
                let command_input = controller.build_command_input(&raw_state, &command);
                WebPrepareOutput::Go2 {
                    policy: observation.values,
                    is_init: controller.is_init(),
                    adapt_hx: controller.adapt_hx().to_vec(),
                    command: command_input.values,
                }
            }
            WebRobotController::Duck(controller) => WebPrepareOutput::Duck {
                observation: controller.build_observation(&raw_state, &command).values,
            },
        };
        to_js(&output)
    })
}

#[wasm_bindgen]
pub fn rust_robotics_fw_finish_step(
    handle: u32,
    raw_state: JsValue,
    policy_output: JsValue,
) -> Result<JsValue, JsValue> {
    let raw_state = into_raw_state(from_js(raw_state)?);
    let policy_output: WebPolicyOutputPayload = from_js(policy_output)?;
    let policy_output = PolicyOutput {
        actions: policy_output.actions,
        recurrent: policy_output.recurrent,
    };

    with_controller_mut(handle, |controller| {
        let output = match controller {
            WebRobotController::Go2(controller) => {
                controller
                    .integrate_policy_output(&policy_output)
                    .map_err(js_error)?;
                let actuation = controller.decode_actuation(&raw_state);
                let (actuation_kind, values) = match actuation {
                    Actuation::JointTorques(values) => ("joint_torques", values.to_vec()),
                    Actuation::JointPositionTargets(values) => ("joint_position_targets", values),
                };
                WebFinishOutput {
                    actuation_kind,
                    values,
                    last_action_preview: controller.last_actions().to_vec(),
                }
            }
            WebRobotController::Duck(controller) => {
                controller
                    .integrate_policy_output(&policy_output)
                    .map_err(js_error)?;
                let actuation = controller.decode_actuation();
                let (actuation_kind, values) = match actuation {
                    Actuation::JointTorques(values) => ("joint_torques", values.to_vec()),
                    Actuation::JointPositionTargets(values) => ("joint_position_targets", values),
                };
                WebFinishOutput {
                    actuation_kind,
                    values,
                    last_action_preview: controller.last_actions().to_vec(),
                }
            }
        };
        to_js(&output)
    })
}
