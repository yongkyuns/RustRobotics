#[cfg(not(target_arch = "wasm32"))]
use egui::emath::GuiRounding;
use egui_plot::{Line, PlotUi};
use egui::{
    pos2, vec2, Align2, Color32, FontId, Painter, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, Ui,
};
#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Ordering;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rust_robotics_algo::robot_fw::duck::DuckController;
use rust_robotics_algo::robot_fw::go2::{Go2CommandMode, Go2Controller};
use rust_robotics_algo::robot_fw::onnx::{NativeOrtBackend, OrtModelMetadata};
use rust_robotics_algo::robot_fw::{Actuation, Command};

#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
use eframe::{egui_wgpu, wgpu};
#[cfg(not(target_arch = "wasm32"))]
use wgpu::util::DeviceExt as _;

#[cfg(not(target_arch = "wasm32"))]
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(not(target_arch = "wasm32"))]
mod sim;
#[cfg(not(target_arch = "wasm32"))]
use sim::MujocoSim;
use crate::data::{IntoValues, TimeTable};
use super::render_scene::{
    append_grid_lines as append_shared_grid_lines, append_primitive_geom,
    append_world_sphere as append_shared_world_sphere,
    display_geom_color as shared_display_geom_color,
    geom_model_matrix as shared_geom_model_matrix, SharedGeomSnapshot, GEOM_BOX, GEOM_CAPSULE,
    GEOM_CYLINDER, GEOM_LINE, GEOM_MESH, GEOM_PLANE, GEOM_SPHERE,
};

#[cfg(not(target_arch = "wasm32"))]
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_imports
)]
mod mujoco_bindings {
    include!(concat!(env!("OUT_DIR"), "/mujoco_bindings.rs"));
}

#[cfg(not(target_arch = "wasm32"))]
use mujoco_bindings::*;

#[cfg(not(target_arch = "wasm32"))]
const MJ_OBJ_JOINT: i32 = mjtObj__mjOBJ_JOINT as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_OBJ_ACTUATOR: i32 = mjtObj__mjOBJ_ACTUATOR as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_OBJ_GEOM: i32 = mjtObj__mjOBJ_GEOM as i32;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
const MJ_CAT_ALL: i32 = mjtCatBit__mjCAT_ALL as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_SPHERE: i32 = mjtGeom__mjGEOM_SPHERE as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_CAPSULE: i32 = mjtGeom__mjGEOM_CAPSULE as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_CYLINDER: i32 = mjtGeom__mjGEOM_CYLINDER as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_BOX: i32 = mjtGeom__mjGEOM_BOX as i32;
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
const MJ_GEOM_PLANE: i32 = mjtGeom__mjGEOM_PLANE as i32;
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
const MJ_GEOM_MESH: i32 = mjtGeom__mjGEOM_MESH as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_LINE: i32 = mjtGeom__mjGEOM_LINE as i32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeRobotPreset {
    Go2,
    OpenDuckMini,
}

impl NativeRobotPreset {
    fn label(self) -> &'static str {
        match self {
            Self::Go2 => "Go2",
            Self::OpenDuckMini => "Open Duck Mini",
        }
    }

    fn default_scene(self) -> PathBuf {
        match self {
            Self::Go2 => local_asset_path("scene.xml"),
            Self::OpenDuckMini => local_duck_asset_path("scene.xml"),
        }
    }

    fn default_policy(self) -> PathBuf {
        match self {
            Self::Go2 => local_asset_path("facet.json"),
            Self::OpenDuckMini => local_duck_asset_path("duck.json"),
        }
    }
}

pub(super) struct NativeMujocoBackend {
    selected_robot: NativeRobotPreset,
    scene_path: String,
    policy_path: String,
    command_vel_x: f32,
    active: bool,
    status: String,
    init_error: Option<String>,
    render_error: Option<String>,
    runtime: Option<MujocoRuntime>,
    plot_history: TimeTable,
}

impl Default for NativeMujocoBackend {
    fn default() -> Self {
        let selected_robot = NativeRobotPreset::Go2;
        Self {
            selected_robot,
            scene_path: selected_robot
                .default_scene()
                .to_string_lossy()
                .into_owned(),
            policy_path: selected_robot
                .default_policy()
                .to_string_lossy()
                .into_owned(),
            command_vel_x: 0.0,
            active: false,
            status: "MuJoCo runtime idle".to_string(),
            init_error: None,
            render_error: None,
            runtime: None,
            plot_history: TimeTable::init_with_names(vec![
                "Base Height",
                "Command Input X",
                "Action[0]",
            ]),
        }
    }
}

impl NativeMujocoBackend {
    pub fn update(&mut self, sim_speed: usize, paused: bool) {
        if !self.active {
            return;
        }

        if paused {
            return;
        }

        let command_vel_x = self.command_vel_x;
        let runtime = match self.ensure_runtime() {
            Ok(runtime) => runtime,
            Err(err) => {
                debug_log(&format!("panel.update: ensure_runtime error: {err}"));
                self.status = err;
                return;
            }
        };

        if let Err(err) = runtime.step(sim_speed, command_vel_x) {
            debug_log(&format!("panel.update: step error: {err}"));
            self.status = err;
        } else {
            let plot_sample = {
                let raw = runtime.sim.read_raw_state();
                let command_input_x = runtime
                    .command
                    .setpoint_world
                    .map(|setpoint| setpoint[0])
                    .unwrap_or(runtime.command.vel_x);
                (
                    runtime.mujoco_time_ms / 1000.0,
                    raw.base_pos[2],
                    command_input_x,
                    runtime.last_action_preview.first().copied().unwrap_or(0.0),
                )
            };
            let (time_s, base_height, command_x, action_0) = plot_sample;
            self.plot_history.add(
                time_s,
                vec![base_height, command_x, action_0],
            );
            self.status = format!(
                "MuJoCo running from {}",
                Path::new(&self.scene_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("scene.xml")
            );
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.ui_native(ui, frame);
    }

    pub fn ui_split(
        &mut self,
        controls_ui: &mut Ui,
        viewport_ui: &mut Ui,
        frame: Option<&eframe::Frame>,
    ) {
        self.ui_native_controls(controls_ui);
        self.ui_native_viewport(viewport_ui, frame);
    }

    pub fn ui_controls(&mut self, ui: &mut Ui) {
        self.ui_native_controls(ui);
    }

    pub fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.ui_native_viewport(ui, frame);
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn set_overlay_occlusions(&mut self, _rects: &[Rect], _interactive: bool) {}

    pub fn reset_state(&mut self) {
        self.plot_history.clear();
        self.runtime = None;
        self.init_error = None;
        self.render_error = None;
        self.status = "MuJoCo reset".to_string();
    }

    pub fn reset_all(&mut self) {
        let active = self.active;
        *self = Self::default();
        self.active = active;
    }

    pub fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        for (index, name) in self.plot_history.names().iter().enumerate() {
            if let Some(values) = self.plot_history.values(index) {
                plot_ui.line(Line::new(name.clone(), values));
            }
        }
    }

    fn ui_native(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        ui.vertical(|ui| {
            self.ui_native_controls(ui);
            ui.separator();
            self.ui_native_viewport(ui, frame);
        });
    }

    fn ui_native_controls(&mut self, ui: &mut Ui) {
        ui.heading("mujoco");
        ui.label("Native MuJoCo running inside RustRobotics.");
        let previous_robot = self.selected_robot;
        egui::ComboBox::from_label("Robot")
            .selected_text(self.selected_robot.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.selected_robot, NativeRobotPreset::Go2, "Go2");
                ui.selectable_value(
                    &mut self.selected_robot,
                    NativeRobotPreset::OpenDuckMini,
                    "Open Duck Mini",
                );
            });
        if self.selected_robot != previous_robot {
            self.scene_path = self
                .selected_robot
                .default_scene()
                .to_string_lossy()
                .into_owned();
            self.policy_path = self
                .selected_robot
                .default_policy()
                .to_string_lossy()
                .into_owned();
            self.runtime = None;
            self.init_error = None;
            self.render_error = None;
            self.status = format!("Switching to {}...", self.selected_robot.label());
        }
        ui.label(match self.selected_robot {
            NativeRobotPreset::Go2 => "Policy: facet",
            NativeRobotPreset::OpenDuckMini => "Policy: BEST_WALK_ONNX",
        });
        ui.label("Control: drag the red setpoint ball in the viewport.");

        if ui.button("Reset view").clicked() {
            if let Some(runtime) = self.runtime.as_mut() {
                runtime.reset_camera();
            }
        }
    }

    fn ui_native_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        let mut desired = ui.available_size_before_wrap();
        desired.x = desired.x.max(320.0);
        desired.y = desired.y.max(240.0);
        if let Some(render_state) = frame.and_then(|frame| frame.wgpu_render_state()) {
            match self.ensure_runtime().and_then(|runtime| {
                runtime.render_wgpu(ui, render_state, desired, false, None)
            }) {
                Ok(()) => {
                    self.render_error = None;
                    return;
                }
                Err(err) => {
                    debug_log(&format!("ui_native: wgpu render error: {err}"));
                    self.render_error = Some(err.clone());
                    self.status = "MuJoCo running with native software fallback renderer".to_string();
                    if let Ok(runtime) = self.ensure_runtime() {
                        runtime.render_software(ui, desired);
                    }
                    return;
                }
            }
        }
        if let Ok(runtime) = self.ensure_runtime() {
            runtime.render_software(ui, desired);
            self.status = "MuJoCo running with native software fallback renderer".to_string();
        }

        let _ = frame;
    }
    fn ensure_runtime(&mut self) -> Result<&mut MujocoRuntime, String> {
        if let Some(err) = &self.init_error {
            return Err(err.clone());
        }

        if self.runtime.is_none() {
            let started = Instant::now();
            debug_log("ensure_runtime: constructing runtime");
            self.status = "MuJoCo runtime loading...".to_string();
            self.runtime = Some(
                MujocoRuntime::load(Path::new(&self.scene_path), Path::new(&self.policy_path))
                    .map_err(|err| {
                        self.init_error = Some(err.clone());
                        err
                    })?,
            );
            debug_log("ensure_runtime: runtime constructed");
            self.status = format!(
                "MuJoCo runtime loaded in {}",
                fmt_duration(started.elapsed())
            );
            self.init_error = None;
            self.render_error = None;
        }

        Ok(self.runtime.as_mut().expect("runtime just initialized"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeMujocoBackend {
    fn _unused_wasm_companion(&mut self) {}
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct PolicyFile {
    onnx: OnnxFile,
    #[serde(default)]
    controller_kind: Option<String>,
    #[serde(default)]
    command_mode: Option<String>,
    #[serde(default)]
    phase_steps: Option<usize>,
    #[serde(default)]
    obs_config: PolicyObsConfig,
    action_scale: f32,
    stiffness: f32,
    damping: f32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize, Default)]
struct PolicyObsConfig {
    #[serde(default)]
    command: Vec<PolicyCommandConfig>,
    #[serde(default, rename = "command_")]
    command_alt: Vec<PolicyCommandConfig>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct PolicyCommandConfig {
    name: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl PolicyFile {
    fn controller_kind(&self) -> &str {
        self.controller_kind.as_deref().unwrap_or_else(|| {
            if self.onnx.meta.in_keys.len() == 1 {
                "open_duck_mini_walk"
            } else {
                "go2_facet"
            }
        })
    }

    fn phase_steps(&self) -> usize {
        self.phase_steps.unwrap_or(0)
    }

    fn command_mode(&self) -> Go2CommandMode {
        if let Some(command_mode) = self.command_mode.as_deref() {
            return if command_mode == "impedance" {
                Go2CommandMode::Impedance
            } else {
                Go2CommandMode::Velocity
            };
        }

        let commands = if !self.obs_config.command.is_empty() {
            &self.obs_config.command
        } else {
            &self.obs_config.command_alt
        };

        if commands.iter().any(|cfg| cfg.name == "ImpedanceCommand") {
            Go2CommandMode::Impedance
        } else {
            Go2CommandMode::Velocity
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct OnnxFile {
    path: String,
    meta: OnnxMeta,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct OnnxMeta {
    #[serde(default)]
    in_keys: Vec<String>,
    out_keys: Vec<OnnxKey>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OnnxKey {
    String(String),
    Path(Vec<String>),
}

#[cfg(not(target_arch = "wasm32"))]
impl OnnxKey {
    fn joined(&self) -> String {
        match self {
            OnnxKey::String(value) => value.clone(),
            OnnxKey::Path(parts) => parts.join("."),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl OnnxMeta {
    fn as_fw_metadata(&self) -> OrtModelMetadata {
        OrtModelMetadata {
            input_keys: self.in_keys.clone(),
            output_keys: self.out_keys.iter().map(OnnxKey::joined).collect(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct AssetMeta {
    joint_names_isaac: Vec<String>,
    default_joint_pos: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
struct MujocoRuntime {
    sim: MujocoSim,
    scene: mjvScene,
    cam: mjvCamera,
    opt: mjvOption,
    policy: NativeOrtBackend,
    robot: NativeRobotController,
    command: Command,
    mujoco_time_ms: f32,
    diagnostics: MujocoDiagnostics,
    last_action_preview: Vec<f32>,
    wgpu_scene: Arc<Mutex<WgpuSceneFrame>>,
    mesh_assets: BTreeMap<usize, MeshAssetCpu>,
    wgpu_renderer: Arc<Mutex<WgpuSceneRenderer>>,
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeRobotController {
    Go2(Go2Controller),
    Duck(DuckController),
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeRobotController {
    fn uses_setpoint_ball(&self) -> bool {
        matches!(self, Self::Duck(_))
            || matches!(
                self,
                Self::Go2(robot) if robot.command_mode() == Go2CommandMode::Impedance
            )
    }

}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct MujocoDiagnostics {
    last_step_ms: f32,
    last_policy_ms: f32,
    last_render_ms: f32,
    frame_count: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for MujocoRuntime {
    fn drop(&mut self) {
        unsafe {
            mjv_freeScene(&mut self.scene);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MujocoRuntime {
    fn load(scene_path: &Path, policy_path: &Path) -> Result<Self, String> {
        debug_log(&format!(
            "runtime.load: begin scene={} policy={}",
            scene_path.display(),
            policy_path.display()
        ));
        let asset_meta_path = scene_path
            .parent()
            .ok_or_else(|| {
                format!(
                    "Scene path has no parent directory: {}",
                    scene_path.display()
                )
            })?
            .join("asset_meta.json");

        let scene_path = scene_path.canonicalize().map_err(|err| {
            format!(
                "Failed to resolve scene path {}: {err}",
                scene_path.display()
            )
        })?;
        let policy_config_path = policy_path.canonicalize().map_err(|err| {
            format!(
                "Failed to resolve policy path {}: {err}",
                policy_path.display()
            )
        })?;

        let policy_file: PolicyFile = serde_json::from_str(
            &std::fs::read_to_string(&policy_config_path).map_err(|err| {
                format!(
                    "Failed to read policy config {}: {err}",
                    policy_config_path.display()
                )
            })?,
        )
        .map_err(|err| {
            format!(
                "Failed to parse policy config {}: {err}",
                policy_config_path.display()
            )
        })?;

        let asset_meta: AssetMeta =
            serde_json::from_str(&std::fs::read_to_string(&asset_meta_path).map_err(|err| {
                format!(
                    "Failed to read asset metadata {}: {err}",
                    asset_meta_path.display()
                )
            })?)
            .map_err(|err| {
                format!(
                    "Failed to parse asset metadata {}: {err}",
                    asset_meta_path.display()
                )
            })?;

        let controller_kind = policy_file.controller_kind().to_string();
        let sensor_names: &[&str] = if controller_kind == "open_duck_mini_walk" {
            &["imu_gyro", "imu_accel", "left_foot_pos", "right_foot_pos"]
        } else {
            &[]
        };

        let sim = MujocoSim::load(&scene_path, &asset_meta.joint_names_isaac, sensor_names)?;
        let mesh_assets = collect_mesh_assets(sim.model_ptr());
        let mut scene = unsafe { std::mem::zeroed::<mjvScene>() };
        let mut cam = unsafe { std::mem::zeroed::<mjvCamera>() };
        let mut opt = unsafe { std::mem::zeroed::<mjvOption>() };
        unsafe {
            mjv_defaultScene(&mut scene);
            mjv_makeScene(sim.model_ptr(), &mut scene, 3000);
            mjv_defaultCamera(&mut cam);
            mjv_defaultFreeCamera(sim.model_ptr(), &mut cam);
            mjv_defaultOption(&mut opt);
        }
        fit_camera_to_model_stat(sim.model_ptr(), &mut cam);

        let robot = if controller_kind == "open_duck_mini_walk" {
            NativeRobotController::Duck(DuckController::new(
                asset_meta.default_joint_pos.clone(),
                policy_file.action_scale,
                policy_file.phase_steps(),
                sim.timestep(),
                sim.decimation(),
            ))
        } else {
            if asset_meta.default_joint_pos.len() != 12 {
                return Err(format!(
                    "Expected 12 default joint positions for Go2, found {}",
                    asset_meta.default_joint_pos.len()
                ));
            }
            let mut default_jpos = [0.0f32; 12];
            default_jpos.copy_from_slice(&asset_meta.default_joint_pos);
            let action_scale = [policy_file.action_scale; 12];
            let kp = [policy_file.stiffness; 12];
            let kd = [policy_file.damping; 12];
            NativeRobotController::Go2(Go2Controller::new(
                policy_file.command_mode(),
                default_jpos,
                action_scale,
                kp,
                kd,
            ))
        };

        let policy_started = Instant::now();
        debug_log("runtime.load: policy load start");
        let policy = NativeOrtBackend::load(
            &scene_path
                .parent()
                .ok_or_else(|| {
                    format!(
                        "Scene path has no parent directory: {}",
                        scene_path.display()
                    )
                })?
                .join(&policy_file.onnx.path),
            &policy_file.onnx.meta.as_fw_metadata(),
            &controller_kind,
        )?;
        let policy_elapsed = policy_started.elapsed();
        debug_log(&format!(
            "runtime.load: policy load done in {}",
            fmt_duration(policy_elapsed)
        ));

        let initial_setpoint = {
            let raw = sim.read_raw_state();
            Some([raw.base_pos[0], raw.base_pos[1], 0.05])
        };

        Ok(Self {
            sim,
            scene,
            cam,
            opt,
            policy,
            robot,
            command: Command {
                setpoint_world: initial_setpoint,
                ..Command::default()
            },
            mujoco_time_ms: 0.0,
            diagnostics: MujocoDiagnostics::default(),
            last_action_preview: Vec::new(),
            wgpu_scene: Arc::new(Mutex::new(WgpuSceneFrame::default())),
            mesh_assets,
            wgpu_renderer: Arc::new(Mutex::new(WgpuSceneRenderer::default())),
        })
    }

    fn reset_camera(&mut self) {
        unsafe {
            mjv_defaultCamera(&mut self.cam);
            mjv_defaultFreeCamera(self.sim.model_ptr(), &mut self.cam);
        }
        fit_camera_to_model_stat(self.sim.model_ptr(), &mut self.cam);
    }

    fn step(&mut self, sim_speed: usize, command_vel_x: f32) -> Result<(), String> {
        let started = Instant::now();
        self.command.vel_x = command_vel_x;
        for _ in 0..sim_speed {
            let policy_started = Instant::now();
            let raw = self.sim.read_raw_state();
            let actuation = match &mut self.robot {
                NativeRobotController::Go2(robot) => robot.step(&raw, &self.command, &mut self.policy)?,
                NativeRobotController::Duck(robot) => {
                    robot.step(&raw, &self.command, &mut self.policy)?
                }
            };
            self.last_action_preview = match &actuation {
                Actuation::JointTorques(values) => values.iter().take(8).copied().collect(),
                Actuation::JointPositionTargets(values) => {
                    values.iter().take(8).copied().collect()
                }
            };
            self.diagnostics.last_policy_ms = policy_started.elapsed().as_secs_f32() * 1000.0;

            for _ in 0..self.sim.decimation() {
                self.sim.apply_actuation(&actuation)?;
                self.sim.step_substeps(1);
                self.mujoco_time_ms += self.sim.timestep() * 1000.0;
            }
        }
        self.diagnostics.last_step_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }


    fn render_software(&mut self, ui: &mut Ui, desired_size: egui::Vec2) {
        self.render_software_2d(ui, desired_size);
        ui.label("WGPU viewport unavailable; showing simplified 2D fallback.");
    }

    fn render_wgpu(
        &mut self,
        ui: &mut Ui,
        render_state: &egui_wgpu::RenderState,
        _desired_size: egui::Vec2,
        diagnostic_colors: bool,
        mesh_filter: Option<usize>,
    ) -> Result<(), String> {
        let started = Instant::now();
        let rect = aligned_viewport_rect(ui);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        self.update_software_camera(ui, &response);
        self.update_visual_scene();
        let frame = self.build_wgpu_scene_frame(
            (rect.width() / rect.height()).max(0.1),
            diagnostic_colors,
            mesh_filter,
        )?;
        {
            let mut scene = self
                .wgpu_scene
                .lock()
                .map_err(|_| "Failed to lock WGPU scene state".to_string())?;
            *scene = frame;
        }
        {
            let mut renderer = self
                .wgpu_renderer
                .lock()
                .map_err(|_| "Failed to lock WGPU renderer".to_string())?;
            renderer.ensure_mesh_assets(&render_state.device, &self.mesh_assets);
        }

        let callback = NativeWgpuCallback {
            rect,
            renderer: self.wgpu_renderer.clone(),
            scene: self.wgpu_scene.clone(),
            target_format: render_state.target_format,
        };
        ui.painter()
            .add(egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
                rect, callback,
            )));

        let overlay = self.software_overlay_text();
        ui.painter().text(
            rect.left_top() + vec2(12.0, 12.0),
            Align2::LEFT_TOP,
            &overlay,
            FontId::monospace(12.0),
            Color32::from_gray(210),
        );

        if let Ok(renderer) = self.wgpu_renderer.lock() {
            self.diagnostics.last_render_ms = renderer.last_render_ms_ms;
        } else {
            self.diagnostics.last_render_ms = started.elapsed().as_secs_f32() * 1000.0;
        }
        self.diagnostics.frame_count += 1;
        Ok(())
    }

    fn render_software_2d(&mut self, ui: &mut Ui, _desired_size: egui::Vec2) {
        let started = Instant::now();
        let rect = aligned_viewport_rect(ui);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        self.update_software_camera(ui, &response);
        self.update_visual_scene();

        let painter = ui.painter_at(rect);
        let camera = SoftwareCamera::from_gl(&self.scene.camera[0], rect.width() / rect.height());

        painter.rect_filled(rect, 6.0, Color32::from_rgb(18, 22, 28));
        self.draw_ground(&painter, rect, camera.as_ref());

        let geoms =
            unsafe { std::slice::from_raw_parts(self.scene.geoms, self.scene.ngeom as usize) };
        let mut draw_order = (0..geoms.len()).collect::<Vec<_>>();
        if let Some(camera) = camera.as_ref() {
            draw_order.sort_by(|a, b| {
                camera
                    .depth_of(geoms[*b].pos)
                    .partial_cmp(&camera.depth_of(geoms[*a].pos))
                    .unwrap_or(Ordering::Equal)
            });
        }

        for idx in draw_order {
            self.draw_geom_software(&painter, rect, &geoms[idx], camera.as_ref());
        }

        self.draw_command_setpoint(&painter, rect, camera.as_ref());

        let overlay = self.software_overlay_text();
        painter.text(
            rect.left_top() + vec2(12.0, 12.0),
            Align2::LEFT_TOP,
            &overlay,
            FontId::monospace(12.0),
            Color32::from_gray(210),
        );

        self.diagnostics.last_render_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.diagnostics.frame_count += 1;
    }

    fn draw_command_setpoint(&self, painter: &Painter, rect: Rect, camera: Option<&SoftwareCamera>) {
        let Some(setpoint) = self.command.setpoint_world else {
            return;
        };
        if let Some(camera) = camera {
            if let Some((setpoint_pos, _)) = camera.project(rect, setpoint)
            {
                let radius = camera.project_radius(rect, setpoint, 0.05).clamp(6.0, 18.0);
                painter.circle_filled(setpoint_pos, radius, Color32::from_rgb(220, 70, 70));
                painter.circle_stroke(
                    setpoint_pos,
                    radius,
                    Stroke::new(1.5, Color32::from_rgb(255, 220, 220)),
                );
            }
        }
    }

    fn update_visual_scene(&mut self) {
        unsafe {
            mjv_updateScene(
                self.sim.model_ptr(),
                self.sim.data_ptr(),
                &self.opt,
                std::ptr::null::<mjvPerturb>(),
                &mut self.cam,
                MJ_CAT_ALL,
                &mut self.scene,
            );
        }
    }

    fn update_software_camera(&mut self, ui: &Ui, response: &egui::Response) {
        if response.double_clicked() {
            self.reset_camera();
        }

        let software_camera = SoftwareCamera::from_gl(&self.scene.camera[0], response.rect.aspect_ratio());
        if self.robot.uses_setpoint_ball() {
            if response.clicked_by(PointerButton::Primary) {
                if let (Some(pointer), Some(camera)) =
                    (response.interact_pointer_pos(), software_camera.as_ref())
                {
                    if let Some(world) = camera.intersect_plane_z(response.rect, pointer, 0.05) {
                        self.command.setpoint_world = Some(world);
                        self.command.vel_x = world[0] - self.sim.read_raw_state().base_pos[0];
                        self.command.vel_y = world[1] - self.sim.read_raw_state().base_pos[1];
                    }
                }
            }
            if response.dragged_by(PointerButton::Primary) {
                if let (Some(pointer), Some(camera)) = (response.interact_pointer_pos(), software_camera.as_ref()) {
                    if let Some(world) = camera.intersect_plane_z(response.rect, pointer, 0.05) {
                        self.command.setpoint_world = Some(world);
                        self.command.vel_x = world[0] - self.sim.read_raw_state().base_pos[0];
                        self.command.vel_y = world[1] - self.sim.read_raw_state().base_pos[1];
                    }
                }
            }
        } else if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            self.cam.azimuth -= delta.x as f64 * 0.25;
            self.cam.elevation = (self.cam.elevation + delta.y as f64 * 0.2).clamp(-89.0, 89.0);
        }

        if response.dragged_by(PointerButton::Secondary) {
            let delta = response.drag_delta();
            self.cam.azimuth -= delta.x as f64 * 0.25;
            self.cam.elevation = (self.cam.elevation - delta.y as f64 * 0.2).clamp(-89.0, 89.0);
        }

        if response.hovered() {
            let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
            if scroll_y.abs() > f32::EPSILON {
                let zoom = (1.0 - scroll_y * 0.0015).clamp(0.8, 1.25);
                self.cam.distance = (self.cam.distance * zoom as f64).clamp(0.35, 40.0);
            }
        }
    }

    fn software_overlay_text(&self) -> String {
        let raw = self.sim.read_raw_state();
        let yaw_deg = quaternion_yaw(raw.base_quat).to_degrees();
        let planar_speed = raw.base_lin_vel[0].hypot(raw.base_lin_vel[1]);
        let (triangles, lines) = self
            .wgpu_scene
            .lock()
            .map(|scene| {
                let mesh_tris = scene
                    .mesh_instances
                    .iter()
                    .map(|instance| {
                        self.mesh_assets
                            .get(&(instance.mesh_id as usize))
                            .map(|asset| asset.vertices.len() / 3)
                            .unwrap_or(0)
                    })
                    .sum::<usize>();
                ((scene.triangles.len() / 3) + mesh_tris, scene.lines.len() / 2)
            })
            .unwrap_or((0, 0));

        let command_hint = if self.robot.uses_setpoint_ball() {
            "Left-drag: setpoint | Right-drag: orbit | Wheel: zoom | Double-click: reset view"
        } else {
            "Drag: orbit | Right-drag: orbit | Wheel: zoom | Double-click: reset view"
        };
        format!(
            concat!(
                "Native MuJoCo fallback renderer\n",
                "Sim t={:.2}s  cmd=({:+.2}, {:+.2})  speed={:.2}  z={:.2}  yaw={:+.1}deg\n",
                "Cam az={:+.1}  el={:+.1}  dist={:.2}  lookat=({:+.2}, {:+.2}, {:+.2})\n",
                "Setpoint {:?}\n",
                "Step {:.2} ms  Policy {:.2} ms  Render {:.2} ms  Frames {}\n",
                "Glow tris {}  lines {}\n",
                "{}"
            ),
            self.mujoco_time_ms / 1000.0,
            self.command.vel_x,
            self.command.vel_y,
            planar_speed,
            raw.base_pos[2],
            yaw_deg,
            self.cam.azimuth as f32,
            self.cam.elevation as f32,
            self.cam.distance as f32,
            self.cam.lookat[0] as f32,
            self.cam.lookat[1] as f32,
            self.cam.lookat[2] as f32,
            self.command.setpoint_world,
            self.diagnostics.last_step_ms,
            self.diagnostics.last_policy_ms,
            self.diagnostics.last_render_ms,
            self.diagnostics.frame_count,
            triangles,
            lines,
            command_hint,
        )
    }

    fn draw_ground(&self, painter: &Painter, rect: Rect, camera: Option<&SoftwareCamera>) {
        if let Some(camera) = camera {
            let stroke = Stroke::new(1.0, Color32::from_gray(40));
            for i in -10..=10 {
                let offset = i as f32 * 0.25;
                let a = camera.project(rect, [-2.5, offset, 0.0]);
                let b = camera.project(rect, [2.5, offset, 0.0]);
                let c = camera.project(rect, [offset, -2.5, 0.0]);
                let d = camera.project(rect, [offset, 2.5, 0.0]);
                if let (Some((a, _)), Some((b, _))) = (a, b) {
                    painter.line_segment([a, b], stroke);
                }
                if let (Some((c, _)), Some((d, _))) = (c, d) {
                    painter.line_segment([c, d], stroke);
                }
            }
            return;
        }

        let stroke = Stroke::new(1.0, Color32::from_gray(45));
        let center = rect.center();
        let spacing = 28.0;
        for i in -8..=8 {
            let offset = i as f32 * spacing;
            painter.line_segment(
                [
                    pos2(rect.left() + 16.0, center.y + offset * 0.35),
                    pos2(rect.right() - 16.0, center.y + offset * 0.35),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    pos2(center.x + offset, rect.top() + 24.0),
                    pos2(center.x + offset * 0.25, rect.bottom() - 24.0),
                ],
                stroke,
            );
        }
    }

    fn draw_geom_software(
        &self,
        painter: &Painter,
        rect: Rect,
        geom: &mjvGeom,
        camera: Option<&SoftwareCamera>,
    ) {
        let color = rgba_to_color32(geom.rgba);
        let Some(camera) = camera else {
            let bounds = scene_bounds(std::slice::from_ref(geom));
            let center = [
                (bounds[0][0] + bounds[1][0]) * 0.5,
                (bounds[0][1] + bounds[1][1]) * 0.5,
                bounds[0][2],
            ];
            let scale = 0.3 * rect.width().min(rect.height());
            let pos = project_point(rect, center, geom.pos, scale);
            painter.circle_filled(pos, (geom.size[0].max(0.03) * scale * 0.05).max(2.0), color);
            return;
        };

        let Some((pos, depth)) = camera.project(rect, geom.pos) else {
            return;
        };
        let axes = geom_axes(geom);
        let shaded_color = shade_color_by_depth(color, depth);

        match geom.type_ {
            MJ_GEOM_SPHERE => {
                painter.circle_filled(
                    pos,
                    camera
                        .project_radius(rect, geom.pos, geom.size[0].max(0.01))
                        .max(1.5),
                    shaded_color,
                );
            }
            MJ_GEOM_CAPSULE | MJ_GEOM_CYLINDER | MJ_GEOM_LINE => {
                let half = [
                    axes[2][0] * geom.size[1],
                    axes[2][1] * geom.size[1],
                    axes[2][2] * geom.size[1],
                ];
                let a_world = [
                    geom.pos[0] - half[0],
                    geom.pos[1] - half[1],
                    geom.pos[2] - half[2],
                ];
                let b_world = [
                    geom.pos[0] + half[0],
                    geom.pos[1] + half[1],
                    geom.pos[2] + half[2],
                ];
                if let (Some((a, _)), Some((b, _))) =
                    (camera.project(rect, a_world), camera.project(rect, b_world))
                {
                    let thickness = camera.project_radius(rect, geom.pos, geom.size[0].max(0.015));
                    painter.line_segment([a, b], Stroke::new(thickness.max(1.5), shaded_color));
                    painter.circle_filled(a, thickness.max(2.0), shaded_color);
                    painter.circle_filled(b, thickness.max(2.0), shaded_color);
                }
            }
            MJ_GEOM_BOX => {
                let mut points = Vec::with_capacity(8);
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        for sz in [-1.0, 1.0] {
                            let corner = [
                                geom.pos[0]
                                    + axes[0][0] * geom.size[0] * sx
                                    + axes[1][0] * geom.size[1] * sy
                                    + axes[2][0] * geom.size[2] * sz,
                                geom.pos[1]
                                    + axes[0][1] * geom.size[0] * sx
                                    + axes[1][1] * geom.size[1] * sy
                                    + axes[2][1] * geom.size[2] * sz,
                                geom.pos[2]
                                    + axes[0][2] * geom.size[0] * sx
                                    + axes[1][2] * geom.size[1] * sy
                                    + axes[2][2] * geom.size[2] * sz,
                            ];
                            if let Some((projected, _)) = camera.project(rect, corner) {
                                points.push(projected);
                            }
                        }
                    }
                }
                let hull = convex_hull(points);
                if hull.len() >= 3 {
                    painter.add(Shape::convex_polygon(
                        hull,
                        shaded_color.gamma_multiply(0.75),
                        Stroke::new(1.0, shaded_color),
                    ));
                } else {
                    painter.circle_filled(
                        pos,
                        camera.project_radius(rect, geom.pos, 0.04),
                        shaded_color,
                    );
                }
            }
            _ => {
                painter.circle_filled(
                    pos,
                    camera
                        .project_radius(rect, geom.pos, geom.size[0].max(0.03))
                        .max((0.5 / depth.max(0.25)) * 6.0),
                    shaded_color,
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MujocoRuntime {
    fn build_wgpu_scene_frame(
        &self,
        aspect: f32,
        diagnostic_colors: bool,
        mesh_filter: Option<usize>,
    ) -> Result<WgpuSceneFrame, String> {
        let camera = SoftwareCamera::from_gl(&self.scene.camera[0], aspect)
            .ok_or_else(|| "MuJoCo scene camera is not available for wgpu renderer".to_string())?;
        let mut frame = WgpuSceneFrame::default();
        let mut depth_points = Vec::new();

        append_shared_grid_lines(&mut frame.lines);
        for line in &frame.lines {
            depth_points.push(line.position);
        }

        let model = self.sim.model_ptr();
        let ngeom = unsafe { (*model).ngeom as usize };
        for geom_id in 0..ngeom {
            let Some(geom) = self.model_geom_instance(geom_id) else {
                continue;
            };
            let shared_geom = SharedGeomSnapshot {
                type_id: geom.type_,
                data_id: geom.dataid,
                size: geom.size,
                rgba: geom.rgba,
                pos: geom.pos,
                mat: geom.mat,
            };

            match geom.type_ {
                GEOM_PLANE => {}
                GEOM_SPHERE | GEOM_CAPSULE | GEOM_CYLINDER | GEOM_BOX | GEOM_LINE => {
                    let start = frame.triangles.len();
                    let start_lines = frame.lines.len();
                    append_primitive_geom(
                        &mut frame.triangles,
                        &mut frame.lines,
                        &shared_geom,
                        diagnostic_colors,
                    );
                    depth_points.extend(frame.triangles[start..].iter().map(|v| v.position));
                    depth_points.extend(frame.lines[start_lines..].iter().map(|v| v.position));
                }
                GEOM_MESH => {
                    if let Some(filter) = mesh_filter {
                        if self.resolve_mesh_id(&geom) != Some(filter) {
                            continue;
                        }
                    }
                    if let Some(instance) = self.build_mesh_instance(&geom, diagnostic_colors) {
                        if let Some(asset) = self.mesh_assets.get(&(instance.mesh_id as usize)) {
                            extend_depth_points_with_mesh_bounds(
                                &mut depth_points,
                                asset,
                                &instance.model,
                            );
                        }
                        frame.mesh_instances.push(instance);
                    }
                }
                MJ_GEOM_LINE => {
                    let start = frame.lines.len();
                    append_line_geom(&mut frame.lines, &geom, diagnostic_colors);
                    depth_points.extend(frame.lines[start..].iter().map(|v| v.position));
                }
                _ => {}
            }
        }

        self.append_command_setpoint_scene(&mut frame.triangles, &mut frame.lines);
        if let Some(setpoint) = self.command.setpoint_world {
            let raw = self.sim.read_raw_state();
            let start = [raw.base_pos[0], raw.base_pos[1], setpoint[2]];
            depth_points.push(start);
            depth_points.push(setpoint);
            let marker_radius = 0.05;
            for sx in [-marker_radius, marker_radius] {
                for sy in [-marker_radius, marker_radius] {
                    for sz in [-marker_radius, marker_radius] {
                        depth_points.push([setpoint[0] + sx, setpoint[1] + sy, setpoint[2] + sz]);
                    }
                }
            }
        }

        frame
            .mesh_instances
            .sort_by_key(|instance| instance.mesh_id);
        frame.view_proj = camera.fitted_view_projection_matrix_points(&depth_points);
        Ok(frame)
    }

    fn append_command_setpoint_scene(
        &self,
        triangles: &mut Vec<GlVertex>,
        _lines: &mut Vec<GlVertex>,
    ) {
        let Some(setpoint) = self.command.setpoint_world else {
            return;
        };
        append_shared_world_sphere(triangles, setpoint, 0.05, [0.92, 0.24, 0.24, 1.0]);
    }

    fn model_geom_instance(&self, geom_id: usize) -> Option<mjvGeom> {
        unsafe {
            let model = self.sim.model_ptr();
            let data = self.sim.data_ptr();
            if geom_id >= (*model).ngeom as usize {
                return None;
            }

            if *(*model).geom_group.add(geom_id) >= 3 {
                return None;
            }

            let type_ = *(*model).geom_type.add(geom_id);
            let dataid = *(*model).geom_dataid.add(geom_id);
            let matid = *(*model).geom_matid.add(geom_id);
            let mut geom = std::mem::zeroed::<mjvGeom>();
            geom.type_ = type_;
            geom.objtype = MJ_OBJ_GEOM;
            geom.objid = geom_id as i32;
            geom.dataid = if dataid >= 0 { dataid * 2 } else { -1 };

            for axis in 0..3 {
                geom.size[axis] = *(*model).geom_size.add(geom_id * 3 + axis) as f32;
                geom.pos[axis] = *(*data).geom_xpos.add(geom_id * 3 + axis) as f32;
            }
            for idx in 0..9 {
                geom.mat[idx] = *(*data).geom_xmat.add(geom_id * 9 + idx) as f32;
            }

            let rgba_src = if matid >= 0 {
                (*model).mat_rgba.add(matid as usize * 4)
            } else {
                (*model).geom_rgba.add(geom_id * 4)
            };
            for idx in 0..4 {
                geom.rgba[idx] = (*rgba_src.add(idx) as f32).clamp(0.0, 1.0);
            }
            if geom.rgba[3] <= 0.01 {
                return None;
            }

            Some(geom)
        }
    }

    fn build_mesh_instance(&self, geom: &mjvGeom, diagnostic_colors: bool) -> Option<MeshInstance> {
        let mesh_id = self.resolve_mesh_id(geom)? as u32;
        let shared_geom = SharedGeomSnapshot {
            type_id: geom.type_,
            data_id: geom.dataid,
            size: geom.size,
            rgba: geom.rgba,
            pos: geom.pos,
            mat: geom.mat,
        };
        let color = shared_display_geom_color(&shared_geom, diagnostic_colors);
        Some(MeshInstance {
            model: shared_geom_model_matrix(&shared_geom),
            color,
            mesh_id,
            _padding: [0; 3],
        })
    }

    fn resolve_mesh_id(&self, geom: &mjvGeom) -> Option<usize> {
        unsafe {
            let model = self.sim.model_ptr();
            if geom.objtype == MJ_OBJ_GEOM && geom.objid >= 0 {
                let geom_id = geom.objid as usize;
                let dataid = *(*model).geom_dataid.add(geom_id);
                if dataid >= 0 {
                    return Some(dataid as usize);
                }
            }

            if geom.dataid >= 0 {
                return Some((geom.dataid as usize) / 2);
            }
        }

        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default, Clone)]
struct WgpuSceneFrame {
    triangles: Vec<GlVertex>,
    lines: Vec<GlVertex>,
    mesh_instances: Vec<MeshInstance>,
    view_proj: [f32; 16],
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWgpuCallback {
    rect: Rect,
    renderer: Arc<Mutex<WgpuSceneRenderer>>,
    scene: Arc<Mutex<WgpuSceneFrame>>,
    target_format: wgpu::TextureFormat,
}

#[cfg(not(target_arch = "wasm32"))]
impl egui_wgpu::CallbackTrait for NativeWgpuCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let render_result = (|| -> Result<wgpu::CommandBuffer, String> {
            let scene = self
                .scene
                .lock()
                .map_err(|_| "Failed to lock WGPU scene frame".to_string())?;
            let mut renderer = self
                .renderer
                .lock()
                .map_err(|_| "Failed to lock WGPU renderer".to_string())?;
            renderer.prepare(
                device,
                queue,
                self.rect,
                screen_descriptor,
                self.target_format,
                &scene,
            )
        })();

        match render_result {
            Ok(command_buffer) => vec![command_buffer],
            Err(err) => {
                if let Ok(mut renderer) = self.renderer.lock() {
                    if renderer.last_error.as_deref() != Some(err.as_str()) {
                        debug_log(&format!("runtime.render_wgpu: prepare error: {err}"));
                    }
                    renderer.last_error = Some(err);
                }
                Vec::new()
            }
        }
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let render_result = (|| -> Result<(), String> {
            let mut renderer = self
                .renderer
                .lock()
                .map_err(|_| "Failed to lock WGPU renderer".to_string())?;
            renderer.paint(render_pass)
        })();

        if let Err(err) = render_result {
            if let Ok(mut renderer) = self.renderer.lock() {
                if renderer.last_error.as_deref() != Some(err.as_str()) {
                    debug_log(&format!("runtime.render_wgpu: paint error: {err}"));
                }
                renderer.last_error = Some(err);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WgpuSceneUniform {
    view_proj: [f32; 16],
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MeshLocalVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MeshInstance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    mesh_id: u32,
    _padding: [u32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
struct MeshAssetCpu {
    vertices: Vec<MeshLocalVertex>,
    local_min: [f32; 3],
    local_max: [f32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
struct MeshAssetGpu {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct WgpuSceneRenderer {
    scene_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    mesh_pipeline: Option<wgpu::RenderPipeline>,
    present_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    triangle_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,
    mesh_instance_buffer: Option<wgpu::Buffer>,
    triangle_capacity: u64,
    line_capacity: u64,
    mesh_instance_capacity: u64,
    sampler: Option<wgpu::Sampler>,
    present_bind_group_layout: Option<wgpu::BindGroupLayout>,
    present_bind_group: Option<wgpu::BindGroup>,
    offscreen_texture: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    offscreen_size: [u32; 2],
    mesh_assets: BTreeMap<usize, MeshAssetGpu>,
    last_error: Option<String>,
    last_render_ms_ms: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl WgpuSceneRenderer {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rect: Rect,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        target_format: wgpu::TextureFormat,
        scene: &WgpuSceneFrame,
    ) -> Result<wgpu::CommandBuffer, String> {
        let started = Instant::now();
        let width = (rect.width() * screen_descriptor.pixels_per_point)
            .round()
            .max(2.0) as u32;
        let height = (rect.height() * screen_descriptor.pixels_per_point)
            .round()
            .max(2.0) as u32;

        self.ensure_initialized(device, target_format)?;
        self.ensure_offscreen_target(device, width, height)?;
        self.ensure_vertex_capacity(device, scene)?;

        if let Some(buffer) = self.uniform_buffer.as_ref() {
            let uniform = WgpuSceneUniform {
                view_proj: scene.view_proj,
            };
            queue.write_buffer(buffer, 0, slice_as_u8(std::slice::from_ref(&uniform)));
        }
        if let Some(buffer) = self.triangle_buffer.as_ref() {
            if !scene.triangles.is_empty() {
                queue.write_buffer(buffer, 0, slice_as_u8(&scene.triangles));
            }
        }
        if let Some(buffer) = self.line_buffer.as_ref() {
            if !scene.lines.is_empty() {
                queue.write_buffer(buffer, 0, slice_as_u8(&scene.lines));
            }
        }
        if let Some(buffer) = self.mesh_instance_buffer.as_ref() {
            if !scene.mesh_instances.is_empty() {
                queue.write_buffer(buffer, 0, slice_as_u8(&scene.mesh_instances));
            }
        }

        let offscreen_view = self
            .offscreen_view
            .as_ref()
            .ok_or_else(|| "WGPU offscreen view missing".to_string())?;
        let depth_view = self
            .depth_view
            .as_ref()
            .ok_or_else(|| "WGPU depth view missing".to_string())?;
        let scene_pipeline = self
            .scene_pipeline
            .as_ref()
            .ok_or_else(|| "WGPU scene pipeline missing".to_string())?;
        let line_pipeline = self
            .line_pipeline
            .as_ref()
            .ok_or_else(|| "WGPU line pipeline missing".to_string())?;
        let uniform_bind_group = self
            .uniform_bind_group
            .as_ref()
            .ok_or_else(|| "WGPU uniform bind group missing".to_string())?;
        let triangle_buffer = self
            .triangle_buffer
            .as_ref()
            .ok_or_else(|| "WGPU triangle buffer missing".to_string())?;
        let line_buffer = self
            .line_buffer
            .as_ref()
            .ok_or_else(|| "WGPU line buffer missing".to_string())?;
        let mesh_pipeline = self
            .mesh_pipeline
            .as_ref()
            .ok_or_else(|| "WGPU mesh pipeline missing".to_string())?;
        let mesh_instance_buffer = self
            .mesh_instance_buffer
            .as_ref()
            .ok_or_else(|| "WGPU mesh instance buffer missing".to_string())?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mujoco_wgpu_prepare"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mujoco_wgpu_offscreen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.07,
                            g: 0.09,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
            pass.set_bind_group(0, uniform_bind_group, &[]);
            if !scene.triangles.is_empty() {
                pass.set_pipeline(scene_pipeline);
                pass.set_vertex_buffer(
                    0,
                    triangle_buffer.slice(..(std::mem::size_of_val(scene.triangles.as_slice()) as u64)),
                );
                pass.draw(0..scene.triangles.len() as u32, 0..1);
            }
            if !scene.lines.is_empty() {
                pass.set_pipeline(line_pipeline);
                pass.set_bind_group(0, uniform_bind_group, &[]);
                pass.set_vertex_buffer(
                    0,
                    line_buffer.slice(..(std::mem::size_of_val(scene.lines.as_slice()) as u64)),
                );
                pass.draw(0..scene.lines.len() as u32, 0..1);
            }
            if !scene.mesh_instances.is_empty() {
                pass.set_pipeline(mesh_pipeline);
                pass.set_bind_group(0, uniform_bind_group, &[]);
                for (mesh_id, range) in mesh_instance_ranges(&scene.mesh_instances) {
                    let Some(mesh_asset) = self.mesh_assets.get(&mesh_id) else {
                        continue;
                    };
                    let start = range.start * std::mem::size_of::<MeshInstance>();
                    let end = range.end * std::mem::size_of::<MeshInstance>();
                    pass.set_vertex_buffer(0, mesh_asset.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, mesh_instance_buffer.slice(start as u64..end as u64));
                    pass.draw(0..mesh_asset.vertex_count, 0..(range.end - range.start) as u32);
                }
            }
        }

        self.last_error = None;
        self.last_render_ms_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok(encoder.finish())
    }

    fn paint(&mut self, render_pass: &mut wgpu::RenderPass<'static>) -> Result<(), String> {
        let present_pipeline = self
            .present_pipeline
            .as_ref()
            .ok_or_else(|| "WGPU present pipeline missing".to_string())?;
        let present_bind_group = self
            .present_bind_group
            .as_ref()
            .ok_or_else(|| "WGPU present bind group missing".to_string())?;
        render_pass.set_pipeline(present_pipeline);
        render_pass.set_bind_group(0, present_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
        Ok(())
    }

    fn ensure_initialized(
        &mut self,
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), String> {
        if self.scene_pipeline.is_some() && self.present_pipeline.is_some() {
            return Ok(());
        }

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mujoco_wgpu_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(WGPU_SCENE_SHADER.into()),
        });
        let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mujoco_wgpu_present_shader"),
            source: wgpu::ShaderSource::Wgsl(WGPU_PRESENT_SHADER.into()),
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mujoco_wgpu_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mujoco_wgpu_uniform_buffer"),
            size: std::mem::size_of::<WgpuSceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mujoco_wgpu_uniform_bind_group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mujoco_wgpu_scene_layout"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
        };
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mujoco_wgpu_scene_pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_lit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mujoco_wgpu_line_pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_unlit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });
        let mesh_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshLocalVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let mesh_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                2 => Float32x4,
                3 => Float32x4,
                4 => Float32x4,
                5 => Float32x4,
                6 => Float32x4
            ],
        };
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mujoco_wgpu_mesh_pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_mesh"),
                buffers: &[mesh_vertex_layout, mesh_instance_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_lit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let present_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mujoco_wgpu_present_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let present_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mujoco_wgpu_present_layout"),
            bind_group_layouts: &[&present_bind_group_layout],
            push_constant_ranges: &[],
        });
        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mujoco_wgpu_present_pipeline"),
            layout: Some(&present_layout),
            vertex: wgpu::VertexState {
                module: &present_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &present_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        self.scene_pipeline = Some(scene_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.present_pipeline = Some(present_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.present_bind_group_layout = Some(present_bind_group_layout);
        self.mesh_instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mujoco_wgpu_mesh_instance_buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.mesh_instance_capacity = 1024;
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mujoco_wgpu_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        Ok(())
    }

    fn ensure_offscreen_target(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if self.offscreen_size == [width, height] && self.present_bind_group.is_some() {
            return Ok(());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mujoco_wgpu_offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mujoco_wgpu_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[wgpu::TextureFormat::Depth24Plus],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let present_bind_group_layout = self
            .present_bind_group_layout
            .as_ref()
            .ok_or_else(|| "WGPU present bind group layout missing".to_string())?;
        let sampler = self
            .sampler
            .as_ref()
            .ok_or_else(|| "WGPU sampler missing".to_string())?;
        let present_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mujoco_wgpu_present_bind_group"),
            layout: present_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        self.offscreen_texture = Some(texture);
        self.offscreen_view = Some(texture_view);
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);
        self.present_bind_group = Some(present_bind_group);
        self.offscreen_size = [width, height];
        Ok(())
    }

    fn ensure_vertex_capacity(
        &mut self,
        device: &wgpu::Device,
        scene: &WgpuSceneFrame,
    ) -> Result<(), String> {
        let triangle_bytes = std::mem::size_of_val(scene.triangles.as_slice()) as u64;
        let line_bytes = std::mem::size_of_val(scene.lines.as_slice()) as u64;
        let mesh_instance_bytes = std::mem::size_of_val(scene.mesh_instances.as_slice()) as u64;

        if self.triangle_buffer.is_none() || triangle_bytes > self.triangle_capacity {
            self.triangle_capacity = triangle_bytes.max(1024).next_power_of_two();
            self.triangle_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mujoco_wgpu_triangle_buffer"),
                size: self.triangle_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if self.line_buffer.is_none() || line_bytes > self.line_capacity {
            self.line_capacity = line_bytes.max(1024).next_power_of_two();
            self.line_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mujoco_wgpu_line_buffer"),
                size: self.line_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if self.mesh_instance_buffer.is_none() || mesh_instance_bytes > self.mesh_instance_capacity {
            self.mesh_instance_capacity = mesh_instance_bytes.max(1024).next_power_of_two();
            self.mesh_instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mujoco_wgpu_mesh_instance_buffer"),
                size: self.mesh_instance_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        Ok(())
    }

    fn ensure_mesh_assets(
        &mut self,
        device: &wgpu::Device,
        mesh_assets: &BTreeMap<usize, MeshAssetCpu>,
    ) {
        for (&mesh_id, asset) in mesh_assets {
            let needs_upload = self
                .mesh_assets
                .get(&mesh_id)
                .map(|gpu_asset| gpu_asset.vertex_count != asset.vertices.len() as u32)
                .unwrap_or(true);
            if needs_upload {
                self.mesh_assets.insert(
                    mesh_id,
                    MeshAssetGpu {
                        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("mujoco_wgpu_mesh_asset_buffer"),
                            contents: slice_as_u8(&asset.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                        vertex_count: asset.vertices.len() as u32,
                    },
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
type GlVertex = super::render_scene::GlVertex;

#[cfg(not(target_arch = "wasm32"))]
fn mesh_instance_ranges(instances: &[MeshInstance]) -> Vec<(usize, std::ops::Range<usize>)> {
    let mut ranges = Vec::new();
    if instances.is_empty() {
        return ranges;
    }
    let mut start = 0usize;
    let mut current = instances[0].mesh_id as usize;
    for (idx, instance) in instances.iter().enumerate().skip(1) {
        let mesh_id = instance.mesh_id as usize;
        if mesh_id != current {
            ranges.push((current, start..idx));
            current = mesh_id;
            start = idx;
        }
    }
    ranges.push((current, start..instances.len()));
    ranges
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_mesh_assets(model: *const mjModel) -> BTreeMap<usize, MeshAssetCpu> {
    let mut assets = BTreeMap::new();
    if model.is_null() {
        return assets;
    }
    unsafe {
        let all_vertices =
            std::slice::from_raw_parts((*model).mesh_vert, (*model).nmeshvert as usize * 3);
        let all_normals =
            std::slice::from_raw_parts((*model).mesh_normal, (*model).nmeshvert as usize * 3);
        let all_faces =
            std::slice::from_raw_parts((*model).mesh_face, (*model).nmeshface as usize * 3);
        for mesh_id in 0..(*model).nmesh as usize {
            let vertadr = *(*model).mesh_vertadr.add(mesh_id) as usize;
            let vertnum = *(*model).mesh_vertnum.add(mesh_id) as usize;
            let faceadr = *(*model).mesh_faceadr.add(mesh_id) as usize;
            let facenum = *(*model).mesh_facenum.add(mesh_id) as usize;
            if vertnum == 0 || facenum == 0 {
                continue;
            }
            let local_vertices = &all_vertices[vertadr * 3..(vertadr + vertnum) * 3];
            let local_normals = &all_normals[vertadr * 3..(vertadr + vertnum) * 3];
            let face_slice = &all_faces[faceadr * 3..(faceadr + facenum) * 3];
            let mut vertices = Vec::with_capacity(facenum * 3);
            let mut local_min = [f32::INFINITY; 3];
            let mut local_max = [f32::NEG_INFINITY; 3];
            for tri in face_slice.chunks_exact(3) {
                for &i in tri {
                    let idx = i as usize;
                    if idx >= vertnum {
                        continue;
                    }
                    let position = mesh_vertex(local_vertices, idx);
                    let normal = mesh_vertex(local_normals, idx);
                    extend_bounds3(&mut local_min, &mut local_max, position);
                    vertices.push(MeshLocalVertex { position, normal });
                }
            }
            assets.insert(
                mesh_id,
                MeshAssetCpu {
                    vertices,
                    local_min,
                    local_max,
                },
            );
        }
    }
    assets
}

#[cfg(not(target_arch = "wasm32"))]
fn extend_depth_points_with_mesh_bounds(
    depth_points: &mut Vec<[f32; 3]>,
    asset: &MeshAssetCpu,
    model: &[[f32; 4]; 4],
) {
    for sx in [asset.local_min[0], asset.local_max[0]] {
        for sy in [asset.local_min[1], asset.local_max[1]] {
            for sz in [asset.local_min[2], asset.local_max[2]] {
                depth_points.push(transform_point_mat4(model, [sx, sy, sz]));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fmt_duration(duration: Duration) -> String {
    format!("{:.2} ms", duration.as_secs_f64() * 1000.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn aligned_viewport_rect(ui: &Ui) -> Rect {
    let pixels_per_point = ui.ctx().pixels_per_point().max(1.0);
    let pixel = 1.0 / pixels_per_point;
    let mut rect = ui
        .available_rect_before_wrap()
        .shrink(pixel)
        .round_to_pixels(pixels_per_point);
    rect.min.x += pixel;
    rect = rect.round_to_pixels(pixels_per_point);
    rect
}

#[cfg(not(target_arch = "wasm32"))]
fn debug_log(message: &str) {
    eprintln!("[mujoco-tab] {message}");
}

fn fit_camera_to_model_stat(model: *const mjModel, cam: &mut mjvCamera) {
    if model.is_null() {
        return;
    }

    unsafe {
        let stat = &(*model).stat;
        cam.lookat[0] = stat.center[0];
        cam.lookat[1] = stat.center[1];
        cam.lookat[2] = stat.center[2];

        let extent = (stat.extent as f64).max(0.25);
        cam.distance = (extent * 2.8).max(1.25);

        if cam.elevation.abs() < 1.0 {
            cam.elevation = -20.0;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn project_point(rect: Rect, center: [f32; 3], point: [f32; 3], scale: f32) -> Pos2 {
    let x = point[0] - center[0];
    let y = point[1] - center[1];
    let z = point[2] - center[2];

    let screen_x = rect.center().x + (x - 0.65 * y) * scale;
    let screen_y = rect.center().y + (0.25 * x + 0.18 * y - 1.2 * z) * scale;
    pos2(screen_x, screen_y)
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct SoftwareCamera {
    eye: [f32; 3],
    forward: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    frustum_center: f32,
    frustum_bottom: f32,
    frustum_top: f32,
    frustum_near: f32,
    frustum_far: f32,
    viewport_aspect: f32,
    orthographic: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl SoftwareCamera {
    fn from_gl(camera: &mjvGLCamera, viewport_aspect: f32) -> Option<Self> {
        let forward = normalize3(camera.forward);
        let up = normalize3(camera.up);
        let right = normalize3(cross3(forward, up));
        if length3(forward) <= f32::EPSILON
            || length3(up) <= f32::EPSILON
            || length3(right) <= f32::EPSILON
        {
            return None;
        }

        Some(Self {
            eye: camera.pos,
            forward,
            up,
            right,
            frustum_center: camera.frustum_center,
            frustum_bottom: camera.frustum_bottom,
            frustum_top: camera.frustum_top,
            frustum_near: camera.frustum_near.max(1e-4),
            frustum_far: camera.frustum_far.max(camera.frustum_near + 1.0),
            viewport_aspect: viewport_aspect.max(0.1),
            orthographic: camera.orthographic != 0,
        })
    }

    fn depth_of(&self, point: [f32; 3]) -> f32 {
        dot3(sub3(point, self.eye), self.forward)
    }

    fn project(&self, rect: Rect, point: [f32; 3]) -> Option<(Pos2, f32)> {
        let rel = sub3(point, self.eye);
        let depth = dot3(rel, self.forward);
        if depth <= 1e-4 {
            return None;
        }

        let x = dot3(rel, self.right);
        let y = dot3(rel, self.up);
        let near_x = if self.orthographic {
            x
        } else {
            x * self.frustum_near / depth
        };
        let near_y = if self.orthographic {
            y
        } else {
            y * self.frustum_near / depth
        };

        let half_height = ((self.frustum_top - self.frustum_bottom) * 0.5)
            .abs()
            .max(1e-4);
        let aspect = (rect.width() / rect.height()).max(0.1);
        let half_width = (half_height * aspect).max(1e-4);
        let center_y = (self.frustum_top + self.frustum_bottom) * 0.5;
        let ndc_x = (near_x - self.frustum_center) / half_width;
        let ndc_y = (near_y - center_y) / half_height;

        Some((
            pos2(
                rect.center().x + ndc_x * rect.width() * 0.5,
                rect.center().y - ndc_y * rect.height() * 0.5,
            ),
            depth,
        ))
    }

    fn project_radius(&self, rect: Rect, point: [f32; 3], radius: f32) -> f32 {
        let center = self.project(rect, point);
        let edge = self.project(rect, add3(point, scale3(self.right, radius)));
        match (center, edge) {
            (Some((center, _)), Some((edge, _))) => center.distance(edge).max(1.0),
            _ => 1.0,
        }
    }

    fn world_ray(&self, rect: Rect, pointer: Pos2) -> Option<([f32; 3], [f32; 3])> {
        if rect.width() <= f32::EPSILON || rect.height() <= f32::EPSILON {
            return None;
        }
        let ndc_x = ((pointer.x - rect.center().x) / (rect.width() * 0.5)).clamp(-1.0, 1.0);
        let ndc_y = (-(pointer.y - rect.center().y) / (rect.height() * 0.5)).clamp(-1.0, 1.0);
        let half_height = ((self.frustum_top - self.frustum_bottom) * 0.5)
            .abs()
            .max(1e-4);
        let half_width = (half_height * self.viewport_aspect).max(1e-4);
        let center_y = (self.frustum_top + self.frustum_bottom) * 0.5;
        let near_x = self.frustum_center + ndc_x * half_width;
        let near_y = center_y + ndc_y * half_height;

        if self.orthographic {
            let origin = add3(
                add3(self.eye, scale3(self.right, near_x)),
                scale3(self.up, near_y),
            );
            Some((origin, self.forward))
        } else {
            let dir = normalize3(add3(
                add3(scale3(self.forward, self.frustum_near), scale3(self.right, near_x)),
                scale3(self.up, near_y),
            ));
            Some((self.eye, dir))
        }
    }

    fn intersect_plane_z(&self, rect: Rect, pointer: Pos2, z: f32) -> Option<[f32; 3]> {
        let (origin, dir) = self.world_ray(rect, pointer)?;
        if dir[2].abs() <= 1e-6 {
            return None;
        }
        let t = (z - origin[2]) / dir[2];
        if t <= 0.0 {
            return None;
        }
        Some(add3(origin, scale3(dir, t)))
    }

    fn view_projection_matrix(&self) -> [f32; 16] {
        mul_mat4(
            self.projection_matrix_with_depth_range(self.frustum_near, self.frustum_far),
            self.view_matrix(),
        )
    }

    fn fitted_view_projection_matrix_points(&self, points: &[[f32; 3]]) -> [f32; 16] {
        let Some((near, far)) = self.fitted_depth_range_points(points) else {
            return self.view_projection_matrix();
        };
        mul_mat4(
            self.projection_matrix_with_depth_range(near, far),
            self.view_matrix(),
        )
    }

    fn fitted_depth_range_points(&self, points: &[[f32; 3]]) -> Option<(f32, f32)> {
        let mut max_depth = f32::NEG_INFINITY;
        for &point in points {
            let depth = self.depth_of(point);
            if depth > 1e-4 {
                max_depth = max_depth.max(depth);
            }
        }
        if !max_depth.is_finite() {
            return None;
        }
        let near = self.frustum_near.max(1e-3);
        let far = (max_depth * 1.2).max(near + 1.0).min(self.frustum_far);
        Some((near, far))
    }

    fn view_matrix(&self) -> [f32; 16] {
        [
            self.right[0],
            self.up[0],
            -self.forward[0],
            0.0,
            self.right[1],
            self.up[1],
            -self.forward[1],
            0.0,
            self.right[2],
            self.up[2],
            -self.forward[2],
            0.0,
            -dot3(self.right, self.eye),
            -dot3(self.up, self.eye),
            dot3(self.forward, self.eye),
            1.0,
        ]
    }

    fn projection_matrix_with_depth_range(&self, near: f32, far: f32) -> [f32; 16] {
        let half_height = ((self.frustum_top - self.frustum_bottom) * 0.5)
            .abs()
            .max(1e-4);
        let half_width = (half_height * self.viewport_aspect).max(1e-4);
        let left = self.frustum_center - half_width;
        let right = self.frustum_center + half_width;
        let bottom = self.frustum_bottom;
        let top = self.frustum_top;
        let near = near.max(1e-4);
        let far = far.max(near + 1.0);

        if self.orthographic {
            [
                2.0 / (right - left),
                0.0,
                0.0,
                0.0,
                0.0,
                2.0 / (top - bottom),
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0 / (far - near),
                0.0,
                -((right + left) / (right - left)),
                -((top + bottom) / (top - bottom)),
                -((far + near) / (far - near)),
                1.0,
            ]
        } else {
            [
                (2.0 * near) / (right - left),
                0.0,
                0.0,
                0.0,
                0.0,
                (2.0 * near) / (top - bottom),
                0.0,
                0.0,
                (right + left) / (right - left),
                (top + bottom) / (top - bottom),
                -((far + near) / (far - near)),
                -1.0,
                0.0,
                0.0,
                -((2.0 * far * near) / (far - near)),
                0.0,
            ]
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mul_mat4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum::<f32>();
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn transform_point_mat4(model: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    [
        model[0][0] * point[0] + model[1][0] * point[1] + model[2][0] * point[2] + model[3][0],
        model[0][1] * point[0] + model[1][1] * point[1] + model[2][1] * point[2] + model[3][1],
        model[0][2] * point[0] + model[1][2] * point[1] + model[2][2] * point[2] + model[3][2],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn append_grid_lines(lines: &mut Vec<GlVertex>) {
    let color = [0.22, 0.28, 0.34, 1.0];
    for i in -16..=16 {
        let offset = i as f32 * 0.25;
        push_line(lines, [-4.0, offset, 0.0], [4.0, offset, 0.0], color);
        push_line(lines, [offset, -4.0, 0.0], [offset, 4.0, 0.0], color);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn append_line_geom(lines: &mut Vec<GlVertex>, geom: &mjvGeom, diagnostic_colors: bool) {
    let axes = geom_axes(geom);
    let half = scale3(axes[2], geom.size[1].max(0.02));
    let a = [
        geom.pos[0] - half[0],
        geom.pos[1] - half[1],
        geom.pos[2] - half[2],
    ];
    let b = [
        geom.pos[0] + half[0],
        geom.pos[1] + half[1],
        geom.pos[2] + half[2],
    ];
    push_line(lines, a, b, display_geom_color(geom, diagnostic_colors));
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn append_box_geom(triangles: &mut Vec<GlVertex>, geom: &mjvGeom, diagnostic_colors: bool) {
    let axes = geom_axes(geom);
    let center = geom.pos;
    let sx = geom.size[0];
    let sy = geom.size[1];
    let sz = geom.size[2];
    let corners = [
        transform_geom_point(geom, [sx, sy, sz]),
        transform_geom_point(geom, [sx, sy, -sz]),
        transform_geom_point(geom, [sx, -sy, sz]),
        transform_geom_point(geom, [sx, -sy, -sz]),
        transform_geom_point(geom, [-sx, sy, sz]),
        transform_geom_point(geom, [-sx, sy, -sz]),
        transform_geom_point(geom, [-sx, -sy, sz]),
        transform_geom_point(geom, [-sx, -sy, -sz]),
    ];
    let color = display_geom_color(geom, diagnostic_colors);
    let faces = [
        ([0, 2, 3, 1], axes[0]),
        ([4, 5, 7, 6], scale3(axes[0], -1.0)),
        ([0, 1, 5, 4], axes[1]),
        ([2, 6, 7, 3], scale3(axes[1], -1.0)),
        ([0, 4, 6, 2], axes[2]),
        ([1, 3, 7, 5], scale3(axes[2], -1.0)),
    ];
    for (indices, normal) in faces {
        push_triangle(
            triangles,
            corners[indices[0]],
            corners[indices[1]],
            corners[indices[2]],
            normalize3(normal),
            color,
        );
        push_triangle(
            triangles,
            corners[indices[0]],
            corners[indices[2]],
            corners[indices[3]],
            normalize3(normal),
            color,
        );
    }
    let _ = center;
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn append_cylinder_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &mjvGeom,
    segments: usize,
    diagnostic_colors: bool,
) {
    let color = display_geom_color(geom, diagnostic_colors);
    let half = geom.size[1].max(0.01);
    let radius = geom.size[0].max(0.01);
    for i in 0..segments {
        let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
        let p0 = [radius * a0.cos(), radius * a0.sin(), -half];
        let p1 = [radius * a1.cos(), radius * a1.sin(), -half];
        let p2 = [radius * a1.cos(), radius * a1.sin(), half];
        let p3 = [radius * a0.cos(), radius * a0.sin(), half];

        let w0 = transform_geom_point(geom, p0);
        let w1 = transform_geom_point(geom, p1);
        let w2 = transform_geom_point(geom, p2);
        let w3 = transform_geom_point(geom, p3);
        let n0 = transform_geom_vector(geom, normalize3([a0.cos(), a0.sin(), 0.0]));
        let n1 = transform_geom_vector(geom, normalize3([a1.cos(), a1.sin(), 0.0]));
        let n_mid = normalize3(add3(n0, n1));

        push_triangle(triangles, w0, w1, w2, n_mid, color);
        push_triangle(triangles, w0, w2, w3, n_mid, color);

        let top_center = transform_geom_point(geom, [0.0, 0.0, half]);
        let bottom_center = transform_geom_point(geom, [0.0, 0.0, -half]);
        push_triangle(
            triangles,
            top_center,
            w3,
            w2,
            transform_geom_vector(geom, [0.0, 0.0, 1.0]),
            color,
        );
        push_triangle(
            triangles,
            bottom_center,
            w1,
            w0,
            transform_geom_vector(geom, [0.0, 0.0, -1.0]),
            color,
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn append_capsule_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &mjvGeom,
    hemi_rings: usize,
    segments: usize,
    diagnostic_colors: bool,
) {
    append_cylinder_geom(triangles, geom, segments, diagnostic_colors);
    let radius = geom.size[0].max(0.01);
    let half = geom.size[1].max(0.01);
    let color = display_geom_color(geom, diagnostic_colors);
    for hemisphere in [-1.0f32, 1.0] {
        for ring in 0..hemi_rings {
            let v0 = ring as f32 / hemi_rings as f32 * std::f32::consts::FRAC_PI_2;
            let v1 = (ring + 1) as f32 / hemi_rings as f32 * std::f32::consts::FRAC_PI_2;
            let z0 = hemisphere * (half + radius * v0.sin());
            let z1 = hemisphere * (half + radius * v1.sin());
            let r0 = radius * v0.cos();
            let r1 = radius * v1.cos();

            for seg in 0..segments {
                let a0 = seg as f32 / segments as f32 * std::f32::consts::TAU;
                let a1 = (seg + 1) as f32 / segments as f32 * std::f32::consts::TAU;

                let p00 = [r0 * a0.cos(), r0 * a0.sin(), z0];
                let p01 = [r0 * a1.cos(), r0 * a1.sin(), z0];
                let p10 = [r1 * a0.cos(), r1 * a0.sin(), z1];
                let p11 = [r1 * a1.cos(), r1 * a1.sin(), z1];

                let n00 = transform_geom_vector(
                    geom,
                    normalize3([p00[0], p00[1], hemisphere * (p00[2] - hemisphere * half)]),
                );
                let n01 = transform_geom_vector(
                    geom,
                    normalize3([p01[0], p01[1], hemisphere * (p01[2] - hemisphere * half)]),
                );
                let n10 = transform_geom_vector(
                    geom,
                    normalize3([p10[0], p10[1], hemisphere * (p10[2] - hemisphere * half)]),
                );
                let n11 = transform_geom_vector(
                    geom,
                    normalize3([p11[0], p11[1], hemisphere * (p11[2] - hemisphere * half)]),
                );

                push_triangle(
                    triangles,
                    transform_geom_point(geom, p00),
                    transform_geom_point(geom, p01),
                    transform_geom_point(geom, p11),
                    normalize3(add3(add3(n00, n01), n11)),
                    color,
                );
                push_triangle(
                    triangles,
                    transform_geom_point(geom, p00),
                    transform_geom_point(geom, p11),
                    transform_geom_point(geom, p10),
                    normalize3(add3(add3(n00, n11), n10)),
                    color,
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn append_sphere_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &mjvGeom,
    rings: usize,
    segments: usize,
    diagnostic_colors: bool,
) {
    let radius = geom.size[0].max(0.01);
    let color = display_geom_color(geom, diagnostic_colors);
    for ring in 0..rings {
        let v0 = ring as f32 / rings as f32 * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let v1 =
            (ring + 1) as f32 / rings as f32 * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let z0 = radius * v0.sin();
        let z1 = radius * v1.sin();
        let r0 = radius * v0.cos();
        let r1 = radius * v1.cos();

        for seg in 0..segments {
            let a0 = seg as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (seg + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            let p00 = [r0 * a0.cos(), r0 * a0.sin(), z0];
            let p01 = [r0 * a1.cos(), r0 * a1.sin(), z0];
            let p10 = [r1 * a0.cos(), r1 * a0.sin(), z1];
            let p11 = [r1 * a1.cos(), r1 * a1.sin(), z1];

            push_triangle(
                triangles,
                transform_geom_point(geom, p00),
                transform_geom_point(geom, p01),
                transform_geom_point(geom, p11),
                normalize3(transform_geom_vector(
                    geom,
                    normalize3(add3(add3(p00, p01), p11)),
                )),
                color,
            );
            push_triangle(
                triangles,
                transform_geom_point(geom, p00),
                transform_geom_point(geom, p11),
                transform_geom_point(geom, p10),
                normalize3(transform_geom_vector(
                    geom,
                    normalize3(add3(add3(p00, p11), p10)),
                )),
                color,
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn world_sphere_geom(pos: [f32; 3], radius: f32, color: [f32; 4]) -> mjvGeom {
    let mut geom = unsafe { std::mem::zeroed::<mjvGeom>() };
    geom.type_ = MJ_GEOM_SPHERE;
    geom.objid = -1;
    geom.dataid = -1;
    geom.pos = pos;
    geom.size[0] = radius;
    geom.size[1] = radius;
    geom.size[2] = radius;
    geom.mat = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    geom.rgba = color;
    geom
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn push_triangle(
    triangles: &mut Vec<GlVertex>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
) {
    triangles.push(GlVertex::world(a, normal, color));
    triangles.push(GlVertex::world(b, normal, color));
    triangles.push(GlVertex::world(c, normal, color));
}

#[cfg(not(target_arch = "wasm32"))]
fn push_line(lines: &mut Vec<GlVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    let normal = [0.0, 0.0, 1.0];
    lines.push(GlVertex::world(a, normal, color));
    lines.push(GlVertex::world(b, normal, color));
}

#[cfg(not(target_arch = "wasm32"))]
fn geom_color(geom: &mjvGeom) -> [f32; 4] {
    [
        geom.rgba[0].clamp(0.0, 1.0),
        geom.rgba[1].clamp(0.0, 1.0),
        geom.rgba[2].clamp(0.0, 1.0),
        1.0,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn display_geom_color(geom: &mjvGeom, diagnostic_colors: bool) -> [f32; 4] {
    if diagnostic_colors {
        diagnostic_geom_color(geom)
    } else {
        geom_color(geom)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn diagnostic_geom_color(geom: &mjvGeom) -> [f32; 4] {
    let seed = (geom.objid as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((geom.dataid as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add((geom.type_ as u32).wrapping_mul(0xC2B2_AE35));
    let hue = (seed % 360) as f32;
    let sat = 0.72;
    let val = 0.92;
    let chroma = val * sat;
    let h = hue / 60.0;
    let x = chroma * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = val - chroma;
    [r1 + m, g1 + m, b1 + m, 1.0]
}

#[cfg(not(target_arch = "wasm32"))]
fn mesh_vertex(vertices: &[f32], index: usize) -> [f32; 3] {
    [
        vertices[index * 3],
        vertices[index * 3 + 1],
        vertices[index * 3 + 2],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn transform_geom_point(geom: &mjvGeom, local: [f32; 3]) -> [f32; 3] {
    [
        geom.pos[0] + geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.pos[1] + geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.pos[2] + geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn transform_geom_vector(geom: &mjvGeom, local: [f32; 3]) -> [f32; 3] {
    normalize3([
        geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ])
}

#[cfg(not(target_arch = "wasm32"))]
fn slice_as_u8<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(not(target_arch = "wasm32"))]
const WGPU_SCENE_SHADER: &str = r#"
struct SceneUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_scene: SceneUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct MeshVertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = u_scene.view_proj * vec4<f32>(input.position, 1.0);
    out.normal = input.normal;
    out.color = input.color;
    return out;
}

@vertex
fn vs_mesh(input: MeshVertexIn) -> VertexOut {
    let model = mat4x4<f32>(
        input.model_0,
        input.model_1,
        input.model_2,
        input.model_3,
    );
    let world_pos = model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    var out: VertexOut;
    out.position = u_scene.view_proj * world_pos;
    out.normal = world_normal;
    out.color = input.color;
    return out;
}

@fragment
fn fs_lit(input: VertexOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.35, 0.55, 0.75));
    let normal = normalize(input.normal);
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.75 + 0.25;
    return vec4<f32>(input.color.rgb * diffuse, input.color.a);
}

@fragment
fn fs_unlit(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

#[cfg(not(target_arch = "wasm32"))]
const WGPU_PRESENT_SHADER: &str = r#"
@group(0) @binding(0)
var u_tex: texture_2d<f32>;
@group(0) @binding(1)
var u_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(u_tex, u_sampler, input.uv);
}
"#;

#[cfg(not(target_arch = "wasm32"))]
fn geom_axes(geom: &mjvGeom) -> [[f32; 3]; 3] {
    [
        [geom.mat[0], geom.mat[3], geom.mat[6]],
        [geom.mat[1], geom.mat[4], geom.mat[7]],
        [geom.mat[2], geom.mat[5], geom.mat[8]],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(not(target_arch = "wasm32"))]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(not(target_arch = "wasm32"))]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(not(target_arch = "wasm32"))]
fn scale3(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

#[cfg(not(target_arch = "wasm32"))]
fn extend_bounds3(min: &mut [f32; 3], max: &mut [f32; 3], point: [f32; 3]) {
    for axis in 0..3 {
        min[axis] = min[axis].min(point[axis]);
        max[axis] = max[axis].max(point[axis]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn convex_hull(mut points: Vec<Pos2>) -> Vec<Pos2> {
    if points.len() <= 3 {
        return points;
    }

    points.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal))
    });

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && cross2(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross2(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(not(target_arch = "wasm32"))]
fn cross2(a: Pos2, b: Pos2, c: Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_bounds(geoms: &[mjvGeom]) -> [[f32; 3]; 2] {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for geom in geoms {
        for axis in 0..3 {
            let radius = geom.size[axis.min(2)];
            min[axis] = min[axis].min(geom.pos[axis] - radius);
            max[axis] = max[axis].max(geom.pos[axis] + radius);
        }
    }

    if geoms.is_empty() {
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    } else {
        [min, max]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn rgba_to_color32(rgba: [f32; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
        (rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
        (rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
        (rgba[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn shade_color_by_depth(color: Color32, depth: f32) -> Color32 {
    let attenuation = (1.25 / (0.35 + depth * 0.12)).clamp(0.45, 1.0);
    color.gamma_multiply(attenuation)
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn name2id(model: *const mjModel, ty: i32, name: &str) -> Result<i32, String> {
    let c_name =
        CString::new(name).map_err(|err| format!("Invalid MuJoCo name {name:?}: {err}"))?;
    let id = mj_name2id(model, ty, c_name.as_ptr());
    if id < 0 {
        Err(format!("MuJoCo object {:?} not found: {}", ty, name))
    } else {
        Ok(id)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn local_asset_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/mujoco/go2")
        .join(file_name)
}

#[cfg(not(target_arch = "wasm32"))]
fn local_duck_asset_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/mujoco/openduckmini")
        .join(file_name)
}

#[cfg(not(target_arch = "wasm32"))]
fn cstring_buffer_to_string(buffer: &[i8]) -> String {
    let bytes = buffer
        .iter()
        .copied()
        .take_while(|&c| c != 0)
        .map(|c| c as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(not(target_arch = "wasm32"))]
fn quaternion_yaw(q: [f32; 4]) -> f32 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let sinz_cosp = 2.0 * (w * z - x * y);
    let cosz_cosp = w * w + x * x - y * y - z * z;
    sinz_cosp.atan2(cosz_cosp)
}
