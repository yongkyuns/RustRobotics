use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[cfg(feature = "web_wgpu_viewport")]
use std::sync::{Arc, Mutex};

use eframe::emath::GuiRounding;
use egui::{vec2, Align2, Color32, FontId, Rect, Sense, Ui};
use egui_plot::{Line, PlotUi};
#[cfg(feature = "web_wgpu_viewport")]
use egui_wgpu;
#[cfg(feature = "web_wgpu_viewport")]
use eframe::wgpu;
use js_sys::{ArrayBuffer, Function, Promise, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::{spawn_local, JsFuture};
#[cfg(feature = "web_wgpu_viewport")]
use web_time::Instant;
#[cfg(feature = "web_wgpu_viewport")]
use wgpu::util::DeviceExt as _;
use crate::data::{IntoValues, TimeTable};
use super::shared_layout::show_stacked_layout;
#[cfg(feature = "web_wgpu_viewport")]
use super::render_scene::{
    append_grid_lines as append_shared_grid_lines, append_primitive_geom,
    geom_model_matrix as shared_geom_model_matrix,
    display_geom_color as shared_display_geom_color,
    GlVertex, SharedGeomSnapshot, GEOM_MESH,
};
use super::shared_panel::show_shared_panel;
#[cfg(feature = "web_wgpu_viewport")]
use super::viewport_interaction::gather_viewport_interaction;

const GO2_ASSET_FILES: &[&str] = &[
    "scene.xml",
    "go2.xml",
    "asset_meta.json",
    "facet.json",
    "facet.onnx",
    "assets/base_0.obj",
    "assets/base_1.obj",
    "assets/base_2.obj",
    "assets/base_3.obj",
    "assets/base_4.obj",
    "assets/hip_0.obj",
    "assets/hip_1.obj",
    "assets/thigh_0.obj",
    "assets/thigh_1.obj",
    "assets/thigh_mirror_0.obj",
    "assets/thigh_mirror_1.obj",
    "assets/calf_0.obj",
    "assets/calf_1.obj",
    "assets/calf_mirror_0.obj",
    "assets/calf_mirror_1.obj",
    "assets/foot.obj",
];
const OPEN_DUCK_MINI_ASSET_FILES: &[&str] = &[
    "scene.xml",
    "open_duck_mini_v2.xml",
    "asset_meta.json",
    "duck.json",
    "BEST_WALK_ONNX.onnx",
    "assets/antenna.stl",
    "assets/battery_pack_lid.stl",
    "assets/bms.stl",
    "assets/bno055.stl",
    "assets/board.stl",
    "assets/body_back.stl",
    "assets/body_front.stl",
    "assets/body_middle_bottom.stl",
    "assets/body_middle_top.stl",
    "assets/cell.stl",
    "assets/drive_palonier.stl",
    "assets/foot_bottom_pla.stl",
    "assets/foot_bottom_tpu.stl",
    "assets/foot_side.stl",
    "assets/foot_top.stl",
    "assets/head.stl",
    "assets/head_bot_sheet.stl",
    "assets/head_pitch_to_yaw.stl",
    "assets/head_roll_mount.stl",
    "assets/head_yaw_to_roll.stl",
    "assets/holder.stl",
    "assets/left_antenna_holder.stl",
    "assets/left_cache.stl",
    "assets/left_knee_to_ankle_left_sheet.stl",
    "assets/left_knee_to_ankle_right_sheet.stl",
    "assets/left_roll_to_pitch.stl",
    "assets/leg_spacer.stl",
    "assets/neck_left_sheet.stl",
    "assets/neck_right_sheet.stl",
    "assets/passive_palonier.stl",
    "assets/power_switch.stl",
    "assets/raspberrypizerow.stl",
    "assets/right_antenna_holder.stl",
    "assets/right_cache.stl",
    "assets/right_roll_to_pitch.stl",
    "assets/roll_bearing.stl",
    "assets/roll_motor_bottom.stl",
    "assets/roll_motor_top.stl",
    "assets/sg90.stl",
    "assets/trunk_bottom.stl",
    "assets/trunk_top.stl",
    "assets/usb_c_charger.stl",
    "assets/wj-wk00-0122topcabinetcase_95.stl",
    "assets/wj-wk00-0123middlecase_56.stl",
    "assets/wj-wk00-0124bottomcase_45.stl",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowserRobotPreset {
    Go2,
    OpenDuckMini,
}

impl BrowserRobotPreset {
    fn asset_base_path(self) -> &'static str {
        match self {
            Self::Go2 => "assets/mujoco/go2/",
            Self::OpenDuckMini => "assets/mujoco/openduckmini/",
        }
    }

    fn asset_files(self) -> &'static [&'static str] {
        match self {
            Self::Go2 => GO2_ASSET_FILES,
            Self::OpenDuckMini => OPEN_DUCK_MINI_ASSET_FILES,
        }
    }

    fn policy_config_name(self) -> &'static str {
        match self {
            Self::Go2 => "facet.json",
            Self::OpenDuckMini => "duck.json",
        }
    }

    fn robot_xml_name(self) -> &'static str {
        match self {
            Self::Go2 => "go2.xml",
            Self::OpenDuckMini => "open_duck_mini_v2.xml",
        }
    }
}

pub(super) struct WasmMujocoBackend {
    active: bool,
    selected_robot: BrowserRobotPreset,
    generation: Rc<RefCell<u64>>,
    asset_state: Rc<RefCell<BrowserAssetState>>,
    ort_state: Rc<RefCell<BrowserOrtState>>,
    mujoco_state: Rc<RefCell<BrowserMujocoState>>,
    mujoco_step_in_flight: Rc<RefCell<bool>>,
    pending_mujoco_steps: Rc<RefCell<usize>>,
    compiled_mesh_assets: Rc<RefCell<Vec<BrowserMeshAsset>>>,
    #[cfg(feature = "web_wgpu_viewport")]
    wgpu_renderer: Arc<Mutex<BrowserWgpuRenderer>>,
    #[cfg(feature = "web_wgpu_viewport")]
    wgpu_scene: Arc<Mutex<BrowserWgpuSceneFrame>>,
    camera: BrowserOrbitCamera,
    camera_initialized: bool,
    plot_history: Rc<RefCell<TimeTable>>,
    overlay_occlusions: Vec<Rect>,
    overlay_interactive: bool,
    #[cfg(feature = "web_wgpu_viewport")]
    render_display_fps: f32,
    #[cfg(feature = "web_wgpu_viewport")]
    last_render_frame_at: Option<Instant>,
}

impl Default for WasmMujocoBackend {
    fn default() -> Self {
        Self {
            active: false,
            selected_robot: BrowserRobotPreset::Go2,
            generation: Rc::new(RefCell::new(0)),
            asset_state: Rc::new(RefCell::new(BrowserAssetState::Idle)),
            ort_state: Rc::new(RefCell::new(BrowserOrtState::Idle)),
            mujoco_state: Rc::new(RefCell::new(BrowserMujocoState::Idle)),
            mujoco_step_in_flight: Rc::new(RefCell::new(false)),
            pending_mujoco_steps: Rc::new(RefCell::new(0)),
            compiled_mesh_assets: Rc::new(RefCell::new(Vec::new())),
            #[cfg(feature = "web_wgpu_viewport")]
            wgpu_renderer: Arc::new(Mutex::new(BrowserWgpuRenderer::default())),
            #[cfg(feature = "web_wgpu_viewport")]
            wgpu_scene: Arc::new(Mutex::new(BrowserWgpuSceneFrame::default())),
            camera: BrowserOrbitCamera::default(),
            camera_initialized: false,
            plot_history: Rc::new(RefCell::new(TimeTable::init_with_names(vec![
                "Base Height",
                "Command Input X",
                "Action[0]",
            ]))),
            overlay_occlusions: Vec::new(),
            overlay_interactive: true,
            #[cfg(feature = "web_wgpu_viewport")]
            render_display_fps: 0.0,
            #[cfg(feature = "web_wgpu_viewport")]
            last_render_frame_at: None,
        }
    }
}

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

#[derive(Debug, Deserialize, Default)]
struct PolicyObsConfig {
    #[serde(default)]
    command: Vec<PolicyCommandConfig>,
    #[serde(default, rename = "command_")]
    command_alt: Vec<PolicyCommandConfig>,
}

#[derive(Debug, Deserialize)]
struct PolicyCommandConfig {
    name: String,
}

impl PolicyFile {
    fn command_mode(&self) -> &str {
        if let Some(command_mode) = self.command_mode.as_deref() {
            return command_mode;
        }
        let commands = if !self.obs_config.command.is_empty() {
            &self.obs_config.command
        } else {
            &self.obs_config.command_alt
        };
        if commands.iter().any(|cfg| cfg.name == "ImpedanceCommand") {
            "impedance"
        } else {
            "velocity"
        }
    }

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
}

#[derive(Debug, Deserialize)]
struct OnnxFile {
    path: String,
    meta: OnnxMeta,
}

#[derive(Debug, Deserialize)]
struct OnnxMeta {
    in_keys: Vec<String>,
    in_shapes: Vec<Vec<Vec<usize>>>,
    out_keys: Vec<OnnxKey>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OnnxKey {
    String(String),
    Path(Vec<String>),
}

impl OnnxKey {
    fn joined(&self) -> String {
        match self {
            OnnxKey::String(value) => value.clone(),
            OnnxKey::Path(parts) => parts.join("."),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AssetMeta {
    joint_names_isaac: Vec<String>,
    default_joint_pos: Vec<f32>,
}

struct BrowserAssetBundle {
    preset: BrowserRobotPreset,
    files: BTreeMap<String, Vec<u8>>,
    policy: PolicyFile,
    asset_meta: AssetMeta,
}

#[derive(Default)]
enum BrowserAssetState {
    #[default]
    Idle,
    Loading,
    Ready(Rc<BrowserAssetBundle>),
    Error(String),
}

enum BrowserAssetUiState {
    Idle,
    Loading,
    Ready(Rc<BrowserAssetBundle>),
    Error(String),
}

enum BrowserPanelStatus {
    Info(String),
    Error(String),
}

#[derive(Default)]
enum BrowserOrtState {
    #[default]
    Idle,
    Loading,
    Ready,
    Error(String),
}

#[derive(Default)]
enum BrowserMujocoState {
    #[default]
    Idle,
    Loading,
    Ready(BrowserMujocoReport),
    Error(String),
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
struct BrowserMujocoReport {
    nbody: usize,
    ngeom: usize,
    nv: usize,
    nu: usize,
    timestep: f64,
    sim_time: f64,
    step_count: usize,
    qpos_preview: Vec<f32>,
    xpos_preview: Vec<f32>,
    last_action_preview: Vec<f32>,
    policy_inputs: Vec<String>,
    policy_outputs: Vec<String>,
    command_vel_x: f32,
    #[serde(default)]
    command_vel_y: f32,
    #[serde(default)]
    use_setpoint_ball: bool,
    #[serde(default)]
    command_mode: String,
    #[serde(default)]
    setpoint_preview: Vec<f32>,
    #[serde(default)]
    debug_drag_mode: String,
    #[serde(default)]
    debug_pointer_downs: u32,
    #[serde(default)]
    debug_pointer_moves: u32,
    #[serde(default)]
    display_fps: f32,
    #[serde(default)]
    last_step_wall_ms: f32,
    #[serde(default)]
    avg_step_wall_ms: f32,
    #[serde(default)]
    last_policy_wall_ms: f32,
    #[serde(default)]
    avg_policy_wall_ms: f32,
    #[serde(default)]
    last_physics_wall_ms: f32,
    #[serde(default)]
    avg_physics_wall_ms: f32,
    #[serde(default)]
    last_overlay_wall_ms: f32,
    #[serde(default)]
    avg_overlay_wall_ms: f32,
    #[serde(default)]
    geoms: Vec<BrowserGeomSnapshot>,
    #[serde(default)]
    mesh_assets: Vec<BrowserMeshAssetSnapshot>,
}

#[cfg_attr(not(feature = "web_wgpu_viewport"), allow(dead_code))]
#[derive(Debug, Deserialize, Clone)]
struct BrowserGeomSnapshot {
    type_id: i32,
    dataid: i32,
    size: [f32; 3],
    rgba: [f32; 4],
    pos: [f32; 3],
    mat: [f32; 9],
}

#[derive(Serialize)]
struct BrowserMujocoConfig {
    controller_kind: String,
    joint_names: Vec<String>,
    default_joint_pos: Vec<f32>,
    action_scale: f32,
    stiffness: f32,
    damping: f32,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
    command_dim: usize,
    command_mode: String,
    phase_steps: usize,
}

#[derive(Clone)]
struct BrowserMeshAsset {
    vertices: Vec<LocalMeshVertex>,
    local_min: [f32; 3],
    local_max: [f32; 3],
}

#[derive(Debug, Deserialize, Clone)]
struct BrowserMeshAssetSnapshot {
    positions: Vec<f32>,
    normals: Vec<f32>,
    faces: Vec<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct LocalMeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[cfg(feature = "web_wgpu_viewport")]
#[derive(Default, Clone)]
struct BrowserWgpuSceneFrame {
    triangles: Vec<GlVertex>,
    lines: Vec<GlVertex>,
    mesh_instances: Vec<MeshInstance>,
    view_proj: [f32; 16],
}

#[cfg(feature = "web_wgpu_viewport")]
struct BrowserWgpuCallback {
    rect: Rect,
    renderer: Arc<Mutex<BrowserWgpuRenderer>>,
    scene: Arc<Mutex<BrowserWgpuSceneFrame>>,
    target_format: wgpu::TextureFormat,
}

#[cfg(feature = "web_wgpu_viewport")]
impl egui_wgpu::CallbackTrait for BrowserWgpuCallback {
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
                .map_err(|_| "Failed to lock wasm wgpu scene frame".to_string())?;
            let mut renderer = self
                .renderer
                .lock()
                .map_err(|_| "Failed to lock wasm wgpu renderer".to_string())?;
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
                .map_err(|_| "Failed to lock wasm wgpu renderer".to_string())?;
            renderer.paint(render_pass)
        })();

        if let Err(err) = render_result {
            if let Ok(mut renderer) = self.renderer.lock() {
                renderer.last_error = Some(err);
            }
        }
    }
}

#[cfg(feature = "web_wgpu_viewport")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BrowserWgpuSceneUniform {
    view_proj: [f32; 16],
}

#[cfg(feature = "web_wgpu_viewport")]
#[derive(Default)]
struct BrowserWgpuRenderer {
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

#[cfg(feature = "web_wgpu_viewport")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MeshInstance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    mesh_id: u32,
    _padding: [u32; 3],
}

#[cfg(feature = "web_wgpu_viewport")]
struct MeshAssetGpu {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

#[derive(Clone, Copy)]
struct BrowserOrbitCamera {
    azimuth_deg: f32,
    elevation_deg: f32,
    distance: f32,
    target: [f32; 3],
}

impl Default for BrowserOrbitCamera {
    fn default() -> Self {
        Self {
            azimuth_deg: 135.0,
            elevation_deg: -20.0,
            distance: 2.2,
            target: [0.0, 0.0, 0.35],
        }
    }
}

impl WasmMujocoBackend {
    pub fn update(&mut self, sim_speed: usize, paused: bool) {
        if !self.active {
            return;
        }

        self.ensure_browser_assets_started();
        let Some(bundle) = self.browser_assets_ready() else {
            return;
        };

        self.ensure_ort_smoke_test_started(bundle.as_ref());
        self.ensure_mujoco_runtime_started(bundle.as_ref());

        if paused {
            return;
        }

        let queued = self.pending_mujoco_steps.borrow().saturating_add(sim_speed.max(1));
        *self.pending_mujoco_steps.borrow_mut() = queued.min(256);
        self.ensure_mujoco_step_started();
    }

    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        show_stacked_layout(
            ui,
            self,
            |this, ui| {
                let previous_robot = this.selected_robot;
                let selected_index = match this.selected_robot {
                    BrowserRobotPreset::Go2 => 0,
                    BrowserRobotPreset::OpenDuckMini => 1,
                };
                let description = match this.selected_robot {
                    BrowserRobotPreset::Go2 => {
                        "Go2 runs the facet policy and uses the draggable setpoint ball as the command input."
                    }
                    BrowserRobotPreset::OpenDuckMini => {
                        "Open Duck Mini runs the BEST_WALK_ONNX policy and uses the draggable setpoint ball to generate walking commands."
                    }
                };
                let outcome = show_shared_panel(
                    ui,
                    selected_index,
                    &["Go2", "Open Duck Mini"],
                    description,
                    |ui| {
                        ui.label(match this.selected_robot {
                            BrowserRobotPreset::Go2 => "Policy: facet",
                            BrowserRobotPreset::OpenDuckMini => "Policy: BEST_WALK_ONNX",
                        });

                        if this.active {
                            this.ensure_browser_assets_started();
                        }

                        let asset_state_kind = {
                            let state = this.asset_state.borrow();
                            match &*state {
                                BrowserAssetState::Idle => BrowserAssetUiState::Idle,
                                BrowserAssetState::Loading => BrowserAssetUiState::Loading,
                                BrowserAssetState::Ready(bundle) => {
                                    BrowserAssetUiState::Ready(Rc::clone(bundle))
                                }
                                BrowserAssetState::Error(err) => {
                                    BrowserAssetUiState::Error(err.clone())
                                }
                            }
                        };

                        match asset_state_kind {
                            BrowserAssetUiState::Idle => {
                                if this.active {
                                    ui.label("Loading robot runtime...");
                                }
                            }
                            BrowserAssetUiState::Loading => {
                                ui.label("Loading robot runtime...");
                            }
                            BrowserAssetUiState::Ready(bundle) => {
                                if this.active {
                                    this.ensure_ort_smoke_test_started(bundle.as_ref());
                                    this.ensure_mujoco_runtime_started(bundle.as_ref());
                                }
                                if let Some(message) = this.browser_panel_status_message() {
                                    match message {
                                        BrowserPanelStatus::Info(text) => {
                                            ui.label(text);
                                        }
                                        BrowserPanelStatus::Error(text) => {
                                            ui.colored_label(Color32::LIGHT_RED, text);
                                        }
                                    }
                                }
                            }
                            BrowserAssetUiState::Error(err) => {
                                ui.colored_label(
                                    Color32::LIGHT_RED,
                                    format!("Failed to load robot assets: {err}"),
                                );
                            }
                        }
                    },
                );
                this.selected_robot = match outcome.selected_index {
                    1 => BrowserRobotPreset::OpenDuckMini,
                    _ => BrowserRobotPreset::Go2,
                };
                if this.selected_robot != previous_robot {
                    this.reset_selected_robot_state();
                }
                if outcome.reset_view {
                    this.camera = BrowserOrbitCamera::default();
                    this.camera_initialized = false;
                }
            },
            |this, ui| this.ui_viewport(ui, frame),
        );
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn set_overlay_occlusions(&mut self, rects: &[Rect], interactive: bool) {
        self.overlay_occlusions.clear();
        self.overlay_occlusions.extend_from_slice(rects);
        self.overlay_interactive = interactive;
    }

    pub fn reset_state(&mut self) {
        self.plot_history.borrow_mut().clear();
        self.reset_loaded_policy_state();
    }

    pub fn reset_all(&mut self) {
        let active = self.active;
        *self = Self::default();
        self.active = active;
    }

    pub fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        let history = self.plot_history.borrow();
        for (index, name) in history.names().iter().enumerate() {
            if let Some(values) = history.values(index) {
                plot_ui.line(Line::new(name.clone(), values));
            }
        }
    }

    fn ensure_browser_assets_started(&mut self) {
        if !matches!(*self.asset_state.borrow(), BrowserAssetState::Idle) {
            return;
        }

        *self.asset_state.borrow_mut() = BrowserAssetState::Loading;
        let state = Rc::clone(&self.asset_state);
        let generation = Rc::clone(&self.generation);
        let expected_generation = *generation.borrow();
        let preset = self.selected_robot;
        spawn_local(async move {
            let result = BrowserAssetBundle::load_from_server(preset)
                .await
                .map(Rc::new);
            if *generation.borrow() != expected_generation {
                return;
            }
            *state.borrow_mut() = match result {
                Ok(bundle) => BrowserAssetState::Ready(bundle),
                Err(err) => BrowserAssetState::Error(err),
            };
        });
    }

    fn browser_assets_ready(&self) -> Option<Rc<BrowserAssetBundle>> {
        match &*self.asset_state.borrow() {
            BrowserAssetState::Ready(bundle) => Some(Rc::clone(bundle)),
            _ => None,
        }
    }

    fn reset_loaded_policy_state(&mut self) {
        *self.ort_state.borrow_mut() = BrowserOrtState::Idle;
        *self.mujoco_state.borrow_mut() = BrowserMujocoState::Idle;
        *self.mujoco_step_in_flight.borrow_mut() = false;
        *self.pending_mujoco_steps.borrow_mut() = 0;
        self.compiled_mesh_assets.borrow_mut().clear();
        self.plot_history.borrow_mut().clear();
        self.reset_browser_runtime();
    }

    fn reset_selected_robot_state(&mut self) {
        *self.generation.borrow_mut() += 1;
        *self.asset_state.borrow_mut() = BrowserAssetState::Idle;
        self.reset_loaded_policy_state();
        self.camera = BrowserOrbitCamera::default();
        self.camera_initialized = false;
    }

    fn browser_panel_status_message(&self) -> Option<BrowserPanelStatus> {
        if let BrowserAssetState::Error(err) = &*self.asset_state.borrow() {
            return Some(BrowserPanelStatus::Error(format!(
                "Failed to load robot assets: {err}"
            )));
        }

        match &*self.ort_state.borrow() {
            BrowserOrtState::Loading => {
                return Some(BrowserPanelStatus::Info(
                    "Loading ONNX policy runtime...".to_string(),
                ));
            }
            BrowserOrtState::Error(err) => {
                return Some(BrowserPanelStatus::Error(err.clone()));
            }
            BrowserOrtState::Idle | BrowserOrtState::Ready => {}
        }

        match &*self.mujoco_state.borrow() {
            BrowserMujocoState::Idle | BrowserMujocoState::Loading => Some(
                BrowserPanelStatus::Info("Loading robot runtime...".to_string()),
            ),
            BrowserMujocoState::Ready(_) => None,
            BrowserMujocoState::Error(err) => Some(BrowserPanelStatus::Error(err.clone())),
        }
    }

    fn ensure_ort_smoke_test_started(&mut self, bundle: &BrowserAssetBundle) {
        if !matches!(*self.ort_state.borrow(), BrowserOrtState::Idle) {
            return;
        }

        *self.ort_state.borrow_mut() = BrowserOrtState::Loading;
        let state = Rc::clone(&self.ort_state);
        let generation = Rc::clone(&self.generation);
        let expected_generation = *generation.borrow();
        let policy = bundle.policy();
        let onnx_path = normalize_rel_path(&policy.onnx.path);
        let Some(model_bytes) = bundle.files.get(onnx_path.as_str()).cloned() else {
            *self.ort_state.borrow_mut() =
                BrowserOrtState::Error(format!("Browser asset bundle missing {onnx_path}"));
            return;
        };
        let wasm_base_path = "./vendor/onnxruntime-web/dist/";
        let config = BrowserMujocoConfig {
            controller_kind: policy.controller_kind().to_string(),
            joint_names: bundle.asset_meta.joint_names_isaac.clone(),
            default_joint_pos: bundle.asset_meta.default_joint_pos.clone(),
            action_scale: policy.action_scale,
            stiffness: policy.stiffness,
            damping: policy.damping,
            input_keys: policy.onnx.meta.in_keys.clone(),
            output_keys: policy
                .onnx
                .meta
                .out_keys
                .iter()
                .map(OnnxKey::joined)
                .collect(),
            command_dim: policy
                .onnx
                .meta
                .in_shapes
                .first()
                .and_then(|group| group.first())
                .and_then(|shape| shape.get(1))
                .copied()
                .unwrap_or(16),
            command_mode: policy.command_mode().to_string(),
            phase_steps: policy.phase_steps(),
        };

        spawn_local(async move {
            let result = async {
                let js_value = ort_smoke_test(&model_bytes, wasm_base_path, &config).await?;
                let _: serde_json::Value = serde_wasm_bindgen::from_value(js_value)
                    .map_err(|err| JsValue::from_str(&err.to_string()))?;
                Ok(())
            }
            .await;

            if *generation.borrow() != expected_generation {
                return;
            }
            *state.borrow_mut() = match result {
                Ok(()) => BrowserOrtState::Ready,
                Err(err) => BrowserOrtState::Error(js_error_to_string(err)),
            };
        });
    }

    fn ensure_mujoco_runtime_started(&mut self, bundle: &BrowserAssetBundle) {
        if !matches!(*self.mujoco_state.borrow(), BrowserMujocoState::Idle) {
            return;
        }

        *self.mujoco_state.borrow_mut() = BrowserMujocoState::Loading;
        let state = Rc::clone(&self.mujoco_state);
        let compiled_mesh_assets = Rc::clone(&self.compiled_mesh_assets);
        let generation = Rc::clone(&self.generation);
        let expected_generation = *generation.borrow();
        let policy_config_name = bundle.preset.policy_config_name();
        let policy_onnx_path = normalize_rel_path(&bundle.policy().onnx.path);
        let file_entries = bundle
            .files
            .iter()
            .filter_map(|(path, bytes)| {
                if path == "asset_meta.json"
                    || path == policy_config_name
                    || path == &policy_onnx_path
                {
                    None
                } else {
                    Some((path.clone(), bytes.clone()))
                }
            })
            .collect::<Vec<_>>();
        let policy = bundle.policy();
        let onnx_path = normalize_rel_path(&policy.onnx.path);
        let Some(policy_bytes) = bundle.files.get(onnx_path.as_str()).cloned() else {
            *self.mujoco_state.borrow_mut() =
                BrowserMujocoState::Error(format!("Browser asset bundle missing {onnx_path}"));
            return;
        };
        let mujoco_wasm_base_path = "./vendor/mujoco/";
        let ort_wasm_base_path = "./vendor/onnxruntime-web/dist/";
        let config = BrowserMujocoConfig {
            controller_kind: policy.controller_kind().to_string(),
            joint_names: bundle.asset_meta.joint_names_isaac.clone(),
            default_joint_pos: bundle.asset_meta.default_joint_pos.clone(),
            action_scale: policy.action_scale,
            stiffness: policy.stiffness,
            damping: policy.damping,
            input_keys: policy.onnx.meta.in_keys.clone(),
            output_keys: policy
                .onnx
                .meta
                .out_keys
                .iter()
                .map(OnnxKey::joined)
                .collect(),
            command_dim: policy
                .onnx
                .meta
                .in_shapes
                .first()
                .and_then(|group| group.first())
                .and_then(|shape| shape.get(1))
                .copied()
                .unwrap_or(16),
            command_mode: policy.command_mode().to_string(),
            phase_steps: policy.phase_steps(),
        };

        spawn_local(async move {
            let result = async {
                let js_value = mujoco_init(
                    &file_entries,
                    &policy_bytes,
                    mujoco_wasm_base_path,
                    ort_wasm_base_path,
                    &config,
                )
                .await?;
                serde_wasm_bindgen::from_value::<BrowserMujocoReport>(js_value)
                    .map_err(|err| JsValue::from_str(&err.to_string()))
            }
            .await;

            if *generation.borrow() != expected_generation {
                return;
            }
            if let Ok(report) = &result {
                *compiled_mesh_assets.borrow_mut() =
                    convert_browser_mesh_assets(&report.mesh_assets);
            }
            *state.borrow_mut() = match result {
                Ok(report) => BrowserMujocoState::Ready(report),
                Err(err) => BrowserMujocoState::Error(js_error_to_string(err)),
            };
        });
    }

    fn ensure_mujoco_step_started(&mut self) {
        if *self.mujoco_step_in_flight.borrow() {
            return;
        }
        if !matches!(*self.mujoco_state.borrow(), BrowserMujocoState::Ready(_)) {
            return;
        }
        let step_count = {
            let pending = *self.pending_mujoco_steps.borrow();
            pending.max(1)
        };
        *self.pending_mujoco_steps.borrow_mut() = 0;

        *self.mujoco_step_in_flight.borrow_mut() = true;
        let state = Rc::clone(&self.mujoco_state);
        let step_in_flight = Rc::clone(&self.mujoco_step_in_flight);
        let generation = Rc::clone(&self.generation);
        let plot_history = Rc::clone(&self.plot_history);
        let expected_generation = *generation.borrow();
        let command_vel_x = 0.0f32;
        let command_vel_y = 0.0f32;
        let use_setpoint_ball = true;

        spawn_local(async move {
            let result = async {
                let js_value =
                    mujoco_step(step_count, command_vel_x, command_vel_y, use_setpoint_ball)
                        .await?;
                serde_wasm_bindgen::from_value::<BrowserMujocoReport>(js_value)
                    .map_err(|err| JsValue::from_str(&err.to_string()))
            }
            .await;

            if *generation.borrow() != expected_generation {
                *step_in_flight.borrow_mut() = false;
                return;
            }
            *step_in_flight.borrow_mut() = false;
            let latest_setpoint_preview = match &*state.borrow() {
                BrowserMujocoState::Ready(current) if current.use_setpoint_ball => {
                    Some(current.setpoint_preview.clone())
                }
                _ => None,
            };
            *state.borrow_mut() = match result {
                Ok(mut report) => {
                    if let Some(setpoint_preview) = latest_setpoint_preview {
                        if setpoint_preview.len() >= 3 {
                            report.setpoint_preview = setpoint_preview;
                        }
                    }
                    let command_input_x = report
                        .setpoint_preview
                        .first()
                        .copied()
                        .unwrap_or(report.command_vel_x);
                    plot_history.borrow_mut().add(
                        report.sim_time as f32,
                        vec![
                            report.qpos_preview.get(2).copied().unwrap_or(0.0),
                            command_input_x,
                            report.last_action_preview.first().copied().unwrap_or(0.0),
                        ],
                    );
                    BrowserMujocoState::Ready(report)
                }
                Err(err) => BrowserMujocoState::Error(js_error_to_string(err)),
            };
        });
    }

    fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        let rect = aligned_viewport_rect(ui);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let state_binding = self.mujoco_state.borrow();
        let mut report = match &*state_binding {
            BrowserMujocoState::Ready(report) => report.clone(),
            BrowserMujocoState::Idle => {
                ui.painter()
                    .rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 24));
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "MuJoCo browser runtime idle",
                    FontId::proportional(18.0),
                    Color32::from_gray(200),
                );
                return;
            }
            BrowserMujocoState::Loading => {
                ui.painter()
                    .rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 24));
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Loading MuJoCo browser runtime...",
                    FontId::proportional(18.0),
                    Color32::from_gray(200),
                );
                return;
            }
            BrowserMujocoState::Error(err) => {
                ui.painter()
                    .rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 24));
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    err,
                    FontId::proportional(16.0),
                    Color32::LIGHT_RED,
                );
                return;
            }
        };
        drop(state_binding);
        self.update_camera(ui, &response, &report);
        if let BrowserMujocoState::Ready(updated_report) = &*self.mujoco_state.borrow() {
            report = updated_report.clone();
        }
        let mesh_assets_empty = self.compiled_mesh_assets.borrow().is_empty();
        if mesh_assets_empty {
            ui.painter()
                .rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 24));
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Waiting for compiled MuJoCo mesh assets...",
                FontId::proportional(16.0),
                Color32::from_gray(210),
            );
            return;
        }

        if !self.camera_initialized {
            self.fit_camera_to_report(&report);
            self.camera_initialized = true;
        }
        let mesh_assets = self.compiled_mesh_assets.borrow();
        let scene = self.build_wgpu_scene_frame(
            &report,
            mesh_assets.as_slice(),
            rect.width() / rect.height(),
        );
        if let Ok(mut scene_state) = self.wgpu_scene.lock() {
            *scene_state = scene;
        }

        let Some(frame) = frame else {
            self.render_software_2d(ui, rect, &report, mesh_assets.as_slice());
            return;
        };
        let Some(render_state) = frame.wgpu_render_state() else {
            self.render_software_2d(ui, rect, &report, mesh_assets.as_slice());
            return;
        };
        if let Ok(mut renderer) = self.wgpu_renderer.lock() {
            let _ = renderer.ensure_mesh_assets(&render_state.device, mesh_assets.as_slice());
        }

        let callback = BrowserWgpuCallback {
            rect,
            renderer: self.wgpu_renderer.clone(),
            scene: self.wgpu_scene.clone(),
            target_format: render_state.target_format,
        };
        ui.painter()
            .add(egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
                rect, callback,
            )));

        let now = Instant::now();
        if let Some(last_frame_at) = self.last_render_frame_at {
            let dt_ms = (now - last_frame_at).as_secs_f32() * 1000.0;
            if dt_ms > f32::EPSILON {
                let instant_fps = 1000.0 / dt_ms.max(1.0);
                self.render_display_fps = if self.render_display_fps > 0.0 {
                    self.render_display_fps * 0.9 + instant_fps * 0.1
                } else {
                    instant_fps
                };
            }
        }
        self.last_render_frame_at = Some(now);

        let overlay = format!(
            concat!(
                "Browser rust wgpu viewport\n",
                "Sim t={:.2}s  steps={}  cmd=({:+.2}, {:+.2})\n",
                "Setpoint {:?}  policy={}  Viewport FPS {:.1}\n",
                "Step {:.2}/{:.2} ms  Policy {:.2}/{:.2} ms  Physics {:.2}/{:.2} ms\n",
                "Overlay {:.2}/{:.2} ms  Drag {} d={} m={}\n",
                "Drag: orbit | Right-drag: pan | Wheel: zoom | Double-click: reset view"
            ),
            report.sim_time,
            report.step_count,
            report.command_vel_x,
            report.command_vel_y,
            report.setpoint_preview,
            report.command_mode,
            self.render_display_fps,
            report.last_step_wall_ms,
            report.avg_step_wall_ms,
            report.last_policy_wall_ms,
            report.avg_policy_wall_ms,
            report.last_physics_wall_ms,
            report.avg_physics_wall_ms,
            report.last_overlay_wall_ms,
            report.avg_overlay_wall_ms,
            report.debug_drag_mode,
            report.debug_pointer_downs,
            report.debug_pointer_moves
        );
        ui.painter().text(
            rect.left_top() + vec2(12.0, 12.0),
            Align2::LEFT_TOP,
            overlay,
            FontId::monospace(12.0),
            Color32::from_gray(210),
        );
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn fit_camera_to_report(&mut self, report: &BrowserMujocoReport) {
        if report.geoms.is_empty() {
            return;
        }

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for geom in &report.geoms {
            let radius = geom.size.iter().copied().fold(0.05f32, f32::max);
            for axis in 0..3 {
                min[axis] = min[axis].min(geom.pos[axis] - radius);
                max[axis] = max[axis].max(geom.pos[axis] + radius);
            }
        }

        self.camera.target = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let extent = ((max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2])).max(0.5);
        self.camera.distance = (extent * 2.8).max(1.5);
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn build_wgpu_scene_frame(
        &self,
        report: &BrowserMujocoReport,
        mesh_assets: &[BrowserMeshAsset],
        aspect: f32,
    ) -> BrowserWgpuSceneFrame {
        let mut frame = BrowserWgpuSceneFrame::default();
        append_shared_grid_lines(&mut frame.lines);
        let mut depth_points = Vec::new();
        depth_points.extend(frame.lines.iter().map(|v| v.position));

        for geom in &report.geoms {
            let shared_geom = SharedGeomSnapshot {
                type_id: geom.type_id,
                data_id: geom.dataid,
                size: geom.size,
                rgba: geom.rgba,
                pos: geom.pos,
                mat: geom.mat,
            };
            match geom.type_id {
                GEOM_MESH => {
                    if let Some(instance) = build_mesh_instance(&shared_geom, false) {
                        if let Some(asset) = mesh_assets.get(instance.mesh_id as usize) {
                            extend_depth_points_with_mesh_bounds(
                                &mut depth_points,
                                asset,
                                &instance.model,
                            );
                        }
                        frame.mesh_instances.push(instance);
                    }
                }
                _ => append_primitive_geom(
                    &mut frame.triangles,
                    &mut frame.lines,
                    &shared_geom,
                    false,
                ),
            }
        }

        self.append_command_setpoint_scene(&mut frame.triangles, report);

        depth_points.extend(frame.triangles.iter().map(|v| v.position));
        depth_points.extend(frame.lines.iter().map(|v| v.position));
        if let Some(setpoint) = report_setpoint(report) {
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

        frame.view_proj = self
            .camera
            .fitted_view_projection_matrix_points(aspect.max(0.1), &depth_points);
        frame
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn append_command_setpoint_scene(
        &self,
        triangles: &mut Vec<GlVertex>,
        report: &BrowserMujocoReport,
    ) {
        let Some(setpoint) = report_setpoint(report) else {
            return;
        };
        super::render_scene::append_world_sphere(
            triangles,
            setpoint,
            0.05,
            [0.92, 0.24, 0.24, 1.0],
        );
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn update_camera(&mut self, ui: &Ui, response: &egui::Response, report: &BrowserMujocoReport) {
        let interaction = gather_viewport_interaction(report.use_setpoint_ball, ui, response, |pointer| {
            self.camera.intersect_plane_z(response.rect, pointer, 0.05)
        });

        if interaction.reset_view {
            self.camera = BrowserOrbitCamera::default();
            self.camera_initialized = false;
        }
        if let Some(world) = interaction.setpoint_world {
            self.set_command_setpoint(world);
        }
        if let Some(delta) = interaction.primary_orbit_delta {
            self.camera.azimuth_deg -= delta.x * 0.25;
            self.camera.elevation_deg =
                (self.camera.elevation_deg + delta.y * 0.2).clamp(-89.0, 89.0);
        }

        if let Some(delta) = interaction.secondary_orbit_delta {
            self.camera.azimuth_deg -= delta.x * 0.25;
            self.camera.elevation_deg =
                (self.camera.elevation_deg - delta.y * 0.2).clamp(-89.0, 89.0);
        }

        if let Some(zoom) = interaction.zoom_factor {
            self.camera.distance = (self.camera.distance * zoom).clamp(0.35, 40.0);
        }
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn set_command_setpoint(&mut self, world: [f32; 3]) {
        if let BrowserMujocoState::Ready(report) = &mut *self.mujoco_state.borrow_mut() {
            report.setpoint_preview = vec![world[0], world[1], world[2]];
        }
        self.set_browser_command_setpoint(world);
    }

    #[cfg(feature = "web_wgpu_viewport")]
    fn set_browser_command_setpoint(&self, world: [f32; 3]) {
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(function) = Reflect::get(
            window.as_ref(),
            &JsValue::from_str("rustRoboticsMujocoSetCommandSetpoint"),
        ) else {
            return;
        };
        let Ok(function) = function.dyn_into::<Function>() else {
            return;
        };
        let _ = function.call3(
            &JsValue::NULL,
            &JsValue::from_f64(world[0] as f64),
            &JsValue::from_f64(world[1] as f64),
            &JsValue::from_f64(world[2] as f64),
        );
    }

    fn render_software_2d(
        &self,
        ui: &mut Ui,
        rect: Rect,
        report: &BrowserMujocoReport,
        _mesh_assets: &[BrowserMeshAsset],
    ) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, Color32::from_rgb(18, 22, 28));
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            format!(
                concat!(
                    "WGPU viewport unavailable\n",
                    "sim_time={:.2}s step_count={}\n",
                    "cmd=({:+.2}, {:+.2}) setpoint={:?}\n",
                    "step {:.2}/{:.2} ms | policy {:.2}/{:.2} ms"
                ),
                report.sim_time,
                report.step_count,
                report.command_vel_x,
                report.command_vel_y,
                report.setpoint_preview,
                report.last_step_wall_ms,
                report.avg_step_wall_ms,
                report.last_policy_wall_ms,
                report.avg_policy_wall_ms
            ),
            FontId::proportional(18.0),
            Color32::from_gray(210),
        );
    }

    fn reset_browser_runtime(&self) {
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(function) = Reflect::get(
            window.as_ref(),
            &JsValue::from_str("rustRoboticsMujocoReset"),
        ) else {
            return;
        };
        let Ok(function) = function.dyn_into::<Function>() else {
            return;
        };
        let _ = function.call0(&JsValue::NULL);
    }
}

impl BrowserAssetBundle {
    fn policy(&self) -> &PolicyFile {
        &self.policy
    }

    async fn load_from_server(preset: BrowserRobotPreset) -> Result<Self, String> {
        let mut files = BTreeMap::new();
        for path in preset.asset_files() {
            let bytes = fetch_browser_asset_bytes(preset, path).await?;
            files.insert((*path).to_string(), bytes);
        }

        let scene_xml = std::str::from_utf8(
            files
                .get("scene.xml")
                .ok_or_else(|| "Missing fetched scene.xml".to_string())?,
        )
        .map_err(|err| format!("scene.xml is not valid UTF-8: {err}"))?;
        let robot_xml_name = preset.robot_xml_name();
        let robot_xml = std::str::from_utf8(
            files
                .get(robot_xml_name)
                .ok_or_else(|| format!("Missing fetched {robot_xml_name}"))?,
        )
        .map_err(|err| format!("{robot_xml_name} is not valid UTF-8: {err}"))?;
        let asset_meta_json = std::str::from_utf8(
            files
                .get("asset_meta.json")
                .ok_or_else(|| "Missing fetched asset_meta.json".to_string())?,
        )
        .map_err(|err| format!("asset_meta.json is not valid UTF-8: {err}"))?;

        let scene_include = single_xml_attribute(scene_xml, "include", "file")
            .ok_or_else(|| "scene.xml missing <include file=\"...\">".to_string())?;
        if !files.contains_key(scene_include.as_str()) {
            return Err(format!("scene.xml includes missing file {}", scene_include));
        }

        let meshdir = single_xml_attribute(robot_xml, "compiler", "meshdir")
            .ok_or_else(|| format!("{robot_xml_name} missing <compiler meshdir=\"...\">"))?;

        let mesh_files = xml_attribute_values(robot_xml, "mesh", "file");
        if mesh_files.is_empty() {
            return Err(format!("{robot_xml_name} contains no mesh assets"));
        }
        for mesh_file in &mesh_files {
            let asset_path = join_asset_path(&meshdir, mesh_file);
            if !files.contains_key(asset_path.as_str()) {
                return Err(format!(
                    "Missing embedded mesh asset {} referenced by {}",
                    asset_path, mesh_file
                ));
            }
        }

        let policy_path = preset.policy_config_name();
        let policy_json = std::str::from_utf8(
            files
                .get(policy_path)
                .ok_or_else(|| format!("Missing fetched {policy_path}"))?,
        )
        .map_err(|err| format!("{policy_path} is not valid UTF-8: {err}"))?;
        let policy: PolicyFile = serde_json::from_str(policy_json)
            .map_err(|err| format!("Failed to parse fetched {policy_path}: {err}"))?;
        let onnx_path = normalize_rel_path(&policy.onnx.path);
        if !files.contains_key(onnx_path.as_str()) {
            return Err(format!(
                "Policy config {policy_path} points to missing fetched ONNX file {}",
                onnx_path
            ));
        }
        if policy.controller_kind() == "open_duck_mini_walk" {
            if policy.onnx.meta.in_keys.is_empty() {
                return Err(format!(
                    "Policy {policy_path} is missing expected ONNX input keys"
                ));
            }
        } else if policy.onnx.meta.in_keys.len() < 4 {
            return Err(format!(
                "Policy {policy_path} is missing expected ONNX input keys"
            ));
        }
        if policy.onnx.meta.in_shapes.is_empty() {
            return Err(format!(
                "Policy {policy_path} metadata is missing input shapes"
            ));
        }
        if policy.onnx.meta.out_keys.is_empty() {
            return Err(format!(
                "Policy {policy_path} metadata is missing output keys"
            ));
        }

        let asset_meta: AssetMeta = serde_json::from_str(asset_meta_json)
            .map_err(|err| format!("Failed to parse fetched asset_meta.json: {err}"))?;
        if asset_meta.joint_names_isaac.is_empty() {
            return Err(format!(
                "Expected non-empty joint metadata, found {} joints",
                asset_meta.joint_names_isaac.len()
            ));
        }
        if asset_meta.default_joint_pos.len() != asset_meta.joint_names_isaac.len() {
            return Err(format!(
                "Expected matching default joint positions and joints, found {} positions for {} joints",
                asset_meta.default_joint_pos.len(),
                asset_meta.joint_names_isaac.len()
            ));
        }

        Ok(Self {
            preset,
            files,
            policy,
            asset_meta,
        })
    }
}

async fn fetch_browser_asset_bytes(
    preset: BrowserRobotPreset,
    path: &str,
) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let asset_url = format!("{}{}", preset.asset_base_path(), path);
    let response_value = JsFuture::from(window.fetch_with_str(&asset_url))
        .await
        .map_err(js_error_to_string)?;
    let response = response_value
        .dyn_into::<web_sys::Response>()
        .map_err(|_| format!("Fetch for {asset_url} did not return a Response"))?;
    if !response.ok() {
        return Err(format!(
            "Failed to fetch {asset_url}: HTTP {}",
            response.status()
        ));
    }

    let array_buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|err| format!("Failed to read array buffer for {asset_url}: {err:?}"))?,
    )
    .await
    .map_err(js_error_to_string)?
    .dyn_into::<ArrayBuffer>()
    .map_err(|_| format!("array_buffer() for {asset_url} did not return an ArrayBuffer"))?;
    Ok(Uint8Array::new(&array_buffer).to_vec())
}

fn single_xml_attribute(xml: &str, tag: &str, attr: &str) -> Option<String> {
    xml_attribute_values(xml, tag, attr).into_iter().next()
}

fn xml_attribute_values(xml: &str, tag: &str, attr: &str) -> Vec<String> {
    let mut values = Vec::new();
    let needle = format!("<{tag}");
    let attr_needle = format!("{attr}=\"");
    let mut rest = xml;

    while let Some(tag_start) = rest.find(&needle) {
        rest = &rest[tag_start + needle.len()..];
        let Some(tag_end) = rest.find('>') else {
            break;
        };
        let tag_body = &rest[..tag_end];
        if let Some(attr_start) = tag_body.find(&attr_needle) {
            let value_start = attr_start + attr_needle.len();
            if let Some(value_end) = tag_body[value_start..].find('"') {
                values.push(tag_body[value_start..value_start + value_end].to_string());
            }
        }
        rest = &rest[tag_end + 1..];
    }

    values
}

fn normalize_rel_path(path: &str) -> String {
    normalize_path_components(path)
}

fn join_asset_path(dir: &str, file: &str) -> String {
    if dir.trim().is_empty() {
        return normalize_path_components(file);
    }
    normalize_path_components(&format!("{dir}/{file}"))
}

fn normalize_path_components(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !parts.is_empty() {
                    parts.pop();
                }
            }
            _ => parts.push(part),
        }
    }
    parts.join("/")
}

fn js_error_to_string(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

async fn ort_smoke_test(
    model_bytes: &[u8],
    wasm_base_path: &str,
    config: &BrowserMujocoConfig,
) -> Result<JsValue, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let function = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("rustRoboticsOrtSmokeTest"),
    )?
    .dyn_into::<Function>()
    .map_err(|_| JsValue::from_str("rustRoboticsOrtSmokeTest is not a function"))?;

    let bytes = Uint8Array::from(model_bytes);
    let config = serde_wasm_bindgen::to_value(config)
        .map_err(|err| JsValue::from_str(&format!("Failed to serialize ORT config: {err}")))?;
    let promise = function
        .call3(
            &JsValue::NULL,
            &bytes.into(),
            &JsValue::from_str(wasm_base_path),
            &config,
        )?
        .dyn_into::<Promise>()
        .map_err(|_| JsValue::from_str("rustRoboticsOrtSmokeTest did not return a Promise"))?;

    wasm_bindgen_futures::JsFuture::from(promise).await
}

async fn mujoco_init(
    file_entries: &[(String, Vec<u8>)],
    policy_bytes: &[u8],
    mujoco_wasm_base_path: &str,
    ort_wasm_base_path: &str,
    config: &BrowserMujocoConfig,
) -> Result<JsValue, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let function = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("rustRoboticsMujocoInit"),
    )?
    .dyn_into::<Function>()
    .map_err(|_| JsValue::from_str("rustRoboticsMujocoInit is not a function"))?;

    let file_array = js_sys::Array::new();
    for (path, bytes) in file_entries {
        let entry = js_sys::Array::new();
        entry.push(&JsValue::from_str(path));
        entry.push(&Uint8Array::from(bytes.as_slice()).into());
        file_array.push(&entry.into());
    }
    let config = serde_wasm_bindgen::to_value(config)
        .map_err(|err| JsValue::from_str(&format!("Failed to serialize MuJoCo config: {err}")))?;

    let promise = function
        .call5(
            &JsValue::NULL,
            &file_array.into(),
            &Uint8Array::from(policy_bytes).into(),
            &JsValue::from_str(mujoco_wasm_base_path),
            &JsValue::from_str(ort_wasm_base_path),
            &config,
        )?
        .dyn_into::<Promise>()
        .map_err(|_| JsValue::from_str("rustRoboticsMujocoInit did not return a Promise"))?;

    wasm_bindgen_futures::JsFuture::from(promise).await
}

async fn mujoco_step(
    step_count: usize,
    command_vel_x: f32,
    command_vel_y: f32,
    use_setpoint_ball: bool,
) -> Result<JsValue, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let function = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("rustRoboticsMujocoStep"),
    )?
    .dyn_into::<Function>()
    .map_err(|_| JsValue::from_str("rustRoboticsMujocoStep is not a function"))?;

    let promise = function
        .call4(
            &JsValue::NULL,
            &JsValue::from_f64(step_count as f64),
            &JsValue::from_f64(command_vel_x as f64),
            &JsValue::from_f64(command_vel_y as f64),
            &JsValue::from_bool(use_setpoint_ball),
        )?
        .dyn_into::<Promise>()
        .map_err(|_| JsValue::from_str("rustRoboticsMujocoStep did not return a Promise"))?;

    wasm_bindgen_futures::JsFuture::from(promise).await
}

fn convert_browser_mesh_assets(meshes: &[BrowserMeshAssetSnapshot]) -> Vec<BrowserMeshAsset> {
    meshes
        .iter()
        .map(|mesh| {
            let mut vertices = Vec::new();
            let mut local_min = [f32::INFINITY; 3];
            let mut local_max = [f32::NEG_INFINITY; 3];
            let vertnum = mesh.positions.len() / 3;
            for tri in mesh.faces.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                if ia >= vertnum || ib >= vertnum || ic >= vertnum {
                    continue;
                }
                let a = [
                    mesh.positions[ia * 3],
                    mesh.positions[ia * 3 + 1],
                    mesh.positions[ia * 3 + 2],
                ];
                let b = [
                    mesh.positions[ib * 3],
                    mesh.positions[ib * 3 + 1],
                    mesh.positions[ib * 3 + 2],
                ];
                let c = [
                    mesh.positions[ic * 3],
                    mesh.positions[ic * 3 + 1],
                    mesh.positions[ic * 3 + 2],
                ];
                let face_normal = triangle_normal(a, b, c);
                let na = if mesh.normals.len() >= (ia + 1) * 3 {
                    normalize3([
                        mesh.normals[ia * 3],
                        mesh.normals[ia * 3 + 1],
                        mesh.normals[ia * 3 + 2],
                    ])
                } else {
                    face_normal
                };
                let nb = if mesh.normals.len() >= (ib + 1) * 3 {
                    normalize3([
                        mesh.normals[ib * 3],
                        mesh.normals[ib * 3 + 1],
                        mesh.normals[ib * 3 + 2],
                    ])
                } else {
                    face_normal
                };
                let nc = if mesh.normals.len() >= (ic + 1) * 3 {
                    normalize3([
                        mesh.normals[ic * 3],
                        mesh.normals[ic * 3 + 1],
                        mesh.normals[ic * 3 + 2],
                    ])
                } else {
                    face_normal
                };
                extend_bounds3(&mut local_min, &mut local_max, a);
                extend_bounds3(&mut local_min, &mut local_max, b);
                extend_bounds3(&mut local_min, &mut local_max, c);
                vertices.push(LocalMeshVertex {
                    position: a,
                    normal: na,
                });
                vertices.push(LocalMeshVertex {
                    position: b,
                    normal: nb,
                });
                vertices.push(LocalMeshVertex {
                    position: c,
                    normal: nc,
                });
            }
            if !local_min[0].is_finite() {
                local_min = [0.0; 3];
                local_max = [0.0; 3];
            }
            BrowserMeshAsset {
                vertices,
                local_min,
                local_max,
            }
        })
        .collect()
}

#[cfg(feature = "web_wgpu_viewport")]
fn report_setpoint(report: &BrowserMujocoReport) -> Option<[f32; 3]> {
    if report.setpoint_preview.len() >= 3 {
        Some([
            report.setpoint_preview[0],
            report.setpoint_preview[1],
            report.setpoint_preview[2],
        ])
    } else {
        None
    }
}

fn aligned_viewport_rect(ui: &Ui) -> Rect {
    let pixels_per_point = ui.ctx().pixels_per_point().max(1.0);
    let pixel = 1.0 / pixels_per_point;
    let mut rect = ui
        .available_rect_before_wrap()
        .shrink(pixel)
        .round_to_pixels(pixels_per_point);
    rect.min.x += pixel;
    rect.round_to_pixels(pixels_per_point)
}

#[cfg(feature = "web_wgpu_viewport")]
impl BrowserOrbitCamera {
    fn view_projection_matrix(&self, aspect: f32) -> [f32; 16] {
        mul_mat4(self.projection_matrix(aspect), self.view_matrix())
    }

    fn fitted_view_projection_matrix_points(&self, aspect: f32, points: &[[f32; 3]]) -> [f32; 16] {
        let Some((near, far)) = self.fitted_depth_range_points(points) else {
            return self.view_projection_matrix(aspect);
        };
        mul_mat4(self.projection_matrix_with_depth_range(aspect, near, far), self.view_matrix())
    }

    fn eye(&self) -> [f32; 3] {
        let forward = orbit_forward(self.azimuth_deg, self.elevation_deg);
        sub3(self.target, scale3(forward, self.distance))
    }

    fn view_matrix(&self) -> [f32; 16] {
        let eye = self.eye();
        let forward = normalize3(sub3(self.target, eye));
        let world_up = [0.0, 0.0, 1.0];
        let right = normalize3(cross3(forward, world_up));
        let up = normalize3(cross3(right, forward));

        [
            right[0],
            up[0],
            -forward[0],
            0.0,
            right[1],
            up[1],
            -forward[1],
            0.0,
            right[2],
            up[2],
            -forward[2],
            0.0,
            -dot3(right, eye),
            -dot3(up, eye),
            dot3(forward, eye),
            1.0,
        ]
    }

    fn projection_matrix(&self, aspect: f32) -> [f32; 16] {
        self.projection_matrix_with_depth_range(aspect, 0.02, 50.0)
    }

    fn projection_matrix_with_depth_range(&self, aspect: f32, near: f32, far: f32) -> [f32; 16] {
        let fov_y = 45.0_f32.to_radians();
        let near = near.max(1e-4);
        let far = far.max(near + 0.1);
        let f = 1.0 / (fov_y * 0.5).tan();

        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            -((far + near) / (far - near)),
            -1.0,
            0.0,
            0.0,
            -((2.0 * far * near) / (far - near)),
            0.0,
        ]
    }

    fn fitted_depth_range_points(&self, points: &[[f32; 3]]) -> Option<(f32, f32)> {
        let eye = self.eye();
        let forward = normalize3(sub3(self.target, eye));
        let mut max_depth = f32::NEG_INFINITY;
        for &point in points {
            let depth = dot3(sub3(point, eye), forward);
            if depth > 1e-4 {
                max_depth = max_depth.max(depth);
            }
        }
        if !max_depth.is_finite() {
            return None;
        }
        let near = 0.02;
        let far = (max_depth * 1.2).max(near + 1.0).min(50.0);
        Some((near, far))
    }

    fn intersect_plane_z(&self, rect: Rect, pointer: egui::Pos2, z: f32) -> Option<[f32; 3]> {
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

    fn world_ray(&self, rect: Rect, pointer: egui::Pos2) -> Option<([f32; 3], [f32; 3])> {
        if rect.width() <= f32::EPSILON || rect.height() <= f32::EPSILON {
            return None;
        }
        let ndc_x = ((pointer.x - rect.center().x) / (rect.width() * 0.5)).clamp(-1.0, 1.0);
        let ndc_y = (-(pointer.y - rect.center().y) / (rect.height() * 0.5)).clamp(-1.0, 1.0);
        let aspect = (rect.width() / rect.height()).max(0.1);
        let fov_y = 45.0_f32.to_radians();
        let half_height = (fov_y * 0.5).tan();
        let half_width = half_height * aspect;
        let eye = self.eye();
        let forward = normalize3(sub3(self.target, eye));
        let world_up = [0.0, 0.0, 1.0];
        let right = normalize3(cross3(forward, world_up));
        let up = normalize3(cross3(right, forward));
        let dir = normalize3(add3(
            add3(forward, scale3(right, ndc_x * half_width)),
            scale3(up, ndc_y * half_height),
        ));
        Some((eye, dir))
    }
}

#[cfg(feature = "web_wgpu_viewport")]
impl BrowserWgpuRenderer {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rect: Rect,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        target_format: wgpu::TextureFormat,
        scene: &BrowserWgpuSceneFrame,
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
            let uniform = BrowserWgpuSceneUniform {
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
        let mesh_pipeline = self
            .mesh_pipeline
            .as_ref()
            .ok_or_else(|| "WGPU mesh pipeline missing".to_string())?;
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
        let mesh_instance_buffer = self
            .mesh_instance_buffer
            .as_ref()
            .ok_or_else(|| "WGPU mesh instance buffer missing".to_string())?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("browser_mujoco_wgpu_prepare"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("browser_mujoco_wgpu_offscreen_pass"),
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
                    triangle_buffer
                        .slice(..(std::mem::size_of_val(scene.triangles.as_slice()) as u64)),
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
            label: Some("browser_mujoco_wgpu_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(WGPU_SCENE_SHADER.into()),
        });
        let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("browser_mujoco_wgpu_present_shader"),
            source: wgpu::ShaderSource::Wgsl(WGPU_PRESENT_SHADER.into()),
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("browser_mujoco_wgpu_uniform_layout"),
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
            label: Some("browser_mujoco_wgpu_uniform_buffer"),
            size: std::mem::size_of::<BrowserWgpuSceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("browser_mujoco_wgpu_uniform_bind_group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("browser_mujoco_wgpu_scene_layout"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
        };
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("browser_mujoco_wgpu_scene_pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_layout),
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
            label: Some("browser_mujoco_wgpu_line_pipeline"),
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
            array_stride: std::mem::size_of::<LocalMeshVertex>() as u64,
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
            label: Some("browser_mujoco_wgpu_mesh_pipeline"),
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
                label: Some("browser_mujoco_wgpu_present_bind_group_layout"),
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
            label: Some("browser_mujoco_wgpu_present_layout"),
            bind_group_layouts: &[&present_bind_group_layout],
            push_constant_ranges: &[],
        });
        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("browser_mujoco_wgpu_present_pipeline"),
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
            label: Some("browser_mujoco_wgpu_mesh_instance_buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.mesh_instance_capacity = 1024;
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("browser_mujoco_wgpu_sampler"),
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
            label: Some("browser_mujoco_wgpu_offscreen"),
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
            label: Some("browser_mujoco_wgpu_depth"),
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
            label: Some("browser_mujoco_wgpu_present_bind_group"),
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
        scene: &BrowserWgpuSceneFrame,
    ) -> Result<(), String> {
        let triangle_bytes = std::mem::size_of_val(scene.triangles.as_slice()) as u64;
        let line_bytes = std::mem::size_of_val(scene.lines.as_slice()) as u64;
        let mesh_instance_bytes = std::mem::size_of_val(scene.mesh_instances.as_slice()) as u64;

        if self.triangle_buffer.is_none() || triangle_bytes > self.triangle_capacity {
            self.triangle_capacity = triangle_bytes.max(1024).next_power_of_two();
            self.triangle_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("browser_mujoco_wgpu_triangle_buffer"),
                size: self.triangle_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if self.line_buffer.is_none() || line_bytes > self.line_capacity {
            self.line_capacity = line_bytes.max(1024).next_power_of_two();
            self.line_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("browser_mujoco_wgpu_line_buffer"),
                size: self.line_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if self.mesh_instance_buffer.is_none() || mesh_instance_bytes > self.mesh_instance_capacity {
            self.mesh_instance_capacity = mesh_instance_bytes.max(1024).next_power_of_two();
            self.mesh_instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("browser_mujoco_wgpu_mesh_instance_buffer"),
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
        mesh_assets: &[BrowserMeshAsset],
    ) -> Result<(), String> {
        for (mesh_id, asset) in mesh_assets.iter().enumerate() {
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
                            label: Some("browser_mujoco_wgpu_mesh_asset_buffer"),
                            contents: slice_as_u8(&asset.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                        vertex_count: asset.vertices.len() as u32,
                    },
                );
            }
        }
        Ok(())
    }
}

#[cfg(feature = "web_wgpu_viewport")]
fn build_mesh_instance(geom: &SharedGeomSnapshot, diagnostic_colors: bool) -> Option<MeshInstance> {
    let mesh_id = geom.data_id.max(0) as u32;
    Some(MeshInstance {
        model: shared_geom_model_matrix(geom),
        color: shared_display_geom_color(geom, diagnostic_colors),
        mesh_id,
        _padding: [0; 3],
    })
}

#[cfg(feature = "web_wgpu_viewport")]
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

#[cfg(feature = "web_wgpu_viewport")]
fn extend_depth_points_with_mesh_bounds(
    depth_points: &mut Vec<[f32; 3]>,
    asset: &BrowserMeshAsset,
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

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize3(cross3(sub3(b, a), sub3(c, a)))
}

#[cfg(feature = "web_wgpu_viewport")]
fn orbit_forward(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    [
        elevation.cos() * azimuth.cos(),
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
    ]
}

#[cfg(feature = "web_wgpu_viewport")]
fn mul_mat4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum::<f32>();
        }
    }
    out
}

#[cfg(feature = "web_wgpu_viewport")]
fn transform_point_mat4(model: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    [
        model[0][0] * point[0] + model[1][0] * point[1] + model[2][0] * point[2] + model[3][0],
        model[0][1] * point[0] + model[1][1] * point[1] + model[2][1] * point[2] + model[3][1],
        model[0][2] * point[0] + model[1][2] * point[1] + model[2][2] * point[2] + model[3][2],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(feature = "web_wgpu_viewport")]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(feature = "web_wgpu_viewport")]
fn scale3(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn extend_bounds3(min: &mut [f32; 3], max: &mut [f32; 3], point: [f32; 3]) {
    for axis in 0..3 {
        min[axis] = min[axis].min(point[axis]);
        max[axis] = max[axis].max(point[axis]);
    }
}

fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(feature = "web_wgpu_viewport")]
fn slice_as_u8<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(feature = "web_wgpu_viewport")]
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

#[cfg(feature = "web_wgpu_viewport")]
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
