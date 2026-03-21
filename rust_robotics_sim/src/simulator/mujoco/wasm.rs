use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[cfg(feature = "web_glow_viewport")]
use std::sync::{Arc, Mutex};

use eframe::emath::GuiRounding;
#[cfg(feature = "web_glow_viewport")]
use eframe::{egui_glow, glow};
use egui::{pos2, vec2, Align2, Color32, FontId, Rect, RichText, Sense, Ui};
#[cfg(feature = "web_glow_viewport")]
use glow::HasContext;
use js_sys::{ArrayBuffer, Function, Promise, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::{spawn_local, JsFuture};

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
    fn label(self) -> &'static str {
        match self {
            Self::Go2 => "Go2",
            Self::OpenDuckMini => "Open Duck Mini",
        }
    }

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
    diagnostic_colors: bool,
    generation: Rc<RefCell<u64>>,
    asset_state: Rc<RefCell<BrowserAssetState>>,
    ort_state: Rc<RefCell<BrowserOrtState>>,
    mujoco_state: Rc<RefCell<BrowserMujocoState>>,
    mujoco_step_in_flight: Rc<RefCell<bool>>,
    compiled_mesh_assets: Rc<RefCell<Vec<BrowserMeshAsset>>>,
    #[cfg(feature = "web_glow_viewport")]
    glow_renderer: Arc<Mutex<BrowserGlowRenderer>>,
    #[cfg(feature = "web_glow_viewport")]
    glow_scene: Arc<Mutex<BrowserGlowSceneFrame>>,
    camera: BrowserOrbitCamera,
    camera_initialized: bool,
}

impl Default for WasmMujocoBackend {
    fn default() -> Self {
        Self {
            active: false,
            selected_robot: BrowserRobotPreset::Go2,
            diagnostic_colors: false,
            generation: Rc::new(RefCell::new(0)),
            asset_state: Rc::new(RefCell::new(BrowserAssetState::Idle)),
            ort_state: Rc::new(RefCell::new(BrowserOrtState::Idle)),
            mujoco_state: Rc::new(RefCell::new(BrowserMujocoState::Idle)),
            mujoco_step_in_flight: Rc::new(RefCell::new(false)),
            compiled_mesh_assets: Rc::new(RefCell::new(Vec::new())),
            #[cfg(feature = "web_glow_viewport")]
            glow_renderer: Arc::new(Mutex::new(BrowserGlowRenderer::default())),
            #[cfg(feature = "web_glow_viewport")]
            glow_scene: Arc::new(Mutex::new(BrowserGlowSceneFrame::default())),
            camera: BrowserOrbitCamera::default(),
            camera_initialized: false,
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
    body_names_isaac: Vec<String>,
    joint_names_isaac: Vec<String>,
    default_joint_pos: Vec<f32>,
}

struct BrowserAssetBundle {
    preset: BrowserRobotPreset,
    files: BTreeMap<String, Vec<u8>>,
    scene_include: String,
    meshdir: String,
    mesh_files: Vec<String>,
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

#[derive(Default)]
enum BrowserOrtState {
    #[default]
    Idle,
    Loading,
    Ready(BrowserOrtReport),
    Error(String),
}

#[derive(Debug, Deserialize)]
struct BrowserOrtReport {
    input_names: Vec<String>,
    output_names: Vec<String>,
    output_summaries: Vec<BrowserOrtTensorSummary>,
}

#[derive(Debug, Deserialize)]
struct BrowserOrtTensorSummary {
    name: String,
    dims: Vec<usize>,
    first: Option<f32>,
}

#[derive(Default)]
enum BrowserMujocoState {
    #[default]
    Idle,
    Loading,
    Ready(BrowserMujocoReport),
    Error(String),
}

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

#[derive(Serialize)]
struct BrowserViewportConfig {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    pixels_per_point: f32,
    visible: bool,
    diagnostic_colors: bool,
}

#[derive(Clone)]
struct BrowserMeshAsset {
    triangles: Vec<LocalMeshVertex>,
}

#[derive(Debug, Deserialize, Clone)]
struct BrowserMeshAssetSnapshot {
    positions: Vec<f32>,
    normals: Vec<f32>,
    faces: Vec<u32>,
}

#[derive(Clone, Copy, Default)]
struct LocalMeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[cfg(feature = "web_glow_viewport")]
#[derive(Default, Clone)]
struct BrowserGlowSceneFrame {
    triangles: Vec<GlVertex>,
    lines: Vec<GlVertex>,
    view_proj: [f32; 16],
}

#[cfg(feature = "web_glow_viewport")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GlVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

#[cfg(feature = "web_glow_viewport")]
impl GlVertex {
    fn world(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }
}

#[cfg(feature = "web_glow_viewport")]
#[derive(Default)]
struct BrowserGlowRenderer {
    program: Option<glow::Program>,
    triangle_vao: Option<glow::VertexArray>,
    triangle_vbo: Option<glow::Buffer>,
    line_vao: Option<glow::VertexArray>,
    line_vbo: Option<glow::Buffer>,
    present_program: Option<glow::Program>,
    present_vao: Option<glow::VertexArray>,
    present_vbo: Option<glow::Buffer>,
    offscreen_fbo: Option<glow::Framebuffer>,
    offscreen_color: Option<glow::Texture>,
    offscreen_depth: Option<glow::Renderbuffer>,
    offscreen_size: [i32; 2],
    u_view_proj: Option<glow::UniformLocation>,
    u_unlit: Option<glow::UniformLocation>,
    u_present_texture: Option<glow::UniformLocation>,
    last_error: Option<String>,
}

// The web build uses this renderer only on the single browser main thread, but
// egui_glow's callback API requires the captured state to be Send + Sync.
#[cfg(feature = "web_glow_viewport")]
unsafe impl Send for BrowserGlowRenderer {}
#[cfg(feature = "web_glow_viewport")]
unsafe impl Sync for BrowserGlowRenderer {}

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

        self.ensure_mujoco_step_started(sim_speed);
    }

    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        let outer = ui.available_rect_before_wrap();
        let gap = 12.0;
        let controls_width = outer.width().min(360.0).max(300.0);
        let controls_rect =
            Rect::from_min_size(outer.min, vec2(controls_width, outer.height().max(280.0)));
        let viewport_rect = Rect::from_min_max(
            pos2((controls_rect.max.x + gap).min(outer.max.x), outer.min.y),
            outer.max,
        );

        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(controls_rect)
                .layout(egui::Layout::top_down(egui::Align::Min)),
            |ui| {
                ui.heading("mujoco");
                ui.label("Browser app shell is active.");
                let previous_robot = self.selected_robot;
                egui::ComboBox::from_label("Robot")
                    .selected_text(self.selected_robot.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.selected_robot,
                            BrowserRobotPreset::Go2,
                            BrowserRobotPreset::Go2.label(),
                        );
                        ui.selectable_value(
                            &mut self.selected_robot,
                            BrowserRobotPreset::OpenDuckMini,
                            BrowserRobotPreset::OpenDuckMini.label(),
                        );
                    });
                if self.selected_robot != previous_robot {
                    self.reset_selected_robot_state();
                }
                ui.label(match self.selected_robot {
                    BrowserRobotPreset::Go2 => {
                        "Go2 runs the facet policy and uses the draggable setpoint ball as the command input."
                    }
                    BrowserRobotPreset::OpenDuckMini => {
                        "Open Duck Mini runs the BEST_WALK_ONNX policy and uses the draggable setpoint ball to generate walking commands."
                    }
                });
                ui.label(format!(
                    "Status: {}",
                    if self.active {
                        "tab active"
                    } else {
                        "tab inactive"
                    }
                ));
                ui.separator();
                ui.label("Control: drag the red setpoint ball in the viewport.");
                ui.checkbox(&mut self.diagnostic_colors, "Diagnostic colors");
                if ui.button("Reset view").clicked() {
                    self.camera = BrowserOrbitCamera::default();
                    self.camera_initialized = false;
                    #[cfg(not(feature = "web_glow_viewport"))]
                    self.reset_browser_viewport_camera();
                }
                ui.separator();

                if self.active {
                    self.ensure_browser_assets_started();
                }

                let asset_state_kind = {
                    let state = self.asset_state.borrow();
                    match &*state {
                        BrowserAssetState::Idle => BrowserAssetUiState::Idle,
                        BrowserAssetState::Loading => BrowserAssetUiState::Loading,
                        BrowserAssetState::Ready(bundle) => {
                            BrowserAssetUiState::Ready(Rc::clone(bundle))
                        }
                        BrowserAssetState::Error(err) => BrowserAssetUiState::Error(err.clone()),
                    }
                };

                match asset_state_kind {
                    BrowserAssetUiState::Idle => {
                        ui.label("Browser asset bundle: idle");
                    }
                    BrowserAssetUiState::Loading => {
                        ui.label("Browser asset bundle: loading...");
                        if self.active {
                            ui.ctx().request_repaint();
                        }
                    }
                    BrowserAssetUiState::Ready(bundle) => {
                        if self.active {
                            self.ensure_ort_smoke_test_started(bundle.as_ref());
                            self.ensure_mujoco_runtime_started(bundle.as_ref());
                            ui.ctx().request_repaint();
                        }
                        self.ui_asset_report(ui, bundle.as_ref());
                        ui.separator();
                        self.ui_ort_report(ui);
                        ui.separator();
                        self.ui_mujoco_report(ui);
                    }
                    BrowserAssetUiState::Error(err) => {
                        ui.colored_label(Color32::LIGHT_RED, format!("Asset bundle error: {err}"));
                    }
                }
            },
        );

        ui.painter().rect_filled(
            Rect::from_min_max(
                pos2(controls_rect.max.x, outer.min.y),
                pos2((controls_rect.max.x + gap).min(outer.max.x), outer.max.y),
            ),
            0.0,
            ui.visuals().panel_fill,
        );

        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(viewport_rect)
                .layout(egui::Layout::top_down(egui::Align::Min)),
            |ui| {
                self.ui_viewport(ui, frame);
            },
        );
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
        #[cfg(not(feature = "web_glow_viewport"))]
        if !active {
            self.configure_browser_viewport(&BrowserViewportConfig {
                left: 0.0,
                top: 0.0,
                width: 0.0,
                height: 0.0,
                pixels_per_point: 1.0,
                visible: false,
                diagnostic_colors: self.diagnostic_colors,
            });
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
        self.compiled_mesh_assets.borrow_mut().clear();
        #[cfg(not(feature = "web_glow_viewport"))]
        self.reset_browser_runtime();
    }

    fn reset_selected_robot_state(&mut self) {
        *self.generation.borrow_mut() += 1;
        *self.asset_state.borrow_mut() = BrowserAssetState::Idle;
        self.reset_loaded_policy_state();
        self.camera = BrowserOrbitCamera::default();
        self.camera_initialized = false;
        #[cfg(not(feature = "web_glow_viewport"))]
        self.hide_browser_viewport();
    }

    fn ui_asset_report(&self, ui: &mut Ui, bundle: &BrowserAssetBundle) {
        let policy = bundle.policy();
        ui.label(
            RichText::new("Browser asset bundle")
                .strong()
                .color(Color32::LIGHT_GREEN),
        );
        ui.label(format!("Files: {} fetched assets", bundle.files.len()));
        ui.label(format!("Robot preset: {}", bundle.preset.label()));
        ui.label(format!("Scene include: {}", bundle.scene_include));
        ui.label(format!(
            "Meshes: {} files under {}",
            bundle.mesh_files.len(),
            bundle.meshdir
        ));
        ui.label(format!(
            "Policy: {} outputs, {:?} input groups, ONNX {}",
            policy.onnx.meta.out_keys.len(),
            policy.onnx.meta.in_shapes,
            bundle
                .files
                .get(normalize_rel_path(&policy.onnx.path).as_str())
                .map(|bytes| format_bytes(bytes.len()))
                .unwrap_or_else(|| "missing".to_string())
        ));
        let preview_outputs = bundle
            .policy()
            .onnx
            .meta
            .out_keys
            .iter()
            .take(4)
            .map(OnnxKey::joined)
            .collect::<Vec<_>>()
            .join(", ");
        ui.label(format!("Output preview: {preview_outputs}"));
        ui.label(format!(
            "Control: mode={} action_scale {:.2}, kp {:.2}, kd {:.2}",
            policy.command_mode(),
            policy.action_scale,
            policy.stiffness,
            policy.damping
        ));
        ui.label(format!(
            "Robot metadata: {} bodies, {} joints, {} default joint positions",
            bundle.asset_meta.body_names_isaac.len(),
            bundle.asset_meta.joint_names_isaac.len(),
            bundle.asset_meta.default_joint_pos.len()
        ));

        egui::CollapsingHeader::new("Mesh assets")
            .default_open(false)
            .show(ui, |ui| {
                for mesh in &bundle.mesh_files {
                    ui.label(mesh);
                }
            });
    }

    fn ui_ort_report(&self, ui: &mut Ui) {
        ui.label(
            RichText::new("Browser ONNX smoke test")
                .strong()
                .color(Color32::LIGHT_GREEN),
        );
        match &*self.ort_state.borrow() {
            BrowserOrtState::Idle => {
                ui.label("Status: idle");
            }
            BrowserOrtState::Loading => {
                ui.label("Status: loading ONNX Runtime Web session...");
            }
            BrowserOrtState::Ready(report) => {
                ui.label("Status: ONNX Runtime Web session created and first inference succeeded");
                ui.label(format!("Model inputs: {}", report.input_names.join(", ")));
                ui.label(format!(
                    "Model outputs: {}",
                    report
                        .output_names
                        .iter()
                        .take(8)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                egui::CollapsingHeader::new("Output summary")
                    .default_open(true)
                    .show(ui, |ui| {
                        for tensor in &report.output_summaries {
                            ui.label(format!(
                                "{} dims={:?} first={:?}",
                                tensor.name, tensor.dims, tensor.first
                            ));
                        }
                    });
            }
            BrowserOrtState::Error(err) => {
                ui.colored_label(Color32::LIGHT_RED, format!("Status: {err}"));
            }
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
                serde_wasm_bindgen::from_value::<BrowserOrtReport>(js_value)
                    .map_err(|err| JsValue::from_str(&err.to_string()))
            }
            .await;

            if *generation.borrow() != expected_generation {
                return;
            }
            *state.borrow_mut() = match result {
                Ok(report) => BrowserOrtState::Ready(report),
                Err(err) => BrowserOrtState::Error(js_error_to_string(err)),
            };
        });
    }

    fn ui_mujoco_report(&self, ui: &mut Ui) {
        ui.label(
            RichText::new("Browser MuJoCo runtime")
                .strong()
                .color(Color32::LIGHT_GREEN),
        );
        match &*self.mujoco_state.borrow() {
            BrowserMujocoState::Idle => {
                ui.label("Status: idle");
            }
            BrowserMujocoState::Loading => {
                ui.label("Status: loading MuJoCo wasm module and scene...");
            }
            BrowserMujocoState::Ready(report) => {
                ui.label("Status: MuJoCo wasm runtime initialized");
                ui.label(format!(
                    "Model: nbody={} ngeom={} nv={} nu={} dt={:.4}",
                    report.nbody, report.ngeom, report.nv, report.nu, report.timestep
                ));
                ui.label(format!(
                    "Live state: sim_time={:.4}s step_count={}",
                    report.sim_time, report.step_count
                ));
                ui.label(format!(
                    "Robot: {} | Command mode: setpoint ball",
                    self.selected_robot.label()
                ));
                ui.label(format!(
                    "Policy command: {} | setpoint={:?}",
                    report.command_mode, report.setpoint_preview
                ));
                ui.label(format!(
                    "Drag debug: mode={} downs={} moves={}",
                    report.debug_drag_mode, report.debug_pointer_downs, report.debug_pointer_moves
                ));
                ui.label(format!(
                    "Display: {:.1} FPS | step: {:.2} ms last, {:.2} ms avg",
                    report.display_fps, report.last_step_wall_ms, report.avg_step_wall_ms
                ));
                ui.label(format!(
                    "Policy: {:.2} ms last, {:.2} ms avg | Physics: {:.2} ms last, {:.2} ms avg",
                    report.last_policy_wall_ms,
                    report.avg_policy_wall_ms,
                    report.last_physics_wall_ms,
                    report.avg_physics_wall_ms
                ));
                ui.label(format!(
                    "Overlay render: {:.2} ms last, {:.2} ms avg",
                    report.last_overlay_wall_ms, report.avg_overlay_wall_ms
                ));
                ui.label(format!(
                    "Policy inputs: {}",
                    report.policy_inputs.join(", ")
                ));
                ui.label(format!(
                    "Policy outputs: {}",
                    report
                        .policy_outputs
                        .iter()
                        .take(8)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                ui.label(format!("Action preview: {:?}", report.last_action_preview));
                ui.label(format!("qpos preview: {:?}", report.qpos_preview));
                ui.label(format!("xpos preview: {:?}", report.xpos_preview));
            }
            BrowserMujocoState::Error(err) => {
                ui.colored_label(Color32::LIGHT_RED, format!("Status: {err}"));
            }
        }
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

    fn ensure_mujoco_step_started(&mut self, sim_speed: usize) {
        if *self.mujoco_step_in_flight.borrow() {
            return;
        }
        if !matches!(*self.mujoco_state.borrow(), BrowserMujocoState::Ready(_)) {
            return;
        }

        *self.mujoco_step_in_flight.borrow_mut() = true;
        let state = Rc::clone(&self.mujoco_state);
        let step_in_flight = Rc::clone(&self.mujoco_step_in_flight);
        let generation = Rc::clone(&self.generation);
        let expected_generation = *generation.borrow();
        let step_count = sim_speed.max(1);
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
            *state.borrow_mut() = match result {
                Ok(report) => BrowserMujocoState::Ready(report),
                Err(err) => BrowserMujocoState::Error(js_error_to_string(err)),
            };
        });
    }

    fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        let rect = aligned_viewport_rect(ui);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        #[cfg(feature = "web_glow_viewport")]
        self.update_camera(ui, &response);
        #[cfg(not(feature = "web_glow_viewport"))]
        let _ = response;

        let state_binding = self.mujoco_state.borrow();
        #[cfg(feature = "web_glow_viewport")]
        let report = match &*state_binding {
            BrowserMujocoState::Ready(report) => report.clone(),
            BrowserMujocoState::Idle => {
                #[cfg(not(feature = "web_glow_viewport"))]
                self.hide_browser_viewport();
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
                #[cfg(not(feature = "web_glow_viewport"))]
                self.hide_browser_viewport();
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
                #[cfg(not(feature = "web_glow_viewport"))]
                self.hide_browser_viewport();
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
        #[cfg(not(feature = "web_glow_viewport"))]
        match &*state_binding {
            BrowserMujocoState::Ready(_) => {}
            BrowserMujocoState::Idle => {
                self.hide_browser_viewport();
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
                self.hide_browser_viewport();
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
                self.hide_browser_viewport();
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

        #[cfg(not(feature = "web_glow_viewport"))]
        {
            let _ = frame;
            self.configure_browser_viewport(&BrowserViewportConfig {
                left: rect.min.x,
                top: rect.min.y,
                width: rect.width(),
                height: rect.height(),
                pixels_per_point: ui.ctx().pixels_per_point(),
                visible: true,
                diagnostic_colors: self.diagnostic_colors,
            });
            ui.painter()
                .rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 24));
            return;
        }

        #[cfg(feature = "web_glow_viewport")]
        {
            let mesh_assets = self.compiled_mesh_assets.borrow().clone();
            if mesh_assets.is_empty() {
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
            let scene = self.build_gl_scene_frame(
                &report,
                mesh_assets.as_slice(),
                rect.width() / rect.height(),
            );
            if let Ok(mut scene_state) = self.glow_scene.lock() {
                *scene_state = scene;
            }

            let Some(frame) = frame else {
                self.render_software_2d(ui, rect, &report, mesh_assets.as_slice());
                return;
            };
            let Some(gl) = frame.gl() else {
                self.render_software_2d(ui, rect, &report, mesh_assets.as_slice());
                return;
            };

            let renderer = self.glow_renderer.clone();
            let scene = self.glow_scene.clone();
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                    let render_result = (|| -> Result<(), String> {
                        let scene = scene
                            .lock()
                            .map_err(|_| "Failed to lock wasm glow scene".to_string())?;
                        let mut renderer = renderer
                            .lock()
                            .map_err(|_| "Failed to lock wasm glow renderer".to_string())?;
                        renderer.paint(painter.gl(), info, &scene)
                    })();

                    if let Err(err) = render_result {
                        if let Ok(mut renderer) = renderer.lock() {
                            renderer.last_error = Some(err);
                        }
                    }
                })),
            });

            let overlay = format!(
                "Browser glow viewport\nSim t={:.2}s  steps={}  cmd=({:+.2}, {:+.2})\nDrag: orbit | Right-drag: pan | Wheel: zoom | Double-click: reset view",
                report.sim_time, report.step_count, report.command_vel_x, report.command_vel_y
            );
            ui.painter().text(
                rect.left_top() + vec2(12.0, 12.0),
                Align2::LEFT_TOP,
                overlay,
                FontId::monospace(12.0),
                Color32::from_gray(210),
            );
            let _ = gl;
        }
    }

    #[cfg(feature = "web_glow_viewport")]
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

    #[cfg(feature = "web_glow_viewport")]
    fn build_gl_scene_frame(
        &self,
        report: &BrowserMujocoReport,
        mesh_assets: &[BrowserMeshAsset],
        aspect: f32,
    ) -> BrowserGlowSceneFrame {
        let mut frame = BrowserGlowSceneFrame::default();
        append_grid_lines(&mut frame.lines);

        for geom in &report.geoms {
            match geom.type_id {
                MJ_GEOM_PLANE => {}
                MJ_GEOM_SPHERE => {
                    append_sphere_geom(&mut frame.triangles, geom, 12, 20, self.diagnostic_colors)
                }
                MJ_GEOM_CAPSULE => {
                    append_capsule_geom(&mut frame.triangles, geom, 10, 18, self.diagnostic_colors)
                }
                MJ_GEOM_CYLINDER => {
                    append_cylinder_geom(&mut frame.triangles, geom, 18, self.diagnostic_colors)
                }
                MJ_GEOM_BOX => append_box_geom(&mut frame.triangles, geom, self.diagnostic_colors),
                MJ_GEOM_MESH => append_mesh_geom(
                    &mut frame.triangles,
                    geom,
                    mesh_assets,
                    self.diagnostic_colors,
                ),
                MJ_GEOM_LINE => append_line_geom(&mut frame.lines, geom, self.diagnostic_colors),
                _ => {}
            }
        }

        frame.view_proj = self.camera.view_projection_matrix(aspect.max(0.1));
        frame
    }

    #[cfg(feature = "web_glow_viewport")]
    fn update_camera(&mut self, ui: &Ui, response: &egui::Response) {
        if response.double_clicked() {
            self.camera = BrowserOrbitCamera::default();
            self.camera_initialized = false;
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.camera.azimuth_deg -= delta.x * 0.25;
            self.camera.elevation_deg =
                (self.camera.elevation_deg + delta.y * 0.2).clamp(-89.0, 89.0);
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            let forward = orbit_forward(self.camera.azimuth_deg, self.camera.elevation_deg);
            let right = normalize3(cross3(forward, [0.0, 0.0, 1.0]));
            let up = normalize3(cross3(right, forward));
            let pan_scale = self.camera.distance * 0.0025;
            for axis in 0..3 {
                self.camera.target[axis] -=
                    (right[axis] * delta.x + up[axis] * delta.y) * pan_scale;
            }
        }

        if response.hovered() {
            let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
            if scroll_y.abs() > f32::EPSILON {
                let zoom = (1.0 - scroll_y * 0.0015).clamp(0.8, 1.25);
                self.camera.distance = (self.camera.distance * zoom).clamp(0.35, 40.0);
            }
        }
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
                "WebGL viewport unavailable\nsim_time={:.2}s step_count={}",
                report.sim_time, report.step_count
            ),
            FontId::proportional(18.0),
            Color32::from_gray(210),
        );
    }

    fn render_wgpu_experiment_placeholder(
        &self,
        ui: &mut Ui,
        rect: Rect,
        report: &BrowserMujocoReport,
    ) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, Color32::from_rgb(18, 22, 28));
        painter.text(
            rect.center_top() + vec2(0.0, 28.0),
            Align2::CENTER_TOP,
            "MuJoCo viewport disabled for web wgpu experiment",
            FontId::proportional(18.0),
            Color32::from_gray(220),
        );
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            format!(
                "sim_time={:.2}s\nstep_count={}\ncmd=({:+.2}, {:+.2})",
                report.sim_time, report.step_count, report.command_vel_x, report.command_vel_y
            ),
            FontId::monospace(15.0),
            Color32::from_gray(210),
        );
    }

    #[cfg(not(feature = "web_glow_viewport"))]
    fn hide_browser_viewport(&self) {
        self.configure_browser_viewport(&BrowserViewportConfig {
            left: 0.0,
            top: 0.0,
            width: 0.0,
            height: 0.0,
            pixels_per_point: 1.0,
            visible: false,
            diagnostic_colors: self.diagnostic_colors,
        });
    }

    #[cfg(not(feature = "web_glow_viewport"))]
    fn configure_browser_viewport(&self, config: &BrowserViewportConfig) {
        let Ok(config_value) = serde_wasm_bindgen::to_value(config) else {
            return;
        };
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(function) = Reflect::get(
            window.as_ref(),
            &JsValue::from_str("rustRoboticsMujocoConfigureViewport"),
        ) else {
            return;
        };
        let Ok(function) = function.dyn_into::<Function>() else {
            return;
        };
        let _ = function.call1(&JsValue::NULL, &config_value);
    }

    #[cfg(not(feature = "web_glow_viewport"))]
    fn reset_browser_viewport_camera(&self) {
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(function) = Reflect::get(
            window.as_ref(),
            &JsValue::from_str("rustRoboticsMujocoResetViewportCamera"),
        ) else {
            return;
        };
        let Ok(function) = function.dyn_into::<Function>() else {
            return;
        };
        let _ = function.call0(&JsValue::NULL);
    }

    #[cfg(not(feature = "web_glow_viewport"))]
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
            scene_include,
            meshdir,
            mesh_files,
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

fn format_bytes(byte_count: usize) -> String {
    const KIB: f32 = 1024.0;
    const MIB: f32 = 1024.0 * 1024.0;

    let bytes = byte_count as f32;
    if bytes >= MIB {
        format!("{:.2} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{byte_count} B")
    }
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
            let mut triangles = Vec::new();
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
                triangles.push(LocalMeshVertex {
                    position: a,
                    normal: na,
                });
                triangles.push(LocalMeshVertex {
                    position: b,
                    normal: nb,
                });
                triangles.push(LocalMeshVertex {
                    position: c,
                    normal: nc,
                });
            }
            BrowserMeshAsset { triangles }
        })
        .collect()
}

const MJ_GEOM_PLANE: i32 = 0;
const MJ_GEOM_SPHERE: i32 = 2;
const MJ_GEOM_CAPSULE: i32 = 3;
const MJ_GEOM_CYLINDER: i32 = 5;
const MJ_GEOM_BOX: i32 = 6;
const MJ_GEOM_MESH: i32 = 7;
const MJ_GEOM_LINE: i32 = 9;

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

#[cfg(feature = "web_glow_viewport")]
impl BrowserOrbitCamera {
    fn view_projection_matrix(&self, aspect: f32) -> [f32; 16] {
        mul_mat4(self.projection_matrix(aspect), self.view_matrix())
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
        let fov_y = 45.0_f32.to_radians();
        let near = 0.02;
        let far = 50.0;
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
}

#[cfg(feature = "web_glow_viewport")]
impl BrowserGlowRenderer {
    fn ensure_initialized(&mut self, gl: &Arc<glow::Context>) -> Result<(), String> {
        if self.program.is_some() {
            return Ok(());
        }

        unsafe {
            let program = create_gl_program(gl, GLOW_VERTEX_SHADER, GLOW_FRAGMENT_SHADER)?;
            let triangle_vao = gl
                .create_vertex_array()
                .map_err(|err| format!("Failed to create triangle VAO: {err}"))?;
            let triangle_vbo = gl
                .create_buffer()
                .map_err(|err| format!("Failed to create triangle VBO: {err}"))?;
            let line_vao = gl
                .create_vertex_array()
                .map_err(|err| format!("Failed to create line VAO: {err}"))?;
            let line_vbo = gl
                .create_buffer()
                .map_err(|err| format!("Failed to create line VBO: {err}"))?;
            let present_program =
                create_gl_program(gl, PRESENT_VERTEX_SHADER, PRESENT_FRAGMENT_SHADER)?;
            let present_vao = gl
                .create_vertex_array()
                .map_err(|err| format!("Failed to create present VAO: {err}"))?;
            let present_vbo = gl
                .create_buffer()
                .map_err(|err| format!("Failed to create present VBO: {err}"))?;
            let offscreen_fbo = gl
                .create_framebuffer()
                .map_err(|err| format!("Failed to create offscreen FBO: {err}"))?;
            let offscreen_color = gl
                .create_texture()
                .map_err(|err| format!("Failed to create offscreen color texture: {err}"))?;
            let offscreen_depth = gl
                .create_renderbuffer()
                .map_err(|err| format!("Failed to create offscreen depth renderbuffer: {err}"))?;

            setup_vertex_array(gl, program, triangle_vao, triangle_vbo)?;
            setup_vertex_array(gl, program, line_vao, line_vbo)?;
            setup_present_quad(gl, present_program, present_vao, present_vbo)?;

            self.u_view_proj = gl.get_uniform_location(program, "u_view_proj");
            self.u_unlit = gl.get_uniform_location(program, "u_unlit");
            self.u_present_texture = gl.get_uniform_location(present_program, "u_texture");
            self.program = Some(program);
            self.triangle_vao = Some(triangle_vao);
            self.triangle_vbo = Some(triangle_vbo);
            self.line_vao = Some(line_vao);
            self.line_vbo = Some(line_vbo);
            self.present_program = Some(present_program);
            self.present_vao = Some(present_vao);
            self.present_vbo = Some(present_vbo);
            self.offscreen_fbo = Some(offscreen_fbo);
            self.offscreen_color = Some(offscreen_color);
            self.offscreen_depth = Some(offscreen_depth);
            self.offscreen_size = [0, 0];
            self.last_error = None;
        }

        Ok(())
    }

    fn ensure_offscreen_target(
        &mut self,
        gl: &Arc<glow::Context>,
        width: i32,
        height: i32,
    ) -> Result<(), String> {
        let width = width.max(2);
        let height = height.max(2);
        if self.offscreen_size == [width, height] {
            return Ok(());
        }

        let fbo = self
            .offscreen_fbo
            .ok_or_else(|| "Offscreen FBO missing".to_string())?;
        let color = self
            .offscreen_color
            .ok_or_else(|| "Offscreen color texture missing".to_string())?;
        let depth = self
            .offscreen_depth
            .ok_or_else(|| "Offscreen depth buffer missing".to_string())?;

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width,
                height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);

            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, width, height);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(color),
                0,
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(depth),
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(format!("Offscreen framebuffer incomplete: 0x{status:04x}"));
            }
        }

        self.offscreen_size = [width, height];
        Ok(())
    }

    fn paint(
        &mut self,
        gl: &Arc<glow::Context>,
        info: egui::PaintCallbackInfo,
        scene: &BrowserGlowSceneFrame,
    ) -> Result<(), String> {
        self.ensure_initialized(gl)?;
        let viewport = info.viewport_in_pixels();
        let clip = info.clip_rect_in_pixels();
        let program = self
            .program
            .ok_or_else(|| "Glow program missing".to_string())?;
        let present_program = self
            .present_program
            .ok_or_else(|| "Present program missing".to_string())?;
        let fbo = self
            .offscreen_fbo
            .ok_or_else(|| "Offscreen FBO missing".to_string())?;
        let offscreen_color = self
            .offscreen_color
            .ok_or_else(|| "Offscreen color missing".to_string())?;
        self.ensure_offscreen_target(gl, viewport.width_px, viewport.height_px)?;

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.viewport(0, 0, viewport.width_px, viewport.height_px);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.clear_color(0.07, 0.09, 0.12, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            gl.use_program(Some(program));
            if let Some(location) = self.u_view_proj.as_ref() {
                gl.uniform_matrix_4_f32_slice(Some(location), false, &scene.view_proj);
            }

            if !scene.triangles.is_empty() {
                if let Some(location) = self.u_unlit.as_ref() {
                    gl.uniform_1_i32(Some(location), 0);
                }
                gl.enable(glow::CULL_FACE);
                gl.cull_face(glow::BACK);
                gl.bind_vertex_array(self.triangle_vao);
                gl.bind_buffer(glow::ARRAY_BUFFER, self.triangle_vbo);
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    slice_as_u8(&scene.triangles),
                    glow::DYNAMIC_DRAW,
                );
                gl.draw_arrays(glow::TRIANGLES, 0, scene.triangles.len() as i32);
                gl.disable(glow::CULL_FACE);
            }

            if !scene.lines.is_empty() {
                if let Some(location) = self.u_unlit.as_ref() {
                    gl.uniform_1_i32(Some(location), 1);
                }
                gl.bind_vertex_array(self.line_vao);
                gl.bind_buffer(glow::ARRAY_BUFFER, self.line_vbo);
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    slice_as_u8(&scene.lines),
                    glow::DYNAMIC_DRAW,
                );
                gl.draw_arrays(glow::LINES, 0, scene.lines.len() as i32);
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(
                viewport.left_px,
                viewport.from_bottom_px,
                viewport.width_px,
                viewport.height_px,
            );
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(
                clip.left_px,
                clip.from_bottom_px,
                clip.width_px,
                clip.height_px,
            );
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.use_program(Some(present_program));
            if let Some(location) = self.u_present_texture.as_ref() {
                gl.uniform_1_i32(Some(location), 0);
            }
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(offscreen_color));
            gl.bind_vertex_array(self.present_vao);
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.use_program(None);
            gl.disable(glow::SCISSOR_TEST);
            gl.enable(glow::BLEND);
        }

        self.last_error = None;
        Ok(())
    }
}

#[cfg(feature = "web_glow_viewport")]
fn append_grid_lines(lines: &mut Vec<GlVertex>) {
    let color = [0.22, 0.28, 0.34, 1.0];
    for i in -16..=16 {
        let offset = i as f32 * 0.25;
        push_line(lines, [-4.0, offset, 0.0], [4.0, offset, 0.0], color);
        push_line(lines, [offset, -4.0, 0.0], [offset, 4.0, 0.0], color);
    }
}

#[cfg(feature = "web_glow_viewport")]
fn append_line_geom(
    lines: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
    diagnostic_colors: bool,
) {
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

#[cfg(feature = "web_glow_viewport")]
fn append_box_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
    diagnostic_colors: bool,
) {
    let axes = geom_axes(geom);
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
}

#[cfg(feature = "web_glow_viewport")]
fn append_cylinder_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
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

#[cfg(feature = "web_glow_viewport")]
fn append_capsule_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
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

#[cfg(feature = "web_glow_viewport")]
fn append_sphere_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
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

#[cfg(feature = "web_glow_viewport")]
fn append_mesh_geom(
    triangles: &mut Vec<GlVertex>,
    geom: &BrowserGeomSnapshot,
    mesh_assets: &[BrowserMeshAsset],
    diagnostic_colors: bool,
) {
    let mesh_id = geom.dataid.max(0) as usize;
    let Some(mesh) = mesh_assets.get(mesh_id) else {
        return;
    };
    let color = display_geom_color(geom, diagnostic_colors);
    for vertex in &mesh.triangles {
        triangles.push(GlVertex::world(
            transform_geom_point(geom, vertex.position),
            transform_geom_vector(geom, vertex.normal),
            color,
        ));
    }
}

#[cfg(feature = "web_glow_viewport")]
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

#[cfg(feature = "web_glow_viewport")]
fn push_line(lines: &mut Vec<GlVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    let normal = [0.0, 0.0, 1.0];
    lines.push(GlVertex::world(a, normal, color));
    lines.push(GlVertex::world(b, normal, color));
}

fn geom_color(geom: &BrowserGeomSnapshot) -> [f32; 4] {
    [geom.rgba[0], geom.rgba[1], geom.rgba[2], 1.0]
}

fn display_geom_color(geom: &BrowserGeomSnapshot, diagnostic_colors: bool) -> [f32; 4] {
    if diagnostic_colors {
        diagnostic_geom_color(geom)
    } else {
        geom_color(geom)
    }
}

fn diagnostic_geom_color(geom: &BrowserGeomSnapshot) -> [f32; 4] {
    let seed = (geom.dataid as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((geom.type_id as u32).wrapping_mul(0x85EB_CA6B));
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

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize3(cross3(sub3(b, a), sub3(c, a)))
}

fn transform_geom_point(geom: &BrowserGeomSnapshot, local: [f32; 3]) -> [f32; 3] {
    [
        geom.pos[0] + geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.pos[1] + geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.pos[2] + geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ]
}

fn transform_geom_vector(geom: &BrowserGeomSnapshot, local: [f32; 3]) -> [f32; 3] {
    normalize3([
        geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ])
}

fn geom_axes(geom: &BrowserGeomSnapshot) -> [[f32; 3]; 3] {
    [
        [geom.mat[0], geom.mat[3], geom.mat[6]],
        [geom.mat[1], geom.mat[4], geom.mat[7]],
        [geom.mat[2], geom.mat[5], geom.mat[8]],
    ]
}

fn orbit_forward(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    [
        elevation.cos() * azimuth.cos(),
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
    ]
}

fn mul_mat4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum::<f32>();
        }
    }
    out
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

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
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

#[cfg(feature = "web_glow_viewport")]
fn create_gl_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|err| format!("Failed to create GL program: {err}"))?;
        let shaders = [
            (glow::VERTEX_SHADER, vertex_src),
            (glow::FRAGMENT_SHADER, fragment_src),
        ];
        let mut compiled = Vec::new();

        for (shader_type, source) in shaders {
            let shader = gl
                .create_shader(shader_type)
                .map_err(|err| format!("Failed to create shader: {err}"))?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                gl.delete_program(program);
                return Err(format!("GL shader compilation failed: {log}"));
            }
            gl.attach_shader(program, shader);
            compiled.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            for shader in compiled {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            gl.delete_program(program);
            return Err(format!("GL program link failed: {log}"));
        }

        for shader in compiled {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        Ok(program)
    }
}

#[cfg(feature = "web_glow_viewport")]
fn setup_vertex_array(
    gl: &glow::Context,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
) -> Result<(), String> {
    unsafe {
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let stride = std::mem::size_of::<GlVertex>() as i32;

        let position = gl
            .get_attrib_location(program, "a_pos")
            .ok_or_else(|| "Glow shader is missing a_pos attribute".to_string())?;
        let normal = gl
            .get_attrib_location(program, "a_normal")
            .ok_or_else(|| "Glow shader is missing a_normal attribute".to_string())?;
        let color = gl
            .get_attrib_location(program, "a_color")
            .ok_or_else(|| "Glow shader is missing a_color attribute".to_string())?;

        gl.enable_vertex_attrib_array(position);
        gl.vertex_attrib_pointer_f32(position, 3, glow::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(normal);
        gl.vertex_attrib_pointer_f32(normal, 3, glow::FLOAT, false, stride, 12);
        gl.enable_vertex_attrib_array(color);
        gl.vertex_attrib_pointer_f32(color, 4, glow::FLOAT, false, stride, 24);

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    Ok(())
}

#[cfg(feature = "web_glow_viewport")]
fn setup_present_quad(
    gl: &glow::Context,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
) -> Result<(), String> {
    const QUAD: [f32; 16] = [
        -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    unsafe {
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, slice_as_u8(&QUAD), glow::STATIC_DRAW);

        let position = gl
            .get_attrib_location(program, "a_pos")
            .ok_or_else(|| "Present shader is missing a_pos attribute".to_string())?;
        let uv = gl
            .get_attrib_location(program, "a_uv")
            .ok_or_else(|| "Present shader is missing a_uv attribute".to_string())?;

        gl.enable_vertex_attrib_array(position);
        gl.vertex_attrib_pointer_f32(position, 2, glow::FLOAT, false, 16, 0);
        gl.enable_vertex_attrib_array(uv);
        gl.vertex_attrib_pointer_f32(uv, 2, glow::FLOAT, false, 16, 8);

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    Ok(())
}

#[cfg(feature = "web_glow_viewport")]
fn slice_as_u8<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(feature = "web_glow_viewport")]
const GLOW_VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;
uniform mat4 u_view_proj;
in vec3 a_pos;
in vec3 a_normal;
in vec4 a_color;
out vec3 v_normal;
out vec4 v_color;

void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
    v_normal = a_normal;
    v_color = a_color;
}
"#;

#[cfg(feature = "web_glow_viewport")]
const GLOW_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
in vec3 v_normal;
in vec4 v_color;
uniform int u_unlit;
out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(0.35, 0.45, 0.82));
    float lit = (u_unlit != 0) ? 1.0 : (0.22 + 0.78 * max(dot(n, light_dir), 0.0));
    out_color = vec4(v_color.rgb * lit, v_color.a);
}
"#;

#[cfg(feature = "web_glow_viewport")]
const PRESENT_VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

#[cfg(feature = "web_glow_viewport")]
const PRESENT_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = texture(u_texture, v_uv);
}
"#;
