#[cfg(not(target_arch = "wasm32"))]
use egui::emath::GuiRounding;
use egui::{
    pos2, vec2, Align2, Color32, ColorImage, FontId, Painter, PointerButton, Pos2, Rect, Sense,
    Shape, Slider, Stroke, TextEdit, TextureHandle, TextureOptions, Ui,
};
#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Ordering;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
use eframe::glow::HasContext;
#[cfg(not(target_arch = "wasm32"))]
use eframe::{egui_glow, glow};

#[cfg(not(target_arch = "wasm32"))]
use std::{
    ffi::{CStr, CString},
    mem::ManuallyDrop,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

#[cfg(not(target_arch = "wasm32"))]
use ort::{
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
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
const MJ_OBJ_MESH: i32 = mjtObj__mjOBJ_MESH as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_FONTSCALE_100: i32 = mjtFontScale__mjFONTSCALE_100 as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_FB_OFFSCREEN: i32 = mjtFramebuffer__mjFB_OFFSCREEN as i32;
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
const MJ_GEOM_PLANE: i32 = mjtGeom__mjGEOM_PLANE as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_MESH: i32 = mjtGeom__mjGEOM_MESH as i32;
#[cfg(not(target_arch = "wasm32"))]
const MJ_GEOM_LINE: i32 = mjtGeom__mjGEOM_LINE as i32;

pub struct MujocoPanel {
    #[cfg(target_arch = "wasm32")]
    demo_url: String,
    #[cfg(not(target_arch = "wasm32"))]
    scene_path: String,
    #[cfg(not(target_arch = "wasm32"))]
    policy_path: String,
    #[cfg(not(target_arch = "wasm32"))]
    command_vel_x: f32,
    #[cfg(not(target_arch = "wasm32"))]
    diagnostic_colors: bool,
    #[cfg(not(target_arch = "wasm32"))]
    mesh_filter: i32,
    active: bool,
    status: String,
    #[cfg(not(target_arch = "wasm32"))]
    init_error: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    render_error: Option<String>,
    #[cfg(target_arch = "wasm32")]
    iframe_id: String,
    #[cfg(not(target_arch = "wasm32"))]
    runtime: Option<MujocoRuntime>,
}

impl Default for MujocoPanel {
    fn default() -> Self {
        Self {
            #[cfg(target_arch = "wasm32")]
            demo_url: "http://127.0.0.1:3000/demo".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            scene_path: local_asset_path("scene.xml").to_string_lossy().into_owned(),
            #[cfg(not(target_arch = "wasm32"))]
            policy_path: local_asset_path("robust.json")
                .to_string_lossy()
                .into_owned(),
            #[cfg(not(target_arch = "wasm32"))]
            command_vel_x: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            diagnostic_colors: true,
            #[cfg(not(target_arch = "wasm32"))]
            mesh_filter: -1,
            active: false,
            status: "MuJoCo runtime idle".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            init_error: None,
            #[cfg(not(target_arch = "wasm32"))]
            render_error: None,
            #[cfg(target_arch = "wasm32")]
            iframe_id: "rust-robotics-mujoco-iframe".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            runtime: None,
        }
    }
}

impl MujocoPanel {
    pub fn update(&mut self, sim_speed: usize, paused: bool) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (sim_speed, paused);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
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
                self.status = format!(
                    "MuJoCo running from {}",
                    Path::new(&self.scene_path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("scene.xml")
                );
            }
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        #[cfg(target_arch = "wasm32")]
        {
            self.ui_wasm(ui);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ui_native(ui, frame);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_split(
        &mut self,
        controls_ui: &mut Ui,
        viewport_ui: &mut Ui,
        frame: Option<&eframe::Frame>,
    ) {
        self.ui_native_controls(controls_ui);
        self.ui_native_viewport(viewport_ui, frame);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_controls(&mut self, ui: &mut Ui) {
        self.ui_native_controls(ui);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.ui_native_viewport(ui, frame);
    }

    pub fn set_active(&mut self, active: bool) {
        if self.active == active {
            return;
        }

        self.active = active;

        #[cfg(target_arch = "wasm32")]
        {
            self.sync_overlay();
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn ui_wasm(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.heading("mujoco");
            ui.label("Host the FACET MuJoCo demo here while keeping the Rust app shell intact.");

            ui.horizontal(|ui| {
                ui.label("Demo URL:");
                let changed = ui
                    .add(TextEdit::singleline(&mut self.demo_url).desired_width(320.0))
                    .changed();

                if ui.button("Reload").clicked() || changed {
                    self.sync_overlay();
                }
            });

            ui.horizontal(|ui| {
                let show_label = if self.active {
                    "Hide overlay"
                } else {
                    "Show overlay"
                };
                if ui.button(show_label).clicked() {
                    self.set_active(!self.active);
                }

                if ui.button("Reset status").clicked() {
                    self.status = "MuJoCo overlay idle".to_string();
                }
            });

            ui.label(format!("Status: {}", self.status));
            ui.label("The current implementation uses a browser overlay for MuJoCo rendering.");
            ui.label("This keeps the Rust side focused on app state and controls.");
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ui_native(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        ui.vertical(|ui| {
            self.ui_native_controls(ui);
            ui.separator();
            self.ui_native_viewport(ui, frame);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ui_native_controls(&mut self, ui: &mut Ui) {
        ui.heading("mujoco");
        ui.label("Native MuJoCo running inside RustRobotics.");

        ui.horizontal(|ui| {
            ui.label("Scene:");
            let changed = ui
                .add(TextEdit::singleline(&mut self.scene_path).desired_width(360.0))
                .changed();

            if ui.button("Reload").clicked() || changed {
                self.runtime = None;
                self.init_error = None;
                self.render_error = None;
                self.status = "Reloading MuJoCo scene...".to_string();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Policy:");
            let changed = ui
                .add(TextEdit::singleline(&mut self.policy_path).desired_width(360.0))
                .changed();

            if ui.button("Reload policy").clicked() || changed {
                self.runtime = None;
                self.init_error = None;
                self.render_error = None;
                self.status = "Reloading MuJoCo policy...".to_string();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Command vx:");
            ui.add(Slider::new(&mut self.command_vel_x, -2.0..=2.0).show_value(true));
            ui.checkbox(&mut self.diagnostic_colors, "Diagnostic colors");
        });

        ui.horizontal(|ui| {
            if ui.button("Reset sim").clicked() {
                if let Some(runtime) = self.runtime.as_mut() {
                    if let Err(err) = runtime.reset() {
                        self.status = err;
                    } else {
                        self.init_error = None;
                        self.render_error = None;
                        self.status = "MuJoCo reset".to_string();
                    }
                } else {
                    self.status = "MuJoCo runtime not loaded yet".to_string();
                }
            }

            if ui.button("Reset view").clicked() {
                if let Some(runtime) = self.runtime.as_mut() {
                    runtime.reset_camera();
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Mesh filter:");
            ui.add(Slider::new(&mut self.mesh_filter, -1..=15).show_value(true));
            if self.mesh_filter < 0 {
                ui.label("all mesh assets");
            } else if let Some(runtime) = self.runtime.as_ref() {
                ui.label(runtime.mesh_filter_label(self.mesh_filter));
            }
        });

        ui.label(format!("Status: {}", self.status));

        if let Some(runtime) = self.runtime.as_ref() {
            ui.label(format!("Load: {}", runtime.diagnostics.load_summary));
            ui.label(format!(
                "Perf: step {:.2} ms | policy {:.2} ms | render {:.2} ms | gl-init {:.2} ms | frames {}",
                runtime.diagnostics.last_step_ms,
                runtime.diagnostics.last_policy_ms,
                runtime.diagnostics.last_render_ms,
                runtime.diagnostics.last_gl_init_ms,
                runtime.diagnostics.frame_count
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ui_native_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        let mut desired = ui.available_size_before_wrap();
        desired.x = desired.x.max(320.0);
        desired.y = desired.y.max(240.0);
        if let Some(err) = self.render_error.clone() {
            let diagnostic_colors = self.diagnostic_colors;
            let mesh_filter = normalize_mesh_filter(self.mesh_filter);
            if let Ok(runtime) = self.ensure_runtime() {
                runtime.render_software(ui, frame, desired, diagnostic_colors, mesh_filter);
            }
            return;
        }
        let render_result = if let Some(err) = &self.render_error {
            Err(err.clone())
        } else {
            (|| -> Result<Option<egui::TextureId>, String> {
                let runtime = self.ensure_runtime()?;
                let Some(frame) = frame else {
                    return Err("Native renderer needs access to the eframe frame.".to_string());
                };

                let Some(gl) = frame.gl() else {
                    return Err(
                        "OpenGL context unavailable; MuJoCo viewport is disabled.".to_string()
                    );
                };

                runtime.ensure_gl(gl)?;
                runtime.render(gl, ui.ctx(), desired)?;
                Ok(runtime.texture.as_ref().map(|texture| texture.id()))
            })()
        };

        match render_result {
            Ok(Some(texture_id)) => {
                self.render_error = None;
                ui.image((texture_id, desired));
            }
            Ok(None) => {
                self.render_error = None;
                ui.label("MuJoCo texture is not ready yet.");
            }
            Err(err) => {
                debug_log(&format!("ui_native: render error: {err}"));
                self.render_error = Some(err.clone());
                self.status = "MuJoCo running with glow 3D fallback renderer".to_string();
                let diagnostic_colors = self.diagnostic_colors;
                let mesh_filter = normalize_mesh_filter(self.mesh_filter);
                if let Ok(runtime) = self.ensure_runtime() {
                    runtime.render_software(ui, frame, desired, diagnostic_colors, mesh_filter);
                }
            }
        }

        let _ = frame;
    }

    #[cfg(target_arch = "wasm32")]
    fn sync_overlay(&mut self) {
        use wasm_bindgen::JsCast;

        let Some(window) = web_sys::window() else {
            self.status = "No browser window available.".to_string();
            return;
        };

        let Some(document) = window.document() else {
            self.status = "No document available.".to_string();
            return;
        };

        if !self.active {
            if let Some(element) = document.get_element_by_id(&self.iframe_id) {
                element.remove();
            }
            self.status = "MuJoCo overlay hidden".to_string();
            return;
        }

        let iframe = match document.get_element_by_id(&self.iframe_id) {
            Some(existing) => match existing.dyn_into::<web_sys::HtmlIFrameElement>() {
                Ok(frame) => frame,
                Err(_) => {
                    self.status = "Existing overlay element is not an iframe.".to_string();
                    return;
                }
            },
            None => {
                let Some(body) = document.body() else {
                    self.status = "No document body available.".to_string();
                    return;
                };

                let Ok(element) = document.create_element("iframe") else {
                    self.status = "Failed to create iframe element.".to_string();
                    return;
                };

                let Ok(frame) = element.dyn_into::<web_sys::HtmlIFrameElement>() else {
                    self.status = "Created overlay element is not an iframe.".to_string();
                    return;
                };

                frame.set_id(&self.iframe_id);
                if let Err(err) = frame.set_attribute("title", "MuJoCo demo") {
                    self.status = format!("Failed to set iframe title: {:?}", err);
                    return;
                }
                if let Err(err) = frame.set_attribute("allow", "fullscreen") {
                    self.status = format!("Failed to set iframe permissions: {:?}", err);
                    return;
                }
                if let Err(err) = frame.set_attribute(
                    "style",
                    "position: fixed; top: 72px; right: 16px; width: min(52vw, 920px); height: calc(100vh - 88px); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; background: #111; z-index: 9998;",
                ) {
                    self.status = format!("Failed to style iframe: {:?}", err);
                    return;
                }

                if let Err(err) = body.append_child(&frame) {
                    self.status = format!("Failed to attach iframe: {:?}", err);
                    return;
                }

                frame
            }
        };

        iframe.set_src(&self.demo_url);
        self.status = format!("MuJoCo overlay attached: {}", self.demo_url);
    }

    #[cfg(not(target_arch = "wasm32"))]
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
impl MujocoPanel {
    fn _unused_wasm_companion(&mut self) {}
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct PolicyFile {
    onnx: OnnxFile,
    action_scale: f32,
    stiffness: f32,
    damping: f32,
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
#[derive(Debug, Deserialize)]
struct AssetMeta {
    joint_names_isaac: Vec<String>,
    default_joint_pos: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
struct MujocoPolicy {
    session: Session,
    action_output_name: String,
    next_hx_output_name: String,
}

#[cfg(not(target_arch = "wasm32"))]
static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static GLOW_SCENE_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static GLOW_CALLBACK_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static MESH_DIAGNOSTICS_LOGGED: OnceLock<()> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
impl MujocoPolicy {
    fn load(path: &Path, meta: &OnnxMeta) -> Result<Self, String> {
        debug_log(&format!("policy.load: begin {}", path.display()));
        let ort_lib = find_onnxruntime_library().ok_or_else(|| {
            "Failed to locate libonnxruntime.dylib; set ORT_DYLIB_PATH or install ONNX Runtime"
                .to_string()
        })?;
        debug_log(&format!("policy.load: ort dylib {}", ort_lib.display()));

        ensure_ort_initialized(&ort_lib)?;

        debug_log("policy.load: Session::builder start");
        let mut builder = Session::builder()
            .map_err(|err| format!("Failed to create ONNX Runtime session builder: {err}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .and_then(|builder| builder.with_log_level(LogLevel::Fatal))
            .map_err(|err| format!("Failed to configure ONNX Runtime session: {err}"))?;
        debug_log("policy.load: Session::builder configured");

        debug_log(&format!(
            "policy.load: commit_from_file start {}",
            path.display()
        ));
        let session = {
            let _stderr_silencer = StderrSilencer::new();
            builder
                .commit_from_file(path)
                .map_err(|err| format!("Failed to load ONNX model {}: {err}", path.display()))?
        };
        debug_log("policy.load: session ready");

        let output_name_map = meta
            .out_keys
            .iter()
            .zip(session.outputs().iter())
            .map(|(semantic_key, outlet)| (semantic_key.joined(), outlet.name().to_string()))
            .collect::<std::collections::HashMap<_, _>>();

        let action_output_name = output_name_map
            .get("action")
            .cloned()
            .ok_or_else(|| "ONNX metadata missing output mapping for action".to_string())?;
        let next_hx_output_name = output_name_map
            .get("next.adapt_hx")
            .cloned()
            .ok_or_else(|| "ONNX metadata missing output mapping for next.adapt_hx".to_string())?;

        debug_log(&format!(
            "policy.load: output mapping action={} next_hx={}",
            action_output_name, next_hx_output_name
        ));

        Ok(Self {
            session,
            action_output_name,
            next_hx_output_name,
        })
    }

    fn run(
        &mut self,
        policy: &[f32; 117],
        is_init: bool,
        adapt_hx: &[f32; 128],
        command: &[f32; 16],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let first_inference = is_init;
        debug_log_if(first_inference, "policy.run: first inference start");
        let policy = Tensor::from_array(([1usize, 117], policy.to_vec().into_boxed_slice()))
            .map_err(|err| format!("Failed to build policy tensor: {err}"))?;
        let is_init = Tensor::from_array(([1usize], vec![is_init].into_boxed_slice()))
            .map_err(|err| format!("Failed to build init tensor: {err}"))?;
        let adapt_hx = Tensor::from_array(([1usize, 128], adapt_hx.to_vec().into_boxed_slice()))
            .map_err(|err| format!("Failed to build recurrent tensor: {err}"))?;
        let command = Tensor::from_array(([1usize, 16], command.to_vec().into_boxed_slice()))
            .map_err(|err| format!("Failed to build command tensor: {err}"))?;

        let outputs = self
            .session
            .run(ort::inputs![policy, is_init, adapt_hx, command])
            .map_err(|err| format!("ONNX inference failed: {err}"))?;
        debug_log_if(first_inference, "policy.run: first inference done");

        let action = outputs[self.action_output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|err| format!("Failed to read action tensor: {err}"))?
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let next_hx = outputs[self.next_hx_output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|err| format!("Failed to read recurrent tensor: {err}"))?
            .iter()
            .copied()
            .collect::<Vec<_>>();

        Ok((action, next_hx))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_ort_initialized(ort_lib: &Path) -> Result<(), String> {
    ORT_INIT_RESULT
        .get_or_init(|| {
            debug_log("policy.load: ort manual init start");
            let library = unsafe { libloading::Library::new(ort_lib) }.map_err(|err| {
                format!(
                    "Failed to load ONNX Runtime dylib {}: {err}",
                    ort_lib.display()
                )
            })?;
            debug_log("policy.load: ort manual library open done");

            let get_api_base: libloading::Symbol<
                '_,
                unsafe extern "C" fn() -> *const ort::sys::OrtApiBase,
            > = unsafe { library.get(b"OrtGetApiBase") }.map_err(|err| {
                format!(
                    "Failed to locate OrtGetApiBase in {}: {err}",
                    ort_lib.display()
                )
            })?;
            debug_log("policy.load: ort manual symbol lookup done");

            let api_base = unsafe { get_api_base() };
            if api_base.is_null() {
                return Err(format!(
                    "OrtGetApiBase returned null for {}",
                    ort_lib.display()
                ));
            }

            let version = unsafe { CStr::from_ptr(((*api_base).GetVersionString)()) }
                .to_string_lossy()
                .into_owned();
            debug_log(&format!("policy.load: ort version {version}"));

            let api_ptr = unsafe { ((*api_base).GetApi)(ort::sys::ORT_API_VERSION) };
            if api_ptr.is_null() {
                return Err(format!(
                    "OrtGetApi({}) returned null for {}",
                    ort::sys::ORT_API_VERSION,
                    ort_lib.display()
                ));
            }

            let api = unsafe { std::ptr::read(api_ptr) };
            let _library = ManuallyDrop::new(library);
            let inserted = ort::set_api(api);
            debug_log(&format!("policy.load: ort set_api inserted={inserted}"));

            let committed = ort::init().commit();
            debug_log(&format!(
                "policy.load: ort env commit committed={committed}"
            ));
            Ok(())
        })
        .clone()
}

#[cfg(not(target_arch = "wasm32"))]
fn find_onnxruntime_library() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    let candidates = [
        "/usr/local/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.1.20.2.dylib",
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/opt/homebrew/lib/libonnxruntime.1.20.2.dylib",
    ];

    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

#[cfg(not(target_arch = "wasm32"))]
struct MujocoRuntime {
    model: *mut mjModel,
    data: *mut mjData,
    scene: mjvScene,
    cam: mjvCamera,
    opt: mjvOption,
    con: mjrContext,
    gl_ready: bool,
    texture: Option<TextureHandle>,
    policy: MujocoPolicy,
    joint_qpos_adr: [usize; 12],
    joint_qvel_adr: [usize; 12],
    ctrl_adr: [usize; 12],
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
    command_vel_x: f32,
    timestep: f32,
    decimation: usize,
    mujoco_time_ms: f32,
    diagnostics: MujocoDiagnostics,
    glow_renderer: Arc<Mutex<GlowSceneRenderer>>,
    glow_scene: Arc<Mutex<GlowSceneFrame>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct MujocoDiagnostics {
    load_summary: String,
    last_step_ms: f32,
    last_policy_ms: f32,
    last_render_ms: f32,
    last_gl_init_ms: f32,
    frame_count: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for MujocoRuntime {
    fn drop(&mut self) {
        unsafe {
            if self.gl_ready {
                mjr_freeContext(&mut self.con);
            }
            mjv_freeScene(&mut self.scene);
            if !self.data.is_null() {
                mj_deleteData(self.data);
            }
            if !self.model.is_null() {
                mj_deleteModel(self.model);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MujocoRuntime {
    fn load(scene_path: &Path, policy_path: &Path) -> Result<Self, String> {
        let load_started = Instant::now();
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

        if asset_meta.joint_names_isaac.len() != 12 {
            return Err(format!(
                "Expected 12 joints in asset metadata, found {}",
                asset_meta.joint_names_isaac.len()
            ));
        }
        if asset_meta.default_joint_pos.len() != 12 {
            return Err(format!(
                "Expected 12 default joint positions in asset metadata, found {}",
                asset_meta.default_joint_pos.len()
            ));
        }

        let model_started = Instant::now();
        debug_log("runtime.load: mj_loadXML start");
        let model = unsafe {
            let scene_c = CString::new(scene_path.as_os_str().to_string_lossy().as_bytes())
                .map_err(|err| format!("Invalid scene path {}: {err}", scene_path.display()))?;
            let mut error = [0i8; 1024];
            let model = mj_loadXML(
                scene_c.as_ptr(),
                std::ptr::null(),
                error.as_mut_ptr(),
                error.len() as i32,
            );
            if model.is_null() {
                let message = cstring_buffer_to_string(&error);
                return Err(format!(
                    "Failed to load MuJoCo scene {}: {}",
                    scene_path.display(),
                    message
                ));
            }
            model
        };
        let model_elapsed = model_started.elapsed();
        debug_log(&format!(
            "runtime.load: mj_loadXML done in {}",
            fmt_duration(model_elapsed)
        ));

        debug_log("runtime.load: mj_makeData start");
        let data = unsafe {
            let data = mj_makeData(model);
            if data.is_null() {
                mj_deleteModel(model);
                return Err(format!(
                    "Failed to allocate MuJoCo data for {}",
                    scene_path.display()
                ));
            }
            data
        };
        debug_log("runtime.load: mj_makeData done");

        debug_log("runtime.load: initial mj_forward start");
        unsafe {
            if (*model).nkey > 0 {
                mj_resetDataKeyframe(model, data, 0);
            } else {
                mj_resetData(model, data);
            }
            mj_forward(model, data);
        }
        debug_log("runtime.load: initial mj_forward done");

        let mut joint_qpos_adr = [0usize; 12];
        let mut joint_qvel_adr = [0usize; 12];
        let mut ctrl_adr = [0usize; 12];

        for (i, joint_name) in asset_meta.joint_names_isaac.iter().enumerate() {
            let joint_id = unsafe { name2id(model, MJ_OBJ_JOINT, joint_name)? };
            let actuator_name = joint_name.strip_suffix("_joint").unwrap_or(joint_name);
            let actuator_id = unsafe { name2id(model, MJ_OBJ_ACTUATOR, actuator_name)? };

            unsafe {
                joint_qpos_adr[i] = *(*model).jnt_qposadr.add(joint_id as usize) as usize;
                joint_qvel_adr[i] = *(*model).jnt_dofadr.add(joint_id as usize) as usize;
            }
            ctrl_adr[i] = actuator_id as usize;
        }

        let timestep = unsafe { (*model).opt.timestep as f32 };
        let decimation = ((0.02 / timestep).round() as usize).max(1);
        let mut scene = unsafe { std::mem::zeroed::<mjvScene>() };
        let mut cam = unsafe { std::mem::zeroed::<mjvCamera>() };
        let mut opt = unsafe { std::mem::zeroed::<mjvOption>() };
        let mut con = unsafe { std::mem::zeroed::<mjrContext>() };

        unsafe {
            mjv_defaultScene(&mut scene);
            mjv_makeScene(model, &mut scene, 3000);
            mjv_defaultCamera(&mut cam);
            mjv_defaultFreeCamera(model, &mut cam);
            mjv_defaultOption(&mut opt);
            mjr_defaultContext(&mut con);
        }
        fit_camera_to_model_stat(model, &mut cam);

        let mut default_jpos = [0.0f32; 12];
        default_jpos.copy_from_slice(&asset_meta.default_joint_pos);

        let action_scale = [policy_file.action_scale; 12];
        let kp = [policy_file.stiffness; 12];
        let kd = [policy_file.damping; 12];

        let policy_started = Instant::now();
        debug_log("runtime.load: policy load start");
        let policy = MujocoPolicy::load(
            &scene_path
                .parent()
                .ok_or_else(|| {
                    format!(
                        "Scene path has no parent directory: {}",
                        scene_path.display()
                    )
                })?
                .join(&policy_file.onnx.path),
            &policy_file.onnx.meta,
        )?;
        let policy_elapsed = policy_started.elapsed();
        debug_log(&format!(
            "runtime.load: policy load done in {}",
            fmt_duration(policy_elapsed)
        ));

        Ok(Self {
            model,
            data,
            scene,
            cam,
            opt,
            con,
            gl_ready: false,
            texture: None,
            policy,
            joint_qpos_adr,
            joint_qvel_adr,
            ctrl_adr,
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
            command_vel_x: 0.0,
            timestep,
            decimation,
            mujoco_time_ms: 0.0,
            diagnostics: MujocoDiagnostics {
                load_summary: format!(
                    "total {} | model {} | policy {}",
                    fmt_duration(load_started.elapsed()),
                    fmt_duration(model_elapsed),
                    fmt_duration(policy_elapsed)
                ),
                ..Default::default()
            },
            glow_renderer: Arc::new(Mutex::new(GlowSceneRenderer::default())),
            glow_scene: Arc::new(Mutex::new(GlowSceneFrame::default())),
        })
    }

    fn ensure_gl(&mut self, gl: &Arc<glow::Context>) -> Result<(), String> {
        if self.gl_ready {
            return Ok(());
        }

        let started = Instant::now();
        let version = gl.version().clone();
        let extensions = gl.supported_extensions();
        let has_arb_fbo = extensions.contains("GL_ARB_framebuffer_object");
        let has_ext_fbo = extensions.contains("GL_EXT_framebuffer_object");
        debug_log(&format!(
            "runtime.ensure_gl: context version={:?} ARB_fbo={} EXT_fbo={}",
            version, has_arb_fbo, has_ext_fbo
        ));
        if version.is_embedded || !(has_arb_fbo || has_ext_fbo) {
            let err = format!(
                "MuJoCo native rendering is not available on this OpenGL context. MuJoCo requires ARB/EXT framebuffer-object support and the current context is {:?} (ARB_fbo={}, EXT_fbo={}). The simulation is still running; only native MuJoCo rendering is unavailable in this eframe/glow context.",
                version, has_arb_fbo, has_ext_fbo,
            );
            debug_log(&format!("runtime.ensure_gl: rejected context: {err}"));
            return Err(err);
        }
        debug_log("runtime.ensure_gl: mjr_makeContext start");
        unsafe {
            mjr_makeContext(self.model, &mut self.con, MJ_FONTSCALE_100);
        }
        self.gl_ready = true;
        self.diagnostics.last_gl_init_ms = started.elapsed().as_secs_f32() * 1000.0;
        debug_log(&format!(
            "runtime.ensure_gl: mjr_makeContext done in {}",
            fmt_duration(started.elapsed())
        ));
        Ok(())
    }

    fn reset(&mut self) -> Result<(), String> {
        unsafe {
            if (*self.model).nkey > 0 {
                mj_resetDataKeyframe(self.model, self.data, 0);
            } else {
                mj_resetData(self.model, self.data);
            }
            mj_forward(self.model, self.data);
        }

        self.last_actions = [0.0; 12];
        self.action_history = [[0.0; 12]; 3];
        self.gravity_history = [[0.0, 0.0, -1.0]; 3];
        self.joint_pos_history = [[0.0; 12]; 3];
        self.joint_vel_history = [[0.0; 12]; 3];
        self.adapt_hx = [0.0; 128];
        self.is_init = true;
        self.mujoco_time_ms = 0.0;
        Ok(())
    }

    fn reset_camera(&mut self) {
        unsafe {
            mjv_defaultCamera(&mut self.cam);
            mjv_defaultFreeCamera(self.model, &mut self.cam);
        }
        fit_camera_to_model_stat(self.model, &mut self.cam);
    }

    fn step(&mut self, sim_speed: usize, command_vel_x: f32) -> Result<(), String> {
        let started = Instant::now();
        self.command_vel_x = command_vel_x;
        for _ in 0..sim_speed {
            self.update_policy_input()?;
            self.run_policy()?;

            for _ in 0..self.decimation {
                self.apply_control();
                unsafe {
                    mj_step(self.model, self.data);
                }
                self.mujoco_time_ms += self.timestep * 1000.0;
            }
        }
        self.diagnostics.last_step_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    fn update_policy_input(&mut self) -> Result<(), String> {
        let qpos =
            unsafe { std::slice::from_raw_parts((*self.data).qpos, (*self.model).nq as usize) };
        let qvel =
            unsafe { std::slice::from_raw_parts((*self.data).qvel, (*self.model).nv as usize) };

        let base_qpos = &qpos[0..7];
        let quat = [
            base_qpos[3] as f32,
            base_qpos[4] as f32,
            base_qpos[5] as f32,
            base_qpos[6] as f32,
        ];
        let gravity = rotate_vector_by_inverse_quaternion(quat, [0.0, 0.0, -1.0]);

        shift_history_3(&mut self.gravity_history, gravity);

        let mut joint_pos = [0.0f32; 12];
        let mut joint_vel = [0.0f32; 12];
        for i in 0..12 {
            joint_pos[i] = qpos[self.joint_qpos_adr[i]] as f32;
            joint_vel[i] = qvel[self.joint_qvel_adr[i]] as f32;
        }
        shift_history_12(&mut self.joint_pos_history, joint_pos);
        shift_history_12(&mut self.joint_vel_history, joint_vel);
        Ok(())
    }

    fn run_policy(&mut self) -> Result<(), String> {
        let started = Instant::now();
        let policy_input = self.flatten_policy_input();
        let command_input = self.compute_command_input();
        let (action, next_hx) =
            self.policy
                .run(&policy_input, self.is_init, &self.adapt_hx, &command_input)?;

        if action.len() < 12 {
            return Err(format!(
                "Policy returned {} actions, expected at least 12",
                action.len()
            ));
        }
        if next_hx.len() < 128 {
            return Err(format!(
                "Policy returned {} recurrent values, expected at least 128",
                next_hx.len()
            ));
        }

        let mut smoothed = [0.0f32; 12];
        for i in 0..12 {
            smoothed[i] = self.last_actions[i] * 0.2 + action[i] * 0.8;
        }

        self.last_actions = smoothed;
        shift_history_12(&mut self.action_history, self.last_actions);
        self.adapt_hx.copy_from_slice(&next_hx[0..128]);
        self.is_init = false;
        self.diagnostics.last_policy_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    fn apply_control(&mut self) {
        let qpos =
            unsafe { std::slice::from_raw_parts((*self.data).qpos, (*self.model).nq as usize) };
        let qvel =
            unsafe { std::slice::from_raw_parts((*self.data).qvel, (*self.model).nv as usize) };
        let ctrl =
            unsafe { std::slice::from_raw_parts_mut((*self.data).ctrl, (*self.model).nu as usize) };

        for i in 0..12 {
            let target = self.action_scale[i] * self.last_actions[i] + self.default_jpos[i];
            let torque = self.kp[i] * (target - qpos[self.joint_qpos_adr[i]] as f32)
                + self.kd[i] * (0.0 - qvel[self.joint_qvel_adr[i]] as f32);
            ctrl[self.ctrl_adr[i]] = torque as f64;
        }
    }

    fn flatten_policy_input(&self) -> [f32; 117] {
        let mut out = [0.0f32; 117];
        let mut idx = 0;

        for step in &self.gravity_history {
            for value in step {
                out[idx] = *value;
                idx += 1;
            }
        }
        for step in &self.joint_pos_history {
            for value in step {
                out[idx] = *value;
                idx += 1;
            }
        }
        for step in &self.joint_vel_history {
            for value in step {
                out[idx] = *value;
                idx += 1;
            }
        }
        for step in &self.action_history {
            for value in step {
                out[idx] = *value;
                idx += 1;
            }
        }

        out
    }

    fn compute_command_input(&self) -> [f32; 16] {
        let qpos =
            unsafe { std::slice::from_raw_parts((*self.data).qpos, (*self.model).nq as usize) };
        let quat = [
            qpos[3] as f32,
            qpos[4] as f32,
            qpos[5] as f32,
            qpos[6] as f32,
        ];
        let command = rotate_vector_by_inverse_quaternion(quat, [self.command_vel_x, 0.0, 0.0]);
        let yaw = quaternion_yaw(quat);
        let oscillator = oscillator(self.mujoco_time_ms / 1000.0);

        [
            command[0],
            command[1],
            -yaw,
            0.0,
            oscillator[0],
            oscillator[1],
            oscillator[2],
            oscillator[3],
            oscillator[4],
            oscillator[5],
            oscillator[6],
            oscillator[7],
            oscillator[8],
            oscillator[9],
            oscillator[10],
            oscillator[11],
        ]
    }

    fn render(
        &mut self,
        gl: &Arc<glow::Context>,
        ctx: &egui::Context,
        desired_size: egui::Vec2,
    ) -> Result<(), String> {
        let started = Instant::now();
        debug_log("runtime.render: begin");
        self.ensure_gl(gl)?;

        let width = desired_size.x.max(2.0).round() as i32;
        let height = desired_size.y.max(2.0).round() as i32;
        unsafe {
            mjr_resizeOffscreen(width, height, &mut self.con);
            mjr_setBuffer(MJ_FB_OFFSCREEN, &mut self.con);
        }

        let viewport = mjrRect {
            left: 0,
            bottom: 0,
            width,
            height,
        };

        unsafe {
            mjv_updateScene(
                self.model,
                self.data,
                &self.opt,
                std::ptr::null::<mjvPerturb>(),
                &mut self.cam,
                MJ_CAT_ALL,
                &mut self.scene,
            );
            mjr_render(viewport, &mut self.scene, &self.con);
        }

        let mut rgb = vec![0u8; (width as usize) * (height as usize) * 3];
        unsafe {
            mjr_readPixels(
                rgb.as_mut_ptr(),
                std::ptr::null_mut::<f32>(),
                viewport,
                &self.con,
            );
            mjr_restoreBuffer(&self.con);
        }

        let rgba = flip_rgb_to_rgba(&rgb, width as usize, height as usize);
        let image = ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba);

        match self.texture.as_mut() {
            Some(texture) => texture.set(image, TextureOptions::LINEAR),
            None => {
                self.texture =
                    Some(ctx.load_texture("mujoco-render", image, TextureOptions::LINEAR));
            }
        }

        self.diagnostics.last_render_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.diagnostics.frame_count += 1;
        debug_log(&format!(
            "runtime.render: done in {}",
            fmt_duration(started.elapsed())
        ));
        Ok(())
    }

    fn render_software(
        &mut self,
        ui: &mut Ui,
        frame: Option<&eframe::Frame>,
        desired_size: egui::Vec2,
        diagnostic_colors: bool,
        mesh_filter: Option<usize>,
    ) {
        let Some(frame) = frame else {
            self.render_software_2d(ui, desired_size);
            ui.label("Native glow frame unavailable; showing simplified 2D fallback.");
            return;
        };
        let Some(gl) = frame.gl() else {
            self.render_software_2d(ui, desired_size);
            ui.label("OpenGL context unavailable; showing simplified 2D fallback.");
            return;
        };

        if let Err(err) = self.ensure_glow_renderer(gl) {
            debug_log(&format!("runtime.render_glow: init error: {err}"));
            self.render_software_2d(ui, desired_size);
            ui.label(format!("Custom glow renderer init failed: {err}"));
            return;
        }

        if let Err(err) = self.render_glow_scene(ui, desired_size, diagnostic_colors, mesh_filter) {
            debug_log(&format!("runtime.render_glow: build error: {err}"));
            self.render_software_2d(ui, desired_size);
            ui.label(format!("Custom glow renderer failed: {err}"));
        }
    }

    fn render_glow_scene(
        &mut self,
        ui: &mut Ui,
        _desired_size: egui::Vec2,
        diagnostic_colors: bool,
        mesh_filter: Option<usize>,
    ) -> Result<(), String> {
        let started = Instant::now();
        let rect = aligned_viewport_rect(ui);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        self.update_software_camera(ui, &response);
        self.update_visual_scene();
        let frame = self.build_gl_scene_frame(
            (rect.width() / rect.height()).max(0.1),
            diagnostic_colors,
            mesh_filter,
        )?;
        {
            let mut scene = self
                .glow_scene
                .lock()
                .map_err(|_| "Failed to lock glow scene state".to_string())?;
            *scene = frame;
        }

        let renderer = self.glow_renderer.clone();
        let scene = self.glow_scene.clone();
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                let render_result = (|| -> Result<(), String> {
                    let scene = scene
                        .lock()
                        .map_err(|_| "Failed to lock glow scene frame".to_string())?;
                    let mut renderer = renderer
                        .lock()
                        .map_err(|_| "Failed to lock glow renderer".to_string())?;
                    renderer.paint(painter.gl(), info, &scene)
                })();

                if let Err(err) = render_result {
                    if let Ok(mut renderer) = renderer.lock() {
                        if renderer.last_error.as_deref() != Some(err.as_str()) {
                            debug_log(&format!("runtime.render_glow: callback error: {err}"));
                        }
                        renderer.last_error = Some(err);
                    }
                }
            })),
        });

        let overlay = self.software_overlay_text();
        ui.painter().text(
            rect.left_top() + vec2(12.0, 12.0),
            Align2::LEFT_TOP,
            &overlay,
            FontId::monospace(12.0),
            Color32::from_gray(210),
        );

        if let Ok(renderer) = self.glow_renderer.lock() {
            self.diagnostics.last_render_ms = renderer.last_render_ms_ms;
        } else {
            self.diagnostics.last_render_ms = started.elapsed().as_secs_f32() * 1000.0;
        }
        self.diagnostics.frame_count += 1;
        Ok(())
    }

    fn render_software_2d(&mut self, ui: &mut Ui, desired_size: egui::Vec2) {
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

    fn update_visual_scene(&mut self) {
        unsafe {
            mjv_updateScene(
                self.model,
                self.data,
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

        if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            self.cam.azimuth -= delta.x as f64 * 0.25;
            self.cam.elevation = (self.cam.elevation + delta.y as f64 * 0.2).clamp(-89.0, 89.0);
        }

        if response.dragged_by(PointerButton::Secondary) {
            let delta = response.drag_delta();
            let forward = orbit_forward(self.cam.azimuth as f32, self.cam.elevation as f32);
            let right = normalize3(cross3(forward, [0.0, 0.0, 1.0]));
            let up = normalize3(cross3(right, forward));
            let pan_scale = self.cam.distance as f32 * 0.0025;
            for axis in 0..3 {
                self.cam.lookat[axis] -=
                    (right[axis] * delta.x + up[axis] * delta.y) as f64 * pan_scale as f64;
            }
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
        let qpos =
            unsafe { std::slice::from_raw_parts((*self.data).qpos, (*self.model).nq as usize) };
        let qvel =
            unsafe { std::slice::from_raw_parts((*self.data).qvel, (*self.model).nv as usize) };
        let quat = [
            qpos[3] as f32,
            qpos[4] as f32,
            qpos[5] as f32,
            qpos[6] as f32,
        ];
        let yaw_deg = quaternion_yaw(quat).to_degrees();
        let planar_speed = (qvel[0] as f32).hypot(qvel[1] as f32);
        let (triangles, lines) = self
            .glow_scene
            .lock()
            .map(|scene| (scene.triangles.len() / 3, scene.lines.len() / 2))
            .unwrap_or((0, 0));

        format!(
            concat!(
                "Glow 3D fallback renderer\n",
                "Sim t={:.2}s  cmd_vx={:+.2}  speed={:.2}  z={:.2}  yaw={:+.1}deg\n",
                "Cam az={:+.1}  el={:+.1}  dist={:.2}  lookat=({:+.2}, {:+.2}, {:+.2})\n",
                "Step {:.2} ms  Policy {:.2} ms  Render {:.2} ms  Frames {}\n",
                "Glow tris {}  lines {}\n",
                "Drag: orbit | Right-drag: pan | Wheel: zoom | Double-click: reset view"
            ),
            self.mujoco_time_ms / 1000.0,
            self.command_vel_x,
            planar_speed,
            qpos[2] as f32,
            yaw_deg,
            self.cam.azimuth as f32,
            self.cam.elevation as f32,
            self.cam.distance as f32,
            self.cam.lookat[0] as f32,
            self.cam.lookat[1] as f32,
            self.cam.lookat[2] as f32,
            self.diagnostics.last_step_ms,
            self.diagnostics.last_policy_ms,
            self.diagnostics.last_render_ms,
            self.diagnostics.frame_count,
            triangles,
            lines,
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
    fn ensure_glow_renderer(&mut self, gl: &Arc<glow::Context>) -> Result<(), String> {
        let mut renderer = self
            .glow_renderer
            .lock()
            .map_err(|_| "Failed to lock glow renderer".to_string())?;
        renderer.ensure_initialized(gl)
    }

    fn build_gl_scene_frame(
        &self,
        aspect: f32,
        diagnostic_colors: bool,
        mesh_filter: Option<usize>,
    ) -> Result<GlowSceneFrame, String> {
        let camera = SoftwareCamera::from_gl(&self.scene.camera[0], aspect)
            .ok_or_else(|| "MuJoCo scene camera is not available for glow fallback".to_string())?;
        let mut frame = GlowSceneFrame::default();

        append_grid_lines(&mut frame.lines);

        let mut mesh_diagnostics = Vec::new();
        let mut mesh_assets = BTreeMap::<usize, MeshAssetStats>::new();
        let ngeom = unsafe { (*self.model).ngeom as usize };
        for geom_id in 0..ngeom {
            let Some(geom) = self.model_geom_instance(geom_id) else {
                continue;
            };

            match geom.type_ {
                MJ_GEOM_PLANE => {}
                MJ_GEOM_SPHERE => {
                    append_sphere_geom(&mut frame.triangles, &geom, 12, 20, diagnostic_colors)
                }
                MJ_GEOM_CAPSULE => {
                    append_capsule_geom(&mut frame.triangles, &geom, 10, 18, diagnostic_colors)
                }
                MJ_GEOM_CYLINDER => {
                    append_cylinder_geom(&mut frame.triangles, &geom, 18, diagnostic_colors)
                }
                MJ_GEOM_BOX => append_box_geom(&mut frame.triangles, &geom, diagnostic_colors),
                MJ_GEOM_MESH => {
                    if let Some(filter) = mesh_filter {
                        if self.resolve_mesh_id(&geom) != Some(filter) {
                            continue;
                        }
                    }
                    if MESH_DIAGNOSTICS_LOGGED.get().is_none() {
                        mesh_diagnostics.push(self.describe_mesh_geom(&geom));
                    }
                    if let Some(stats) =
                        self.append_mesh_geom(&mut frame.triangles, &geom, diagnostic_colors)
                    {
                        mesh_assets
                            .entry(stats.mesh_id)
                            .or_insert_with(|| MeshAssetStats::new(stats.mesh_id))
                            .record(&stats);
                    }
                }
                MJ_GEOM_LINE => append_line_geom(&mut frame.lines, &geom, diagnostic_colors),
                _ => {}
            }
        }

        frame.view_proj = camera.fitted_view_projection_matrix(&frame.triangles);

        if GLOW_SCENE_LOGGED.get().is_none() {
            let (min, max) = vertex_bounds(&frame.triangles);
            debug_log(&format!(
                "runtime.render_glow: scene built triangles={} lines={} world_min=({:.2},{:.2},{:.2}) world_max=({:.2},{:.2},{:.2})",
                frame.triangles.len() / 3,
                frame.lines.len() / 2,
                min[0],
                min[1],
                min[2],
                max[0],
                max[1],
                max[2],
            ));
            let _ = GLOW_SCENE_LOGGED.set(());
        }
        if MESH_DIAGNOSTICS_LOGGED.get().is_none() {
            debug_log(&format!(
                "runtime.render_glow: mesh geoms seen={}",
                mesh_diagnostics.len()
            ));
            for line in mesh_diagnostics.iter().take(48) {
                debug_log(line);
            }
            for stats in mesh_assets.values() {
                debug_log(&stats.describe(self.model));
            }
            let _ = MESH_DIAGNOSTICS_LOGGED.set(());
        }

        Ok(frame)
    }

    fn model_geom_instance(&self, geom_id: usize) -> Option<mjvGeom> {
        unsafe {
            if geom_id >= (*self.model).ngeom as usize {
                return None;
            }

            if *(*self.model).geom_group.add(geom_id) >= 3 {
                return None;
            }

            let type_ = *(*self.model).geom_type.add(geom_id);
            let dataid = *(*self.model).geom_dataid.add(geom_id);
            let matid = *(*self.model).geom_matid.add(geom_id);
            let mut geom = std::mem::zeroed::<mjvGeom>();
            geom.type_ = type_;
            geom.objtype = MJ_OBJ_GEOM;
            geom.objid = geom_id as i32;
            geom.dataid = if dataid >= 0 { dataid * 2 } else { -1 };

            for axis in 0..3 {
                geom.size[axis] = *(*self.model).geom_size.add(geom_id * 3 + axis) as f32;
                geom.pos[axis] = *(*self.data).geom_xpos.add(geom_id * 3 + axis) as f32;
            }
            for idx in 0..9 {
                geom.mat[idx] = *(*self.data).geom_xmat.add(geom_id * 9 + idx) as f32;
            }

            let rgba_src = if matid >= 0 {
                (*self.model).mat_rgba.add(matid as usize * 4)
            } else {
                (*self.model).geom_rgba.add(geom_id * 4)
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

    fn append_mesh_geom(
        &self,
        triangles: &mut Vec<GlVertex>,
        geom: &mjvGeom,
        diagnostic_colors: bool,
    ) -> Option<MeshGeomStats> {
        let Some(mesh_id) = self.resolve_mesh_id(geom) else {
            return None;
        };

        unsafe {
            let vertadr = *(*self.model).mesh_vertadr.add(mesh_id) as usize;
            let vertnum = *(*self.model).mesh_vertnum.add(mesh_id) as usize;
            let faceadr = *(*self.model).mesh_faceadr.add(mesh_id) as usize;
            let facenum = *(*self.model).mesh_facenum.add(mesh_id) as usize;
            if vertnum == 0 || facenum == 0 {
                return None;
            }

            let all_vertices = std::slice::from_raw_parts(
                (*self.model).mesh_vert,
                (*self.model).nmeshvert as usize * 3,
            );
            let all_normals = std::slice::from_raw_parts(
                (*self.model).mesh_normal,
                (*self.model).nmeshvert as usize * 3,
            );
            let all_faces = std::slice::from_raw_parts(
                (*self.model).mesh_face,
                (*self.model).nmeshface as usize * 3,
            );
            let local_vertices = &all_vertices[vertadr * 3..(vertadr + vertnum) * 3];
            let local_normals = &all_normals[vertadr * 3..(vertadr + vertnum) * 3];
            let face_slice = &all_faces[faceadr * 3..(faceadr + facenum) * 3];
            let mut world_min = [f32::INFINITY; 3];
            let mut world_max = [f32::NEG_INFINITY; 3];
            let starting_vertices = triangles.len();
            let color = display_geom_color(geom, diagnostic_colors);

            for tri in face_slice.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                if ia >= vertnum || ib >= vertnum || ic >= vertnum {
                    continue;
                }

                let a_local = mesh_vertex(local_vertices, ia);
                let b_local = mesh_vertex(local_vertices, ib);
                let c_local = mesh_vertex(local_vertices, ic);
                let a_normal = transform_geom_vector(geom, mesh_vertex(local_normals, ia));
                let b_normal = transform_geom_vector(geom, mesh_vertex(local_normals, ib));
                let c_normal = transform_geom_vector(geom, mesh_vertex(local_normals, ic));

                let a_world = transform_geom_point(geom, a_local);
                let b_world = transform_geom_point(geom, b_local);
                let c_world = transform_geom_point(geom, c_local);
                extend_bounds3(&mut world_min, &mut world_max, a_world);
                extend_bounds3(&mut world_min, &mut world_max, b_world);
                extend_bounds3(&mut world_min, &mut world_max, c_world);

                triangles.push(GlVertex::world(a_world, a_normal, color));
                triangles.push(GlVertex::world(b_world, b_normal, color));
                triangles.push(GlVertex::world(c_world, c_normal, color));
            }

            Some(MeshGeomStats {
                mesh_id,
                triangle_count: (triangles.len() - starting_vertices) / 3,
                world_min,
                world_max,
            })
        }
    }

    fn resolve_mesh_id(&self, geom: &mjvGeom) -> Option<usize> {
        unsafe {
            if geom.objtype == MJ_OBJ_GEOM && geom.objid >= 0 {
                let geom_id = geom.objid as usize;
                let dataid = *(*self.model).geom_dataid.add(geom_id);
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

    fn describe_mesh_geom(&self, geom: &mjvGeom) -> String {
        let resolved = self.resolve_mesh_id(geom);
        let geom_name =
            object_name(self.model, MJ_OBJ_GEOM, geom.objid).unwrap_or_else(|| "-".to_string());
        let (vertnum, facenum, vertadr, faceadr, local_mode) = if let Some(mesh_id) = resolved {
            unsafe {
                let vertadr = *(*self.model).mesh_vertadr.add(mesh_id) as usize;
                let vertnum = *(*self.model).mesh_vertnum.add(mesh_id) as usize;
                let faceadr = *(*self.model).mesh_faceadr.add(mesh_id) as usize;
                let facenum = *(*self.model).mesh_facenum.add(mesh_id) as usize;
                let all_faces = std::slice::from_raw_parts(
                    (*self.model).mesh_face,
                    (*self.model).nmeshface as usize * 3,
                );
                let face_slice = &all_faces[faceadr * 3..(faceadr + facenum) * 3];
                let local_mode = face_slice
                    .iter()
                    .copied()
                    .max()
                    .map(|max_index| (max_index as usize) < vertnum)
                    .unwrap_or(true);
                (vertnum, facenum, vertadr, faceadr, local_mode)
            }
        } else {
            (0, 0, 0, 0, false)
        };

        format!(
            "mesh geom name={} objid={} objtype={} dataid={} resolved_mesh={} vertadr={} vertnum={} faceadr={} facenum={} local_faces={} rgba=({:.2},{:.2},{:.2},{:.2})",
            geom_name,
            geom.objid,
            geom.objtype,
            geom.dataid,
            resolved
                .map(|id| id.to_string())
                .unwrap_or_else(|| "none".to_string()),
            vertadr,
            vertnum,
            faceadr,
            facenum,
            local_mode,
            geom.rgba[0],
            geom.rgba[1],
            geom.rgba[2],
            geom.rgba[3],
        )
    }

    fn mesh_filter_label(&self, mesh_filter: i32) -> String {
        normalize_mesh_filter(mesh_filter)
            .and_then(|mesh_id| object_name(self.model, MJ_OBJ_MESH, mesh_id as i32))
            .unwrap_or_else(|| "unknown".to_string())
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default, Clone)]
struct GlowSceneFrame {
    triangles: Vec<GlVertex>,
    lines: Vec<GlVertex>,
    view_proj: [f32; 16],
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct GlowSceneRenderer {
    program: Option<glow::NativeProgram>,
    triangle_vao: Option<glow::NativeVertexArray>,
    triangle_vbo: Option<glow::NativeBuffer>,
    line_vao: Option<glow::NativeVertexArray>,
    line_vbo: Option<glow::NativeBuffer>,
    present_program: Option<glow::NativeProgram>,
    present_vao: Option<glow::NativeVertexArray>,
    present_vbo: Option<glow::NativeBuffer>,
    offscreen_fbo: Option<glow::NativeFramebuffer>,
    offscreen_color: Option<glow::NativeTexture>,
    offscreen_depth: Option<glow::NativeRenderbuffer>,
    offscreen_size: [i32; 2],
    u_view_proj: Option<glow::NativeUniformLocation>,
    u_unlit: Option<glow::NativeUniformLocation>,
    u_present_texture: Option<glow::NativeUniformLocation>,
    last_error: Option<String>,
    last_render_ms_ms: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl GlowSceneRenderer {
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
            .ok_or_else(|| "Offscreen FBO is not initialized".to_string())?;
        let color = self
            .offscreen_color
            .ok_or_else(|| "Offscreen color texture is not initialized".to_string())?;
        let depth = self
            .offscreen_depth
            .ok_or_else(|| "Offscreen depth renderbuffer is not initialized".to_string())?;

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
        scene: &GlowSceneFrame,
    ) -> Result<(), String> {
        self.ensure_initialized(gl)?;

        let started = Instant::now();
        let viewport = info.viewport_in_pixels();
        let clip = info.clip_rect_in_pixels();
        if GLOW_CALLBACK_LOGGED.get().is_none() {
            debug_log(&format!(
                "runtime.render_glow: callback viewport=({},{} {}x{}) clip=({},{} {}x{}) triangles={} lines={}",
                viewport.left_px,
                viewport.from_bottom_px,
                viewport.width_px,
                viewport.height_px,
                clip.left_px,
                clip.from_bottom_px,
                clip.width_px,
                clip.height_px,
                scene.triangles.len() / 3,
                scene.lines.len() / 2
            ));
            let _ = GLOW_CALLBACK_LOGGED.set(());
        }
        let program = self
            .program
            .ok_or_else(|| "Glow program is not initialized".to_string())?;
        let present_program = self
            .present_program
            .ok_or_else(|| "Present program is not initialized".to_string())?;
        let fbo = self
            .offscreen_fbo
            .ok_or_else(|| "Offscreen FBO is not initialized".to_string())?;
        let offscreen_color = self
            .offscreen_color
            .ok_or_else(|| "Offscreen color texture is not initialized".to_string())?;
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
        self.last_render_ms_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GlVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

#[cfg(not(target_arch = "wasm32"))]
impl GlVertex {
    fn world(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct MeshGeomStats {
    mesh_id: usize,
    triangle_count: usize,
    world_min: [f32; 3],
    world_max: [f32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
struct MeshAssetStats {
    mesh_id: usize,
    instances: usize,
    triangle_count: usize,
    world_min: [f32; 3],
    world_max: [f32; 3],
}

#[cfg(not(target_arch = "wasm32"))]
impl MeshAssetStats {
    fn new(mesh_id: usize) -> Self {
        Self {
            mesh_id,
            instances: 0,
            triangle_count: 0,
            world_min: [f32::INFINITY; 3],
            world_max: [f32::NEG_INFINITY; 3],
        }
    }

    fn record(&mut self, stats: &MeshGeomStats) {
        self.instances += 1;
        self.triangle_count += stats.triangle_count;
        extend_bounds3(&mut self.world_min, &mut self.world_max, stats.world_min);
        extend_bounds3(&mut self.world_min, &mut self.world_max, stats.world_max);
    }

    fn describe(&self, model: *const mjModel) -> String {
        let mesh_name =
            object_name(model, MJ_OBJ_MESH, self.mesh_id as i32).unwrap_or_else(|| "-".to_string());
        format!(
            "mesh asset name={} id={} instances={} tris={} world_min=({:.2},{:.2},{:.2}) world_max=({:.2},{:.2},{:.2})",
            mesh_name,
            self.mesh_id,
            self.instances,
            self.triangle_count,
            self.world_min[0],
            self.world_min[1],
            self.world_min[2],
            self.world_max[0],
            self.world_max[1],
            self.world_max[2],
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vertex_bounds(vertices: &[GlVertex]) -> ([f32; 3], [f32; 3]) {
    if vertices.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            let value = vertex.position[axis];
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    (min, max)
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

#[cfg(not(target_arch = "wasm32"))]
fn debug_log_if(condition: bool, message: &str) {
    if condition {
        debug_log(message);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn object_name(model: *const mjModel, objtype: i32, objid: i32) -> Option<String> {
    if model.is_null() || objid < 0 {
        return None;
    }

    unsafe {
        let ptr = mj_id2name(model, objtype, objid);
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_mesh_filter(mesh_filter: i32) -> Option<usize> {
    (mesh_filter >= 0).then_some(mesh_filter as usize)
}

#[cfg(not(target_arch = "wasm32"))]
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

    fn view_projection_matrix(&self) -> [f32; 16] {
        mul_mat4(
            self.projection_matrix_with_depth_range(self.frustum_near, self.frustum_far),
            self.view_matrix(),
        )
    }

    fn fitted_view_projection_matrix(&self, points: &[GlVertex]) -> [f32; 16] {
        let Some((near, far)) = self.fitted_depth_range(points) else {
            return self.view_projection_matrix();
        };
        mul_mat4(
            self.projection_matrix_with_depth_range(near, far),
            self.view_matrix(),
        )
    }

    fn fitted_depth_range(&self, points: &[GlVertex]) -> Option<(f32, f32)> {
        let mut max_depth = f32::NEG_INFINITY;
        for vertex in points {
            let depth = self.depth_of(vertex.position);
            if depth > 1e-4 {
                max_depth = max_depth.max(depth);
            }
        }

        if !max_depth.is_finite() {
            return None;
        }

        // Keep MuJoCo's original near plane so perspective scale stays stable.
        // Tighten only the far plane to improve depth precision.
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
fn transform_mesh_point(
    point: [f32; 3],
    scale: [f32; 3],
    pos: [f32; 3],
    quat: [f32; 4],
) -> [f32; 3] {
    add3(rotate_quat(scale3_components(point, scale), quat), pos)
}

#[cfg(not(target_arch = "wasm32"))]
fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize3(cross3(sub3(b, a), sub3(c, a)))
}

#[cfg(not(target_arch = "wasm32"))]
fn scale3_components(v: [f32; 3], scale: [f32; 3]) -> [f32; 3] {
    [v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]]
}

#[cfg(not(target_arch = "wasm32"))]
fn rotate_quat(v: [f32; 3], quat: [f32; 4]) -> [f32; 3] {
    let q = normalize4(quat);
    let qv = [q[1], q[2], q[3]];
    let t = scale3(cross3(qv, v), 2.0);
    add3(v, add3(scale3(t, q[0]), cross3(qv, t)))
}

#[cfg(not(target_arch = "wasm32"))]
fn transform_geom_point(geom: &mjvGeom, local: [f32; 3]) -> [f32; 3] {
    [
        geom.pos[0] + geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.pos[1] + geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.pos[2] + geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn transform_geom_vector(geom: &mjvGeom, local: [f32; 3]) -> [f32; 3] {
    normalize3([
        geom.mat[0] * local[0] + geom.mat[1] * local[1] + geom.mat[2] * local[2],
        geom.mat[3] * local[0] + geom.mat[4] * local[1] + geom.mat[5] * local[2],
        geom.mat[6] * local[0] + geom.mat[7] * local[1] + geom.mat[8] * local[2],
    ])
}

#[cfg(not(target_arch = "wasm32"))]
fn create_gl_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::NativeProgram, String> {
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

#[cfg(not(target_arch = "wasm32"))]
fn setup_vertex_array(
    gl: &glow::Context,
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    vbo: glow::NativeBuffer,
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

#[cfg(not(target_arch = "wasm32"))]
fn setup_present_quad(
    gl: &glow::Context,
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    vbo: glow::NativeBuffer,
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

#[cfg(not(target_arch = "wasm32"))]
fn slice_as_u8<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(not(target_arch = "wasm32"))]
const GLOW_VERTEX_SHADER: &str = r#"#version 150 core
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

#[cfg(not(target_arch = "wasm32"))]
const GLOW_FRAGMENT_SHADER: &str = r#"#version 150 core
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

#[cfg(not(target_arch = "wasm32"))]
const PRESENT_VERTEX_SHADER: &str = r#"#version 150 core
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

#[cfg(not(target_arch = "wasm32"))]
const PRESENT_FRAGMENT_SHADER: &str = r#"#version 150 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = texture(u_texture, v_uv);
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
fn orbit_forward(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    [
        elevation.cos() * azimuth.cos(),
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
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
fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if len <= f32::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len, v[3] / len]
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

#[cfg(all(not(target_arch = "wasm32"), unix))]
struct StderrSilencer {
    saved_stderr: i32,
    dev_null: i32,
}

#[cfg(all(not(target_arch = "wasm32"), unix))]
impl StderrSilencer {
    fn new() -> Option<Self> {
        unsafe {
            let dev_null_path = CString::new("/dev/null").ok()?;
            let dev_null = open(dev_null_path.as_ptr(), O_WRONLY);
            if dev_null < 0 {
                return None;
            }

            let saved_stderr = dup(STDERR_FILENO);
            if saved_stderr < 0 {
                close(dev_null);
                return None;
            }

            if dup2(dev_null, STDERR_FILENO) < 0 {
                close(saved_stderr);
                close(dev_null);
                return None;
            }

            Some(Self {
                saved_stderr,
                dev_null,
            })
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), unix))]
impl Drop for StderrSilencer {
    fn drop(&mut self) {
        unsafe {
            let _ = dup2(self.saved_stderr, STDERR_FILENO);
            let _ = close(self.saved_stderr);
            let _ = close(self.dev_null);
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), unix))]
const O_WRONLY: i32 = 0x0001;
#[cfg(all(not(target_arch = "wasm32"), unix))]
const STDERR_FILENO: i32 = 2;

#[cfg(all(not(target_arch = "wasm32"), unix))]
unsafe extern "C" {
    fn open(path: *const std::os::raw::c_char, oflag: i32) -> i32;
    fn dup(fd: i32) -> i32;
    fn dup2(src: i32, dst: i32) -> i32;
    fn close(fd: i32) -> i32;
}

#[cfg(all(not(target_arch = "wasm32"), not(unix)))]
struct StderrSilencer;

#[cfg(all(not(target_arch = "wasm32"), not(unix)))]
impl StderrSilencer {
    fn new() -> Option<Self> {
        None
    }
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
fn oscillator(time_s: f32) -> [f32; 12] {
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

#[cfg(not(target_arch = "wasm32"))]
fn quaternion_yaw(q: [f32; 4]) -> f32 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    siny_cosp.atan2(cosy_cosp)
}

#[cfg(not(target_arch = "wasm32"))]
fn rotate_vector_by_inverse_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let inv = [q[0], -q[1], -q[2], -q[3]];
    rotate_vector_by_quaternion(inv, v)
}

#[cfg(not(target_arch = "wasm32"))]
fn rotate_vector_by_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
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

#[cfg(not(target_arch = "wasm32"))]
fn flip_rgb_to_rgba(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let src_y = height - 1 - y;
        let src_row = src_y * width * 3;
        let dst_row = y * width * 4;
        for x in 0..width {
            let src = src_row + x * 3;
            let dst = dst_row + x * 4;
            rgba[dst] = rgb[src];
            rgba[dst + 1] = rgb[src + 1];
            rgba[dst + 2] = rgb[src + 2];
            rgba[dst + 3] = 255;
        }
    }
    rgba
}

#[cfg(not(target_arch = "wasm32"))]
fn shift_history_3(history: &mut [[f32; 3]; 3], sample: [f32; 3]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}

#[cfg(not(target_arch = "wasm32"))]
fn shift_history_12(history: &mut [[f32; 12]; 3], sample: [f32; 12]) {
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}
