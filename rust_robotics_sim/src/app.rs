#[cfg(target_arch = "wasm32")]
use crate::simulator::{
    LocalizationDriveMode, LocalizationPatch, PathPlannerPatch, PathPlanningAlgorithm,
    PendulumPatch, SlamDriveMode, SlamPatch,
};
use crate::{simulator::Simulator, theme};

use eframe::egui::{self, CornerRadius, Stroke};
use std::time::Duration;
use web_time::Instant;

#[cfg(target_arch = "wasm32")]
use crate::simulator::SimMode;
#[cfg(target_arch = "wasm32")]
use js_sys::{Object, Reflect};
#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_TEST_BRIDGE: RefCell<WebTestBridge> = RefCell::new(WebTestBridge::default());
    static LAST_EMBED_HEIGHT_PX: RefCell<Option<i32>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WebTestBridge {
    state: Option<WebEmbedState>,
    pending_actions: Vec<WebEmbedAction>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WebEmbedMode {
    #[default]
    FullApp,
    FocusedMainContent,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WebThemeMode {
    Light,
    #[default]
    Dark,
}

#[cfg(target_arch = "wasm32")]
impl WebThemeMode {
    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value {
            "light" => Some(Self::Light),
            "dark" => Some(Self::Dark),
            _ => None,
        }
    }

    fn ui_theme(self) -> theme::UiTheme {
        match self {
            Self::Light => theme::UiTheme::Light,
            Self::Dark => theme::UiTheme::Dark,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl WebEmbedMode {
    pub(crate) fn from_query_value(value: &str) -> Option<Self> {
        match value {
            "full" | "app" => Some(Self::FullApp),
            "focused" | "content" | "main_content" => Some(Self::FocusedMainContent),
            _ => None,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WebEmbedState {
    mode: &'static str,
    paused: bool,
    time: f32,
    toolbar: WebEmbedToolbarState,
    view: WebEmbedViewState,
    payload: WebEmbedPayload,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WebEmbedToolbarState {
    sim_speed: usize,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WebEmbedViewState {
    show_graph: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WebEmbedPayload {
    Localization {
        vehicle_count: usize,
        vehicles: Vec<crate::simulator::LocalizationCardState>,
    },
    Slam {
        slam_count: usize,
        demos: Vec<crate::simulator::SlamCardState>,
    },
    Pendulum {
        noise_enabled: bool,
        noise_scale: f32,
        pendulum_count: usize,
        pendulums: Vec<crate::simulator::PendulumCardState>,
    },
    PathPlanning {
        env_mode: crate::simulator::path_planning::EnvironmentMode,
        continuous_obstacle_radius: f32,
        planner_count: usize,
        planners: Vec<crate::simulator::PathPlannerCardState>,
    },
    Unsupported,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WebEmbedAction {
    SetMode {
        mode: String,
    },
    SetPaused {
        paused: bool,
    },
    Restart,
    SetEmbedMode {
        mode: String,
    },
    SetTheme {
        theme: String,
    },
    SetShowGraph {
        show_graph: bool,
    },
    SetSimSpeed {
        sim_speed: usize,
    },
    SetPendulumNoiseEnabled {
        enabled: bool,
    },
    SetPendulumNoiseScale {
        scale: f32,
    },
    AddActiveSimulation,
    RemoveLocalizationVehicle {
        vehicle_id: usize,
    },
    SetLocalizationDriveMode {
        vehicle_id: usize,
        drive_mode: LocalizationDriveMode,
    },
    PatchLocalizationVehicle {
        vehicle_id: usize,
        patch: LocalizationPatch,
    },
    SetSlamDriveMode {
        slam_id: usize,
        drive_mode: SlamDriveMode,
    },
    RemoveSlamDemo {
        slam_id: usize,
    },
    SetSlamEkfEnabled {
        slam_id: usize,
        enabled: bool,
    },
    SetSlamGraphEnabled {
        slam_id: usize,
        enabled: bool,
    },
    PatchSlam {
        slam_id: usize,
        patch: SlamPatch,
    },
    AddPendulum,
    RemovePendulum {
        pendulum_id: usize,
    },
    SetPendulumController {
        pendulum_id: usize,
        controller: crate::simulator::pendulum::ControllerKind,
    },
    PatchPendulum {
        pendulum_id: usize,
        patch: PendulumPatch,
    },
    SetPathPlanningEnvMode {
        mode: crate::simulator::path_planning::EnvironmentMode,
    },
    SetPathPlanningContinuousObstacleRadius {
        radius: f32,
    },
    RemovePathPlanner {
        planner_id: usize,
    },
    SetPathPlannerAlgorithm {
        planner_id: usize,
        algorithm: PathPlanningAlgorithm,
    },
    SetPathPlannerShowVisited {
        planner_id: usize,
        show_visited: bool,
    },
    PatchPathPlanner {
        planner_id: usize,
        patch: PathPlannerPatch,
    },
}

pub struct App {
    sim: Simulator,
    last_frame_at: Option<Instant>,
    fps_ema: f32,
    theme_mode: theme::UiTheme,
    #[cfg(target_arch = "wasm32")]
    embed_mode: WebEmbedMode,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let theme_mode = theme::UiTheme::Dark;
        theme::install(&_cc.egui_ctx, theme::UiDensity::Comfortable, theme_mode);
        let app = Self {
            sim: Simulator::default(),
            last_frame_at: None,
            fps_ema: 0.0,
            theme_mode,
            #[cfg(target_arch = "wasm32")]
            embed_mode: WebEmbedMode::FullApp,
        };
        #[cfg(target_arch = "wasm32")]
        publish_web_test_state(&app.sim);
        app
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let now = Instant::now();
        if let Some(last_frame_at) = self.last_frame_at.replace(now) {
            let dt = (now - last_frame_at).as_secs_f32();
            if dt > 0.0 {
                let fps = 1.0 / dt;
                self.fps_ema = if self.fps_ema > 0.0 {
                    self.fps_ema * 0.9 + fps * 0.1
                } else {
                    fps
                };
            }
        }

        #[cfg(target_arch = "wasm32")]
        apply_web_test_commands(self);

        self.sim.update();

        #[cfg(target_arch = "wasm32")]
        let panel_margin = if self.embed_mode == WebEmbedMode::FocusedMainContent {
            12
        } else {
            14
        };
        #[cfg(not(target_arch = "wasm32"))]
        let panel_margin = 14;

        #[cfg(target_arch = "wasm32")]
        let density = if self.embed_mode == WebEmbedMode::FocusedMainContent {
            theme::UiDensity::Compact
        } else {
            theme::UiDensity::Comfortable
        };
        #[cfg(not(target_arch = "wasm32"))]
        let density = theme::UiDensity::Comfortable;

        theme::apply(ctx, density, self.theme_mode);

        let panel_fill = ctx.style().visuals.panel_fill;
        let panel_stroke = ctx.style().visuals.widgets.noninteractive.bg_stroke;
        #[cfg(target_arch = "wasm32")]
        let panel_frame = if self.embed_mode == WebEmbedMode::FocusedMainContent {
            egui::Frame::group(ctx.style().as_ref())
                .fill(panel_fill)
                .corner_radius(CornerRadius::same(16))
                .stroke(Stroke::new(panel_stroke.width.max(1.0), panel_stroke.color))
        } else {
            egui::Frame::new()
                .fill(panel_fill)
                .corner_radius(CornerRadius::same(16))
                .stroke(Stroke::new(panel_stroke.width.max(1.0), panel_stroke.color))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let panel_frame = egui::Frame::new()
            .fill(panel_fill)
            .corner_radius(CornerRadius::same(16))
            .stroke(Stroke::new(panel_stroke.width.max(1.0), panel_stroke.color));

        egui::CentralPanel::default()
            .frame(panel_frame.inner_margin(egui::Margin::same(panel_margin)))
            .show(ctx, |ui| {
                #[cfg(target_arch = "wasm32")]
                if self.embed_mode == WebEmbedMode::FocusedMainContent {
                    self.sim.ui_embedded(ui, Some(frame), self.fps_ema);
                    return;
                }

                self.sim.ui(ui, Some(frame), self.fps_ema);
            });

        #[cfg(target_arch = "wasm32")]
        {
            publish_web_test_state(&self.sim);
            publish_embed_height(ctx, &self.sim, self.embed_mode);
        }

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_web_test_commands(app: &mut App) {
    WEB_TEST_BRIDGE.with(|bridge| {
        let mut bridge = bridge.borrow_mut();
        for action in bridge.pending_actions.drain(..) {
            match action {
                WebEmbedAction::SetMode { mode } => {
                    if let Some(mode) = SimMode::from_test_id(&mode) {
                        app.sim.set_mode(mode);
                    }
                }
                WebEmbedAction::SetPaused { paused } => app.sim.set_paused(paused),
                WebEmbedAction::Restart => app.sim.restart(),
                WebEmbedAction::SetEmbedMode { mode } => {
                    if let Some(embed_mode) = WebEmbedMode::from_query_value(&mode) {
                        app.embed_mode = embed_mode;
                    }
                }
                WebEmbedAction::SetTheme { theme } => {
                    if let Some(theme_mode) = WebThemeMode::from_str(&theme) {
                        app.theme_mode = theme_mode.ui_theme();
                    }
                }
                WebEmbedAction::SetShowGraph { show_graph } => app.sim.set_show_graph(show_graph),
                WebEmbedAction::SetSimSpeed { sim_speed } => app.sim.set_sim_speed(sim_speed),
                WebEmbedAction::SetPendulumNoiseEnabled { enabled } => {
                    app.sim.set_pendulum_noise_enabled(enabled);
                }
                WebEmbedAction::SetPendulumNoiseScale { scale } => {
                    app.sim.set_pendulum_noise_scale(scale);
                }
                WebEmbedAction::AddActiveSimulation => app.sim.add_active_simulation(),
                WebEmbedAction::RemoveLocalizationVehicle { vehicle_id } => {
                    app.sim.remove_localization_vehicle(vehicle_id);
                }
                WebEmbedAction::SetLocalizationDriveMode {
                    vehicle_id,
                    drive_mode,
                } => {
                    app.sim.set_localization_drive_mode(vehicle_id, drive_mode);
                }
                WebEmbedAction::PatchLocalizationVehicle { vehicle_id, patch } => {
                    app.sim.patch_localization_vehicle(vehicle_id, patch);
                }
                WebEmbedAction::SetSlamDriveMode {
                    slam_id,
                    drive_mode,
                } => {
                    app.sim.set_slam_drive_mode(slam_id, drive_mode);
                }
                WebEmbedAction::RemoveSlamDemo { slam_id } => {
                    app.sim.remove_slam_demo(slam_id);
                }
                WebEmbedAction::SetSlamEkfEnabled { slam_id, enabled } => {
                    app.sim.set_slam_ekf_enabled(slam_id, enabled);
                }
                WebEmbedAction::SetSlamGraphEnabled { slam_id, enabled } => {
                    app.sim.set_slam_graph_enabled(slam_id, enabled);
                }
                WebEmbedAction::PatchSlam { slam_id, patch } => {
                    app.sim.patch_slam(slam_id, patch);
                }
                WebEmbedAction::AddPendulum => app.sim.add_active_simulation(),
                WebEmbedAction::RemovePendulum { pendulum_id } => {
                    app.sim.remove_pendulum(pendulum_id);
                }
                WebEmbedAction::SetPendulumController {
                    pendulum_id,
                    controller,
                } => {
                    app.sim
                        .set_pendulum_controller_kind(pendulum_id, controller);
                }
                WebEmbedAction::PatchPendulum { pendulum_id, patch } => {
                    app.sim.patch_pendulum(pendulum_id, patch);
                }
                WebEmbedAction::SetPathPlanningEnvMode { mode } => {
                    app.sim.set_path_planning_env_mode(mode);
                }
                WebEmbedAction::SetPathPlanningContinuousObstacleRadius { radius } => {
                    app.sim.set_path_planning_continuous_obstacle_radius(radius);
                }
                WebEmbedAction::RemovePathPlanner { planner_id } => {
                    app.sim.remove_path_planner(planner_id);
                }
                WebEmbedAction::SetPathPlannerAlgorithm {
                    planner_id,
                    algorithm,
                } => {
                    app.sim.set_path_planner_algorithm(planner_id, algorithm);
                }
                WebEmbedAction::SetPathPlannerShowVisited {
                    planner_id,
                    show_visited,
                } => {
                    app.sim
                        .set_path_planner_show_visited(planner_id, show_visited);
                }
                WebEmbedAction::PatchPathPlanner { planner_id, patch } => {
                    app.sim.patch_path_planner(planner_id, patch);
                }
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn publish_web_test_state(sim: &Simulator) {
    WEB_TEST_BRIDGE.with(|bridge| {
        let vehicles = sim.localization_ui_state();
        let slam_demos = sim.slam_ui_state();
        let pendulums = sim.pendulum_ui_state();
        let planners = sim.path_planning_ui_state();

        let mut bridge = bridge.borrow_mut();
        bridge.state = Some(WebEmbedState {
            mode: sim.mode().test_id(),
            paused: sim.paused(),
            time: sim.time(),
            toolbar: WebEmbedToolbarState {
                sim_speed: sim.sim_speed(),
            },
            view: WebEmbedViewState {
                show_graph: sim.show_graph(),
            },
            payload: if sim.mode() == crate::simulator::SimMode::Localization {
                WebEmbedPayload::Localization {
                    vehicle_count: vehicles.len(),
                    vehicles,
                }
            } else if sim.mode() == crate::simulator::SimMode::Slam {
                WebEmbedPayload::Slam {
                    slam_count: slam_demos.len(),
                    demos: slam_demos,
                }
            } else if sim.mode() == crate::simulator::SimMode::InvertedPendulum {
                WebEmbedPayload::Pendulum {
                    noise_enabled: sim.pendulum_noise_enabled(),
                    noise_scale: sim.pendulum_noise_scale(),
                    pendulum_count: sim.pendulum_count(),
                    pendulums,
                }
            } else if sim.mode() == crate::simulator::SimMode::PathPlanning {
                WebEmbedPayload::PathPlanning {
                    env_mode: sim.path_planning_env_mode(),
                    continuous_obstacle_radius: sim.path_planning_continuous_obstacle_radius(),
                    planner_count: planners.len(),
                    planners,
                }
            } else {
                WebEmbedPayload::Unsupported
            },
        });
    });
}

#[cfg(target_arch = "wasm32")]
fn publish_embed_height(ctx: &egui::Context, sim: &Simulator, embed_mode: WebEmbedMode) {
    if embed_mode != WebEmbedMode::FocusedMainContent {
        return;
    }

    // The focused pendulum tutorial uses native DOM controls outside egui and
    // reports total document height from `docs/index.html`. If the wasm app
    // also posts its scene-only height, the parent iframe can be shrunk back
    // down and crop the DOM controls.
    if matches!(
        sim.mode(),
        crate::simulator::SimMode::Localization
            | crate::simulator::SimMode::InvertedPendulum
            | crate::simulator::SimMode::PathPlanning
            | crate::simulator::SimMode::Slam
    ) {
        return;
    }

    let Some(height_points) = sim.embedded_content_height() else {
        return;
    };

    let height_px = (height_points * ctx.pixels_per_point()).ceil() as i32;
    let should_send = LAST_EMBED_HEIGHT_PX.with(|last| {
        let mut last = last.borrow_mut();
        if last.as_ref().is_some_and(|previous| *previous == height_px) {
            false
        } else {
            *last = Some(height_px);
            true
        }
    });

    if !should_send {
        return;
    }

    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(parent) = window.parent().ok().flatten() else {
        return;
    };

    let payload = Object::new();
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("type"),
        &JsValue::from_str("rust-robotics-embed-size"),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("height"),
        &JsValue::from_f64(height_px as f64),
    );
    let _ = parent.post_message(&payload.into(), "*");
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_embed_state() -> Option<WebEmbedState> {
    WEB_TEST_BRIDGE.with(|bridge| bridge.borrow().state.clone())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_embed_dispatch(action: WebEmbedAction) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_actions.push(action);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_state() -> Option<WebEmbedState> {
    web_embed_state()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_mode(mode: SimMode) {
    web_embed_dispatch(WebEmbedAction::SetMode {
        mode: mode.test_id().to_owned(),
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_paused(paused: bool) {
    web_embed_dispatch(WebEmbedAction::SetPaused { paused });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_restart() {
    web_embed_dispatch(WebEmbedAction::Restart);
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_embed_mode(mode: WebEmbedMode) {
    web_embed_dispatch(WebEmbedAction::SetEmbedMode {
        mode: match mode {
            WebEmbedMode::FullApp => "full".to_owned(),
            WebEmbedMode::FocusedMainContent => "focused".to_owned(),
        },
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_theme_mode(mode: WebThemeMode) {
    web_embed_dispatch(WebEmbedAction::SetTheme {
        theme: match mode {
            WebThemeMode::Light => "light".to_owned(),
            WebThemeMode::Dark => "dark".to_owned(),
        },
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_show_graph(show_graph: bool) {
    web_embed_dispatch(WebEmbedAction::SetShowGraph { show_graph });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_sim_speed(sim_speed: usize) {
    web_embed_dispatch(WebEmbedAction::SetSimSpeed { sim_speed });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_noise_enabled(enabled: bool) {
    web_embed_dispatch(WebEmbedAction::SetPendulumNoiseEnabled { enabled });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_noise_scale(scale: f32) {
    web_embed_dispatch(WebEmbedAction::SetPendulumNoiseScale { scale });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_add_pendulum() {
    web_embed_dispatch(WebEmbedAction::AddPendulum);
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_controller(
    pendulum_id: usize,
    kind: crate::simulator::pendulum::ControllerKind,
) {
    web_embed_dispatch(WebEmbedAction::SetPendulumController {
        pendulum_id,
        controller: kind,
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_remove_pendulum(pendulum_id: usize) {
    web_embed_dispatch(WebEmbedAction::RemovePendulum { pendulum_id });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_patch_pendulum(pendulum_id: usize, patch: PendulumPatch) {
    web_embed_dispatch(WebEmbedAction::PatchPendulum { pendulum_id, patch });
}
