use crate::{
    simulator::{PendulumPatch, Simulator},
    theme,
};

use eframe::egui::{self, CornerRadius, Stroke};
use std::time::Duration;
use web_time::Instant;

#[cfg(target_arch = "wasm32")]
use crate::simulator::SimMode;
#[cfg(target_arch = "wasm32")]
use js_sys::{Object, Reflect};
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
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
    state: Option<WebTestState>,
    pending_mode: Option<SimMode>,
    pending_paused: Option<bool>,
    pending_embed_mode: Option<WebEmbedMode>,
    pending_theme: Option<WebThemeMode>,
    pending_show_graph: Option<bool>,
    pending_sim_speed: Option<usize>,
    pending_pendulum_noise_enabled: Option<bool>,
    pending_pendulum_noise_scale: Option<f32>,
    pending_pendulum_controller: Option<(usize, SimPendulumControllerKind)>,
    pending_pendulum_patch: Option<(usize, PendulumPatch)>,
    pending_remove_pendulum: Option<usize>,
    add_pendulum_requested: bool,
    restart_requested: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimPendulumControllerKind {
    Lqr,
    Pid,
    Mpc,
    Policy,
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
pub(crate) struct WebTestState {
    mode: &'static str,
    paused: bool,
    time: f32,
    show_graph: bool,
    sim_speed: usize,
    pendulum_noise_enabled: bool,
    pendulum_noise_scale: f32,
    pendulum_count: usize,
    pendulums: Vec<crate::simulator::PendulumCardState>,
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
        if let Some(mode) = bridge.pending_mode.take() {
            app.sim.set_mode(mode);
        }
        if let Some(paused) = bridge.pending_paused.take() {
            app.sim.set_paused(paused);
        }
        if let Some(embed_mode) = bridge.pending_embed_mode.take() {
            app.embed_mode = embed_mode;
        }
        if let Some(theme_mode) = bridge.pending_theme.take() {
            app.theme_mode = theme_mode.ui_theme();
        }
        if let Some(show_graph) = bridge.pending_show_graph.take() {
            app.sim.set_show_graph(show_graph);
        }
        if let Some(sim_speed) = bridge.pending_sim_speed.take() {
            app.sim.set_sim_speed(sim_speed);
        }
        if let Some(enabled) = bridge.pending_pendulum_noise_enabled.take() {
            app.sim.set_pendulum_noise_enabled(enabled);
        }
        if let Some(scale) = bridge.pending_pendulum_noise_scale.take() {
            app.sim.set_pendulum_noise_scale(scale);
        }
        if let Some((pendulum_id, kind)) = bridge.pending_pendulum_controller.take() {
            app.sim.set_pendulum_controller_kind(
                pendulum_id,
                match kind {
                    SimPendulumControllerKind::Lqr => {
                        crate::simulator::pendulum::ControllerKind::Lqr
                    }
                    SimPendulumControllerKind::Pid => {
                        crate::simulator::pendulum::ControllerKind::Pid
                    }
                    SimPendulumControllerKind::Mpc => {
                        crate::simulator::pendulum::ControllerKind::Mpc
                    }
                    SimPendulumControllerKind::Policy => {
                        crate::simulator::pendulum::ControllerKind::Policy
                    }
                },
            );
        }
        if let Some((pendulum_id, patch)) = bridge.pending_pendulum_patch.take() {
            app.sim.patch_pendulum(pendulum_id, patch);
        }
        if let Some(pendulum_id) = bridge.pending_remove_pendulum.take() {
            app.sim.remove_pendulum(pendulum_id);
        }
        if bridge.add_pendulum_requested {
            app.sim.add_active_simulation();
            bridge.add_pendulum_requested = false;
        }
        if bridge.restart_requested {
            app.sim.restart();
            bridge.restart_requested = false;
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn publish_web_test_state(sim: &Simulator) {
    WEB_TEST_BRIDGE.with(|bridge| {
        let pendulums = sim.pendulum_ui_state();

        let mut bridge = bridge.borrow_mut();
        bridge.state = Some(WebTestState {
            mode: sim.mode().test_id(),
            paused: sim.paused(),
            time: sim.time(),
            show_graph: sim.show_graph(),
            sim_speed: sim.sim_speed(),
            pendulum_noise_enabled: sim.pendulum_noise_enabled(),
            pendulum_noise_scale: sim.pendulum_noise_scale(),
            pendulum_count: sim.pendulum_count(),
            pendulums,
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
    if sim.mode() == crate::simulator::SimMode::InvertedPendulum {
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
pub(crate) fn web_test_state() -> Option<WebTestState> {
    WEB_TEST_BRIDGE.with(|bridge| bridge.borrow().state.clone())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_mode(mode: SimMode) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_mode = Some(mode);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_paused(paused: bool) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_paused = Some(paused);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_restart() {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().restart_requested = true;
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_embed_mode(mode: WebEmbedMode) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_embed_mode = Some(mode);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_theme_mode(mode: WebThemeMode) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_theme = Some(mode);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_show_graph(show_graph: bool) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_show_graph = Some(show_graph);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_sim_speed(sim_speed: usize) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_sim_speed = Some(sim_speed);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_noise_enabled(enabled: bool) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_pendulum_noise_enabled = Some(enabled);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_noise_scale(scale: f32) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_pendulum_noise_scale = Some(scale);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_add_pendulum() {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().add_pendulum_requested = true;
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_set_pendulum_controller(
    pendulum_id: usize,
    kind: SimPendulumControllerKind,
) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_pendulum_controller = Some((pendulum_id, kind));
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_remove_pendulum(pendulum_id: usize) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_remove_pendulum = Some(pendulum_id);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn web_test_patch_pendulum(pendulum_id: usize, patch: PendulumPatch) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().pending_pendulum_patch = Some((pendulum_id, patch));
    });
}
