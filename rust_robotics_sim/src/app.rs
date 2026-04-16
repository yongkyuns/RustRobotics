use crate::{simulator::Simulator, theme};

use eframe::egui;
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
    restart_requested: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WebEmbedMode {
    #[default]
    FullApp,
    FocusedMainContent,
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
}

pub struct App {
    sim: Simulator,
    last_frame_at: Option<Instant>,
    fps_ema: f32,
    #[cfg(target_arch = "wasm32")]
    embed_mode: WebEmbedMode,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&_cc.egui_ctx);
        let app = Self {
            sim: Simulator::default(),
            last_frame_at: None,
            fps_ema: 0.0,
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
        theme::apply_density(
            ctx,
            if self.embed_mode == WebEmbedMode::FocusedMainContent {
                theme::UiDensity::Compact
            } else {
                theme::UiDensity::Comfortable
            },
        );

        #[cfg(target_arch = "wasm32")]
        let panel_frame = if self.embed_mode == WebEmbedMode::FocusedMainContent {
            egui::Frame::NONE
        } else {
            egui::Frame::new().fill(egui::Color32::from_rgb(12, 16, 22))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let panel_frame = egui::Frame::new().fill(egui::Color32::from_rgb(12, 16, 22));

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
        if bridge.restart_requested {
            app.sim.restart();
            bridge.restart_requested = false;
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn publish_web_test_state(sim: &Simulator) {
    WEB_TEST_BRIDGE.with(|bridge| {
        bridge.borrow_mut().state = Some(WebTestState {
            mode: sim.mode().test_id(),
            paused: sim.paused(),
            time: sim.time(),
        });
    });
}

#[cfg(target_arch = "wasm32")]
fn publish_embed_height(ctx: &egui::Context, sim: &Simulator, embed_mode: WebEmbedMode) {
    if embed_mode != WebEmbedMode::FocusedMainContent {
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
