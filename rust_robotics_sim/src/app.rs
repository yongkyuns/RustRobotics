use crate::{simulator::Simulator, theme};

use eframe::egui;
use std::time::Duration;
use web_time::Instant;

#[cfg(target_arch = "wasm32")]
use crate::simulator::SimMode;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_TEST_BRIDGE: RefCell<WebTestBridge> = RefCell::new(WebTestBridge::default());
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WebTestBridge {
    state: Option<WebTestState>,
    pending_mode: Option<SimMode>,
    pending_paused: Option<bool>,
    restart_requested: bool,
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
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&_cc.egui_ctx);
        let app = Self {
            sim: Simulator::default(),
            last_frame_at: None,
            fps_ema: 0.0,
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
        apply_web_test_commands(&mut self.sim);

        self.sim.update();

        #[cfg(target_arch = "wasm32")]
        publish_web_test_state(&self.sim);

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(12, 16, 22))
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                self.sim.ui(ui, Some(frame), self.fps_ema);
            });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_web_test_commands(sim: &mut Simulator) {
    WEB_TEST_BRIDGE.with(|bridge| {
        let mut bridge = bridge.borrow_mut();
        if let Some(mode) = bridge.pending_mode.take() {
            sim.set_mode(mode);
        }
        if let Some(paused) = bridge.pending_paused.take() {
            sim.set_paused(paused);
        }
        if bridge.restart_requested {
            sim.restart();
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
