use crate::simulator::Simulator;

use eframe::egui;
use std::time::Duration;
use web_time::Instant;

pub struct App {
    sim: Simulator,
    last_frame_at: Option<Instant>,
    fps_ema: f32,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            sim: Simulator::default(),
            last_frame_at: None,
            fps_ema: 0.0,
        }
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

        self.sim.update();

        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.set_visuals(egui::Visuals::dark());
            self.sim.ui(ui, Some(frame));
        });

        egui::Area::new("perf_overlay".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(180))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.label(format!("Display FPS: {:.1}", self.fps_ema));
                    });
            });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}
