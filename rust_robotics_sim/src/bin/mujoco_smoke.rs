#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use eframe::egui::emath::GuiRounding;
#[cfg(not(target_arch = "wasm32"))]
use rust_robotics_sim::simulator::mujoco::MujocoPanel;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    tracing_subscriber::fmt::init();

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        ..eframe::NativeOptions::default()
    };

    let _ = eframe::run_native(
        "MuJoCo Smoke Test",
        native_options,
        Box::new(|_cc| Ok(Box::new(MujocoSmokeApp::default()))),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
struct MujocoSmokeApp {
    panel: MujocoPanel,
    paused: bool,
    sim_speed: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for MujocoSmokeApp {
    fn default() -> Self {
        let mut panel = MujocoPanel::default();
        panel.set_active(true);
        Self {
            panel,
            paused: false,
            sim_speed: 2,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl eframe::App for MujocoSmokeApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.panel.update(self.sim_speed, self.paused);

        ctx.set_visuals(egui::Visuals::dark());
        egui::TopBottomPanel::top("mujoco_smoke_controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.paused, "Paused");
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.sim_speed, 1..=20).show_value(true));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let full = ui.max_rect();
            let sidebar_width = 360.0;
            let gap = 12.0;

            let sidebar_rect = egui::Rect::from_min_max(
                full.min,
                egui::pos2((full.min.x + sidebar_width).min(full.max.x), full.max.y),
            );
            let viewport_outer_rect = egui::Rect::from_min_max(
                egui::pos2((sidebar_rect.max.x + gap).min(full.max.x), full.min.y),
                full.max,
            );

            ui.scope_builder(egui::UiBuilder::new().max_rect(sidebar_rect), |ui| {
                self.panel.ui_controls(ui);
            });
            ui.scope_builder(egui::UiBuilder::new().max_rect(viewport_outer_rect), |ui| {
                let frame_response = egui::Frame::canvas(ui.style())
                    .inner_margin(egui::Margin::same(4))
                    .show(ui, |ui| {
                        self.panel.ui_viewport(ui, Some(frame));
                    });
                ui.painter().rect_stroke(
                    frame_response.response.rect,
                    0.0,
                    ui.visuals().widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
            });

            let divider_rect = egui::Rect::from_min_max(
                egui::pos2(sidebar_rect.max.x, full.min.y),
                egui::pos2((sidebar_rect.max.x + gap).min(full.max.x), full.max.y),
            );
            ui.painter()
                .rect_filled(divider_rect, 0.0, ui.visuals().panel_fill);
            let separator_x = divider_rect
                .center()
                .x
                .round_to_pixel_center(ui.ctx().pixels_per_point());
            ui.painter().line_segment(
                [
                    egui::pos2(separator_x, divider_rect.top()),
                    egui::pos2(separator_x, divider_rect.bottom()),
                ],
                ui.visuals().widgets.noninteractive.bg_stroke,
            );
        });

        ctx.request_repaint();
    }
}
