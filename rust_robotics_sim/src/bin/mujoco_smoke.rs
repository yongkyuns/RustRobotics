#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use rust_robotics_sim::simulator::mujoco::MujocoPanel;

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

struct MujocoSmokeApp {
    panel: MujocoPanel,
    paused: bool,
    sim_speed: usize,
}

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

        egui::SidePanel::left("mujoco_smoke_sidebar")
            .resizable(false)
            .default_width(360.0)
            .min_width(360.0)
            .max_width(360.0)
            .show(ctx, |ui| {
                self.panel.ui_controls(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.panel.ui_viewport(ui, Some(frame));
        });

        ctx.request_repaint();
    }
}
