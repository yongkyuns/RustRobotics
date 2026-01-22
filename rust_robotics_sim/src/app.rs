use crate::simulator::Simulator;

use eframe::egui;

#[derive(Default)]
pub struct App {
    sim: Simulator,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.sim.update();

        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.set_visuals(egui::Visuals::dark());
            self.sim.ui(ui);
        });

        ctx.request_repaint();
    }
}
