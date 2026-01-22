use crate::simulator::Simulator;
use crate::View;

use eframe::egui;

#[derive(Default)]
pub struct State {
    sim: Simulator,
}

#[derive(Default)]
pub struct App {
    state: State,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.state.sim.update();
        egui::CentralPanel::default().show(ctx, |_ui| {
            ctx.set_visuals(egui::Visuals::dark());
            self.state.sim.show(ctx, &mut true);
        });

        ctx.request_repaint();
    }
}
