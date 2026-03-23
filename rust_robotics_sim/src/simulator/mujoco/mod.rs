use egui::{Rect, Ui};
use egui_plot::PlotUi;

#[cfg(not(target_arch = "wasm32"))]
mod native;
mod render_scene;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
use native::NativeMujocoBackend as Backend;
#[cfg(target_arch = "wasm32")]
use wasm::WasmMujocoBackend as Backend;

pub struct MujocoPanel {
    backend: Backend,
}

impl Default for MujocoPanel {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
        }
    }
}

impl MujocoPanel {
    pub fn fixed_dt(&self) -> f32 {
        0.02
    }

    pub fn update(&mut self, sim_speed: usize, paused: bool) {
        self.backend.update(sim_speed, paused);
    }

    pub fn reset_state(&mut self) {
        self.backend.reset_state();
    }

    pub fn reset_all(&mut self) {
        self.backend.reset_all();
    }

    pub fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        self.backend.plot(plot_ui);
    }

    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.backend.ui(ui, frame);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_split(
        &mut self,
        controls_ui: &mut Ui,
        viewport_ui: &mut Ui,
        frame: Option<&eframe::Frame>,
    ) {
        self.backend.ui_split(controls_ui, viewport_ui, frame);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_controls(&mut self, ui: &mut Ui) {
        self.backend.ui_controls(ui);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.backend.ui_viewport(ui, frame);
    }

    pub fn set_active(&mut self, active: bool) {
        self.backend.set_active(active);
    }

    pub fn set_overlay_occlusions(&mut self, rects: &[Rect], interactive: bool) {
        self.backend.set_overlay_occlusions(rects, interactive);
    }
}
