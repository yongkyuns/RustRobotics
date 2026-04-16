use egui::{Rect, Ui};
use egui_plot::PlotUi;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;

#[cfg(not(target_arch = "wasm32"))]
mod native;
mod render_scene;
mod shared_layout;
mod shared_panel;
mod viewport_interaction;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
use native::NativeMujocoBackend as Backend;
#[cfg(target_arch = "wasm32")]
use wasm::WasmMujocoBackend as Backend;

pub struct MujocoPanel {
    backend: Backend,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct MujocoEmbedState {
    pub(crate) selected_robot: String,
    pub(crate) robot_label: String,
    pub(crate) policy_label: String,
    pub(crate) status: String,
    pub(crate) ready: bool,
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

    pub fn ui_split(
        &mut self,
        controls_ui: &mut Ui,
        viewport_ui: &mut Ui,
        frame: Option<&eframe::Frame>,
    ) {
        self.backend.ui_split(controls_ui, viewport_ui, frame);
    }

    pub fn ui_controls(&mut self, ui: &mut Ui) {
        self.backend.ui_controls(ui);
    }

    pub fn ui_viewport(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        self.backend.ui_viewport(ui, frame);
    }

    pub fn set_active(&mut self, active: bool) {
        self.backend.set_active(active);
    }

    pub fn set_overlay_occlusions(&mut self, rects: &[Rect], interactive: bool) {
        self.backend.set_overlay_occlusions(rects, interactive);
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn embed_state(&self) -> MujocoEmbedState {
        self.backend.embed_state()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn set_embed_robot(&mut self, robot: &str) {
        self.backend.set_embed_robot(robot);
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn reset_embed_view(&mut self) {
        self.backend.reset_embed_view();
    }
}
