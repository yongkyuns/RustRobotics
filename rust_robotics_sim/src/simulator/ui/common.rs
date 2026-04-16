//! Top-level shared UI layout for the simulator shell.
//!
//! This module owns the pieces of UI that are common regardless of which mode
//! is active:
//!
//! - shared keyboard shortcuts
//! - mode selector row
//! - playback / reset / comparison controls
//! - responsive sidebar + scene layout
//! - graph window orchestration
use super::super::{SimMode, Simulator};
use egui::*;
use egui_plot::{Corner, Legend, Plot};

impl Simulator {
    /// Clears the cached rectangles used by the help overlay before a new frame
    /// lays out the UI again.
    pub(in crate::simulator) fn reset_help_regions(&mut self) {
        self.help_state.help_mode_selector_rect = None;
        self.help_state.help_controls_rect = None;
        self.help_state.help_options_rect = None;
        self.help_state.help_scene_rect = None;
    }

    /// Handles shared keyboard shortcuts and forwards mode-specific input.
    pub(in crate::simulator) fn handle_ui_input(&mut self, ctx: &Context) {
        if ctx.input(|i| i.key_pressed(Key::Space)) {
            self.paused = !self.paused;
        }

        if ctx.input(|i| i.key_pressed(Key::Enter)) {
            self.restart();
        }

        match self.mode {
            SimMode::Localization => self.handle_localization_input(ctx),
            SimMode::Slam => self.handle_slam_input(ctx),
            _ => {}
        }
    }

    /// Renders the top row used to switch between simulator modes.
    pub(in crate::simulator) fn show_mode_selector_row(&mut self, ui: &mut Ui) {
        let mode_selector_response = ui.horizontal(|ui| {
            ui.label("Simulation:");
            for mode in [
                SimMode::InvertedPendulum,
                SimMode::Localization,
                SimMode::Mujoco,
                SimMode::PathPlanning,
                SimMode::Slam,
            ] {
                if ui
                    .selectable_label(self.mode == mode, mode.label())
                    .clicked()
                {
                    self.set_mode(mode);
                }
            }
        });
        self.help_state.help_mode_selector_rect = Some(mode_selector_response.response.rect);
    }

    /// Starts the first-time tutorial for a mode when appropriate.
    pub(in crate::simulator) fn maybe_start_tutorial_for_mode(&mut self) {
        if self.help_state.visited_modes.contains(&self.mode) {
            return;
        }

        match self.help_state.tutorial_enabled {
            Some(true) => {
                self.help_state.visited_modes.insert(self.mode);
                self.help_state.show_help_popup = true;
                self.reset_help_tour();
            }
            Some(false) => {}
            None => {
                self.help_state.show_tutorial_prompt = true;
            }
        }
    }

    /// Renders controls that are meaningful across all modes.
    pub(in crate::simulator) fn show_shared_controls_row(&mut self, ui: &mut Ui, display_fps: f32) {
        let controls_response = ui.horizontal(|ui| {
            let btn_text = if self.paused { "Play" } else { "Pause" };
            if ui.button(btn_text).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("Restart").clicked() {
                self.restart();
            }
            if ui.button("Reset All").clicked() {
                self.time = 0.0;
                self.reset_all();
            }

            if self.mode != SimMode::Slam && self.mode != SimMode::Mujoco {
                let add_label = match self.mode {
                    SimMode::InvertedPendulum => "Add Pendulum",
                    SimMode::Localization => "Add Vehicle",
                    SimMode::Mujoco => unreachable!(),
                    SimMode::PathPlanning => "Add Planner",
                    SimMode::Slam => unreachable!(),
                };
                if ui.button(add_label).clicked() {
                    self.add_simulation();
                }
            }

            ui.checkbox(&mut self.ui_state.show_graph, "Show Graph");

            ui.separator();
            ui.label("Speed:");
            ui.add(Slider::new(&mut self.sim_speed, 1..=20).show_value(true));

            if self.mode == SimMode::InvertedPendulum {
                ui.separator();
                ui.checkbox(&mut self.pendulum_noise_enabled, "Noise");
                ui.add_enabled(
                    self.pendulum_noise_enabled,
                    Slider::new(&mut self.pendulum_noise_scale, 0.0..=3.0)
                        .text("Magnitude")
                        .show_value(true),
                );
            }

            if self.mode == SimMode::Slam {
                if let Some(slam) = self.simulations.slam_demos.first_mut() {
                    ui.separator();
                    ui.label("Landmarks:");
                    let response =
                        ui.add(Slider::new(&mut slam.n_landmarks, 1..=50).show_value(true));
                    if response.drag_stopped() {
                        slam.regenerate_landmarks();
                    }
                }
            }

            ui.separator();
            ui.label(
                RichText::new(format!("Display FPS: {:.1}", display_fps))
                    .strong()
                    .color(Color32::from_rgb(238, 243, 249)),
            );
        });
        self.help_state.help_controls_rect = Some(controls_response.response.rect);
    }

    /// Builds the responsive two-pane layout for options and the scene view.
    pub(in crate::simulator) fn show_main_content(
        &mut self,
        ui: &mut Ui,
        frame: Option<&eframe::Frame>,
    ) {
        let content_height = ui.available_height().max(360.0);
        let total_width = ui.available_width();
        let is_phone = total_width < 760.0;
        let is_tablet = !is_phone && total_width < 1100.0;
        let cards_vertical = is_phone;
        let min_viewport_width = if is_tablet {
            (total_width * 0.45).max(340.0)
        } else {
            (total_width * 0.5).max(360.0)
        };
        let max_sidebar_width = (total_width - min_viewport_width).max(280.0);
        let sidebar_width = if is_phone {
            total_width
        } else {
            (total_width * if is_tablet { 0.42 } else { 0.34 }).clamp(280.0, max_sidebar_width)
        };

        if is_phone {
            let options_height = (content_height * 0.45).clamp(260.0, 420.0);
            let scene_height = (content_height - options_height - 14.0).max(280.0);

            ui.vertical(|ui| {
                let options_response = ui.allocate_ui_with_layout(
                    vec2(total_width, options_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_sidebar(ui, cards_vertical),
                );
                self.help_state.help_options_rect = Some(options_response.response.rect);

                ui.add_space(14.0);

                ui.allocate_ui_with_layout(
                    vec2(total_width, scene_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_scene_pane(ui, frame, false),
                );
            });
        } else {
            ui.horizontal_top(|ui| {
                let options_response = ui.allocate_ui_with_layout(
                    vec2(sidebar_width, content_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_sidebar(ui, cards_vertical),
                );
                self.help_state.help_options_rect = Some(options_response.response.rect);

                ui.add_space(14.0);

                ui.allocate_ui_with_layout(
                    vec2(ui.available_width(), content_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_scene_pane(ui, frame, false),
                );
            });
        }
    }

    /// Builds the focused tutorial/embed layout with controls across the top
    /// and the main visualization below.
    pub(in crate::simulator) fn show_embedded_content(
        &mut self,
        ui: &mut Ui,
        frame: Option<&eframe::Frame>,
        display_fps: f32,
    ) {
        let total_width = ui.available_width();
        let is_narrow = total_width < 900.0;
        let controls_max_height = match self.mode {
            SimMode::Mujoco => {
                if is_narrow {
                    (total_width * 0.42).clamp(220.0, 360.0)
                } else {
                    (total_width * 0.24).clamp(200.0, 300.0)
                }
            }
            SimMode::PathPlanning | SimMode::Slam => {
                if is_narrow {
                    (total_width * 0.36).clamp(190.0, 280.0)
                } else {
                    (total_width * 0.2).clamp(170.0, 240.0)
                }
            }
            SimMode::InvertedPendulum | SimMode::Localization => {
                if is_narrow {
                    (total_width * 0.22).clamp(130.0, 180.0)
                } else {
                    (total_width * 0.12).clamp(120.0, 160.0)
                }
            }
        };
        let scene_height = match self.mode {
            SimMode::Mujoco => {
                if is_narrow {
                    (total_width * 0.9).clamp(360.0, 580.0)
                } else {
                    (total_width * 0.58).clamp(360.0, 520.0)
                }
            }
            SimMode::InvertedPendulum => {
                if is_narrow {
                    (total_width * 0.28).clamp(180.0, 240.0)
                } else {
                    (total_width * 0.18).clamp(160.0, 220.0)
                }
            }
            SimMode::Localization | SimMode::PathPlanning | SimMode::Slam => {
                if is_narrow {
                    (total_width * 0.72).clamp(320.0, 460.0)
                } else {
                    (total_width * 0.48).clamp(320.0, 420.0)
                }
            }
        };

        let controls_response = egui::Frame::NONE
            .inner_margin(Margin {
                left: 8,
                right: 8,
                top: 12,
                bottom: 8,
            })
            .show(ui, |ui| {
                self.show_embedded_shared_controls_row(ui, display_fps);
                ui.add_space(6.0);

                ScrollArea::vertical()
                    .max_height((controls_max_height - 48.0).max(120.0))
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.show_embedded_controls_panel(ui))
            });
        self.help_state.help_options_rect = Some(controls_response.response.rect);

        ui.add_space(4.0);

        ui.allocate_ui_with_layout(
            vec2(total_width, scene_height),
            Layout::top_down(Align::Min),
            |ui| {
                egui::Frame::group(ui.style())
                    .inner_margin(Margin::same(6))
                    .show(ui, |ui| self.show_scene_pane(ui, frame, true));
            },
        );
    }

    fn show_embedded_shared_controls_row(&mut self, ui: &mut Ui, display_fps: f32) {
        ui.horizontal_wrapped(|ui| {
            let btn_text = if self.paused { "Play" } else { "Pause" };
            if ui.button(btn_text).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("Restart").clicked() {
                self.restart();
            }
            if ui.button("Reset All").clicked() {
                self.time = 0.0;
                self.reset_all();
            }

            if self.mode != SimMode::Slam && self.mode != SimMode::Mujoco {
                let add_label = match self.mode {
                    SimMode::InvertedPendulum => "Add Pendulum",
                    SimMode::Localization => "Add Vehicle",
                    SimMode::Mujoco => unreachable!(),
                    SimMode::PathPlanning => "Add Planner",
                    SimMode::Slam => unreachable!(),
                };
                if ui.button(add_label).clicked() {
                    self.add_simulation();
                }
            }

            ui.separator();
            ui.label("Speed:");
            ui.add(Slider::new(&mut self.sim_speed, 1..=20).show_value(true));

            if self.mode == SimMode::InvertedPendulum {
                ui.separator();
                ui.checkbox(&mut self.pendulum_noise_enabled, "Noise");
                ui.add_enabled(
                    self.pendulum_noise_enabled,
                    Slider::new(&mut self.pendulum_noise_scale, 0.0..=3.0)
                        .text("Magnitude")
                        .show_value(true),
                );
            }

            if self.mode == SimMode::Slam {
                if let Some(slam) = self.simulations.slam_demos.first_mut() {
                    ui.separator();
                    ui.label("Landmarks:");
                    let response =
                        ui.add(Slider::new(&mut slam.n_landmarks, 1..=50).show_value(true));
                    if response.drag_stopped() {
                        slam.regenerate_landmarks();
                    }
                }
            }

            ui.separator();
            ui.label(
                RichText::new(format!("FPS: {:.1}", display_fps))
                    .strong()
                    .color(Color32::from_rgb(238, 243, 249)),
            );
        });
    }

    fn show_embedded_controls_panel(&mut self, ui: &mut Ui) {
        let use_vertical_cards = matches!(
            self.mode,
            SimMode::InvertedPendulum | SimMode::Localization | SimMode::Slam
        );
        self.show_option_cards(ui, use_vertical_cards);

        let show_keyboard_controls = (self.mode == SimMode::Localization
            && self
                .simulations
                .vehicles
                .iter()
                .any(|vehicle| vehicle.is_dynamic_mode()))
            || (self.mode == SimMode::Slam
                && self
                    .simulations
                    .slam_demos
                    .iter()
                    .any(|slam| slam.is_manual_mode()));

        if show_keyboard_controls {
            ui.add_space(6.0);
            ui.collapsing("Keyboard Controls", |ui| match self.mode {
                SimMode::Localization => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("← → : Steering");
                        ui.label("↑ ↓ : Accelerate / Brake");
                        ui.label("Space : Pause");
                        ui.label("Enter : Restart");
                    });
                }
                SimMode::Slam => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("← → : Turn Left / Right");
                        ui.label("↑ ↓ : Speed Up / Slow Down");
                        ui.label("Space : Pause");
                        ui.label("Enter : Restart");
                    });
                }
                _ => {}
            });
        }

        ui.add_space(6.0);
        ui.collapsing("Instructions", |ui| {
            self.show_mode_instructions(ui);
        });
    }

    /// Renders the sidebar containing option cards, contextual keyboard help,
    /// and mode-specific instructions.
    pub(in crate::simulator) fn show_sidebar(&mut self, ui: &mut Ui, cards_vertical: bool) {
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.show_option_cards(ui, cards_vertical);

                if self.mode == SimMode::Localization
                    && self
                        .simulations
                        .vehicles
                        .iter()
                        .any(|v| v.is_dynamic_mode())
                {
                    ui.collapsing("Keyboard Controls", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("← → : Steering");
                            ui.label("   ↑ ↓ : Accelerate/Brake");
                            ui.label("   Space : Pause");
                            ui.label("   Enter : Restart");
                        });
                    });
                }

                if self.mode == SimMode::Slam
                    && self
                        .simulations
                        .slam_demos
                        .iter()
                        .any(|s| s.is_manual_mode())
                {
                    ui.collapsing("Keyboard Controls", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("← → : Turn Left/Right");
                            ui.label("   ↑ ↓ : Speed Up/Slow Down");
                            ui.label("   Space : Pause");
                            ui.label("   Enter : Restart");
                        });
                    });
                }

                ui.separator();

                ui.collapsing("Instructions", |ui| {
                    self.show_mode_instructions(ui);

                    ui.add_space(5.0);
                    ui.label(RichText::new("Navigation").strong());
                    ui.label("• Pan by dragging, or scroll (+ shift = horizontal).");
                    ui.label(
                        "• Box zooming: Right click to zoom in and zoom out using a selection.",
                    );
                    if cfg!(target_arch = "wasm32") {
                        ui.label("• Zoom with ctrl / ⌘ + pointer wheel, or with pinch gesture.");
                    } else if cfg!(target_os = "macos") {
                        ui.label("• Zoom with ctrl / ⌘ + scroll.");
                    } else {
                        ui.label("• Zoom with ctrl + scroll.");
                    }
                    ui.label("• Reset view with double-click.");
                });
            });
    }

    /// Renders the main visualization pane for the active mode.
    pub(in crate::simulator) fn show_scene_pane(
        &mut self,
        ui: &mut Ui,
        frame: Option<&eframe::Frame>,
        embedded: bool,
    ) {
        if self.mode == SimMode::Mujoco {
            self.simulations.mujoco_panel.ui_viewport(ui, frame);
            self.help_state.help_scene_rect = Some(ui.min_rect());
        } else if self.mode == SimMode::InvertedPendulum {
            self.help_state.help_scene_rect = Some(self.render_pendulum_scene(ui, embedded));
        } else {
            let mut plot = Plot::new("Scene")
                .legend(Legend::default().position(Corner::RightTop))
                .show_x(false)
                .show_y(false)
                .data_aspect(1.0)
                .allow_boxed_zoom(self.mode != SimMode::PathPlanning);

            if self.mode == SimMode::PathPlanning {
                plot = plot
                    .show_grid(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false);
            }

            let plot_response = plot.show(ui, |plot_ui| self.draw_mode_scene_plot(plot_ui));

            if self.mode == SimMode::PathPlanning {
                self.sync_path_planning_scene_interaction(&plot_response);
            }
            self.help_state.help_scene_rect = Some(plot_response.response.rect);
        }
    }

    /// Renders the detached graph window when the user enables it.
    pub(in crate::simulator) fn show_graph_window(&mut self, ctx: &Context) -> Option<Rect> {
        Window::new("Signal Plot")
            .default_size(vec2(400.0, 300.0))
            .show(ctx, |ui| self.show_mode_graph_contents(ui))
            .map(|window| window.response.rect)
    }
}
