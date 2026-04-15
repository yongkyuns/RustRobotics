//! Mode-specific UI helpers used by the shared simulator shell.
//!
//! The goal of this module is to keep `ui::common` focused on layout while this
//! file owns the branching logic that differs per simulator mode.
use super::super::{Draw, SimMode, Simulate, Simulator};
use crate::simulator::path_planning::EnvironmentMode;
use egui::*;
use egui_plot::{Corner, Legend, Plot};

impl Simulator {
    /// Forwards keyboard input to localization demos that support manual control.
    pub(super) fn handle_localization_input(&mut self, ctx: &Context) {
        for vehicle in &mut self.simulations.vehicles {
            vehicle.handle_keyboard(ctx);
        }
    }

    /// Forwards keyboard input to SLAM demos that support manual control.
    pub(super) fn handle_slam_input(&mut self, ctx: &Context) {
        for slam in &mut self.simulations.slam_demos {
            slam.handle_keyboard(ctx);
        }
    }

    /// Dispatches the mode-specific option-card rendering path.
    pub(super) fn show_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        match self.mode {
            SimMode::InvertedPendulum => self.show_pendulum_option_cards(ui, cards_vertical),
            SimMode::Localization => self.show_localization_option_cards(ui, cards_vertical),
            SimMode::Mujoco => self.show_mujoco_option_cards(ui),
            SimMode::PathPlanning => self.show_path_planning_option_cards(ui, cards_vertical),
            SimMode::Slam => self.show_slam_option_cards(ui, cards_vertical),
        }
    }

    fn show_pendulum_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        if cards_vertical {
            ui.vertical(|ui| {
                self.simulations
                    .pendulums
                    .retain_mut(|sim| sim.options_with_policy(ui));
            });
        } else {
            ScrollArea::horizontal()
                .id_salt("pendulum_cards")
                .max_height(300.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        self.simulations
                            .pendulums
                            .retain_mut(|sim| sim.options_with_policy(ui));
                    });
                });
        }
    }

    fn show_localization_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        if cards_vertical {
            ui.vertical(|ui| {
                self.simulations.vehicles.retain_mut(|sim| sim.options(ui));
            });
        } else {
            ScrollArea::horizontal()
                .id_salt("localization_cards")
                .max_height(320.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        self.simulations.vehicles.retain_mut(|sim| sim.options(ui));
                    });
                });
        }
    }

    fn show_mujoco_option_cards(&mut self, ui: &mut Ui) {
        self.simulations.mujoco_panel.ui_controls(ui);
    }

    fn show_path_planning_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Env:");
            if ui
                .selectable_label(self.path_settings.env_mode == EnvironmentMode::Grid, "Grid")
                .clicked()
            {
                self.path_settings.env_mode = EnvironmentMode::Grid;
            }
            if ui
                .selectable_label(
                    self.path_settings.env_mode == EnvironmentMode::Continuous,
                    "Continuous",
                )
                .clicked()
            {
                self.path_settings.env_mode = EnvironmentMode::Continuous;
            }

            if self.path_settings.env_mode == EnvironmentMode::Continuous {
                ui.label("Obs Radius:");
                ui.add(
                    DragValue::new(&mut self.path_settings.continuous_obstacle_radius)
                        .range(0.1..=5.0)
                        .speed(0.1),
                );
            }
        });

        if self.path_settings.env_mode == EnvironmentMode::Grid {
            ui.horizontal_wrapped(|ui| {
                ui.label("Grid:");
                ui.label("Width:");
                let width_changed = ui
                    .add(DragValue::new(&mut self.path_settings.grid_width).range(10..=100))
                    .changed();
                ui.label("Height:");
                let height_changed = ui
                    .add(DragValue::new(&mut self.path_settings.grid_height).range(10..=100))
                    .changed();
                ui.label("Resolution:");
                let res_changed = ui
                    .add(
                        DragValue::new(&mut self.path_settings.grid_resolution)
                            .range(0.1..=5.0)
                            .speed(0.1),
                    )
                    .changed();

                for planner in &mut self.simulations.planners {
                    if width_changed || height_changed || res_changed {
                        planner.update_grid_settings(
                            self.path_settings.grid_width,
                            self.path_settings.grid_height,
                            self.path_settings.grid_resolution,
                        );
                    }
                }
            });
        }

        for planner in &mut self.simulations.planners {
            planner.set_env_mode(self.path_settings.env_mode);
            planner.set_continuous_obstacle_radius(self.path_settings.continuous_obstacle_radius);
        }

        ui.separator();

        if cards_vertical {
            ui.vertical(|ui| {
                self.simulations.planners.retain_mut(|sim| sim.options(ui));
            });
        } else {
            ScrollArea::horizontal()
                .id_salt("path_planning_cards")
                .max_height(320.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        self.simulations.planners.retain_mut(|sim| sim.options(ui));
                    });
                });
        }
    }

    fn show_slam_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        if cards_vertical {
            ui.vertical(|ui| {
                self.simulations
                    .slam_demos
                    .retain_mut(|sim| sim.options(ui));
            });
        } else {
            ScrollArea::horizontal()
                .id_salt("slam_cards")
                .max_height(260.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        self.simulations
                            .slam_demos
                            .retain_mut(|sim| sim.options(ui));
                    });
                });
        }
    }

    /// Renders the instructional text block for the current mode.
    pub(super) fn show_mode_instructions(&self, ui: &mut Ui) {
        match self.mode {
            SimMode::PathPlanning => {
                ui.label(RichText::new("1. Start & Goal").strong());
                ui.label("• Left-click on the map to set the Start point (Green).");
                ui.label("• Left-click again to set the Goal point (Red).");
                ui.label("• Once both are set, all active planners will run automatically.");

                ui.add_space(5.0);
                ui.label(RichText::new("2. Environment").strong());
                if self.path_settings.env_mode == EnvironmentMode::Grid {
                    ui.label("• Grid Mode: Discrete cells.");
                    ui.label("• Right-click (or drag): Paint/Erase obstacle cells.");
                } else {
                    ui.label("• Continuous Mode: Free space.");
                    ui.label("• Right-click: Add a circular obstacle at cursor.");
                    ui.label("• Right-click on existing obstacle: Remove it.");
                }

                ui.add_space(5.0);
                ui.label(RichText::new("3. Planners").strong());
                ui.label("• Click 'Add Planner' (top) to compare multiple algorithms.");
                ui.label("• Select different algorithms (A*, RRT, Theta*) for each planner.");
                ui.label("• Compare Path Length, Execution Time, and Optimality Ratio.");
            }
            SimMode::Localization | SimMode::Slam => {
                ui.label("• Use Keyboard arrows to drive.");
            }
            SimMode::Mujoco => {
                ui.label(
                    "• The MuJoCo tab runs the native MuJoCo model and ONNX policy inside Rust.",
                );
                ui.label("• The viewport uses the shared Rust-owned 3D renderer path across native and web.");
            }
            SimMode::InvertedPendulum => {}
        }
    }

    /// Draws the active mode's main scene into the shared plot surface.
    pub(super) fn draw_mode_scene_plot(&self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.simulations
                    .pendulums
                    .iter()
                    .for_each(|sim| sim.scene(plot_ui));
            }
            SimMode::Localization => {
                self.simulations
                    .vehicles
                    .iter()
                    .for_each(|sim| sim.scene(plot_ui));
            }
            SimMode::Slam => {
                self.simulations
                    .slam_demos
                    .iter()
                    .for_each(|sim| sim.scene(plot_ui));
            }
            SimMode::PathPlanning => {
                self.simulations
                    .planners
                    .iter()
                    .for_each(|sim| sim.scene(plot_ui));
            }
            SimMode::Mujoco => unreachable!(),
        }
    }

    pub(super) fn sync_path_planning_scene_interaction(
        &mut self,
        plot_response: &egui_plot::PlotResponse<()>,
    ) {
        if let Some((first, rest)) = self.simulations.planners.split_first_mut() {
            first.handle_mouse(plot_response);
            rest.iter_mut().for_each(|sim| sim.match_state_with(first));
        }
    }

    pub(super) fn show_mode_graph_contents(&mut self, ui: &mut Ui) {
        match self.mode {
            SimMode::InvertedPendulum => self.pendulum_plot_ui(ui),
            SimMode::Localization => {
                Plot::new("Plot")
                    .legend(Legend::default().position(Corner::RightTop))
                    .show(ui, |plot_ui| {
                        self.simulations
                            .vehicles
                            .iter()
                            .for_each(|sim| sim.plot(plot_ui));
                    });
            }
            SimMode::PathPlanning => {
                Plot::new("Plot")
                    .legend(Legend::default().position(Corner::RightTop))
                    .show(ui, |plot_ui| {
                        self.simulations
                            .planners
                            .iter()
                            .for_each(|sim| sim.plot(plot_ui));
                    });
            }
            SimMode::Slam => {
                Plot::new("Plot")
                    .legend(Legend::default().position(Corner::RightTop))
                    .show(ui, |plot_ui| {
                        self.simulations
                            .slam_demos
                            .iter()
                            .for_each(|sim| sim.plot(plot_ui));
                    });
            }
            SimMode::Mujoco => {
                Plot::new("Plot")
                    .legend(Legend::default().position(Corner::RightTop))
                    .show(ui, |plot_ui| {
                        self.simulations.mujoco_panel.plot(plot_ui);
                    });
            }
        }
    }
}
