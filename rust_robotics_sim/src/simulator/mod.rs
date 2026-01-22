pub mod localization;
pub mod pendulum;

use localization::ParticleFilter;
use pendulum::InvertedPendulum;

use egui::*;
use egui_plot::{Corner, Legend, Plot, PlotUi};

/// Base trait to make simulation work within `rust robotics`.
///
/// Users can implement this trait to make custom simulations.
pub trait Simulate {
    /// Getter method for internal state of an object that implements [`Simulate`]
    ///
    /// This method allows simulations of same type to communicate its internal
    /// state. The usecase for this method is when we want to align the initial
    /// conditions of multiple simulations, so that they can be compared with
    /// respect to each other throughout the simulation.
    fn get_state(&self) -> &dyn std::any::Any;

    /// Match the current simulation's state with that of another object, as long
    /// as it's state is compatible with the current simulation.
    fn match_state_with(&mut self, other: &dyn Simulate);

    /// Take a single step through simulation based on the given time delta
    fn step(&mut self, dt: f32);

    /// Reset the dynamic states of the current simulation object.
    ///
    /// Any dynamic states that get updated with [`Simulate::step`] should be
    /// reset to the default values using this method. Anything that is **not a
    /// dynamic state of the system (e.g. tunable parameters) should not be
    /// reset using this method.**
    fn reset_state(&mut self);

    /// Reset the dynamic states, as well as any other parameters into its default
    /// values
    ///
    /// This is a hard reset on the simulation, instead of restarting the
    /// simulation with same parameters.
    fn reset_all(&mut self);
}

/// Trait to allow visually representing simulation (simulation graphics + GUI)
pub trait Draw {
    /// Draw the simulation onto a 2D scene
    fn scene(&self, plot_ui: &mut PlotUi<'_>);
    /// Draw any GUI elements to interact with the simulation
    fn options(&mut self, ui: &mut Ui);
    /// Draw time-domain plot (optional)
    fn plot(&self, _plot_ui: &mut PlotUi<'_>) {}
}

/// Super-trait for objects which implement both [`Simulate`] and [`Draw`]
///
/// This trait is required in order to simulate and draw using [`egui`].
pub trait SimulateEgui: Simulate + Draw {
    /// A downcast method to access another simulation object as a generic [`Simulate`]
    /// object, instead of [`SimulateEgui`].
    ///
    /// The primary usecase for this method is for state synchronization between
    /// multiple simulations via [`Simulate::match_state_with`]
    fn as_base(&self) -> &dyn Simulate;
}

impl<T> SimulateEgui for T
where
    T: Simulate + Draw,
{
    fn as_base(&self) -> &dyn Simulate {
        self
    }
}

/// Available simulation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimMode {
    InvertedPendulum,
    Localization,
}

impl SimMode {
    fn label(&self) -> &'static str {
        match self {
            SimMode::InvertedPendulum => "Inverted Pendulum",
            SimMode::Localization => "Localization (Particle Filter)",
        }
    }
}

/// A concrete type for containing simulations and executing them
pub struct Simulator {
    /// Current simulation mode
    mode: SimMode,
    /// Simulations for inverted pendulum mode
    pendulums: Vec<InvertedPendulum>,
    /// Simulations for localization mode
    vehicles: Vec<ParticleFilter>,
    /// Current simulation time in seconds.
    time: f32,
    /// The speed with which to execute the simulation. This is actually a
    /// multiplier to indicate how many times to call [`step`](Simulate::step) when
    /// [`update`](Self::update) is called.
    sim_speed: usize,
    /// Settings to indicate whether to show the graph of simulation signals
    show_graph: bool,
    paused: bool,
}

impl Default for Simulator {
    fn default() -> Self {
        Self {
            mode: SimMode::InvertedPendulum,
            pendulums: vec![InvertedPendulum::default()],
            vehicles: vec![ParticleFilter::new(1, 0.0)],
            time: 0.0,
            sim_speed: 2,
            show_graph: false,
            paused: false,
        }
    }
}

impl Simulator {
    /// Update the simulation for a single time step
    pub fn update(&mut self) {
        if !self.paused {
            let dt = 0.01;
            self.time += dt * self.sim_speed as f32;

            match self.mode {
                SimMode::InvertedPendulum => {
                    self.pendulums
                        .iter_mut()
                        .for_each(|sim| (0..self.sim_speed).for_each(|_| sim.step(dt)));
                }
                SimMode::Localization => {
                    self.vehicles
                        .iter_mut()
                        .for_each(|sim| (0..self.sim_speed).for_each(|_| sim.step(dt)));
                }
            }
        }
    }

    /// Reset the states of all simulations within the current mode
    fn reset_state(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.pendulums.iter_mut().for_each(|sim| sim.reset_state());
                // Sync states
                if let Some((first, rest)) = self.pendulums.split_first_mut() {
                    rest.iter_mut()
                        .for_each(|sim| sim.match_state_with(first));
                }
            }
            SimMode::Localization => {
                self.vehicles.iter_mut().for_each(|sim| sim.reset_state());
            }
        }
    }

    /// Reset all simulations to default
    fn reset_all(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.pendulums.iter_mut().for_each(|sim| sim.reset_all());
            }
            SimMode::Localization => {
                self.vehicles.iter_mut().for_each(|sim| sim.reset_all());
            }
        }
    }

    /// Add a new simulation instance to the current mode
    fn add_simulation(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                let id = self.pendulums.len() + 1;
                self.pendulums.push(InvertedPendulum::new(id, self.time));
            }
            SimMode::Localization => {
                let id = self.vehicles.len() + 1;
                self.vehicles.push(ParticleFilter::new(id, self.time));
            }
        }
    }

    /// Draw the UI directly into a Ui (for embedding in CentralPanel)
    pub fn ui(&mut self, ui: &mut Ui) {
        // Mode selector at the top
        ui.horizontal(|ui| {
            ui.label("Simulation:");
            for mode in [SimMode::InvertedPendulum, SimMode::Localization] {
                if ui.selectable_label(self.mode == mode, mode.label()).clicked() {
                    self.mode = mode;
                    self.time = 0.0;
                }
            }
        });

        ui.separator();

        // Control buttons
        ui.horizontal(|ui| {
            let btn_text = if self.paused { "Play" } else { "Pause" };
            if ui.button(btn_text).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("Restart").clicked() {
                self.time = 0.0;
                self.reset_state();
            }
            if ui.button("Reset All").clicked() {
                self.time = 0.0;
                self.reset_all();
            }

            let add_label = match self.mode {
                SimMode::InvertedPendulum => "Add Pendulum",
                SimMode::Localization => "Add Vehicle",
            };
            if ui.button(add_label).clicked() {
                self.add_simulation();
            }

            ui.checkbox(&mut self.show_graph, "Show Graph");
        });

        ui.separator();

        // Options panel for current simulations
        ui.horizontal(|ui| {
            match self.mode {
                SimMode::InvertedPendulum => {
                    self.pendulums.iter_mut().for_each(|sim| sim.options(ui));
                }
                SimMode::Localization => {
                    self.vehicles.iter_mut().for_each(|sim| sim.options(ui));
                }
            }
        });

        ui.separator();

        // Instructions (collapsible)
        ui.collapsing("Instructions", |ui| {
            ui.label("Pan by dragging, or scroll (+ shift = horizontal).");
            ui.label("Box zooming: Right click to zoom in and zoom out using a selection.");
            if cfg!(target_arch = "wasm32") {
                ui.label("Zoom with ctrl / ⌘ + pointer wheel, or with pinch gesture.");
            } else if cfg!(target_os = "macos") {
                ui.label("Zoom with ctrl / ⌘ + scroll.");
            } else {
                ui.label("Zoom with ctrl + scroll.");
            }
            ui.label("Reset view with double-click.");
        });

        // Main scene plot
        let plot = Plot::new("Scene")
            .legend(Legend::default().position(Corner::RightTop))
            .show_x(false)
            .show_y(false)
            .data_aspect(1.0);

        plot.show(ui, |plot_ui| {
            match self.mode {
                SimMode::InvertedPendulum => {
                    self.pendulums.iter().for_each(|sim| sim.scene(plot_ui));
                }
                SimMode::Localization => {
                    self.vehicles.iter().for_each(|sim| sim.scene(plot_ui));
                }
            }
        });

        // Optional graph window
        if self.show_graph {
            egui::Window::new("Signal Plot")
                .default_size(vec2(400.0, 300.0))
                .show(ui.ctx(), |ui| {
                    Plot::new("Plot")
                        .legend(Legend::default().position(Corner::RightTop))
                        .show(ui, |plot_ui| {
                            match self.mode {
                                SimMode::InvertedPendulum => {
                                    self.pendulums.iter().for_each(|sim| sim.plot(plot_ui));
                                }
                                SimMode::Localization => {
                                    self.vehicles.iter().for_each(|sim| sim.plot(plot_ui));
                                }
                            }
                        });
                });
        }
    }
}
