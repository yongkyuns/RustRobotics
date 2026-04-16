//! Top-level simulator orchestration for the Rust Robotics application.
//!
//! This module defines the public simulator traits and the `Simulator`
//! coordinator that stitches all interactive demos together. The internal split
//! is responsibility-driven:
//!
//! - `runtime`: stepping, reset, pause, mode switching, timebase management
//! - `help`: guided tutorial overlays and highlighted UI regions
//! - `ui`: shared UI layout plus mode-specific panels
//! - `pendulum`, `localization`, `path_planning`, `slam`, `mujoco`: concrete
//!   mode implementations
//!
//! The central design rule is that `Simulator` owns cross-mode application
//! state, while each mode owns its own domain logic.
pub mod common;
mod help;
pub mod localization;
pub mod mujoco;
pub mod path_planning;
pub mod pendulum;
mod ppo_trainer;
mod runtime;
pub mod slam;
mod ui;

use localization::ParticleFilter;
#[cfg(target_arch = "wasm32")]
pub(crate) use localization::{
    DriveMode as LocalizationDriveMode, LocalizationCardState, LocalizationPatch,
};
#[cfg(target_arch = "wasm32")]
pub(crate) use mujoco::MujocoEmbedState;
use mujoco::MujocoPanel;
#[cfg(target_arch = "wasm32")]
pub(crate) use path_planning::{
    Algorithm as PathPlanningAlgorithm, PathPlannerCardState, PathPlannerPatch,
};
use path_planning::{EnvironmentMode, PathPlanning};
use pendulum::{InvertedPendulum, NoiseConfig as PendulumNoiseConfig, PENDULUM_FIXED_DT};
#[cfg(target_arch = "wasm32")]
pub(crate) use pendulum::{PendulumCardState, PendulumPatch};
use slam::SlamDemo;
#[cfg(target_arch = "wasm32")]
pub(crate) use slam::{DriveMode as SlamDriveMode, SlamCardState, SlamPatch};
use ui::PendulumPlotTab;

use egui::*;
use egui_plot::PlotUi;
use std::collections::HashSet;
use web_time::Instant;

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
    /// Returns true to keep the simulation, false to remove it
    fn options(&mut self, ui: &mut Ui) -> bool;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimMode {
    InvertedPendulum,
    Localization,
    Mujoco,
    PathPlanning,
    Slam,
}

impl SimMode {
    fn label(&self) -> &'static str {
        match self {
            SimMode::InvertedPendulum => "Inverted Pendulum",
            SimMode::Localization => "Localization",
            SimMode::Mujoco => "Robot",
            SimMode::PathPlanning => "Path Planning",
            SimMode::Slam => "SLAM",
        }
    }

    pub fn test_id(self) -> &'static str {
        match self {
            SimMode::InvertedPendulum => "inverted_pendulum",
            SimMode::Localization => "localization",
            SimMode::Mujoco => "robot",
            SimMode::PathPlanning => "path_planning",
            SimMode::Slam => "slam",
        }
    }

    pub fn from_test_id(value: &str) -> Option<Self> {
        match value {
            "inverted_pendulum" => Some(Self::InvertedPendulum),
            "localization" => Some(Self::Localization),
            "robot" => Some(Self::Mujoco),
            "path_planning" => Some(Self::PathPlanning),
            "slam" => Some(Self::Slam),
            _ => None,
        }
    }
}

/// Width-driven layout contract for documentation/tutorial embeds.
///
/// The full desktop app is allowed to be viewport-driven because it owns the
/// whole window. The focused docs embed is different: the article owns the
/// width, while the simulator should expose a natural intrinsic height for a
/// compact "teaching widget". This spec captures those per-mode defaults.
#[derive(Clone, Copy)]
pub(crate) struct EmbedLayoutSpec {
    pub(crate) controls_max_height: f32,
    pub(crate) scene_height: f32,
    pub(crate) controls_margin: Margin,
    pub(crate) controls_gap: f32,
    pub(crate) scene_gap: f32,
    pub(crate) scene_margin: i8,
    pub(crate) cards_vertical: bool,
    pub(crate) collapse_keyboard_controls: bool,
    pub(crate) collapse_instructions: bool,
}

impl SimMode {
    pub(crate) fn embed_layout_spec(self, width: f32) -> EmbedLayoutSpec {
        let is_narrow = width < 900.0;
        match self {
            SimMode::InvertedPendulum => EmbedLayoutSpec {
                controls_max_height: if is_narrow {
                    (width * 0.22).clamp(130.0, 180.0)
                } else {
                    (width * 0.12).clamp(120.0, 160.0)
                },
                scene_height: if is_narrow {
                    (width * 0.34).clamp(220.0, 320.0)
                } else {
                    (width * 0.22).clamp(200.0, 280.0)
                },
                controls_margin: Margin {
                    left: 8,
                    right: 8,
                    top: 12,
                    bottom: 8,
                },
                controls_gap: 6.0,
                scene_gap: 4.0,
                scene_margin: 6,
                cards_vertical: true,
                collapse_keyboard_controls: true,
                collapse_instructions: true,
            },
            SimMode::Localization => EmbedLayoutSpec {
                controls_max_height: if is_narrow {
                    (width * 0.24).clamp(150.0, 220.0)
                } else {
                    (width * 0.14).clamp(130.0, 190.0)
                },
                scene_height: if is_narrow {
                    (width * 0.58).clamp(260.0, 360.0)
                } else {
                    (width * 0.38).clamp(260.0, 340.0)
                },
                controls_margin: Margin {
                    left: 8,
                    right: 8,
                    top: 12,
                    bottom: 8,
                },
                controls_gap: 6.0,
                scene_gap: 6.0,
                scene_margin: 6,
                cards_vertical: true,
                collapse_keyboard_controls: true,
                collapse_instructions: true,
            },
            SimMode::PathPlanning => EmbedLayoutSpec {
                controls_max_height: if is_narrow {
                    (width * 0.30).clamp(170.0, 250.0)
                } else {
                    (width * 0.18).clamp(150.0, 220.0)
                },
                scene_height: if is_narrow {
                    (width * 0.70).clamp(300.0, 420.0)
                } else {
                    (width * 0.46).clamp(300.0, 400.0)
                },
                controls_margin: Margin {
                    left: 8,
                    right: 8,
                    top: 12,
                    bottom: 8,
                },
                controls_gap: 8.0,
                scene_gap: 8.0,
                scene_margin: 6,
                cards_vertical: false,
                collapse_keyboard_controls: true,
                collapse_instructions: true,
            },
            SimMode::Slam => EmbedLayoutSpec {
                controls_max_height: if is_narrow {
                    (width * 0.28).clamp(160.0, 240.0)
                } else {
                    (width * 0.18).clamp(150.0, 220.0)
                },
                scene_height: if is_narrow {
                    (width * 0.72).clamp(300.0, 430.0)
                } else {
                    (width * 0.48).clamp(300.0, 410.0)
                },
                controls_margin: Margin {
                    left: 8,
                    right: 8,
                    top: 12,
                    bottom: 8,
                },
                controls_gap: 8.0,
                scene_gap: 8.0,
                scene_margin: 6,
                cards_vertical: true,
                collapse_keyboard_controls: true,
                collapse_instructions: true,
            },
            SimMode::Mujoco => EmbedLayoutSpec {
                controls_max_height: if is_narrow {
                    (width * 0.40).clamp(220.0, 340.0)
                } else {
                    (width * 0.24).clamp(200.0, 300.0)
                },
                scene_height: if is_narrow {
                    (width * 0.90).clamp(360.0, 560.0)
                } else {
                    (width * 0.58).clamp(360.0, 520.0)
                },
                controls_margin: Margin {
                    left: 10,
                    right: 10,
                    top: 12,
                    bottom: 8,
                },
                controls_gap: 8.0,
                scene_gap: 8.0,
                scene_margin: 6,
                cards_vertical: false,
                collapse_keyboard_controls: true,
                collapse_instructions: true,
            },
        }
    }
}

/// Shared UI state that is independent of a specific simulation mode.
struct SharedUiState {
    show_graph: bool,
    pendulum_plot_tab: PendulumPlotTab,
}

impl Default for SharedUiState {
    fn default() -> Self {
        Self {
            show_graph: false,
            pendulum_plot_tab: PendulumPlotTab::LateralPosition,
        }
    }
}

/// Shared planner/environment settings applied to newly created planning demos.
///
/// These values live at the simulator level so multiple planners can be kept in
/// sync while still allowing each planner instance to own its own search state.
struct PathPlanningSettings {
    grid_width: usize,
    grid_height: usize,
    grid_resolution: f32,
    env_mode: EnvironmentMode,
    continuous_obstacle_radius: f32,
}

impl Default for PathPlanningSettings {
    fn default() -> Self {
        Self {
            grid_width: 40,
            grid_height: 40,
            grid_resolution: 1.0,
            env_mode: EnvironmentMode::Grid,
            continuous_obstacle_radius: 1.0,
        }
    }
}

/// State required to drive the guided help tour and highlight overlays.
struct HelpUiState {
    visited_modes: HashSet<SimMode>,
    tutorial_enabled: Option<bool>,
    show_tutorial_prompt: bool,
    show_help_popup: bool,
    shared_help_intro_completed: bool,
    help_step_index: usize,
    help_mode_selector_rect: Option<Rect>,
    help_controls_rect: Option<Rect>,
    help_options_rect: Option<Rect>,
    help_scene_rect: Option<Rect>,
}

impl Default for HelpUiState {
    fn default() -> Self {
        Self {
            visited_modes: HashSet::new(),
            tutorial_enabled: None,
            show_tutorial_prompt: false,
            show_help_popup: false,
            shared_help_intro_completed: false,
            help_step_index: 0,
            help_mode_selector_rect: None,
            help_controls_rect: None,
            help_options_rect: None,
            help_scene_rect: None,
        }
    }
}

/// Per-mode simulation collections owned by the app coordinator.
///
/// Grouping these together keeps the top-level `Simulator` from becoming a flat
/// bag of unrelated vectors and makes the distinction between "global app state"
/// and "per-mode simulation instances" explicit.
struct SimulationCollections {
    pendulums: Vec<InvertedPendulum>,
    vehicles: Vec<ParticleFilter>,
    planners: Vec<PathPlanning>,
    slam_demos: Vec<SlamDemo>,
    mujoco_panel: MujocoPanel,
}

impl Default for SimulationCollections {
    fn default() -> Self {
        Self {
            pendulums: vec![InvertedPendulum::default()],
            vehicles: vec![ParticleFilter::new(1, 0.0)],
            planners: vec![PathPlanning::new(1, 0.0)],
            slam_demos: vec![SlamDemo::new(1, 0.0)],
            mujoco_panel: MujocoPanel::default(),
        }
    }
}

/// A concrete type for containing simulations and executing them
pub struct Simulator {
    /// Current simulation mode
    mode: SimMode,
    /// Per-mode simulation collections and MuJoCo panel state.
    simulations: SimulationCollections,
    /// Current simulation time in seconds.
    time: f32,
    /// Fixed-step simulation speed multiplier relative to wall-clock time.
    /// A value of 1 advances one simulation step of `dt` for each `dt` of
    /// accumulated wall-clock time on average.
    sim_speed: usize,
    /// Shared UI state across simulator modes.
    ui_state: SharedUiState,
    paused: bool,
    /// Shared planner/environment settings for path planning mode.
    path_settings: PathPlanningSettings,
    /// Whether measurement/action noise is enabled for inverted pendulum mode.
    pendulum_noise_enabled: bool,
    /// Dimensionless scale factor applied to the inverted pendulum noise profile.
    pendulum_noise_scale: f32,
    /// Guided tutorial/help overlay state.
    help_state: HelpUiState,
    /// Last wall-clock tick used for fixed-step simulation pacing.
    last_tick: Option<Instant>,
    /// Accumulator for non-MuJoCo fixed-step simulation.
    sim_accumulator: f32,
}

impl Default for Simulator {
    fn default() -> Self {
        Self {
            mode: SimMode::InvertedPendulum,
            simulations: SimulationCollections::default(),
            time: 0.0,
            sim_speed: 1,
            ui_state: SharedUiState::default(),
            paused: false,
            path_settings: PathPlanningSettings::default(),
            pendulum_noise_enabled: false,
            pendulum_noise_scale: 0.0,
            help_state: HelpUiState::default(),
            last_tick: None,
            sim_accumulator: 0.0,
        }
    }
}

impl Simulator {
    /// Draw the UI directly into a Ui (for embedding in CentralPanel)
    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>, display_fps: f32) {
        self.sync_mujoco_overlay_active();
        self.reset_help_regions();
        self.handle_ui_input(ui.ctx());
        self.show_mode_selector_row(ui);
        self.maybe_start_tutorial_for_mode();

        self.sync_mujoco_overlay_active();

        ui.separator();
        self.show_shared_controls_row(ui, display_fps);

        ui.separator();
        self.show_main_content(ui, frame);

        self.paint_help_highlight(ui);
        self.sync_mujoco_overlay_active();

        let mut overlay_occlusions = self.show_help_windows(ui.ctx());
        self.sync_mujoco_overlay_active();

        if self.ui_state.show_graph {
            if let Some(rect) = self.show_graph_window(ui.ctx()) {
                overlay_occlusions.push(rect);
            }
        }
        self.sync_mujoco_overlay_occlusions(&overlay_occlusions);
        self.sync_mujoco_overlay_active();
    }

    /// Draws only the mode-specific main content area for tightly embedded
    /// tutorial/documentation contexts.
    ///
    /// This intentionally omits the global mode selector, shared controls row,
    /// help overlays, and detached graph window so the caller can embed just
    /// the active simulator surface and its mode-specific sidebar.
    pub fn ui_embedded(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>, display_fps: f32) {
        self.sync_mujoco_overlay_active();
        self.reset_help_regions();
        self.handle_ui_input(ui.ctx());
        self.show_embedded_content(ui, frame, display_fps);
        self.sync_mujoco_overlay_occlusions(&[]);
        self.sync_mujoco_overlay_active();
    }

    /// Returns the rendered content height for the focused embed layout.
    ///
    /// This is used by the docs iframe host to size the embed from actual egui
    /// layout measurements instead of a heuristic guess.
    pub fn embedded_content_height(&self) -> Option<f32> {
        if matches!(
            self.mode,
            SimMode::Localization
                | SimMode::InvertedPendulum
                | SimMode::PathPlanning
                | SimMode::Slam
        ) {
            let width = self
                .help_state
                .help_scene_rect
                .map(|rect| rect.width())
                .unwrap_or(640.0)
                .max(320.0);
            let spec = self.mode.embed_layout_spec(width);
            return Some(spec.scene_margin as f32 + spec.scene_height + 20.0);
        }

        let mut bottom = 0.0_f32;
        let mut saw_rect = false;

        for rect in [
            self.help_state.help_options_rect,
            self.help_state.help_scene_rect,
        ]
        .into_iter()
        .flatten()
        {
            saw_rect = true;
            bottom = bottom.max(rect.bottom());
        }

        saw_rect.then_some(bottom + 20.0)
    }
}
