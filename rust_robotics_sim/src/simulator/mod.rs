pub mod common;
pub mod localization;
pub mod mujoco;
pub mod path_planning;
pub mod pendulum;
mod ppo_trainer;
pub mod slam;

use localization::ParticleFilter;
use mujoco::MujocoPanel;
use path_planning::{EnvironmentMode, PathPlanning};
use pendulum::{InvertedPendulum, NoiseConfig as PendulumNoiseConfig, PENDULUM_FIXED_DT};
use slam::SlamDemo;

use egui::{epaint::Hsva, *};
use egui_plot::{Corner, Legend, Plot, PlotUi};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HelpTarget {
    ModeSelector,
    Controls,
    Options,
    Scene,
}

#[derive(Debug, Clone, Copy)]
struct HelpStep {
    title: &'static str,
    body: &'static str,
    target: Option<HelpTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendulumPlotTab {
    LateralPosition,
    LateralVelocity,
    RodAngle,
    RodAngularVelocity,
    ControlInput,
}

impl PendulumPlotTab {
    const ALL: [Self; 5] = [
        Self::LateralPosition,
        Self::LateralVelocity,
        Self::RodAngle,
        Self::RodAngularVelocity,
        Self::ControlInput,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::LateralPosition => "Lateral Position",
            Self::LateralVelocity => "Lateral Velocity",
            Self::RodAngle => "Rod Angle",
            Self::RodAngularVelocity => "Rod Angular Velocity",
            Self::ControlInput => "Control Input",
        }
    }

    fn signal_index(self) -> usize {
        match self {
            Self::LateralPosition => 0,
            Self::LateralVelocity => 1,
            Self::RodAngle => 2,
            Self::RodAngularVelocity => 3,
            Self::ControlInput => 4,
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
    /// Simulations for path planning mode
    planners: Vec<PathPlanning>,
    /// Simulations for SLAM mode
    slam_demos: Vec<SlamDemo>,
    /// Browser-hosted MuJoCo demo overlay
    mujoco_panel: MujocoPanel,
    /// Current simulation time in seconds.
    time: f32,
    /// Fixed-step simulation speed multiplier relative to wall-clock time.
    /// A value of 1 advances one simulation step of `dt` for each `dt` of
    /// accumulated wall-clock time on average.
    sim_speed: usize,
    /// Settings to indicate whether to show the graph of simulation signals
    show_graph: bool,
    /// Active tab for inverted pendulum signal plots.
    pendulum_plot_tab: PendulumPlotTab,
    paused: bool,
    /// Shared grid width for path planning (in cells)
    grid_width: usize,
    /// Shared grid height for path planning (in cells)
    grid_height: usize,
    /// Shared grid resolution for path planning (meters per cell)
    grid_resolution: f32,
    /// Shared environment mode for path planning
    env_mode: EnvironmentMode,
    /// Shared obstacle radius for continuous path planning
    continuous_obstacle_radius: f32,
    /// Whether measurement/action noise is enabled for inverted pendulum mode.
    pendulum_noise_enabled: bool,
    /// Dimensionless scale factor applied to the inverted pendulum noise profile.
    pendulum_noise_scale: f32,
    /// Modes visited to track help popup
    visited_modes: HashSet<SimMode>,
    /// Global tutorial preference. `None` means the user has not chosen yet.
    tutorial_enabled: Option<bool>,
    /// Whether to show the initial welcome/splash prompt.
    show_tutorial_prompt: bool,
    /// Whether to show the help popup
    show_help_popup: bool,
    /// Whether the shared simulator chrome has already been explained in the guided tour.
    shared_help_intro_completed: bool,
    /// Current walkthrough step index for the help popup.
    help_step_index: usize,
    /// Rect for the mode selector row, used by the guided help highlight.
    help_mode_selector_rect: Option<Rect>,
    /// Rect for the global controls row, used by the guided help highlight.
    help_controls_rect: Option<Rect>,
    /// Rect for the current mode's options panel, used by the guided help highlight.
    help_options_rect: Option<Rect>,
    /// Rect for the main scene/viewport area, used by the guided help highlight.
    help_scene_rect: Option<Rect>,
    /// Last wall-clock tick used for fixed-step simulation pacing.
    last_tick: Option<Instant>,
    /// Accumulator for non-MuJoCo fixed-step simulation.
    sim_accumulator: f32,
}

impl Default for Simulator {
    fn default() -> Self {
        Self {
            mode: SimMode::InvertedPendulum,
            pendulums: vec![InvertedPendulum::default()],
            vehicles: vec![ParticleFilter::new(1, 0.0)],
            planners: vec![PathPlanning::new(1, 0.0)],
            slam_demos: vec![SlamDemo::new(1, 0.0)],
            mujoco_panel: MujocoPanel::default(),
            time: 0.0,
            sim_speed: 1,
            show_graph: false,
            pendulum_plot_tab: PendulumPlotTab::LateralPosition,
            paused: false,
            grid_width: 40,
            grid_height: 40,
            grid_resolution: 1.0,
            env_mode: EnvironmentMode::Grid,
            continuous_obstacle_radius: 1.0,
            pendulum_noise_enabled: false,
            pendulum_noise_scale: 0.0,
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
            last_tick: None,
            sim_accumulator: 0.0,
        }
    }
}

impl Simulator {
    pub fn mode(&self) -> SimMode {
        self.mode
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn restart(&mut self) {
        self.time = 0.0;
        self.sim_accumulator = 0.0;
        self.last_tick = None;
        self.reset_state();
    }

    pub fn set_mode(&mut self, mode: SimMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.time = 0.0;
        self.sim_accumulator = 0.0;
        self.last_tick = None;
        self.reset_help_tour();
    }

    fn sync_mujoco_overlay_active(&mut self) {
        let active = self.mode == SimMode::Mujoco;
        self.mujoco_panel.set_active(active);
    }

    fn sync_mujoco_overlay_occlusions(&mut self, rects: &[Rect]) {
        let interactive = self.mode == SimMode::Mujoco && rects.is_empty();
        self.mujoco_panel.set_overlay_occlusions(rects, interactive);
    }

    fn help_steps(&self) -> &'static [HelpStep] {
        const INVERTED_PENDULUM_STEPS: [HelpStep; 9] = [
            HelpStep {
                title: "Mode Selector",
                body: "Choose which simulator is active here. Switching modes resets the local timebase and opens the relevant controls.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Play And Reset",
                body: "Use Play or Pause to freeze the current response, Restart to re-run with the same parameters, and Reset All to return this mode to its defaults. That makes it easy to test one change at a time and replay from the same starting condition.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed And Noise",
                body: "Simulation speed changes how quickly simulated time advances. The Noise toggle injects sensor noise and actuator noise. As you raise it, expect noisier measurements, rougher control effort, and more visible jitter in sensitive controllers.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Comparing Pendulums",
                body: "Add Pendulum creates another copy of the same task so you can compare controllers or parameter choices side by side. This is the easiest way to build intuition, because you can keep one baseline untouched while you experiment on the other.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Physical Model",
                body: "The Cart section changes the plant itself. Increasing Beam Length slows the pendulum and gives the controller more leverage. Increasing Ball Mass makes the top heavier and usually harder to catch after a disturbance. Increasing Cart Mass makes lateral correction more sluggish for the same controller.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Controller Choice",
                body: "The Controller selector changes the control strategy, not just the tuning. LQR is a clean linear stabilizer near upright, PID reacts directly to angle error, and MPC looks ahead but can cost more computation. If behavior changes a lot after a plant change, switching controllers helps show whether the limitation is the tuning or the control law itself.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Tuning Intuition",
                body: "For LQR, increasing the Rod Angle or Rod Angular Velocity weights makes the controller fight tipping more aggressively. Increasing the Lateral Position or Velocity weights makes it care more about keeping the cart centered. Increasing the Control Input weight makes it more conservative. For PID, higher P reacts harder to tilt, higher I removes steady bias but can add overshoot, and higher D damps oscillation.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Scene And Plot",
                body: "Watch the cart motion and rod angle here, then open the Signal Plot to compare one channel across multiple pendulums. If a heavier ball causes larger swings or slower recovery, you should see that immediately both in the scene and in the angle or control traces.",
                target: Some(HelpTarget::Scene),
            },
            HelpStep {
                title: "Scene Overview",
                body: "This is the live pendulum scene. Use it to connect parameter changes to visible behavior: quicker recovery, more oscillation, slower cart motion, or loss of stability.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const LOCALIZATION_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use this row to switch between localization, SLAM, pendulum, robot, and planning demos.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the estimation process, Restart replays the same setup, and Add Vehicle creates another estimator so you can compare settings side by side.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed changes how fast the vehicle and filter move through time. Slowing the demo down is useful when you want to watch drift build up or see how the filter recovers after a turn.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Drive Mode",
                body: "Kinematic mode uses simple commanded velocity and yaw rate, so it is easier to reason about. Dynamic mode adds tire and body dynamics, so the vehicle can slip or lag. Dynamic mode is where model mismatch becomes more visible.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Motion And Vehicle Knobs",
                body: "In kinematic mode, increasing speed makes dead-reckoning drift accumulate faster, and increasing yaw rate creates tighter turns that are harder to estimate cleanly. In dynamic mode, mass, axle distances, and tire parameters change how quickly the car rotates and settles, which directly affects how hard localization becomes.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "What To Compare",
                body: "Compare the true pose, dead-reckoning path, and particle-filter estimate. If the estimate stays close to truth while dead reckoning drifts away, the filter is using observations effectively. If all curves separate, the motion or sensor assumptions are too weak for the chosen settings.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Localization Scene",
                body: "Drive and inspect the localization behavior in this scene. Arrow keys control dynamic vehicle modes. The most informative moments are hard turns and higher-speed segments, where model error is easiest to see.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const MUJOCO_STEPS: [HelpStep; 6] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use this selector to enter the Robot mode or return to the other simulators. Each mode focuses on a different robotics demo.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the live robot, Restart resets the current run, and Reset All restores the mode defaults. Use these when you want a clean comparison after changing the target or switching robots.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed changes how quickly robot time advances. Slowing it down is useful when you want to study gait timing, recovery steps, or how the controller responds after you move the target.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Robot Panel",
                body: "Choose which robot to run here. Different robots expose different control behaviors and policy styles, so switching robots is really switching the task and controller together. Read the description text as a clue for what motion you should expect to see.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Setpoint Interaction",
                body: "For robots that support setpoint control, dragging the red ball changes the commanded target in the scene. Move it farther away and expect the controller to lean, step, or reorient more aggressively to chase it. Small target changes should produce small posture corrections; large jumps expose controller limits much more clearly.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Viewport Note",
                body: "The viewport is where you judge stability and tracking quality. Watch whether the robot approaches the target smoothly, oscillates around it, or fails to recover after a disturbance. Those visual cues are often more informative than a single scalar metric.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const PATH_PLANNING_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Switch between planners and the other demos from this selector.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback And Comparison",
                body: "Use Restart to replay the same planning scene, Add Planner to create another planner panel, and Show Graph if you want plotted data later. The main value here is comparing multiple planners in the same environment.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Speed",
                body: "Simulation speed mainly affects how quickly animations and repeated replans advance. Slowing it down is useful when you want to watch visited cells or RRT growth more carefully.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Environment Controls",
                body: "Grid mode gives discrete cells and naturally fits A*, Dijkstra, and Theta*. Continuous mode gives free space with circular obstacles and is better for RRT. Changing grid size or resolution changes both difficulty and cost: finer grids can produce cleaner paths, but they usually make the search do more work.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planner Choice",
                body: "A* is a strong baseline when you want efficient shortest-path search with a heuristic. Dijkstra ignores the heuristic and usually explores more broadly. Theta* can cut across line of sight and often produces shorter-looking paths. RRT is sampling-based, so it is better matched to continuous spaces than dense grids.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planner Knobs",
                body: "For RRT, larger Expand values make the tree jump farther each step, which can explore faster but miss narrow passages. Higher Goal Bias pulls the tree toward the goal more often, which can speed up easy scenes but reduce exploration. Max Iter controls how long the planner is allowed to keep searching before it gives up.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Planning Scene",
                body: "Set a green start and a red goal in the scene, then compare how each planner moves through the same environment. If you enable visited cells or the RRT tree, you can study not just the final path but the search strategy that produced it.",
                target: Some(HelpTarget::Scene),
            },
        ];
        const SLAM_STEPS: [HelpStep; 7] = [
            HelpStep {
                title: "Mode Selector",
                body: "Use the selector to move between SLAM and the other simulations.",
                target: Some(HelpTarget::ModeSelector),
            },
            HelpStep {
                title: "Playback Controls",
                body: "Play or Pause freezes the current trajectory, Restart replays the run from the same map, and simulation speed changes how quickly observations accumulate. Slowing the demo down makes it easier to see when estimates start drifting apart.",
                target: Some(HelpTarget::Controls),
            },
            HelpStep {
                title: "Algorithm Toggles",
                body: "Enable EKF, Graph SLAM, or both. This lets you compare filtering versus optimization on the same sensor stream. If one estimate drifts less or recovers better after revisiting a place, that difference is exactly the lesson this demo is meant to show.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Graph SLAM Options",
                body: "Robust kernels reduce the influence of bad measurements, Sparse solves the optimization more efficiently, and Loop Closure allows the estimate to use revisited places to correct accumulated drift. When loop closure is on, expect the map and trajectory to tighten up after revisits.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Motion And Landmarks",
                body: "Increasing linear or angular speed makes localization harder because error accumulates faster between observations. The landmark count controls how much information the world provides; more landmarks usually give the estimators more chances to correct drift.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "Display And Error",
                body: "Use the display toggles to show covariance, observations, dead reckoning, and landmarks. The error box is the quickest summary of estimator quality: if EKF or Graph error stays below dead reckoning, the estimator is adding useful information instead of just replaying odometry.",
                target: Some(HelpTarget::Options),
            },
            HelpStep {
                title: "SLAM Scene",
                body: "Drive the robot in the scene and compare how the map and pose estimate evolve over time. The most instructive moments are turns, revisits, and longer runs where drift has time to build up.",
                target: Some(HelpTarget::Scene),
            },
        ];

        match self.mode {
            SimMode::InvertedPendulum => &INVERTED_PENDULUM_STEPS,
            SimMode::Localization => &LOCALIZATION_STEPS,
            SimMode::Mujoco => &MUJOCO_STEPS,
            SimMode::PathPlanning => &PATH_PLANNING_STEPS,
            SimMode::Slam => &SLAM_STEPS,
        }
    }

    fn active_help_steps(&self) -> Vec<HelpStep> {
        self.help_steps()
            .iter()
            .copied()
            .filter(|step| {
                !self.shared_help_intro_completed
                    || !matches!(
                        step.target,
                        Some(HelpTarget::ModeSelector | HelpTarget::Controls)
                    )
            })
            .collect()
    }

    fn reset_help_tour(&mut self) {
        self.help_step_index = 0;
    }

    fn help_target_rect(&self, target: HelpTarget) -> Option<Rect> {
        match target {
            HelpTarget::ModeSelector => self.help_mode_selector_rect,
            HelpTarget::Controls => self.help_controls_rect,
            HelpTarget::Options => self.help_options_rect,
            HelpTarget::Scene => self.help_scene_rect,
        }
    }

    fn paint_help_highlight(&self, ui: &Ui) {
        if !self.show_help_popup {
            return;
        }

        let steps = self.active_help_steps();
        let Some(step) = steps.get(self.help_step_index) else {
            return;
        };
        let Some(target) = step.target else {
            return;
        };
        let Some(rect) = self.help_target_rect(target) else {
            return;
        };

        let pulse = (ui.ctx().input(|i| i.time) as f32 * 2.5).sin().abs();
        let highlight_rect = rect.expand(4.0);
        let stroke_color = Color32::from_rgb(255, 210, 64);
        let fill_color = stroke_color.linear_multiply(0.08 + pulse * 0.05);
        let painter = ui.ctx().layer_painter(LayerId::new(
            Order::Background,
            Id::new("simulation_help_highlight"),
        ));
        painter.rect_filled(highlight_rect, 8.0, fill_color);
        painter.rect_stroke(
            highlight_rect,
            8.0,
            Stroke::new(1.5 + pulse * 0.5, stroke_color),
            StrokeKind::Outside,
        );
    }

    fn pendulum_noise_config(&self) -> PendulumNoiseConfig {
        PendulumNoiseConfig {
            enabled: self.pendulum_noise_enabled,
            scale: self.pendulum_noise_scale,
        }
    }

    fn pendulum_plot_ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for tab in PendulumPlotTab::ALL {
                ui.selectable_value(&mut self.pendulum_plot_tab, tab, tab.label());
            }
        });
        ui.separator();
        Plot::new("PendulumPlot")
            .legend(Legend::default().position(Corner::RightTop))
            .show(ui, |plot_ui| {
                self.pendulums.iter().for_each(|sim| {
                    sim.plot_signal(plot_ui, self.pendulum_plot_tab.signal_index());
                });
            });
    }

    fn pendulum_color(index: usize) -> Color32 {
        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0;
        Hsva::new(index as f32 * golden_ratio, 0.85, 0.5, 1.0).into()
    }

    fn nice_grid_step(span: f32, target_lines: f32) -> f32 {
        let raw = (span / target_lines).max(1.0e-3);
        let magnitude = 10.0_f32.powf(raw.log10().floor());
        let normalized = raw / magnitude;
        let nice = if normalized < 1.5 {
            1.0
        } else if normalized < 3.0 {
            2.0
        } else if normalized < 7.0 {
            5.0
        } else {
            10.0
        };
        nice * magnitude
    }

    fn render_pendulum_scene(&self, ui: &mut Ui) -> Rect {
        let desired_size = vec2(
            ui.available_width().max(240.0),
            ui.available_height().max(240.0),
        );
        let (response, painter) = ui.allocate_painter(desired_size, Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let plot_rect = rect;
        let scene_rect = plot_rect.shrink(12.0);
        let pad = 12.0;
        let painter = painter.with_clip_rect(plot_rect);

        painter.rect_filled(plot_rect, 2.0, visuals.extreme_bg_color);
        painter.rect_stroke(
            plot_rect,
            2.0,
            Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
            StrokeKind::Inside,
        );

        if self.pendulums.is_empty() {
            painter.text(
                plot_rect.center(),
                Align2::CENTER_CENTER,
                "No pendulums",
                FontId::proportional(16.0),
                visuals.weak_text_color(),
            );
            return plot_rect;
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = 0.0_f32;
        let mut max_y = 0.0_f32;

        for sim in &self.pendulums {
            let model = sim.model();
            let x = sim.x_position();
            let th = sim.rod_angle();
            let r_ball = 0.1 * model.m_ball;
            let r_whl = 0.1 * model.m_cart;
            let w = model.m_cart;
            let h = 0.5 * model.m_cart;
            let len = model.l_bar;
            let rod_bottom_y = h + 2.0 * r_whl;
            let rod_top_x = x - len * th.sin();
            let rod_top_y = rod_bottom_y + len * th.cos();

            min_x = min_x.min(x - w * 0.5);
            min_x = min_x.min(x - w * 0.25 - r_whl);
            min_x = min_x.min(rod_top_x - r_ball);

            max_x = max_x.max(x + w * 0.5);
            max_x = max_x.max(x + w * 0.25 + r_whl);
            max_x = max_x.max(rod_top_x + r_ball);

            min_y = min_y.min(0.0);
            max_y = max_y.max(rod_top_y + r_ball);
            max_y = max_y.max(rod_bottom_y);
        }

        let width = (max_x - min_x).max(2.0);
        let height = (max_y - min_y).max(2.0);
        let usable_width = (scene_rect.width() - 2.0 * pad).max(1.0);
        let usable_height = (scene_rect.height() - 2.0 * pad).max(1.0);
        let scale = (usable_width / width).min(usable_height / height);
        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let scene_center = pos2(
            scene_rect.center().x,
            scene_rect.bottom() - pad - usable_height * 0.5,
        );

        let to_screen = |x: f32, y: f32| -> Pos2 {
            pos2(
                scene_center.x + (x - center_x) * scale,
                scene_center.y - (y - center_y) * scale,
            )
        };

        let ground_y = to_screen(0.0, 0.0).y;
        let grid_color = visuals
            .widgets
            .noninteractive
            .bg_stroke
            .color
            .linear_multiply(0.35);
        let target_grid_px = 56.0_f32;
        let grid_step = Self::nice_grid_step(target_grid_px / scale.max(1.0e-3), 1.0);
        let x_start = (min_x / grid_step).floor() as i32 - 1;
        let x_end = (max_x / grid_step).ceil() as i32 + 1;
        let y_start = (min_y / grid_step).floor() as i32 - 1;
        let y_end = (max_y / grid_step).ceil() as i32 + 1;

        for ix in x_start..=x_end {
            let x = ix as f32 * grid_step;
            let screen_x = to_screen(x, 0.0).x;
            if screen_x < scene_rect.left() || screen_x > scene_rect.right() {
                continue;
            }
            painter.line_segment(
                [
                    pos2(screen_x, scene_rect.top()),
                    pos2(screen_x, scene_rect.bottom()),
                ],
                Stroke::new(1.0, grid_color),
            );
        }

        for iy in y_start..=y_end {
            let y = iy as f32 * grid_step;
            let screen_y = to_screen(0.0, y).y;
            if screen_y < scene_rect.top() || screen_y > scene_rect.bottom() {
                continue;
            }
            painter.line_segment(
                [
                    pos2(scene_rect.left(), screen_y),
                    pos2(scene_rect.right(), screen_y),
                ],
                Stroke::new(1.0, grid_color),
            );
        }

        let origin_x = to_screen(0.0, 0.0).x;
        if origin_x >= scene_rect.left() && origin_x <= scene_rect.right() {
            painter.line_segment(
                [
                    pos2(origin_x, scene_rect.top()),
                    pos2(origin_x, scene_rect.bottom()),
                ],
                Stroke::new(1.5, visuals.widgets.noninteractive.fg_stroke.color),
            );
        }

        painter.line_segment(
            [
                pos2(scene_rect.left() + pad * 0.5, ground_y),
                pos2(scene_rect.right() - pad * 0.5, ground_y),
            ],
            Stroke::new(2.0, visuals.widgets.noninteractive.fg_stroke.color),
        );

        let tick_y0 = ground_y - 4.0;
        let tick_y1 = ground_y + 4.0;
        let label_y = ground_y + 8.0;
        let tick_start = min_x.floor() as i32 - 1;
        let tick_end = max_x.ceil() as i32 + 1;
        for tick in tick_start..=tick_end {
            let x = tick as f32;
            let screen_x = to_screen(x, 0.0).x;
            if screen_x < scene_rect.left() || screen_x > scene_rect.right() {
                continue;
            }
            painter.line_segment(
                [pos2(screen_x, tick_y0), pos2(screen_x, tick_y1)],
                Stroke::new(1.0, visuals.widgets.noninteractive.fg_stroke.color),
            );
            painter.text(
                pos2(screen_x, label_y),
                Align2::CENTER_TOP,
                tick.to_string(),
                FontId::proportional(12.5),
                visuals.text_color(),
            );
        }

        for (index, sim) in self.pendulums.iter().enumerate() {
            let model = sim.model();
            let x = sim.x_position();
            let th = sim.rod_angle();
            let base_color = Self::pendulum_color(index);
            let stroke = Stroke::new(2.0, visuals.widgets.noninteractive.fg_stroke.color);
            let wheel_stroke = Stroke::new(1.5, stroke.color);
            let fill = base_color.linear_multiply(0.05);

            let r_ball = 0.1 * model.m_ball;
            let r_whl = 0.1 * model.m_cart;
            let w = model.m_cart;
            let h = 0.5 * model.m_cart;
            let len = model.l_bar;

            let body_center = pos2(x, h * 0.5 + 2.0 * r_whl);
            let rod_bottom = pos2(x, h + 2.0 * r_whl);
            let rod_top = pos2(x - len * th.sin(), rod_bottom.y + len * th.cos());
            let left_wheel = pos2(x - w * 0.25, r_whl);
            let right_wheel = pos2(x + w * 0.25, r_whl);
            let wheel_angle = -x / r_whl.max(1.0e-3);

            let body_rect = Rect::from_center_size(
                to_screen(body_center.x, body_center.y),
                vec2(w * scale, h * scale),
            );
            painter.rect_filled(body_rect, 4.0, fill);
            painter.rect_stroke(body_rect, 4.0, stroke, StrokeKind::Inside);

            for wheel_center in [left_wheel, right_wheel] {
                let center = to_screen(wheel_center.x, wheel_center.y);
                let radius = (r_whl * scale).max(2.0);
                painter.circle_filled(center, radius, visuals.panel_fill);
                painter.circle_stroke(center, radius, wheel_stroke);
                let tick = pos2(
                    center.x + radius * wheel_angle.cos(),
                    center.y - radius * wheel_angle.sin(),
                );
                painter.line_segment([center, tick], wheel_stroke);
            }

            let rod_bottom_screen = to_screen(rod_bottom.x, rod_bottom.y);
            let rod_top_screen = to_screen(rod_top.x, rod_top.y);
            painter.line_segment(
                [rod_bottom_screen, rod_top_screen],
                Stroke::new(3.0, base_color),
            );

            let ball_radius = (r_ball * scale).max(3.0);
            painter.circle_filled(rod_top_screen, ball_radius, base_color);
            painter.circle_stroke(rod_top_screen, ball_radius, stroke);
        }

        let legend_font = FontId::proportional(13.5);
        let legend_line_width = 18.0;
        let legend_row_height = 18.0;
        let legend_padding = vec2(8.0, 6.0);
        let legend_entries: Vec<String> = self
            .pendulums
            .iter()
            .map(|sim| format!("Cart {}", sim.id()))
            .collect();
        let legend_text_width = legend_entries
            .iter()
            .map(|label| {
                ui.painter()
                    .layout_no_wrap(label.clone(), legend_font.clone(), visuals.text_color())
                    .size()
                    .x
            })
            .fold(0.0, f32::max);
        let legend_size = vec2(
            legend_padding.x * 2.0 + legend_line_width + 8.0 + legend_text_width,
            legend_padding.y * 2.0 + legend_row_height * legend_entries.len() as f32,
        );
        let legend_rect = Rect::from_min_size(
            pos2(
                plot_rect.right() - 10.0 - legend_size.x,
                plot_rect.top() + 10.0,
            ),
            legend_size,
        );

        painter.rect_filled(
            legend_rect,
            2.0,
            visuals.extreme_bg_color.gamma_multiply(0.92),
        );
        painter.rect_stroke(
            legend_rect,
            2.0,
            visuals.widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        for (index, label) in legend_entries.iter().enumerate() {
            let y = legend_rect.top() + legend_padding.y + legend_row_height * index as f32;
            let color = Self::pendulum_color(index);
            let line_mid_y = y + legend_row_height * 0.5;
            let line_start = pos2(legend_rect.left() + legend_padding.x, line_mid_y);
            let line_end = pos2(line_start.x + legend_line_width, line_mid_y);
            painter.line_segment([line_start, line_end], Stroke::new(3.0, color));
            painter.text(
                pos2(line_end.x + 8.0, y + 1.0),
                Align2::LEFT_TOP,
                label,
                legend_font.clone(),
                visuals.text_color(),
            );
        }

        plot_rect
    }

    fn show_option_cards(&mut self, ui: &mut Ui, cards_vertical: bool) {
        match self.mode {
            SimMode::InvertedPendulum => {
                if cards_vertical {
                    ui.vertical(|ui| {
                        self.pendulums.retain_mut(|sim| sim.options_with_policy(ui));
                    });
                } else {
                    egui::ScrollArea::horizontal()
                        .id_salt("pendulum_cards")
                        .max_height(300.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                self.pendulums.retain_mut(|sim| sim.options_with_policy(ui));
                            });
                        });
                }
            }
            SimMode::Localization => {
                if cards_vertical {
                    ui.vertical(|ui| {
                        self.vehicles.retain_mut(|sim| sim.options(ui));
                    });
                } else {
                    egui::ScrollArea::horizontal()
                        .id_salt("localization_cards")
                        .max_height(320.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                self.vehicles.retain_mut(|sim| sim.options(ui));
                            });
                        });
                }
            }
            SimMode::Mujoco => {
                self.mujoco_panel.ui_controls(ui);
            }
            SimMode::PathPlanning => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Env:");
                    if ui
                        .selectable_label(self.env_mode == EnvironmentMode::Grid, "Grid")
                        .clicked()
                    {
                        self.env_mode = EnvironmentMode::Grid;
                    }
                    if ui
                        .selectable_label(
                            self.env_mode == EnvironmentMode::Continuous,
                            "Continuous",
                        )
                        .clicked()
                    {
                        self.env_mode = EnvironmentMode::Continuous;
                    }

                    if self.env_mode == EnvironmentMode::Continuous {
                        ui.label("Obs Radius:");
                        ui.add(
                            DragValue::new(&mut self.continuous_obstacle_radius)
                                .range(0.1..=5.0)
                                .speed(0.1),
                        );
                    }
                });

                if self.env_mode == EnvironmentMode::Grid {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Grid:");
                        ui.label("Width:");
                        let width_changed = ui
                            .add(DragValue::new(&mut self.grid_width).range(10..=100))
                            .changed();
                        ui.label("Height:");
                        let height_changed = ui
                            .add(DragValue::new(&mut self.grid_height).range(10..=100))
                            .changed();
                        ui.label("Resolution:");
                        let res_changed = ui
                            .add(
                                DragValue::new(&mut self.grid_resolution)
                                    .range(0.1..=5.0)
                                    .speed(0.1),
                            )
                            .changed();

                        for planner in &mut self.planners {
                            if width_changed || height_changed || res_changed {
                                planner.update_grid_settings(
                                    self.grid_width,
                                    self.grid_height,
                                    self.grid_resolution,
                                );
                            }
                        }
                    });
                }

                for planner in &mut self.planners {
                    planner.set_env_mode(self.env_mode);
                    planner.set_continuous_obstacle_radius(self.continuous_obstacle_radius);
                }

                ui.separator();

                if cards_vertical {
                    ui.vertical(|ui| {
                        self.planners.retain_mut(|sim| sim.options(ui));
                    });
                } else {
                    egui::ScrollArea::horizontal()
                        .id_salt("path_planning_cards")
                        .max_height(320.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                self.planners.retain_mut(|sim| sim.options(ui));
                            });
                        });
                }
            }
            SimMode::Slam => {
                if cards_vertical {
                    ui.vertical(|ui| {
                        self.slam_demos.retain_mut(|sim| sim.options(ui));
                    });
                } else {
                    egui::ScrollArea::horizontal()
                        .id_salt("slam_cards")
                        .max_height(260.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                self.slam_demos.retain_mut(|sim| sim.options(ui));
                            });
                        });
                }
            }
        }
    }

    fn show_sidebar(&mut self, ui: &mut Ui, cards_vertical: bool) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.show_option_cards(ui, cards_vertical);

                if self.mode == SimMode::Localization && self.vehicles.iter().any(|v| v.is_dynamic_mode()) {
                    ui.collapsing("Keyboard Controls", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("← → : Steering");
                            ui.label("   ↑ ↓ : Accelerate/Brake");
                            ui.label("   Space : Pause");
                            ui.label("   Enter : Restart");
                        });
                    });
                }

                if self.mode == SimMode::Slam && self.slam_demos.iter().any(|s| s.is_manual_mode()) {
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
                    match self.mode {
                        SimMode::PathPlanning => {
                            ui.label(RichText::new("1. Start & Goal").strong());
                            ui.label("• Left-click on the map to set the Start point (Green).");
                            ui.label("• Left-click again to set the Goal point (Red).");
                            ui.label("• Once both are set, all active planners will run automatically.");

                            ui.add_space(5.0);
                            ui.label(RichText::new("2. Environment").strong());
                            if self.env_mode == EnvironmentMode::Grid {
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
                            ui.label("• The MuJoCo tab runs the native MuJoCo model and ONNX policy inside Rust.");
                            ui.label("• The viewport uses the shared Rust-owned 3D renderer path across native and web.");
                        }
                        _ => {}
                    }

                    ui.add_space(5.0);
                    ui.label(RichText::new("Navigation").strong());
                    ui.label("• Pan by dragging, or scroll (+ shift = horizontal).");
                    ui.label("• Box zooming: Right click to zoom in and zoom out using a selection.");
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

    fn show_scene_pane(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>) {
        if self.mode == SimMode::Mujoco {
            self.mujoco_panel.ui_viewport(ui, frame);
            self.help_scene_rect = Some(ui.min_rect());
        } else if self.mode == SimMode::InvertedPendulum {
            self.help_scene_rect = Some(self.render_pendulum_scene(ui));
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

            let plot_response = plot.show(ui, |plot_ui| match self.mode {
                SimMode::InvertedPendulum => {
                    self.pendulums.iter().for_each(|sim| sim.scene(plot_ui));
                }
                SimMode::Localization => {
                    self.vehicles.iter().for_each(|sim| sim.scene(plot_ui));
                }
                SimMode::Slam => {
                    self.slam_demos.iter().for_each(|sim| sim.scene(plot_ui));
                }
                SimMode::PathPlanning => {
                    self.planners.iter().for_each(|sim| sim.scene(plot_ui));
                }
                SimMode::Mujoco => unreachable!(),
            });

            if self.mode == SimMode::PathPlanning {
                if let Some((first, rest)) = self.planners.split_first_mut() {
                    first.handle_mouse(&plot_response);
                    rest.iter_mut().for_each(|sim| sim.match_state_with(first));
                }
            }
            self.help_scene_rect = Some(plot_response.response.rect);
        }
    }

    /// Update the simulation for a single time step
    pub fn update(&mut self) {
        self.sync_mujoco_overlay_active();
        let now = Instant::now();
        let elapsed = self
            .last_tick
            .replace(now)
            .map(|last| (now - last).as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1);

        if !self.paused {
            let dt = PENDULUM_FIXED_DT;
            match self.mode {
                SimMode::InvertedPendulum => {
                    let noise = self.pendulum_noise_config();
                    self.pendulums
                        .iter_mut()
                        .for_each(InvertedPendulum::tick_training);
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.pendulums
                            .iter_mut()
                            .for_each(|pendulum| pendulum.step_with_noise(dt, noise));
                    });
                }
                SimMode::Localization => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.vehicles.iter_mut().for_each(|vehicle| vehicle.step(dt));
                    });
                }
                SimMode::Mujoco => {
                    let mujoco_dt = self.mujoco_panel.fixed_dt();
                    self.step_fixed_dt(elapsed, mujoco_dt, |sim, _dt| {
                        sim.mujoco_panel.update(1, false);
                    });
                }
                SimMode::PathPlanning => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.planners.iter_mut().for_each(|planner| planner.step(dt));
                    });
                }
                SimMode::Slam => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.slam_demos.iter_mut().for_each(|slam| slam.step(dt));
                    });
                }
            }
        }
        if self.mode == SimMode::InvertedPendulum && self.paused {
            self.pendulums
                .iter_mut()
                .for_each(InvertedPendulum::tick_training);
        }
    }

    fn step_fixed_dt<F>(&mut self, elapsed: f32, dt: f32, mut step_once: F)
    where
        F: FnMut(&mut Self, f32),
    {
        self.sim_accumulator =
            (self.sim_accumulator + elapsed * self.sim_speed as f32).min(dt * 32.0);
        while self.sim_accumulator >= dt {
            step_once(self, dt);
            self.time += dt;
            self.sim_accumulator -= dt;
        }
    }

    /// Reset the states of all simulations within the current mode
    fn reset_state(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.pendulums.iter_mut().for_each(|sim| sim.reset_state());
                // Sync states
                if let Some((first, rest)) = self.pendulums.split_first_mut() {
                    rest.iter_mut().for_each(|sim| sim.match_state_with(first));
                }
            }
            SimMode::Localization => {
                self.vehicles.iter_mut().for_each(|sim| sim.reset_state());
            }
            SimMode::Mujoco => {
                self.mujoco_panel.reset_state();
            }
            SimMode::PathPlanning => {
                self.planners.iter_mut().for_each(|sim| sim.reset_state());
            }
            SimMode::Slam => {
                self.slam_demos.iter_mut().for_each(|sim| sim.reset_state());
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
            SimMode::Mujoco => {
                self.mujoco_panel.reset_all();
            }
            SimMode::PathPlanning => {
                self.planners.iter_mut().for_each(|sim| sim.reset_all());
            }
            SimMode::Slam => {
                self.slam_demos.iter_mut().for_each(|sim| sim.reset_all());
            }
        }
    }

    /// Add a new simulation instance to the current mode
    fn add_simulation(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                let id = self.pendulums.iter().map(|s| s.id()).max().unwrap_or(0) + 1;
                self.pendulums.push(InvertedPendulum::new(id, self.time));
            }
            SimMode::Localization => {
                let id = self.vehicles.iter().map(|s| s.id()).max().unwrap_or(0) + 1;
                self.vehicles.push(ParticleFilter::new(id, self.time));
            }
            SimMode::Mujoco => {}
            SimMode::PathPlanning => {
                let id = self.planners.iter().map(|s| s.id()).max().unwrap_or(0) + 1;
                let mut new_planner = PathPlanning::new(id, self.time);
                // Apply shared settings
                new_planner.update_grid_settings(
                    self.grid_width,
                    self.grid_height,
                    self.grid_resolution,
                );
                new_planner.set_env_mode(self.env_mode);
                new_planner.set_continuous_obstacle_radius(self.continuous_obstacle_radius);

                // Copy state from first existing planner if available
                if let Some(first) = self.planners.first() {
                    new_planner.copy_state_from(first);
                }
                self.planners.push(new_planner);
            }
            SimMode::Slam => {
                let id = self.slam_demos.iter().map(|s| s.id()).max().unwrap_or(0) + 1;
                self.slam_demos.push(SlamDemo::new(id, self.time));
            }
        }
    }

    /// Draw the UI directly into a Ui (for embedding in CentralPanel)
    pub fn ui(&mut self, ui: &mut Ui, frame: Option<&eframe::Frame>, display_fps: f32) {
        self.sync_mujoco_overlay_active();
        self.help_mode_selector_rect = None;
        self.help_controls_rect = None;
        self.help_options_rect = None;
        self.help_scene_rect = None;

        // Handle space key to pause/resume simulation
        if ui.ctx().input(|i| i.key_pressed(egui::Key::Space)) {
            self.paused = !self.paused;
        }

        // Handle enter key to restart simulation
        if ui.ctx().input(|i| i.key_pressed(egui::Key::Enter)) {
            self.restart();
        }

        // Handle keyboard input for vehicle simulations
        if self.mode == SimMode::Localization {
            let ctx = ui.ctx().clone();
            for vehicle in &mut self.vehicles {
                vehicle.handle_keyboard(&ctx);
            }
        }

        // Handle keyboard input for SLAM simulations
        if self.mode == SimMode::Slam {
            let ctx = ui.ctx().clone();
            for slam in &mut self.slam_demos {
                slam.handle_keyboard(&ctx);
            }
        }

        // Mode selector at the top
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
        self.help_mode_selector_rect = Some(mode_selector_response.response.rect);

        // Check if we need to show tutorial UI for the current mode.
        if !self.visited_modes.contains(&self.mode) {
            match self.tutorial_enabled {
                Some(true) => {
                    self.visited_modes.insert(self.mode);
                    self.show_help_popup = true;
                    self.reset_help_tour();
                }
                Some(false) => {}
                None => {
                    self.show_tutorial_prompt = true;
                }
            }
        }

        self.sync_mujoco_overlay_active();

        ui.separator();

        // Control buttons
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

            // SLAM has single vehicle with multiple algorithms - no "Add" button
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

            ui.checkbox(&mut self.show_graph, "Show Graph");

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

            // Landmark count slider for SLAM mode
            if self.mode == SimMode::Slam {
                if let Some(slam) = self.slam_demos.first_mut() {
                    ui.separator();
                    ui.label("Landmarks:");
                    let response =
                        ui.add(Slider::new(&mut slam.n_landmarks, 1..=50).show_value(true));
                    // Only regenerate when slider is released, not during drag
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
        self.help_controls_rect = Some(controls_response.response.rect);

        ui.separator();
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
                self.help_options_rect = Some(options_response.response.rect);

                ui.add_space(14.0);

                ui.allocate_ui_with_layout(
                    vec2(total_width, scene_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_scene_pane(ui, frame),
                );
            });
        } else {
            ui.horizontal_top(|ui| {
                let options_response = ui.allocate_ui_with_layout(
                    vec2(sidebar_width, content_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_sidebar(ui, cards_vertical),
                );
                self.help_options_rect = Some(options_response.response.rect);

                ui.add_space(14.0);

                ui.allocate_ui_with_layout(
                    vec2(ui.available_width(), content_height),
                    Layout::top_down(Align::Min),
                    |ui| self.show_scene_pane(ui, frame),
                );
            });
        }

        self.paint_help_highlight(ui);
        self.sync_mujoco_overlay_active();

        let mut overlay_occlusions = Vec::new();

        // Welcome / tutorial opt-in prompt
        let mut start_tutorial = false;
        let mut skip_tutorial = false;
        if self.show_tutorial_prompt {
            let mut show_tutorial_prompt = self.show_tutorial_prompt;
            if let Some(window) = egui::Window::new("Welcome")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .open(&mut show_tutorial_prompt)
                .show(ui.ctx(), |ui| {
                    ui.heading("Welcome to Rust Robotics");
                    ui.separator();
                    ui.label("This app includes several robotics demos, including pendulum control, localization, path planning, SLAM, and robot simulation.");
                    ui.add_space(6.0);
                    ui.label("Would you like a short guided tutorial when you open each simulator for the first time?");
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Start Tutorial").clicked() {
                            start_tutorial = true;
                        }
                        if ui.button("Skip").clicked() {
                            skip_tutorial = true;
                        }
                    });
                })
            {
                overlay_occlusions.push(window.response.rect);
            }
            self.show_tutorial_prompt = show_tutorial_prompt;
        }
        if start_tutorial {
            self.tutorial_enabled = Some(true);
            self.show_tutorial_prompt = false;
            self.visited_modes.insert(self.mode);
            self.show_help_popup = true;
            self.reset_help_tour();
        }
        if skip_tutorial || !self.show_tutorial_prompt && self.tutorial_enabled.is_none() {
            // Closing the splash acts as skipping tutorials.
            self.tutorial_enabled = Some(false);
            self.show_tutorial_prompt = false;
            self.show_help_popup = false;
        }

        // Help Popup
        let mut close_popup = false;
        if self.show_help_popup {
            let steps = self.active_help_steps();
            let step_index = self.help_step_index.min(steps.len().saturating_sub(1));
            let step = steps[step_index];
            let mode = self.mode;
            let mut next_help_step = step_index;
            let mut show_help_popup = self.show_help_popup;
            if let Some(window) = egui::Window::new("Simulation Help")
                .collapsible(false)
                .resizable(false)
                .fixed_size(vec2(560.0, 320.0))
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .open(&mut show_help_popup)
                .show(ui.ctx(), |ui| {
                    ui.set_min_size(vec2(520.0, 280.0));
                    ui.heading(format!("{} Instructions", mode.label()));
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(210.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("Step {} of {}", step_index + 1, steps.len()))
                                    .small()
                                    .weak(),
                            );
                            ui.heading(step.title);
                            ui.label(step.body);

                            if mode == SimMode::Localization || mode == SimMode::Slam {
                                ui.add_space(6.0);
                                ui.label("Keyboard: arrows drive, `Space` pauses, `Enter` restarts.");
                            }
                            if mode == SimMode::PathPlanning && step_index == steps.len() - 1 {
                                ui.add_space(6.0);
                                ui.label("Left-click sets start/goal. Right-click paints or places obstacles depending on environment mode.");
                            }
                            if mode == SimMode::Mujoco && step_index == steps.len() - 1 {
                                ui.add_space(6.0);
                                ui.label("Try moving the setpoint a short distance first, then make larger moves to see how the controller transitions from fine tracking to more aggressive recovery.");
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        let button_size = vec2(96.0, 28.0);

                        if ui
                            .add_enabled(
                                step_index > 0,
                                Button::new("Back").min_size(button_size),
                            )
                            .clicked()
                        {
                            next_help_step = step_index.saturating_sub(1);
                        }

                        let forward_label = if step_index + 1 < steps.len() {
                            "Next"
                        } else {
                            "Done"
                        };
                        if ui
                            .add(Button::new(forward_label).min_size(button_size))
                            .clicked()
                        {
                            if step_index + 1 < steps.len() {
                                next_help_step = step_index + 1;
                            } else {
                                close_popup = true;
                            }
                        }

                        ui.add_space((ui.available_width() - button_size.x).max(0.0));

                        if ui
                            .add(Button::new("Close Tour").min_size(button_size))
                            .clicked()
                        {
                            close_popup = true;
                        }
                    });
                })
            {
                overlay_occlusions.push(window.response.rect);
            }
            self.show_help_popup = show_help_popup;
            self.help_step_index = next_help_step;
        }
        if close_popup {
            self.show_help_popup = false;
            self.shared_help_intro_completed = true;
        }
        self.sync_mujoco_overlay_active();

        // Optional graph window
        if self.show_graph {
            if let Some(window) = egui::Window::new("Signal Plot")
                .default_size(vec2(400.0, 300.0))
                .show(ui.ctx(), |ui| {
                    match self.mode {
                        SimMode::InvertedPendulum => self.pendulum_plot_ui(ui),
                        SimMode::Localization => {
                            Plot::new("Plot")
                                .legend(Legend::default().position(Corner::RightTop))
                                .show(ui, |plot_ui| {
                                    self.vehicles.iter().for_each(|sim| sim.plot(plot_ui));
                                });
                        }
                        SimMode::PathPlanning => {
                            Plot::new("Plot")
                                .legend(Legend::default().position(Corner::RightTop))
                                .show(ui, |plot_ui| {
                                    self.planners.iter().for_each(|sim| sim.plot(plot_ui));
                                });
                        }
                        SimMode::Slam => {
                            Plot::new("Plot")
                                .legend(Legend::default().position(Corner::RightTop))
                                .show(ui, |plot_ui| {
                                    self.slam_demos.iter().for_each(|sim| sim.plot(plot_ui));
                                });
                        }
                        SimMode::Mujoco => {
                            Plot::new("Plot")
                                .legend(Legend::default().position(Corner::RightTop))
                                .show(ui, |plot_ui| {
                                    self.mujoco_panel.plot(plot_ui);
                                });
                        }
                    };
                })
            {
                overlay_occlusions.push(window.response.rect);
            }
        }
        self.sync_mujoco_overlay_occlusions(&overlay_occlusions);
        self.sync_mujoco_overlay_active();
    }
}
