//! Runtime and timebase management for the interactive simulator.
//!
//! This module is the execution side of `Simulator`. It owns:
//!
//! - the active mode and pause state
//! - fixed-step accumulation against wall-clock time
//! - restart / reset semantics
//! - per-mode stepping dispatch
//! - MuJoCo overlay synchronization hooks
//!
//! Separating this logic from the UI modules keeps update semantics readable and
//! makes it easier to test the simulator state machine without involving `egui`.
#[cfg(target_arch = "wasm32")]
use super::mujoco::MujocoEmbedState;
use super::{
    localization::{DriveMode as LocalizationDriveMode, LocalizationCardState, LocalizationPatch},
    path_planning::{
        Algorithm as PathPlanningAlgorithm, EnvironmentMode as PathPlanningEnvironmentMode,
        PathPlannerCardState, PathPlannerPatch,
    },
    pendulum::{ControllerKind, PendulumCardState, PendulumPatch},
    slam::{DriveMode as SlamDriveMode, SlamCardState, SlamPatch},
    InvertedPendulum, ParticleFilter, PathPlanning, PendulumNoiseConfig, SimMode, Simulate,
    Simulator, SlamDemo, PENDULUM_FIXED_DT,
};
use web_time::Instant;

impl Simulator {
    /// Returns the currently selected simulator mode.
    pub fn mode(&self) -> SimMode {
        self.mode
    }

    /// Returns whether simulation stepping is currently paused.
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Returns the current simulation time in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Returns the current global simulation speed multiplier.
    pub fn sim_speed(&self) -> usize {
        self.sim_speed
    }

    /// Returns whether the shared graph panel is enabled.
    pub fn show_graph(&self) -> bool {
        self.ui_state.show_graph
    }

    /// Enables or disables the shared graph panel.
    pub fn set_show_graph(&mut self, show_graph: bool) {
        self.ui_state.show_graph = show_graph;
    }

    /// Sets the global simulation speed multiplier.
    pub fn set_sim_speed(&mut self, sim_speed: usize) {
        self.sim_speed = sim_speed.clamp(1, 20);
    }

    /// Returns whether pendulum measurement/action noise is enabled.
    pub fn pendulum_noise_enabled(&self) -> bool {
        self.pendulum_noise_enabled
    }

    /// Enables or disables pendulum noise in the active simulator.
    pub fn set_pendulum_noise_enabled(&mut self, enabled: bool) {
        self.pendulum_noise_enabled = enabled;
    }

    /// Returns the current pendulum noise scale.
    pub fn pendulum_noise_scale(&self) -> f32 {
        self.pendulum_noise_scale
    }

    /// Sets the pendulum noise scale.
    pub fn set_pendulum_noise_scale(&mut self, scale: f32) {
        self.pendulum_noise_scale = scale.clamp(0.0, 3.0);
    }

    /// Returns the number of active pendulum instances.
    pub fn pendulum_count(&self) -> usize {
        self.simulations.pendulums.len()
    }

    /// Returns the serialized DOM-card state for each active pendulum.
    pub(crate) fn pendulum_ui_state(&self) -> Vec<PendulumCardState> {
        self.simulations
            .pendulums
            .iter()
            .map(InvertedPendulum::card_state)
            .collect()
    }

    /// Sets the selected controller kind for the pendulum with the matching id.
    pub(crate) fn set_pendulum_controller_kind(
        &mut self,
        pendulum_id: usize,
        kind: ControllerKind,
    ) {
        if let Some(pendulum) = self
            .simulations
            .pendulums
            .iter_mut()
            .find(|pendulum| pendulum.id() == pendulum_id)
        {
            pendulum.select_controller_kind(kind);
        }
    }

    /// Removes a pendulum instance by id, keeping at least one pendulum alive.
    pub(crate) fn remove_pendulum(&mut self, pendulum_id: usize) {
        if self.simulations.pendulums.len() <= 1 {
            return;
        }
        self.simulations
            .pendulums
            .retain(|pendulum| pendulum.id() != pendulum_id);
    }

    /// Applies a partial configuration patch to one pendulum instance.
    pub(crate) fn patch_pendulum(&mut self, pendulum_id: usize, patch: PendulumPatch) {
        if let Some(pendulum) = self
            .simulations
            .pendulums
            .iter_mut()
            .find(|pendulum| pendulum.id() == pendulum_id)
        {
            pendulum.apply_patch(patch);
        }
    }

    /// Returns the serialized DOM-card state for each active localization demo.
    pub(crate) fn localization_ui_state(&self) -> Vec<LocalizationCardState> {
        self.simulations
            .vehicles
            .iter()
            .map(ParticleFilter::card_state)
            .collect()
    }

    /// Removes a localization vehicle instance by id, keeping at least one alive.
    pub(crate) fn remove_localization_vehicle(&mut self, vehicle_id: usize) {
        if self.simulations.vehicles.len() <= 1 {
            return;
        }
        self.simulations
            .vehicles
            .retain(|vehicle| vehicle.id() != vehicle_id);
    }

    /// Sets the drive mode for one localization vehicle.
    pub(crate) fn set_localization_drive_mode(
        &mut self,
        vehicle_id: usize,
        drive_mode: LocalizationDriveMode,
    ) {
        if let Some(vehicle) = self
            .simulations
            .vehicles
            .iter_mut()
            .find(|vehicle| vehicle.id() == vehicle_id)
        {
            vehicle.set_drive_mode(drive_mode);
        }
    }

    /// Applies a partial configuration patch to one localization vehicle.
    pub(crate) fn patch_localization_vehicle(
        &mut self,
        vehicle_id: usize,
        patch: LocalizationPatch,
    ) {
        if let Some(vehicle) = self
            .simulations
            .vehicles
            .iter_mut()
            .find(|vehicle| vehicle.id() == vehicle_id)
        {
            vehicle.apply_patch(patch);
        }
    }

    /// Returns the current path-planning environment mode used by the web embed.
    pub(crate) fn path_planning_env_mode(&self) -> PathPlanningEnvironmentMode {
        self.path_settings.env_mode
    }

    /// Returns the current continuous obstacle radius used by new and existing planners.
    pub(crate) fn path_planning_continuous_obstacle_radius(&self) -> f32 {
        self.path_settings.continuous_obstacle_radius
    }

    /// Returns the serialized DOM-card state for each active path-planning comparison panel.
    pub(crate) fn path_planning_ui_state(&self) -> Vec<PathPlannerCardState> {
        self.simulations
            .planners
            .iter()
            .map(PathPlanning::card_state)
            .collect()
    }

    /// Sets the global path-planning environment mode and propagates it to all planners.
    pub(crate) fn set_path_planning_env_mode(&mut self, mode: PathPlanningEnvironmentMode) {
        self.path_settings.env_mode = mode;
        for planner in &mut self.simulations.planners {
            planner.set_env_mode(mode);
            planner.set_continuous_obstacle_radius(self.path_settings.continuous_obstacle_radius);
        }
    }

    /// Sets the continuous obstacle radius used by all path-planning demos.
    pub(crate) fn set_path_planning_continuous_obstacle_radius(&mut self, radius: f32) {
        let radius = radius.clamp(0.1, 5.0);
        self.path_settings.continuous_obstacle_radius = radius;
        for planner in &mut self.simulations.planners {
            planner.set_continuous_obstacle_radius(radius);
        }
    }

    /// Removes a path-planning comparison panel by id, keeping at least one alive.
    pub(crate) fn remove_path_planner(&mut self, planner_id: usize) {
        if self.simulations.planners.len() <= 1 {
            return;
        }
        self.simulations
            .planners
            .retain(|planner| planner.id() != planner_id);
    }

    /// Updates the selected primary algorithm for one path-planning comparison panel.
    pub(crate) fn set_path_planner_algorithm(
        &mut self,
        planner_id: usize,
        algorithm: PathPlanningAlgorithm,
    ) {
        if let Some(planner) = self
            .simulations
            .planners
            .iter_mut()
            .find(|planner| planner.id() == planner_id)
        {
            planner.set_primary_algorithm(algorithm);
        }
    }

    /// Updates whether one planner card shows visited cells / the RRT tree.
    pub(crate) fn set_path_planner_show_visited(&mut self, planner_id: usize, show_visited: bool) {
        if let Some(planner) = self
            .simulations
            .planners
            .iter_mut()
            .find(|planner| planner.id() == planner_id)
        {
            planner.set_primary_show_visited(show_visited);
        }
    }

    /// Applies a partial patch to one path-planning comparison panel.
    pub(crate) fn patch_path_planner(&mut self, planner_id: usize, patch: PathPlannerPatch) {
        if let Some(planner) = self
            .simulations
            .planners
            .iter_mut()
            .find(|planner| planner.id() == planner_id)
        {
            planner.apply_primary_patch(patch);
        }
    }

    /// Returns the serialized DOM-card state for each active SLAM demo.
    pub(crate) fn slam_ui_state(&self) -> Vec<SlamCardState> {
        self.simulations
            .slam_demos
            .iter()
            .map(SlamDemo::card_state)
            .collect()
    }

    /// Removes one SLAM demo by id while keeping at least one demo alive.
    pub(crate) fn remove_slam_demo(&mut self, slam_id: usize) {
        if self.simulations.slam_demos.len() <= 1 {
            return;
        }
        self.simulations
            .slam_demos
            .retain(|slam| slam.id() != slam_id);
    }

    /// Sets the drive mode for one SLAM demo.
    pub(crate) fn set_slam_drive_mode(&mut self, slam_id: usize, drive_mode: SlamDriveMode) {
        if let Some(slam) = self
            .simulations
            .slam_demos
            .iter_mut()
            .find(|slam| slam.id() == slam_id)
        {
            slam.set_drive_mode(drive_mode);
        }
    }

    /// Enables or disables EKF-SLAM for one demo.
    pub(crate) fn set_slam_ekf_enabled(&mut self, slam_id: usize, enabled: bool) {
        if let Some(slam) = self
            .simulations
            .slam_demos
            .iter_mut()
            .find(|slam| slam.id() == slam_id)
        {
            slam.set_ekf_enabled(enabled);
        }
    }

    /// Enables or disables Graph-SLAM for one demo.
    pub(crate) fn set_slam_graph_enabled(&mut self, slam_id: usize, enabled: bool) {
        if let Some(slam) = self
            .simulations
            .slam_demos
            .iter_mut()
            .find(|slam| slam.id() == slam_id)
        {
            slam.set_graph_enabled(enabled);
        }
    }

    /// Applies a partial configuration patch to one SLAM demo.
    pub(crate) fn patch_slam(&mut self, slam_id: usize, patch: SlamPatch) {
        if let Some(slam) = self
            .simulations
            .slam_demos
            .iter_mut()
            .find(|slam| slam.id() == slam_id)
        {
            slam.apply_patch(patch);
        }
    }

    /// Returns the compact state used by the focused MuJoCo web embed.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn mujoco_embed_state(&self) -> MujocoEmbedState {
        self.simulations.mujoco_panel.embed_state()
    }

    /// Sets the selected robot preset for the focused MuJoCo web embed.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn set_mujoco_embed_robot(&mut self, robot: &str) {
        self.simulations.mujoco_panel.set_embed_robot(robot);
    }

    /// Resets the focused MuJoCo viewport camera.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn reset_mujoco_embed_view(&mut self) {
        self.simulations.mujoco_panel.reset_embed_view();
    }

    /// Sets the paused state without mutating any other simulator state.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Restarts the active mode from the same configured parameters.
    ///
    /// This clears the current timebase and dynamic state, but intentionally
    /// keeps user-selected tuning and configuration intact.
    pub fn restart(&mut self) {
        self.time = 0.0;
        self.sim_accumulator = 0.0;
        self.last_tick = None;
        self.reset_state();
    }

    /// Switches the active mode and resets cross-mode transient state.
    ///
    /// Mode changes reset the local time accumulator and the contextual help
    /// tour so each mode starts from a clean interaction baseline.
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

    /// Synchronizes whether the MuJoCo overlay should be active this frame.
    pub(super) fn sync_mujoco_overlay_active(&mut self) {
        let active = self.mode == SimMode::Mujoco;
        self.simulations.mujoco_panel.set_active(active);
    }

    /// Publishes rectangles that should occlude interactive MuJoCo overlays.
    pub(super) fn sync_mujoco_overlay_occlusions(&mut self, rects: &[egui::Rect]) {
        let interactive = self.mode == SimMode::Mujoco && rects.is_empty();
        self.simulations
            .mujoco_panel
            .set_overlay_occlusions(rects, interactive);
    }

    /// Builds the current pendulum noise configuration from top-level toggles.
    fn pendulum_noise_config(&self) -> PendulumNoiseConfig {
        PendulumNoiseConfig {
            enabled: self.pendulum_noise_enabled,
            scale: self.pendulum_noise_scale,
        }
    }

    /// Advances the active mode using a fixed-step wall-clock accumulator.
    ///
    /// All non-MuJoCo modes currently share the pendulum demo's `dt`, while
    /// MuJoCo uses its own panel-defined fixed step. The accumulator is capped
    /// to avoid an unbounded catch-up burst after long frame stalls.
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
                    self.simulations
                        .pendulums
                        .iter_mut()
                        .for_each(InvertedPendulum::tick_training);
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.simulations
                            .pendulums
                            .iter_mut()
                            .for_each(|pendulum| pendulum.step_with_noise(dt, noise));
                    });
                }
                SimMode::Localization => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.simulations
                            .vehicles
                            .iter_mut()
                            .for_each(|vehicle| vehicle.step(dt));
                    });
                }
                SimMode::Mujoco => {
                    let mujoco_dt = self.simulations.mujoco_panel.fixed_dt();
                    self.step_fixed_dt(elapsed, mujoco_dt, |sim, _dt| {
                        sim.simulations.mujoco_panel.update(1, false);
                    });
                }
                SimMode::PathPlanning => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.simulations
                            .planners
                            .iter_mut()
                            .for_each(|planner| planner.step(dt));
                    });
                }
                SimMode::Slam => {
                    self.step_fixed_dt(elapsed, dt, |sim, dt| {
                        sim.simulations
                            .slam_demos
                            .iter_mut()
                            .for_each(|slam| slam.step(dt));
                    });
                }
            }
        }
        if self.mode == SimMode::InvertedPendulum && self.paused {
            self.simulations
                .pendulums
                .iter_mut()
                .for_each(InvertedPendulum::tick_training);
        }
    }

    /// Accumulates elapsed wall-clock time and runs as many fixed-size steps as
    /// needed to catch the simulation up.
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

    /// Resets only dynamic state for the active mode.
    fn reset_state(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.simulations
                    .pendulums
                    .iter_mut()
                    .for_each(|sim| sim.reset_state());
                if let Some((first, rest)) = self.simulations.pendulums.split_first_mut() {
                    rest.iter_mut().for_each(|sim| sim.match_state_with(first));
                }
            }
            SimMode::Localization => {
                self.simulations
                    .vehicles
                    .iter_mut()
                    .for_each(|sim| sim.reset_state());
            }
            SimMode::Mujoco => {
                self.simulations.mujoco_panel.reset_state();
            }
            SimMode::PathPlanning => {
                self.simulations
                    .planners
                    .iter_mut()
                    .for_each(|sim| sim.reset_state());
            }
            SimMode::Slam => {
                self.simulations
                    .slam_demos
                    .iter_mut()
                    .for_each(|sim| sim.reset_state());
            }
        }
    }

    /// Restores the active mode to its full default state.
    pub(super) fn reset_all(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                self.simulations
                    .pendulums
                    .iter_mut()
                    .for_each(|sim| sim.reset_all());
            }
            SimMode::Localization => {
                self.simulations
                    .vehicles
                    .iter_mut()
                    .for_each(|sim| sim.reset_all());
            }
            SimMode::Mujoco => {
                self.simulations.mujoco_panel.reset_all();
            }
            SimMode::PathPlanning => {
                self.simulations
                    .planners
                    .iter_mut()
                    .for_each(|sim| sim.reset_all());
            }
            SimMode::Slam => {
                self.simulations
                    .slam_demos
                    .iter_mut()
                    .for_each(|sim| sim.reset_all());
            }
        }
    }

    /// Adds another simulation instance for the active mode when supported.
    ///
    /// The simulator uses this to enable side-by-side comparisons under a shared
    /// timebase and mostly shared initial conditions.
    pub(super) fn add_simulation(&mut self) {
        match self.mode {
            SimMode::InvertedPendulum => {
                let id = self
                    .simulations
                    .pendulums
                    .iter()
                    .map(|s| s.id())
                    .max()
                    .unwrap_or(0)
                    + 1;
                self.simulations
                    .pendulums
                    .push(InvertedPendulum::new(id, self.time));
            }
            SimMode::Localization => {
                let id = self
                    .simulations
                    .vehicles
                    .iter()
                    .map(|s| s.id())
                    .max()
                    .unwrap_or(0)
                    + 1;
                self.simulations
                    .vehicles
                    .push(ParticleFilter::new(id, self.time));
            }
            SimMode::Mujoco => {}
            SimMode::PathPlanning => {
                let id = self
                    .simulations
                    .planners
                    .iter()
                    .map(|s| s.id())
                    .max()
                    .unwrap_or(0)
                    + 1;
                let mut new_planner = PathPlanning::new(id, self.time);
                new_planner.update_grid_settings(
                    self.path_settings.grid_width,
                    self.path_settings.grid_height,
                    self.path_settings.grid_resolution,
                );
                new_planner.set_env_mode(self.path_settings.env_mode);
                new_planner
                    .set_continuous_obstacle_radius(self.path_settings.continuous_obstacle_radius);

                if let Some(first) = self.simulations.planners.first() {
                    new_planner.copy_state_from(first);
                }
                self.simulations.planners.push(new_planner);
            }
            SimMode::Slam => {
                let id = self
                    .simulations
                    .slam_demos
                    .iter()
                    .map(|s| s.id())
                    .max()
                    .unwrap_or(0)
                    + 1;
                self.simulations
                    .slam_demos
                    .push(SlamDemo::new(id, self.time));
            }
        }
    }

    /// Public wrapper used by web/native UI bridges to add a simulation for
    /// the currently active mode.
    pub fn add_active_simulation(&mut self) {
        self.add_simulation();
    }
}
