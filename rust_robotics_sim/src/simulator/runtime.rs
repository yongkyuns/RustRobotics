use super::{
    InvertedPendulum, ParticleFilter, PathPlanning, PendulumNoiseConfig, SimMode, Simulate,
    Simulator, SlamDemo, PENDULUM_FIXED_DT,
};
use web_time::Instant;

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

    pub(super) fn sync_mujoco_overlay_active(&mut self) {
        let active = self.mode == SimMode::Mujoco;
        self.simulations.mujoco_panel.set_active(active);
    }

    pub(super) fn sync_mujoco_overlay_occlusions(&mut self, rects: &[egui::Rect]) {
        let interactive = self.mode == SimMode::Mujoco && rects.is_empty();
        self.simulations
            .mujoco_panel
            .set_overlay_occlusions(rects, interactive);
    }

    fn pendulum_noise_config(&self) -> PendulumNoiseConfig {
        PendulumNoiseConfig {
            enabled: self.pendulum_noise_enabled,
            scale: self.pendulum_noise_scale,
        }
    }

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
}
