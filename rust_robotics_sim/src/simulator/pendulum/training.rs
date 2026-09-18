//! PPO integration for the pendulum simulator mode.
//!
//! This module is the glue between:
//!
//! - the user-facing controller selection in the pendulum UI
//! - the `PpoTrainerCoordinator`, which owns one learner and one or more environments
//! - the live controller instance embedded in `InvertedPendulum`
//!
//! The key rule is that selecting `PPO Policy` in the UI should immediately
//! synchronize the visible controller with the latest available trainer
//! snapshot, even if the underlying trainer is being refreshed asynchronously.
use super::super::Simulate;
use super::domain::{ControllerKind, InvertedPendulum, NoiseConfig, PENDULUM_FIXED_DT};
use rust_robotics_algo::cart_pole::CartPoleParameters;
use rust_robotics_train::PendulumEnvConfig;

impl InvertedPendulum {
    /// Keep physical settings explicit and reject a stale trainer/policy rather
    /// than silently deploying it on a different task. Restart is user-visible.
    pub(crate) fn validate_training_environment(&mut self) {
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        let requested = (self.trainer_config.plant, self.trainer_config.env);
        if self
            .active_training_environment
            .is_some_and(|active| active != requested)
        {
            self.training_active = false;
            self.trainer_backend.destroy();
            self.active_training_environment = None;
            self.policy_observation = None;
            self.policy_environment_stale = true;
            self.last_control_error = Some("Plant/noise changed: restart PPO training.".to_owned());
        }
    }

    /// The global noise control affects BOTH PPO training and live inference.
    /// The original PPO amplitudes are the scale-one reference; no hidden second
    /// PPO disturbance model lives in the viewer.
    pub(crate) fn configure_training_noise(&mut self, noise: NoiseConfig) {
        let scale = if noise.enabled {
            noise.scale.max(0.0)
        } else {
            0.0
        };
        let base = PendulumEnvConfig::default();
        let env = &mut self.trainer_config.env;
        let previous = *env;
        env.observation_position_noise_m = base.observation_position_noise_m * scale;
        env.observation_velocity_noise_mps = base.observation_velocity_noise_mps * scale;
        env.observation_angle_noise_rad = base.observation_angle_noise_rad * scale;
        env.observation_angular_velocity_noise_radps =
            base.observation_angular_velocity_noise_radps * scale;
        env.action_noise_force_n = base.action_noise_force_n * scale;
        env.disturbance_force_n = base.disturbance_force_n * scale;
        env.disturbance_probability_per_step = if scale > 0.0 {
            base.disturbance_probability_per_step
        } else {
            0.0
        };
        if previous != *env {
            self.policy_observation = None;
        }
        self.validate_training_environment();
    }

    /// Advances the embedded PPO coordinator if training is active and applies
    /// any newly available policy snapshot to the live controller.
    pub fn tick_training(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale {
            return;
        }
        if self.training_active {
            self.trainer_config.env.dt = PENDULUM_FIXED_DT;
            let updates = self.training_updates_per_tick.max(1);
            self.trainer_backend.tick(updates);
        } else {
            // A previously requested browser update may finish after Stop.
            // Poll it even when the focused UI does not draw an egui panel.
            // This publishes its final snapshot/status, but schedules no work.
            self.trainer_backend.refresh();
        }
        if let Some(snapshot) = self.trainer_backend.snapshot().cloned() {
            if self.controller_selection == ControllerKind::Policy
                && self.controller.kind() == ControllerKind::Policy
            {
                self.controller.sync_policy(&snapshot);
            } else if self.controller_selection == ControllerKind::Policy {
                self.set_policy_controller(&snapshot);
                self.reset_state();
            }
        }
    }

    /// Rebuilds the learner and its environment streams from the current configuration.
    pub(crate) fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        self.active_training_environment =
            Some((self.trainer_config.plant, self.trainer_config.env));
        self.policy_environment_stale = false;
        self.last_control_error = None;
        self.policy_observation = None;
        self.trainer_backend
            .reset(&self.trainer_config, self.parallel_trainers);
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    /// Enables PPO training and switches the active controller to the learned
    /// policy as soon as a snapshot is available.
    pub(crate) fn start_training(&mut self) {
        self.validate_training_environment();
        self.controller_selection = ControllerKind::Policy;
        if !self.trainer_backend.is_initialized() {
            self.reset_trainer();
        }
        self.training_active = true;
        if let Some(snapshot) = self.trainer_backend.snapshot().cloned() {
            self.set_policy_controller(&snapshot);
            self.reset_state();
        }
    }

    /// Stops PPO updates while keeping the latest available snapshot.
    pub(crate) fn stop_training(&mut self) {
        self.training_active = false;
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    /// Synchronizes UI controller selection with the latest trainer snapshot.
    pub(crate) fn sync_policy_selection(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale {
            return;
        }
        if self.controller_selection != ControllerKind::Policy {
            return;
        }

        if !self.trainer_backend.is_initialized() {
            self.reset_trainer();
        } else {
            self.trainer_backend.refresh();
        }

        let Some(snapshot) = self.trainer_backend.snapshot().cloned() else {
            return;
        };

        if self.controller.kind() == ControllerKind::Policy {
            self.controller.sync_policy(&snapshot);
        } else {
            self.set_policy_controller(&snapshot);
        }
    }

    /// Handles a controller-kind selection change coming from the UI.
    pub(crate) fn select_controller_kind(&mut self, selected: ControllerKind) {
        if selected == self.controller_selection {
            if selected == ControllerKind::Policy {
                self.sync_policy_selection();
            }
            return;
        }

        self.controller_selection = selected;
        if selected == ControllerKind::Policy {
            self.sync_policy_selection();
        } else {
            let available_policy = self.trainer_backend.snapshot().cloned();
            self.stop_training();
            self.controller
                .set_kind(selected, self.model, available_policy.as_ref());
        }
    }
}
