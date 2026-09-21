//! PPO integration for the pendulum simulator mode.
//!
//! Physical/environment settings define a training contract. Changes invalidate
//! an incompatible learner and require an explicit restart. Policy publication
//! does not reset a running episode; activation and explicit restart do.
use super::super::Simulate;
use super::domain::{
    ControllerKind, InvertedPendulum, NoiseConfig, PpoLearnerSettings, PENDULUM_FIXED_DT,
};
use rust_robotics_algo::cart_pole::CartPoleParameters;
use rust_robotics_train::PendulumEnvConfig;

impl InvertedPendulum {
    /// Reject a stale trainer/policy rather than silently deploying it on a
    /// different task. The worker is destroyed so late results cannot reappear.
    pub(crate) fn validate_training_environment(&mut self) {
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        let requested_environment = (self.trainer_config.plant, self.trainer_config.env);
        if self
            .active_training_environment
            .is_some_and(|active| active != requested_environment)
        {
            self.training_active = false;
            self.trainer_backend.destroy();
            self.active_training_environment = None;
            self.active_training_settings = None;
            self.policy_observation = None;
            self.policy_environment_stale = true;
            self.last_control_error = Some("Plant/noise changed: restart PPO training.".to_owned());
            return;
        }

        let requested_settings =
            PpoLearnerSettings::from_config(&self.trainer_config, self.parallel_trainers);
        if self
            .active_training_settings
            .is_some_and(|active| active != requested_settings)
        {
            // Optimizer/collection settings belong to the learner, not the physical
            // deployment contract. Tear down the stale learner while allowing the
            // already-published policy to keep driving the unchanged plant.
            self.training_active = false;
            self.trainer_backend.destroy();
            self.active_training_environment = None;
            self.active_training_settings = None;
            self.policy_environment_stale = false;
            self.last_control_error =
                Some("PPO trainer settings changed: restart PPO training.".to_owned());
        }
    }

    /// Scale one uses the original PPO noise amplitudes, shared with the viewer.
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

    /// Advance training, or poll the last requested browser update after Stop.
    /// Neither ordinary polling nor snapshot publication resets an episode.
    pub fn tick_training(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale {
            return;
        }
        if self.training_active {
            self.trainer_config.env.dt = PENDULUM_FIXED_DT;
            self.trainer_backend
                .tick(self.training_updates_per_tick.max(1));
        } else {
            self.trainer_backend.refresh();
        }
        if self.controller_selection != ControllerKind::Policy {
            return;
        }
        if let Some(snapshot) = self.trainer_backend.snapshot().cloned() {
            if self.controller.kind() == ControllerKind::Policy {
                self.controller.sync_policy(&snapshot);
            } else {
                self.set_policy_controller(&snapshot);
            }
        }
    }

    /// Create a matching learner and reset an already-active policy episode now,
    /// including when a browser worker has not yet published its new snapshot.
    pub(crate) fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        self.active_training_environment =
            Some((self.trainer_config.plant, self.trainer_config.env));
        self.active_training_settings = Some(PpoLearnerSettings::from_config(
            &self.trainer_config,
            self.parallel_trainers,
        ));
        self.policy_environment_stale = false;
        self.last_control_error = None;
        self.policy_observation = None;
        self.trainer_backend
            .reset(&self.trainer_config, self.parallel_trainers);
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
        if self.controller.kind() == ControllerKind::Policy {
            self.reset_state();
        }
    }

    /// An explicit Start activates PPO and starts a fresh matching episode.
    pub(crate) fn start_training(&mut self) {
        self.validate_training_environment();
        self.controller_selection = ControllerKind::Policy;
        let initialize = !self.trainer_backend.is_initialized();
        if initialize {
            self.reset_trainer();
        }
        self.training_active = true;
        if let Some(snapshot) = self.trainer_backend.snapshot().cloned() {
            let already_policy = self.controller.kind() == ControllerKind::Policy;
            self.set_policy_controller(&snapshot);
            // reset_trainer handles an existing policy on initialization;
            // set_policy_controller handles entering Policy from another mode.
            if already_policy && !initialize {
                self.reset_state();
            }
        }
    }

    pub(crate) fn stop_training(&mut self) {
        self.training_active = false;
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    pub(crate) fn sync_policy_selection(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale || self.controller_selection != ControllerKind::Policy {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use rust_robotics_algo::Vector4;

    #[test]
    fn replacing_a_policy_resets_before_the_new_worker_can_publish() {
        let mut sim = InvertedPendulum::default();
        sim.start_training();
        sim.state = Vector4::new(3.0, 5.0, 0.8, 6.0);
        sim.visual_episode_steps = 23;
        sim.model.l_bar = 0.7;
        sim.validate_training_environment();
        assert!(sim.policy_environment_stale);
        sim.reset_trainer();
        assert_eq!(sim.visual_episode_steps, 0);
        assert!(sim.state[0].abs() <= sim.trainer_config.env.reset_position_range_m);
        assert!(sim.state[2].abs() <= sim.trainer_config.env.reset_angle_range_rad);
        assert!(sim.policy_observation.is_some());
        // No published snapshot models the actual asynchronous creation interval.
        sim.trainer_backend.destroy();
        let reset_state = sim.state;
        sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut StdRng::seed_from_u64(17));
        assert_eq!(sim.state, reset_state);
        assert!(sim.last_control_error().unwrap().contains("Waiting"));
    }
}
