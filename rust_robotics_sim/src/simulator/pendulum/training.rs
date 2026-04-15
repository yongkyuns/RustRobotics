use super::super::Simulate;
use super::domain::{ControllerKind, InvertedPendulum, PENDULUM_FIXED_DT};

impl InvertedPendulum {
    pub fn tick_training(&mut self) {
        if !self.training_active {
            return;
        }

        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        let updates = self.training_updates_per_tick.max(1);
        self.trainer_backend.tick(updates);
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

    pub(crate) fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        self.trainer_backend
            .reset(&self.trainer_config, self.parallel_trainers);
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    pub(crate) fn start_training(&mut self) {
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

    pub(crate) fn stop_training(&mut self) {
        self.training_active = false;
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    pub(crate) fn sync_policy_selection(&mut self) {
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
