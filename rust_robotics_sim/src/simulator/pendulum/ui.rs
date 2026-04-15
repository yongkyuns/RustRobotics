use super::super::ppo_trainer::PpoReplicaStatus;
use super::domain::{Controller, ControllerKind, InvertedPendulum, PENDULUM_FIXED_DT};
use egui::{ComboBox, DragValue, Grid, Ui};
use rust_robotics_core::PolicySnapshot;

impl Controller {
    pub fn options(&mut self, ui: &mut Ui, available_policy: Option<&PolicySnapshot>) {
        match self {
            Self::LQR(model) => {
                ui.vertical(|ui| {
                    ui.label("LQR Parameters:");
                    ui.add(
                        DragValue::new(&mut model.l_bar)
                            .speed(0.01)
                            .range(0.1_f32..=10.0)
                            .prefix("Beam Length: ")
                            .suffix(" m"),
                    );
                    ui.add(
                        DragValue::new(&mut model.m_cart)
                            .speed(0.01)
                            .range(0.1_f32..=3.0)
                            .prefix("Cart Mass: ")
                            .suffix(" kg"),
                    );
                    ui.add(
                        DragValue::new(&mut model.m_ball)
                            .speed(0.01)
                            .range(0.1_f32..=10.0)
                            .prefix("Ball Mass: ")
                            .suffix(" kg"),
                    );
                    ui.label("Weights");
                    ui.add(
                        DragValue::new(model.Q.get_mut(0).unwrap())
                            .speed(0.01)
                            .range(0.0_f32..=100.0)
                            .prefix("Lateral Position: "),
                    );
                    ui.add(
                        DragValue::new(model.Q.get_mut(5).unwrap())
                            .speed(0.01)
                            .range(0.0_f32..=100.0)
                            .prefix("Lateral Velocity: "),
                    );
                    ui.add(
                        DragValue::new(model.Q.get_mut(10).unwrap())
                            .speed(0.01)
                            .range(0.0_f32..=100.0)
                            .prefix("Rod Angle: "),
                    );
                    ui.add(
                        DragValue::new(model.Q.get_mut(15).unwrap())
                            .speed(0.01)
                            .range(0.0_f32..=100.0)
                            .prefix("Rod Angular Vel: "),
                    );
                    ui.add(
                        DragValue::new(model.R.get_mut(0).unwrap())
                            .speed(0.01)
                            .range(0.0_f32..=100.0)
                            .prefix("Control Input: "),
                    );
                });
            }
            Self::PID(pid) => {
                ui.vertical(|ui| {
                    ui.label("PID Parameters:");
                    ui.add(
                        DragValue::new(&mut pid.P)
                            .speed(0.01)
                            .range(0.01_f32..=10000.0)
                            .prefix("P gain: "),
                    );
                    ui.add(
                        DragValue::new(&mut pid.I)
                            .speed(0.01)
                            .range(0.01_f32..=10000.0)
                            .prefix("I gain: "),
                    );
                    ui.add(
                        DragValue::new(&mut pid.D)
                            .speed(0.01)
                            .range(0.01_f32..=10000.0)
                            .prefix("D gain: "),
                    );
                });
            }
            Self::MPC(model) => {
                ui.vertical(|ui| {
                    ui.label("MPC Parameters:");
                    ui.add(
                        DragValue::new(&mut model.l_bar)
                            .speed(0.01)
                            .range(0.1_f32..=10.0)
                            .prefix("Beam Length: ")
                            .suffix(" m"),
                    );
                    ui.add(
                        DragValue::new(&mut model.m_cart)
                            .speed(0.01)
                            .range(0.1_f32..=3.0)
                            .prefix("Cart Mass: ")
                            .suffix(" kg"),
                    );
                    ui.add(
                        DragValue::new(&mut model.m_ball)
                            .speed(0.01)
                            .range(0.1_f32..=10.0)
                            .prefix("Ball Mass: ")
                            .suffix(" kg"),
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    ui.label("(Horizon: 12, Control bounds: ±50N)");
                    #[cfg(target_arch = "wasm32")]
                    ui.label("(Falls back to LQR on web)");
                });
            }
            Self::Policy(policy) => {
                let available = available_policy.is_some();
                ui.vertical(|ui| {
                    ui.label("PPO Policy:");
                    ui.label(format!("Action std: {:.3}", policy.action_std()));
                    ui.label("Trained on this pendulum's PPO trainer state.");
                    if !available {
                        ui.label("No active trainer snapshot is loaded.");
                    }
                });
            }
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Self::LQR(_) => "LQR".to_owned(),
            Self::PID(_) => "PID".to_owned(),
            Self::MPC(_) => "MPC".to_owned(),
            Self::Policy(_) => "PPO Policy".to_owned(),
        }
    }
}

impl InvertedPendulum {
    pub fn options_with_policy(&mut self, ui: &mut Ui) -> bool {
        let mut keep = true;
        if self.controller_selection == ControllerKind::Policy
            || self.trainer_backend.is_initialized()
        {
            self.sync_policy_selection();
        }
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("Pendulum {}", self.id));
                            if ui.small_button("🗙").clicked() {
                                keep = false;
                            }
                        });
                        ui.group(|ui| {
                            ui.label("Cart:");
                            ui.add(
                                DragValue::new(&mut self.model.l_bar)
                                    .speed(0.01)
                                    .range(0.1_f32..=10.0)
                                    .prefix("Beam Length: ")
                                    .suffix(" m"),
                            );
                            ui.add(
                                DragValue::new(&mut self.model.m_cart)
                                    .speed(0.01)
                                    .range(0.1_f32..=3.0)
                                    .prefix("Cart Mass: ")
                                    .suffix(" kg"),
                            );
                            ui.add(
                                DragValue::new(&mut self.model.m_ball)
                                    .speed(0.01)
                                    .range(0.1_f32..=10.0)
                                    .prefix("Ball Mass: ")
                                    .suffix(" kg"),
                            );
                        });
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(220.0);
                                ui.set_max_width(220.0);
                                ui.label("Controller:");
                                let available_policy = self.trainer_backend.snapshot().cloned();
                                let mut selected = self.controller_selection;
                                ui.push_id(self.id, |ui| {
                                    ComboBox::from_label("")
                                        .width(140.0)
                                        .selected_text(selected.label())
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut selected,
                                                ControllerKind::Lqr,
                                                "LQR",
                                            );
                                            ui.selectable_value(
                                                &mut selected,
                                                ControllerKind::Pid,
                                                "PID",
                                            );
                                            ui.selectable_value(
                                                &mut selected,
                                                ControllerKind::Mpc,
                                                "MPC",
                                            );
                                            ui.selectable_value(
                                                &mut selected,
                                                ControllerKind::Policy,
                                                "PPO Policy",
                                            );
                                        });
                                });
                                if selected != self.controller_selection {
                                    self.select_controller_kind(selected);
                                }
                                if self.controller_selection == ControllerKind::Policy {
                                    if self.controller.kind() == ControllerKind::Policy {
                                        self.controller.options(ui, available_policy.as_ref());
                                    } else {
                                        ui.vertical(|ui| {
                                            ui.label("PPO Policy:");
                                            ui.label("No snapshot yet.");
                                        });
                                    }
                                } else {
                                    self.controller.options(ui, available_policy.as_ref());
                                }

                                if self.controller_selection == ControllerKind::Policy {
                                    ui.separator();
                                    ui.collapsing("PPO Trainer", |ui| {
                                        #[cfg(target_arch = "wasm32")]
                                        ui.label("Web: CPU workers.");

                                        ui.horizontal_wrapped(|ui| {
                                            if ui
                                                .button(if self.training_active {
                                                    "Stop"
                                                } else {
                                                    "Train"
                                                })
                                                .clicked()
                                            {
                                                if self.training_active {
                                                    self.stop_training();
                                                } else {
                                                    self.start_training();
                                                }
                                            }
                                            if ui.button("Reset").clicked() {
                                                let was_training = self.training_active;
                                                self.reset_trainer();
                                                if was_training {
                                                    self.start_training();
                                                }
                                            }
                                            if ui.button("Use").clicked() {
                                                if let Some(snapshot) =
                                                    self.trainer_backend.snapshot().cloned()
                                                {
                                                    self.set_policy_controller(&snapshot);
                                                }
                                            }
                                        });

                                        Grid::new(("ppo_trainer_grid", self.id))
                                            .num_columns(2)
                                            .spacing([12.0, 4.0])
                                            .show(ui, |ui| {
                                                ui.label("Parallel");
                                                ui.add(
                                                    DragValue::new(&mut self.parallel_trainers)
                                                        .range(1..=32)
                                                        .speed(1),
                                                );
                                                ui.end_row();

                                                ui.label("Updates/tick");
                                                ui.add(
                                                    DragValue::new(
                                                        &mut self.training_updates_per_tick,
                                                    )
                                                    .range(1..=32)
                                                    .speed(1),
                                                );
                                                ui.end_row();

                                                ui.label("Rollout");
                                                ui.add(
                                                    DragValue::new(
                                                        &mut self.trainer_config.ppo.rollout_steps,
                                                    )
                                                    .range(32..=8192)
                                                    .speed(16),
                                                );
                                                ui.end_row();

                                                ui.label("Epochs");
                                                ui.add(
                                                    DragValue::new(
                                                        &mut self
                                                            .trainer_config
                                                            .ppo
                                                            .epochs_per_update,
                                                    )
                                                    .range(1..=16)
                                                    .speed(1),
                                                );
                                                ui.end_row();

                                                ui.label("LR");
                                                ui.add(
                                                    DragValue::new(
                                                        &mut self.trainer_config.ppo.learning_rate,
                                                    )
                                                    .range(1e-5..=1e-2)
                                                    .speed(1e-4),
                                                );
                                                ui.end_row();

                                                ui.label("Action std");
                                                ui.add(
                                                    DragValue::new(
                                                        &mut self.trainer_config.action_std,
                                                    )
                                                    .range(0.05..=10.0)
                                                    .speed(0.05),
                                                );
                                                ui.end_row();
                                            });

                                        ui.label(format!(
                                            "dt = sim step ({:.3}s)",
                                            PENDULUM_FIXED_DT
                                        ));
                                        ui.label("Changes apply after `Reset`.");

                                        let PpoReplicaStatus { total, ready, busy } =
                                            self.trainer_backend.status();

                                        if let Some(metrics) = self.trainer_backend.metrics() {
                                            if total > 0 {
                                                ui.label(format!(
                                                    "Replicas {}/{}/{}",
                                                    total, ready, busy
                                                ));
                                            }
                                            ui.label(format!(
                                                "Upd {}  Step {}  Ep {}",
                                                metrics.total_updates,
                                                metrics.total_env_steps,
                                                metrics.total_episodes
                                            ));
                                            ui.label(format!(
                                                "Ret {:.2}/{:.2}/{:.2}",
                                                metrics.last_episode_return,
                                                metrics.mean_episode_return,
                                                metrics.best_episode_return
                                            ));
                                            ui.label(format!(
                                                "Loss {:.3}/{:.3}",
                                                metrics.last_policy_loss, metrics.last_value_loss
                                            ));
                                        } else {
                                            ui.label("Trainer not initialized.");
                                        }

                                        if self.training_active && self.trainer_backend.busy() {
                                            ui.label("Training...");
                                        }
                                        if let Some(error) = self.trainer_backend.last_error() {
                                            ui.label(format!("Trainer error: {error}"));
                                        }
                                    });
                                }
                            });
                        });
                    });
                });
            });
        });
        keep
    }
}
