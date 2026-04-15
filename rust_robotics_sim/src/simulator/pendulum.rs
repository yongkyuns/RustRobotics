#![allow(non_snake_case)]

use super::ppo_trainer::{PpoReplicaStatus, PpoTrainerCoordinator};
use super::Draw;
use crate::data::{IntoValues, TimeTable};
use crate::prelude::draw_cart;

use egui::{ComboBox, DragValue, Grid, Ui};
use egui_plot::{Line, PlotUi};
use rand::Rng;
use rb::inverted_pendulum::*;
use rb::prelude::*;
use rust_robotics_algo as rb;
use rust_robotics_train::{PolicySnapshot, PpoTrainerConfig};

use super::Simulate;

pub type State = rb::Vector4;
pub const PENDULUM_FIXED_DT: f32 = 0.01;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NoiseConfig {
    pub enabled: bool,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct NoiseProfile {
    position_m: f32,
    velocity_mps: f32,
    angle_rad: f32,
    angular_velocity_radps: f32,
    force_n: f32,
}

impl NoiseConfig {
    fn is_active(self) -> bool {
        self.enabled && self.scale > 0.0
    }

    fn profile(self) -> NoiseProfile {
        let scale = self.scale.max(0.0);
        NoiseProfile {
            position_m: 0.005 * scale,
            velocity_mps: 0.02 * scale,
            angle_rad: 0.004 * scale,
            angular_velocity_radps: 0.02 * scale,
            force_n: 0.2 * scale,
        }
    }
}

/// Controller for the inverted pendulum simulation
#[derive(Debug, PartialEq, Clone)]
pub enum Controller {
    LQR(Model),
    PID(PID),
    MPC(Model),
    Policy(PolicySnapshot),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControllerKind {
    Lqr,
    Pid,
    Mpc,
    Policy,
}

impl ControllerKind {
    fn label(self) -> &'static str {
        match self {
            Self::Lqr => "LQR",
            Self::Pid => "PID",
            Self::Mpc => "MPC",
            Self::Policy => "PPO Policy",
        }
    }
}

impl Controller {
    pub fn control(&mut self, x: State, dt: f32) -> f32 {
        match self {
            Self::LQR(model) => *model.control(x, dt).index(0),
            Self::PID(pid) => pid.control(0.0 - x[2], dt),
            Self::MPC(model) => {
                #[cfg(target_arch = "wasm32")]
                let start = web_sys::window().unwrap().performance().unwrap().now();

                let u = mpc_control(x, *model, dt);

                #[cfg(target_arch = "wasm32")]
                {
                    let elapsed = web_sys::window().unwrap().performance().unwrap().now() - start;
                    web_sys::console::log_1(&format!("MPC took {:.2}ms", elapsed).into());
                }

                u
            }
            Self::Policy(policy) => policy.act([x[0], x[1], x[2], x[3]]),
        }
    }
    /// Instantiate a new LQR controller for [`InvertedPendulum`]
    pub fn lqr(model: Model) -> Self {
        Self::LQR(model)
    }
    /// Instantiate a new PID controller for [`InvertedPendulum`]
    pub fn pid() -> Self {
        Self::PID(PID::with_gains(25.0, 3.0, 3.0))
    }
    /// Instantiate a new MPC controller for [`InvertedPendulum`]
    pub fn mpc(model: Model) -> Self {
        Self::MPC(model)
    }
    /// Instantiate a learned policy controller for [`InvertedPendulum`]
    pub fn policy(snapshot: PolicySnapshot) -> Self {
        Self::Policy(snapshot)
    }

    fn kind(&self) -> ControllerKind {
        match self {
            Self::LQR(_) => ControllerKind::Lqr,
            Self::PID(_) => ControllerKind::Pid,
            Self::MPC(_) => ControllerKind::Mpc,
            Self::Policy(_) => ControllerKind::Policy,
        }
    }

    fn set_kind(
        &mut self,
        kind: ControllerKind,
        model: Model,
        available_policy: Option<&PolicySnapshot>,
    ) {
        match kind {
            ControllerKind::Lqr => *self = Self::lqr(model),
            ControllerKind::Pid => *self = Self::pid(),
            ControllerKind::Mpc => *self = Self::mpc(model),
            ControllerKind::Policy => {
                if let Some(policy) = available_policy {
                    *self = Self::policy(policy.clone());
                }
            }
        }
    }

    pub fn sync_policy(&mut self, snapshot: &PolicySnapshot) {
        if let Self::Policy(policy) = self {
            *policy = snapshot.clone();
        }
    }
    /// Reset the states of the current [`Controller`]
    ///
    /// If there are parameters related to the controller (e.g. PID gains),
    /// this method retains those parameters unchanged and only resets the
    /// internal states (e.g. integral error in [`PID controller`](PID))
    pub fn reset_state(&mut self) {
        match self {
            Self::LQR(_) => (),
            Self::PID(pid) => pid.reset_state(),
            Self::MPC(_) => (),
            Self::Policy(_) => (),
        }
    }
    /// Reset the states and any parameters to it's default values
    ///
    /// This method only retains the [`Controller`] selection but resets
    /// any internal states and parameters. If you want to only reset the
    /// the state of a controller (e.g. integral error of PID control), use
    /// [`reset_state`](Controller::reset_state) instead.
    pub fn reset_all(&mut self) {
        match self {
            Self::LQR(_) => *self = Self::lqr(Model::default()),
            Self::PID(_) => *self = Self::pid(),
            Self::MPC(_) => *self = Self::mpc(Model::default()),
            Self::Policy(policy) => *self = Self::policy(policy.clone()),
        }
    }
    /// Method to draw onto [`egui`] UI.
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

    /// Output the [`String`] for the currrent controller
    pub fn to_string(&self) -> String {
        match self {
            Self::LQR(_) => "LQR".to_owned(),
            Self::PID(_) => "PID".to_owned(),
            Self::MPC(_) => "MPC".to_owned(),
            Self::Policy(_) => "PPO Policy".to_owned(),
        }
    }
}

/// Inverted pendulum simulation
pub struct InvertedPendulum {
    state: State,
    controller: Controller,
    controller_selection: ControllerKind,
    model: Model,
    id: usize,
    data: TimeTable,
    time_init: f32,
    visual_episode_steps: usize,
    trainer_config: PpoTrainerConfig,
    trainer_backend: PpoTrainerCoordinator,
    training_active: bool,
    training_updates_per_tick: usize,
    parallel_trainers: usize,
}

impl Default for InvertedPendulum {
    fn default() -> Self {
        let state = vector![0., 0., rand(0.4), 0.];
        let data = TimeTable::init_with_names(vec![
            "Lateral Position",
            "Lateral Velocity",
            "Rod Angle",
            "Rod Angular Velocity",
            "Control Input",
        ]);
        let mut trainer_config = PpoTrainerConfig::default();
        trainer_config.env.dt = PENDULUM_FIXED_DT;

        Self {
            state,
            controller: Controller::lqr(Model::default()),
            controller_selection: ControllerKind::Lqr,
            model: Model::default(),
            id: 1,
            time_init: 0.0,
            data,
            visual_episode_steps: 0,
            trainer_config,
            trainer_backend: PpoTrainerCoordinator::default(),
            training_active: false,
            training_updates_per_tick: 1,
            parallel_trainers: 1,
        }
    }
}

impl InvertedPendulum {
    pub const SIGNAL_LABELS: [&'static str; 5] = [
        "Lateral Position",
        "Lateral Velocity",
        "Rod Angle",
        "Rod Angular Velocity",
        "Control Input",
    ];

    pub fn new(id: usize, time: f32) -> Self {
        Self {
            id,
            time_init: time,
            ..Default::default()
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn x_position(&self) -> f32 {
        self.state[0]
    }

    pub fn rod_angle(&self) -> f32 {
        self.state[2]
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn plot_signal(&self, plot_ui: &mut PlotUi<'_>, signal_index: usize) {
        if signal_index >= Self::SIGNAL_LABELS.len() {
            return;
        }

        let line_name = format!("{}_{}", Self::SIGNAL_LABELS[signal_index], self.id);
        self.data
            .values_shifted(signal_index, self.time_init, 0.0)
            .map(|values| plot_ui.line(Line::new(&line_name, values)));
    }

    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        let mut measured_state = self.state;
        if noise.is_active() {
            let profile = noise.profile();
            measured_state[0] += rand(profile.position_m);
            measured_state[1] += rand(profile.velocity_mps);
            measured_state[2] += rand(profile.angle_rad);
            measured_state[3] += rand(profile.angular_velocity_radps);
        }

        let (A, B) = self.model.model(dt);
        let control_command = self.controller.control(measured_state, dt);
        let applied_control = if noise.is_active() {
            control_command + rand(noise.profile().force_n)
        } else {
            control_command
        };

        self.state = A * self.state + B * applied_control;
        self.visual_episode_steps += 1;
        self.data.add(
            self.data.time_last() + dt,
            vec![
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                applied_control,
            ],
        );

        if self.training_active && self.visual_episode_done() {
            self.reset_state();
        }
    }

    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        self.controller = Controller::policy(snapshot.clone());
    }

    pub fn sync_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        self.controller.sync_policy(snapshot);
    }

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

    fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        self.trainer_backend
            .reset(&self.trainer_config, self.parallel_trainers);
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    fn start_training(&mut self) {
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

    fn stop_training(&mut self) {
        self.training_active = false;
        if let Some(snapshot) = self.trainer_backend.snapshot() {
            self.controller.sync_policy(snapshot);
        }
    }

    fn sync_policy_selection(&mut self) {
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

    fn select_controller_kind(&mut self, selected: ControllerKind) {
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

    fn visual_episode_done(&self) -> bool {
        let env = &self.trainer_config.env;
        self.state[2].abs() > env.max_angle_rad
            || self.state[0].abs() > env.max_position_m
            || self.visual_episode_steps >= env.max_steps
    }

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

impl Simulate for InvertedPendulum {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.state
    }

    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<State>() {
            // Then set self's data from `other` if the type matches
            self.state.clone_from(data);
        }
    }

    fn step(&mut self, dt: f32) {
        self.step_with_noise(dt, NoiseConfig::default());
    }

    fn reset_state(&mut self) {
        self.state = vector![0., 0., rand(0.4), 0.];
        self.time_init = 0.0;
        self.visual_episode_steps = 0;
        self.controller.reset_state();
        self.data.clear();
    }

    fn reset_all(&mut self) {
        self.trainer_backend.destroy();
        *self = Self::default();
    }
}

impl Draw for InvertedPendulum {
    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        let names: Vec<String> = self
            .data
            .names()
            .iter()
            .map(|name| format!("{}_{}", name, self.id))
            .collect();

        (0..self.data.ncols()).for_each(|i| {
            self.data
                .values_shifted(i, self.time_init, 0.0)
                .map(|values| plot_ui.line(Line::new(&names[i], values)));
        });
    }

    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        draw_cart(
            plot_ui,
            self.x_position(),
            self.rod_angle(),
            &self.model,
            &format!("Cart {}", self.id),
        );
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        self.options_with_policy(ui)
    }
}

pub fn rand(max: f32) -> f32 {
    rand::thread_rng().gen_range(-max..max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_noise_is_a_noop() {
        let config = NoiseConfig {
            enabled: false,
            scale: 1.0,
        };
        let mut sim = InvertedPendulum::default();
        sim.state = vector![0.1, -0.2, 0.05, 0.3];
        let expected = {
            let (a, b) = sim.model.model(0.01);
            let u = sim.controller.control(sim.state, 0.01);
            a * sim.state + b * u
        };

        sim.step_with_noise(0.01, config);

        assert_eq!(sim.state, expected);
    }

    #[test]
    fn noise_profile_uses_signal_specific_units() {
        let profile = NoiseConfig {
            enabled: true,
            scale: 1.0,
        }
        .profile();

        assert_eq!(profile.position_m, 0.005);
        assert_eq!(profile.velocity_mps, 0.02);
        assert_eq!(profile.angle_rad, 0.004);
        assert_eq!(profile.angular_velocity_radps, 0.02);
        assert_eq!(profile.force_n, 0.2);
    }

    #[test]
    fn selecting_policy_initializes_policy_controller() {
        let mut sim = InvertedPendulum::default();

        sim.select_controller_kind(ControllerKind::Policy);

        assert_eq!(sim.controller_selection, ControllerKind::Policy);
        assert_eq!(sim.controller.kind(), ControllerKind::Policy);
        assert!(sim.trainer_backend.is_initialized());
    }
}
