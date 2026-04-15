use super::super::ppo_trainer::PpoTrainerCoordinator;
use super::super::{Draw, Simulate};
use crate::data::{IntoValues, TimeTable};
use crate::prelude::draw_cart;

use egui::Ui;
use egui_plot::{Line, PlotUi};
use rand::Rng;
use rb::inverted_pendulum::*;
use rb::prelude::*;
use rust_robotics_algo as rb;
use rust_robotics_core::PolicySnapshot;
use rust_robotics_train::PpoTrainerConfig;

pub type State = rb::Vector4;
pub const PENDULUM_FIXED_DT: f32 = 0.01;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NoiseConfig {
    pub enabled: bool,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NoiseProfile {
    pub(crate) position_m: f32,
    pub(crate) velocity_mps: f32,
    pub(crate) angle_rad: f32,
    pub(crate) angular_velocity_radps: f32,
    pub(crate) force_n: f32,
}

impl NoiseConfig {
    pub(super) fn is_active(self) -> bool {
        self.enabled && self.scale > 0.0
    }

    pub(super) fn profile(self) -> NoiseProfile {
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

#[derive(Debug, PartialEq, Clone)]
pub enum Controller {
    LQR(Model),
    PID(PID),
    MPC(Model),
    Policy(PolicySnapshot),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControllerKind {
    Lqr,
    Pid,
    Mpc,
    Policy,
}

impl ControllerKind {
    pub(crate) fn label(self) -> &'static str {
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

    pub fn lqr(model: Model) -> Self {
        Self::LQR(model)
    }

    pub fn pid() -> Self {
        Self::PID(PID::with_gains(25.0, 3.0, 3.0))
    }

    pub fn mpc(model: Model) -> Self {
        Self::MPC(model)
    }

    pub fn policy(snapshot: PolicySnapshot) -> Self {
        Self::Policy(snapshot)
    }

    pub(crate) fn kind(&self) -> ControllerKind {
        match self {
            Self::LQR(_) => ControllerKind::Lqr,
            Self::PID(_) => ControllerKind::Pid,
            Self::MPC(_) => ControllerKind::Mpc,
            Self::Policy(_) => ControllerKind::Policy,
        }
    }

    pub(crate) fn set_kind(
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

    pub fn reset_state(&mut self) {
        match self {
            Self::LQR(_) => (),
            Self::PID(pid) => pid.reset_state(),
            Self::MPC(_) => (),
            Self::Policy(_) => (),
        }
    }

    pub fn reset_all(&mut self) {
        match self {
            Self::LQR(_) => *self = Self::lqr(Model::default()),
            Self::PID(_) => *self = Self::pid(),
            Self::MPC(_) => *self = Self::mpc(Model::default()),
            Self::Policy(policy) => *self = Self::policy(policy.clone()),
        }
    }
}

pub struct InvertedPendulum {
    pub(crate) state: State,
    pub(crate) controller: Controller,
    pub(crate) controller_selection: ControllerKind,
    pub(crate) model: Model,
    pub(crate) id: usize,
    pub(crate) data: TimeTable,
    pub(crate) time_init: f32,
    pub(crate) visual_episode_steps: usize,
    pub(crate) trainer_config: PpoTrainerConfig,
    pub(crate) trainer_backend: PpoTrainerCoordinator,
    pub(crate) training_active: bool,
    pub(crate) training_updates_per_tick: usize,
    pub(crate) parallel_trainers: usize,
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

    pub(crate) fn visual_episode_done(&self) -> bool {
        let env = &self.trainer_config.env;
        self.state[2].abs() > env.max_angle_rad
            || self.state[0].abs() > env.max_position_m
            || self.visual_episode_steps >= env.max_steps
    }
}

impl Simulate for InvertedPendulum {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.state
    }

    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<State>() {
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

pub(super) fn rand(max: f32) -> f32 {
    rand::thread_rng().gen_range(-max..max)
}
