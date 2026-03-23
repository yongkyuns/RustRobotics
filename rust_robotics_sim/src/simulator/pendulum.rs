#![allow(non_snake_case)]

use super::Draw;
use crate::data::{IntoValues, TimeTable};
use crate::prelude::draw_cart;

use egui::{ComboBox, DragValue, Ui};
use egui_plot::{Line, PlotUi};
use rand::Rng;
use rb::inverted_pendulum::*;
use rb::prelude::*;
use rust_robotics_algo as rb;

use super::Simulate;

pub type State = rb::Vector4;

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
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Controller {
    LQR(Model),
    PID(PID),
    MPC(Model),
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
        }
    }
    /// Method to draw onto [`egui`] UI.
    pub fn options(&mut self, ui: &mut Ui) {
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
        }
    }

    /// Output the [`String`] for the currrent controller
    pub fn to_string(&self) -> String {
        match self {
            Self::LQR(_) => "LQR".to_owned(),
            Self::PID(_) => "PID".to_owned(),
            Self::MPC(_) => "MPC".to_owned(),
        }
    }
}

/// Inverted pendulum simulation
pub struct InvertedPendulum {
    state: State,
    controller: Controller,
    model: Model,
    id: usize,
    data: TimeTable,
    time_init: f32,
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

        Self {
            state,
            controller: Controller::lqr(Model::default()),
            model: Model::default(),
            id: 1,
            time_init: 0.0,
            data,
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
        self.controller.reset_state();
        self.data.clear();
    }

    fn reset_all(&mut self) {
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
        let mut keep = true;
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
                                ui.label("Controller:");
                                // `ComboBox` label can't be a static string
                                // due to id clashes when adding multiple `ComboBox`s
                                // ui.push_id is used here to create unique ID
                                ui.push_id(self.id, |ui| {
                                    ComboBox::from_label("")
                                        .selected_text(self.controller.to_string())
                                        .show_ui(ui, |ui| {
                                            for options in [
                                                Controller::lqr(self.model),
                                                Controller::pid(),
                                                Controller::mpc(self.model),
                                            ]
                                            .iter()
                                            {
                                                ui.selectable_value(
                                                    &mut self.controller,
                                                    *options,
                                                    options.to_string(),
                                                );
                                            }
                                        });
                                });
                                self.controller.options(ui);
                            });
                        });
                    });
                });
            });
        });
        keep
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
}
