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

/// Controller for the inverted pendulum simulation
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Controller {
    LQR(Model),
    PID(PID),
}

impl Controller {
    pub fn control(&mut self, x: State, dt: f32) -> f32 {
        match self {
            Self::LQR(model) => *model.control(x, dt).index(0),
            Self::PID(pid) => pid.control(0.0 - x[2], dt),
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
    /// Reset the states of the current [`Controller`]
    ///
    /// If there are parameters related to the controller (e.g. PID gains),
    /// this method retains those parameters unchanged and only resets the
    /// internal states (e.g. integral error in [`PID controller`](PID))
    pub fn reset_state(&mut self) {
        match self {
            Self::LQR(_) => (),
            Self::PID(pid) => pid.reset_state(),
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
        }
    }

    /// Output the [`String`] for the currrent controller
    pub fn to_string(&self) -> String {
        match self {
            Self::LQR(_) => "LQR".to_owned(),
            Self::PID(_) => "PID".to_owned(),
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
    pub fn new(id: usize, time: f32) -> Self {
        Self {
            id,
            time_init: time,
            ..Default::default()
        }
    }

    pub fn x_position(&self) -> f32 {
        self.state[0]
    }

    pub fn rod_angle(&self) -> f32 {
        self.state[2]
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
        let mut x = self.state.clone();
        let (A, B) = self.model.model(dt);

        // Compute control command
        let u = self.controller.control(x, dt);

        // Update simulation based on control input
        x = A * x + B * u;
        self.state = x;

        // Log data
        self.data.add(
            self.data.time_last() + dt,
            vec![
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                u,
            ],
        );
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

    fn options(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.vertical(|ui| {
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
                                            for options in
                                                [Controller::lqr(self.model), Controller::pid()]
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
    }
}

pub fn rand(max: f32) -> f32 {
    rand::thread_rng().gen_range(-max..max)
}
