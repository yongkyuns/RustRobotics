use super::*;

use crate::data::VehiclePlot;
use crate::item::draw_vehicle;

use egui::{DragValue, Ui};
use egui_plot::{Line, LineStyle, PlotPoints, PlotUi, Points};
use rb::localization::{particle_filter::*, StateVector};
use rb::prelude::*;
use rust_robotics_algo as rb;

pub type State = rb::Vector4;

/// Maximum history length for trajectory visualization
const HISTORY_LEN: usize = 1000;

/// Landmark positions for observation
const MARKERS: [rb::Vector2; 4] = [
    vector![10.0_f32, 0.0_f32],
    vector![10.0, 10.0],
    vector![0.0, 15.0],
    vector![-5.0, 20.0],
];

/// Configurable parameters for particle filter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PFConfig {
    /// Forward velocity (m/s)
    pub velocity: f32,
    /// Yaw rate (rad/s)
    pub yaw_rate: f32,
    /// Maximum detection range for landmarks
    pub max_range: f32,
    /// Observation noise (Q_sim)
    pub obs_noise: f32,
    /// Motion noise - velocity component
    pub motion_noise_v: f32,
    /// Motion noise - yaw rate component (degrees)
    pub motion_noise_yaw: f32,
}

impl Default for PFConfig {
    fn default() -> Self {
        Self {
            velocity: 1.0,
            yaw_rate: 0.1,
            max_range: MAX_RANGE,
            obs_noise: 0.2,
            motion_noise_v: 1.0,
            motion_noise_yaw: 40.0,
        }
    }
}

pub struct ParticleFilter {
    /// Estimated state from particle filter
    x_est: State,
    /// True state (ground truth)
    x_true: State,
    /// Dead reckoning state
    x_dr: State,
    /// Estimation covariance
    p_est: rb::Matrix3,
    /// Particle weights
    pw: PW,
    /// Particle states
    px: PX,
    /// History of estimated states
    h_x_est: Vec<State>,
    /// History of true states
    h_x_true: Vec<State>,
    /// History of dead reckoning states
    h_x_dr: Vec<State>,
    /// History of estimation errors
    h_err_est: Vec<f32>,
    /// History of dead reckoning errors
    h_err_dr: Vec<f32>,
    /// Simulation time step counter
    step_count: usize,
    /// Unique ID for this simulation instance
    id: usize,
    /// Configuration parameters
    config: PFConfig,
}

impl ParticleFilter {
    pub fn new(id: usize, _time: f32) -> Self {
        Self {
            x_est: zeros!(4, 1),
            x_true: zeros!(4, 1),
            x_dr: zeros!(4, 1),
            p_est: zeros!(3, 3),
            pw: ones!(1, NP) * (1. / NP as f32),
            px: zeros!(4, NP),
            h_x_est: vec![zeros!(4, 1)],
            h_x_true: vec![zeros!(4, 1)],
            h_x_dr: vec![zeros!(4, 1)],
            h_err_est: vec![0.0],
            h_err_dr: vec![0.0],
            step_count: 0,
            id,
            config: PFConfig::default(),
        }
    }

    fn update_history(&mut self) {
        self.h_x_est.push(self.x_est);
        self.h_x_true.push(self.x_true);
        self.h_x_dr.push(self.x_dr);

        // Calculate and store errors
        let err_est = ((self.x_true.x() - self.x_est.x()).powi(2)
            + (self.x_true.y() - self.x_est.y()).powi(2))
        .sqrt();
        let err_dr = ((self.x_true.x() - self.x_dr.x()).powi(2)
            + (self.x_true.y() - self.x_dr.y()).powi(2))
        .sqrt();
        self.h_err_est.push(err_est);
        self.h_err_dr.push(err_dr);

        self.step_count += 1;

        if self.h_x_est.len() > HISTORY_LEN {
            self.h_x_est.remove(0);
            self.h_x_true.remove(0);
            self.h_x_dr.remove(0);
            self.h_err_est.remove(0);
            self.h_err_dr.remove(0);
        }
    }

    fn calc_input(&self) -> rb::Vector2 {
        vector![self.config.velocity, self.config.yaw_rate]
    }
}

impl Default for ParticleFilter {
    fn default() -> Self {
        Self::new(1, 0.0)
    }
}

impl Simulate for ParticleFilter {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.x_true
    }

    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<State>() {
            self.x_true.clone_from(data);
        }
    }

    fn step(&mut self, dt: f32) {
        let u = self.calc_input();
        let (z, ud) = observation(&mut self.x_true, &mut self.x_dr, u, &MARKERS, dt);
        self.p_est = pf_localization(&mut self.x_est, &mut self.px, &mut self.pw, z, ud, dt);
        self.update_history();
    }

    fn reset_state(&mut self) {
        // Reset all states to origin
        self.x_est = zeros!(4, 1);
        self.x_true = zeros!(4, 1);
        self.x_dr = zeros!(4, 1);
        self.p_est = zeros!(3, 3);
        // Reset particles
        self.pw = ones!(1, NP) * (1. / NP as f32);
        self.px = zeros!(4, NP);
        // Clear history
        self.h_x_est = vec![zeros!(4, 1)];
        self.h_x_true = vec![zeros!(4, 1)];
        self.h_x_dr = vec![zeros!(4, 1)];
        self.h_err_est = vec![0.0];
        self.h_err_dr = vec![0.0];
        self.step_count = 0;
    }

    fn reset_all(&mut self) {
        self.reset_state();
        self.config = PFConfig::default();
    }
}

impl Draw for ParticleFilter {
    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        // Calculate time offset for x-axis (based on step count and history length)
        let start_step = if self.step_count > HISTORY_LEN {
            self.step_count - HISTORY_LEN
        } else {
            0
        };

        // Create error plot data with time on x-axis
        let est_error_points: PlotPoints = self
            .h_err_est
            .iter()
            .enumerate()
            .map(|(i, &err)| [(start_step + i) as f64 * 0.01, err as f64])
            .collect();

        let dr_error_points: PlotPoints = self
            .h_err_dr
            .iter()
            .enumerate()
            .map(|(i, &err)| [(start_step + i) as f64 * 0.01, err as f64])
            .collect();

        plot_ui.line(
            Line::new(format!("PF Est Error {}", self.id), est_error_points)
                .name(format!("Estimation Error {} (m)", self.id)),
        );
        plot_ui.line(
            Line::new(format!("DR Error {}", self.id), dr_error_points)
                .name(format!("Dead Reckoning Error {} (m)", self.id)),
        );
    }

    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        // Draw particles
        plot_ui.points(Points::new(
            format!("Particles {}", self.id),
            PlotPoints::new(
                self.px
                    .column_iter()
                    .map(|state| [
                        *state.get(0).unwrap() as f64,
                        *state.get(1).unwrap() as f64,
                    ])
                    .collect(),
            ),
        ));

        // Draw landmarks and detection lines
        MARKERS.iter().for_each(|marker| {
            plot_ui.points(Points::new("Landmarks", marker_values()).radius(5.0));
            if is_detected(marker, &self.x_true, self.config.max_range) {
                plot_ui.line(
                    Line::new("", values_from_marker_state(marker, &self.x_true))
                        .style(LineStyle::Dotted { spacing: 10.0 }),
                );
            }
        });

        // Draw trajectories
        plot_ui.line(Line::new(
            format!("True Path {}", self.id),
            self.h_x_true.positions(),
        ));
        draw_vehicle(
            plot_ui,
            self.x_true,
            &format!("Vehicle {} (True)", self.id),
        );

        plot_ui.line(Line::new(
            format!("DR Path {}", self.id),
            self.h_x_dr.positions(),
        ));
        draw_vehicle(
            plot_ui,
            self.x_dr,
            &format!("Vehicle {} (Dead Reckoning)", self.id),
        );

        plot_ui.line(Line::new(
            format!("Est Path {}", self.id),
            self.h_x_est.positions(),
        ));
        draw_vehicle(
            plot_ui,
            self.x_est,
            &format!("Vehicle {} (Estimate)", self.id),
        );
    }

    fn options(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(format!("Vehicle {}", self.id));

                ui.group(|ui| {
                    ui.label("Motion:");
                    ui.add(
                        DragValue::new(&mut self.config.velocity)
                            .speed(0.1)
                            .range(0.0_f32..=10.0)
                            .prefix("Velocity: ")
                            .suffix(" m/s"),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.yaw_rate)
                            .speed(0.01)
                            .range(-1.0_f32..=1.0)
                            .prefix("Yaw Rate: ")
                            .suffix(" rad/s"),
                    );
                });

                ui.group(|ui| {
                    ui.label("Sensor:");
                    ui.add(
                        DragValue::new(&mut self.config.max_range)
                            .speed(0.5)
                            .range(1.0_f32..=50.0)
                            .prefix("Detection Range: ")
                            .suffix(" m"),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.obs_noise)
                            .speed(0.01)
                            .range(0.01_f32..=2.0)
                            .prefix("Obs Noise: "),
                    );
                });

                ui.group(|ui| {
                    ui.label("Noise:");
                    ui.add(
                        DragValue::new(&mut self.config.motion_noise_v)
                            .speed(0.1)
                            .range(0.1_f32..=5.0)
                            .prefix("Motion (v): "),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.motion_noise_yaw)
                            .speed(1.0)
                            .range(1.0_f32..=90.0)
                            .prefix("Motion (yaw): ")
                            .suffix("°"),
                    );
                });
            });
        });
    }
}

fn is_detected(marker: &rb::Vector2, state: &rb::Vector4, max_range: f32) -> bool {
    let dx = state.x() - marker.x();
    let dy = state.y() - marker.y();
    let d = hypot(dx, dy);
    d <= max_range
}

fn values_from_marker_state(marker: &rb::Vector2, state: &rb::Vector4) -> PlotPoints<'static> {
    PlotPoints::new(vec![
        [marker.x() as f64, marker.y() as f64],
        [state.x() as f64, state.y() as f64],
    ])
}

fn marker_values() -> PlotPoints<'static> {
    PlotPoints::new(
        MARKERS
            .iter()
            .map(|marker| [marker.x() as f64, marker.y() as f64])
            .collect(),
    )
}
