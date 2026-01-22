use super::*;

use crate::data::VehiclePlot;
use crate::item::draw_vehicle;

use egui::Ui;
use egui_plot::{Line, LineStyle, PlotPoints, PlotUi, Points};
use rb::localization::{particle_filter::*, StateVector};
use rb::prelude::*;
use rust_robotics_algo as rb;

pub type State = rb::Vector4;

const N: usize = 1000;

const MARKERS: [rb::Vector2; 4] = [
    vector![10.0_f32, 0.0_f32],
    vector![10.0, 10.0],
    vector![0.0, 15.0],
    vector![-5.0, 20.0],
];

pub struct ParticleFilter {
    x_est: State,
    x_true: State,
    x_dr: State,
    p_est: rb::Matrix3,
    pw: PW,
    px: PX,
    h_x_est: Vec<State>,
    h_x_true: Vec<State>,
    h_x_dr: Vec<State>,
    id: usize,
    init_time: f32,
}

impl ParticleFilter {
    pub fn new(id: usize, time: f32) -> Self {
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
            id,
            init_time: time,
        }
    }

    pub fn update_history(&mut self) {
        self.h_x_est.push(self.x_est);
        self.h_x_true.push(self.x_true);
        self.h_x_dr.push(self.x_dr);

        if self.h_x_est.len() > N {
            self.h_x_est.remove(0);
            self.h_x_true.remove(0);
            self.h_x_dr.remove(0);
        }
    }
}

impl Simulate for ParticleFilter {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.x_true
    }
    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<State>() {
            // Then set self's data from `other` if the type matches
            self.x_true.clone_from(data);
        }
    }
    fn step(&mut self, dt: f32) {
        let u = calc_input();
        let (z, ud) = observation(&mut self.x_true, &mut self.x_dr, u, &MARKERS, dt);
        self.p_est = pf_localization(&mut self.x_est, &mut self.px, &mut self.pw, z, ud, dt);

        self.update_history();
    }
    fn reset_state(&mut self) {}
    fn reset_all(&mut self) {}
}

impl Draw for ParticleFilter {
    fn plot(&self, _plot_ui: &mut PlotUi<'_>) {}
    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        plot_ui.points(Points::new(
            "",
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
        MARKERS.iter().for_each(|marker| {
            plot_ui.points(Points::new("", marker_values()).radius(2.0));
            if is_detected(marker, &self.x_true) {
                plot_ui.line(
                    Line::new("", values_from_marker_state(marker, &self.x_true))
                        .style(LineStyle::Dotted { spacing: 10.0 }),
                );
            }
        });
        plot_ui.line(Line::new("", self.h_x_true.positions()));
        draw_vehicle(
            plot_ui,
            self.x_true,
            &format!("Vehicle {} (Actual)", self.id),
        );
        plot_ui.line(Line::new("", self.h_x_dr.positions()));
        draw_vehicle(plot_ui, self.x_dr, &format!("Vehicle {} (DR)", self.id));
        plot_ui.line(Line::new("", self.h_x_est.positions()));
        draw_vehicle(
            plot_ui,
            self.x_est,
            &format!("Vehicle {} (Estimate)", self.id),
        );
    }
    fn options(&mut self, _ui: &mut Ui) {}
}

fn is_detected(marker: &rb::Vector2, state: &rb::Vector4) -> bool {
    let dx = state.x() - marker.x();
    let dy = state.y() - marker.y();
    let d = hypot(dx, dy);
    d <= MAX_RANGE
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

// pub struct Vehicle {
//     state: State,
//     id: usize,
//     data: TimeTable,
//     time_init: f32,
// }
