use super::*;
use super::common::{
    rand_noise,
    visualization::{colors, covariance_ellipse_points, draw_labeled_vehicle, draw_trajectory},
};

use egui::{Color32, DragValue, Ui};
use egui_plot::{Line, LineStyle, PlotPoints, PlotUi, Points, Polygon};
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use rb::control::vehicle::VehicleParams;
use rb::prelude::*;
use rb::slam::{
    generate_observations, motion_model, predict, update, EkfSlamConfig, EkfSlamState, Observation,
    GraphSlam, Pose2D, Landmark2D,
};
use rust_robotics_algo as rb;
use std::time::Instant;

/// Maximum history length for trajectory visualization
const HISTORY_LEN: usize = 1000;

/// Default world bounds for landmark generation
const WORLD_SIZE: f32 = 40.0;

/// Minimum spacing between landmarks
const MIN_LANDMARK_SPACING: f32 = 5.0;

/// Colors for different algorithms
const EKF_COLOR: Color32 = Color32::from_rgb(66, 135, 245);   // Blue
const GRAPH_COLOR: Color32 = Color32::from_rgb(156, 39, 176); // Purple

/// EKF-SLAM algorithm instance
struct EkfSlamInstance {
    enabled: bool,
    state: EkfSlamState,
    h_est: Vec<rb::Vector3>,
    /// Last update time in microseconds
    last_update_us: f64,
    /// Exponential moving average of update time
    avg_update_us: f64,
}

impl EkfSlamInstance {
    fn new() -> Self {
        Self {
            enabled: true,
            state: EkfSlamState::new(),
            h_est: vec![rb::Vector3::zeros()],
            last_update_us: 0.0,
            avg_update_us: 0.0,
        }
    }

    fn reset(&mut self) {
        self.state = EkfSlamState::new();
        self.h_est = vec![rb::Vector3::zeros()];
        self.last_update_us = 0.0;
        self.avg_update_us = 0.0;
    }

    fn step(&mut self, config: &EkfSlamConfig, v: f32, w: f32, dt: f32, observations: &[(usize, Observation)]) {
        if !self.enabled {
            return;
        }

        let start = Instant::now();
        predict(&mut self.state, config, v, w, dt);
        update(&mut self.state, config, observations);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0; // microseconds

        self.last_update_us = elapsed;
        // Exponential moving average (alpha = 0.1)
        self.avg_update_us = 0.9 * self.avg_update_us + 0.1 * elapsed;

        self.h_est.push(self.state.robot_pose());
        if self.h_est.len() > HISTORY_LEN {
            self.h_est.remove(0);
        }
    }

    fn get_pose(&self) -> rb::Vector3 {
        self.state.robot_pose()
    }
}

/// Graph-SLAM algorithm instance
struct GraphSlamInstance {
    enabled: bool,
    graph: GraphSlam,
    landmark_to_graph_idx: Vec<Option<usize>>,
    prev_keyframe_idx: usize,
    accumulated_motion: (f32, f32, f32),
    keyframe_trans_threshold: f32,
    keyframe_rot_threshold: f32,
    h_est: Vec<rb::Vector3>,
    /// Last update time in microseconds (only when keyframe added)
    last_update_us: f64,
    /// Exponential moving average of update time
    avg_update_us: f64,
}

impl GraphSlamInstance {
    fn new(n_landmarks: usize) -> Self {
        let mut graph = GraphSlam::new();
        graph.add_pose(Pose2D::origin());
        Self {
            enabled: false, // Disabled by default
            graph,
            landmark_to_graph_idx: vec![None; n_landmarks],
            prev_keyframe_idx: 0,
            accumulated_motion: (0.0, 0.0, 0.0),
            keyframe_trans_threshold: 1.0,
            keyframe_rot_threshold: 0.3,
            h_est: vec![rb::Vector3::zeros()],
            last_update_us: 0.0,
            avg_update_us: 0.0,
        }
    }

    fn reset(&mut self, n_landmarks: usize) {
        self.graph = GraphSlam::new();
        self.graph.add_pose(Pose2D::origin());
        self.landmark_to_graph_idx = vec![None; n_landmarks];
        self.prev_keyframe_idx = 0;
        self.accumulated_motion = (0.0, 0.0, 0.0);
        self.h_est = vec![rb::Vector3::zeros()];
        self.last_update_us = 0.0;
        self.avg_update_us = 0.0;
    }

    fn step(&mut self, v: f32, w: f32, dt: f32, observations: &[(usize, Observation)]) {
        if !self.enabled {
            return;
        }

        let start = Instant::now();

        // Accumulate motion since last keyframe
        let dx = v * dt;
        let dtheta = w * dt;

        let prev_kf = self.graph.poses[self.prev_keyframe_idx];
        let prev_kf_theta = prev_kf.theta;

        let (acc_x, acc_y, acc_theta) = self.accumulated_motion;
        let new_acc_theta = acc_theta + dtheta;
        let new_acc_x = acc_x + dx * (prev_kf_theta + acc_theta).cos();
        let new_acc_y = acc_y + dx * (prev_kf_theta + acc_theta).sin();
        self.accumulated_motion = (new_acc_x, new_acc_y, new_acc_theta);

        // Update history with interpolated pose
        let current_pose = rb::Vector3::new(
            prev_kf.x + new_acc_x,
            prev_kf.y + new_acc_y,
            prev_kf_theta + new_acc_theta,
        );
        self.h_est.push(current_pose);
        if self.h_est.len() > HISTORY_LEN {
            self.h_est.remove(0);
        }

        // Check if we should create a new keyframe
        let trans = (new_acc_x * new_acc_x + new_acc_y * new_acc_y).sqrt();
        let rot = new_acc_theta.abs();

        if trans < self.keyframe_trans_threshold && rot < self.keyframe_rot_threshold {
            // No keyframe - just measure the small overhead
            let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
            self.last_update_us = elapsed;
            self.avg_update_us = 0.9 * self.avg_update_us + 0.1 * elapsed;
            return;
        }

        // Create new keyframe
        let new_x = prev_kf.x + new_acc_x;
        let new_y = prev_kf.y + new_acc_y;
        let new_theta = prev_kf_theta + new_acc_theta;

        let new_pose_idx = self.graph.add_pose(Pose2D::new(new_x, new_y, new_theta));

        // Odometry constraint
        let c = prev_kf_theta.cos();
        let s = prev_kf_theta.sin();
        let dx_local = c * new_acc_x + s * new_acc_y;
        let dy_local = -s * new_acc_x + c * new_acc_y;
        let odom_measurement = Vector3::new(dx_local, dy_local, new_acc_theta);

        let odom_cov = Matrix3::from_diagonal(&Vector3::new(
            0.1 * trans.max(0.1),
            0.1 * trans.max(0.1),
            0.05 * rot.max(0.01),
        ));
        self.graph.add_odometry(self.prev_keyframe_idx, new_pose_idx, odom_measurement, &odom_cov);

        // Observation constraints
        let obs_cov = Matrix2::from_diagonal(&nalgebra::Vector2::new(0.5, 0.05));
        for (true_lm_idx, obs) in observations {
            // Ensure landmark_to_graph_idx is large enough
            while self.landmark_to_graph_idx.len() <= *true_lm_idx {
                self.landmark_to_graph_idx.push(None);
            }

            let graph_lm_idx = if let Some(idx) = self.landmark_to_graph_idx[*true_lm_idx] {
                idx
            } else {
                let pose = &self.graph.poses[new_pose_idx];
                let lm_x = pose.x + obs.range * (pose.theta + obs.bearing).cos();
                let lm_y = pose.y + obs.range * (pose.theta + obs.bearing).sin();
                let idx = self.graph.add_landmark(Landmark2D::new(lm_x, lm_y));
                self.landmark_to_graph_idx[*true_lm_idx] = Some(idx);
                idx
            };

            self.graph.add_observation(new_pose_idx, graph_lm_idx, obs.range, obs.bearing, &obs_cov);
        }

        self.prev_keyframe_idx = new_pose_idx;
        self.accumulated_motion = (0.0, 0.0, 0.0);

        // Optimize after adding keyframe
        self.graph.optimize();

        // Measure time including optimization
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        self.last_update_us = elapsed;
        self.avg_update_us = 0.9 * self.avg_update_us + 0.1 * elapsed;
    }

    fn get_pose(&self) -> rb::Vector3 {
        let prev_kf = &self.graph.poses[self.prev_keyframe_idx];
        let (acc_x, acc_y, acc_theta) = self.accumulated_motion;
        rb::Vector3::new(prev_kf.x + acc_x, prev_kf.y + acc_y, prev_kf.theta + acc_theta)
    }
}

/// SLAM demonstration with shared vehicle and multiple algorithms
pub struct SlamDemo {
    /// True robot state [x, y, θ]
    x_true: rb::Vector3,
    /// True landmark positions
    landmarks_true: Vec<Vector2<f32>>,
    /// Dead reckoning state (for comparison)
    x_dr: rb::Vector3,
    /// History of true robot poses
    h_true: Vec<rb::Vector3>,
    /// History of dead reckoning poses
    h_dr: Vec<rb::Vector3>,
    /// Current observations (landmark_idx, observation)
    current_observations: Vec<(usize, Observation)>,

    /// EKF-SLAM algorithm instance
    ekf: EkfSlamInstance,
    /// Graph-SLAM algorithm instance
    graph: GraphSlamInstance,

    /// Shared sensor configuration
    config: EkfSlamConfig,
    /// Forward velocity (m/s)
    velocity: f32,
    /// Angular velocity (rad/s)
    yaw_rate: f32,
    /// Number of landmarks
    pub n_landmarks: usize,

    /// UI options
    show_covariance: bool,
    show_observations: bool,
    show_dr: bool,
    show_true_landmarks: bool,

    /// Vehicle parameters for visualization
    vehicle_params: VehicleParams,
    /// Step counter
    step_count: usize,
}

impl SlamDemo {
    pub fn new(_id: usize, _time: f32) -> Self {
        let n_landmarks = 8;
        let landmarks_true = generate_landmarks(n_landmarks);

        Self {
            x_true: rb::Vector3::zeros(),
            landmarks_true,
            x_dr: rb::Vector3::zeros(),
            h_true: vec![rb::Vector3::zeros()],
            h_dr: vec![rb::Vector3::zeros()],
            current_observations: Vec::new(),
            ekf: EkfSlamInstance::new(),
            graph: GraphSlamInstance::new(n_landmarks),
            config: EkfSlamConfig::default(),
            velocity: 1.5,
            yaw_rate: 0.15,
            n_landmarks,
            show_covariance: true,
            show_observations: true,
            show_dr: true,
            show_true_landmarks: true,
            vehicle_params: VehicleParams::default(),
            step_count: 0,
        }
    }

    fn update_history(&mut self) {
        self.h_true.push(self.x_true);
        self.h_dr.push(self.x_dr);

        if self.h_true.len() > HISTORY_LEN {
            self.h_true.remove(0);
            self.h_dr.remove(0);
        }
    }

    /// Regenerate landmarks and reset simulation state
    pub fn regenerate_landmarks(&mut self) {
        // Reset vehicle state
        self.x_true = rb::Vector3::zeros();
        self.x_dr = rb::Vector3::zeros();
        self.h_true = vec![rb::Vector3::zeros()];
        self.h_dr = vec![rb::Vector3::zeros()];
        self.current_observations.clear();
        self.step_count = 0;

        // Regenerate landmarks
        self.landmarks_true = generate_landmarks(self.n_landmarks);

        // Reset algorithm states
        self.ekf.reset();
        self.graph.reset(self.n_landmarks);
    }
}

impl Default for SlamDemo {
    fn default() -> Self {
        Self::new(1, 0.0)
    }
}

impl Simulate for SlamDemo {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.x_true
    }

    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<rb::Vector3>() {
            self.x_true = *data;
        }
    }

    fn step(&mut self, dt: f32) {
        let v = self.velocity;
        let w = self.yaw_rate;

        // Update true robot pose
        self.x_true = motion_model(&self.x_true, v, w, dt);

        // Update dead reckoning
        let v_bias = 1.05;
        let w_bias = 0.02;
        let v_noisy = v * v_bias + rand_noise() * 0.1;
        let w_noisy = w + w_bias + rand_noise() * 0.02;
        self.x_dr = motion_model(&self.x_dr, v_noisy, w_noisy, dt);

        // Generate observations from true pose
        self.current_observations =
            generate_observations(&self.x_true, &self.landmarks_true, &self.config, true);

        // Run all enabled algorithms with the same observations
        self.ekf.step(&self.config, v, w, dt, &self.current_observations);
        self.graph.step(v, w, dt, &self.current_observations);

        self.step_count += 1;
        self.update_history();
    }

    fn reset_state(&mut self) {
        self.x_true = rb::Vector3::zeros();
        self.x_dr = rb::Vector3::zeros();
        self.h_true = vec![rb::Vector3::zeros()];
        self.h_dr = vec![rb::Vector3::zeros()];
        self.current_observations.clear();
        self.step_count = 0;

        self.ekf.reset();
        self.graph.reset(self.n_landmarks);

        // Regenerate landmarks
        self.landmarks_true = generate_landmarks(self.n_landmarks);
    }

    fn reset_all(&mut self) {
        self.reset_state();
        self.velocity = 1.5;
        self.yaw_rate = 0.15;
        self.n_landmarks = 8;
        self.config = EkfSlamConfig::default();
        self.show_covariance = true;
        self.show_observations = true;
        self.show_dr = true;
        self.show_true_landmarks = true;
        self.ekf.enabled = true;
        self.graph.enabled = false;
    }
}

impl Draw for SlamDemo {
    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        // Draw true landmarks
        if self.show_true_landmarks {
            let landmark_points: Vec<[f64; 2]> = self
                .landmarks_true
                .iter()
                .map(|lm| [lm[0] as f64, lm[1] as f64])
                .collect();
            plot_ui.points(
                Points::new("True Landmarks", PlotPoints::new(landmark_points))
                    .shape(egui_plot::MarkerShape::Cross)
                    .radius(6.0)
                    .color(colors::TRUE),
            );
        }

        // Draw EKF-SLAM landmarks and estimates
        if self.ekf.enabled {
            for i in 0..self.ekf.state.n_landmarks {
                if let Some(lm) = self.ekf.state.landmark(i) {
                    plot_ui.points(
                        Points::new("", PlotPoints::new(vec![[lm[0] as f64, lm[1] as f64]]))
                            .shape(egui_plot::MarkerShape::Circle)
                            .radius(5.0)
                            .color(EKF_COLOR),
                    );

                    if self.show_covariance {
                        if let Some(cov) = self.ekf.state.landmark_covariance(i) {
                            let ellipse = covariance_ellipse_points(lm[0], lm[1], &cov, 2.0);
                            plot_ui.polygon(
                                Polygon::new("", PlotPoints::new(ellipse))
                                    .stroke(egui::Stroke::new(1.5, EKF_COLOR.gamma_multiply(0.7)))
                                    .fill_color(EKF_COLOR.gamma_multiply(0.15)),
                            );
                        }
                    }
                }
            }

            // EKF trajectory
            draw_trajectory(plot_ui, self.ekf.h_est.iter(), EKF_COLOR, 2.0, None);

            // EKF vehicle
            let ekf_pose = self.ekf.get_pose();
            draw_labeled_vehicle(
                plot_ui, &ekf_pose, self.velocity, "EKF", 3.0, 0.0,
                &self.vehicle_params, "EKF Estimate",
            );

            // EKF robot covariance
            if self.show_covariance {
                let cov = self.ekf.state.robot_covariance();
                let pos_cov = nalgebra::Matrix2::new(cov[(0, 0)], cov[(0, 1)], cov[(1, 0)], cov[(1, 1)]);
                let ellipse = covariance_ellipse_points(ekf_pose[0], ekf_pose[1], &pos_cov, 2.0);
                plot_ui.polygon(
                    Polygon::new("", PlotPoints::new(ellipse))
                        .stroke(egui::Stroke::new(1.5, EKF_COLOR.gamma_multiply(0.7)))
                        .fill_color(EKF_COLOR.gamma_multiply(0.15)),
                );
            }
        }

        // Draw Graph-SLAM landmarks and estimates
        if self.graph.enabled {
            for lm in self.graph.graph.landmarks.iter() {
                plot_ui.points(
                    Points::new("", PlotPoints::new(vec![[lm.x as f64, lm.y as f64]]))
                        .shape(egui_plot::MarkerShape::Diamond)
                        .radius(5.0)
                        .color(GRAPH_COLOR),
                );
            }

            // Graph trajectory
            draw_trajectory(plot_ui, self.graph.h_est.iter(), GRAPH_COLOR, 2.0, None);

            // Graph vehicle
            let graph_pose = self.graph.get_pose();
            draw_labeled_vehicle(
                plot_ui, &graph_pose, self.velocity, "Graph", 5.0, 0.0,
                &self.vehicle_params, "Graph Estimate",
            );
        }

        // Draw observation lines
        if self.show_observations {
            for (landmark_idx, _obs) in &self.current_observations {
                if *landmark_idx < self.landmarks_true.len() {
                    let lm = &self.landmarks_true[*landmark_idx];
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![
                            [self.x_true[0] as f64, self.x_true[1] as f64],
                            [lm[0] as f64, lm[1] as f64],
                        ]))
                        .style(LineStyle::Dotted { spacing: 5.0 })
                        .color(colors::OBSERVATION)
                        .width(1.5),
                    );
                }
            }
        }

        // Draw true trajectory
        draw_trajectory(plot_ui, self.h_true.iter(), colors::TRUE, 2.0, None);

        // Draw dead reckoning
        if self.show_dr {
            draw_trajectory(
                plot_ui, self.h_dr.iter(), colors::DR, 1.5,
                Some(LineStyle::Dashed { length: 8.0 }),
            );
            draw_labeled_vehicle(
                plot_ui, &self.x_dr, self.velocity, "DR", 3.0, 0.0,
                &self.vehicle_params, "Dead Reckoning",
            );
        }

        // Draw true vehicle
        draw_labeled_vehicle(
            plot_ui, &self.x_true, self.velocity, "GT", 3.0, 0.0,
            &self.vehicle_params, "Ground Truth",
        );
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        ui.horizontal_top(|ui| {
            // Algorithm toggles
            ui.group(|ui| {
                ui.set_min_width(100.0);
                ui.vertical(|ui| {
                    ui.label("Algorithms");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.ekf.enabled, "EKF");
                        ui.colored_label(EKF_COLOR, "●");
                    });
                    ui.horizontal(|ui| {
                        let was_enabled = self.graph.enabled;
                        ui.checkbox(&mut self.graph.enabled, "Graph");
                        ui.colored_label(GRAPH_COLOR, "◆");
                        if self.graph.enabled && !was_enabled {
                            self.graph.reset(self.n_landmarks);
                        }
                    });
                });
            });

            // Motion & Landmarks
            ui.group(|ui| {
                ui.set_min_width(100.0);
                ui.vertical(|ui| {
                    ui.label("Motion");
                    ui.add(DragValue::new(&mut self.velocity).speed(0.05).range(0.1_f32..=3.0).prefix("v: "));
                    ui.add(DragValue::new(&mut self.yaw_rate).speed(0.01).range(-0.5_f32..=0.5).prefix("ω: "));
                });
            });

            // Display options
            ui.group(|ui| {
                ui.set_min_width(90.0);
                ui.vertical(|ui| {
                    ui.label("Display");
                    ui.checkbox(&mut self.show_covariance, "Cov");
                    ui.checkbox(&mut self.show_observations, "Obs");
                    ui.checkbox(&mut self.show_dr, "DR");
                    ui.checkbox(&mut self.show_true_landmarks, "LM");
                });
            });

            // Error stats
            ui.group(|ui| {
                ui.set_min_width(80.0);
                ui.vertical(|ui| {
                    ui.label("Error");
                    if self.ekf.enabled {
                        let ekf_pose = self.ekf.get_pose();
                        let ekf_err = ((self.x_true[0] - ekf_pose[0]).powi(2)
                            + (self.x_true[1] - ekf_pose[1]).powi(2)).sqrt();
                        ui.colored_label(EKF_COLOR, format!("{:.2}m", ekf_err));
                    }
                    if self.graph.enabled {
                        let graph_pose = self.graph.get_pose();
                        let graph_err = ((self.x_true[0] - graph_pose[0]).powi(2)
                            + (self.x_true[1] - graph_pose[1]).powi(2)).sqrt();
                        ui.colored_label(GRAPH_COLOR, format!("{:.2}m", graph_err));
                    }
                    let dr_err = ((self.x_true[0] - self.x_dr[0]).powi(2)
                        + (self.x_true[1] - self.x_dr[1]).powi(2)).sqrt();
                    ui.label(format!("DR {:.2}m", dr_err));
                });
            });

            // Timing stats
            ui.group(|ui| {
                ui.set_min_width(80.0);
                ui.vertical(|ui| {
                    ui.label("Time (μs)");
                    if self.ekf.enabled {
                        ui.colored_label(EKF_COLOR, format!("{:.0}", self.ekf.avg_update_us));
                    }
                    if self.graph.enabled {
                        ui.colored_label(GRAPH_COLOR, format!("{:.0}", self.graph.avg_update_us));
                    }
                });
            });
        });
        true // Always keep - single instance
    }

    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        // EKF error over time
        if self.ekf.enabled {
            let ekf_errors: PlotPoints<'_> = self.h_true.iter()
                .zip(self.ekf.h_est.iter())
                .enumerate()
                .map(|(i, (t, e))| {
                    let err = ((t[0] - e[0]).powi(2) + (t[1] - e[1]).powi(2)).sqrt();
                    [i as f64 * 0.01, err as f64]
                })
                .collect();
            plot_ui.line(
                Line::new("EKF Error", ekf_errors)
                    .name("EKF Error (m)")
                    .color(EKF_COLOR),
            );
        }

        // Graph error over time
        if self.graph.enabled {
            let graph_errors: PlotPoints<'_> = self.h_true.iter()
                .zip(self.graph.h_est.iter())
                .enumerate()
                .map(|(i, (t, e))| {
                    let err = ((t[0] - e[0]).powi(2) + (t[1] - e[1]).powi(2)).sqrt();
                    [i as f64 * 0.01, err as f64]
                })
                .collect();
            plot_ui.line(
                Line::new("Graph Error", graph_errors)
                    .name("Graph Error (m)")
                    .color(GRAPH_COLOR),
            );
        }

        // DR error
        let dr_errors: PlotPoints<'_> = self.h_true.iter()
            .zip(self.h_dr.iter())
            .enumerate()
            .map(|(i, (t, d))| {
                let err = ((t[0] - d[0]).powi(2) + (t[1] - d[1]).powi(2)).sqrt();
                [i as f64 * 0.01, err as f64]
            })
            .collect();
        plot_ui.line(
            Line::new("DR Error", dr_errors)
                .name("DR Error (m)")
                .color(colors::DR),
        );
    }
}

/// Generate random landmarks within the world bounds
fn generate_landmarks(n: usize) -> Vec<Vector2<f32>> {
    let mut landmarks = Vec::with_capacity(n);
    let mut attempts = 0;
    let max_attempts = n * 100;

    while landmarks.len() < n && attempts < max_attempts {
        attempts += 1;

        let x = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 2.0;
        let y = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 2.0;

        if (x * x + y * y).sqrt() < 3.0 {
            continue;
        }

        let too_close = landmarks.iter().any(|lm: &Vector2<f32>| {
            let dx = lm[0] - x;
            let dy = lm[1] - y;
            (dx * dx + dy * dy).sqrt() < MIN_LANDMARK_SPACING
        });

        if !too_close {
            landmarks.push(Vector2::new(x, y));
        }
    }

    landmarks
}
