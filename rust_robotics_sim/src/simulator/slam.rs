use super::*;
use super::common::{
    rand_noise,
    visualization::{colors, covariance_ellipse_points, draw_labeled_vehicle, draw_trajectory},
};

use egui::{DragValue, Ui};
use egui_plot::{Line, LineStyle, PlotPoint, PlotPoints, PlotUi, Points, Polygon, Text};
use nalgebra::Vector2;
use rb::control::vehicle::VehicleParams;
use rb::prelude::*;
use rb::slam::{
    generate_observations, motion_model, predict, update, EkfSlamConfig, EkfSlamState, Observation,
};
use rust_robotics_algo as rb;
use std::fs::File;
use std::io::Write;

/// Log entry for a single simulation step
#[derive(Clone)]
struct StepLog {
    step: usize,
    // True state
    true_x: f32,
    true_y: f32,
    true_theta: f32,
    // Estimated state
    est_x: f32,
    est_y: f32,
    est_theta: f32,
    // Position error
    pos_error: f32,
    heading_error: f32,
    // Observations
    n_observations: usize,
    observed_landmark_indices: Vec<usize>,
    // State info
    n_landmarks_in_state: usize,
    // Data association info (which estimated landmark each observation matched to)
    associations: Vec<Option<usize>>,
}

/// Maximum number of log entries to keep
const MAX_LOG_ENTRIES: usize = 5000;

/// Maximum history length for trajectory visualization
const HISTORY_LEN: usize = 1000;

/// Default world bounds for landmark generation
const WORLD_SIZE: f32 = 40.0;

/// Minimum spacing between landmarks
const MIN_LANDMARK_SPACING: f32 = 5.0;

/// EKF-SLAM demonstration simulation
pub struct SlamDemo {
    /// Unique identifier for this simulation instance
    id: usize,

    /// True robot state [x, y, θ]
    x_true: rb::Vector3,
    /// True landmark positions
    landmarks_true: Vec<Vector2<f32>>,

    /// EKF-SLAM state estimate
    slam_state: EkfSlamState,
    /// EKF-SLAM configuration
    config: EkfSlamConfig,

    /// Dead reckoning state (for comparison)
    x_dr: rb::Vector3,

    /// History of true robot poses
    h_true: Vec<rb::Vector3>,
    /// History of estimated robot poses
    h_est: Vec<rb::Vector3>,
    /// History of dead reckoning poses
    h_dr: Vec<rb::Vector3>,

    /// Current observations (landmark_idx, observation) for visualization
    current_observations: Vec<(usize, Observation)>,

    /// Forward velocity (m/s)
    velocity: f32,
    /// Angular velocity (rad/s)
    yaw_rate: f32,

    /// Number of landmarks to generate
    n_landmarks: usize,

    /// UI option: show covariance ellipses
    show_covariance: bool,
    /// UI option: show observation lines
    show_observations: bool,
    /// UI option: show dead reckoning trajectory
    show_dr: bool,
    /// UI option: show true landmarks
    show_true_landmarks: bool,

    /// Vehicle parameters for visualization
    vehicle_params: VehicleParams,

    /// Step counter for logging
    step_count: usize,
    /// Log buffer for debugging
    logs: Vec<StepLog>,
    /// Enable logging
    logging_enabled: bool,
}

impl SlamDemo {
    pub fn new(id: usize, _time: f32) -> Self {
        let n_landmarks = 8;
        let landmarks_true = generate_landmarks(n_landmarks);

        Self {
            id,
            x_true: rb::Vector3::zeros(),
            landmarks_true,
            slam_state: EkfSlamState::new(),
            config: EkfSlamConfig::default(),
            x_dr: rb::Vector3::zeros(),
            h_true: vec![rb::Vector3::zeros()],
            h_est: vec![rb::Vector3::zeros()],
            h_dr: vec![rb::Vector3::zeros()],
            current_observations: Vec::new(),
            velocity: 1.5,
            yaw_rate: 0.15,
            n_landmarks,
            show_covariance: true,
            show_observations: true,
            show_dr: true,
            show_true_landmarks: true,
            vehicle_params: VehicleParams::default(),
            step_count: 0,
            logs: Vec::new(),
            logging_enabled: true,
        }
    }

    /// Save logs to a file
    fn save_logs(&self) -> Result<String, std::io::Error> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("slam_log_{}.csv", timestamp);

        let mut file = File::create(&filename)?;

        // Write header
        writeln!(file, "step,true_x,true_y,true_theta,est_x,est_y,est_theta,pos_error,heading_error,n_obs,n_landmarks,observed_indices,associations")?;

        // Write data
        for log in &self.logs {
            let obs_indices: String = log.observed_landmark_indices
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(";");
            let assoc_str: String = log.associations
                .iter()
                .map(|a| match a {
                    Some(i) => i.to_string(),
                    None => "new".to_string(),
                })
                .collect::<Vec<_>>()
                .join(";");

            writeln!(
                file,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
                log.step,
                log.true_x, log.true_y, log.true_theta,
                log.est_x, log.est_y, log.est_theta,
                log.pos_error, log.heading_error,
                log.n_observations, log.n_landmarks_in_state,
                obs_indices, assoc_str
            )?;
        }

        // Also save landmark positions
        let lm_filename = format!("slam_landmarks_{}.csv", timestamp);
        let mut lm_file = File::create(&lm_filename)?;
        writeln!(lm_file, "type,index,x,y")?;

        // True landmarks
        for (i, lm) in self.landmarks_true.iter().enumerate() {
            writeln!(lm_file, "true,{},{:.4},{:.4}", i, lm[0], lm[1])?;
        }

        // Estimated landmarks
        for i in 0..self.slam_state.n_landmarks {
            if let Some(lm) = self.slam_state.landmark(i) {
                writeln!(lm_file, "estimated,{},{:.4},{:.4}", i, lm[0], lm[1])?;
            }
        }

        Ok(filename)
    }

    fn update_history(&mut self) {
        self.h_true.push(self.x_true);
        self.h_est.push(self.slam_state.robot_pose());
        self.h_dr.push(self.x_dr);

        if self.h_true.len() > HISTORY_LEN {
            self.h_true.remove(0);
            self.h_est.remove(0);
            self.h_dr.remove(0);
        }
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

        // 1. Update true robot pose
        self.x_true = motion_model(&self.x_true, v, w, dt);

        // 2. Update dead reckoning (with biased noise to show SLAM benefit)
        // Simulates systematic odometry errors (wheel diameter, encoder resolution)
        // Bias causes consistent drift that doesn't cancel out
        let v_bias = 1.05;  // 5% consistent velocity overestimate
        let w_bias = 0.02;  // Consistent angular bias (robot turns slightly more)
        let v_noisy = v * v_bias + rand_noise() * 0.1;
        let w_noisy = w + w_bias + rand_noise() * 0.02;
        self.x_dr = motion_model(&self.x_dr, v_noisy, w_noisy, dt);

        // 3. EKF-SLAM prediction step
        predict(&mut self.slam_state, &self.config, v, w, dt);

        // 4. Generate observations from true pose
        self.current_observations =
            generate_observations(&self.x_true, &self.landmarks_true, &self.config, true);

        // Record state before update for logging
        let n_landmarks_before = self.slam_state.n_landmarks;

        // 5. EKF-SLAM update step
        update(&mut self.slam_state, &self.config, &self.current_observations);

        // 6. Log this step
        if self.logging_enabled {
            let est_pose = self.slam_state.robot_pose();
            let pos_error = ((self.x_true[0] - est_pose[0]).powi(2)
                + (self.x_true[1] - est_pose[1]).powi(2)).sqrt();
            let heading_error = (self.x_true[2] - est_pose[2]).abs();

            // Determine associations (simplified: compare n_landmarks before/after)
            let observed_indices: Vec<usize> = self.current_observations
                .iter()
                .map(|(idx, _)| *idx)
                .collect();

            // For associations, we track how many were new vs matched
            let n_new = self.slam_state.n_landmarks - n_landmarks_before;
            let n_matched = self.current_observations.len() - n_new;
            let mut associations: Vec<Option<usize>> = Vec::new();
            for i in 0..self.current_observations.len() {
                if i < n_matched {
                    // Matched to existing (we don't have exact index, mark as Some(0) placeholder)
                    associations.push(Some(i));
                } else {
                    // New landmark
                    associations.push(None);
                }
            }

            let log_entry = StepLog {
                step: self.step_count,
                true_x: self.x_true[0],
                true_y: self.x_true[1],
                true_theta: self.x_true[2],
                est_x: est_pose[0],
                est_y: est_pose[1],
                est_theta: est_pose[2],
                pos_error,
                heading_error,
                n_observations: self.current_observations.len(),
                observed_landmark_indices: observed_indices,
                n_landmarks_in_state: self.slam_state.n_landmarks,
                associations,
            };

            self.logs.push(log_entry);
            if self.logs.len() > MAX_LOG_ENTRIES {
                self.logs.remove(0);
            }
        }

        self.step_count += 1;

        // 7. Update history
        self.update_history();
    }

    fn reset_state(&mut self) {
        self.x_true = rb::Vector3::zeros();
        self.x_dr = rb::Vector3::zeros();
        self.slam_state = EkfSlamState::new();
        self.h_true = vec![rb::Vector3::zeros()];
        self.h_est = vec![rb::Vector3::zeros()];
        self.h_dr = vec![rb::Vector3::zeros()];
        self.current_observations.clear();
        self.step_count = 0;
        self.logs.clear();
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
    }
}

impl Draw for SlamDemo {
    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        // Colors for landmarks and trajectories
        let obs_color = colors::OBSERVATION;
        let landmark_est_color = colors::LANDMARK_EST;

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

        // Draw estimated landmarks with covariance ellipses
        for i in 0..self.slam_state.n_landmarks {
            if let Some(lm) = self.slam_state.landmark(i) {
                // Draw landmark point
                plot_ui.points(
                    Points::new(
                        "",
                        PlotPoints::new(vec![[lm[0] as f64, lm[1] as f64]]),
                    )
                    .shape(egui_plot::MarkerShape::Circle)
                    .radius(5.0)
                    .color(landmark_est_color),
                );

                // Draw covariance ellipse
                if self.show_covariance {
                    if let Some(cov) = self.slam_state.landmark_covariance(i) {
                        let ellipse = covariance_ellipse_points(lm[0], lm[1], &cov, 2.0);
                        plot_ui.polygon(
                            Polygon::new("", PlotPoints::new(ellipse))
                                .stroke(egui::Stroke::new(1.5, landmark_est_color.gamma_multiply(0.7)))
                                .fill_color(landmark_est_color.gamma_multiply(0.15)),
                        );
                    }
                }

                // Draw landmark index label
                plot_ui.text(Text::new(
                    "",
                    PlotPoint::new(lm[0] as f64 + 1.0, lm[1] as f64 + 1.0),
                    format!("{}", i),
                ));
            }
        }

        // Draw observation lines
        if self.show_observations {
            for (landmark_idx, _obs) in &self.current_observations {
                if *landmark_idx < self.landmarks_true.len() {
                    let lm = &self.landmarks_true[*landmark_idx];
                    plot_ui.line(
                        Line::new(
                            "",
                            PlotPoints::new(vec![
                                [self.x_true[0] as f64, self.x_true[1] as f64],
                                [lm[0] as f64, lm[1] as f64],
                            ]),
                        )
                        .style(LineStyle::Dotted { spacing: 5.0 })
                        .color(obs_color)
                        .width(1.5),
                    );
                }
            }
        }

        // Draw trajectories using common helper
        draw_trajectory(plot_ui, self.h_true.iter(), colors::TRUE, 2.0, None);
        draw_trajectory(plot_ui, self.h_est.iter(), colors::ESTIMATED, 2.0, None);

        // Dead reckoning trajectory
        if self.show_dr {
            draw_trajectory(
                plot_ui,
                self.h_dr.iter(),
                colors::DR,
                1.5,
                Some(LineStyle::Dashed { length: 8.0 }),
            );
        }

        // Draw vehicles using common helper
        let label_offset = 3.0;

        // True vehicle
        draw_labeled_vehicle(
            plot_ui,
            &self.x_true,
            self.velocity,
            "GT",
            label_offset,
            0.0, // No steering for differential drive
            &self.vehicle_params,
            &format!("Vehicle {} (True)", self.id),
        );

        // Estimated vehicle with covariance
        let est_pose = self.slam_state.robot_pose();
        draw_labeled_vehicle(
            plot_ui,
            &est_pose,
            self.velocity,
            "EKF",
            label_offset,
            0.0,
            &self.vehicle_params,
            &format!("Vehicle {} (Estimate)", self.id),
        );

        // Robot pose covariance ellipse
        if self.show_covariance {
            let cov = self.slam_state.robot_covariance();
            // Extract 2x2 position covariance
            let pos_cov = nalgebra::Matrix2::new(cov[(0, 0)], cov[(0, 1)], cov[(1, 0)], cov[(1, 1)]);
            let ellipse = covariance_ellipse_points(est_pose[0], est_pose[1], &pos_cov, 2.0);
            plot_ui.polygon(
                Polygon::new("", PlotPoints::new(ellipse))
                    .stroke(egui::Stroke::new(1.5, colors::ESTIMATED.gamma_multiply(0.7)))
                    .fill_color(colors::ESTIMATED.gamma_multiply(0.15)),
            );
        }

        // Dead reckoning vehicle
        if self.show_dr {
            draw_labeled_vehicle(
                plot_ui,
                &self.x_dr,
                self.velocity,
                "DR",
                label_offset,
                0.0,
                &self.vehicle_params,
                &format!("Vehicle {} (Dead Reckoning)", self.id),
            );
        }
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        let mut keep = true;
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("SLAM {}", self.id));
                    if ui.small_button("x").clicked() {
                        keep = false;
                    }
                });

                ui.separator();

                // Motion controls
                ui.group(|ui| {
                    ui.label("Motion:");
                    ui.add(
                        DragValue::new(&mut self.velocity)
                            .speed(0.05)
                            .range(0.1_f32..=3.0)
                            .prefix("v: ")
                            .suffix(" m/s"),
                    );
                    ui.add(
                        DragValue::new(&mut self.yaw_rate)
                            .speed(0.01)
                            .range(-0.5_f32..=0.5)
                            .prefix("w: ")
                            .suffix(" rad/s"),
                    );
                });

                // Landmarks
                ui.group(|ui| {
                    ui.label("Landmarks:");
                    let old_n = self.n_landmarks;
                    ui.add(
                        DragValue::new(&mut self.n_landmarks)
                            .speed(0.5)
                            .range(2_usize..=20)
                            .prefix("n: "),
                    );
                    if self.n_landmarks != old_n {
                        // Regenerate landmarks and reset SLAM state
                        self.landmarks_true = generate_landmarks(self.n_landmarks);
                        // Reset SLAM but keep current robot pose estimate
                        let mut new_state = EkfSlamState::new();
                        new_state.mu[0] = self.x_true[0];
                        new_state.mu[1] = self.x_true[1];
                        new_state.mu[2] = self.x_true[2];
                        self.slam_state = new_state;
                        self.current_observations.clear();
                    }
                    ui.label(format!("Discovered: {}", self.slam_state.n_landmarks));
                });

                // Sensor config
                ui.group(|ui| {
                    ui.label("Sensor:");
                    ui.add(
                        DragValue::new(&mut self.config.max_range)
                            .speed(1.0)
                            .range(5.0_f32..=60.0)
                            .prefix("Range: ")
                            .suffix(" m"),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.association_gate)
                            .speed(0.1)
                            .range(1.0_f32..=20.0)
                            .prefix("Gate: "),
                    );
                });

                // Display options
                ui.group(|ui| {
                    ui.label("Display:");
                    ui.checkbox(&mut self.show_covariance, "Covariance");
                    ui.checkbox(&mut self.show_observations, "Observations");
                    ui.checkbox(&mut self.show_dr, "Dead Reckoning");
                    ui.checkbox(&mut self.show_true_landmarks, "True Landmarks");
                });

                // Stats
                ui.separator();
                let est_pose = self.slam_state.robot_pose();
                let pos_err = ((self.x_true[0] - est_pose[0]).powi(2)
                    + (self.x_true[1] - est_pose[1]).powi(2))
                .sqrt();
                let dr_err = ((self.x_true[0] - self.x_dr[0]).powi(2)
                    + (self.x_true[1] - self.x_dr[1]).powi(2))
                .sqrt();
                ui.label(format!("EKF err: {:.2} m", pos_err));
                ui.label(format!("DR err: {:.2} m", dr_err));

                // Logging controls
                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.logging_enabled, "Log");
                    ui.label(format!("({} entries)", self.logs.len()));
                });
                if ui.button("Save Logs").clicked() {
                    match self.save_logs() {
                        Ok(filename) => {
                            println!("Saved logs to: {}", filename);
                        }
                        Err(e) => {
                            println!("Failed to save logs: {}", e);
                        }
                    }
                }
                if ui.button("Clear Logs").clicked() {
                    self.logs.clear();
                    self.step_count = 0;
                }
            });
        });
        keep
    }

    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        // Position error over time
        let est_errors: PlotPoints<'_> = self
            .h_true
            .iter()
            .zip(self.h_est.iter())
            .enumerate()
            .map(|(i, (t, e))| {
                let err = ((t[0] - e[0]).powi(2) + (t[1] - e[1]).powi(2)).sqrt();
                [i as f64 * 0.01, err as f64]
            })
            .collect();

        let dr_errors: PlotPoints<'_> = self
            .h_true
            .iter()
            .zip(self.h_dr.iter())
            .enumerate()
            .map(|(i, (t, d))| {
                let err = ((t[0] - d[0]).powi(2) + (t[1] - d[1]).powi(2)).sqrt();
                [i as f64 * 0.01, err as f64]
            })
            .collect();

        plot_ui.line(
            Line::new(format!("EKF Error {}", self.id), est_errors)
                .name(format!("EKF Error {} (m)", self.id))
                .color(colors::ESTIMATED),
        );

        plot_ui.line(
            Line::new(format!("DR Error {}", self.id), dr_errors)
                .name(format!("DR Error {} (m)", self.id))
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

        // Generate random position in world
        let x = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 2.0;
        let y = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 2.0;

        // Avoid placing landmarks too close to origin (starting position)
        if (x * x + y * y).sqrt() < 3.0 {
            continue;
        }

        // Check minimum spacing from existing landmarks
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
