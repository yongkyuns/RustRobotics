use super::*;
use super::common::{
    ErrorTracker, HistoryManager,
    draw_labeled_vehicle_state4, draw_trajectory,
};

use egui::{DragValue, Ui};
use egui_plot::{Line, LineStyle, PlotPoints, PlotUi, Points};
use rb::control::vehicle::{BicycleModel, TireModel, TireParams, VehicleParams, VehicleState};
use rb::localization::{particle_filter::*, StateVector};
use rb::prelude::*;
use rust_robotics_algo as rb;

pub type State = rb::Vector4;

/// Maximum history length for trajectory visualization
const HISTORY_LEN: usize = 1000;

/// Landmark positions for observation (in meters, spread for real-world scale)
const MARKERS: [rb::Vector2; 4] = [
    vector![30.0_f32, 0.0_f32],
    vector![30.0, 30.0],
    vector![0.0, 45.0],
    vector![-15.0, 60.0],
];

/// Drive mode for vehicle control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveMode {
    /// Kinematic mode: specify velocity and yaw rate directly
    Kinematic,
    /// Dynamic mode: keyboard-controlled with bicycle model dynamics
    Dynamic,
}

impl DriveMode {
    fn label(&self) -> &'static str {
        match self {
            DriveMode::Kinematic => "Fixed Input",
            DriveMode::Dynamic => "User Control",
        }
    }
}

/// Configurable parameters for particle filter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PFConfig {
    /// Drive mode (Kinematic or Dynamic)
    pub drive_mode: DriveMode,
    /// Forward velocity (m/s) - used in Kinematic mode
    pub velocity: f32,
    /// Yaw rate (rad/s) - used in Kinematic mode
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
            drive_mode: DriveMode::Kinematic,
            velocity: 5.0,        // 5 m/s (~18 km/h)
            yaw_rate: 0.1,
            max_range: 50.0,      // 50m detection range
            obs_noise: 1.0,       // Observation noise (distance std dev in m)
            motion_noise_v: 3.0,  // Motion noise velocity (m/s std dev)
            motion_noise_yaw: 30.0, // Motion noise yaw rate (deg std dev)
        }
    }
}

/// Keyboard input state for dynamic driving mode
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyboardInput {
    /// Current steering angle [rad]
    pub steering: f32,
    /// Current acceleration [m/s^2]
    pub acceleration: f32,
}

impl KeyboardInput {
    /// Maximum steering angle [rad] (~30 degrees)
    const MAX_STEERING: f32 = 0.52;
    /// Steering increment per key press [rad]
    const STEERING_RATE: f32 = 0.02;
    /// Maximum acceleration [m/s^2]
    const MAX_ACCEL: f32 = 3.0;
    /// Acceleration increment per key press [m/s^2] (50% faster)
    const ACCEL_RATE: f32 = 0.3;
    /// Braking increment per key press [m/s^2]
    const BRAKE_RATE: f32 = 0.8;
    /// Steering return rate (spring back to center)
    const STEERING_DECAY: f32 = 0.95;
    /// Acceleration decay rate
    const ACCEL_DECAY: f32 = 0.9;

    /// Update input based on keyboard state
    pub fn update(&mut self, left: bool, right: bool, up: bool, down: bool) {
        // Steering: left/right arrows
        if left {
            self.steering = (self.steering + Self::STEERING_RATE).min(Self::MAX_STEERING);
        } else if right {
            self.steering = (self.steering - Self::STEERING_RATE).max(-Self::MAX_STEERING);
        } else {
            // Spring back to center
            self.steering *= Self::STEERING_DECAY;
            if self.steering.abs() < 0.01 {
                self.steering = 0.0;
            }
        }

        // Acceleration: up/down arrows
        if up {
            self.acceleration = (self.acceleration + Self::ACCEL_RATE).min(Self::MAX_ACCEL);
        } else if down {
            self.acceleration = (self.acceleration - Self::BRAKE_RATE).max(-Self::MAX_ACCEL);
        } else {
            // Decay acceleration
            self.acceleration *= Self::ACCEL_DECAY;
            if self.acceleration.abs() < 0.1 {
                self.acceleration = 0.0;
            }
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
    /// History manager for trajectories
    history: HistoryManager<State>,
    /// Error tracker for estimation accuracy
    errors: ErrorTracker,
    /// Unique ID for this simulation instance
    id: usize,
    /// Configuration parameters
    config: PFConfig,
    /// Keyboard input state
    pub keyboard: KeyboardInput,
    /// Vehicle state for dynamic mode
    vehicle_state: VehicleState,
    /// Bicycle model for dynamic mode
    bicycle_model: BicycleModel,
    /// Vehicle parameters
    vehicle_params: VehicleParams,
    /// Front tire parameters for Pacejka model
    tire_front: TireParams,
    /// Rear tire parameters for Pacejka model (less grip for oversteer)
    tire_rear: TireParams,
    /// Particle filter state for tracking recovery phases
    pf_state: PFState,
}

impl ParticleFilter {
    pub fn new(id: usize, _time: f32) -> Self {
        let vehicle_params = VehicleParams::default();
        let tire_front = TireParams::default();
        let tire_rear = TireParams::with_peak_force(3600.0); // 80% of front for oversteer
        let mut vehicle_state = VehicleState::new();
        vehicle_state.vx = 5.0; // Start with 5 m/s

        let mut history = HistoryManager::new(HISTORY_LEN);
        history.init_with(zeros!(4, 1));

        Self {
            x_est: zeros!(4, 1),
            x_true: zeros!(4, 1),
            x_dr: zeros!(4, 1),
            p_est: zeros!(3, 3),
            pw: ones!(1, NP) * (1. / NP as f32),
            px: zeros!(4, NP),
            history,
            errors: ErrorTracker::default(),
            id,
            config: PFConfig::default(),
            keyboard: KeyboardInput::default(),
            vehicle_state,
            bicycle_model: BicycleModel::new(vehicle_params, 5.0),
            vehicle_params,
            tire_front,
            tire_rear,
            pf_state: PFState::default(),
        }
    }

    fn update_history(&mut self) {
        self.history.push(self.x_true, self.x_est, self.x_dr);

        // Calculate and store errors
        self.errors.track_positions(
            self.x_true.x(), self.x_true.y(),
            self.x_est.x(), self.x_est.y(),
            self.x_dr.x(), self.x_dr.y(),
        );
    }

    fn calc_input(&self) -> rb::Vector2 {
        match self.config.drive_mode {
            DriveMode::Kinematic => {
                vector![self.config.velocity, self.config.yaw_rate]
            }
            DriveMode::Dynamic => {
                // Use velocity from vehicle state and yaw rate from dynamics
                vector![self.vehicle_state.vx, self.vehicle_state.yaw_rate]
            }
        }
    }

    /// Update keyboard input state
    pub fn handle_keyboard(&mut self, ctx: &egui::Context) {
        if self.config.drive_mode == DriveMode::Dynamic {
            ctx.input(|i| {
                let left = i.key_down(egui::Key::ArrowLeft);
                let right = i.key_down(egui::Key::ArrowRight);
                let up = i.key_down(egui::Key::ArrowUp);
                let down = i.key_down(egui::Key::ArrowDown);
                self.keyboard.update(left, right, up, down);
            });
        }
    }

    /// Check if this vehicle is in dynamic drive mode
    pub fn is_dynamic_mode(&self) -> bool {
        self.config.drive_mode == DriveMode::Dynamic
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
        // Create noise params from config
        let noise = PFNoiseParams {
            obs_noise: self.config.obs_noise,
            motion_noise_v: self.config.motion_noise_v,
            motion_noise_yaw: self.config.motion_noise_yaw.to_radians(),
        };

        match self.config.drive_mode {
            DriveMode::Kinematic => {
                // Original kinematic mode using particle filter observation model
                let u = self.calc_input();
                let (z, ud) = observation(&mut self.x_true, &mut self.x_dr, u, &MARKERS, dt, self.config.max_range);
                self.p_est = pf_localization_with_state(
                    &mut self.x_est, &mut self.px, &mut self.pw, z, ud, dt, &mut self.pf_state, &noise
                );
            }
            DriveMode::Dynamic => {
                // Dynamic mode using bicycle model
                let steering = self.keyboard.steering;
                let acceleration = self.keyboard.acceleration;

                // Sync tire params
                self.bicycle_model.tire_front = self.tire_front;
                self.bicycle_model.tire_rear = self.tire_rear;

                // Step vehicle dynamics
                self.vehicle_state.step(&mut self.bicycle_model, steering, acceleration, dt);

                // Update x_true from vehicle state
                self.x_true = self.vehicle_state.to_vector4();

                // For particle filter, use computed velocity and yaw rate
                let u = self.calc_input();
                let (z, ud) = observation_from_state(&self.x_true, &mut self.x_dr, u, &MARKERS, dt, self.config.max_range);
                self.p_est = pf_localization_with_state(
                    &mut self.x_est, &mut self.px, &mut self.pw, z, ud, dt, &mut self.pf_state, &noise
                );
            }
        }
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
        self.history.clear();
        self.history.init_with(zeros!(4, 1));
        self.errors.clear();
        self.errors.init_with_zero();
        // Reset vehicle state
        self.vehicle_state = VehicleState::new();
        self.vehicle_state.vx = 5.0;
        self.bicycle_model = BicycleModel::new(self.vehicle_params, 5.0);
        self.keyboard = KeyboardInput::default();
        // Reset particle filter state
        self.pf_state = PFState::default();
    }

    fn reset_all(&mut self) {
        self.reset_state();
        self.config = PFConfig::default();
        self.vehicle_params = VehicleParams::default();
        self.tire_front = TireParams::default();
        self.tire_rear = TireParams::with_peak_force(3600.0);
    }
}

impl Draw for ParticleFilter {
    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        // Calculate time offset for x-axis (based on step count and history length)
        let start_step = self.history.start_step();
        let dt = 0.01;

        // Create error plot data with time on x-axis
        let est_error_points: PlotPoints<'_> = self
            .errors
            .get_est_errors()
            .enumerate()
            .map(|(i, &err)| [(start_step + i) as f64 * dt, err as f64])
            .collect();

        let dr_error_points: PlotPoints<'_> = self
            .errors
            .get_dr_errors()
            .enumerate()
            .map(|(i, &err)| [(start_step + i) as f64 * dt, err as f64])
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
        // Draw particles (hidden from legend)
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

        // Draw landmarks and detection lines (hidden from legend)
        MARKERS.iter().for_each(|marker| {
            plot_ui.points(Points::new("", marker_values()).radius(5.0));
            if is_detected(marker, &self.x_true, self.config.max_range) {
                plot_ui.line(
                    Line::new("", values_from_marker_state(marker, &self.x_true))
                        .style(LineStyle::Dotted { spacing: 10.0 }),
                );
            }
        });

        // Label offset above vehicle (in plot coordinates)
        let label_offset = 3.0;

        // Draw trajectories using shared helper
        let true_points: Vec<_> = self.history.get_true().collect();
        draw_trajectory(plot_ui, true_points.into_iter(), egui::Color32::LIGHT_GREEN, 2.0, None);

        let dr_points: Vec<_> = self.history.get_dr().collect();
        draw_trajectory(plot_ui, dr_points.into_iter(), egui::Color32::GRAY, 2.0, None);

        let est_points: Vec<_> = self.history.get_estimated().collect();
        draw_trajectory(plot_ui, est_points.into_iter(), egui::Color32::LIGHT_BLUE, 2.0, None);

        // Draw vehicles with labels using shared helper
        draw_labeled_vehicle_state4(
            plot_ui,
            &self.x_true,
            "GT",
            label_offset,
            self.keyboard.steering,
            &self.vehicle_params,
            &format!("Vehicle {} (True)", self.id),
        );

        draw_labeled_vehicle_state4(
            plot_ui,
            &self.x_dr,
            "DR",
            label_offset,
            self.keyboard.steering,
            &self.vehicle_params,
            &format!("Vehicle {} (Dead Reckoning)", self.id),
        );

        draw_labeled_vehicle_state4(
            plot_ui,
            &self.x_est,
            "PF",
            label_offset,
            self.keyboard.steering,
            &self.vehicle_params,
            &format!("Vehicle {} (Estimate)", self.id),
        );
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        let mut keep = true;
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Vehicle {}", self.id));
                    if ui.small_button("x").clicked() {
                        keep = false;
                    }
                });

                // Drive mode selector
                ui.horizontal(|ui| {
                    ui.label("Drive Mode:");
                    for mode in [DriveMode::Kinematic, DriveMode::Dynamic] {
                        if ui.selectable_label(self.config.drive_mode == mode, mode.label()).clicked() {
                            self.config.drive_mode = mode;
                            if mode == DriveMode::Dynamic {
                                // Initialize vehicle state from current position
                                self.vehicle_state.x = self.x_true.x();
                                self.vehicle_state.y = self.x_true.y();
                                self.vehicle_state.yaw = self.x_true[2];
                                self.vehicle_state.vx = self.config.velocity.max(1.0);
                                self.bicycle_model.vx = self.vehicle_state.vx;
                            }
                        }
                    }
                });

                match self.config.drive_mode {
                    DriveMode::Kinematic => {
                        ui.group(|ui| {
                            ui.label("Motion:");
                            ui.add(
                                DragValue::new(&mut self.config.velocity)
                                    .speed(0.5)
                                    .range(0.0_f32..=30.0)
                                    .prefix("v: ")
                                    .suffix(" m/s"),
                            );
                            ui.add(
                                DragValue::new(&mut self.config.yaw_rate)
                                    .speed(0.01)
                                    .range(-1.0_f32..=1.0)
                                    .prefix("w: ")
                                    .suffix(" rad/s"),
                            );
                        });
                    }
                    DriveMode::Dynamic => {
                        // Compact state display
                        ui.horizontal(|ui| {
                            ui.label(format!("d:{:.0}deg", self.keyboard.steering.to_degrees()));
                            ui.label(format!("v:{:.1} m/s", self.vehicle_state.vx));
                            ui.label(format!("vy:{:.1}", self.vehicle_state.vy));
                        });

                        ui.collapsing("Params", |ui| {
                            // Tire model selector
                            ui.horizontal(|ui| {
                                ui.label("Tire Model:");
                                let current = self.bicycle_model.tire_model;
                                if ui.selectable_label(current == TireModel::Pacejka, "Pacejka").clicked() {
                                    self.bicycle_model.tire_model = TireModel::Pacejka;
                                }
                                if ui.selectable_label(current == TireModel::Linear, "Linear").clicked() {
                                    self.bicycle_model.tire_model = TireModel::Linear;
                                }
                            });

                            ui.add(
                                DragValue::new(&mut self.vehicle_params.mass)
                                    .speed(10.0)
                                    .range(500.0_f32..=3000.0)
                                    .prefix("Mass: ")
                                    .suffix(" kg"),
                            );
                            ui.add(
                                DragValue::new(&mut self.vehicle_params.lf)
                                    .speed(0.1)
                                    .range(0.5_f32..=3.0)
                                    .prefix("Lf: ")
                                    .suffix(" m"),
                            );
                            ui.add(
                                DragValue::new(&mut self.vehicle_params.lr)
                                    .speed(0.1)
                                    .range(0.5_f32..=3.0)
                                    .prefix("Lr: ")
                                    .suffix(" m"),
                            );
                            // Tire parameters (Pacejka only)
                            if self.bicycle_model.tire_model == TireModel::Pacejka {
                                ui.separator();
                                ui.label("Tires (Front/Rear):");
                                ui.horizontal(|ui| {
                                    ui.label("D:");
                                    ui.add(
                                        DragValue::new(&mut self.tire_front.d)
                                            .speed(50.0)
                                            .range(500.0_f32..=8000.0)
                                            .prefix("F:")
                                            .suffix("N"),
                                    );
                                    ui.add(
                                        DragValue::new(&mut self.tire_rear.d)
                                            .speed(50.0)
                                            .range(500.0_f32..=8000.0)
                                            .prefix("R:")
                                            .suffix("N"),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("B:");
                                    ui.add(
                                        DragValue::new(&mut self.tire_front.b)
                                            .speed(0.5)
                                            .range(1.0_f32..=20.0)
                                            .prefix("F:"),
                                    );
                                    ui.add(
                                        DragValue::new(&mut self.tire_rear.b)
                                            .speed(0.5)
                                            .range(1.0_f32..=20.0)
                                            .prefix("R:"),
                                    );
                                });
                            }
                            // Update bicycle model when params change
                            self.bicycle_model.params = self.vehicle_params;
                        });
                    }
                }

                ui.group(|ui| {
                    ui.label("Sensor:");
                    ui.add(
                        DragValue::new(&mut self.config.max_range)
                            .speed(1.0)
                            .range(5.0_f32..=100.0)
                            .prefix("Range: ")
                            .suffix(" m"),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.obs_noise)
                            .speed(0.05)
                            .range(0.1_f32..=5.0)
                            .prefix("Noise: "),
                    );
                });

                ui.group(|ui| {
                    ui.label("PF Noise:");
                    ui.add(
                        DragValue::new(&mut self.config.motion_noise_v)
                            .speed(0.2)
                            .range(0.5_f32..=10.0)
                            .prefix("v: "),
                    );
                    ui.add(
                        DragValue::new(&mut self.config.motion_noise_yaw)
                            .speed(1.0)
                            .range(1.0_f32..=90.0)
                            .prefix("yaw: ")
                            .suffix("deg"),
                    );
                });
            });
        });
        keep
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
