//! Reusable UI control groups for simulation parameters.

use egui::{DragValue, Ui};

/// Motion configuration for robot velocity and yaw rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionConfig {
    /// Forward velocity (m/s)
    pub velocity: f32,
    /// Yaw rate (rad/s)
    pub yaw_rate: f32,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            velocity: 1.5,
            yaw_rate: 0.15,
        }
    }
}

impl MotionConfig {
    /// Create a motion config with higher default velocity (for localization).
    pub fn with_high_velocity() -> Self {
        Self {
            velocity: 5.0,
            yaw_rate: 0.1,
        }
    }
}

/// Sensor configuration for range and noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorConfig {
    /// Maximum detection range (m)
    pub max_range: f32,
    /// Observation noise standard deviation
    pub obs_noise: f32,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            max_range: 50.0,
            obs_noise: 1.0,
        }
    }
}

impl SensorConfig {
    /// Create a sensor config for SLAM (shorter range).
    pub fn for_slam() -> Self {
        Self {
            max_range: 30.0,
            obs_noise: 1.0,
        }
    }
}

/// Display toggles for visualization options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayToggles {
    /// Show covariance ellipses
    pub covariance: bool,
    /// Show observation lines
    pub observations: bool,
    /// Show dead reckoning trajectory
    pub dead_reckoning: bool,
    /// Show true path/trajectory
    pub true_path: bool,
    /// Show true landmarks (SLAM-specific)
    pub true_landmarks: bool,
}

impl Default for DisplayToggles {
    fn default() -> Self {
        Self {
            covariance: true,
            observations: true,
            dead_reckoning: true,
            true_path: true,
            true_landmarks: true,
        }
    }
}

/// Draw motion controls UI group.
///
/// Returns true if any value changed.
pub fn motion_controls_ui(ui: &mut Ui, config: &mut MotionConfig) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label("Motion:");
        changed |= ui
            .add(
                DragValue::new(&mut config.velocity)
                    .speed(0.05)
                    .range(0.1_f32..=10.0)
                    .prefix("v: ")
                    .suffix(" m/s"),
            )
            .changed();
        changed |= ui
            .add(
                DragValue::new(&mut config.yaw_rate)
                    .speed(0.01)
                    .range(-0.5_f32..=0.5)
                    .prefix("w: ")
                    .suffix(" rad/s"),
            )
            .changed();
    });
    changed
}

/// Draw motion controls with extended velocity range (for localization).
///
/// Returns true if any value changed.
pub fn motion_controls_extended_ui(ui: &mut Ui, config: &mut MotionConfig) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label("Motion:");
        changed |= ui
            .add(
                DragValue::new(&mut config.velocity)
                    .speed(0.5)
                    .range(0.0_f32..=30.0)
                    .prefix("v: ")
                    .suffix(" m/s"),
            )
            .changed();
        changed |= ui
            .add(
                DragValue::new(&mut config.yaw_rate)
                    .speed(0.01)
                    .range(-1.0_f32..=1.0)
                    .prefix("w: ")
                    .suffix(" rad/s"),
            )
            .changed();
    });
    changed
}

/// Draw sensor controls UI group.
///
/// Returns true if any value changed.
pub fn sensor_controls_ui(ui: &mut Ui, config: &mut SensorConfig) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label("Sensor:");
        changed |= ui
            .add(
                DragValue::new(&mut config.max_range)
                    .speed(1.0)
                    .range(5.0_f32..=100.0)
                    .prefix("Range: ")
                    .suffix(" m"),
            )
            .changed();
        changed |= ui
            .add(
                DragValue::new(&mut config.obs_noise)
                    .speed(0.05)
                    .range(0.1_f32..=5.0)
                    .prefix("Noise: "),
            )
            .changed();
    });
    changed
}

/// Draw display toggles UI group.
///
/// # Arguments
/// * `ui` - The UI to draw on
/// * `toggles` - Display toggles to modify
/// * `show_landmarks` - Whether to show the landmarks toggle (SLAM-specific)
pub fn display_toggles_ui(ui: &mut Ui, toggles: &mut DisplayToggles, show_landmarks: bool) {
    ui.group(|ui| {
        ui.label("Display:");
        ui.checkbox(&mut toggles.covariance, "Covariance");
        ui.checkbox(&mut toggles.observations, "Observations");
        ui.checkbox(&mut toggles.dead_reckoning, "Dead Reckoning");
        if show_landmarks {
            ui.checkbox(&mut toggles.true_landmarks, "True Landmarks");
        }
    });
}

/// Draw error statistics display.
pub fn error_stats_ui(ui: &mut Ui, est_label: &str, est_err: f32, dr_err: f32) {
    ui.separator();
    ui.label(format!("{} err: {:.2} m", est_label, est_err));
    ui.label(format!("DR err: {:.2} m", dr_err));
}
