//! Common utilities shared between localization and SLAM simulations.
//!
//! This module provides reusable components to reduce code duplication:
//! - [`HistoryManager`]: Manages trajectory history with circular buffers
//! - [`ErrorTracker`]: Tracks position errors over time
//! - Visualization helpers for drawing trajectories, vehicles, and covariance ellipses
//! - UI builders for common control groups
//! - Noise generation utilities

pub mod error_tracking;
pub mod history;
pub mod noise;
pub mod ui_builders;
pub mod visualization;

pub use error_tracking::ErrorTracker;
pub use history::HistoryManager;
pub use noise::{gaussian_noise, rand_noise};
pub use ui_builders::{DisplayToggles, MotionConfig, SensorConfig};
pub use visualization::{draw_covariance_ellipse, draw_labeled_vehicle, draw_labeled_vehicle_state4, draw_trajectory};
