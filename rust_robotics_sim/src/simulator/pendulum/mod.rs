//! Inverted pendulum simulator mode.
//!
//! This module is intentionally split by concern:
//!
//! - `domain`: plant state, classical controllers, stepping, and noise model
//! - `training`: PPO coordinator integration and policy-selection logic
//! - `ui`: user-facing controls for controller choice and tuning
//! - `tests`: contract and regression coverage
#![allow(non_snake_case)]

mod domain;
#[cfg(test)]
mod tests;
mod training;
mod ui;

pub use domain::{Controller, InvertedPendulum, NoiseConfig, State, PENDULUM_FIXED_DT};
