//! PPO training runtime for the Rust Robotics workspace.
//!
//! This crate owns:
//!
//! - environment rollouts
//! - Burn-backed actor / critic models
//! - PPO optimization logic
//! - conversion between framework-specific models and portable snapshot DTOs
//!
//! It intentionally does **not** own UI, simulator stepping, or MuJoCo world
//! state. The simulator interacts with this crate through exported config
//! structs and snapshot types, so trained policies can be consumed without
//! embedding the full training runtime inside the UI layer.
pub mod algorithm;
pub mod backend;
pub mod env;
pub mod model;
pub mod trainer;

/// Portable snapshot and metric types shared with the simulator.
pub use rust_robotics_core::{
    LinearSnapshot, PolicySnapshot, PpoMetrics, PpoSharedState, ValueSnapshot,
};

pub use algorithm::{AlgorithmKind, DqnConfig, PpoConfig, SacConfig};
pub use backend::{default_train_device, AutodiffBackend, TrainBackend, TrainDevice};
pub use env::{PendulumEnv, PendulumEnvConfig, StepResult};
pub use trainer::{PpoTrainerConfig, PpoTrainerSession};
