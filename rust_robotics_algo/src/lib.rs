//! Reusable robotics algorithms and robot-framework logic for the workspace.
//!
//! This crate is the algorithmic counterpart to `rust_robotics_sim`. It is
//! meant to own reusable robotics logic rather than live world state or UI.
//! The main subsystems are:
//!
//! - `control`: classical control models and dynamics
//! - `localization`: particle-filter-based localization helpers
//! - `path_planning`: grid and continuous-space planners
//! - `slam`: EKF and graph-based SLAM implementations
//! - `robot_fw`: shared observation / command / actuation logic for robots
//!
//! The design goal is that these modules remain usable outside the simulator.
//! They should encode robotics semantics and algorithms, while the simulator
//! owns rendering, MuJoCo world stepping, and interactive orchestration.
// #![no_std]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod control;
pub mod localization;
pub mod path_planning;
pub mod robot_fw;
pub mod slam;
pub mod util;
pub mod prelude {
    pub use crate::util::*;
    pub use crate::*;
    pub use control::inverted_pendulum;
    pub use localization::particle_filter as pf;
    pub use nalgebra;
    pub use nalgebra::{matrix, vector};
    pub use path_planning::{
        AStarPlanner, AStarResult, CircleObstacle, DijkstraPlanner, DijkstraResult, Grid,
        RrtConfig, RrtNode, RrtPlanner, RrtResult,
    };

    // #[cfg(not(feature = "libm"))]
    // pub mod std {
    //     extern crate std;
    //     pub use std::{println, vec::Vec};
    // }
}

#[cfg(feature = "numpy")]
pub use nalgebra_numpy::matrix_from_numpy;

pub use prelude::*;
