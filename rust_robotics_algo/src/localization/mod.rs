//! Localization helpers and state conventions.
//!
//! The localization demos in this workspace use a common vehicle state ordering:
//!
//! `x = [pos_x, pos_y, heading, speed]`
//!
//! This module provides small shared traits and the particle-filter
//! implementation used by the simulator.
use crate::prelude::*;

pub mod particle_filter;

/// Getter methods for state vector
pub trait StateVector {
    /// X Position [m]
    fn x(&self) -> f32;
    /// Y Position [m]
    fn y(&self) -> f32;
    /// Heading Angle [rad]
    fn phi(&self) -> f32;
    /// Velocity [m/s]
    fn v(&self) -> f32;
}

impl StateVector for Vector4 {
    fn x(&self) -> f32 {
        *self.get(0).expect("Cannot get 1st element of Vector4")
    }
    fn y(&self) -> f32 {
        *self.get(1).expect("Cannot get 2nd element of Vector4")
    }
    fn phi(&self) -> f32 {
        *self.get(2).expect("Cannot get 3rd element of Vector4")
    }
    fn v(&self) -> f32 {
        *self.get(3).expect("Cannot get 4th element of Vector4")
    }
}
