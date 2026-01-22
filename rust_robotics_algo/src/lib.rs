// #![no_std]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod control;
pub mod localization;
pub mod util;
pub mod prelude {
    pub use crate::util::*;
    pub use crate::*;
    pub use control::inverted_pendulum;
    pub use localization::particle_filter as pf;
    pub use nalgebra;
    pub use nalgebra::{matrix, vector};


    // #[cfg(not(feature = "libm"))]
    // pub mod std {
    //     extern crate std;
    //     pub use std::{println, vec::Vec};
    // }
}

#[cfg(feature = "numpy")]
pub use nalgebra_numpy::matrix_from_numpy;

pub use prelude::*;
