//! Shared simulator UI orchestration.
//!
//! The UI layer is split into:
//!
//! - `common`: top-level layout and controls shared across modes
//! - `modes`: mode-specific cards, instructions, and scene/graph dispatch
//! - `pendulum`: pendulum-specific plots and scene rendering helpers
mod common;
mod modes;
mod pendulum;

pub(super) use pendulum::PendulumPlotTab;
