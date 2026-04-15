//! Path planning algorithms
//!
//! This module provides path planning algorithms including:
//! - A* (grid-based, optimal with heuristic)
//! - Dijkstra (grid-based, optimal without heuristic)
//! - RRT (continuous space, probabilistically complete)
//! - Theta* (grid-based, any-angle path planning)
//!
//! The module intentionally exposes both discrete and continuous-space planners
//! because the simulator supports both environment styles:
//!
//! - `Grid` + A* / Dijkstra / Theta*
//! - circular obstacles + RRT

pub mod astar;
pub mod dijkstra;
pub mod grid;
pub mod rrt;
pub mod theta_star;

pub use astar::{AStarPlanner, AStarResult};
pub use dijkstra::{DijkstraPlanner, DijkstraResult};
pub use grid::Grid;
pub use rrt::{CircleObstacle, RrtConfig, RrtNode, RrtPlanner, RrtResult};
pub use theta_star::{ThetaStarPlanner, ThetaStarResult};
