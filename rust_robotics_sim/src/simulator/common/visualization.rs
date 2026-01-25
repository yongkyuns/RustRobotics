//! Visualization helpers for trajectory, vehicle, and covariance rendering.

use egui::Color32;
use egui_plot::{Line, LineStyle, PlotPoint, PlotPoints, PlotUi, Polygon, Text};
use rust_robotics_algo::nalgebra::{Matrix2, SymmetricEigen};
use rust_robotics_algo::control::vehicle::VehicleParams;
use rust_robotics_algo::nalgebra::Vector4 as NaVector4;
use rust_robotics_algo::Vector4;

use crate::item::draw_vehicle;

/// Trait for types that can provide 2D position coordinates.
pub trait HasPosition {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
}

/// Implement HasPosition for Vector4 (state: [x, y, phi, v])
impl HasPosition for rust_robotics_algo::Vector4 {
    fn x(&self) -> f32 {
        self[0]
    }
    fn y(&self) -> f32 {
        self[1]
    }
}

/// Implement HasPosition for Vector3 (pose: [x, y, theta])
impl HasPosition for rust_robotics_algo::Vector3 {
    fn x(&self) -> f32 {
        self[0]
    }
    fn y(&self) -> f32 {
        self[1]
    }
}

/// Draw a trajectory line from position history.
///
/// # Arguments
/// * `plot_ui` - The plot UI to draw on
/// * `history` - Iterator of positions
/// * `color` - Line color
/// * `width` - Line width
/// * `style` - Optional line style (solid if None)
pub fn draw_trajectory<'a, T: HasPosition + 'a>(
    plot_ui: &mut PlotUi<'_>,
    history: impl Iterator<Item = &'a T>,
    color: Color32,
    width: f32,
    style: Option<LineStyle>,
) {
    let points: PlotPoints<'_> = history
        .map(|p| [p.x() as f64, p.y() as f64])
        .collect();

    let mut line = Line::new("", points).color(color).width(width);

    if let Some(s) = style {
        line = line.style(s);
    }

    plot_ui.line(line);
}

/// Draw a vehicle with a text label above it.
///
/// # Arguments
/// * `plot_ui` - The plot UI to draw on
/// * `pose` - Robot pose [x, y, theta] (Vector3)
/// * `velocity` - Current velocity for state conversion
/// * `label` - Text label to display above the vehicle
/// * `label_offset` - Vertical offset for the label in plot units
/// * `steering` - Front wheel steering angle in radians
/// * `params` - Vehicle visualization parameters
/// * `vehicle_name` - Name for the vehicle in legend
pub fn draw_labeled_vehicle(
    plot_ui: &mut PlotUi<'_>,
    pose: &rust_robotics_algo::Vector3,
    velocity: f32,
    label: &str,
    label_offset: f32,
    steering: f32,
    params: &VehicleParams,
    vehicle_name: &str,
) {
    // Convert Vector3 (x, y, theta) to Vector4 (x, y, phi, v) for draw_vehicle
    let state: Vector4 = NaVector4::new(pose[0], pose[1], pose[2], velocity);

    draw_vehicle(plot_ui, state, vehicle_name, steering, params);

    plot_ui.text(Text::new(
        "",
        PlotPoint::new(pose[0] as f64, pose[1] as f64 + label_offset as f64),
        label,
    ));
}

/// Draw a vehicle from Vector4 state with a text label above it.
///
/// # Arguments
/// * `plot_ui` - The plot UI to draw on
/// * `state` - Robot state [x, y, phi, v] (Vector4)
/// * `label` - Text label to display above the vehicle
/// * `label_offset` - Vertical offset for the label in plot units
/// * `steering` - Front wheel steering angle in radians
/// * `params` - Vehicle visualization parameters
/// * `vehicle_name` - Name for the vehicle in legend
pub fn draw_labeled_vehicle_state4(
    plot_ui: &mut PlotUi<'_>,
    state: &rust_robotics_algo::Vector4,
    label: &str,
    label_offset: f32,
    steering: f32,
    params: &VehicleParams,
    vehicle_name: &str,
) {
    draw_vehicle(plot_ui, *state, vehicle_name, steering, params);

    plot_ui.text(Text::new(
        "",
        PlotPoint::new(state[0] as f64, state[1] as f64 + label_offset as f64),
        label,
    ));
}

/// Compute points for a 2D covariance ellipse.
///
/// # Arguments
/// * `cx` - Center x coordinate
/// * `cy` - Center y coordinate
/// * `cov` - 2x2 covariance matrix
/// * `n_sigma` - Number of standard deviations for the ellipse size
///
/// # Returns
/// Vector of [x, y] points forming the ellipse
pub fn covariance_ellipse_points(
    cx: f32,
    cy: f32,
    cov: &Matrix2<f32>,
    n_sigma: f32,
) -> Vec<[f64; 2]> {
    // Eigendecomposition of covariance matrix
    let eigen = SymmetricEigen::new(cov.clone());
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Semi-axes lengths (n_sigma standard deviations)
    let a = n_sigma * eigenvalues[0].abs().sqrt();
    let b = n_sigma * eigenvalues[1].abs().sqrt();

    // Rotation angle from eigenvectors
    let angle = eigenvectors[(1, 0)].atan2(eigenvectors[(0, 0)]);

    // Generate ellipse points
    let n_points = 32;
    (0..=n_points)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n_points as f32;
            let ex = a * t.cos();
            let ey = b * t.sin();
            // Rotate and translate
            let rx = ex * angle.cos() - ey * angle.sin() + cx;
            let ry = ex * angle.sin() + ey * angle.cos() + cy;
            [rx as f64, ry as f64]
        })
        .collect()
}

/// Draw a covariance ellipse on the plot.
///
/// # Arguments
/// * `plot_ui` - The plot UI to draw on
/// * `cx` - Center x coordinate
/// * `cy` - Center y coordinate
/// * `cov` - 2x2 covariance matrix
/// * `n_sigma` - Number of standard deviations for the ellipse size
/// * `color` - Fill and stroke color (fill will be semi-transparent)
pub fn draw_covariance_ellipse(
    plot_ui: &mut PlotUi<'_>,
    cx: f32,
    cy: f32,
    cov: &Matrix2<f32>,
    n_sigma: f32,
    color: Color32,
) {
    let ellipse = covariance_ellipse_points(cx, cy, cov, n_sigma);
    plot_ui.polygon(
        Polygon::new("", PlotPoints::new(ellipse))
            .stroke(egui::Stroke::new(1.5, color.gamma_multiply(0.7)))
            .fill_color(color.gamma_multiply(0.15)),
    );
}

/// Standard colors for different trajectory types.
pub mod colors {
    use egui::Color32;

    /// Color for true/ground truth trajectory (lime green)
    pub const TRUE: Color32 = Color32::from_rgb(50, 205, 50);

    /// Color for estimated trajectory (royal blue)
    pub const ESTIMATED: Color32 = Color32::from_rgb(65, 105, 225);

    /// Color for dead reckoning trajectory (gray)
    pub const DR: Color32 = Color32::from_rgb(100, 100, 100);

    /// Color for observations (yellow-orange)
    pub const OBSERVATION: Color32 = Color32::from_rgb(255, 200, 50);

    /// Color for estimated landmarks (cornflower blue)
    pub const LANDMARK_EST: Color32 = Color32::from_rgb(100, 149, 237);
}
