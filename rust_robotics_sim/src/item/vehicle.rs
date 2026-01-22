use super::{Rectangle, Shape, WithAngle, WithPosition, WithSize};
use egui_plot::PlotUi;
use rust_robotics_algo::localization::StateVector;
use rust_robotics_algo::Vector4;

//x, y, phi, v
pub fn draw_vehicle(plot_ui: &mut PlotUi<'_>, state: Vector4, name: &str) {
    let w = 0.4;
    let h = 0.2;
    let x = state.x() as f64;
    let y = state.y() as f64;
    let ang = state.phi() as f64;

    let body = Rectangle::new()
        .with_width(w)
        .with_height(h)
        .with_angle(ang)
        .at(x, y)
        .into_polygon();

    plot_ui.polygon(body.name(name));
}
