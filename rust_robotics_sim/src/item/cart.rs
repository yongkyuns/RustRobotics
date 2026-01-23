use super::{Circle, Point, Rectangle, Shape, WithPosition, WithSize};
use crate::math::{cos, sin};
use egui_plot::{Line, PlotUi, PlotPoints};
use rust_robotics_algo::inverted_pendulum::Model;

pub fn draw_cart(plot_ui: &mut PlotUi<'_>, x_pos: f32, rod_angle: f32, model: &Model, name: &str) {
    let x = x_pos as f64;
    let y = 0.0;

    let r_ball = 0.1 * model.m_ball as f64;
    let r_whl = 0.1 * model.m_cart as f64;
    let w = 1.0 * model.m_cart as f64;
    let h = 0.5 * model.m_cart as f64;
    let len = model.l_bar as f64;
    let th = rod_angle as f64;

    let rod_bottom = Point::new(x, y + h + 2.0 * r_whl);
    let rod_top = Point::new(rod_bottom.x - len * sin(th), rod_bottom.y + len * cos(th));

    let body = Rectangle::new()
        .with_width(w)
        .with_height(h)
        .at(x, y + h / 2.0 + 2.0 * r_whl)
        .into_polygon();
    let left_wheel = Circle::new()
        .with_radius(r_whl)
        .at(x - w / 4.0, y + r_whl)
        .into_polygon();
    let right_wheel = Circle::new()
        .with_radius(r_whl)
        .at(x + w / 4.0, y + r_whl)
        .into_polygon();
    let ball = Circle::new()
        .with_radius(r_ball)
        .at(rod_top.x, rod_top.y)
        .into_polygon();
    let rod = Line::new(
        "",
        PlotPoints::new(vec![
            [rod_bottom.x, rod_bottom.y],
            [rod_top.x, rod_top.y],
        ]),
    );

    plot_ui.polygon(body.name(name));
    plot_ui.polygon(left_wheel.name(""));  // Hide from legend
    plot_ui.polygon(right_wheel.name("")); // Hide from legend
    plot_ui.polygon(ball.name(""));        // Hide from legend
    plot_ui.line(rod.name(""));            // Hide from legend
}
