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

    // Draw wheel tick marks to show rotation
    // Wheel rotation angle = distance traveled / radius
    let wheel_angle = x / r_whl;

    // Left wheel tick
    let left_wheel_cx = x - w / 4.0;
    let left_wheel_cy = y + r_whl;
    let left_tick = Line::new(
        "",
        PlotPoints::new(vec![
            [left_wheel_cx, left_wheel_cy],
            [left_wheel_cx + r_whl * cos(wheel_angle), left_wheel_cy + r_whl * sin(wheel_angle)],
        ]),
    );
    plot_ui.line(left_tick);

    // Right wheel tick
    let right_wheel_cx = x + w / 4.0;
    let right_wheel_cy = y + r_whl;
    let right_tick = Line::new(
        "",
        PlotPoints::new(vec![
            [right_wheel_cx, right_wheel_cy],
            [right_wheel_cx + r_whl * cos(wheel_angle), right_wheel_cy + r_whl * sin(wheel_angle)],
        ]),
    );
    plot_ui.line(right_tick);
}
