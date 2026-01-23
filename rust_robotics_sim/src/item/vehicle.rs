use super::{Rectangle, Shape, WithAngle, WithPosition, WithSize};
use egui::Color32;
use egui_plot::{PlotPoints, PlotUi, Polygon};
use rust_robotics_algo::control::vehicle::VehicleParams;
use rust_robotics_algo::localization::StateVector;
use rust_robotics_algo::Vector4;

/// Transform a local point to global coordinates
fn local_to_global(local_x: f64, local_y: f64, x: f64, y: f64, ang: f64) -> [f64; 2] {
    let cos_a = ang.cos();
    let sin_a = ang.sin();
    [
        x + local_x * cos_a - local_y * sin_a,
        y + local_x * sin_a + local_y * cos_a,
    ]
}

/// Reference values for scaling (all dimensions in meters to match world coordinates)
const REF_MASS: f64 = 1500.0; // Reference mass in kg
const BASE_WIDTH: f64 = 1.8; // Base vehicle width at reference mass [m]
const FRONT_OVERHANG: f64 = 0.8; // Fixed front overhang beyond front axle [m]
const REAR_OVERHANG: f64 = 0.5; // Fixed rear overhang beyond rear axle [m]

/// Draw a vehicle with steering angle and physical parameters
/// state: [x, y, phi, v]
/// steering: front wheel steering angle in radians
/// params: vehicle parameters (lf, lr, mass) for sizing
pub fn draw_vehicle(plot_ui: &mut PlotUi<'_>, state: Vector4, name: &str, steering: f32, params: &VehicleParams) {
    let x = state.x() as f64;
    let y = state.y() as f64;
    let ang = state.phi() as f64;
    let steer = steering as f64;

    // Vehicle dimensions based on parameters (in meters)
    let lf = params.lf as f64;
    let lr = params.lr as f64;
    let mass = params.mass as f64;

    // Wheel positions from CG (in meters)
    let front_axle = lf;
    let rear_axle = lr;

    // Body extends from rear overhang to front overhang
    let front_overhang = FRONT_OVERHANG;
    let rear_overhang = REAR_OVERHANG;

    // Body dimensions
    let body_front = front_axle + front_overhang; // Front edge from CG
    let body_rear = rear_axle + rear_overhang;    // Rear edge from CG
    let body_w = body_front + body_rear;          // Total body length
    let body_center_offset = (body_front - body_rear) / 2.0; // Body center offset from CG

    // Width scales with sqrt of mass
    let mass_scale = (mass / REF_MASS).sqrt();
    let body_h = BASE_WIDTH * mass_scale;

    // Draw main body (centered at body center, not CG)
    let [body_cx, body_cy] = local_to_global(body_center_offset, 0.0, x, y, ang);
    let body = Rectangle::new()
        .with_width(body_w)
        .with_height(body_h)
        .with_angle(ang)
        .at(body_cx, body_cy)
        .into_polygon();
    plot_ui.polygon(body.name(name));

    // Draw front triangle indicator at front of body
    let tri_base = body_h; // Same width as car body
    let tri_height = body_w * 0.15; // Proportional to body length
    let tri_offset = body_front; // Front of vehicle body from CG

    let tri_points = PlotPoints::new(vec![
        local_to_global(tri_offset, tri_base * 0.5, x, y, ang),
        local_to_global(tri_offset, -tri_base * 0.5, x, y, ang),
        local_to_global(tri_offset + tri_height, 0.0, x, y, ang),
    ]);
    let triangle = Polygon::new(name, tri_points)
        .fill_color(Color32::TRANSPARENT);
    plot_ui.polygon(triangle);

    // Draw wheels - fixed size in meters
    let wheel_w = 0.6; // Wheel diameter [m]
    let wheel_h = 0.25; // Wheel width [m]
    let wheel_y_offset = body_h * 0.5 + wheel_h * 0.5; // Distance from center along y

    // Wheel positions based on actual lf/lr from CG: (local_x, local_y, is_front)
    let wheel_configs = [
        (front_axle, wheel_y_offset, true),    // Front-left
        (front_axle, -wheel_y_offset, true),   // Front-right
        (-rear_axle, wheel_y_offset, false),   // Rear-left
        (-rear_axle, -wheel_y_offset, false),  // Rear-right
    ];

    for (wx, wy, is_front) in wheel_configs {
        let [gx, gy] = local_to_global(wx, wy, x, y, ang);
        // Front wheels rotate with steering angle
        let wheel_ang = if is_front { ang + steer } else { ang };
        let wheel = Rectangle::new()
            .with_width(wheel_w)
            .with_height(wheel_h)
            .with_angle(wheel_ang)
            .at(gx, gy)
            .into_polygon()
            .fill_color(Color32::from_gray(60));
        plot_ui.polygon(wheel);
    }
}
