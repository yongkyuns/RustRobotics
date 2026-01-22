#![allow(non_snake_case)]

use nannou::prelude::*;
use rust_robotics_algo as rb;
use rust_robotics_algo::control::LQR;
use rust_robotics_algo::inverted_pendulum::*;
use rust_robotics_algo::prelude::*;

mod util;
use util::{draw_cart, draw_grid, Color};

/// Available controller types
#[derive(Debug, Clone, Copy, PartialEq)]
enum ControllerType {
    None,
    LQR,
    PID,
    MPC,
}

impl ControllerType {
    fn name(&self) -> &'static str {
        match self {
            ControllerType::None => "None (Open Loop)",
            ControllerType::LQR => "LQR",
            ControllerType::PID => "PID",
            ControllerType::MPC => "MPC",
        }
    }

    fn next(&self) -> Self {
        match self {
            ControllerType::None => ControllerType::LQR,
            ControllerType::LQR => ControllerType::PID,
            ControllerType::PID => ControllerType::MPC,
            ControllerType::MPC => ControllerType::None,
        }
    }
}

struct InvertedPendulum {
    state: rb::Vector4,
    model: Model,
    controller: ControllerType,
    pid: PID,
    control_input: f32,
}

impl InvertedPendulum {
    pub fn new() -> Self {
        Self {
            state: vector![0., 0., random_range(-0.3, 0.3), 0.],
            model: Model::default(),
            controller: ControllerType::MPC,
            pid: PID::with_gains(100.0, 1.0, 20.0),
            control_input: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.state = vector![0., 0., random_range(-0.3, 0.3), 0.];
        self.pid.reset_state();
        self.control_input = 0.0;
    }

    pub fn step(&mut self, dt: f32) {
        let x = self.state;
        let (A, B) = self.model.model(dt);

        // Compute control input based on selected controller
        let u = match self.controller {
            ControllerType::None => 0.0,
            ControllerType::LQR => {
                // LQR control: minimize state error
                let u_vec = self.model.control(x, dt);
                u_vec[0]
            }
            ControllerType::PID => {
                // PID control: control based on angle error
                // Target angle is 0 (upright)
                let angle_error = -x[2]; // Negative because we want to correct the angle
                self.pid.control(angle_error, dt)
            }
            ControllerType::MPC => {
                // MPC control: optimal control over prediction horizon
                mpc_control(x, self.model, dt)
            }
        };

        // Clamp control input to reasonable bounds
        let u = u.clamp(-50.0, 50.0);
        self.control_input = u;

        // Update simulation based on control input
        self.state = A * x + B * u;
    }
}

fn main() {
    nannou::app(model).update(update).view(draw).run()
}

fn model(app: &App) -> InvertedPendulum {
    app.new_window()
        .size(800, 600)
        .key_pressed(key_pressed)
        .build()
        .unwrap();
    InvertedPendulum::new()
}

fn update(app: &App, pendulum: &mut InvertedPendulum, _update: Update) {
    let dt = 1.0 / app.fps();
    pendulum.step(dt);
}

fn draw(app: &App, pendulum: &InvertedPendulum, frame: Frame) {
    let draw = app.draw();
    let win_rect = app.main_window().rect();
    draw.background().rgb(0.11, 0.12, 0.13);

    let x_pos = pendulum.state[0];
    let angle = pendulum.state[2];

    let zoom = 100.0;

    draw_grid(&draw, &win_rect, 100.0, 1.0, zoom, true);
    draw_grid(&draw, &win_rect, 25.0, 0.5, zoom, false);

    // Draw HUD
    let hud_color = nannou::color::rgb(0.8, 0.8, 0.8);
    let info = format!(
        "Controller: {} (Press C to cycle)\n\
         State: pos={:.2}m, vel={:.2}m/s, angle={:.2}°, ω={:.2}°/s\n\
         Control: {:.2}N\n\
         Press SPACE to reset",
        pendulum.controller.name(),
        pendulum.state[0],
        pendulum.state[1],
        pendulum.state[2].to_degrees(),
        pendulum.state[3].to_degrees(),
        pendulum.control_input,
    );
    draw.text(&info)
        .color(hud_color)
        .font_size(14)
        .left_justify()
        .line_spacing(4.0)
        .x_y(win_rect.left() + 150.0, win_rect.top() - 50.0);

    // Draw controller indicator
    let controller_color = match pendulum.controller {
        ControllerType::None => nannou::color::rgb(0.5, 0.5, 0.5),
        ControllerType::LQR => Color::coral(),
        ControllerType::PID => Color::yellow(),
        ControllerType::MPC => Color::orange(),
    };
    draw.ellipse()
        .color(controller_color)
        .x_y(win_rect.right() - 30.0, win_rect.top() - 30.0)
        .radius(10.0);

    let draw = draw.scale(zoom);

    draw_cart(&draw, x_pos, angle);

    draw.to_frame(app, &frame).unwrap();
}

fn key_pressed(_app: &App, pendulum: &mut InvertedPendulum, key: Key) {
    match key {
        Key::Return | Key::Space => {
            pendulum.reset();
        }
        Key::C => {
            pendulum.controller = pendulum.controller.next();
            pendulum.pid.reset_state();
            println!("Switched to: {}", pendulum.controller.name());
        }
        Key::Key1 => {
            pendulum.controller = ControllerType::None;
            println!("Switched to: {}", pendulum.controller.name());
        }
        Key::Key2 => {
            pendulum.controller = ControllerType::LQR;
            println!("Switched to: {}", pendulum.controller.name());
        }
        Key::Key3 => {
            pendulum.controller = ControllerType::PID;
            pendulum.pid.reset_state();
            println!("Switched to: {}", pendulum.controller.name());
        }
        Key::Key4 => {
            pendulum.controller = ControllerType::MPC;
            println!("Switched to: {}", pendulum.controller.name());
        }
        _ => {}
    }
}
