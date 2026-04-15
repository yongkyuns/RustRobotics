use super::super::Simulator;
use egui::{epaint::Hsva, *};
use egui_plot::{Corner, Legend, Plot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendulumPlotTab {
    LateralPosition,
    LateralVelocity,
    RodAngle,
    RodAngularVelocity,
    ControlInput,
}

impl PendulumPlotTab {
    const ALL: [Self; 5] = [
        Self::LateralPosition,
        Self::LateralVelocity,
        Self::RodAngle,
        Self::RodAngularVelocity,
        Self::ControlInput,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::LateralPosition => "Lateral Position",
            Self::LateralVelocity => "Lateral Velocity",
            Self::RodAngle => "Rod Angle",
            Self::RodAngularVelocity => "Rod Angular Velocity",
            Self::ControlInput => "Control Input",
        }
    }

    fn signal_index(self) -> usize {
        match self {
            Self::LateralPosition => 0,
            Self::LateralVelocity => 1,
            Self::RodAngle => 2,
            Self::RodAngularVelocity => 3,
            Self::ControlInput => 4,
        }
    }
}

impl Simulator {
    pub(super) fn pendulum_plot_ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for tab in PendulumPlotTab::ALL {
                ui.selectable_value(&mut self.ui_state.pendulum_plot_tab, tab, tab.label());
            }
        });
        ui.separator();
        Plot::new("PendulumPlot")
            .legend(Legend::default().position(Corner::RightTop))
            .show(ui, |plot_ui| {
                self.simulations.pendulums.iter().for_each(|sim| {
                    sim.plot_signal(plot_ui, self.ui_state.pendulum_plot_tab.signal_index());
                });
            });
    }

    fn pendulum_color(index: usize) -> Color32 {
        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0;
        Hsva::new(index as f32 * golden_ratio, 0.85, 0.5, 1.0).into()
    }

    fn nice_grid_step(span: f32, target_lines: f32) -> f32 {
        let raw = (span / target_lines).max(1.0e-3);
        let magnitude = 10.0_f32.powf(raw.log10().floor());
        let normalized = raw / magnitude;
        let nice = if normalized < 1.5 {
            1.0
        } else if normalized < 3.0 {
            2.0
        } else if normalized < 7.0 {
            5.0
        } else {
            10.0
        };
        nice * magnitude
    }

    pub(super) fn render_pendulum_scene(&self, ui: &mut Ui) -> Rect {
        let desired_size = vec2(
            ui.available_width().max(240.0),
            ui.available_height().max(240.0),
        );
        let (response, painter) = ui.allocate_painter(desired_size, Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let plot_rect = rect;
        let scene_rect = plot_rect.shrink(12.0);
        let pad = 12.0;
        let painter = painter.with_clip_rect(plot_rect);

        painter.rect_filled(plot_rect, 2.0, visuals.extreme_bg_color);
        painter.rect_stroke(
            plot_rect,
            2.0,
            Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
            StrokeKind::Inside,
        );

        if self.simulations.pendulums.is_empty() {
            painter.text(
                plot_rect.center(),
                Align2::CENTER_CENTER,
                "No pendulums",
                FontId::proportional(16.0),
                visuals.weak_text_color(),
            );
            return plot_rect;
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = 0.0_f32;
        let mut max_y = 0.0_f32;

        for sim in &self.simulations.pendulums {
            let model = sim.model();
            let x = sim.x_position();
            let th = sim.rod_angle();
            let r_ball = 0.1 * model.m_ball;
            let r_whl = 0.1 * model.m_cart;
            let w = model.m_cart;
            let h = 0.5 * model.m_cart;
            let len = model.l_bar;
            let rod_bottom_y = h + 2.0 * r_whl;
            let rod_top_x = x - len * th.sin();
            let rod_top_y = rod_bottom_y + len * th.cos();

            min_x = min_x.min(x - w * 0.5);
            min_x = min_x.min(x - w * 0.25 - r_whl);
            min_x = min_x.min(rod_top_x - r_ball);

            max_x = max_x.max(x + w * 0.5);
            max_x = max_x.max(x + w * 0.25 + r_whl);
            max_x = max_x.max(rod_top_x + r_ball);

            min_y = min_y.min(0.0);
            max_y = max_y.max(rod_top_y + r_ball);
            max_y = max_y.max(rod_bottom_y);
        }

        let width = (max_x - min_x).max(2.0);
        let height = (max_y - min_y).max(2.0);
        let usable_width = (scene_rect.width() - 2.0 * pad).max(1.0);
        let usable_height = (scene_rect.height() - 2.0 * pad).max(1.0);
        let scale = (usable_width / width).min(usable_height / height);
        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let scene_center = pos2(
            scene_rect.center().x,
            scene_rect.bottom() - pad - usable_height * 0.5,
        );

        let to_screen = |x: f32, y: f32| -> Pos2 {
            pos2(
                scene_center.x + (x - center_x) * scale,
                scene_center.y - (y - center_y) * scale,
            )
        };

        let ground_y = to_screen(0.0, 0.0).y;
        let grid_color = visuals
            .widgets
            .noninteractive
            .bg_stroke
            .color
            .linear_multiply(0.35);
        let target_grid_px = 56.0_f32;
        let grid_step = Self::nice_grid_step(target_grid_px / scale.max(1.0e-3), 1.0);
        let x_start = (min_x / grid_step).floor() as i32 - 1;
        let x_end = (max_x / grid_step).ceil() as i32 + 1;
        let y_start = (min_y / grid_step).floor() as i32 - 1;
        let y_end = (max_y / grid_step).ceil() as i32 + 1;

        for ix in x_start..=x_end {
            let x = ix as f32 * grid_step;
            let screen_x = to_screen(x, 0.0).x;
            if screen_x < scene_rect.left() || screen_x > scene_rect.right() {
                continue;
            }
            painter.line_segment(
                [
                    pos2(screen_x, scene_rect.top()),
                    pos2(screen_x, scene_rect.bottom()),
                ],
                Stroke::new(1.0, grid_color),
            );
        }

        for iy in y_start..=y_end {
            let y = iy as f32 * grid_step;
            let screen_y = to_screen(0.0, y).y;
            if screen_y < scene_rect.top() || screen_y > scene_rect.bottom() {
                continue;
            }
            painter.line_segment(
                [
                    pos2(scene_rect.left(), screen_y),
                    pos2(scene_rect.right(), screen_y),
                ],
                Stroke::new(1.0, grid_color),
            );
        }

        let origin_x = to_screen(0.0, 0.0).x;
        if origin_x >= scene_rect.left() && origin_x <= scene_rect.right() {
            painter.line_segment(
                [
                    pos2(origin_x, scene_rect.top()),
                    pos2(origin_x, scene_rect.bottom()),
                ],
                Stroke::new(1.5, visuals.widgets.noninteractive.fg_stroke.color),
            );
        }

        painter.line_segment(
            [
                pos2(scene_rect.left() + pad * 0.5, ground_y),
                pos2(scene_rect.right() - pad * 0.5, ground_y),
            ],
            Stroke::new(2.0, visuals.widgets.noninteractive.fg_stroke.color),
        );

        let tick_y0 = ground_y - 4.0;
        let tick_y1 = ground_y + 4.0;
        let label_y = ground_y + 8.0;
        let tick_start = min_x.floor() as i32 - 1;
        let tick_end = max_x.ceil() as i32 + 1;
        for tick in tick_start..=tick_end {
            let x = tick as f32;
            let screen_x = to_screen(x, 0.0).x;
            if screen_x < scene_rect.left() || screen_x > scene_rect.right() {
                continue;
            }
            painter.line_segment(
                [pos2(screen_x, tick_y0), pos2(screen_x, tick_y1)],
                Stroke::new(1.0, visuals.widgets.noninteractive.fg_stroke.color),
            );
            painter.text(
                pos2(screen_x, label_y),
                Align2::CENTER_TOP,
                tick.to_string(),
                FontId::proportional(12.5),
                visuals.text_color(),
            );
        }

        for (index, sim) in self.simulations.pendulums.iter().enumerate() {
            let model = sim.model();
            let x = sim.x_position();
            let th = sim.rod_angle();
            let base_color = Self::pendulum_color(index);
            let stroke = Stroke::new(2.0, visuals.widgets.noninteractive.fg_stroke.color);
            let wheel_stroke = Stroke::new(1.5, stroke.color);
            let fill = base_color.linear_multiply(0.05);

            let r_ball = 0.1 * model.m_ball;
            let r_whl = 0.1 * model.m_cart;
            let w = model.m_cart;
            let h = 0.5 * model.m_cart;
            let len = model.l_bar;

            let body_center = pos2(x, h * 0.5 + 2.0 * r_whl);
            let rod_bottom = pos2(x, h + 2.0 * r_whl);
            let rod_top = pos2(x - len * th.sin(), rod_bottom.y + len * th.cos());
            let left_wheel = pos2(x - w * 0.25, r_whl);
            let right_wheel = pos2(x + w * 0.25, r_whl);
            let wheel_angle = -x / r_whl.max(1.0e-3);

            let body_rect = Rect::from_center_size(
                to_screen(body_center.x, body_center.y),
                vec2(w * scale, h * scale),
            );
            painter.rect_filled(body_rect, 4.0, fill);
            painter.rect_stroke(body_rect, 4.0, stroke, StrokeKind::Inside);

            for wheel_center in [left_wheel, right_wheel] {
                let center = to_screen(wheel_center.x, wheel_center.y);
                let radius = (r_whl * scale).max(2.0);
                painter.circle_filled(center, radius, visuals.panel_fill);
                painter.circle_stroke(center, radius, wheel_stroke);
                let tick = pos2(
                    center.x + radius * wheel_angle.cos(),
                    center.y - radius * wheel_angle.sin(),
                );
                painter.line_segment([center, tick], wheel_stroke);
            }

            let rod_bottom_screen = to_screen(rod_bottom.x, rod_bottom.y);
            let rod_top_screen = to_screen(rod_top.x, rod_top.y);
            painter.line_segment(
                [rod_bottom_screen, rod_top_screen],
                Stroke::new(3.0, base_color),
            );

            let ball_radius = (r_ball * scale).max(3.0);
            painter.circle_filled(rod_top_screen, ball_radius, base_color);
            painter.circle_stroke(rod_top_screen, ball_radius, stroke);
        }

        let legend_font = FontId::proportional(13.5);
        let legend_line_width = 18.0;
        let legend_row_height = 18.0;
        let legend_padding = vec2(8.0, 6.0);
        let legend_entries: Vec<String> = self
            .simulations
            .pendulums
            .iter()
            .map(|sim| format!("Cart {}", sim.id()))
            .collect();
        let legend_text_width = legend_entries
            .iter()
            .map(|label| {
                ui.painter()
                    .layout_no_wrap(label.clone(), legend_font.clone(), visuals.text_color())
                    .size()
                    .x
            })
            .fold(0.0, f32::max);
        let legend_size = vec2(
            legend_padding.x * 2.0 + legend_line_width + 8.0 + legend_text_width,
            legend_padding.y * 2.0 + legend_row_height * legend_entries.len() as f32,
        );
        let legend_rect = Rect::from_min_size(
            pos2(
                plot_rect.right() - 10.0 - legend_size.x,
                plot_rect.top() + 10.0,
            ),
            legend_size,
        );

        painter.rect_filled(
            legend_rect,
            2.0,
            visuals.extreme_bg_color.gamma_multiply(0.92),
        );
        painter.rect_stroke(
            legend_rect,
            2.0,
            visuals.widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        for (index, label) in legend_entries.iter().enumerate() {
            let y = legend_rect.top() + legend_padding.y + legend_row_height * index as f32;
            let color = Self::pendulum_color(index);
            let line_mid_y = y + legend_row_height * 0.5;
            let line_start = pos2(legend_rect.left() + legend_padding.x, line_mid_y);
            let line_end = pos2(line_start.x + legend_line_width, line_mid_y);
            painter.line_segment([line_start, line_end], Stroke::new(3.0, color));
            painter.text(
                pos2(line_end.x + 8.0, y + 1.0),
                Align2::LEFT_TOP,
                label,
                legend_font.clone(),
                visuals.text_color(),
            );
        }

        plot_rect
    }
}
