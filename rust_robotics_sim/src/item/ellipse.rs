use super::{Point, Size, WithAngle, WithPosition, WithSize};
use crate::math::{cos, sin};
use egui::{Color32, Stroke};
use egui_plot::{LineStyle, PlotPoints, Polygon};

pub struct Ellipse {
    size: Size,
    angle: f64,
    position: Point,
    stroke: Stroke,
    fill_alpha: f32,
    style: LineStyle,
}

impl Default for Ellipse {
    fn default() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            size: Size::default(),
            angle: 0.0,
            stroke: Stroke::new(1.0, Color32::TRANSPARENT),
            style: LineStyle::Solid,
            fill_alpha: 0.05,
        }
    }
}

impl Ellipse {
    pub fn into_polygon(self) -> Polygon<'static> {
        let a = self.width() / 2.0; // Horizontal axis
        let b = self.height() / 2.0; // Vertical axis
        let x = self.position().x;
        let y = self.position().y;
        let ang = self.angle();
        let n = (a.max(b) as usize * 50).max(20); // Number of points

        // Equation for ellipse with given range of angle (0 - 2PI)
        Polygon::new(
            "",
            PlotPoints::from_parametric_callback(
                move |t| {
                    let x1 = a * cos(t); // Equation of ellipse for x coordinate
                    let y1 = b * sin(t); // Equation of ellipse for y coordinate

                    let x2 = x1 * cos(ang) - y1 * sin(ang); // Rotate by current angle
                    let y2 = x1 * sin(ang) + y1 * cos(ang);

                    [x2 + x, y2 + y].into() // Translate by current position
                },
                0.0..std::f64::consts::TAU,
                n,
            ),
        )
        .fill_color(Color32::from_black_alpha((self.fill_alpha * 255.0) as u8))
        .stroke(self.stroke)
        .style(self.style)
    }
}

crate::impl_size!(Ellipse);
crate::impl_angle!(Ellipse);
crate::impl_position!(Ellipse);

pub struct Circle {
    size: Size,
    angle: f64,
    position: Point,
    stroke: Stroke,
    fill_alpha: f32,
    style: LineStyle,
}

impl Default for Circle {
    fn default() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            size: Size::default(),
            angle: 0.0,
            stroke: Stroke::new(1.0, Color32::TRANSPARENT),
            style: LineStyle::Solid,
            fill_alpha: 0.05,
        }
    }
}

impl Circle {
    pub fn with_radius(self, radius: f64) -> Self {
        self.with_size(Size {
            width: radius,
            height: radius,
        })
    }
    pub fn into_polygon(self) -> Polygon<'static> {
        let r = self.width().max(self.height());
        let x = self.position().x;
        let y = self.position().y;
        let n = (r as usize * 50).max(20); // Number of points

        // Equation for circle with given range of angle (0 - 2PI)
        Polygon::new(
            "",
            PlotPoints::from_parametric_callback(
                move |t| [r * cos(t) + x, r * sin(t) + y].into(),
                0.0..std::f64::consts::TAU,
                n,
            ),
        )
        .fill_color(Color32::from_black_alpha((self.fill_alpha * 255.0) as u8))
        .stroke(self.stroke)
        .style(self.style)
    }
}

crate::impl_size!(Circle);
crate::impl_angle!(Circle);
crate::impl_position!(Circle);
