use super::{Point, Shape, Size, WithAngle, WithPosition, WithSize};
use egui::{Color32, Stroke};
use egui_plot::{LineStyle, Polygon};

pub struct Rectangle {
    size: Size,
    angle: f64,
    position: Point,
    stroke: Stroke,
    fill_alpha: f32,
    style: LineStyle,
}

impl Default for Rectangle {
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

impl Rectangle {
    pub fn into_polygon(self) -> Polygon<'static> {
        Polygon::new("", self.bounding_box())
            .fill_color(Color32::from_black_alpha((self.fill_alpha * 255.0) as u8))
            .stroke(self.stroke)
            .style(self.style)
    }
}

crate::impl_size!(Rectangle);
crate::impl_angle!(Rectangle);
crate::impl_position!(Rectangle);
