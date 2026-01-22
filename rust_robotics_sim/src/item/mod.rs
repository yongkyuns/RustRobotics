mod cart;
mod ellipse;
mod rectangle;
mod vehicle;

pub use cart::draw_cart;
pub use ellipse::{Circle, Ellipse};
pub use rectangle::Rectangle;
pub use vehicle::draw_vehicle;

use crate::prelude::*;
use egui_plot::{PlotPoint, PlotPoints};

pub type Point = PlotPoint;

impl WithAngle for Point {
    fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }
    fn with_angle(self, rad: f64) -> Self {
        self.rotate_to(rad)
    }
    fn rotate_by(self, rad: f64) -> Self {
        Point::new(
            self.x * cos(rad) - self.y * sin(rad),
            self.x * sin(rad) + self.y * cos(rad),
        )
    }
    fn rotate_to(self, rad: f64) -> Self {
        let rad = rad - self.angle();
        self.rotate_by(rad)
    }
}

impl WithPosition for Point {
    fn position(&self) -> Point {
        *self
    }
    fn at(self, x: f64, y: f64) -> Self {
        Point::new(x, y)
    }
    fn move_to(self, position: Point) -> Self {
        position
    }
    fn move_by(mut self, vector: Vector) -> Self {
        self.x += vector.x;
        self.y += vector.y;
        self
    }
}

pub type Vector = Point;

pub struct Size {
    width: f64,
    height: f64,
}

impl Default for Size {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
        }
    }
}

pub trait WithSize {
    fn with_width(self, width: f64) -> Self;
    fn with_height(self, height: f64) -> Self;
    fn with_size(self, size: Size) -> Self;
    fn scale(self, by: f64) -> Self;
    fn width(&self) -> f64;
    fn height(&self) -> f64;
}

pub trait WithAngle {
    fn angle(&self) -> f64;
    fn with_angle(self, rad: f64) -> Self;
    fn rotate_to(self, rad: f64) -> Self;
    fn rotate_by(self, rad: f64) -> Self;
}

pub trait WithPosition {
    fn position(&self) -> Point;
    fn at(self, x: f64, y: f64) -> Self;
    fn move_to(self, position: Point) -> Self;
    fn move_by(self, vector: Vector) -> Self;
}

pub trait Shape {
    fn new() -> Self;
    fn bounding_box(&self) -> PlotPoints<'static>;
    fn local2global(&self, x: f64, y: f64) -> Point;
    fn upper_left(&self) -> Point;
    fn upper_right(&self) -> Point;
    fn lower_left(&self) -> Point;
    fn lower_right(&self) -> Point;
}

impl<T> Shape for T
where
    T: WithAngle + WithSize + WithPosition + Default,
{
    fn new() -> Self {
        Self::default()
    }
    fn bounding_box(&self) -> PlotPoints<'static> {
        PlotPoints::new(vec![
            [self.upper_left().x, self.upper_left().y],
            [self.upper_right().x, self.upper_right().y],
            [self.lower_right().x, self.lower_right().y],
            [self.lower_left().x, self.lower_left().y],
        ])
    }
    fn local2global(&self, x: f64, y: f64) -> Point {
        Point::new(x, y)
            .rotate_by(self.angle())
            .move_by(self.position())
    }
    fn upper_left(&self) -> Point {
        self.local2global(-self.width() / 2.0, self.height() / 2.0)
    }
    fn upper_right(&self) -> Point {
        self.local2global(self.width() / 2.0, self.height() / 2.0)
    }
    fn lower_left(&self) -> Point {
        self.local2global(-self.width() / 2.0, -self.height() / 2.0)
    }
    fn lower_right(&self) -> Point {
        self.local2global(self.width() / 2.0, -self.height() / 2.0)
    }
}

#[macro_export]
macro_rules! impl_position {
    ($name:ident) => {
        impl WithPosition for $name {
            fn position(&self) -> Point {
                self.position
            }
            fn at(mut self, x: f64, y: f64) -> Self {
                self.position = Point::new(x, y);
                self
            }
            fn move_to(mut self, position: Point) -> Self {
                self.position = position;
                self
            }
            fn move_by(mut self, vector: crate::item::Vector) -> Self {
                self.position = Point::new(self.position.x + vector.x, self.position.y + vector.y);
                self
            }
        }
    };
}

#[macro_export]
macro_rules! impl_angle {
    ($name:ident) => {
        impl WithAngle for $name {
            fn angle(&self) -> f64 {
                self.angle
            }
            fn with_angle(self, rad: f64) -> Self {
                self.rotate_to(rad)
            }
            fn rotate_to(mut self, rad: f64) -> Self {
                self.angle = rad;
                self
            }
            fn rotate_by(mut self, rad: f64) -> Self {
                self.angle += rad;
                self
            }
        }
    };
}

#[macro_export]
macro_rules! impl_size {
    ($name:ident) => {
        impl WithSize for $name {
            fn with_width(mut self, width: f64) -> Self {
                self.size.width = width;
                self
            }
            fn with_height(mut self, height: f64) -> Self {
                self.size.height = height;
                self
            }
            fn with_size(mut self, size: Size) -> Self {
                self.size = size;
                self
            }
            fn scale(mut self, by: f64) -> Self {
                self.size.width *= by;
                self.size.height *= by;
                self
            }
            fn width(&self) -> f64 {
                self.size.width
            }
            fn height(&self) -> f64 {
                self.size.height
            }
        }
    };
}
