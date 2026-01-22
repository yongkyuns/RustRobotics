use super::*;

use egui_plot::PlotPoints;
use rust_robotics_algo as rb;

/// Allows converting from a column in [`TimeTable`] into [`egui`] [`PlotPoints`].
///
/// This is convenience trait for plotting on [`egui`] [`Plot`](egui_plot::Plot)
pub trait IntoValues {
    fn values(&self, column: usize) -> Option<PlotPoints<'static>>;
    fn values_shifted(&self, column: usize, x: f32, y: f32) -> Option<PlotPoints<'static>>;
}

impl IntoValues for TimeTable<f32> {
    fn values(&self, column: usize) -> Option<PlotPoints<'static>> {
        self.values_shifted(column, 0.0, 0.0)
    }
    fn values_shifted(&self, column: usize, x: f32, y: f32) -> Option<PlotPoints<'static>> {
        self.zipped_iter(column).map(|zip| {
            PlotPoints::new(
                zip.into_iter()
                    .map(|(t, v)| [(*t + x) as f64, (*v + y) as f64])
                    .collect(),
            )
        })
    }
}

/// Allows extracting information for [`plot`](egui_plot::Plot) from array of
/// 4-element vectors. The following assignments are assumed:
///
/// - [0] = x position
/// - [1] = y position
pub trait VehiclePlot {
    fn positions(&self) -> PlotPoints<'static>;
}

impl VehiclePlot for Vec<rb::Vector4> {
    fn positions(&self) -> PlotPoints<'static> {
        PlotPoints::new(
            self.into_iter()
                .map(|&state| [
                    *state.get(0).unwrap() as f64,
                    *state.get(1).unwrap() as f64,
                ])
                .collect(),
        )
    }
}
