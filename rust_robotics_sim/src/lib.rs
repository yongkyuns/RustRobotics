#![warn(clippy::all, rust_2018_idioms)]

pub mod app;
pub mod data;
pub mod item;
pub mod math;
pub mod simulator;
pub mod time;

pub use app::App;

pub mod prelude {
    pub use crate::item::{
        draw_cart, Circle, Ellipse, Rectangle, Shape, WithAngle, WithPosition, WithSize,
    };
    pub use crate::math::{cos, sin};
    pub use crate::time::Timer;
}

// ----------------------------------------------------------------------------
// When compiling for web:

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let document = web_sys::window()
        .ok_or("No window")?
        .document()
        .ok_or("No document")?;

    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("No canvas element found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| "Element is not a canvas")?;

    let web_options = eframe::WebOptions::default();

    let runner = eframe::WebRunner::new();

    runner
        .start(
            canvas,
            web_options,
            Box::new(|cc| Ok(Box::new(App::new(cc)))),
        )
        .await?;

    Ok(())
}
