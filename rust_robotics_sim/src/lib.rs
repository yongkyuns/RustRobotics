#![warn(clippy::all, rust_2018_idioms)]

pub mod app;
pub mod data;
pub mod item;
pub mod math;
pub mod simulator;
pub mod theme;
pub mod time;
#[cfg(target_arch = "wasm32")]
mod web_ppo_worker;

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

#[cfg(target_arch = "wasm32")]
fn js_err(message: impl Into<String>) -> JsValue {
    js_sys::Error::new(&message.into()).into()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_get_state() -> Result<JsValue, JsValue> {
    let state = app::web_test_state().ok_or_else(|| js_err("web test state is not initialized"))?;
    serde_wasm_bindgen::to_value(&state).map_err(|err| js_err(err.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_embed_get_state() -> Result<JsValue, JsValue> {
    let state =
        app::web_embed_state().ok_or_else(|| js_err("web embed state is not initialized"))?;
    serde_wasm_bindgen::to_value(&state).map_err(|err| js_err(err.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_mode(mode: &str) -> Result<(), JsValue> {
    let mode = simulator::SimMode::from_test_id(mode)
        .ok_or_else(|| js_err(format!("unsupported simulator mode: {mode}")))?;
    app::web_test_set_mode(mode);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_paused(paused: bool) {
    app::web_test_set_paused(paused);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_restart() {
    app::web_test_restart();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_sim_speed(sim_speed: usize) {
    app::web_test_set_sim_speed(sim_speed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_pendulum_noise_enabled(enabled: bool) {
    app::web_test_set_pendulum_noise_enabled(enabled);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_pendulum_noise_scale(scale: f32) {
    app::web_test_set_pendulum_noise_scale(scale);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_add_pendulum() {
    app::web_test_add_pendulum();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_pendulum_controller(
    pendulum_id: usize,
    controller: &str,
) -> Result<(), JsValue> {
    let kind = match controller {
        "lqr" => simulator::pendulum::ControllerKind::Lqr,
        "pid" => simulator::pendulum::ControllerKind::Pid,
        "mpc" => simulator::pendulum::ControllerKind::Mpc,
        "policy" => simulator::pendulum::ControllerKind::Policy,
        _ => {
            return Err(js_err(format!(
                "unsupported pendulum controller: {controller}"
            )))
        }
    };
    app::web_test_set_pendulum_controller(pendulum_id, kind);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_remove_pendulum(pendulum_id: usize) {
    app::web_test_remove_pendulum(pendulum_id);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_patch_pendulum(
    pendulum_id: usize,
    patch_json: &str,
) -> Result<(), JsValue> {
    let patch: simulator::PendulumPatch =
        serde_json::from_str(patch_json).map_err(|err| js_err(err.to_string()))?;
    app::web_test_patch_pendulum(pendulum_id, patch);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_set_embed_mode(mode: &str) -> Result<(), JsValue> {
    let mode = app::WebEmbedMode::from_query_value(mode)
        .ok_or_else(|| js_err(format!("unsupported embed mode: {mode}")))?;
    app::web_test_set_embed_mode(mode);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_show_graph(show_graph: bool) {
    app::web_test_set_show_graph(show_graph);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_test_set_theme(theme: &str) -> Result<(), JsValue> {
    let theme = app::WebThemeMode::from_str(theme)
        .ok_or_else(|| js_err(format!("unsupported web theme: {theme}")))?;
    app::web_test_set_theme_mode(theme);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_embed_dispatch_action(action_json: &str) -> Result<(), JsValue> {
    let action: app::WebEmbedAction =
        serde_json::from_str(action_json).map_err(|err| js_err(err.to_string()))?;
    app::web_embed_dispatch(action);
    Ok(())
}
