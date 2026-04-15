#[cfg(target_arch = "wasm32")]
use rust_robotics_train::{
    PolicySnapshot, PpoMetrics, PpoSharedState, PpoTrainerConfig, PpoTrainerSession,
};
#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static NEXT_SESSION_ID: Cell<u32> = const { Cell::new(1) };
    static TRAINER_SESSIONS: RefCell<HashMap<u32, PpoTrainerSession>> = RefCell::new(HashMap::new());
}

#[cfg(target_arch = "wasm32")]
#[derive(Serialize)]
struct WorkerTrainerSnapshot {
    metrics: PpoMetrics,
    snapshot: PolicySnapshot,
    shared_state: PpoSharedState,
}

#[cfg(target_arch = "wasm32")]
#[derive(Serialize)]
struct WorkerTrainerCreated {
    session_id: u32,
    metrics: PpoMetrics,
    snapshot: PolicySnapshot,
    shared_state: PpoSharedState,
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
struct WorkerSharedStateInput {
    shared_state: PpoSharedState,
}

#[cfg(target_arch = "wasm32")]
fn next_session_id() -> u32 {
    NEXT_SESSION_ID.with(|next| {
        let id = next.get();
        next.set(id.saturating_add(1));
        id
    })
}

#[cfg(target_arch = "wasm32")]
fn js_err(message: impl Into<String>) -> JsValue {
    js_sys::Error::new(&message.into()).into()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_ppo_worker_create_trainer(config: JsValue) -> Result<JsValue, JsValue> {
    let config: PpoTrainerConfig =
        serde_wasm_bindgen::from_value(config).map_err(|err| js_err(err.to_string()))?;
    let session = PpoTrainerSession::new(config);
    let response = WorkerTrainerCreated {
        session_id: next_session_id(),
        metrics: session.metrics().clone(),
        snapshot: session.snapshot(),
        shared_state: session.shared_state(),
    };

    TRAINER_SESSIONS.with(|sessions| {
        sessions.borrow_mut().insert(response.session_id, session);
    });

    serde_wasm_bindgen::to_value(&response).map_err(|err| js_err(err.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_ppo_worker_train(session_id: u32, updates: usize) -> Result<JsValue, JsValue> {
    TRAINER_SESSIONS.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        let session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| js_err(format!("trainer session {session_id} is missing")))?;
        session.train_updates(updates.max(1));
        let response = WorkerTrainerSnapshot {
            metrics: session.metrics().clone(),
            snapshot: session.snapshot(),
            shared_state: session.shared_state(),
        };
        serde_wasm_bindgen::to_value(&response).map_err(|err| js_err(err.to_string()))
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_ppo_worker_load_shared_state(
    session_id: u32,
    state: JsValue,
) -> Result<(), JsValue> {
    let state: WorkerSharedStateInput =
        serde_wasm_bindgen::from_value(state).map_err(|err| js_err(err.to_string()))?;
    TRAINER_SESSIONS.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        let session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| js_err(format!("trainer session {session_id} is missing")))?;
        session.load_shared_state(&state.shared_state);
        Ok(())
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rust_robotics_ppo_worker_destroy_trainer(session_id: u32) {
    TRAINER_SESSIONS.with(|sessions| {
        sessions.borrow_mut().remove(&session_id);
    });
}
