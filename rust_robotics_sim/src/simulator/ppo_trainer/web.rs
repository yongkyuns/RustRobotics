use rust_robotics_train::{PolicySnapshot, PpoMetrics, PpoSharedState, PpoTrainerConfig};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Default)]
pub struct WebPpoReplicaExecutor {
    handle: Option<u32>,
    snapshot: Option<PolicySnapshot>,
    shared_state: Option<PpoSharedState>,
    metrics: Option<PpoMetrics>,
    last_error: Option<String>,
    busy: bool,
    ready: bool,
}

#[derive(Deserialize)]
struct PollState {
    ready: bool,
    busy: bool,
    snapshot: Option<PolicySnapshot>,
    shared_state: Option<PpoSharedState>,
    metrics: Option<PpoMetrics>,
    error: Option<String>,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerCreate)]
    fn js_ppo_trainer_create(config: JsValue) -> u32;

    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerRecreate)]
    fn js_ppo_trainer_recreate(handle: u32, config: JsValue);

    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerRequestTrain)]
    fn js_ppo_trainer_request_train(handle: u32, updates: usize) -> bool;

    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerPoll)]
    fn js_ppo_trainer_poll(handle: u32) -> JsValue;

    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerDestroy)]
    fn js_ppo_trainer_destroy(handle: u32);

    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerLoadSharedState)]
    fn js_ppo_trainer_load_shared_state(handle: u32, state: JsValue);
}

impl WebPpoReplicaExecutor {
    pub fn new(config: PpoTrainerConfig) -> Self {
        let mut executor = Self::default();
        executor.reset(config);
        executor
    }

    fn reset(&mut self, config: PpoTrainerConfig) {
        let value = match serde_wasm_bindgen::to_value(&config) {
            Ok(value) => value,
            Err(err) => {
                self.last_error = Some(err.to_string());
                self.ready = false;
                self.busy = false;
                return;
            }
        };

        self.snapshot = None;
        self.shared_state = None;
        self.metrics = None;
        self.last_error = None;
        self.ready = false;
        self.busy = true;

        if let Some(handle) = self.handle {
            js_ppo_trainer_recreate(handle, value);
        } else {
            self.handle = Some(js_ppo_trainer_create(value));
        }
    }

    pub fn destroy(&mut self) {
        if let Some(handle) = self.handle.take() {
            js_ppo_trainer_destroy(handle);
        }
        self.snapshot = None;
        self.shared_state = None;
        self.metrics = None;
        self.last_error = None;
        self.busy = false;
        self.ready = false;
    }

    pub fn tick(&mut self, updates: usize) {
        self.poll();
        if self.ready && !self.busy {
            self.request_train(updates.max(1));
        }
        self.poll();
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn busy(&self) -> bool {
        self.busy
    }

    pub fn accepts_shared_state(&self) -> bool {
        self.ready && !self.busy
    }

    pub fn metrics(&self) -> Option<&PpoMetrics> {
        self.metrics.as_ref()
    }

    pub fn shared_state(&self) -> Option<PpoSharedState> {
        self.shared_state.clone()
    }

    pub fn load_shared_state(&mut self, state: &PpoSharedState) {
        let Some(handle) = self.handle else {
            return;
        };
        if let Ok(value) = serde_wasm_bindgen::to_value(state) {
            js_ppo_trainer_load_shared_state(handle, value);
            self.shared_state = Some(state.clone());
            self.snapshot = Some(state.policy.clone());
        }
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    fn request_train(&mut self, updates: usize) {
        let Some(handle) = self.handle else {
            return;
        };
        if js_ppo_trainer_request_train(handle, updates.max(1)) {
            self.busy = true;
        }
    }

    pub fn poll(&mut self) {
        let Some(handle) = self.handle else {
            return;
        };
        let value = js_ppo_trainer_poll(handle);
        if value.is_null() || value.is_undefined() {
            return;
        }

        match serde_wasm_bindgen::from_value::<PollState>(value) {
            Ok(state) => {
                self.ready = state.ready;
                self.busy = state.busy;
                self.snapshot = state.snapshot;
                self.shared_state = state.shared_state;
                self.metrics = state.metrics;
                self.last_error = state.error;
            }
            Err(err) => {
                self.last_error = Some(err.to_string());
                self.busy = false;
                self.ready = false;
                self.shared_state = None;
            }
        }
    }
}
