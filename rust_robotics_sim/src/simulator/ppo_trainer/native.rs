use rust_robotics_train::{PpoMetrics, PpoSharedState, PpoTrainerConfig, PpoTrainerSession};

pub struct NativePpoReplicaExecutor {
    session: PpoTrainerSession,
}

impl NativePpoReplicaExecutor {
    pub fn new(config: PpoTrainerConfig) -> Self {
        Self {
            session: PpoTrainerSession::new(config),
        }
    }

    pub fn destroy(&mut self) {}

    pub fn tick(&mut self, updates: usize) {
        self.session.train_updates(updates.max(1));
    }

    pub fn poll(&mut self) {}

    pub fn ready(&self) -> bool {
        true
    }

    pub fn busy(&self) -> bool {
        false
    }

    pub fn accepts_shared_state(&self) -> bool {
        true
    }

    pub fn metrics(&self) -> Option<&PpoMetrics> {
        Some(self.session.metrics())
    }

    pub fn shared_state(&self) -> Option<PpoSharedState> {
        Some(self.session.shared_state())
    }

    pub fn load_shared_state(&mut self, state: &PpoSharedState) {
        self.session.load_shared_state(state);
    }

    pub fn last_error(&self) -> Option<&str> {
        None
    }
}
