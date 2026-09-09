//! One persistent PPO learner for the native app or one browser worker.
//!
//! The selected environment count controls rollout coverage, not a collection
//! of independently initialized/optimized agents. No parameter averaging or
//! automatic external weight transfer occurs on ticks or readouts.
use rust_robotics_core::{PolicySnapshot, PpoMetrics};
use rust_robotics_train::PpoTrainerConfig;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(not(target_arch = "wasm32"))]
use native::NativePpoReplicaExecutor as PlatformPpoReplicaExecutor;
#[cfg(target_arch = "wasm32")]
use web::WebPpoReplicaExecutor as PlatformPpoReplicaExecutor;

/// Legacy type name retained for the UI readout contract. Counts now describe
/// environments served by the one learner, NOT independent policy replicas.
#[derive(Debug, Clone, Copy, Default)]
pub struct PpoReplicaStatus {
    pub total: usize,
    pub ready: usize,
    pub busy: usize,
}

#[derive(Default)]
pub struct PpoTrainerCoordinator {
    executor: Option<PlatformPpoReplicaExecutor>,
    environment_count: usize,
    snapshot: Option<PolicySnapshot>,
    metrics: Option<PpoMetrics>,
    last_error: Option<String>,
    busy: bool,
    status: PpoReplicaStatus,
}

impl PpoTrainerCoordinator {
    /// Starts one learner with the requested number of environment streams.
    /// The old zero-count UI convention still means one environment.
    pub fn reset(&mut self, config: &PpoTrainerConfig, environments: usize) {
        self.destroy();
        self.environment_count = environments.max(1);
        self.executor = Some(PlatformPpoReplicaExecutor::new(
            config.clone(),
            self.environment_count,
        ));
        self.refresh_summary();
    }

    pub fn destroy(&mut self) {
        if let Some(mut executor) = self.executor.take() {
            executor.destroy();
        }
        self.environment_count = 0;
        self.snapshot = None;
        self.metrics = None;
        self.last_error = None;
        self.busy = false;
        self.status = PpoReplicaStatus::default();
    }

    pub fn is_initialized(&self) -> bool {
        self.executor.is_some()
    }

    /// Advances the learner without rebuilding networks or optimizer state.
    pub fn tick(&mut self, updates: usize) {
        if let Some(executor) = &mut self.executor {
            executor.tick(updates.max(1));
        }
        self.refresh_summary();
    }

    /// Readout/polling never performs a model synchronization or an update.
    pub fn refresh(&mut self) {
        if let Some(executor) = &mut self.executor {
            executor.poll();
        }
        self.refresh_summary();
    }

    pub fn snapshot(&self) -> Option<&PolicySnapshot> {
        self.snapshot.as_ref()
    }
    pub fn metrics(&self) -> Option<&PpoMetrics> {
        self.metrics.as_ref()
    }
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
    pub fn busy(&self) -> bool {
        self.busy
    }
    pub fn status(&self) -> PpoReplicaStatus {
        self.status
    }

    fn refresh_summary(&mut self) {
        let Some(executor) = &self.executor else {
            return;
        };
        self.busy = executor.busy();
        self.status = PpoReplicaStatus {
            total: self.environment_count,
            ready: if executor.ready() {
                self.environment_count
            } else {
                0
            },
            busy: if self.busy { self.environment_count } else { 0 },
        };
        self.last_error = executor.last_error().map(str::to_owned);
        self.snapshot = executor.shared_state().map(|state| state.policy);
        // In particular, preserve a negative best return, rather than max(0, x).
        self.metrics = executor.metrics().cloned();
    }
}

impl Drop for PpoTrainerCoordinator {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lifecycle_tests;
