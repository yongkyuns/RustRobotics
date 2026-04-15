use rust_robotics_core::{
    LinearSnapshot, PolicySnapshot, PpoMetrics, PpoSharedState, ValueSnapshot,
};
use rust_robotics_train::PpoTrainerConfig;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
use native::NativePpoReplicaExecutor as PlatformPpoReplicaExecutor;
#[cfg(target_arch = "wasm32")]
use web::WebPpoReplicaExecutor as PlatformPpoReplicaExecutor;

#[derive(Debug, Clone, Copy, Default)]
pub struct PpoReplicaStatus {
    pub total: usize,
    pub ready: usize,
    pub busy: usize,
}

#[derive(Default)]
pub struct PpoTrainerCoordinator {
    executors: Vec<PlatformPpoReplicaExecutor>,
    snapshot: Option<PolicySnapshot>,
    metrics: Option<PpoMetrics>,
    last_error: Option<String>,
    busy: bool,
    status: PpoReplicaStatus,
}

impl PpoTrainerCoordinator {
    pub fn reset(&mut self, config: &PpoTrainerConfig, replicas: usize) {
        let replicas = replicas.max(1);
        self.destroy();
        self.executors = (0..replicas)
            .map(|_| PlatformPpoReplicaExecutor::new(config.clone()))
            .collect();
        self.last_error = None;
        self.refresh_summary();
    }

    pub fn destroy(&mut self) {
        for executor in &mut self.executors {
            executor.destroy();
        }
        self.executors.clear();
        self.snapshot = None;
        self.metrics = None;
        self.last_error = None;
        self.busy = false;
        self.status = PpoReplicaStatus::default();
    }

    pub fn is_initialized(&self) -> bool {
        self.status.total > 0
    }

    pub fn tick(&mut self, updates: usize) {
        let updates = updates.max(1);

        for executor in &mut self.executors {
            executor.tick(updates);
        }

        if let Some(shared_state) = average_shared_states(
            &self
                .executors
                .iter()
                .filter_map(PlatformPpoReplicaExecutor::shared_state)
                .collect::<Vec<_>>(),
        ) {
            for executor in &mut self.executors {
                if executor.accepts_shared_state() {
                    executor.load_shared_state(&shared_state);
                }
            }
        }

        self.refresh_summary();
    }

    pub fn refresh(&mut self) {
        for executor in &mut self.executors {
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
        let ready = self
            .executors
            .iter()
            .filter(|executor| executor.ready())
            .count();
        let busy = self
            .executors
            .iter()
            .filter(|executor| executor.busy())
            .count();
        self.status = PpoReplicaStatus {
            total: self.executors.len(),
            ready,
            busy,
        };
        self.busy = busy > 0;
        self.last_error = self
            .executors
            .iter()
            .find_map(|executor| executor.last_error().map(str::to_owned));
        self.snapshot = self
            .executors
            .first()
            .and_then(PlatformPpoReplicaExecutor::shared_state)
            .map(|state| state.policy);
        self.metrics = aggregate_metrics(&self.executors);
    }
}

impl Drop for PpoTrainerCoordinator {
    fn drop(&mut self) {
        self.destroy();
    }
}

fn aggregate_metrics(executors: &[PlatformPpoReplicaExecutor]) -> Option<PpoMetrics> {
    let mut counted = 0usize;
    let mut metrics = PpoMetrics::default();

    for executor in executors {
        let Some(replica) = executor.metrics() else {
            continue;
        };
        counted += 1;
        metrics.total_updates = metrics.total_updates.max(replica.total_updates);
        metrics.total_env_steps += replica.total_env_steps;
        metrics.total_episodes += replica.total_episodes;
        metrics.best_episode_return = metrics.best_episode_return.max(replica.best_episode_return);
        metrics.last_episode_return += replica.last_episode_return;
        metrics.mean_episode_return += replica.mean_episode_return;
        metrics.last_policy_loss += replica.last_policy_loss;
        metrics.last_value_loss += replica.last_value_loss;
        metrics.last_mean_advantage += replica.last_mean_advantage;
    }

    if counted == 0 {
        return None;
    }

    let count = counted as f32;
    metrics.last_episode_return /= count;
    metrics.mean_episode_return /= count;
    metrics.last_policy_loss /= count;
    metrics.last_value_loss /= count;
    metrics.last_mean_advantage /= count;
    Some(metrics)
}

fn average_shared_states(states: &[PpoSharedState]) -> Option<PpoSharedState> {
    let first = states.first()?;
    let policy = PolicySnapshot {
        input: average_linear_snapshots(states.iter().map(|state| &state.policy.input).collect())?,
        hidden: average_linear_snapshots(
            states.iter().map(|state| &state.policy.hidden).collect(),
        )?,
        output: average_linear_snapshots(
            states.iter().map(|state| &state.policy.output).collect(),
        )?,
        action_limit: first.policy.action_limit,
        action_std: first.policy.action_std,
    };
    let value = ValueSnapshot {
        input: average_linear_snapshots(states.iter().map(|state| &state.value.input).collect())?,
        hidden: average_linear_snapshots(states.iter().map(|state| &state.value.hidden).collect())?,
        output: average_linear_snapshots(states.iter().map(|state| &state.value.output).collect())?,
    };
    Some(PpoSharedState { policy, value })
}

fn average_linear_snapshots(snapshots: Vec<&LinearSnapshot>) -> Option<LinearSnapshot> {
    let first = snapshots.first()?;
    let mut weight = vec![0.0; first.weight.len()];
    let mut bias = vec![0.0; first.bias.len()];
    let count = snapshots.len() as f32;

    for snapshot in &snapshots {
        if snapshot.in_dim != first.in_dim
            || snapshot.out_dim != first.out_dim
            || snapshot.weight.len() != first.weight.len()
            || snapshot.bias.len() != first.bias.len()
        {
            return None;
        }
        for (dst, src) in weight.iter_mut().zip(snapshot.weight.iter()) {
            *dst += *src;
        }
        for (dst, src) in bias.iter_mut().zip(snapshot.bias.iter()) {
            *dst += *src;
        }
    }

    for value in &mut weight {
        *value /= count;
    }
    for value in &mut bias {
        *value /= count;
    }

    Some(LinearSnapshot {
        in_dim: first.in_dim,
        out_dim: first.out_dim,
        weight,
        bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_snapshot(
        in_dim: usize,
        out_dim: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> LinearSnapshot {
        LinearSnapshot {
            in_dim,
            out_dim,
            weight: weight.to_vec(),
            bias: bias.to_vec(),
        }
    }

    fn shared_state(scale: f32) -> PpoSharedState {
        PpoSharedState {
            policy: PolicySnapshot {
                input: linear_snapshot(
                    4,
                    2,
                    &[
                        scale,
                        2.0 * scale,
                        3.0 * scale,
                        4.0 * scale,
                        5.0 * scale,
                        6.0 * scale,
                        7.0 * scale,
                        8.0 * scale,
                    ],
                    &[0.5 * scale, 1.5 * scale],
                ),
                hidden: linear_snapshot(
                    2,
                    2,
                    &[1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale],
                    &[0.25 * scale, 0.75 * scale],
                ),
                output: linear_snapshot(2, 1, &[2.0 * scale, 4.0 * scale], &[1.25 * scale]),
                action_limit: 5.0 + scale,
                action_std: 0.2 + scale,
            },
            value: ValueSnapshot {
                input: linear_snapshot(
                    4,
                    2,
                    &[
                        1.0 * scale,
                        3.0 * scale,
                        5.0 * scale,
                        7.0 * scale,
                        9.0 * scale,
                        11.0 * scale,
                        13.0 * scale,
                        15.0 * scale,
                    ],
                    &[2.0 * scale, 4.0 * scale],
                ),
                hidden: linear_snapshot(
                    2,
                    2,
                    &[0.5 * scale, 1.5 * scale, 2.5 * scale, 3.5 * scale],
                    &[4.0 * scale, 6.0 * scale],
                ),
                output: linear_snapshot(2, 1, &[8.0 * scale, 10.0 * scale], &[12.0 * scale]),
            },
        }
    }

    #[test]
    fn average_linear_snapshots_returns_none_on_shape_mismatch() {
        let first = linear_snapshot(2, 2, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0]);
        let mismatched = linear_snapshot(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[7.0, 8.0]);

        let averaged = average_linear_snapshots(vec![&first, &mismatched]);

        assert!(averaged.is_none());
    }

    #[test]
    fn average_shared_states_averages_weights_and_preserves_policy_metadata() {
        let first = shared_state(1.0);
        let second = shared_state(3.0);

        let averaged = average_shared_states(&[first.clone(), second]).expect("average state");

        assert_eq!(
            averaged.policy.input,
            linear_snapshot(
                4,
                2,
                &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                &[1.0, 3.0],
            )
        );
        assert_eq!(
            averaged.policy.hidden,
            linear_snapshot(2, 2, &[2.0, 4.0, 6.0, 8.0], &[0.5, 1.5])
        );
        assert_eq!(
            averaged.policy.output,
            linear_snapshot(2, 1, &[4.0, 8.0], &[2.5])
        );
        assert_eq!(
            averaged.value.output,
            linear_snapshot(2, 1, &[16.0, 20.0], &[24.0])
        );
        assert_eq!(averaged.policy.action_limit, first.policy.action_limit);
        assert_eq!(averaged.policy.action_std, first.policy.action_std);
    }
}
