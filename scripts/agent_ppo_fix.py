#!/usr/bin/env python3
"""Temporary exact-source patch application; removed before review."""
from pathlib import Path
import hashlib

BASE = Path('rust_robotics_sim/src/simulator/ppo_trainer')

def original(name, expected):
    p = BASE / name
    data = p.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == expected, (name, actual)
    return p, data.decode()

def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

p, s = original('mod.rs', '7ac4aa49ebe15ea5a29cd79fd2865f059ac62f0d')
s = once(s, '        if let Some(shared_state) = average_shared_states(\n', '''        // A readout is not an external weight transfer. Reloading even identical
        // weights reconstructs the networks and resets both Adam optimizers.
        if self.executors.len() == 1 {
            self.refresh_summary();
            return;
        }

        if let Some(shared_state) = average_shared_states(
''')
s = once(s, '        self.metrics = aggregate_metrics(&self.executors);', '''        self.metrics = if self.executors.len() == 1 {
            self.executors[0].metrics().cloned()
        } else {
            aggregate_metrics(&self.executors)
        };''')
s += '\n#[cfg(all(test, not(target_arch = "wasm32")))]\nmod lifecycle_tests;\n'
p.write_text(s)
p, s = original('native.rs', '52cf70d26f95623449c78ff7e612d601bbfb10af')
s += '''
#[cfg(test)]
impl NativePpoReplicaExecutor {
    pub(super) fn new_seeded(config: PpoTrainerConfig, seed: u64) -> Self {
        Self {
            session: PpoTrainerSession::new_seeded(config, seed),
        }
    }
}
'''
p.write_text(s)
(BASE / 'lifecycle_tests.rs').write_text('''//! Permanent regressions for real coordinator/session optimizer continuity.
use super::*;
use rust_robotics_train::PpoTrainerSession;

fn seeded(config: PpoTrainerConfig, seed: u64) -> PpoTrainerCoordinator {
    let mut coordinator = PpoTrainerCoordinator::default();
    coordinator.executors = vec![PlatformPpoReplicaExecutor::new_seeded(config, seed)];
    coordinator.refresh_summary();
    coordinator
}

fn assert_same(coordinator: &PpoTrainerCoordinator, session: &PpoTrainerSession) {
    let actual = coordinator.executors[0].shared_state().unwrap();
    let expected = session.shared_state();
    assert_eq!(actual.policy, expected.policy, "actor/Adam history must survive ticks");
    assert_eq!(actual.value, expected.value, "critic/Adam history must survive ticks");
    assert_eq!(coordinator.snapshot(), Some(&expected.policy));
    assert_eq!(coordinator.metrics(), Some(session.metrics()));
    let status = coordinator.status();
    assert_eq!((status.total, status.ready, status.busy), (1, 1, 0));
    assert!(!coordinator.busy());
    assert!(coordinator.last_error().is_none());
}

#[test]
fn single_environment_ticks_preserve_persistent_optimizer_history() {
    for seed in [0, 201, 301] {
        let config = PpoTrainerConfig::default();
        let mut coordinator = seeded(config.clone(), seed);
        let mut direct = PpoTrainerSession::new_seeded(config, seed);
        assert_same(&coordinator, &direct);
        for _ in 0..4 {
            coordinator.tick(1);
            direct.train_updates(1);
            assert_same(&coordinator, &direct);
        }
    }
}

#[test]
fn grouped_split_ticks_and_readouts_have_identical_training_state() {
    let config = PpoTrainerConfig::default();
    let mut grouped = seeded(config.clone(), 201);
    let mut split = seeded(config.clone(), 201);
    let mut direct = PpoTrainerSession::new_seeded(config, 201);
    grouped.tick(4);
    for _ in 0..4 {
        split.refresh();
        let _ = split.snapshot();
        let _ = split.metrics();
        split.tick(1);
        split.refresh();
    }
    direct.train_updates(4);
    assert_same(&grouped, &direct);
    assert_same(&split, &direct);
}

#[test]
fn single_environment_metrics_retain_negative_best_return() {
    let mut config = PpoTrainerConfig::default();
    config.env.max_angle_rad = 0.0;
    config.ppo.rollout_steps = 8;
    config.ppo.mini_batch_size = 4;
    let mut coordinator = seeded(config.clone(), 201);
    let mut direct = PpoTrainerSession::new_seeded(config, 201);
    coordinator.tick(1);
    direct.train_updates(1);
    assert_eq!(direct.metrics().total_episodes, 8);
    assert_eq!(direct.metrics().best_episode_return, -10.0);
    assert_same(&coordinator, &direct);
}

#[test]
fn reset_and_destroy_preserve_readiness_and_clear_cached_results() {
    let config = PpoTrainerConfig::default();
    let mut coordinator = seeded(config.clone(), 201);
    coordinator.tick(1);
    coordinator.destroy();
    assert!(!coordinator.is_initialized());
    assert!(coordinator.snapshot().is_none());
    assert!(coordinator.metrics().is_none());
    assert!(coordinator.last_error().is_none());
    assert!(!coordinator.busy());
    let status = coordinator.status();
    assert_eq!((status.total, status.ready, status.busy), (0, 0, 0));
    // The existing UI contract treats zero as a request for one environment.
    coordinator.reset(&config, 0);
    assert!(coordinator.is_initialized());
    assert_eq!(coordinator.metrics().unwrap().total_updates, 0);
    assert_eq!(coordinator.status().total, 1);
    assert_eq!(coordinator.status().ready, 1);
}
''')
print('Applied lifecycle fix and four permanent native coordinator tests.')
