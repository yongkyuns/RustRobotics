//! Permanent regressions for real coordinator/session optimizer continuity.
use super::*;
use rust_robotics_train::{PpoConfig, PpoTrainerSession};

fn seeded_pool(config: PpoTrainerConfig, seed: u64, count: usize) -> PpoTrainerCoordinator {
    let mut coordinator = PpoTrainerCoordinator {
        executor: Some(PlatformPpoReplicaExecutor::new_seeded(config, seed, count)),
        environment_count: count,
        snapshot: None,
        metrics: None,
        last_error: None,
        busy: false,
        status: PpoReplicaStatus::default(),
    };
    coordinator.refresh_summary();
    coordinator
}

fn seeded(config: PpoTrainerConfig, seed: u64) -> PpoTrainerCoordinator {
    seeded_pool(config, seed, 1)
}

fn assert_same(coordinator: &PpoTrainerCoordinator, session: &PpoTrainerSession) {
    let actual = coordinator
        .executor
        .as_ref()
        .unwrap()
        .shared_state()
        .unwrap();
    let expected = session.shared_state();
    assert_eq!(
        actual.policy, expected.policy,
        "actor/Adam history must survive ticks"
    );
    assert_eq!(
        actual.value, expected.value,
        "critic/Adam history must survive ticks"
    );
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

#[test]
fn multiple_environments_use_one_persistent_pooled_learner() {
    let config = PpoTrainerConfig {
        hidden_dim: 8,
        ppo: PpoConfig {
            rollout_steps: 16,
            mini_batch_size: 16,
            ..PpoConfig::default()
        },
        ..PpoTrainerConfig::default()
    };
    for environments in [2, 8] {
        let mut coordinator = seeded_pool(config.clone(), 201, environments);
        let mut direct =
            PpoTrainerSession::new_seeded_with_environments(config.clone(), 201, environments);
        for _ in 0..3 {
            coordinator.tick(1);
            coordinator.refresh();
            direct.train_updates(1);
            let actual = coordinator
                .executor
                .as_ref()
                .unwrap()
                .shared_state()
                .unwrap();
            let expected = direct.shared_state();
            assert_eq!(actual.policy, expected.policy);
            assert_eq!(actual.value, expected.value);
            assert_eq!(coordinator.metrics(), Some(direct.metrics()));
            assert_eq!(coordinator.status().total, environments);
            assert_eq!(coordinator.status().ready, environments);
            assert_eq!(coordinator.status().busy, 0);
        }
        assert_eq!(direct.metrics().total_env_steps, 3 * environments * 16);
        assert_eq!(direct.metrics().total_updates, 3);
    }
}
