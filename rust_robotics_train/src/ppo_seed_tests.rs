//! Exact replay is checked within one numerical build, not across platforms.
//! These fixed seeds are not selected for learning quality. Noise, disturbances,
//! reset boundaries, entropy sampling and real Adam updates remain enabled.
use super::*;
use crate::{LinearSnapshot, ValueSnapshot};
use rust_robotics_algo::{control::StateSpace, prelude::Vector4};

const SEED: u64 = 0x5050_0031;

fn config() -> PpoTrainerConfig {
    PpoTrainerConfig {
        hidden_dim: 8,
        env: PendulumEnvConfig {
            max_steps: 7,
            disturbance_probability_per_step: 0.35,
            ..PendulumEnvConfig::default()
        },
        ppo: PpoConfig {
            rollout_steps: 24,
            mini_batch_size: 8,
            epochs_per_update: 2,
            entropy_coef: 0.01,
            ..PpoConfig::default()
        },
        ..PpoTrainerConfig::default()
    }
}

fn cursor(rng: &StdRng) -> [u64; 4] {
    let mut copy = rng.clone();
    std::array::from_fn(|_| copy.gen())
}

#[derive(Debug, PartialEq)]
struct NumericalState {
    policy: PolicySnapshot,
    value: ValueSnapshot,
    metrics: PpoMetrics,
    observation: [f32; 4],
    state: [f32; 4],
    partial_return: f32,
    recent_returns: Vec<f32>,
    environment_rng: [u64; 4],
    action_rng: [u64; 4],
    update_rng: [u64; 4],
}

fn state(s: &PpoTrainerSession) -> NumericalState {
    let x = s.env.state();
    NumericalState {
        policy: s.snapshot(),
        value: s.shared_state().value,
        metrics: s.metrics().clone(),
        observation: s.current_observation,
        state: [x[0], x[1], x[2], x[3]],
        partial_return: s.episode_return,
        recent_returns: s.recent_episode_returns.clone(),
        environment_rng: cursor(&s.environment_rng),
        action_rng: cursor(&s.action_rng),
        update_rng: cursor(&s.update_rng),
    }
}

fn same_rollout(a: &RolloutBatch, b: &RolloutBatch) {
    assert_eq!(a.observations, b.observations, "seeded rollout observations");
    assert_eq!(a.latent_actions, b.latent_actions, "seeded rollout actions");
    assert_eq!(a.old_log_probs, b.old_log_probs, "seeded rollout likelihoods");
    assert_eq!(a.returns, b.returns, "seeded rollout targets");
    assert_eq!(a.advantages, b.advantages, "seeded rollout advantages");
}

#[test]
fn equal_seeds_replay_initial_state() {
    for seed in [0, SEED, u64::MAX] {
        let a = PpoTrainerSession::new_seeded(config(), seed);
        // Materialize unrelated backend-random layers between seeded sessions.
        let _unrelated = PolicyNetwork::<AutodiffBackend>::new(&Default::default(), 4, 8, 20.0)
            .snapshot(2.0);
        let b = PpoTrainerSession::new_seeded(config(), seed);
        assert_eq!(state(&a), state(&b), "seeded initialization replay");
    }
}

#[test]
fn different_seeds_change_initial_models_and_environment() {
    let a = PpoTrainerSession::new_seeded(config(), SEED);
    let b = PpoTrainerSession::new_seeded(config(), SEED + 1);
    assert_ne!(a.snapshot(), b.snapshot(), "master seed must affect actor");
    assert_ne!(a.shared_state().value, b.shared_state().value, "master seed must affect critic");
    assert_ne!(a.env.state(), b.env.state(), "master seed must affect environment");
    assert_ne!(cursor(&a.action_rng), cursor(&b.action_rng));
    assert_ne!(cursor(&a.update_rng), cursor(&b.update_rng));
}

#[test]
fn initialization_matches_explicit_uniform_reference() {
    let mut actual_rng = StdRng::seed_from_u64(SEED);
    let mut reference_rng = StdRng::seed_from_u64(SEED);
    let net = PolicyNetwork::<AutodiffBackend> {
        mlp: Mlp::new_with_rng(&Default::default(), 3, 7, 2, &mut actual_rng),
        action_limit: 1.0,
    };
    let snapshot = net.snapshot(1.0);
    for (layer, dimensions) in [&snapshot.input, &snapshot.hidden, &snapshot.output]
        .into_iter().zip([(3, 7), (7, 7), (7, 2)]) {
        assert_eq!((layer.in_dim, layer.out_dim), dimensions, "seeded layer layout");
        // Burn's documented default uniform law, calculated in f64 first.
        let bound = (1.0 / dimensions.0 as f64).sqrt() as f32;
        let weight: Vec<f32> = (0..dimensions.0 * dimensions.1)
            .map(|_| reference_rng.gen_range(-bound..bound)).collect();
        let bias: Vec<f32> = (0..dimensions.1)
            .map(|_| reference_rng.gen_range(-bound..bound)).collect();
        assert_eq!(layer.weight, weight, "seeded layer weight");
        assert_eq!(layer.bias, bias, "seeded layer bias");
        assert!(layer.weight.iter().chain(&layer.bias).all(|x| x.is_finite() && x.abs() <= bound));
    }
    assert_eq!(cursor(&actual_rng), cursor(&reference_rng), "initialization draw count");
}

fn draw(rng: &mut StdRng, magnitude: f32) -> f32 {
    if magnitude == 0.0 { 0.0 } else { rng.gen_range(-magnitude..magnitude) }
}

fn reset_reference(c: PendulumEnvConfig, rng: &mut StdRng) -> [f32; 4] {
    [c.reset_position_range_m, c.reset_velocity_range_mps,
        c.reset_angle_range_rad, c.reset_angular_velocity_range_radps]
        .map(|m| draw(rng, m))
}

fn observe_reference(x: [f32; 4], c: PendulumEnvConfig, rng: &mut StdRng) -> [f32; 4] {
    let magnitudes = [c.observation_position_noise_m, c.observation_velocity_noise_mps,
        c.observation_angle_noise_rad, c.observation_angular_velocity_noise_radps];
    std::array::from_fn(|i| x[i] + draw(rng, magnitudes[i]))
}

#[test]
fn environment_draws_match_independent_reference() {
    let c = config().env;
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut reference = StdRng::seed_from_u64(SEED);
    let mut env = PendulumEnv::new_with_rng(Default::default(), c, &mut rng);
    let mut x = reset_reference(c, &mut reference);
    // new_with_rng performs one reset and samples its (discarded) observation.
    observe_reference(x, c, &mut reference);
    assert_eq!(env.state(), Vector4::from_column_slice(&x), "initial environment RNG");
    let mut disturbances = 0;
    for index in 0..64 {
        assert_eq!(env.observation_with_rng(&mut rng), observe_reference(x, c, &mut reference),
            "observation RNG reference");
        let action = (index % 9) as f32 * 9.0 - 36.0;
        let clipped = action.clamp(-c.max_force, c.max_force);
        let action_noise = draw(&mut reference, c.action_noise_force_n);
        let disturbance = if reference.gen_bool(f64::from(c.disturbance_probability_per_step)) {
            disturbances += 1;
            draw(&mut reference, c.disturbance_force_n)
        } else { 0.0 };
        // Reuse only the unchanged plant matrix, not production noise helpers.
        let (a, b) = env.model().model(c.dt);
        let next = a * Vector4::from_column_slice(&x) + b * (clipped + action_noise + disturbance);
        x = [next[0], next[1], next[2], next[3]];
        let expected_observation = observe_reference(x, c, &mut reference);
        let step = env.step_with_rng(action, &mut rng);
        assert_eq!(env.state(), next, "action and disturbance RNG reference");
        assert_eq!(step.observation, expected_observation, "step observation RNG reference");
        assert_eq!(cursor(&rng), cursor(&reference), "step consumes only caller RNG");
        if step.done {
            x = reset_reference(c, &mut reference);
            assert_eq!(env.reset_with_rng(&mut rng), observe_reference(x, c, &mut reference),
                "reset observation RNG reference");
            assert_eq!(cursor(&rng), cursor(&reference), "reset consumes only caller RNG");
        }
    }
    assert!(disturbances > 0 && disturbances < 64, "fixture must exercise both disturbance branches");
}

#[test]
fn resets_advance_rather_than_repeat_the_initial_seed() {
    let mut a_rng = StdRng::seed_from_u64(SEED);
    let mut b_rng = StdRng::seed_from_u64(SEED);
    let mut a = PendulumEnv::new_with_rng(Default::default(), config().env, &mut a_rng);
    let mut b = PendulumEnv::new_with_rng(Default::default(), config().env, &mut b_rng);
    let initial = a.state();
    for _ in 0..8 {
        assert_eq!(a.reset_with_rng(&mut a_rng), b.reset_with_rng(&mut b_rng), "reset replay");
        assert_eq!(a.state(), b.state());
        assert_ne!(a.state(), initial, "reset must advance its stream");
    }
}

#[test]
fn noisy_rollouts_replay_every_stored_value() {
    for seed in [0, SEED, u64::MAX] {
        let mut a = PpoTrainerSession::new_seeded(config(), seed);
        let mut b = PpoTrainerSession::new_seeded(config(), seed);
        for _ in 0..4 {
            same_rollout(&a.collect_rollout(), &b.collect_rollout());
            assert_eq!(state(&a), state(&b), "seeded collection state");
        }
        assert_eq!(a.metrics.total_env_steps, 96);
        assert!(a.metrics.total_episodes >= 12, "fixture must cross reset boundaries");
    }
}

#[test]
fn grouped_and_split_training_calls_replay_real_updates() {
    for seed in [0, SEED, 91] {
        let mut a = PpoTrainerSession::new_seeded(config(), seed);
        let mut b = PpoTrainerSession::new_seeded(config(), seed);
        let initial = state(&a);
        a.train_updates(3);
        for _ in 0..3 { b.train_updates(1); }
        assert_eq!(state(&a), state(&b), "seeded optimizer replay");
        assert_ne!(a.snapshot(), initial.policy, "fixture must actually train actor");
        assert_ne!(a.shared_state().value, initial.value, "fixture must actually train critic");
        assert_eq!(a.metrics.total_updates, 3);
        assert_eq!(a.metrics.total_env_steps, 72);
    }
}

#[test]
fn interleaved_unrelated_sessions_do_not_change_replay() {
    let mut control = PpoTrainerSession::new_seeded(config(), SEED);
    control.train_updates(3);
    let mut candidate = PpoTrainerSession::new_seeded(config(), SEED);
    for seed in [301, 302, 303] {
        let mut unrelated = PpoTrainerSession::new_seeded(config(), seed);
        unrelated.train_updates(1);
        // Also consume the public entropy-backed environment and backend APIs.
        let mut other_env = PendulumEnv::default();
        other_env.step(3.0);
        let _ = PolicyNetwork::<AutodiffBackend>::new(&Default::default(), 4, 3, 20.0).snapshot(2.0);
        candidate.train_updates(1);
    }
    assert_eq!(state(&control), state(&candidate), "independent interleaved sessions");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn concurrent_sessions_match_sequential_training() {
    let mut sequential = PpoTrainerSession::new_seeded(config(), SEED);
    sequential.train_updates(3);
    let handles: Vec<_> = (0..3).map(|_| std::thread::spawn(|| {
        let mut session = PpoTrainerSession::new_seeded(config(), SEED);
        session.train_updates(3);
        state(&session)
    })).collect();
    for handle in handles {
        assert_eq!(state(&sequential), handle.join().unwrap(), "independent concurrent sessions");
    }
}

#[test]
fn readouts_do_not_consume_training_randomness() {
    let mut a = PpoTrainerSession::new_seeded(config(), SEED);
    let mut b = PpoTrainerSession::new_seeded(config(), SEED);
    for _ in 0..3 {
        for _ in 0..8 {
            let _ = a.config();
            let _ = a.metrics();
            let _ = a.snapshot();
            let _ = a.shared_state();
        }
        a.train_updates(1);
        b.train_updates(1);
        assert_eq!(state(&a), state(&b), "readouts must not change replay");
    }
}

#[test]
fn weight_transfer_preserves_local_random_cursors() {
    let mut a = PpoTrainerSession::new_seeded(config(), SEED);
    let mut b = PpoTrainerSession::new_seeded(config(), SEED);
    a.train_updates(1);
    b.train_updates(1);
    let donor = PpoTrainerSession::new_seeded(config(), SEED + 1).shared_state();
    let before = state(&a);
    a.load_shared_state(&donor);
    b.load_shared_state(&donor);
    let after = state(&a);
    assert_eq!(after.environment_rng, before.environment_rng, "weight transfer must retain environment RNG");
    assert_eq!(after.action_rng, before.action_rng, "weight transfer must retain action RNG");
    assert_eq!(after.update_rng, before.update_rng, "weight transfer must retain optimizer RNG");
    assert_eq!(after.metrics, before.metrics);
    assert_eq!(after.observation, before.observation);
    assert_eq!(after.partial_return, before.partial_return);
    a.train_updates(1);
    b.train_updates(1);
    assert_eq!(state(&a), state(&b), "replay continues after weight transfer");
}

#[test]
fn optimizer_draw_counts_do_not_change_collection_streams() {
    let mut a = PpoTrainerSession::new_seeded(config(), SEED);
    let mut b = PpoTrainerSession::new_seeded(config(), SEED);
    let rollout_a = a.collect_rollout();
    same_rollout(&rollout_a, &b.collect_rollout());
    a.optimize(&rollout_a);
    a.config.ppo.epochs_per_update = 3;
    a.optimize(&rollout_a);
    assert_eq!(cursor(&a.action_rng), cursor(&b.action_rng), "optimizer must not consume action RNG");
    assert_eq!(cursor(&a.environment_rng), cursor(&b.environment_rng), "optimizer must not consume environment RNG");
    // Equalize the models so only accidental RNG coupling can alter collection.
    b.load_shared_state(&a.shared_state());
    same_rollout(&a.collect_rollout(), &b.collect_rollout());
}

#[test]
fn model_size_does_not_consume_live_random_streams() {
    let a = PpoTrainerSession::new_seeded(config(), SEED);
    let mut larger = config();
    larger.hidden_dim = 17;
    let b = PpoTrainerSession::new_seeded(larger, SEED);
    assert_eq!(a.current_observation, b.current_observation, "model size must not change environment");
    assert_eq!(cursor(&a.environment_rng), cursor(&b.environment_rng));
    assert_eq!(cursor(&a.action_rng), cursor(&b.action_rng), "model size must not change action stream");
    assert_eq!(cursor(&a.update_rng), cursor(&b.update_rng));
}

#[test]
fn random_cursors_advance_only_in_their_own_phase() {
    let mut a = PpoTrainerSession::new_seeded(config(), SEED);
    let before = state(&a);
    let batch = a.collect_rollout();
    let collected = state(&a);
    assert_ne!(collected.environment_rng, before.environment_rng, "environment cursor must advance");
    assert_ne!(collected.action_rng, before.action_rng, "action cursor must advance");
    assert_eq!(collected.update_rng, before.update_rng, "collection must not consume optimizer RNG");
    a.optimize(&batch);
    let optimized = state(&a);
    assert_ne!(optimized.update_rng, collected.update_rng, "optimizer cursor must advance");
    assert_eq!(optimized.action_rng, collected.action_rng);
    assert_eq!(optimized.environment_rng, collected.environment_rng);
}

#[test]
fn empty_work_preserves_random_cursors_and_episode() {
    let mut c = config();
    c.ppo.rollout_steps = 0;
    let mut a = PpoTrainerSession::new_seeded(c, SEED);
    let before = state(&a);
    a.train_updates(0);
    let batch = a.collect_rollout();
    assert!(batch.observations.is_empty());
    a.optimize(&batch);
    assert_eq!(state(&a), before, "empty work must preserve session");
}

#[test]
fn zero_hidden_dimension_is_rejected() {
    let mut c = config();
    c.hidden_dim = 0;
    assert!(std::panic::catch_unwind(|| PpoTrainerSession::new_seeded(c, SEED)).is_err());
}
