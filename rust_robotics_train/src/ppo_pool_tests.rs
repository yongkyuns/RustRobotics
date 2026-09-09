//! Shared-policy collection regressions, including an independent forward GAE oracle.
use super::*;

fn config() -> PpoTrainerConfig {
    PpoTrainerConfig {
        hidden_dim: 8,
        ppo: PpoConfig {
            rollout_steps: 7,
            mini_batch_size: 8,
            epochs_per_update: 2,
            ..PpoConfig::default()
        },
        ..PpoTrainerConfig::default()
    }
}

fn assert_models(a: &PpoTrainerSession, b: &PpoTrainerSession) {
    assert_eq!(a.snapshot(), b.snapshot());
    assert_eq!(a.shared_state().value, b.shared_state().value);
}

fn assert_batch(a: &RolloutBatch, b: &RolloutBatch) {
    assert_eq!(a.observations, b.observations);
    assert_eq!(a.latent_actions, b.latent_actions);
    assert_eq!(a.old_log_probs, b.old_log_probs);
    assert_eq!(a.returns, b.returns);
    assert_eq!(a.advantages, b.advantages);
}

#[test]
fn one_environment_constructor_preserves_original_seeded_training() {
    let cfg = config();
    let mut original = PpoTrainerSession::new_seeded(cfg.clone(), 201);
    let mut pooled = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 1);
    for _ in 0..4 {
        assert_models(&original, &pooled);
        assert_eq!(original.metrics(), pooled.metrics());
        assert_batch(&original.collect_rollout(), &pooled.collect_rollout());
        original.train_updates(1);
        pooled.train_updates(1);
    }
    assert_models(&original, &pooled);
    assert_eq!(original.metrics(), pooled.metrics());
}

#[test]
fn pooled_updates_are_grouping_invariant_and_count_actual_interactions() {
    for count in [2, 3, 8] {
        let cfg = config();
        let mut grouped = PpoTrainerSession::new_seeded_with_environments(cfg.clone(), 201, count);
        let mut split = PpoTrainerSession::new_seeded_with_environments(cfg, 201, count);
        grouped.train_updates(4);
        split.train_updates(1);
        split.train_updates(2);
        split.train_updates(1);
        assert_models(&grouped, &split);
        assert_eq!(grouped.metrics(), split.metrics());
        assert_eq!(grouped.metrics().total_env_steps, 4 * count * 7);
        assert_eq!(grouped.metrics().total_updates, 4);
        assert_eq!(grouped.environment_count(), count);
        assert_batch(&grouped.collect_rollout(), &split.collect_rollout());
    }
}

#[test]
fn pooled_update_optimizes_the_union_once_not_separate_agents() {
    let cfg = config();
    let mut automatic = PpoTrainerSession::new_seeded_with_environments(cfg.clone(), 201, 3);
    let mut explicit = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 3);
    for _ in 0..3 {
        automatic.train_updates(1);
        let union = explicit.collect_rollout();
        assert_eq!(union.observations.len(), 21);
        explicit.optimize(&union);
        explicit.metrics.total_updates += 1;
        assert_models(&automatic, &explicit);
        assert_eq!(automatic.metrics(), explicit.metrics());
    }
}

#[test]
fn extra_environment_count_does_not_perturb_existing_streams_before_update() {
    let cfg = config();
    let mut a = PpoTrainerSession::new_seeded_with_environments(cfg.clone(), 201, 2);
    let mut b = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 5);
    assert_models(&a, &b);
    for _ in 0..3 {
        let x = a.collect_rollout();
        let y = b.collect_rollout();
        assert_eq!(x.observations, y.observations[..14]);
        assert_eq!(x.latent_actions, y.latent_actions[..14]);
        assert_eq!(x.old_log_probs, y.old_log_probs[..14]);
        assert_eq!(x.returns, y.returns[..14]);
        // Global normalization is intentionally different for a larger pool.
        assert_eq!(a.env.state(), b.env.state());
        assert_eq!(a.episode_return, b.episode_return);
        assert_eq!(a.additional_environments[0].env.state(), b.additional_environments[0].env.state());
    }
}

#[test]
fn optimizer_randomness_cannot_change_any_environment_stream() {
    let cfg = config();
    let mut a = PpoTrainerSession::new_seeded_with_environments(cfg.clone(), 201, 3);
    let mut b = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 3);
    for _ in 0..101 { let _: u64 = b.update_rng.gen(); }
    assert_batch(&a.collect_rollout(), &b.collect_rollout());
}

fn stream_fingerprint(stream: &RolloutEnvironment) -> ([f32; 4], [f32; 4], f32, u64, u64) {
    let x = stream.env.state();
    (stream.current_observation, [x[0], x[1], x[2], x[3]], stream.episode_return,
        stream.environment_rng.clone().gen(), stream.action_rng.clone().gen())
}

#[test]
fn external_weight_transfer_preserves_all_live_environment_streams() {
    let cfg = config();
    let mut session = PpoTrainerSession::new_seeded_with_environments(cfg.clone(), 201, 3);
    session.collect_rollout();
    let before = session.additional_environments.iter().map(stream_fingerprint).collect::<Vec<_>>();
    let observation = session.current_observation;
    let metrics = session.metrics().clone();
    let state = PpoTrainerSession::new_seeded(cfg, 202).shared_state();
    session.load_shared_state(&state);
    assert_eq!(session.snapshot(), state.policy);
    assert_eq!(session.shared_state().value, state.value);
    assert_eq!(before, session.additional_environments.iter().map(stream_fingerprint).collect::<Vec<_>>());
    assert_eq!(session.current_observation, observation);
    assert_eq!(session.metrics(), &metrics);
}

#[test]
fn empty_collection_and_zero_updates_preserve_all_streams() {
    let mut cfg = config();
    cfg.ppo.rollout_steps = 0;
    let mut session = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 3);
    let before = session.additional_environments.iter().map(stream_fingerprint).collect::<Vec<_>>();
    let state = session.shared_state();
    let action_cursor: u64 = session.action_rng.clone().gen();
    session.train_updates(0);
    assert_eq!(session.metrics().total_updates, 0);
    let batch = session.collect_rollout();
    assert!(batch.observations.is_empty() && batch.advantages.is_empty());
    assert_eq!(before, session.additional_environments.iter().map(stream_fingerprint).collect::<Vec<_>>());
    assert_eq!(session.action_rng.clone().gen::<u64>(), action_cursor);
    assert_eq!(session.snapshot(), state.policy);
    assert_eq!(session.shared_state().value, state.value);
    assert_eq!(session.metrics().total_env_steps, 0);
}

#[test]
fn invalid_pool_sizes_fail_before_any_rollout_allocation() {
    assert!(std::panic::catch_unwind(|| PpoTrainerSession::new_seeded_with_environments(config(), 201, 0)).is_err());
    assert!(std::panic::catch_unwind(|| {
        let mut cfg = config();
        cfg.ppo.rollout_steps = usize::MAX;
        PpoTrainerSession::new_seeded_with_environments(cfg, 201, 2)
    }).is_err());
}

#[derive(Clone)]
struct ReferenceStream {
    state: RolloutEnvironment,
    age: usize,
}

// Constant networks avoid conflating portable/Burn reduction differences with
// boundary/collection correctness. The real stochastic actions/noisy Rust
// environment are still executed and every continuation cursor is checked.
fn constant_session(terminal: bool) -> PpoTrainerSession {
    let mut cfg = config();
    cfg.ppo.gamma = 0.75;
    cfg.ppo.gae_lambda = 0.5;
    cfg.ppo.rollout_steps = 2;
    cfg.env.max_steps = 3;
    if terminal { cfg.env.max_angle_rad = 0.0; }
    let mut session = PpoTrainerSession::new_seeded_with_environments(cfg, 201, 3);
    let mut state = session.shared_state();
    for layer in [&mut state.policy.input, &mut state.policy.hidden, &mut state.policy.output,
        &mut state.value.input, &mut state.value.hidden, &mut state.value.output] {
        layer.weight.fill(0.0);
        layer.bias.fill(0.0);
    }
    state.policy.output.bias[0] = 0.25;
    state.value.output.bias[0] = 2.0;
    session.load_shared_state(&state);
    session
}

fn references(session: &PpoTrainerSession) -> Vec<ReferenceStream> {
    std::iter::once(RolloutEnvironment {
        env: session.env.clone(), current_observation: session.current_observation,
        episode_return: session.episode_return, environment_rng: session.environment_rng.clone(),
        action_rng: session.action_rng.clone(),
    }).chain(session.additional_environments.iter().cloned())
      .map(|state| ReferenceStream { state, age: 0 }).collect()
}

#[test]
fn pooled_gae_matches_forward_oracle_across_timeouts_and_live_cutoffs() {
    let mut session = constant_session(false);
    let mut streams = references(&session);
    let distribution = SquashedGaussian::new(2.0, 20.0);
    let mut completed = Vec::new();
    for _ in 0..4 {
        let mut expected_obs = Vec::new();
        let mut expected_actions = Vec::new();
        let mut expected_returns = Vec::new();
        let mut raw_advantages = Vec::new();
        for stream in &mut streams {
            let mut rows = Vec::new();
            for _ in 0..2 {
                expected_obs.push(stream.state.current_observation);
                let action = distribution.sample(0.25, &mut stream.state.action_rng);
                expected_actions.push(action.latent);
                let result = stream.state.env.step_with_rng(action.action, &mut stream.state.environment_rng);
                stream.age += 1;
                let x = stream.state.env.state();
                let env = stream.state.env.config();
                let terminal = x[0].abs() > env.max_position_m || x[2].abs() > env.max_angle_rad;
                let ended = terminal || stream.age >= env.max_steps;
                assert_eq!(result.done, ended);
                assert_eq!(result.truncated, ended && !terminal);
                rows.push((f64::from(result.reward), terminal, ended));
                stream.state.episode_return += result.reward;
                stream.state.current_observation = result.observation;
                if ended {
                    completed.push(std::mem::take(&mut stream.state.episode_return));
                    stream.state.current_observation = stream.state.env.reset_with_rng(&mut stream.state.environment_rng);
                    stream.age = 0;
                }
            }
            // Forward sum in f64: no production GAE or normalization helper.
            for start in 0..rows.len() {
                let mut advantage = 0.0;
                let mut weight = 1.0;
                for &(reward, terminal, ended) in &rows[start..] {
                    advantage += weight * (reward + (if terminal { 0.0 } else { 0.75 * 2.0 }) - 2.0);
                    if ended { break; }
                    weight *= 0.75 * 0.5;
                }
                expected_returns.push(advantage + 2.0);
                raw_advantages.push(advantage);
            }
        }
        let actual = session.collect_rollout();
        assert_eq!(actual.observations, expected_obs);
        assert_eq!(actual.latent_actions, expected_actions);
        let mean = raw_advantages.iter().sum::<f64>() / raw_advantages.len() as f64;
        let std = (raw_advantages.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / raw_advantages.len() as f64).sqrt().max(1e-6);
        for i in 0..actual.returns.len() {
            assert!((f64::from(actual.returns[i]) - expected_returns[i]).abs() < 2e-5, "stream-local return target");
            let expected = (raw_advantages[i]-mean)/std;
            assert!((f64::from(actual.advantages[i])-expected).abs() < 2e-4, "one normalization over the pool");
        }
        assert_eq!(session.env.state(), streams[0].state.env.state());
        assert_eq!(session.episode_return, streams[0].state.episode_return);
        for (actual, expected) in session.additional_environments.iter().zip(&streams[1..]) {
            assert_eq!(stream_fingerprint(actual), stream_fingerprint(&expected.state));
        }
        assert_eq!(session.metrics().total_episodes, completed.len());
        if let Some(last) = completed.last() {
            assert_eq!(session.metrics().last_episode_return, *last);
            assert_eq!(session.metrics().best_episode_return, completed.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
    }
    assert_eq!(session.metrics().total_env_steps, 24);
}

#[test]
fn true_terminals_do_not_bootstrap_or_share_advantages_across_streams() {
    let mut session = constant_session(true);
    let batch = session.collect_rollout();
    assert_eq!(batch.returns, vec![-10.0; 6]);
    assert_eq!(batch.advantages, vec![0.0; 6]);
    assert_eq!(session.metrics().total_episodes, 6);
    assert_eq!(session.metrics().best_episode_return, -10.0);
    assert!(session.additional_environments.iter().all(|s| s.episode_return == 0.0));
}
