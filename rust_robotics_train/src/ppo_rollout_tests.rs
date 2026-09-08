//! Boundary tests use noise-free environments and caller-owned action RNGs.
//! Actor/critic weights are explicit, not random-initialization-dependent. The
//! oracle sums TD errors forward in f64 and never calls production compute_gae.
use super::*;
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_core::{LinearSnapshot, ValueSnapshot};

const SEED: u64 = 0x5050_0030;
const MEAN: f32 = 0.4;

fn close(actual: f32, expected: f64, tolerance: f64, context: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{context}: nonfinite"
    );
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{context}: actual={actual}, expected={expected}"
    );
}

fn config(max_steps: usize) -> PendulumEnvConfig {
    PendulumEnvConfig {
        max_steps,
        max_angle_rad: 1.0e6,
        max_position_m: 1.0e6,
        reset_position_range_m: 0.0,
        reset_velocity_range_mps: 0.0,
        reset_angle_range_rad: 0.0,
        reset_angular_velocity_range_radps: 0.0,
        observation_position_noise_m: 0.0,
        observation_velocity_noise_mps: 0.0,
        observation_angle_noise_rad: 0.0,
        observation_angular_velocity_noise_radps: 0.0,
        action_noise_force_n: 0.0,
        disturbance_force_n: 0.0,
        disturbance_probability_per_step: 0.0,
        reward_position_weight: 0.0,
        reward_velocity_weight: 0.0,
        reward_angle_weight: 0.0,
        reward_angular_velocity_weight: 0.0,
        reward_action_weight: 0.0,
        ..PendulumEnvConfig::default()
    }
}

fn layer(in_dim: usize, weights: Vec<f32>, bias: f32) -> LinearSnapshot {
    LinearSnapshot {
        in_dim,
        out_dim: 1,
        weight: weights,
        bias: vec![bias],
    }
}

fn session(env: PendulumEnvConfig, rollout_steps: usize) -> PpoTrainerSession {
    let mut session = PpoTrainerSession::new(PpoTrainerConfig {
        env,
        hidden_dim: 1,
        ppo: PpoConfig {
            rollout_steps,
            gamma: 0.75,
            gae_lambda: 0.5,
            // Boundary-only fixtures freeze optimization; objective/optimizer
            // behavior is covered separately by ppo_objective_tests.rs.
            epochs_per_update: 0,
            ..PpoConfig::default()
        },
        ..PpoTrainerConfig::default()
    });
    let policy = PolicySnapshot {
        input: layer(4, vec![0.0; 4], 1.0),
        hidden: layer(1, vec![1.0], 0.0),
        output: layer(1, vec![0.0], MEAN),
        action_limit: 20.0,
        action_std: 2.0,
    };
    let value = ValueSnapshot {
        input: layer(4, vec![0.0, 0.2, 0.0, 0.3], 4.0),
        hidden: layer(1, vec![1.0], 0.0),
        output: layer(1, vec![1.0], -2.0),
    };
    session.load_shared_state(&PpoSharedState { policy, value });
    session
}

fn reference_value(observation: [f32; 4]) -> f64 {
    (4.0 + f64::from(0.2_f32) * f64::from(observation[1])
        + f64::from(0.3_f32) * f64::from(observation[3]))
    .max(0.0)
        - 2.0
}

struct Transition {
    observation: [f32; 4],
    reward: f64,
    value: f64,
    next_value: f64,
    ended: bool,
    terminated: bool,
}

fn trace(session: &PpoTrainerSession, count: usize) -> (Vec<Transition>, [f32; 4]) {
    let mut env = session.env.clone();
    let config = env.config();
    let mut observation = session.current_observation;
    let mut elapsed = 0;
    let mut rng = StdRng::seed_from_u64(SEED);
    let distribution = SquashedGaussian::new(2.0, 20.0);
    let mut result = Vec::new();
    for _ in 0..count {
        let step = env.step(distribution.sample(MEAN, &mut rng).action);
        elapsed += 1;
        // Independently classify raw state, not the production status helper.
        let state = env.state();
        let terminated =
            state[2].abs() > config.max_angle_rad || state[0].abs() > config.max_position_m;
        let ended = terminated || elapsed >= config.max_steps;
        let reward = if terminated { -10.0 } else { 1.0 };
        assert_eq!(f64::from(step.reward), reward);
        result.push(Transition {
            observation,
            reward,
            value: reference_value(observation),
            next_value: if terminated {
                0.0
            } else {
                reference_value(step.observation)
            },
            ended,
            terminated,
        });
        observation = step.observation;
        if ended {
            observation = env.reset();
            elapsed = 0;
        }
    }
    (result, observation)
}

fn reference_targets(trace: &[Transition], gamma: f64, lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let mut returns = Vec::new();
    let mut advantages = Vec::new();
    for start in 0..trace.len() {
        let mut weight = 1.0;
        let mut advantage = 0.0;
        for item in &trace[start..] {
            advantage += weight * (item.reward + gamma * item.next_value - item.value);
            if item.ended {
                break;
            }
            weight *= gamma * lambda;
        }
        returns.push(advantage + trace[start].value);
        advantages.push(advantage);
    }
    (returns, advantages)
}

fn check_collector(session: &mut PpoTrainerSession) -> Vec<Transition> {
    let (expected, final_observation) = trace(session, session.config.ppo.rollout_steps);
    let (returns, advantages) = reference_targets(
        &expected,
        f64::from(session.config.ppo.gamma),
        f64::from(session.config.ppo.gae_lambda),
    );
    let batch = session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    assert_eq!(batch.returns.len(), expected.len());
    assert_eq!(batch.advantages.len(), expected.len());
    assert_eq!(batch.observations.len(), expected.len());
    assert_eq!(
        session.current_observation, final_observation,
        "collector must retain next observation"
    );
    let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
    let variance =
        advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
    for (index, item) in expected.iter().enumerate() {
        assert_eq!(
            batch.observations[index], item.observation,
            "collector observation sequence"
        );
        // Short f32 TD sums/MLP inference versus independent f64 arithmetic.
        close(
            batch.returns[index],
            returns[index],
            2e-5,
            "collector return oracle",
        );
        // Normalization additionally amplifies rounding by the batch deviation.
        close(
            batch.advantages[index],
            (advantages[index] - mean) / variance.sqrt().max(1e-6),
            2e-4,
            "whole-rollout advantage normalization",
        );
    }
    expected
}

#[test]
fn gae_matches_forward_sum_across_terminal_masks() {
    let rewards = [1.0_f32, -2.0, 3.0, 10.0, -1.0, 0.5];
    let values = [0.4_f32, 0.3, -0.6, 7.0, 2.0, -1.0];
    for mask in 0..64 {
        let terminals: Vec<bool> = (0..6).map(|i| mask & (1 << i) != 0).collect();
        let trace: Vec<Transition> = (0..6)
            .map(|i| Transition {
                observation: [0.0; 4],
                reward: rewards[i].into(),
                value: values[i].into(),
                next_value: if terminals[i] {
                    0.0
                } else {
                    f64::from(*values.get(i + 1).unwrap_or(&2.0))
                },
                ended: terminals[i],
                terminated: terminals[i],
            })
            .collect();
        for gamma in [0.0_f32, 0.5, 0.99, 1.0] {
            for lambda in [0.0_f32, 0.5, 0.95, 1.0] {
                let (expected_returns, expected_advantages) =
                    reference_targets(&trace, gamma.into(), lambda.into());
                let (returns, advantages) =
                    compute_gae(&rewards, &values, &terminals, 2.0, gamma, lambda);
                for i in 0..6 {
                    close(returns[i], expected_returns[i], 2e-5, "GAE forward return");
                    close(
                        advantages[i],
                        expected_advantages[i],
                        2e-5,
                        "GAE forward advantage",
                    );
                }
            }
        }
    }
}

#[test]
fn gae_empty_input_is_empty() {
    let (returns, advantages) = compute_gae(&[], &[], &[], 3.0, 0.9, 0.8);
    assert!(returns.is_empty() && advantages.is_empty());
}

#[test]
fn gae_rejects_mismatched_lengths() {
    for (values, terminals) in [
        (vec![], vec![false]),
        (vec![0.0, 0.0], vec![false]),
        (vec![0.0], vec![]),
        (vec![0.0], vec![false, false]),
    ] {
        assert!(std::panic::catch_unwind(|| compute_gae(
            &[1.0],
            &values,
            &terminals,
            0.0,
            0.9,
            0.8
        ))
        .is_err());
    }
}

#[test]
fn collector_bootstraps_timeouts_before_reset() {
    let mut session = session(config(2), 5);
    let trace = check_collector(&mut session);
    assert_eq!(trace.iter().filter(|t| t.ended).count(), 2);
    assert!(trace.iter().all(|t| !t.terminated));
    assert!(
        trace
            .iter()
            .filter(|t| t.ended)
            .all(|t| (t.next_value - 2.0).abs() > 1e-3),
        "fixture must distinguish final observation from reset"
    );
}

#[test]
fn single_step_timeouts_do_not_share_advantages() {
    let mut session = session(config(1), 5);
    let trace = check_collector(&mut session);
    assert!(trace.iter().all(|t| t.ended && !t.terminated));
    assert_eq!(session.metrics.total_episodes, 5);
}

#[test]
fn collector_stops_at_true_termination() {
    let mut env = config(100);
    env.max_angle_rad = 0.0;
    let mut session = session(env, 7);
    let trace = check_collector(&mut session);
    assert!(
        trace.iter().any(|t| t.terminated),
        "fixture must include actual failure"
    );
    assert!(trace.iter().filter(|t| t.ended).all(|t| t.terminated));
}

#[test]
fn collector_terminal_on_timeout_uses_zero_value() {
    let mut env = config(2);
    env.max_angle_rad = 0.0;
    let mut session = session(env, 6);
    let trace = check_collector(&mut session);
    assert!(
        trace[1].terminated,
        "fixture must fail on its second-step time limit"
    );
    assert_eq!(trace[1].next_value, 0.0);
}

#[test]
fn collector_continues_unfinished_rollout() {
    let mut session = session(config(100), 3);
    let trace = check_collector(&mut session);
    assert!(trace.iter().all(|t| !t.ended));
    assert_eq!(session.metrics.total_episodes, 0);
    let before = session.current_observation;
    let next = session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED + 1));
    assert_eq!(
        next.observations[0], before,
        "rollout boundary must not reset the environment"
    );
    assert_eq!(session.metrics.total_env_steps, 6);
}

#[test]
fn partial_episode_returns_cross_rollouts() {
    let mut session = session(config(5), 2);
    for update in 1..=5 {
        session.train_updates(1);
        assert_eq!(session.metrics.total_env_steps, update * 2);
        assert_eq!(session.metrics.total_updates, update);
        assert_eq!(session.metrics.total_episodes, update * 2 / 5);
        if update >= 3 {
            assert_eq!(
                session.metrics.last_episode_return, 5.0,
                "whole episode return spans rollout buffers"
            );
            assert_eq!(session.metrics.mean_episode_return, 5.0);
            assert_eq!(session.metrics.best_episode_return, 5.0);
        }
    }
    assert_eq!(session.recent_episode_returns, vec![5.0, 5.0]);
    assert_eq!(session.episode_return, 0.0);
}

#[test]
fn multiple_completed_episodes_count_once() {
    let mut session = session(config(3), 8);
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    assert_eq!(session.metrics.total_env_steps, 8);
    assert_eq!(session.metrics.total_episodes, 2);
    assert_eq!(
        session.recent_episode_returns,
        vec![3.0, 3.0],
        "completed episodes must not accumulate one another"
    );
    assert_eq!(session.episode_return, 2.0);
    assert_eq!(session.metrics.last_episode_return, 3.0);
    assert_eq!(session.metrics.mean_episode_return, 3.0);
    assert_eq!(session.metrics.best_episode_return, 3.0);
}

#[test]
fn negative_first_episode_sets_best_return() {
    let mut env = config(100);
    env.max_angle_rad = 0.0;
    let mut session = session(env, 2);
    let expected = trace(&session, 2).0;
    assert!(expected.last().unwrap().terminated);
    let total = expected.iter().map(|t| t.reward).sum::<f64>();
    assert!(total < 0.0);
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    close(
        session.metrics.best_episode_return,
        total,
        1e-6,
        "negative first episode best return",
    );
    close(
        session.metrics.last_episode_return,
        total,
        1e-6,
        "negative episode return",
    );
}

#[test]
fn weight_transfer_preserves_partial_episode() {
    let mut session = session(config(5), 2);
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    let observation = session.current_observation;
    let metrics = session.metrics.clone();
    session.load_shared_state(&session.shared_state());
    assert_eq!(session.current_observation, observation);
    assert_eq!(session.metrics, metrics);
    session.config.ppo.rollout_steps = 3;
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED + 1));
    assert_eq!(
        session.metrics.last_episode_return, 5.0,
        "weight transfer must retain unfinished return"
    );
    assert_eq!(session.metrics.total_episodes, 1);
    assert_eq!(session.episode_return, 0.0);
}

#[test]
fn zero_length_rollout_preserves_episode() {
    let mut session = session(config(5), 2);
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    let before = session.current_observation;
    let state = session.env.state();
    let partial = session.episode_return;
    session.config.ppo.rollout_steps = 0;
    let batch = session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    assert!(
        batch.observations.is_empty() && batch.returns.is_empty() && batch.advantages.is_empty()
    );
    assert_eq!(session.current_observation, before);
    assert_eq!(session.env.state(), state);
    assert_eq!(session.episode_return, partial);
    assert_eq!(session.metrics.total_env_steps, 2);
    assert_eq!(session.metrics.total_episodes, 0);
}

#[test]
fn zero_updates_leave_session_unchanged() {
    let mut session = session(config(5), 2);
    session.collect_rollout_with_rng(&mut StdRng::seed_from_u64(SEED));
    let metrics = session.metrics.clone();
    let observation = session.current_observation;
    let partial = session.episode_return;
    let snapshot = session.snapshot();
    session.train_updates(0);
    assert_eq!(session.metrics, metrics);
    assert_eq!(session.current_observation, observation);
    assert_eq!(session.episode_return, partial);
    assert_eq!(session.snapshot(), snapshot);
}
