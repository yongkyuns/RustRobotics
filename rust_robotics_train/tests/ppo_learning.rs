//! Protocol v1 was fixed in issue #5 before reading exploratory or confirmation
//! outcomes. Statistical units are 12 training seeds, NOT 384 evaluation pairs.
//! Run the expensive endpoint explicitly in release mode; normal workspace tests
//! run the evaluator/statistical controls. See LEARNING_QUALIFICATION.md.
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_train::{
    LinearSnapshot, PendulumEnv, PendulumEnvConfig, PolicySnapshot, PpoConfig, PpoTrainerConfig,
    PpoTrainerSession,
};

const SEEDS: std::ops::RangeInclusive<u64> = 101..=112;
const UPDATES: usize = 128;
const EPISODES: usize = 32;
const EVAL_CAP: usize = 1000;
const MARGIN: f64 = 25.0;
const MIN_MEAN_GAIN: f64 = 50.0;

// Explicitly pin the default recipe at #31. A separate test catches drift in
// production defaults instead of silently changing the qualification task.
fn recipe() -> PpoTrainerConfig {
    PpoTrainerConfig {
        env: PendulumEnvConfig {
            dt: 0.01,
            max_force: 20.0,
            reset_position_range_m: 0.2,
            reset_velocity_range_mps: 0.4,
            reset_angle_range_rad: 0.25,
            reset_angular_velocity_range_radps: 0.5,
            max_angle_rad: 0.6,
            max_position_m: 2.4,
            max_steps: 5000,
            observation_position_noise_m: 0.002,
            observation_velocity_noise_mps: 0.01,
            observation_angle_noise_rad: 0.002,
            observation_angular_velocity_noise_radps: 0.01,
            action_noise_force_n: 0.15,
            disturbance_force_n: 1.0,
            disturbance_probability_per_step: 0.005,
            reward_position_weight: 0.2,
            reward_velocity_weight: 0.02,
            reward_angle_weight: 1.0,
            reward_angular_velocity_weight: 0.05,
            reward_action_weight: 0.001,
        },
        ppo: PpoConfig {
            rollout_steps: 512,
            mini_batch_size: 128,
            epochs_per_update: 4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.0,
            learning_rate: 3e-4,
        },
        hidden_dim: 64,
        action_std: 2.0,
        sync_policy_each_update: true,
    }
}

fn validate_layer(layer: &LinearSnapshot) -> Result<(), &'static str> {
    if layer.in_dim == 0
        || layer.out_dim == 0
        || layer.in_dim.checked_mul(layer.out_dim) != Some(layer.weight.len())
        || layer.bias.len() != layer.out_dim
    {
        return Err("malformed snapshot layer");
    }
    if !layer
        .weight
        .iter()
        .chain(&layer.bias)
        .all(|v| v.is_finite())
    {
        return Err("nonfinite snapshot parameter");
    }
    Ok(())
}

fn validate_policy(policy: &PolicySnapshot, limit: f32) -> Result<(), &'static str> {
    for layer in [&policy.input, &policy.hidden, &policy.output] {
        validate_layer(layer)?;
    }
    if policy.input.in_dim != 4
        || policy.input.out_dim != policy.hidden.in_dim
        || policy.hidden.out_dim != policy.output.in_dim
        || policy.output.out_dim != 1
    {
        return Err("incompatible snapshot dimensions");
    }
    if !limit.is_finite()
        || limit <= 0.0
        || policy.action_limit != limit
        || !policy.action_std.is_finite()
        || policy.action_std <= 0.0
    {
        return Err("invalid snapshot action metadata");
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct EpisodeScore {
    seed: u64,
    total_return: f64,
    steps: usize,
    terminated: bool,
    truncated: bool,
}

fn evaluate(
    policy: &PolicySnapshot,
    config: PendulumEnvConfig,
    seeds: &[u64],
) -> Result<Vec<EpisodeScore>, &'static str> {
    validate_policy(policy, config.max_force)?;
    if seeds.is_empty() || config.max_steps == 0 {
        return Err("empty evaluation");
    }
    let mut scores = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        // A new stream per episode is essential: an earlier policy-dependent
        // episode ending must not shift the random conditions of later pairs.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut env = PendulumEnv::new_with_rng(Default::default(), config, &mut rng);
        let mut observation = env.observation_with_rng(&mut rng);
        let mut total_return = 0.0;
        for steps in 1..=config.max_steps {
            if !observation.iter().all(|v| v.is_finite()) {
                return Err("nonfinite observation");
            }
            let action = policy.act(observation);
            if !action.is_finite() || action.abs() > config.max_force {
                return Err("invalid evaluated action");
            }
            let result = env.step_with_rng(action, &mut rng);
            if !result.reward.is_finite()
                || !result.observation.iter().all(|v| v.is_finite())
                || !env.state().iter().all(|v| v.is_finite())
            {
                return Err("nonfinite evaluation transition");
            }
            total_return += f64::from(result.reward);
            observation = result.observation;
            if result.done {
                if result.truncated && steps != config.max_steps {
                    return Err("premature timeout");
                }
                scores.push(EpisodeScore {
                    seed,
                    total_return,
                    steps,
                    terminated: result.terminated(),
                    truncated: result.truncated,
                });
                break;
            }
            if steps == config.max_steps {
                return Err("missing episode end");
            }
        }
    }
    if scores.len() != seeds.len() {
        return Err("missing episode score");
    }
    Ok(scores)
}

fn mean(scores: &[EpisodeScore]) -> f64 {
    scores.iter().map(|s| s.total_return).sum::<f64>() / scores.len() as f64
}

// Exact integer binomial sum; n is restricted to this small fixture's range.
fn binomial_tail(n: usize, wins: usize) -> f64 {
    assert!(n <= 12 && wins <= n);
    let mut choose = 1_u64;
    let mut tail = 0_u64;
    for i in 0..=n {
        if i >= wins {
            tail += choose;
        }
        if i < n {
            choose = choose * (n - i) as u64 / (i + 1) as u64;
        }
    }
    tail as f64 / (1_u64 << n) as f64
}

#[derive(Debug, PartialEq)]
struct Summary {
    wins: usize,
    mean_gain: f64,
    median_gain: f64,
    median_low: f64,
    median_high: f64,
    p_margin: f64,
    accepted: bool,
}

fn summarize(gains: &[f64]) -> Result<Summary, &'static str> {
    if gains.len() != 12 || !gains.iter().all(|v| v.is_finite()) {
        return Err("expected twelve finite independent trial gains");
    }
    let wins = gains.iter().filter(|&&gain| gain > MARGIN).count();
    let mean_gain = gains.iter().sum::<f64>() / 12.0;
    if !mean_gain.is_finite() {
        return Err("nonfinite aggregate");
    }
    let mut ordered = gains.to_vec();
    ordered.sort_by(f64::total_cmp);
    let p_margin = binomial_tail(12, wins);
    Ok(Summary {
        wins,
        mean_gain,
        median_gain: ordered[5] / 2.0 + ordered[6] / 2.0,
        // P(Binomial(12, .5) <= 2) = 79/4096; central coverage 96.142578125%.
        // This interval concerns trial-level finite-panel gains, not true
        // infinite-episode expected returns or trajectories within a seed.
        median_low: ordered[2],
        median_high: ordered[9],
        p_margin,
        accepted: wins >= 10 && mean_gain >= MIN_MEAN_GAIN,
    })
}

fn emit_policy(seed: u64, phase: &str, policy: &PolicySnapshot) {
    // Full numeric actor evidence in big-endian f32 hex, no serializer dependency.
    // Order: action_limit, action_std, then input/hidden/output weight and bias.
    let mut values = vec![policy.action_limit, policy.action_std];
    for layer in [&policy.input, &policy.hidden, &policy.output] {
        values.extend(&layer.weight);
        values.extend(&layer.bias);
    }
    let hex = values
        .iter()
        .map(|x| format!("{:08x}", x.to_bits()))
        .collect::<String>();
    println!("PPO_POLICY\t{seed}\t{phase}\t{hex}");
}

fn emit_scores(seed: u64, phase: &str, scores: &[EpisodeScore]) {
    for s in scores {
        println!(
            "PPO_EPISODE\t{seed}\t{phase}\t{}\t{:.17}\t{}\t{}\t{}",
            s.seed,
            s.total_return,
            s.steps,
            u8::from(s.terminated),
            u8::from(s.truncated)
        );
    }
}

#[test]
#[ignore = "fixed-budget learning qualification; run scripts/qualify_ppo_learning.py"]
fn seeded_short_learning_improves_held_out_returns() {
    println!("PPO_PROTOCOL\tv1\t12\t128\t512\t32\t1000\t25\t50");
    println!("PPO_CONFIG\t{:?}", recipe());
    let mut gains = Vec::new();
    for seed in SEEDS {
        let mut trainer = PpoTrainerSession::new_seeded(recipe(), seed);
        let initial = trainer.snapshot();
        let eval_seeds = (0..EPISODES)
            .map(|i| 1_000_000 + (seed - 101) * 1000 + i as u64)
            .collect::<Vec<_>>();
        let eval_config = PendulumEnvConfig {
            max_steps: EVAL_CAP,
            ..recipe().env
        };
        let before =
            evaluate(&initial, eval_config, &eval_seeds).expect("valid initial evaluation");
        emit_policy(seed, "initial", &initial);
        emit_scores(seed, "initial", &before);
        for _ in 0..UPDATES {
            trainer.train_updates(1);
            let shared = trainer.shared_state();
            validate_policy(&shared.policy, recipe().env.max_force)
                .expect("finite training policy");
            for layer in [
                &shared.value.input,
                &shared.value.hidden,
                &shared.value.output,
            ] {
                validate_layer(layer).expect("finite training critic");
            }
            let m = trainer.metrics();
            assert!(
                [
                    m.last_policy_loss,
                    m.last_value_loss,
                    m.last_mean_advantage,
                    m.last_episode_return,
                    m.mean_episode_return,
                    m.best_episode_return
                ]
                .iter()
                .all(|v| v.is_finite()),
                "finite training metrics"
            );
        }
        assert_eq!(
            trainer.metrics().total_updates,
            UPDATES,
            "fixed update budget"
        );
        assert_eq!(
            trainer.metrics().total_env_steps,
            UPDATES * 512,
            "fixed transition budget"
        );
        let trained = trainer.snapshot();
        let after = evaluate(&trained, eval_config, &eval_seeds).expect("valid trained evaluation");
        emit_policy(seed, "final", &trained);
        emit_scores(seed, "final", &after);
        let gain = mean(&after) - mean(&before);
        gains.push(gain);
        println!(
            "PPO_TRIAL\t{seed}\t{UPDATES}\t{}\t{:.17}\t{:.17}\t{gain:.17}",
            trainer.metrics().total_env_steps,
            mean(&before),
            mean(&after)
        );
    }
    let s = summarize(&gains).expect("complete cohort");
    println!(
        "PPO_SUMMARY\t{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}",
        s.wins,
        s.mean_gain,
        s.median_gain,
        s.median_low,
        s.median_high,
        s.p_margin,
        u8::from(s.accepted)
    );
    assert!(
        s.accepted,
        "predeclared PPO learning criterion failed: {s:?}; all seeds retained"
    );
}

#[test]
fn frozen_recipe_matches_current_defaults() {
    assert_eq!(
        format!("{:?}", recipe()),
        format!("{:?}", PpoTrainerConfig::default()),
        "default recipe drift requires an explicit protocol revision"
    );
}

#[test]
fn evaluator_replays_equal_policies() {
    let policy = PpoTrainerSession::new_seeded(recipe(), 40).snapshot();
    let config = PendulumEnvConfig {
        max_steps: 50,
        ..recipe().env
    };
    assert_eq!(
        evaluate(&policy, config, &[41, 42, 43]),
        evaluate(&policy, config, &[41, 42, 43])
    );
}

#[test]
fn episode_streams_are_order_independent() {
    let policy = PpoTrainerSession::new_seeded(recipe(), 40).snapshot();
    let config = PendulumEnvConfig {
        max_steps: 150,
        ..recipe().env
    };
    let a = evaluate(&policy, config, &[41, 42, 43]).unwrap();
    let b = evaluate(&policy, config, &[43, 41, 42]).unwrap();
    assert_eq!(
        [a[2].clone(), a[0].clone(), a[1].clone()],
        b.as_slice(),
        "episode RNGs must not depend on preceding episode lengths"
    );
}

#[test]
fn evaluator_rejects_invalid_snapshots_and_empty_evaluation() {
    let policy = PpoTrainerSession::new_seeded(recipe(), 40).snapshot();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = policy.clone();
        invalid.hidden.weight[0] = bad;
        assert!(
            evaluate(&invalid, recipe().env, &[41]).is_err(),
            "nonfinite parameters must not be masked by ReLU"
        );
    }
    let mut malformed = policy.clone();
    malformed.hidden.in_dim += 1;
    assert!(evaluate(&malformed, recipe().env, &[41]).is_err());
    let mut wrong_limit = policy.clone();
    wrong_limit.action_limit = 1.0;
    assert!(evaluate(&wrong_limit, recipe().env, &[41]).is_err());
    assert!(evaluate(&policy, recipe().env, &[]).is_err());
    assert!(evaluate(
        &policy,
        PendulumEnvConfig {
            max_steps: 0,
            ..recipe().env
        },
        &[41]
    )
    .is_err());
}

fn constant_policy(bias: f32) -> PolicySnapshot {
    fn layer(input: usize, output: usize, bias: f32) -> LinearSnapshot {
        LinearSnapshot {
            in_dim: input,
            out_dim: output,
            weight: vec![0.0; input * output],
            bias: vec![bias; output],
        }
    }
    PolicySnapshot {
        input: layer(4, 1, 0.0),
        hidden: layer(1, 1, 0.0),
        output: layer(1, 1, bias),
        action_limit: 20.0,
        action_std: 2.0,
    }
}

#[test]
fn evaluator_obeys_policy_rewards_and_episode_ends() {
    let config = PendulumEnvConfig {
        dt: 0.0,
        max_steps: 3,
        reset_position_range_m: 0.0,
        reset_velocity_range_mps: 0.0,
        reset_angle_range_rad: 0.0,
        reset_angular_velocity_range_radps: 0.0,
        observation_position_noise_m: 0.0,
        observation_velocity_noise_mps: 0.0,
        observation_angle_noise_rad: 0.0,
        observation_angular_velocity_noise_radps: 0.0,
        action_noise_force_n: 0.0,
        disturbance_probability_per_step: 0.0,
        ..recipe().env
    };
    let zero = evaluate(&constant_policy(0.0), config, &[41]).unwrap();
    let forced = evaluate(&constant_policy(100.0), config, &[41]).unwrap();
    assert_eq!(zero[0].total_return, 3.0, "zero-action reward reference");
    assert!(
        (forced[0].total_return - 1.8).abs() < 1e-6,
        "evaluation must execute the supplied policy"
    );
    assert_eq!(zero[0].steps, 3);
    assert!(!zero[0].terminated && zero[0].truncated);
    let failure = PendulumEnvConfig {
        reset_angle_range_rad: 0.2,
        max_angle_rad: 1e-8,
        ..config
    };
    let result = evaluate(&constant_policy(0.0), failure, &[41]).unwrap();
    assert_eq!(result[0].steps, 1, "stop at first termination");
    assert_eq!(
        result[0].total_return, -10.0,
        "terminal penalty counted exactly once"
    );
    assert!(result[0].terminated && !result[0].truncated);
}

#[test]
fn binomial_tail_matches_exhaustive_coin_sequences() {
    for n in 0..=12 {
        for wins in 0..=n {
            let outcomes = 1_u32 << n;
            let count = (0..outcomes)
                .filter(|bits| bits.count_ones() as usize >= wins)
                .count();
            assert_eq!(
                binomial_tail(n, wins),
                count as f64 / outcomes as f64,
                "independent binomial oracle"
            );
        }
    }
    assert_eq!(binomial_tail(12, 10), 79.0 / 4096.0);
}

#[test]
fn acceptance_rejects_ties_outliers_missing_and_nonfinite_trials() {
    assert!(
        !summarize(&[25.0; 12]).unwrap().accepted,
        "margin ties are failures"
    );
    assert!(
        !summarize(&[0.0; 12]).unwrap().accepted,
        "no learning cannot qualify"
    );
    assert!(
        !summarize(&[26.0; 12]).unwrap().accepted,
        "mean effect floor"
    );
    let mut outlier = [0.0; 12];
    outlier[0] = 10000.0;
    assert!(
        !summarize(&outlier).unwrap().accepted,
        "one lucky seed cannot qualify"
    );
    let mut good = [70.0; 12];
    good[0] = 25.0;
    good[1] = 25.0;
    assert!(
        summarize(&good).unwrap().accepted,
        "ten material gains suffice"
    );
    good[2] = 25.0;
    assert!(
        !summarize(&good).unwrap().accepted,
        "nine gains do not suffice"
    );
    assert!(
        summarize(&[50.0; 12]).unwrap().accepted,
        "mean floor includes equality"
    );
    assert!(summarize(&[50.0; 11]).is_err());
    assert!(summarize(&[f64::NAN; 12]).is_err());
    assert!(summarize(&[f64::INFINITY; 12]).is_err());
}

#[test]
fn median_interval_matches_known_order_statistics() {
    let gains = (0..12).rev().map(f64::from).collect::<Vec<_>>();
    let s = summarize(&gains).unwrap();
    assert_eq!(
        (s.median_low, s.median_gain, s.median_high),
        (2.0, 5.5, 9.0)
    );
}

#[test]
fn zero_learning_rate_is_a_real_negative_control() {
    let mut config = recipe();
    config.ppo.learning_rate = 0.0;
    let mut trainer = PpoTrainerSession::new_seeded(config, 77);
    let before = trainer.shared_state();
    trainer.train_updates(4);
    let after = trainer.shared_state();
    assert_eq!(trainer.metrics().total_env_steps, 2048);
    assert_eq!(
        before.policy, after.policy,
        "zero-learning-rate actor must not change"
    );
    assert_eq!(
        before.value, after.value,
        "zero-learning-rate critic must not change"
    );
    let config = PendulumEnvConfig {
        max_steps: 100,
        ..recipe().env
    };
    let initial = evaluate(&before.policy, config, &[81, 82, 83, 84]).unwrap();
    let trained = evaluate(&after.policy, config, &[81, 82, 83, 84]).unwrap();
    assert_eq!(
        initial, trained,
        "no update must not manufacture evaluation improvement"
    );
    // Synthetic zeros check the gate, not 12 additional independent trainings.
    assert!(
        !summarize(&[mean(&trained) - mean(&initial); 12])
            .unwrap()
            .accepted
    );
}
