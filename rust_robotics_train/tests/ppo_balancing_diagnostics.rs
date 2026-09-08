//! Development-only measurements for #33; NOT a sustained-balancing gate.
//! Frozen seed/checkpoint plan: issuecomment-5585901307. No production tuning.
//! Run the ignored endpoint in release mode with PPO_DIAGNOSTIC_DIR set.
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_algo::{
    control::StateSpace,
    inverted_pendulum::Model,
    nalgebra::{SMatrix, SVector},
};
use rust_robotics_train::{
    PendulumEnv, PendulumEnvConfig, PolicySnapshot, PpoSharedState, PpoTrainerConfig,
    PpoTrainerSession, ValueSnapshot,
};
use std::{fs, io::{self, BufWriter, Write}, path::Path};

const SEEDS: [u64; 4] = [201, 202, 203, 204];
const CHECKPOINTS: [usize; 4] = [0, 128, 512, 2048];
const EPISODES: usize = 32;
const CAP: usize = 1000;

type Matrix = SMatrix<f64, 4, 4>;
type Vector = SVector<f64, 4>;

fn evaluation_config() -> PendulumEnvConfig {
    PendulumEnvConfig { max_steps: CAP, ..Default::default() }
}

fn episode_seed(seed: u64, episode: usize) -> u64 {
    0x3300_0000 + seed * 1024 + episode as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending { Angle, Position, Both, Timeout }

fn ending(state: [f32; 4], c: PendulumEnvConfig, steps: usize) -> Option<Ending> {
    match (state[2].abs() > c.max_angle_rad, state[0].abs() > c.max_position_m) {
        (true, true) => Some(Ending::Both),
        (true, false) => Some(Ending::Angle),
        (false, true) => Some(Ending::Position),
        (false, false) if steps == c.max_steps => Some(Ending::Timeout),
        _ => None,
    }
}

// An independent f64 test-only DARE solve on the actual f32 plant matrices.
// Q/R use the task's quadratic penalty weights, not the model's LQR defaults.
// This undiscounted design ignores stopping penalties and saturation; its
// clipped execution is a diagnostic reference, not the optimal PPO policy.
fn lqr_reference(c: PendulumEnvConfig) -> ([f32; 4], Matrix, usize) {
    let (a, b) = Model::default().model(c.dt);
    let a = a.cast::<f64>();
    let b = b.cast::<f64>();
    let q = Matrix::from_diagonal(&Vector::new(
        c.reward_position_weight.into(), c.reward_velocity_weight.into(),
        c.reward_angle_weight.into(), c.reward_angular_velocity_weight.into(),
    ));
    let r = f64::from(c.reward_action_weight);
    assert!(r > 0.0 && r.is_finite());
    let mut p = q;
    for iteration in 1..=50_000 {
        let k = (b.transpose() * p * a) / (r + (b.transpose() * p * b)[0]);
        let next = q + a.transpose() * p * a - a.transpose() * p * b * k;
        let next = (next + next.transpose()) * 0.5;
        let residual = (next - p).amax() / (1.0 + next.amax());
        p = next;
        assert!(p.iter().all(|x| x.is_finite()));
        if residual < 1e-12 {
            assert!(p.cholesky().is_some(), "DARE solution must be positive definite");
            let k = (b.transpose() * p * a) / (r + (b.transpose() * p * b)[0]);
            return ([k[0] as f32, k[1] as f32, k[2] as f32, k[3] as f32], p, iteration);
        }
    }
    panic!("diagnostic DARE failed to converge");
}

fn feedback(gain: [f32; 4], observation: [f32; 4], limit: f32) -> f32 {
    (-gain.iter().zip(observation).map(|(k, x)| k * x).sum::<f32>()).clamp(-limit, limit)
}

fn value(snapshot: &ValueSnapshot, observation: [f32; 4]) -> f64 {
    let mut first = snapshot.input.forward(&observation);
    first.iter_mut().for_each(|x| *x = x.max(0.0));
    let mut second = snapshot.hidden.forward(&first);
    second.iter_mut().for_each(|x| *x = x.max(0.0));
    f64::from(snapshot.output.forward(&second)[0])
}

fn check_snapshot(shared: &PpoSharedState) {
    for layer in [&shared.policy.input, &shared.policy.hidden, &shared.policy.output,
        &shared.value.input, &shared.value.hidden, &shared.value.output] {
        assert_eq!(layer.in_dim.checked_mul(layer.out_dim), Some(layer.weight.len()));
        assert_eq!(layer.out_dim, layer.bias.len());
        assert!(layer.weight.iter().chain(&layer.bias).all(|x| x.is_finite()));
    }
    assert_eq!(shared.policy.action_limit, 20.0);
    assert_eq!(shared.policy.action_std, 2.0);
}

#[derive(Debug, PartialEq)]
struct Episode {
    total_return: f64,
    discounted_return: f64,
    steps: usize,
    ending: Ending,
    initial_observation: [f32; 4],
    final_state: [f32; 4],
    maximum_absolute_state: [f32; 4],
    absolute_force_sum: f64,
    near_limit_steps: usize,
    initial_value: Option<f64>,
}

// A policy only receives the same noisy observation used by PPO. The true
// state is read for diagnostics after choosing an action, never for control.
fn episode(
    c: PendulumEnvConfig,
    seed: u64,
    policy: impl Fn([f32; 4]) -> f32,
    critic: Option<&ValueSnapshot>,
    trace: &mut impl Write,
    trace_key: Option<&str>,
) -> io::Result<Episode> {
    assert!(c.max_steps > 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = PendulumEnv::new_with_rng(Default::default(), c, &mut rng);
    let mut observation = env.observation_with_rng(&mut rng);
    let initial_observation = observation;
    let initial_value = critic.map(|v| value(v, observation));
    assert!(initial_value.is_none_or(f64::is_finite));
    let mut total_return = 0.0;
    let mut discounted_return = 0.0;
    let mut discount = 1.0;
    let mut absolute_force_sum = 0.0;
    let mut near_limit_steps = 0;
    let mut maximum_absolute_state = env.state().map(f32::abs);
    for steps in 1..=c.max_steps {
        assert!(observation.iter().all(|x| x.is_finite()));
        let action = policy(observation);
        assert!(action.is_finite() && action.abs() <= c.max_force, "finite bounded action required");
        absolute_force_sum += f64::from(action.abs());
        near_limit_steps += usize::from(action.abs() >= 0.95 * c.max_force);
        let before = env.state();
        let result = env.step_with_rng(action, &mut rng);
        let state = env.state();
        assert!(state.iter().chain(result.observation.iter()).all(|x| x.is_finite()));
        assert!(result.reward.is_finite());
        for i in 0..4 { maximum_absolute_state[i] = maximum_absolute_state[i].max(state[i].abs()); }
        total_return += f64::from(result.reward);
        discounted_return += discount * f64::from(result.reward);
        discount *= f64::from(PpoTrainerConfig::default().ppo.gamma);
        let physical = [state[0], state[1], state[2], state[3]];
        let reason = ending(physical, c, steps);
        assert_eq!(result.done, reason.is_some(), "independent episode-end classification");
        assert_eq!(result.truncated, reason == Some(Ending::Timeout));
        if let Some(key) = trace_key {
            writeln!(trace, "{key}\t{seed}\t{steps}\t{:?}\t{observation:?}\t{action}\t{}\t{physical:?}\t{}\t{}",
                [before[0], before[1], before[2], before[3]], result.reward, result.done, result.truncated)?;
        }
        if let Some(reason) = reason {
            return Ok(Episode {
                total_return, discounted_return, steps, ending: reason, initial_observation,
                final_state: physical,
                maximum_absolute_state: [maximum_absolute_state[0], maximum_absolute_state[1], maximum_absolute_state[2], maximum_absolute_state[3]],
                absolute_force_sum, near_limit_steps, initial_value,
            });
        }
        observation = result.observation;
    }
    unreachable!("external cap must end the episode")
}

fn emit_episode(writer: &mut impl Write, key: &str, seed: u64, e: &Episode) -> io::Result<()> {
    let initial_value = e.initial_value.map(|x| x.to_string()).unwrap_or_else(|| "none".into());
    writeln!(writer, "{key}\t{seed}\t{}\t{}\t{}\t{:?}\t{}\t{}\t{initial_value}\t{:?}\t{:?}\t{:?}",
        e.total_return, e.discounted_return, e.steps, e.ending, e.absolute_force_sum,
        e.near_limit_steps, e.initial_observation, e.final_state, e.maximum_absolute_state)
}

fn emit_snapshot(writer: &mut impl Write, seed: u64, updates: usize, shared: &PpoSharedState) -> io::Result<()> {
    // Architecture is the unchanged 4x64x64x1 recipe. Hex encodes exact f32 bits;
    // actor metadata precedes layers; each layer is weights followed by biases.
    for (kind, layers, metadata) in [
        ("actor", [&shared.policy.input, &shared.policy.hidden, &shared.policy.output], vec![shared.policy.action_limit, shared.policy.action_std]),
        ("critic", [&shared.value.input, &shared.value.hidden, &shared.value.output], Vec::new()),
    ] {
        let hex = metadata.iter().chain(layers.iter().flat_map(|l| l.weight.iter().chain(&l.bias)))
            .map(|x| format!("{:08x}", x.to_bits())).collect::<String>();
        writeln!(writer, "{seed}\t{updates}\t{kind}\t{hex}")?;
    }
    Ok(())
}

fn output(dir: &Path, name: &str, header: &str) -> io::Result<BufWriter<fs::File>> {
    let mut writer = BufWriter::new(fs::File::create(dir.join(name))?);
    writeln!(writer, "{header}")?;
    Ok(writer)
}

#[test]
#[ignore = "development measurements only; PPO_DIAGNOSTIC_DIR required; use release mode"]
fn measure_default_balancing_development_panel() -> io::Result<()> {
    let directory = std::env::var_os("PPO_DIAGNOSTIC_DIR").expect("PPO_DIAGNOSTIC_DIR required");
    let directory = Path::new(&directory);
    fs::create_dir_all(directory)?;
    let c = evaluation_config();
    let (gain, _, iterations) = lqr_reference(c);
    fs::write(directory.join("lqr.txt"), format!("gain={gain:?}\niterations={iterations}\n"))?;
    fs::write(directory.join("config.txt"), format!("training={:?}\nevaluation={c:?}\nseeds={SEEDS:?}\ncheckpoints={CHECKPOINTS:?}\n", PpoTrainerConfig::default()))?;
    let mut scores = output(directory, "episodes.tsv", "controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\treturn\tdiscounted_return\tsteps\tending\tabs_force_sum\tnear_limit_steps\tinitial_value\tinitial_observation\tfinal_state\tmax_abs_state")?;
    let mut traces = output(directory, "traces.tsv", "controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\tstep\tbefore_state\tobservation\taction\treward\tafter_state\tdone\ttruncated")?;
    let mut snapshots = output(directory, "snapshots.tsv", "training_seed\tupdates\tkind\tf32_hex")?;
    let mut metrics = output(directory, "metrics.tsv", "training_seed\tupdates\tenv_steps\tepisodes\tmean_return\tpolicy_loss\tvalue_loss")?;
    for seed in SEEDS {
        for controller in ["zero", "lqr"] {
            for index in 0..EPISODES {
                let key = format!("{controller}\t{seed}\t0\t{index}");
                let e_seed = episode_seed(seed, index);
                let result = episode(c, e_seed,
                    |obs| if controller == "zero" { 0.0 } else { feedback(gain, obs, c.max_force) },
                    None, &mut traces, (index < 2).then_some(key.as_str()))?;
                emit_episode(&mut scores, &key, e_seed, &result)?;
            }
        }
        let mut session = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
        let mut previous = 0;
        for updates in CHECKPOINTS {
            session.train_updates(updates - previous);
            previous = updates;
            let m = session.metrics();
            assert_eq!(m.total_updates, updates);
            assert_eq!(m.total_env_steps, updates * 512);
            assert!([m.mean_episode_return, m.last_policy_loss, m.last_value_loss].iter().all(|x| x.is_finite()));
            writeln!(metrics, "{seed}\t{updates}\t{}\t{}\t{}\t{}\t{}", m.total_env_steps, m.total_episodes, m.mean_episode_return, m.last_policy_loss, m.last_value_loss)?;
            let shared = session.shared_state();
            check_snapshot(&shared);
            emit_snapshot(&mut snapshots, seed, updates, &shared)?;
            for index in 0..EPISODES {
                let key = format!("ppo\t{seed}\t{updates}\t{index}");
                let e_seed = episode_seed(seed, index);
                let result = episode(c, e_seed, |obs| shared.policy.act(obs), Some(&shared.value),
                    &mut traces, (index < 2).then_some(key.as_str()))?;
                emit_episode(&mut scores, &key, e_seed, &result)?;
            }
            scores.flush()?; traces.flush()?; snapshots.flush()?; metrics.flush()?;
            println!("development seed={seed} updates={updates} transitions={}", updates * 512);
        }
    }
    scores.flush()?; traces.flush()?; snapshots.flush()?; metrics.flush()?;
    Ok(())
}

#[test]
fn failure_labels_respect_strict_limits_and_precedence() {
    let c = evaluation_config();
    assert_eq!(ending([c.max_position_m, 0.0, c.max_angle_rad, 0.0], c, 1), None);
    assert_eq!(ending([3.0, 0.0, 0.0, 0.0], c, CAP), Some(Ending::Position));
    assert_eq!(ending([0.0, 0.0, -0.7, 0.0], c, CAP), Some(Ending::Angle));
    assert_eq!(ending([-3.0, 0.0, 0.7, 0.0], c, CAP), Some(Ending::Both));
    assert_eq!(ending([0.0; 4], c, CAP), Some(Ending::Timeout));
}

#[test]
fn lqr_matches_independent_schur_reference_and_stable_closed_loop() {
    let c = evaluation_config();
    let (gain, p, _) = lqr_reference(c);
    // Independently computed with scipy.linalg.solve_discrete_are before any
    // episode outcomes, using the same float32 A/B/Q/R promoted to float64.
    let reference = [-13.26732451, -19.68876114, 155.46190748, 64.60218293];
    for (actual, expected) in gain.into_iter().zip(reference) {
        assert!((f64::from(actual) - expected).abs() < 2e-5, "independent LQR gain reference");
    }
    let (a, b) = Model::default().model(c.dt);
    let a = a.cast::<f64>(); let b = b.cast::<f64>();
    let k = SMatrix::<f64, 1, 4>::from_row_slice(&gain.map(f64::from));
    let closed = a - b * k;
    assert!(closed.complex_eigenvalues().iter().all(|x| x.norm() < 1.0));
    let q = Matrix::from_diagonal(&Vector::new(c.reward_position_weight.into(), c.reward_velocity_weight.into(), c.reward_angle_weight.into(), c.reward_angular_velocity_weight.into()));
    let residual = p - (q + closed.transpose() * p * closed + k.transpose() * k * f64::from(c.reward_action_weight));
    assert!(residual.amax() / p.amax() < 1e-8, "independent Bellman residual");
    assert_eq!(feedback(gain, [0.0; 4], c.max_force), 0.0);
    assert_eq!(feedback(gain, [0.0, 0.0, 100.0, 0.0], c.max_force), -c.max_force);
}

fn quiet_config() -> PendulumEnvConfig {
    PendulumEnvConfig {
        reset_position_range_m: 0.0, reset_velocity_range_mps: 0.0,
        reset_angle_range_rad: 0.0, reset_angular_velocity_range_radps: 0.0,
        observation_position_noise_m: 0.0, observation_velocity_noise_mps: 0.0,
        observation_angle_noise_rad: 0.0, observation_angular_velocity_noise_radps: 0.0,
        action_noise_force_n: 0.0, disturbance_force_n: 0.0,
        disturbance_probability_per_step: 0.0, max_steps: 5, ..Default::default()
    }
}

#[test]
fn accounting_matches_exact_stationary_rewards_and_timeout() {
    let e = episode(quiet_config(), 1, |_| 0.0, None, &mut io::sink(), None).unwrap();
    assert_eq!(e.steps, 5);
    assert_eq!(e.ending, Ending::Timeout);
    assert_eq!(e.total_return, 5.0, "reward must be counted once");
    let gamma = f64::from(PpoTrainerConfig::default().ppo.gamma);
    let reference = (0..5).map(|i| gamma.powi(i)).sum::<f64>();
    assert!((e.discounted_return - reference).abs() < 1e-14);
    assert_eq!(e.near_limit_steps, 0);
    assert_eq!(e.final_state, [0.0; 4]);
}

#[test]
fn noisy_observation_control_matches_explicit_environment_loop() {
    let c = evaluation_config();
    let seed = episode_seed(201, 0);
    let policy = |obs: [f32; 4]| (12.0 * obs[2] + obs[0]).clamp(-20.0, 20.0);
    let measured = episode(c, seed, policy, None, &mut io::sink(), None).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = PendulumEnv::new_with_rng(Default::default(), c, &mut rng);
    let mut obs = env.observation_with_rng(&mut rng);
    let mut total = 0.0;
    let mut force = 0.0;
    for i in 1..=CAP {
        let action = policy(obs);
        force += f64::from(action.abs());
        let step = env.step_with_rng(action, &mut rng);
        total += f64::from(step.reward);
        if step.done {
            assert_eq!(measured.steps, i);
            assert_eq!(measured.total_return, total, "noisy policy rollout reference");
            assert_eq!(measured.absolute_force_sum, force);
            assert_eq!(measured.final_state, [env.state()[0], env.state()[1], env.state()[2], env.state()[3]]);
            return;
        }
        obs = step.observation;
    }
    panic!("missing end in reference rollout");
}

#[test]
fn episode_streams_and_trace_logging_do_not_change_outcomes() {
    let c = evaluation_config();
    let (gain, _, _) = lqr_reference(c);
    let a = episode(c, 3301, |obs| feedback(gain, obs, 20.0), None, &mut io::sink(), None).unwrap();
    let mut trace = Vec::new();
    let _unrelated = episode(c, 9876, |_| 20.0, None, &mut io::sink(), None).unwrap();
    let b = episode(c, 3301, |obs| feedback(gain, obs, 20.0), None, &mut trace, Some("lqr\t0\t0\t0")).unwrap();
    assert_eq!(a, b, "trace and unrelated episodes must not consume policy randomness");
    assert_eq!(String::from_utf8(trace).unwrap().lines().count(), b.steps);
}

#[test]
fn invalid_policy_actions_fail_before_environment_mutation() {
    for action in [f32::NAN, f32::INFINITY, 21.0] {
        let result = std::panic::catch_unwind(|| {
            episode(quiet_config(), 1, |_| action, None, &mut io::sink(), None).unwrap()
        });
        assert!(result.is_err());
    }
}

#[test]
fn value_snapshot_forward_matches_constant_reference() {
    let session = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 201);
    let mut snapshot = session.shared_state().value;
    snapshot.output.weight.fill(0.0);
    snapshot.output.bias[0] = 7.25;
    assert_eq!(value(&snapshot, [0.2, -0.1, 0.3, 0.0]), 7.25);
}
