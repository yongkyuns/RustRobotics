// C1 development tests appended to the hash-verified attribution module.
// Every update runs PpoTrainerSession::train_updates; no replacement optimizer.

fn c1_arm(session: &mut PpoTrainerSession, arm: &str) {
    assert!(matches!(arm, "baseline" | "coverage" | "returns" | "joint"));
    session.minimum_rollout_episodes = if matches!(arm, "coverage" | "joint") { 16 } else { 0 };
    session.config.ppo.gae_lambda = if matches!(arm, "returns" | "joint") { 1.0 } else { 0.95 };
}

fn c1_seed() -> u64 {
    let seed = std::env::var("C1_SEED").unwrap().parse().unwrap();
    assert!((201..=204).contains(&seed));
    seed
}

fn c1_verify_prior(name: &str, actual: &Path) {
    let prior = PathBuf::from(std::env::var_os("C1_PRIOR_SEED").unwrap());
    assert_eq!(fs::read(actual).unwrap(), fs::read(prior.join(name)).unwrap(), "original witness bytes: {name}");
}

fn c1_finite(session: &PpoTrainerSession) {
    let state = session.shared_state();
    for p in [parameters(&state.policy.input, &state.policy.hidden, &state.policy.output),
        parameters(&state.value.input, &state.value.hidden, &state.value.output)] {
        assert!(p.iter().all(|x| x.is_finite()));
    }
    assert_eq!(state.policy.action_limit, 20.0);
    assert_eq!(state.policy.action_std, 2.0);
    let m = session.metrics();
    assert!([m.last_policy_loss, m.last_value_loss, m.last_mean_advantage,
        m.last_episode_return, m.mean_episode_return, m.best_episode_return]
        .iter().all(|x| x.is_finite()));
}

fn c1_evaluate(policy: &PolicySnapshot, seed: u64, checkpoint: usize, writer: &mut impl Write) {
    let fast = Fast::policy(policy);
    for (mode, count, horizon, domain) in [
        ("deterministic", 32, 1000, 0xC135_1000_0000_0000_u64),
        ("stochastic", 64, 1500, 0xC135_2000_0000_0000_u64),
    ] {
        for i in 0..count {
            // Deliberately the same episode innovations across arms/checkpoints.
            // None of these streams is used by training.
            let key = domain + seed * 0x100000 + i as u64;
            let mut erng = StdRng::seed_from_u64(key ^ 0x126A_B672_37CD_58EB);
            let mut arng = StdRng::seed_from_u64(key ^ 0xA761_ABC3_D299_74E5);
            let mut env = PendulumEnv::new_with_rng(Default::default(), PendulumEnvConfig {
                max_steps: horizon, ..Default::default()
            }, &mut erng);
            let mut obs = env.observation_with_rng(&mut erng);
            let mut sum = 0.0_f64;
            let mut discounted = 0.0_f64;
            let mut discount = 1.0_f64;
            for steps in 1..=horizon {
                let action = if mode == "deterministic" {
                    policy.act(obs)
                } else {
                    SquashedGaussian::new(policy.action_std, policy.action_limit)
                        .sample(fast.forward(obs), &mut arng).action
                };
                assert!(action.is_finite() && action.abs() <= policy.action_limit);
                let result = env.step_with_rng(action, &mut erng);
                assert!(result.reward.is_finite() && result.observation.iter().all(|v| v.is_finite()));
                sum += f64::from(result.reward);
                discounted += discount * f64::from(result.reward);
                discount *= f64::from(0.99_f32);
                obs = result.observation;
                if result.done {
                    let state = env.state();
                    let angle = state[2].abs() > env.config().max_angle_rad;
                    let position = state[0].abs() > env.config().max_position_m;
                    let ending = match (angle, position) {
                        (true, true) => "angle_position", (true, false) => "angle",
                        (false, true) => "position", (false, false) => "timeout",
                    };
                    assert_eq!(result.truncated, !angle && !position);
                    assert!(angle || position || steps == horizon);
                    writeln!(writer, "{checkpoint}\t{mode}\t{i}\t{key}\t{sum}\t{discounted}\t{steps}\t{ending}\t{}\t{}\t{}\t{}", state[0], state[1], state[2], state[3]).unwrap();
                    break;
                }
                assert!(steps < horizon, "evaluation must close at its external cap");
            }
        }
    }
}

#[test]
#[ignore = "explicit, fixed-budget C1 development comparison"]
fn c1_from_scratch() {
    let seed = c1_seed();
    let arm = std::env::var("C1_ARM").unwrap();
    let mut session = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
    c1_arm(&mut session, &arm);
    let dir = root().join("from-scratch");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.txt"), format!("arm={arm}\nseed={seed}\nminimum_episodes={}\n{:?}\n", session.minimum_rollout_episodes, session.config())).unwrap();
    let mut episodes = BufWriter::new(fs::File::create(dir.join("episodes.tsv")).unwrap());
    writeln!(episodes, "checkpoint\tmode\tepisode\tseed\treturn\tdiscounted\tsteps\tending\tx\tv\ttheta\tomega").unwrap();
    let mut metrics = BufWriter::new(fs::File::create(dir.join("checkpoints.tsv")).unwrap());
    writeln!(metrics, "checkpoint\tupdates\tsteps\tepisodes\tminibatches\ttraining_seconds").unwrap();
    let mut updates = BufWriter::new(fs::File::create(dir.join("updates.tsv")).unwrap());
    writeln!(updates, "update\tsteps\tbatch_steps\tepisodes\tbatch_episodes\tminibatches\tpolicy_loss\tvalue_loss\tseconds").unwrap();
    let mut minibatches = 0;
    let mut seconds = 0.0;
    for checkpoint in [0, 65_536, 262_144, 1_048_576] {
        while session.metrics().total_env_steps < checkpoint {
            let old_steps = session.metrics().total_env_steps;
            let old_episodes = session.metrics().total_episodes;
            let start = std::time::Instant::now();
            session.train_updates(1);
            let elapsed = start.elapsed().as_secs_f64();
            seconds += elapsed;
            c1_finite(&session);
            let m = session.metrics();
            let n = m.total_env_steps - old_steps;
            let e = m.total_episodes - old_episodes;
            assert!(n >= 512);
            assert!(n <= 80_000, "declared per-update upper bound");
            if session.minimum_rollout_episodes > 0 {
                assert!(e >= 16);
                assert_eq!(session.episode_return, 0.0, "complete episode boundary");
            } else {
                assert_eq!(n, 512);
            }
            let batches = n.div_ceil(128) * 4;
            minibatches += batches;
            writeln!(updates, "{}\t{}\t{n}\t{}\t{e}\t{batches}\t{}\t{}\t{elapsed}", m.total_updates, m.total_env_steps, m.total_episodes, m.last_policy_loss, m.last_value_loss).unwrap();
        }
        let m = session.metrics();
        let p = dir.join(format!("actor-{checkpoint}.bin"));
        let v = dir.join(format!("critic-{checkpoint}.bin"));
        save_policy(&p, &session.snapshot());
        save_value(&v, &session.shared_state().value);
        if arm == "baseline" && checkpoint == 262_144 {
            c1_verify_prior("actor-512.bin", &p);
            c1_verify_prior("critic-512.bin", &v);
        }
        writeln!(metrics, "{checkpoint}\t{}\t{}\t{}\t{minibatches}\t{seconds}", m.total_updates, m.total_env_steps, m.total_episodes).unwrap();
        c1_evaluate(&session.snapshot(), seed, checkpoint, &mut episodes);
        episodes.flush().unwrap(); metrics.flush().unwrap(); updates.flush().unwrap();
        println!("C1 checkpoint seed={seed} arm={arm} target={checkpoint} steps={} updates={}", m.total_env_steps, m.total_updates);
    }
    println!("C1 FROM SCRATCH COMPLETE seed={seed} arm={arm}");
}

#[test]
#[ignore = "explicit frozen-update C1 development witness"]
fn c1_frozen_witness() {
    let seed = c1_seed();
    let selected = match seed { 201 => 516, 202 => 537, 203 => 514, 204 => 526, _ => unreachable!() };
    let dir = root().join("witness"); fs::create_dir_all(&dir).unwrap();
    let mut baseline = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
    baseline.train_updates(selected - 1);
    let old = baseline.snapshot();
    let critic = baseline.shared_state().value;
    save_policy(dir.join("old-actor.bin"), &old);
    save_value(dir.join("old-critic.bin"), &critic);
    c1_verify_prior("selected/old-actor.bin", &dir.join("old-actor.bin"));
    c1_verify_prior("selected/old-critic.bin", &dir.join("old-critic.bin"));
    CAPTURE.with(|v| { assert!(v.borrow().is_none()); *v.borrow_mut() = Some(Vec::new()); });
    baseline.train_updates(1);
    let rows = CAPTURE.with(|v| v.borrow_mut().take().unwrap());
    assert_eq!(rows.len(), 512);
    let baseline_new = baseline.snapshot();
    save_policy(dir.join("baseline-new.bin"), &baseline_new);
    c1_verify_prior("selected/new-actor.bin", &dir.join("baseline-new.bin"));
    let mut candidate = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
    candidate.train_updates(selected - 1);
    assert_eq!(format!("{:?}", candidate.shared_state()), format!("{:?}", baseline_shared(&old, &critic)));
    c1_arm(&mut candidate, "joint");
    let previous_steps = candidate.metrics().total_env_steps;
    let previous_episodes = candidate.metrics().total_episodes;
    candidate.train_updates(1);
    c1_finite(&candidate);
    let new = candidate.snapshot();
    save_policy(dir.join("joint-new.bin"), &new);
    save_value(dir.join("joint-new-critic.bin"), &candidate.shared_state().value);
    fs::write(dir.join("cost.txt"), format!("seed={seed}\nselected={selected}\nbaseline_prefix_updates={}\nprefix_replays=2\nbaseline_update_steps=512\njoint_update_steps={}\njoint_update_episodes={}\n", selected-1, candidate.metrics().total_env_steps-previous_steps, candidate.metrics().total_episodes-previous_episodes)).unwrap();
    let mut resets = BufWriter::new(fs::File::create(dir.join("resets.tsv")).unwrap());
    outcome_header(&mut resets);
    for (label, p) in [("old", &old), ("baseline", &baseline_new), ("joint", &new)] {
        for (i, r) in reset_panel(p, seed, 0xC135_3000_0000_0000, 256).into_iter().enumerate() {
            outcome_line(&mut resets, &format!("{label}/{i}"), r);
        }
    }
    let mut branches = BufWriter::new(fs::File::create(dir.join("branches.tsv")).unwrap());
    outcome_header(&mut branches);
    let f_old = Fast::policy(&old); let f_base = Fast::policy(&baseline_new); let f_new = Fast::policy(&new);
    for index in (0..512).step_by(16) {
        let row = &rows[index];
        for draw in 0..32 {
            let key = 0xC135_4000_0000_0000 + seed * 0x100000 + index as u64 * 64 + draw;
            for (label, first, follow) in [("old", &f_old, &f_old), ("baseline-first", &f_base, &f_old),
                ("joint-first", &f_new, &f_old), ("baseline-full", &f_base, &f_base), ("joint-full", &f_new, &f_new)] {
                let r = simulate(&row.env, row.obs, key, Spec { first, follow, critic: None,
                    sigma: old.action_std / 20.0, gae_len: 0, forced_latent: None }, None);
                outcome_line(&mut branches, &format!("{label}/{index}/{draw}"), r);
            }
        }
    }
    resets.flush().unwrap(); branches.flush().unwrap();
    println!("C1 FROZEN WITNESS COMPLETE seed={seed} update={selected}");
}

fn baseline_shared(policy: &PolicySnapshot, value: &ValueSnapshot) -> PpoSharedState {
    PpoSharedState { policy: policy.clone(), value: value.clone() }
}

fn c1_fixture(terminal: bool, rollout: usize, minimum: usize, critic_value: f32) -> PpoTrainerSession {
    let config = PpoTrainerConfig {
        ppo: PpoConfig { rollout_steps: rollout, gamma: 0.75, gae_lambda: 1.0, ..Default::default() },
        env: PendulumEnvConfig {
            max_steps: 2, max_angle_rad: if terminal { 0.0 } else { 1e6 }, max_position_m: 1e6,
            reset_position_range_m: 0.0, reset_velocity_range_mps: 0.0,
            reset_angle_range_rad: 0.0, reset_angular_velocity_range_radps: 0.0,
            observation_position_noise_m: 0.0, observation_velocity_noise_mps: 0.0,
            observation_angle_noise_rad: 0.0, observation_angular_velocity_noise_radps: 0.0,
            action_noise_force_n: 0.0, disturbance_probability_per_step: 0.0,
            reward_position_weight: 0.0, reward_velocity_weight: 0.0,
            reward_angle_weight: 0.0, reward_angular_velocity_weight: 0.0, reward_action_weight: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut session = PpoTrainerSession::new_seeded(config, 135);
    let mut state = session.shared_state();
    for layer in [&mut state.policy.input, &mut state.policy.hidden, &mut state.policy.output,
        &mut state.value.input, &mut state.value.hidden, &mut state.value.output] {
        layer.weight.fill(0.0); layer.bias.fill(0.0);
    }
    state.policy.output.bias[0] = 1.0; state.value.output.bias[0] = critic_value;
    session.load_shared_state(&state);
    session.minimum_rollout_episodes = minimum;
    session
}

#[test]
fn c1_episode_floor_and_transition_floor_are_both_required() {
    for (floor, minimum, expected) in [(5, 3, 6), (9, 3, 10), (0, 3, 6), (5, 0, 5), (0, 0, 0)] {
        let mut s = c1_fixture(false, floor, minimum, 5.0);
        let (batch, rows) = collect(&mut s);
        assert_eq!(batch.observations.len(), expected);
        assert_eq!(rows.iter().filter(|r| r.done).count(), expected / 2);
        assert_eq!(s.metrics().total_env_steps, expected);
        if minimum > 0 { assert!(rows.last().unwrap().done); assert_eq!(s.episode_return, 0.0); }
    }
}

#[test]
fn c1_true_terminal_reward_to_go_ignores_future_critic_values() {
    for value in [0.0, 5.0, 1000.0] {
        let mut s = c1_fixture(true, 5, 3, value);
        let (batch, rows) = collect(&mut s);
        assert_eq!(batch.returns.len(), 6);
        assert!(rows.iter().filter(|r| r.done).all(|r| r.terminated));
        for (i, r) in batch.returns.iter().enumerate() {
            let expected = if i % 2 == 0 { -6.5 } else { -10.0 };
            assert!((r-expected).abs() < 0.0002, "reward-to-go {r} != {expected}");
        }
    }
}

#[test]
fn c1_timeouts_still_bootstrap_before_reset() {
    let mut s = c1_fixture(false, 5, 3, 5.0);
    let (batch, rows) = collect(&mut s);
    assert!(rows.iter().all(|r| !r.terminated));
    for (i, r) in batch.returns.iter().enumerate() {
        let expected = if i % 2 == 0 { 4.5625 } else { 4.75 };
        assert!((r-expected).abs() < 0.0002);
    }
}

#[test]
fn c1_complete_collection_replays_and_keeps_optimizer_stream_separate() {
    let mut a = c1_fixture(false, 5, 3, 5.0);
    let mut b = c1_fixture(false, 5, 3, 5.0);
    for _ in 0..101 { let _: u64 = b.update_rng.gen(); }
    let (x, xr) = collect(&mut a); let (y, yr) = collect(&mut b);
    assert_eq!(x.observations, y.observations); assert_eq!(x.latent_actions, y.latent_actions);
    assert_eq!(x.old_log_probs, y.old_log_probs); assert_eq!(x.returns, y.returns);
    assert_eq!(x.advantages, y.advantages); assert_eq!(xr.len(), yr.len());
    assert_eq!(a.current_observation, b.current_observation);
}
