//! Test-only sampled-tail intervention, a child of the retained horizon probe.
//! No counterfactual evaluation outcome enters an optimizer.
use super::*;

const TAIL_STEPS: usize = 512;
const TAIL_DRAWS: usize = 4;
const EVAL_REPS: usize = 256;
const TAIL_DOMAIN: u64 = 0x5441_494c_4452_0001;
const FRESH_DOMAIN: u64 = 0x5441_494c_4556_0001;
const VALUE_DOMAIN: u64 = 0x5441_494c_5641_0001;

#[derive(Debug, PartialEq)]
struct Fingerprint {
    actor: Vec<f32>, critic: Vec<f32>, state: [f32; 4], observation: [f32; 4],
    metrics: PpoMetrics, episode_return: f32, recent: Vec<f32>, rngs: [[u64; 4]; 3],
}
fn cursor(rng: &StdRng) -> [u64; 4] {
    let mut copy = rng.clone(); [copy.gen(), copy.gen(), copy.gen(), copy.gen()]
}
fn fingerprint(s: &PpoTrainerSession) -> Fingerprint {
    let x = s.env.state();
    Fingerprint { actor: flat(&s.snapshot()), critic: flat(&value_snapshot(s)),
        state: [x[0], x[1], x[2], x[3]], observation: s.current_observation,
        metrics: s.metrics.clone(), episode_return: s.episode_return,
        recent: s.recent_episode_returns.clone(),
        rngs: [cursor(&s.environment_rng), cursor(&s.action_rng), cursor(&s.update_rng)] }
}
fn predict_value(s: &PpoTrainerSession, obs: [f32; 4]) -> f32 {
    tensor_scalar(&s.critic.valid().forward(obs_tensor::<Inner>(&s.device, &[obs])))
}
#[derive(Clone, Debug)]
struct Tail {
    observations: Vec<[f32; 4]>, next_observations: Vec<[f32; 4]>,
    states: Vec<[f32; 4]>, next_states: Vec<[f32; 4]>,
    means: Vec<f32>, latents: Vec<f32>, commands: Vec<f32>, applied: Vec<f32>,
    rewards: Vec<f32>, terminal: bool, truncated: bool, bootstrap: f32, estimate: f32,
}
fn array(x: Vector4) -> [f32; 4] { [x[0], x[1], x[2], x[3]] }
fn discounted_tail(rewards: &[f32], bootstrap: f32, gamma: f32) -> f32 {
    rewards.iter().rev().fold(bootstrap, |v, r| *r + gamma * v)
}
fn draw_tail(s: &PpoTrainerSession, seed: u64, draw: usize) -> Tail {
    assert!(draw < TAIL_DRAWS);
    let before = fingerprint(s);
    let mut env_rng = if draw == 0 { s.environment_rng.clone() } else {
        StdRng::seed_from_u64(TAIL_DOMAIN ^ (seed << 32) ^ draw as u64)
    };
    let mut action_rng = if draw == 0 { s.action_rng.clone() } else {
        StdRng::seed_from_u64(TAIL_DOMAIN ^ (seed << 32) ^ draw as u64 ^ 0x4143_5449_4f4e_0001)
    };
    // Reset only the external collection clock, not the physical state. This
    // uses a cloneable simulator and does not claim hardware lookahead data.
    let cfg = PendulumEnvConfig { max_steps: TAIL_STEPS, ..s.env.config() };
    let mut env = PendulumEnv::from_state(s.env.model(), cfg, s.env.state(), 0);
    let mut observation = s.current_observation;
    let distribution = SquashedGaussian::new(s.config.action_std, s.config.env.max_force);
    let mut tail = Tail { observations: Vec::new(), next_observations: Vec::new(),
        states: Vec::new(), next_states: Vec::new(), means: Vec::new(), latents: Vec::new(),
        commands: Vec::new(), applied: Vec::new(), rewards: Vec::new(), terminal: false,
        truncated: false, bootstrap: 0.0, estimate: 0.0 };
    for _ in 0..TAIL_STEPS {
        let mu = actor_means(s, &[observation])[0];
        let action = distribution.sample(mu, &mut action_rng);
        tail.states.push(array(env.state())); tail.observations.push(observation);
        let step = env.step_with_rng(action.action, &mut env_rng);
        tail.means.push(mu); tail.latents.push(action.latent); tail.commands.push(action.action);
        tail.applied.push(env.last_applied_force()); tail.rewards.push(step.reward);
        tail.next_states.push(array(env.state())); tail.next_observations.push(step.observation);
        observation = step.observation;
        tail.terminal = step.terminated(); tail.truncated = step.truncated;
        if step.done { break; }
    }
    tail.bootstrap = if tail.terminal { 0.0 } else { predict_value(s, observation) };
    tail.estimate = discounted_tail(&tail.rewards, tail.bootstrap, s.config.ppo.gamma);
    assert!(tail.estimate.is_finite());
    assert!(tail.terminal || (tail.truncated && tail.rewards.len() == TAIL_STEPS));
    assert_eq!(fingerprint(s), before, "tail sampling changed live training state");
    tail
}
fn save_tail(path: &Path, tail: &Tail, draw: usize) {
    fs::write(path, format!(concat!(
        "{{\"draw\":{},\"observations\":{:?},\"next_observations\":{:?},",
        "\"states\":{:?},\"next_states\":{:?},\"means\":{:?},\"latents\":{:?},",
        "\"commands\":{:?},\"applied\":{:?},\"rewards\":{:?},\"terminal\":{},",
        "\"truncated\":{},\"bootstrap\":{:?},\"estimate\":{:?}}}\n"),
        draw, tail.observations, tail.next_observations, tail.states, tail.next_states,
        tail.means, tail.latents, tail.commands, tail.applied, tail.rewards,
        tail.terminal, tail.truncated, tail.bootstrap, tail.estimate)).unwrap();
}
fn substitute(s: &PpoTrainerSession, batch: &RolloutBatch, rows: &[Row], value: f32)
    -> (RolloutBatch, Vec<f32>) {
    let mut replacement = rows.to_vec();
    let end = replacement.last_mut().unwrap();
    assert!(end.end);
    if !end.terminal { end.bootstrap = value; }
    let (returns, raw, _) = targets(s, &replacement, 1.0);
    let mut result = batch.clone(); result.returns = returns; result.advantages = normalize(&raw);
    assert_eq!(result.observations, batch.observations);
    assert_eq!(result.latent_actions, batch.latent_actions);
    assert_eq!(result.old_log_probs, batch.old_log_probs);
    (result, raw)
}
fn save_targets(path: &Path, bootstrap: f32, batch: &RolloutBatch, raw: &[f32]) {
    fs::write(path, format!(
        "{{\"bootstrap\":{bootstrap:?},\"returns\":{:?},\"raw\":{raw:?},\"normalized\":{:?}}}\n",
        batch.returns, batch.advantages)).unwrap();
}
fn selected_trace(out: &Path, label: &str, rep: usize, branch: usize) -> Option<BufWriter<fs::File>> {
    (rep == 0).then(|| BufWriter::new(fs::File::create(
        out.join(format!("trace-{label}-{branch}.csv"))).unwrap()))
}

#[test]
#[ignore = "explicit sampled-tail correction diagnostic, not controller qualification"]
fn emit_tail_support() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((201..=204).contains(&seed));
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    let mut s = PpoTrainerSession::new_seeded(config(), seed);
    for u in 1..8192 { s.train_updates(1); if u % 1024 == 0 { println!("TAIL PREFIX {seed} {u}"); } }
    let (batch, rows) = captured(&mut s);
    let (r1, a1, remaining) = targets(&s, &rows, 1.0);
    let (r95, a95, _) = targets(&s, &rows, 0.95);
    assert_eq!(r1, batch.returns); assert_eq!(normalize(&a1), batch.advantages);
    dump_batch(&out, &rows, &batch, &r95, &a95, &a1, &remaining);
    let old = s.snapshot(); let critic = value_snapshot(&s);
    save(&out.join("actor-before.bin"), &old); save(&out.join("critic-before.bin"), &critic);
    for name in ["batch.json", "actor-before.bin", "critic-before.bin"] {
        assert_eq!(fs::read(out.join(name)).unwrap(), fs::read(prior.join(name)).unwrap(),
            "incoming history mismatch: {name}");
    }
    let anchor = Anchor::new(&s);
    optimize_traced(&mut s, &batch, &out.join("actual1"));
    let actual = s.snapshot(); let actual_critic = value_snapshot(&s);
    save(&out.join("actor-actual1.bin"), &actual);
    save(&out.join("critic-actual1.bin"), &actual_critic);
    for name in ["actor-actual1.bin", "critic-actual1.bin"] {
        assert_eq!(fs::read(out.join(name)).unwrap(), fs::read(prior.join(name)).unwrap(),
            "actual historical endpoint mismatch: {name}");
    }
    let actual_metrics = s.metrics.clone();
    anchor.restore(&mut s);
    let cutoff_state = array(s.env.state()); let cutoff_observation = s.current_observation;
    let old_bootstrap = rows.last().unwrap().bootstrap;
    assert_eq!(old_bootstrap, predict_value(&s, cutoff_observation));
    assert!(!rows.last().unwrap().terminal);
    let untouched = fingerprint(&s);
    let tails: Vec<_> = (0..TAIL_DRAWS).map(|d| draw_tail(&s, seed, d)).collect();
    assert_eq!(fingerprint(&s), untouched);
    for (d, tail) in tails.iter().enumerate() { save_tail(&out.join(format!("tail-{d}.json")), tail, d); }
    let estimates: Vec<_> = tails.iter().map(|t| t.estimate).collect();
    let average = (estimates.iter().map(|v| f64::from(*v)).sum::<f64>() / TAIL_DRAWS as f64) as f32;
    let mut policies = vec![("old".to_string(), old.clone()), ("actual1".to_string(), actual.clone())];
    for (d, value) in estimates.iter().copied().chain(std::iter::once(average)).enumerate() {
        let label = if d < TAIL_DRAWS { format!("tail{d}") } else { "mean4".to_string() };
        anchor.restore(&mut s);
        let (candidate, raw) = substitute(&s, &batch, &rows, value);
        save_targets(&out.join(format!("targets-{label}.json")), value, &candidate, &raw);
        optimize_traced(&mut s, &candidate, &out.join(&label));
        let policy = s.snapshot();
        save(&out.join(format!("actor-{label}.bin")), &policy);
        save(&out.join(format!("critic-{label}.bin")), &value_snapshot(&s));
        policies.push((label, policy));
    }
    anchor.restore(&mut s); s.optimize(&batch);
    assert_eq!(s.snapshot(), actual); assert_eq!(flat(&value_snapshot(&s)), flat(&actual_critic));
    assert_eq!(s.metrics, actual_metrics);
    s.metrics.total_updates += 1;
    assert_eq!(s.metrics.total_env_steps, 4_194_304); assert_eq!(s.metrics.total_updates, 8192);
    let mut fresh = BufWriter::new(fs::File::create(out.join("fresh-reset.csv")).unwrap());
    header(&mut fresh); let mut fresh_steps = 0_u64;
    for rep in 0..EVAL_REPS {
        let trial = Trial { state: None, key: FRESH_DOMAIN ^ (seed << 32) ^ rep as u64,
            remaining: 512, full_policy: true };
        for (branch, (_, policy)) in policies.iter().enumerate() {
            let mut trace = selected_trace(&out, "fresh", rep, branch);
            let result = rollout(&old, policy, &critic, &trial, trace.as_mut().map(|t| t as &mut dyn Write));
            outcome(&mut fresh, "fresh", 0, rep, branch, &trial, &result);
            fresh_steps += result.steps as u64;
        }
    }
    fresh.flush().unwrap();
    let mut historical = BufWriter::new(fs::File::create(out.join("historical-reset.csv")).unwrap());
    header(&mut historical); let mut historical_steps = 0_u64;
    for rep in 0..REPS {
        let trial = Trial { state: None, key: (DOMAIN ^ 0x5245_5345_5400_0000) ^ (seed << 32) ^ rep as u64,
            remaining: 512, full_policy: true };
        for (branch, (_, policy)) in policies[..2].iter().enumerate() {
            let mut trace = selected_trace(&out, "historical", rep, branch);
            let result = rollout(&old, policy, &critic, &trial, trace.as_mut().map(|t| t as &mut dyn Write));
            outcome(&mut historical, "reset", 0, rep, branch, &trial, &result);
            historical_steps += result.steps as u64;
        }
    }
    historical.flush().unwrap();
    let mut reference = BufWriter::new(fs::File::create(out.join("cutoff-reference.csv")).unwrap());
    header(&mut reference); let mut value_steps = 0_u64;
    for rep in 0..EVAL_REPS {
        let trial = Trial { state: Some((cutoff_state, cutoff_observation)),
            key: VALUE_DOMAIN ^ (seed << 32) ^ rep as u64, remaining: 512, full_policy: false };
        let mut trace = selected_trace(&out, "value", rep, 0);
        let result = rollout(&old, &old, &critic, &trial, trace.as_mut().map(|t| t as &mut dyn Write));
        outcome(&mut reference, "value", 0, rep, 0, &trial, &result); value_steps += result.steps as u64;
    }
    reference.flush().unwrap();
    let tail_steps: usize = tails.iter().map(|t| t.rewards.len()).sum();
    let names: Vec<_> = policies.iter().map(|(name, _)| name).collect();
    let text = format!(concat!(
        "{{\"seed\":{seed},\"training_steps\":4194304,\"update\":8192,",
        "\"policies\":{names:?},\"old_bootstrap\":{old_bootstrap:?},\"tail_values\":{estimates:?},",
        "\"mean4\":{average:?},\"cutoff_state\":{cutoff_state:?},\"cutoff_observation\":{cutoff_observation:?},",
        "\"tail_steps\":{tail_steps},\"fresh_steps\":{fresh_steps},\"historical_steps\":{historical_steps},",
        "\"value_steps\":{value_steps},\"prefix_adam_steps_per_network\":131056,",
        "\"target_adam_steps_per_network\":112,\"historical_weights_exact\":true,",
        "\"batch_exact\":true,\"internal_sham_exact\":true}}\n"),
        seed=seed, names=names, old_bootstrap=old_bootstrap, estimates=estimates, average=average,
        cutoff_state=cutoff_state, cutoff_observation=cutoff_observation, tail_steps=tail_steps,
        fresh_steps=fresh_steps, historical_steps=historical_steps, value_steps=value_steps);
    fs::write(out.join("complete.json"), text).unwrap();
    println!("SAMPLED TAIL COMPLETE {seed}");
}

#[test]
fn unchanged_bootstrap_is_an_exact_sham_target() {
    let mut s = PpoTrainerSession::new_seeded(config(), 201);
    let (b, rows) = captured(&mut s);
    let (copy, _) = substitute(&s, &b, &rows, rows.last().unwrap().bootstrap);
    assert_eq!(copy.returns, b.returns); assert_eq!(copy.advantages, b.advantages);
}
#[test]
fn copied_stream_tail_matches_ordinary_future_until_collection_cutoff() {
    let s = PpoTrainerSession::new_seeded(config(), 202);
    let tail = draw_tail(&s, 202, 0);
    let mut env = s.env.clone(); let mut er = s.environment_rng.clone();
    let mut ar = s.action_rng.clone(); let mut obs = s.current_observation;
    let distribution = SquashedGaussian::new(s.config.action_std, s.config.env.max_force);
    for i in 0..tail.rewards.len() {
        let sampled = distribution.sample(actor_means(&s, &[obs])[0], &mut ar);
        let step = env.step_with_rng(sampled.action, &mut er);
        assert_eq!(sampled.latent, tail.latents[i]); assert_eq!(step.reward, tail.rewards[i]);
        assert_eq!(step.observation, tail.next_observations[i]);
        assert_eq!(array(env.state()), tail.next_states[i]); obs = step.observation;
    }
}
#[test]
fn terminal_tail_has_no_value_bootstrap() {
    let mut s = PpoTrainerSession::new_seeded(config(), 203);
    s.env = PendulumEnv::from_state(s.env.model(), s.env.config(), Vector4::new(2.5, 0.0, 0.0, 0.0), 0);
    s.current_observation = [2.5, 0.0, 0.0, 0.0];
    let tail = draw_tail(&s, 203, 0);
    assert!(tail.terminal && !tail.truncated); assert_eq!(tail.rewards, vec![-10.0]);
    assert_eq!(tail.bootstrap, 0.0); assert_eq!(tail.estimate, -10.0);
}
#[test]
fn tail_return_keeps_cutoff_bootstrap_and_does_not_reset_terminal_path() {
    let g = 0.99_f32;
    assert_eq!(discounted_tail(&[1.0, 2.0], 5.0, g), 1.0 + g * (2.0 + g * 5.0));
    assert_eq!(discounted_tail(&[1.0, -10.0], 0.0, g), 1.0 + g * -10.0);
    let s = PpoTrainerSession::new_seeded(config(), 201);
    let a = draw_tail(&s, 201, 1); let b = draw_tail(&s, 201, 1);
    assert_eq!(a.rewards, b.rewards); assert_eq!(a.estimate, b.estimate);
    if a.truncated { assert_eq!(a.rewards.len(), 512); assert_eq!(a.bootstrap,
        predict_value(&s, *a.next_observations.last().unwrap())); }
}
