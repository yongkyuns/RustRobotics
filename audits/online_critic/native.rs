//! Bounded continuing critic fit. The original PPO update is never replaced.
use super::*;

const ONLINE_PREFIX: usize = 4096;
const ONLINE_END: usize = 4224;
const ONLINE_FIT_STEPS: usize = 48;
const ONLINE_BATCH: usize = 256;
const ONLINE_SELECTED: usize = 512;
const ONLINE_STREAMS: usize = 8;
const ONLINE_EVAL: u64 = 0x7200_0000;
const ONLINE_ARMS: [&str; 3] = ["baseline", "extra-old-batch", "extra-fresh"];

#[derive(Clone, Debug, PartialEq)]
struct ValueData {
    observations: Vec<[f32; 4]>,
    targets: Vec<f32>,
}
impl ValueData {
    fn validate(&self) {
        assert!(!self.targets.is_empty());
        assert_eq!(self.observations.len(), self.targets.len());
        assert!(self.observations.iter().flatten().all(|x| x.is_finite()));
        assert!(self.targets.iter().all(|x| x.is_finite()));
    }
    fn save(&self, path: &Path) {
        self.validate();
        let mut w = BufWriter::new(fs::File::create(path).unwrap());
        for (obs, target) in self.observations.iter().zip(&self.targets) {
            for value in obs.iter().chain(std::iter::once(target)) {
                w.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }
}

/// A terminal stops return propagation even if the next row is a reset episode.
fn online_returns(rewards: &[f32], terminals: &[bool], bootstrap: f32, gamma: f32) -> Vec<f32> {
    assert!(!rewards.is_empty() && rewards.len() == terminals.len());
    assert!(bootstrap.is_finite() && gamma.is_finite() && gamma > 0.0 && gamma <= 1.0);
    assert!(rewards.iter().all(|r| r.is_finite()));
    let mut value = bootstrap;
    let mut result = vec![0.0; rewards.len()];
    for i in (0..rewards.len()).rev() {
        value = if terminals[i] { rewards[i] } else { rewards[i] + gamma * value };
        assert!(value.is_finite());
        result[i] = value;
    }
    result
}

/// Independent starts, no simulator-state cloning and no live RNG consumption.
/// Training labels have a frozen far bootstrap; holdout labels never do.
fn collect_online_data(s: &PpoTrainerSession, id: u64, training: bool, out: Option<&Path>) -> ValueData {
    let before = fingerprint(s);
    let cap = if training { 1536 } else { 2048 };
    let cfg = PendulumEnvConfig { max_steps: cap, ..s.env.config() };
    let law = SquashedGaussian::new(s.config.action_std, s.config.env.max_force);
    let mut data = ValueData { observations: Vec::new(), targets: Vec::new() };
    let _mixed = Mode::enter(true);
    if let Some(path) = out { fs::create_dir_all(path).unwrap(); }
    for stream in 0..ONLINE_STREAMS {
        let key = DATA_DOMAIN ^ (id << 32) ^ stream as u64;
        let mut er = StdRng::seed_from_u64(key);
        let mut ar = StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
        let mut env = PendulumEnv::new_with_rng(s.env.model(), cfg, &mut er);
        let mut obs = env.observation_with_rng(&mut er);
        if let Some((replacement, observation)) = training_start(s, id, stream, cfg) {
            env = replacement;
            obs = observation;
        }
        let initial = array(env.state());
        let mut observations = Vec::with_capacity(cap);
        let mut rewards = Vec::with_capacity(cap);
        let mut terminals = Vec::with_capacity(cap);
        let mut bootstrap = 0.0;
        for t in 0..cap {
            observations.push(obs);
            let mu = actor_means(s, &[obs])[0];
            let sample = law.sample(mu, &mut ar);
            let step = env.step_with_rng(sample.action, &mut er);
            assert!(!step.truncated || t + 1 == cap);
            rewards.push(step.reward);
            terminals.push(step.terminated());
            if t + 1 == cap && training && !step.terminated() {
                bootstrap = predict_value(s, step.observation);
            }
            obs = if step.done { env.reset_with_rng(&mut er) } else { step.observation };
        }
        let returns = online_returns(&rewards, &terminals, bootstrap, s.config.ppo.gamma);
        for t in 0..ONLINE_SELECTED {
            let end = terminals[t..].iter().position(|v| *v).map(|k| t + k);
            assert!(end.is_some() || cap - t > 1024);
        }
        data.observations.extend_from_slice(&observations[..ONLINE_SELECTED]);
        data.targets.extend_from_slice(&returns[..ONLINE_SELECTED]);
        if let Some(path) = out {
            fs::write(path.join(format!("stream-{stream}.json")), format!(
                "{{\"id\":{id},\"stream\":{stream},\"key\":{key},\"training\":{training},\"steps\":{cap},\"selected\":512,\"initial\":{initial:?},\"observations\":{:?},\"rewards\":{rewards:?},\"terminals\":{terminals:?},\"bootstrap\":{bootstrap},\"targets\":{:?}}}\n",
                &observations[..ONLINE_SELECTED], &returns[..ONLINE_SELECTED]
            )).unwrap();
        }
    }
    data.validate();
    assert_eq!(data.targets.len(), ONLINE_STREAMS * ONLINE_SELECTED);
    assert_eq!(before, fingerprint(s), "independent value sampling mutated live state");
    data
}

fn online_values(s: &PpoTrainerSession, observations: &[[f32; 4]]) -> Vec<f32> {
    s.critic.valid().forward(obs_tensor::<Inner>(&s.device, observations))
        .to_data().to_vec::<f32>().unwrap()
}
fn online_mse(s: &PpoTrainerSession, data: &ValueData) -> f64 {
    online_values(s, &data.observations).iter().zip(&data.targets)
        .map(|(v, y)| (f64::from(*v) - f64::from(*y)).powi(2)).sum::<f64>() / data.targets.len() as f64
}

/// Retain live critic parameter IDs/moments. The actor and its optimizer are not touched.
fn fit_online_value(s: &mut PpoTrainerSession, data: &ValueData, rng: &mut StdRng,
                    steps: usize, out: Option<&Path>) -> (f64, f64) {
    data.validate();
    assert_eq!(data.targets.len() % ONLINE_BATCH, 0);
    let batches = data.targets.len() / ONLINE_BATCH;
    assert_eq!(steps % batches, 0);
    let before = fingerprint(s);
    let pre = online_mse(s, data);
    let mut log = out.map(|p| BufWriter::new(fs::File::create(p).unwrap()));
    let mut actual = 0;
    for _ in 0..steps / batches {
        let mut indices = (0..data.targets.len()).collect::<Vec<_>>();
        indices.shuffle(rng);
        for chunk in indices.chunks_exact(ONLINE_BATCH) {
            let x = gather_observations(&data.observations, chunk);
            let y = gather_scalars(&data.targets, chunk);
            let values = s.critic.forward(obs_tensor::<AutodiffBackend>(&s.device, &x));
            let y = scalar_tensor::<AutodiffBackend>(&s.device, &y);
            let loss = (values - y).square().mean();
            let unweighted = tensor_scalar(&loss);
            assert!(unweighted.is_finite());
            let objective = weighted_value_loss(loss, s.config.ppo.value_loss_coef);
            assert!(s.config.ppo.value_loss_coef > 0.0);
            let gradients = GradientsParams::from_grads(objective.backward(), &s.critic);
            s.critic = s.critic_optimizer.step(s.config.ppo.learning_rate, s.critic.clone(), gradients);
            actual += 1;
            if let Some(w) = log.as_mut() {
                writeln!(w, "{{\"step\":{actual},\"indices\":{chunk:?},\"mse\":{unweighted}}}").unwrap();
            }
        }
    }
    if let Some(w) = log.as_mut() { w.flush().unwrap(); }
    assert_eq!(actual, steps);
    let mut after = fingerprint(s);
    after.critic = before.critic.clone();
    assert_eq!(after, before, "extra fitting changed actor, environment, metrics or ordinary RNGs");
    (pre, online_mse(s, data))
}

fn calibration(before: &PpoTrainerSession, after: &PpoTrainerSession, data: &ValueData, path: &Path) {
    assert_eq!(before.snapshot(), after.snapshot());
    let vb = online_values(before, &data.observations);
    let va = online_values(after, &data.observations);
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    writeln!(w, "row,stream,phase,target,before,after").unwrap();
    for i in 0..data.targets.len() {
        writeln!(w, "{i},{},{},{},{},{}", i / ONLINE_SELECTED, i % ONLINE_SELECTED,
                 data.targets[i], vb[i], va[i]).unwrap();
    }
    w.flush().unwrap();
}

fn evaluate_online(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, out: &Path, log: &mut dyn Write) -> usize {
    let before = fingerprint(s);
    let actor = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == ONLINE_END { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut steps = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        let reps = if cp == ONLINE_END && cap == 2048 { 512 } else { 64 };
        for rep in 0..reps {
            let mut trace = (rep == 0 && cp == ONLINE_END).then(||
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(&actor, seed + ONLINE_EVAL, panel, rep,
                            trace.as_mut().map(|w| w as &mut dyn Write));
            writeln!(log, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key, r.steps, r.ending, r.total, r.discounted, r.max_position,
                r.max_angle, r.centered, r.force_rms).unwrap();
            steps += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s));
    steps
}

#[test]
#[ignore = "registered two-history online critic pilot, not production qualification"]
fn emit_online_critic() {
    let seed: u64 = std::env::var("ONLINE_SEED").unwrap().parse().unwrap();
    assert!([41004, 41006].contains(&seed));
    let out = PathBuf::from(std::env::var("ONLINE_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("ONLINE_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let text = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old = text.lines().skip(1).collect::<Vec<_>>();
    assert_eq!(old.len(), ONLINE_END);
    let witness = if seed == 41004 { 4137 } else { 4140 };
    let mut initial = PpoTrainerSession::new_seeded(learning_config("gamma995"), seed);
    compare_weights(&initial, &out, &prior, 0);
    let mut prefix_cost = Costs::default();
    for u in 1..=ONLINE_PREFIX {
        let c = update_support(&mut initial, seed, u, 1024, None);
        prefix_cost.add(c);
        assert_eq!(update_row(&initial, u, c), old[u - 1], "prefix mismatch at {u}");
        if u == 1024 || u == ONLINE_PREFIX { compare_weights(&initial, &out, &prior, u); }
        if u % 512 == 0 { println!("ONLINE PREFIX {seed} {u}"); }
    }
    fs::write(out.join("prefix-costs.json"), cost_json(prefix_cost)).unwrap();
    let initial_fingerprint = fingerprint(&initial);
    let mut first_actor = None;
    let mut first_critic = None;
    let mut first_data = None;
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    let mut eval_steps = 0usize;
    let mut extra_total = 0usize;
    for arm in ONLINE_ARMS {
        let dir = out.join(arm);
        fs::create_dir_all(&dir).unwrap();
        let mut s = clone_session(&initial);
        let mut rng = StdRng::seed_from_u64(0x4f4e_4c43_5249_5455 ^ seed);
        let mut cost = Costs::default();
        let mut log = BufWriter::new(fs::File::create(dir.join("updates.csv")).unwrap());
        writeln!(log, "global,primary,tail,supplement,actor_steps,ordinary_critic_steps,fresh_steps,extra_critic_steps,extra_visits,fit_rows,fit_pre_mse,fit_post_mse,policy_loss,value_loss").unwrap();
        save_state(&s, &dir, ONLINE_PREFIX);
        eval_steps += evaluate_online(&s, seed, arm, ONLINE_PREFIX, &dir, &mut eval);
        for u in ONLINE_PREFIX + 1..=ONLINE_END {
            let local = u - ONLINE_PREFIX;
            let detailed = [1, 32, 128].contains(&local) || u == witness;
            let update_dir = dir.join(format!("update-{u}"));
            fs::create_dir_all(&update_dir).unwrap();
            let ordinary = detailed.then(|| update_dir.join("ordinary"));
            let scope = Scope::enter();
            let c = update_support(&mut s, seed, u, 1024, ordinary.as_deref());
            let captured = scope.finish();
            let batch = captured.batch.unwrap();
            cost.add(c);
            assert_eq!(batch.observations.len(), 1024);
            if arm == "baseline" {
                assert_eq!(update_row(&s, u, c), old[u - 1], "baseline mismatch at {u}");
                if [4128, ONLINE_END, witness - 1, witness].contains(&u) {
                    compare_weights(&s, &dir, &prior, u);
                }
            }
            let before_extra = clone_session(&s);
            let id = stream_id(seed, u);
            let fresh = collect_online_data(&s, id ^ 0x8000_0000, true, Some(&update_dir.join("fresh")));
            if local == 1 {
                let actor = s.snapshot();
                let critic = value_snapshot(&s);
                if arm == "baseline" {
                    first_actor = Some(actor); first_critic = Some(critic); first_data = Some(fresh.clone());
                } else {
                    assert_eq!(Some(actor), first_actor);
                    assert_eq!(Some(critic), first_critic);
                    assert_eq!(Some(&fresh), first_data.as_ref());
                }
            }
            let old_data = ValueData { observations: batch.observations, targets: batch.returns };
            let fitting = if arm == "extra-fresh" { &fresh } else { &old_data };
            fitting.save(&update_dir.join("fit-data.bin"));
            let extra = if arm == "baseline" { 0 } else { ONLINE_FIT_STEPS };
            let (pre, post) = fit_online_value(&mut s, fitting, &mut rng, extra,
                                              Some(&update_dir.join("extra-optimizer.jsonl")));
            extra_total += extra;
            writeln!(log, "{u},{},{},{},16,16,12288,{extra},{},{},{pre},{post},{},{}",
                c.primary, c.tails, c.supplement, extra * ONLINE_BATCH, fitting.targets.len(),
                s.metrics.last_policy_loss, s.metrics.last_value_loss).unwrap();
            if [1, 32, 128].contains(&local) {
                let holdout = collect_online_data(&s, id ^ 0x4000_0000, false, Some(&update_dir.join("holdout")));
                calibration(&before_extra, &s, &holdout, &update_dir.join("calibration.csv"));
                save(&update_dir.join("critic-before-extra.bin"), &value_snapshot(&before_extra));
                save_state(&s, &dir, u);
                eval_steps += evaluate_online(&s, seed, arm, u, &dir, &mut eval);
                eval.flush().unwrap();
            } else if u == witness { save_state(&s, &dir, u); }
            if local % 16 == 0 { log.flush().unwrap(); println!("ONLINE CONTINUATION {seed} {arm} {local}"); }
            assert_eq!(s.metrics.total_updates, u);
            assert_eq!(s.metrics.total_env_steps, u * 512);
        }
        log.flush().unwrap();
        fs::write(dir.join("ordinary-costs.json"), cost_json(cost)).unwrap();
        assert_eq!(fingerprint(&initial), initial_fingerprint);
    }
    eval.flush().unwrap();
    assert_eq!(extra_total, 2 * 128 * ONLINE_FIT_STEPS);
    fs::write(out.join("complete.json"), format!(
        "{{\"seed\":{seed},\"execution\":\"complete\",\"prefix_updates\":4096,\"continuation_per_arm\":128,\"arms\":3,\"extra_critic_steps\":{extra_total},\"fresh_transitions\":4718592,\"calibration_transitions\":147456,\"evaluation_steps\":{eval_steps},\"evaluation_records\":6912,\"history_exact\":true,\"first_update_identical\":true,\"from_scratch_candidate\":false}}\n"
    )).unwrap();
    println!("ONLINE CRITIC COMPLETE {seed}");
}

#[test]
fn online_return_boundaries_and_bootstrap_are_separate() {
    assert_eq!(online_returns(&[1.0, -10.0, 2.0], &[false, true, false], 4.0, 0.5), vec![-4.0, -10.0, 4.0]);
    assert_eq!(online_returns(&[2.0], &[true], 99.0, 0.5), vec![2.0]);
    assert_eq!(online_returns(&[2.0], &[false], 0.0, 0.5), vec![2.0]);
    assert_eq!(online_returns(&[2.0], &[false], 4.0, 0.5), vec![4.0]);
}
fn toy_online_data() -> ValueData {
    ValueData { observations: vec![[0.1, -0.2, 0.05, 0.3]; 256], targets: vec![2.0; 256] }
}
#[test]
fn online_fit_zero_is_exact_and_rejects_invalid_data() {
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), 201);
    let mut rng = StdRng::seed_from_u64(7);
    let before = fingerprint(&s);
    let cursor_before = cursor(&rng);
    fit_online_value(&mut s, &toy_online_data(), &mut rng, 0, None);
    assert_eq!(before, fingerprint(&s)); assert_eq!(cursor_before, cursor(&rng));
    let mut invalid = toy_online_data(); invalid.targets[0] = f32::NAN;
    assert!(std::panic::catch_unwind(|| invalid.validate()).is_err());
}
#[test]
fn online_fit_preserves_actor_optimizer_and_continuing_critic_moments() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 202);
    a.train_updates(1);
    let original = clone_session(&a);
    let mut b = clone_session(&a);
    let data = toy_online_data();
    let mut ra = StdRng::seed_from_u64(9); let mut rb = ra.clone();
    fit_online_value(&mut a, &data, &mut ra, 4, None);
    fit_online_value(&mut b, &data, &mut rb, 2, None);
    b = clone_session(&b);
    fit_online_value(&mut b, &data, &mut rb, 2, None);
    assert_eq!(fingerprint(&a), fingerprint(&b)); assert_eq!(cursor(&ra), cursor(&rb));
    assert_ne!(fingerprint(&a).critic, fingerprint(&original).critic);
    let mut unchanged_actor = clone_session(&original);
    let batch = unchanged_actor.collect_rollout();
    a.config.ppo.value_loss_coef = 0.0;
    unchanged_actor.config.ppo.value_loss_coef = 0.0;
    a.optimize(&batch); unchanged_actor.optimize(&batch);
    assert_eq!(a.snapshot(), unchanged_actor.snapshot());
    assert_eq!(cursor(&a.update_rng), cursor(&unchanged_actor.update_rng));
}
#[test]
fn online_domains_are_distinct_and_observation_collection_is_pure() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"), 203);
    let id = stream_id(203, 4097);
    assert_ne!(id, id ^ 0x8000_0000); assert_ne!(id ^ 0x4000_0000, id ^ 0x8000_0000);
    let before = fingerprint(&s);
    let a = collect_online_data(&s, id ^ 0x8000_0000, true, None);
    let b = collect_online_data(&s, id ^ 0x8000_0000, true, None);
    assert_eq!(a, b); assert_eq!(before, fingerprint(&s));
    let mut altered_critic = clone_session(&s);
    let mut rng = StdRng::seed_from_u64(4);
    fit_online_value(&mut altered_critic, &toy_online_data(), &mut rng, 1, None);
    let hold_a = collect_online_data(&s, id ^ 0x4000_0000, false, None);
    let hold_b = collect_online_data(&altered_critic, id ^ 0x4000_0000, false, None);
    assert_eq!(hold_a, hold_b, "calibration labels must not depend on the tested critic");
}
