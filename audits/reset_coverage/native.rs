//! Matched, from-random-initialization actor/critic coverage comparison.
use super::*;
use crate::trainer::reset_coverage_hooks::{self as hooks, Receipt, Scope};

const COVERAGE_SEEDS: [u64; 4] = [41001, 41002, 41004, 41006];
const COVERAGE_ARMS: [&str; 2] = ["extra-ordinary", "extra-transient"];
const COVERAGE_UPDATES: usize = 4608;
const COVERAGE_CHECKPOINTS: [usize; 4] = [0, 1024, 4096, 4608];
const ADDED_STREAMS: usize = 4;
const SELECTED: usize = 128;
const LOOKAHEAD: usize = 1024;
const ADDED_STEPS: usize = SELECTED + LOOKAHEAD;
const TRAIN_DOMAIN: u64 = 0x5243_5452_4149_0001;
const COVERAGE_EVAL: u64 = 0x7800_0000;
const STRESS_OFFSET: u64 = 0x0080_0000;

fn training_key(seed: u64, update: usize, stream: usize) -> u64 {
    assert!(COVERAGE_SEEDS.contains(&seed));
    assert!((1..=COVERAGE_UPDATES).contains(&update) && stream < ADDED_STREAMS);
    TRAIN_DOMAIN ^ (seed << 32) ^ ((update as u64) << 8) ^ stream as u64
}

fn collect_added(s: &PpoTrainerSession, seed: u64, update: usize, stress: bool, out: Option<&Path>) -> Supplemental {
    let before = fingerprint(s);
    let cfg = PendulumEnvConfig { max_steps: ADDED_STEPS, ..s.env.config() };
    let law = SquashedGaussian::new(s.config.action_std, cfg.max_force);
    let mut result = Supplemental { batch: empty_batch(), raw: Vec::new(), remaining: Vec::new(), steps: 0, terminals: 0 };
    if let Some(out) = out { fs::create_dir_all(out).unwrap(); }
    for stream in 0..ADDED_STREAMS {
        let key = training_key(seed, update, stream);
        let mut er = StdRng::seed_from_u64(key);
        let mut ar = StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
        let mut sr = StdRng::seed_from_u64(key ^ 0x5354_4154_4500_0001);
        let mut env = if stress {
            PendulumEnv::from_state(s.env.model(), cfg, Vector4::from_column_slice(&hooks::stress_state(cfg, &mut sr)), 0)
        } else { PendulumEnv::new_with_rng(s.env.model(), cfg, &mut sr) };
        let mut obs = env.observation_with_rng(&mut sr);
        let initial = array(env.state());
        let mut rows = Vec::with_capacity(ADDED_STEPS);
        let mut batch = empty_batch();
        for t in 0..ADDED_STEPS {
            let state = array(env.state());
            let mu = actor_means(s, &[obs])[0];
            let value = predict_value(s, obs);
            let sample = law.sample(mu, &mut ar);
            let step = env.step_with_rng(sample.action, &mut er);
            assert!(!step.truncated || t + 1 == ADDED_STEPS);
            let end = step.done || t + 1 == ADDED_STEPS;
            let bootstrap = if end && !step.terminated() { predict_value(s, step.observation) } else { 0.0 };
            rows.push(Row { state, observation: obs, mean: mu, value, latent: sample.latent,
                reward: step.reward, terminal: step.terminated(), timeout: step.truncated, end,
                final_observation: step.observation, bootstrap });
            batch.observations.push(obs);
            batch.latent_actions.push(sample.latent);
            batch.old_log_probs.push(sample.log_prob);
            result.steps += 1;
            if step.terminated() { result.terminals += 1; }
            // Deliberately ordinary resets after failure, in BOTH arms.
            obs = if step.done { env.reset_with_rng(&mut er) } else { step.observation };
        }
        let (returns, raw, remaining) = targets(s, &rows, 1.0);
        batch.returns = returns;
        batch.advantages = normalize(&raw);
        for (t, left) in remaining.iter().copied().enumerate().take(SELECTED) {
            assert!(left > LOOKAHEAD || rows[t + left - 1].terminal);
        }
        add_rows(&mut result.batch, &batch, SELECTED);
        result.raw.extend_from_slice(&raw[..SELECTED]);
        result.remaining.extend_from_slice(&remaining[..SELECTED]);
        if let Some(out) = out {
            let path = out.join(format!("stream-{stream}"));
            fs::create_dir_all(&path).unwrap();
            let (r95, a95, _) = targets(s, &rows, 0.95);
            dump_batch(&path, &rows, &batch, &r95, &a95, &raw, &remaining);
            fs::write(path.join("identity.json"), format!("{{\"seed\":{seed},\"update\":{update},\"stream\":{stream},\"key\":{key},\"stress\":{stress},\"initial\":{initial:?},\"selected\":128,\"steps\":1152}}\n")).unwrap();
        }
    }
    result.batch.advantages = normalize(&result.raw);
    assert_eq!(result.raw.len(), 512);
    assert_eq!(result.steps, 4608);
    assert_eq!(before, fingerprint(s), "new collection mutated the live learner");
    result
}

fn check_prefix(r: &Receipt) {
    let n = r.original_raw.len();
    assert_eq!(n, 1024);
    assert_eq!(&r.training.observations[..n], &r.original.observations);
    for (whole, original) in [(&r.training.latent_actions, &r.original.latent_actions),
        (&r.training.old_log_probs, &r.original.old_log_probs), (&r.training.returns, &r.original.returns),
        (&r.raw, &r.original_raw)] {
        assert!(whole[..n].iter().zip(original).all(|(a,b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(r.training.advantages, normalize(&r.raw));
}
fn coverage_update(s: &mut PpoTrainerSession, seed: u64, update: usize, stress: bool, out: Option<&Path>) -> Costs {
    let added_path = out.map(|p| p.join("added"));
    let extra = collect_added(s, seed, update, stress, added_path.as_deref());
    let scope = Scope::enter(Some((extra.batch, extra.raw)));
    let cost = update_support(s, seed, update, LOOKAHEAD, out);
    let receipt = scope.finish();
    check_prefix(&receipt);
    assert_eq!(cost.actor_steps, 24);
    assert_eq!(cost.sample_visits, 6144);
    if let Some(out) = out {
        save_union(&out.join("original-union.json"), &receipt.original, &receipt.original_raw);
        save_union(&out.join("augmented-union.json"), &receipt.training, &receipt.raw);
    }
    cost
}

fn coverage_evaluate(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, out: &Path, w: &mut dyn Write) -> usize {
    let before = fingerprint(s);
    let actor = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch", "stress-det", "stress-stoch"];
    if cp == COVERAGE_UPDATES { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut count = 0;
    for panel in panels {
        let stress = panel.starts_with("stress-");
        let native_panel = if panel == "stress-det" { "reset-det" } else if panel == "stress-stoch" { "reset-stoch" } else { panel };
        let (_, _, cap, _) = panel_spec(native_panel);
        let reps = if cap > 2048 { 32 } else if cp == COVERAGE_UPDATES { 256 } else { 64 };
        let _stress_scope = stress.then(hooks::Evaluation::enter);
        for rep in 0..reps {
            let mut trace = (cp == COVERAGE_UPDATES && rep == 0).then(|| BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(&actor, seed + COVERAGE_EVAL + if stress { STRESS_OFFSET } else { 0 }, native_panel, rep, trace.as_mut().map(|x| x as &mut dyn Write));
            writeln!(w, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}", r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
            count += r.steps;
        }
    }
    assert_eq!(fingerprint(s), before);
    count
}

#[test]
#[ignore = "registered eight-run from-scratch development experiment"]
fn emit_reset_coverage() {
    let seed: u64 = std::env::var("COVERAGE_SEED").unwrap().parse().unwrap();
    let arm = std::env::var("COVERAGE_ARM").unwrap();
    assert!(COVERAGE_SEEDS.contains(&seed) && COVERAGE_ARMS.contains(&arm.as_str()));
    let stress = arm == "extra-transient";
    let out = PathBuf::from(std::env::var("COVERAGE_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), seed);
    assert_eq!(s.metrics.total_updates, 0);
    let mut costs = Costs::default();
    fs::write(out.join("config.json"), format!("{{\"protocol\":5816968405,\"seed\":{seed},\"arm\":\"{arm}\",\"from_scratch\":true,\"updates\":4608,\"training_gamma\":0.995,\"lambda\":1,\"rows\":1536,\"minibatch\":256,\"epochs\":4,\"added_streams\":4,\"selected\":128,\"stream_steps\":1152,\"critic_only_steps\":0}}\n")).unwrap();
    let mut log = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(log,"update,primary,tail,supplement,added,actor_steps,critic_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    save_state(&s, &out, 0);
    let mut evaluation_steps = coverage_evaluate(&s, seed, &arm, 0, &out, &mut eval);
    for u in 1..=COVERAGE_UPDATES {
        let path = [1,4097,COVERAGE_UPDATES].contains(&u).then(|| out.join(format!("update-{u}")));
        if let Some(p) = &path { fs::create_dir_all(p).unwrap(); save_state(&s,p,0); }
        let cost = coverage_update(&mut s, seed, u, stress, path.as_deref());
        assert_eq!(s.metrics.total_updates,u);
        assert_eq!(s.metrics.total_env_steps,u*512);
        assert_eq!(cost.primary,512);
        assert_eq!(cost.supplement,8704);
        assert!(cost.tails <= 4096);
        costs.add(cost);
        writeln!(log,"{u},{},{},{},4608,24,24,6144,{},{},{}",cost.primary,cost.tails,cost.supplement,s.metrics.total_episodes,s.metrics.last_policy_loss,s.metrics.last_value_loss).unwrap();
        if u.is_multiple_of(256) { log.flush().unwrap(); println!("COVERAGE PROGRESS {seed} {arm} {u}"); }
        if COVERAGE_CHECKPOINTS.contains(&u) {
            save_state(&s,&out,u);
            evaluation_steps += coverage_evaluate(&s,seed,&arm,u,&out,&mut eval);
            eval.flush().unwrap();
        }
    }
    log.flush().unwrap(); eval.flush().unwrap();
    assert_eq!(costs.actor_steps,110592);
    assert_eq!(costs.sample_visits,28311552);
    fs::write(out.join("costs.json"), cost_json(costs)).unwrap();
    fs::write(out.join("complete.json"),format!("{{\"execution\":\"complete\",\"protocol\":5816968405,\"seed\":{seed},\"arm\":\"{arm}\",\"from_scratch\":true,\"updates\":4608,\"added_transitions\":21233664,\"actor_steps\":110592,\"critic_steps\":110592,\"sample_visits\":28311552,\"evaluation_records\":2336,\"evaluation_steps\":{evaluation_steps}}}\n")).unwrap();
    println!("RESET COVERAGE COMPLETE {seed} {arm}");
}

#[test]
fn reset_coverage_stress_bounds_and_domains_are_explicit() {
    let cfg = PendulumEnvConfig::default();
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..128 {
        let a = hooks::stress_state(cfg, &mut rng);
        assert!(a[0].abs() < 0.2 && a[1].abs() < 0.4);
        assert!((0.125..0.25).contains(&a[2].abs()));
        assert!((0.25..0.5).contains(&a[3].abs()) && a[2]*a[3] > 0.0);
    }
    assert_ne!(training_key(41004,1,0),training_key(41004,2,0));
    assert_ne!(training_key(41004,1,0),training_key(41006,1,0));
}
#[test]
fn reset_coverage_disabled_scope_preserves_update_and_next_optimizer() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"),41001);
    let mut b = clone_session(&a);
    let ca = update_support(&mut a,41001,1,1024,None);
    let scope = Scope::enter(None);
    let cb = update_support(&mut b,41001,1,1024,None);
    let receipt = scope.finish();
    check_prefix(&receipt);
    assert_eq!(ca,cb); assert_eq!(fingerprint(&a),fingerprint(&b));
    a.train_updates(1); b.train_updates(1);
    assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn reset_coverage_added_collection_is_pure_reproducible_and_supported() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"),41002);
    let before = fingerprint(&s);
    for stress in [false,true] {
        let a = collect_added(&s,41002,1,stress,None);
        let b = collect_added(&s,41002,1,stress,None);
        assert_eq!(a.batch.observations,b.batch.observations);
        assert_eq!(a.batch.latent_actions,b.batch.latent_actions);
        assert_eq!(a.batch.returns,b.batch.returns); assert_eq!(a.raw,b.raw);
        assert_eq!(a.steps,4608); assert_eq!(a.remaining.len(),512);
    }
    assert_eq!(before,fingerprint(&s));
}
#[test]
fn reset_coverage_prefix_and_manual_replay_preserve_live_adam() {
    let initial = PpoTrainerSession::new_seeded(learning_config("gamma995"),41004);
    let extra = collect_added(&initial,41004,1,true,None);
    let mut original = clone_session(&initial);
    let plain_scope = Scope::enter(None);
    update_support(&mut original,41004,1,1024,None);
    let plain = plain_scope.finish();
    let mut actual = clone_session(&initial);
    let scope = Scope::enter(Some((extra.batch,extra.raw)));
    let costs = update_support(&mut actual,41004,1,1024,None);
    let r = scope.finish(); check_prefix(&r);
    assert_eq!(r.original.observations,plain.original.observations);
    assert_eq!(r.original_raw,plain.original_raw);
    assert_eq!(r.training.observations.len(),1536);assert_eq!(costs.actor_steps,24);
    let mut manual = clone_session(&actual);
    manual.actor=initial.actor.clone();manual.critic=initial.critic.clone();
    manual.actor_optimizer=initial.actor_optimizer.clone();manual.critic_optimizer=initial.critic_optimizer.clone();
    manual.update_rng=initial.update_rng.clone();manual.config.ppo.mini_batch_size=256;
    manual.optimize(&r.training);manual.config.ppo.mini_batch_size=128;
    assert_eq!(fingerprint(&actual),fingerprint(&manual));
    actual.train_updates(1);manual.train_updates(1);
    assert_eq!(fingerprint(&actual),fingerprint(&manual));
}
#[test]
fn reset_coverage_stress_evaluation_is_scoped_and_does_not_change_controls() {
    let seed = 41004+COVERAGE_EVAL+STRESS_OFFSET;
    let a = evaluation_start(seed,"reset-det",0);
    {
        let _scope = hooks::Evaluation::enter();
        let b = evaluation_start(seed,"reset-det",0);
        let state = array(b.0.state());
        assert!(state[2]*state[3]>0.0 && state[2].abs()>=0.125 && state[3].abs()>=0.25);
        assert_eq!(cursor(&a.2),cursor(&b.2));assert_eq!(cursor(&a.3),cursor(&b.3));
    }
    let c = evaluation_start(seed,"reset-det",0);
    assert_eq!(array(a.0.state()),array(c.0.state()));assert_eq!(a.1,c.1);
}
#[test]
fn reset_coverage_terminal_targets_do_not_cross_reset_boundaries() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"),41006);
    let row = |reward, terminal, end, bootstrap| Row {state:[0.0;4],observation:[0.0;4],mean:0.0,value:0.0,latent:0.0,reward,terminal,timeout:false,end,final_observation:[0.0;4],bootstrap};
    let rows = [row(-10.0,true,true,0.0),row(2.0,false,true,4.0)];
    let (returns,_,remaining) = targets(&s,&rows,1.0);
    assert_eq!(returns[0],-10.0);assert_eq!(returns[1],2.0+0.995*4.0);assert_eq!(remaining,vec![1,1]);
}
