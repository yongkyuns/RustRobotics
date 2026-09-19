//! Fresh-cohort from-scratch driver. No trained checkpoint is accepted as input.
use super::*;

const UPDATES: usize = 4096;
const CHECKPOINTS: [usize; 4] = [0, 256, 1024, 4096];
const ARMS: [&str; 4] = ["baseline95", "ordinary", "near-union", "recovery-union"];
const EVALUATION_OFFSET: u64 = 0x0100_0000;

fn fresh_config(arm: &str) -> PpoTrainerConfig {
    assert!(ARMS.contains(&arm));
    if arm == "baseline95" { PpoTrainerConfig::default() } else { config() }
}
fn evaluation_seed(seed: u64) -> u64 {
    assert!(seed < 65536);
    seed + EVALUATION_OFFSET
}
fn ordinary95(s: &mut PpoTrainerSession, out: Option<&Path>) -> Costs {
    assert_eq!(s.config.ppo.gae_lambda, 0.95);
    let before = s.metrics.total_env_steps;
    let (batch, rows, _) = capture(s);
    let (r1, a1, remaining) = targets(s, &rows, 1.0);
    let (r95, a95, other) = targets(s, &rows, 0.95);
    assert_eq!(remaining, other);
    assert_eq!(r95, batch.returns);
    assert_eq!(normalize(&a95), batch.advantages);
    if let Some(out) = out {
        fs::create_dir_all(out).unwrap();
        // The inherited diagnostic schema labels both target constructions.
        // The actual optimizer receives the untouched production lambda-.95 batch.
        let mut lambda1 = batch.clone();
        lambda1.returns = r1;
        lambda1.advantages = normalize(&a1);
        dump_batch(out, &rows, &lambda1, &r95, &a95, &a1, &remaining);
        save_union(&out.join("corrected.json"), &batch, &a95);
        optimize_traced(s, &batch, &out.join("optimizer"));
    } else {
        s.optimize(&batch);
    }
    s.metrics.total_updates += 1;
    assert_eq!(s.metrics.total_env_steps - before, 512);
    Costs {primary:512, updates:1, actor_steps:16, sample_visits:2048, ..Default::default()}
}
fn fresh_step(s: &mut PpoTrainerSession, arm: &str, seed: u64, local: usize, out: Option<&Path>) -> Costs {
    if arm == "baseline95" { ordinary95(s, out) } else { update(s, arm, seed, local, out) }
}
fn fresh_evaluate(s: &PpoTrainerSession, seed:u64, arm:&str, cp:usize, out:&Path, log:&mut dyn Write) -> usize {
    let before = fingerprint(s);
    let policy = s.snapshot();
    let eval_seed = evaluation_seed(seed);
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == UPDATES { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut cost = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        for rep in 0..64 {
            let mut trace = (rep == 0 && (cp == 0 || cp == UPDATES)).then(|| {
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap())
            });
            let r = eval_one(&policy, eval_seed, panel, rep, trace.as_mut().map(|v| v as &mut dyn Write));
            writeln!(log,"{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
            cost += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s), "held-out evaluation mutated the learner");
    cost
}
#[test]
#[ignore = "explicit fresh-cohort experiment; pass is not automatic controller acceptance"]
fn emit_fresh_recovery() {
    let seed:u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    let arm = std::env::var("FRESH_ARM").unwrap();
    assert!((41001..=41008).contains(&seed) && ARMS.contains(&arm.as_str()));
    // Deliberately no HORIZON_PRIOR read, ancestor history or shared-state import.
    let mut s = PpoTrainerSession::new_seeded(fresh_config(&arm), seed);
    assert_eq!(s.metrics.total_updates, 0);
    assert_eq!(s.metrics.total_env_steps, 0);
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("config.json"),format!(concat!(
        "{{\"schema\":1,\"seed\":{},\"arm\":\"{}\",\"from_random_initialization\":true,",
        "\"checkpoint_imports\":0,\"warmup_updates\":0,\"updates\":4096,\"primary_budget\":2097152,",
        "\"gamma\":0.99,\"lambda\":{},\"epsilon\":1e-5,\"learning_rate\":0.0003,",
        "\"epochs\":4,\"rollout_rows\":512,\"union_rows\":1024,\"selected_per_stream\":64,",
        "\"supplemental_streams\":8,\"stream_length\":576,\"lookahead_draws\":4,\"lookahead_length\":512,",
        "\"checkpoint_updates\":[0,256,1024,4096],\"evaluation_seed\":{},\"repetitions_per_panel\":64}}\n"),
        seed,arm,s.config.ppo.gae_lambda,evaluation_seed(seed))).unwrap();
    save_state(&s,&out,0);
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    let mut eval_steps = fresh_evaluate(&s,seed,&arm,0,&out,&mut eval);
    eval.flush().unwrap();
    let mut log = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(log,"arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut costs = Costs::default();
    for local in 1..=UPDATES {
        let traced = [1,256,4096].contains(&local);
        let path = traced.then(||out.join(format!("update-{local}")));
        let c = fresh_step(&mut s,&arm,seed,local,path.as_deref());
        costs.add(c);
        assert_eq!(s.metrics.total_updates,local);
        assert_eq!(s.metrics.total_env_steps,local*512);
        assert!(s.metrics.last_policy_loss.is_finite() && s.metrics.last_value_loss.is_finite());
        writeln!(log,"{arm},{local},{},{},{},{},{},{},{},{},{}",s.metrics.total_updates,c.primary,c.tails,c.supplement,
            c.actor_steps,c.sample_visits,s.metrics.total_episodes,s.metrics.last_policy_loss,s.metrics.last_value_loss).unwrap();
        if local % 128 == 0 { log.flush().unwrap(); println!("FRESH PROGRESS {arm} {seed} {local}"); }
        if CHECKPOINTS.contains(&local) {
            save_state(&s,&out,local);
            eval_steps += fresh_evaluate(&s,seed,&arm,local,&out,&mut eval);
            eval.flush().unwrap();
            fs::write(out.join("costs.json"),cost_json(costs)).unwrap();
            println!("FRESH CHECKPOINT {arm} {seed} {local}");
        }
    }
    assert_eq!(costs.primary,2097152);
    assert_eq!(costs.updates,4096);
    assert_eq!(costs.actor_steps,65536);
    assert_eq!(costs.supplement,if arm.ends_with("union") {18874368} else {0});
    assert_eq!(costs.sample_visits,if arm.ends_with("union") {16777216} else {8388608});
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"arm\":\"{}\",\"execution\":\"complete\",\"from_scratch\":true,",
        "\"primary_steps\":2097152,\"updates\":4096,\"tail_steps\":{},\"supplement_steps\":{},",
        "\"actor_adam_steps\":65536,\"critic_adam_steps\":65536,\"sample_visits_each\":{},",
        "\"evaluation_steps\":{},\"evaluation_records\":960,\"checkpoint_imports\":0}}\n"),
        seed,arm,costs.tails,costs.supplement,costs.sample_visits,eval_steps)).unwrap();
    log.flush().unwrap();
    println!("FRESH RECOVERY COMPLETE {arm} {seed}");
}
#[test]
fn random_starts_match_across_all_recipes() {
    // A development seed for implementation controls, not a held-out training run.
    let original = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),201);
    for arm in ARMS {
        let s = PpoTrainerSession::new_seeded(fresh_config(arm),201);
        assert_eq!(s.snapshot(),original.snapshot());
        assert_eq!(s.shared_state().value,original.shared_state().value);
        assert_eq!(s.metrics.total_updates,0);
        assert_eq!(s.metrics.total_env_steps,0);
        assert_eq!(format!("{:?}",s.env),format!("{:?}",original.env));
    }
}
#[test]
fn baseline_driver_is_ordinary_production_with_warm_adam() {
    let mut a = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),202);
    let mut b = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),202);
    for local in 1..=4 {
        a.train_updates(1); fresh_step(&mut b,"baseline95",202,local,None);
        assert_eq!(fingerprint(&a),fingerprint(&b));
    }
    a.train_updates(1); b.train_updates(1);
    assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn fresh_domains_are_unique_and_preserve_historical_prefix_keys() {
    let mut keys = std::collections::BTreeSet::new();
    for seed in 41001..=41008 {
        for local in 1..=UPDATES { assert!(keys.insert(stream_id(seed,local))); }
        assert!(evaluation_seed(seed)>u64::from(u16::MAX));
    }
    assert_eq!(keys.len(),8*UPDATES);
    assert_eq!(stream_id(201,1),201);
    assert_eq!(stream_id(204,512),204+511*65536);
}
#[test]
fn fresh_evaluation_does_not_consume_training_state() {
    let s=PpoTrainerSession::new_seeded(config(),203);
    let before=fingerprint(&s);
    let eval_seed=evaluation_seed(203);
    let a=eval_one(&s.snapshot(),eval_seed,"reset-stoch",0,None);
    let b=eval_one(&s.snapshot(),eval_seed,"reset-stoch",0,None);
    assert_eq!(a,b);
    assert_eq!(before,fingerprint(&s));
}
