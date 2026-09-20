//! Test-only actor-advantage intervention. No new controller or data mixture.
use super::*;

const ARMS: [&str; 3] = ["global", "source-center", "source-standardize"];
const EVALUATION_OFFSET: u64 = 0x0400_0000;
const SPLIT: usize = 768;
thread_local! {
    static NORMALIZATION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
struct NormGuard;
impl NormGuard {
    fn enter(mode: usize) -> Self {
        assert!(mode < ARMS.len());
        NORMALIZATION.with(|v| { assert!(v.get().is_none(), "nested normalization scope"); v.set(Some(mode)); });
        Self
    }
}
impl Drop for NormGuard {
    fn drop(&mut self) { NORMALIZATION.with(|v| v.set(None)); }
}
fn moments(values: &[f32]) -> (f32, f32) {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let mean = mean_slice(values).unwrap();
    let variance = values.iter().map(|v| { let d = *v - mean; d * d }).sum::<f32>() / values.len() as f32;
    (mean, variance.sqrt().max(1.0e-6))
}
fn transform(raw: &[f32], mode: usize) -> Vec<f32> {
    assert_eq!(raw.len(), 1024);
    assert!(mode < ARMS.len());
    if mode == 0 { return normalize(raw); }
    if mode == 2 {
        let mut values = normalize(&raw[..SPLIT]);
        values.extend(normalize(&raw[SPLIT..]));
        return values;
    }
    let (_, common_std) = moments(raw);
    let mut values = Vec::with_capacity(raw.len());
    for group in [&raw[..SPLIT], &raw[SPLIT..]] {
        let (mean, _) = moments(group);
        values.extend(group.iter().map(|v| (*v - mean) / common_std));
    }
    values
}
/// The inherited collector invokes this after joining raw advantages, before
/// saving the union and calling the ordinary production optimize() method.
pub(crate) fn apply(batch: &mut RolloutBatch, raw: &[f32], out: Option<&Path>) {
    let mode = NORMALIZATION.with(|v| v.get()).unwrap_or(0);
    assert_eq!(batch.observations.len(), 1024);
    assert_eq!(batch.advantages, normalize(raw));
    if mode != 0 { batch.advantages = transform(raw, mode); }
    assert!(batch.advantages.iter().all(|v| v.is_finite()));
    if let Some(out) = out {
        let (gm, gs) = moments(raw);
        let (nm, ns) = moments(&raw[..SPLIT]);
        let (om, os) = moments(&raw[SPLIT..]);
        fs::write(out.join("normalization.json"), format!(concat!(
            "{{\"mode\":{},\"arm\":\"{}\",\"split\":768,\"floor\":1e-6,",
            "\"global_mean\":{},\"global_std\":{},\"nominal_mean\":{},\"nominal_std\":{},",
            "\"outward_mean\":{},\"outward_std\":{}}}\n"), mode, ARMS[mode], gm, gs, nm, ns, om, os)).unwrap();
    }
}
fn normalized_update(s: &mut PpoTrainerSession, seed: u64, global: usize, mode: usize, out: Option<&Path>) -> Costs {
    let _guard = NormGuard::enter(mode);
    if let Some(path) = out {
        fs::create_dir_all(path).unwrap();
        save_state(s, path, 0); // Actual incoming modules for every selected update.
    }
    one_update(s, seed, global, true, out)
}
fn evaluate(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, out: &Path, log: &mut dyn Write) -> usize {
    let before = fingerprint(s);
    let policy = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == 0 || cp == EXTRA { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut steps = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        for rep in 0..64 {
            let mut trace = (rep == 0 && (cp == 0 || cp == EXTRA)).then(||
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(&policy, seed + EVALUATION_OFFSET, panel, rep, trace.as_mut().map(|t| t as &mut dyn Write));
            writeln!(log, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key, r.steps, r.ending, r.total, r.discounted, r.max_position, r.max_angle, r.centered, r.force_rms).unwrap();
            steps += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s), "evaluation mutated learner");
    steps
}
fn witnesses(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, prior: &Path, out: &Path, log: &mut dyn Write) -> (usize, usize) {
    if seed != 41008 || ![128,512].contains(&cp) { return (0,0); }
    let before = fingerprint(s);
    let source = fs::read_to_string(prior.join("evaluation.csv")).unwrap();
    let selected: Vec<Vec<&str>> = source.lines().skip(1).map(|l| l.split(',').collect())
        .filter(|r: &Vec<&str>| r[1] == "half-outward" && r[2].parse::<usize>().unwrap() == cp && r[8] == "angle").collect();
    assert_eq!(selected.len(), if cp == 128 { 2 } else { 10 });
    let mut steps = 0;
    let policy = s.snapshot();
    for row in &selected {
        let panel = row[3]; let rep: usize = row[4].parse().unwrap(); let cap: usize = row[6].parse().unwrap();
        let mut trace = BufWriter::new(fs::File::create(out.join(format!("witness-{cp}-{panel}-{rep}.csv"))).unwrap());
        let r = eval_one(&policy, seed + 0x0200_0000, panel, rep, Some(&mut trace));
        assert_eq!(r.key, row[5].parse::<u64>().unwrap());
        if arm == "global" {
            assert_eq!(r.steps, row[7].parse::<usize>().unwrap()); assert_eq!(r.ending, row[8]);
            assert_eq!(r.total, row[9].parse::<f64>().unwrap()); assert_eq!(r.discounted, row[10].parse::<f64>().unwrap());
            assert_eq!(r.max_position, row[11].parse::<f32>().unwrap()); assert_eq!(r.max_angle, row[12].parse::<f32>().unwrap());
            assert_eq!(r.centered, row[13].parse::<bool>().unwrap()); assert_eq!(r.force_rms, row[14].parse::<f64>().unwrap());
        }
        writeln!(log,"{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
            r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
        steps += r.steps;
    }
    assert_eq!(before, fingerprint(s));
    (selected.len(),steps)
}
#[test]
#[ignore = "explicit source-normalization development experiment; not a production gate"]
fn emit_source_normalization() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((41001..=41008).contains(&seed));
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let mut initial = PpoTrainerSession::new_seeded(config(), seed);
    let mut prefix = Costs::default();
    for global in 1..=PREFIX {
        prefix.add(one_update(&mut initial,seed,global,false,None));
        if global % 512 == 0 { println!("SOURCE PREFIX {seed} {global}"); }
    }
    save_state(&initial,&out,0);
    for kind in ["actor","critic"] {
        assert_eq!(fs::read(out.join(format!("{kind}-0.bin"))).unwrap(),fs::read(prior.join(format!("{kind}-0.bin"))).unwrap());
    }
    assert_eq!(cost_json(prefix),fs::read_to_string(prior.join("prefix-costs.json")).unwrap());
    fs::write(out.join("prefix-costs.json"),cost_json(prefix)).unwrap();
    fs::write(out.join("config.json"),format!(concat!(
        "{{\"seed\":{},\"prefix_updates\":4096,\"continuation_updates\":512,\"checkpoints\":[0,32,128,512],",
        "\"arms\":[\"global\",\"source-center\",\"source-standardize\"],\"candidate\":\"source-standardize\",",
        "\"split\":768,\"rows\":1024,\"std_floor\":1e-6,\"lambda\":1,\"gamma\":0.99,\"epsilon\":1e-5,\"lr\":0.0003,",
        "\"minibatch\":256,\"epochs\":4,\"start_abs_position\":[0.4,1.4],\"start_abs_outward_velocity\":[0.2,1.0],",
        "\"start_angle\":[-0.2,0.2],\"start_angular_velocity\":[-0.5,0.5],\"evaluation_seed\":{},\"repetitions\":64}}\n"),seed,seed+EVALUATION_OFFSET)).unwrap();
    let ancestor = fingerprint(&initial);
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap()); eval_header(&mut eval);
    let mut witness_log = BufWriter::new(fs::File::create(out.join("witnesses.csv")).unwrap()); eval_header(&mut witness_log);
    let mut eval_steps = evaluate(&initial,seed,"incoming",0,&out,&mut eval); eval.flush().unwrap();
    let mut records = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(records,"arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut combined = Costs::default(); let mut witness_steps = 0; let mut witness_count = 0;
    for (mode,arm) in ARMS.into_iter().enumerate() {
        let mut session = clone_session(&initial); assert_eq!(fingerprint(&session),ancestor);
        let dir = out.join(arm); fs::create_dir_all(&dir).unwrap(); let mut costs = Costs::default();
        for local in 1..=EXTRA {
            let path = [1,128,512].contains(&local).then(||dir.join(format!("update-{local}")));
            let c = normalized_update(&mut session,seed,PREFIX+local,mode,path.as_deref()); costs.add(c);
            assert_eq!(session.metrics.total_updates,PREFIX+local);
            assert_eq!(session.metrics.total_env_steps,(PREFIX+local)*512);
            writeln!(records,"{arm},{local},{},{},{},{},{},{},{},{},{}",session.metrics.total_updates,
                c.primary,c.tails,c.supplement,c.actor_steps,c.sample_visits,session.metrics.total_episodes,
                session.metrics.last_policy_loss,session.metrics.last_value_loss).unwrap();
            if local % 32 == 0 { records.flush().unwrap(); println!("SOURCE PROGRESS {seed} {arm} {local}"); }
            if CHECKPOINTS.contains(&local) {
                save_state(&session,&dir,local);
                if mode == 0 {
                    for kind in ["actor","critic"] {
                        assert_eq!(fs::read(dir.join(format!("{kind}-{local}.bin"))).unwrap(),
                            fs::read(prior.join("half-outward").join(format!("{kind}-{local}.bin"))).unwrap(),"historical global arm mismatch");
                    }
                }
                eval_steps += evaluate(&session,seed,arm,local,&dir,&mut eval); eval.flush().unwrap();
                let (count,steps) = witnesses(&session,seed,arm,local,&prior,&dir,&mut witness_log);
                witness_count += count; witness_steps += steps; witness_log.flush().unwrap();
                fs::write(dir.join("costs.json"),cost_json(costs)).unwrap();
            }
        }
        assert_eq!(costs.primary,262144); assert_eq!(costs.supplement,2359296);
        assert_eq!(costs.actor_steps,8192); assert_eq!(costs.sample_visits,2097152);
        combined.add(costs); assert_eq!(fingerprint(&initial),ancestor);
    }
    records.flush().unwrap(); eval.flush().unwrap(); witness_log.flush().unwrap();
    assert_eq!(witness_count,if seed == 41008 { 36 } else { 0 });
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"execution\":\"complete\",\"prefix_exact\":true,\"global_checkpoints_exact\":true,",
        "\"branches\":3,\"continuation_updates\":1536,\"primary_steps\":{},\"tail_steps\":{},\"supplement_steps\":{},",
        "\"actor_steps\":{},\"critic_steps\":{},\"sample_visits_each\":{},\"evaluation_steps\":{},",
        "\"evaluation_records\":2688,\"witness_steps\":{},\"witness_records\":{},\"ancestor_unchanged\":true}}\n"),
        seed,combined.primary,combined.tails,combined.supplement,combined.actor_steps,combined.actor_steps,
        combined.sample_visits,eval_steps,witness_steps,witness_count)).unwrap();
    println!("SOURCE NORMALIZATION COMPLETE {seed}");
}
#[test]
fn transforms_use_original_population_moments_and_floor() {
    let raw: Vec<f32> = (0..1024).map(|i| if i < SPLIT { (i % 17) as f32 * 0.1 } else { (i % 13) as f32 - 20.0 }).collect();
    assert_eq!(transform(&raw,0),normalize(&raw));
    let centered = transform(&raw,1); let standardized = transform(&raw,2); let (_,std) = moments(&raw);
    for (start,end) in [(0,SPLIT),(SPLIT,1024)] {
        let (mean,_) = moments(&raw[start..end]);
        for i in start..end { assert_eq!(centered[i],(raw[i]-mean)/std); }
        assert_eq!(&standardized[start..end],normalize(&raw[start..end]).as_slice());
    }
    assert_eq!(transform(&vec![4.0;1024],1),vec![0.0;1024]);
    assert_eq!(transform(&vec![4.0;1024],2),vec![0.0;1024]);
}
#[test]
fn standardized_nominal_values_do_not_depend_on_outward_rows() {
    let a: Vec<f32> = (0..1024).map(|i| (i % 31) as f32 / 17.0).collect(); let mut b = a.clone();
    for v in &mut b[SPLIT..] { *v = *v * 100.0 - 80.0; }
    assert_eq!(&transform(&a,2)[..SPLIT],&transform(&b,2)[..SPLIT]);
    assert_ne!(&transform(&a,0)[..SPLIT],&transform(&b,0)[..SPLIT]);
}
#[test]
fn mode_restores_after_unwind() {
    let result = std::panic::catch_unwind(|| { let _guard = NormGuard::enter(2); panic!("intentional scope test"); });
    assert!(result.is_err()); assert_eq!(NORMALIZATION.with(|v| v.get()),None);
}
#[test]
fn first_critic_update_is_identical_and_actor_only_intervention_executes() {
    let mut initial = PpoTrainerSession::new_seeded(config(),201); initial.train_updates(2);
    let before = fingerprint(&initial); let mut actors = Vec::new(); let mut critics = Vec::new();
    for mode in 0..3 {
        let mut s = clone_session(&initial); normalized_update(&mut s,201,4097,mode,None);
        actors.push(s.snapshot()); critics.push(s.shared_state().value);
    }
    assert_eq!(critics[0],critics[1]); assert_eq!(critics[0],critics[2]);
    assert_ne!(actors[0],actors[1]); assert_ne!(actors[0],actors[2]); assert_eq!(before,fingerprint(&initial));
}
#[test]
fn global_mode_preserves_original_and_next_optimizer_update() {
    let mut a = PpoTrainerSession::new_seeded(config(),202); a.train_updates(2); let mut b = clone_session(&a);
    for local in 4097..4100 {
        one_update(&mut a,202,local,true,None); normalized_update(&mut b,202,local,0,None);
        assert_eq!(fingerprint(&a),fingerprint(&b));
    }
    a.train_updates(1); b.train_updates(1); assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn source_modes_keep_full_state_and_optimizer_grouping() {
    let initial = PpoTrainerSession::new_seeded(config(),203);
    for mode in [1,2] {
        let mut a = clone_session(&initial); let mut b = clone_session(&initial);
        for global in 4097..4100 { normalized_update(&mut a,203,global,mode,None); }
        for global in 4097..4100 { let _ = b.shared_state(); normalized_update(&mut b,203,global,mode,None); }
        assert_eq!(fingerprint(&a),fingerprint(&b));
        a.train_updates(1); b.train_updates(1); assert_eq!(fingerprint(&a),fingerprint(&b));
    }
}
