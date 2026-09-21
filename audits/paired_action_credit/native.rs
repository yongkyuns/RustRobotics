//! Fixed same-input correction on two historical updates, not production training.
use super::*;
use crate::trainer::paired_credit_capture::{Sample, Scope};

const DRAWS: usize = 8;
const CREDIT_STEPS: usize = 1024;
const CREDIT_DOMAIN: u64 = 0x5041_4952_4352_0001;
const NEW_DOMAIN: u64 = 0x4000_0000;
const TRACE_ROWS: [usize; 6] = [0, 511, 512, 767, 768, 1023];
fn target_update(seed: u64) -> usize {
    match seed { 41006 => 4140, 41004 => 4137, _ => panic!("unknown fixed history") }
}
fn credit_key(seed: u64, update: usize, index: usize, draw: usize) -> u64 {
    assert!([41004, 41006].contains(&seed) && index < 1024 && draw < DRAWS);
    CREDIT_DOMAIN ^ (seed << 32) ^ ((update as u64) << 20) ^ ((index as u64) << 4) ^ draw as u64
}
fn baseline_action(policy: &PolicySnapshot, observation: [f32; 4], key: u64) -> f32 {
    let mut rng = StdRng::seed_from_u64(key ^ 0x4241_5345_0000_0001);
    mean(policy, observation) + (policy.action_std / policy.action_limit) * standard_normal(&mut rng)
}
#[derive(Debug, PartialEq)]
struct CreditResult {
    steps: usize, terminal: bool, total: f64, discounted: f64,
    bootstrap: f32, coefficient: f64, estimate: f64,
}
fn credit_path(
    session: &PpoTrainerSession, actor: &PolicySnapshot, critic: &PolicySnapshot,
    sample: Sample, first: f32, key: u64, mut trace: Option<&mut dyn Write>,
) -> CreditResult {
    let cfg = PendulumEnvConfig { max_steps: CREDIT_STEPS, ..session.env.config() };
    let mut env = PendulumEnv::from_state(session.env.model(), cfg, Vector4::from_column_slice(&sample.state), 0);
    let mut obs = sample.observation;
    let mut er = StdRng::seed_from_u64(key);
    let mut ar = StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
    let gamma = f64::from(session.config.ppo.gamma);
    let mut result = CreditResult { steps: 0, terminal: false, total: 0.0, discounted: 0.0,
        bootstrap: 0.0, coefficient: 1.0, estimate: 0.0 };
    if let Some(w) = trace.as_deref_mut() {
        writeln!(w, "t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated").unwrap();
    }
    for t in 0..CREDIT_STEPS {
        let before = array(env.state());
        let mu = mean(actor, obs);
        let innovation = if t == 0 { 0.0 } else { standard_normal(&mut ar) };
        let latent = if t == 0 { first } else { mu + (actor.action_std / actor.action_limit) * innovation };
        let command = actor.action_limit * latent.tanh();
        let step = env.step_with_rng(command, &mut er);
        let after = array(env.state());
        assert!(after.iter().all(|v| v.is_finite()) && step.reward.is_finite());
        if let Some(w) = trace.as_deref_mut() {
            writeln!(w, "{t},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{command},{},{},{},{},{},{},{},{},{},{},{},{}",
                before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],env.last_applied_force(),step.reward,
                after[0],after[1],after[2],after[3],step.observation[0],step.observation[1],step.observation[2],step.observation[3],step.terminated(),step.truncated).unwrap();
        }
        result.steps = t + 1;
        result.terminal = step.terminated();
        result.total += f64::from(step.reward);
        result.discounted += result.coefficient * f64::from(step.reward);
        result.coefficient *= gamma;
        obs = step.observation;
        if step.done { break; }
    }
    assert!(result.terminal || result.steps == CREDIT_STEPS);
    result.bootstrap = if result.terminal { 0.0 } else { mean(critic, obs) };
    result.estimate = result.discounted + result.coefficient * f64::from(result.bootstrap);
    assert!(result.estimate.is_finite());
    result
}
fn restored_optimizer(incoming: &PpoTrainerSession, collected: &PpoTrainerSession) -> PpoTrainerSession {
    // Preserve the actual post-collection environment/episode/metrics, but use
    // the exact incoming modules, parameter IDs, Adam records and shuffle cursor.
    let mut s = clone_session(collected);
    s.actor = incoming.actor.clone(); s.critic = incoming.critic.clone();
    s.actor_optimizer = incoming.actor_optimizer.clone();
    s.critic_optimizer = incoming.critic_optimizer.clone();
    s.update_rng = incoming.update_rng.clone();
    s
}
fn compare_tree(current: &Path, expected: &Path) -> usize {
    let mut checked = 0;
    let mut entries: Vec<_> = fs::read_dir(current).unwrap().map(|v|v.unwrap()).collect();
    entries.sort_by_key(|v|v.file_name());
    for entry in entries {
        let dest = expected.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() { checked += compare_tree(&entry.path(), &dest); }
        else { assert_eq!(fs::read(entry.path()).unwrap(), fs::read(&dest).unwrap(), "historical file differs: {}", dest.display()); checked += 1; }
    }
    checked
}
fn fit_from(incoming: &PpoTrainerSession, original: &PpoTrainerSession, batch: &RolloutBatch, out: &Path) -> PpoTrainerSession {
    let mut s = restored_optimizer(incoming, original);
    s.config.ppo.mini_batch_size = 256;
    optimize_traced(&mut s, batch, out);
    s.config.ppo.mini_batch_size = 128;
    s
}
fn write_credit_header(w: &mut dyn Write) {
    writeln!(w, "row,draw,key,recorded_latent,baseline_latent,q_steps,q_terminal,q_total,q_discounted,q_bootstrap,q_coefficient,q_estimate,v_steps,v_terminal,v_total,v_discounted,v_bootstrap,v_coefficient,v_estimate,difference").unwrap();
}
#[test]
#[ignore = "explicit paired-action-credit correction; not controller qualification"]
fn emit_paired_action_credit() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    let target = target_update(seed);
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let old_text = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old_rows: Vec<_> = old_text.lines().skip(1).collect();
    assert_eq!(old_rows.len(), 4224);
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), seed);
    compare_weights(&s, &out, &prior, 0);
    let mut costs = Costs::default();
    let mut prefix = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(prefix, "{}", old_text.lines().next().unwrap()).unwrap();
    for u in 1..target {
        let c = update_support(&mut s, seed, u, 1024, None);
        costs.add(c);
        let record = update_row(&s, u, c);
        assert_eq!(record, old_rows[u - 1], "prefix differs at {u}");
        writeln!(prefix, "{record}").unwrap();
        if [1024, 4096, 4128, target - 1].contains(&u) { compare_weights(&s, &out, &prior, u); }
        if u % 512 == 0 { prefix.flush().unwrap(); println!("PAIRED CREDIT PREFIX {seed} {u}"); }
    }
    let incoming = clone_session(&s);
    let original_dir = out.join("original"); fs::create_dir_all(&original_dir).unwrap();
    save_state(&s, &original_dir, 0);
    fs::write(original_dir.join("training-gamma.txt"), "0.995\n").unwrap();
    fs::write(original_dir.join("support-steps.txt"), "1024\n").unwrap();
    let scope = Scope::enter();
    let c = update_support(&mut s, seed, target, 1024, Some(&original_dir));
    let captured = scope.finish();
    let batch = captured.batch.unwrap(); let samples = captured.samples;
    assert_eq!(samples.len(), 1024);
    costs.add(c);
    let record = update_row(&s, target, c);
    assert_eq!(record, old_rows[target - 1]); writeln!(prefix, "{record}").unwrap(); prefix.flush().unwrap();
    let compared_files = compare_tree(&original_dir, &prior.join(format!("update-{target}")));
    compare_weights(&s, &out, &prior, target);
    let original = clone_session(&s);
    let old_actor = incoming.snapshot(); let old_critic = value_snapshot(&incoming);
    let ancestor = fingerprint(&incoming);
    let mut states = BufWriter::new(fs::File::create(out.join("fit-states.csv")).unwrap());
    writeln!(states, "row,x,v,theta,omega,o0,o1,o2,o3,latent").unwrap();
    for (i,r) in samples.iter().enumerate() {
        writeln!(states, "{i},{},{},{},{},{},{},{},{},{}",r.state[0],r.state[1],r.state[2],r.state[3],r.observation[0],r.observation[1],r.observation[2],r.observation[3],r.latent).unwrap();
    }
    states.flush().unwrap();
    let mut raw = Vec::with_capacity(1024);
    let mut added_steps = 0_u64;
    let mut w = BufWriter::new(fs::File::create(out.join("paired-credit.csv")).unwrap()); write_credit_header(&mut w);
    for (index, sample) in samples.iter().copied().enumerate() {
        let mut sum = 0.0;
        for draw in 0..DRAWS {
            let key = credit_key(seed, target, index, draw);
            let alternate = baseline_action(&old_actor, sample.observation, key);
            let keep = draw == 0 && TRACE_ROWS.contains(&index);
            let mut qt = keep.then(|| BufWriter::new(fs::File::create(out.join(format!("credit-{index}-{draw}-q.csv"))).unwrap()));
            let mut vt = keep.then(|| BufWriter::new(fs::File::create(out.join(format!("credit-{index}-{draw}-v.csv"))).unwrap()));
            let q = credit_path(&incoming, &old_actor, &old_critic, sample, sample.latent, key, qt.as_mut().map(|v|v as &mut dyn Write));
            let v = credit_path(&incoming, &old_actor, &old_critic, sample, alternate, key, vt.as_mut().map(|v|v as &mut dyn Write));
            let difference = q.estimate - v.estimate;
            sum += difference; added_steps += (q.steps + v.steps) as u64;
            writeln!(w, "{index},{draw},{key},{},{alternate},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{difference}", sample.latent,
                q.steps,q.terminal,q.total,q.discounted,q.bootstrap,q.coefficient,q.estimate,
                v.steps,v.terminal,v.total,v.discounted,v.bootstrap,v.coefficient,v.estimate).unwrap();
        }
        raw.push((sum / DRAWS as f64) as f32);
        if index % 64 == 63 { w.flush().unwrap(); println!("PAIRED CREDIT ROWS {seed} {}", index + 1); }
    }
    w.flush().unwrap(); assert_eq!(fingerprint(&incoming), ancestor);
    let mut candidate_batch = batch.clone(); candidate_batch.advantages = normalize(&raw);
    save_union(&out.join("paired8.json"), &candidate_batch, &raw);
    let candidate = fit_from(&incoming, &original, &candidate_batch, &out.join("paired8-optimizer"));
    let sham = fit_from(&incoming, &original, &batch, &out.join("sham-optimizer"));
    assert_eq!(fingerprint(&sham), fingerprint(&original));
    compare_tree(&out.join("sham-optimizer"), &original_dir.join("optimizer"));
    for k in 1..=16 {
        let file = format!("critic-{k}.bin");
        assert_eq!(fs::read(out.join("paired8-optimizer").join(&file)).unwrap(), fs::read(original_dir.join("optimizer").join(file)).unwrap());
    }
    assert_eq!(candidate.update_rng.clone().gen::<[u64;4]>(), original.update_rng.clone().gen::<[u64;4]>());
    // Verify persistent history on a subsequent IDENTICAL fixed-data transaction.
    let mut a = clone_session(&original); let mut b = clone_session(&sham);
    a.config.ppo.mini_batch_size = 256; b.config.ppo.mini_batch_size = 256;
    a.optimize(&batch); b.optimize(&batch); assert_eq!(fingerprint(&a), fingerprint(&b));
    let mut check = BufWriter::new(fs::File::create(out.join("historical-independent.csv")).unwrap()); eval_header(&mut check);
    let mut history_steps = 0;
    for (name,cp,actor) in [("before",target-1,&old_actor),("after",target,&original.snapshot())] {
        history_steps += evaluate_policy(actor,seed,FRESH_DOMAIN,name,cp,256,None,None,&mut check).steps;
    }
    check.flush().unwrap();
    assert_eq!(fs::read(out.join("historical-independent.csv")).unwrap(),fs::read(prior.join("independent/evaluation.csv")).unwrap());
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap()); eval_header(&mut eval);
    let mut eval_steps = 0;
    for (name,cp,actor) in [("incoming",target-1,&old_actor),("original",target,&original.snapshot()),("paired8",target,&candidate.snapshot())] {
        save(&out.join(format!("actor-{name}.bin")),actor);
        eval_steps += evaluate_policy(actor,seed,NEW_DOMAIN,name,cp,512,Some(&out),None,&mut eval).steps;
    }
    eval.flush().unwrap(); assert_eq!(fingerprint(&incoming), ancestor);
    fs::write(out.join("costs.json"), cost_json(costs)).unwrap();
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"target_update\":{},\"training_gamma\":0.995,\"evaluation_gamma\":0.99,",
        "\"fitting_rows\":1024,\"draws\":8,\"credit_steps\":1024,\"pair_records\":8192,",
        "\"credit_transitions\":{},\"historical_evaluation_steps\":{},\"new_evaluation_steps\":{},",
        "\"new_evaluation_records\":4608,\"historical_selected_files_exact\":{},",
        "\"sham_exact\":true,\"next_identical_optimizer_exact\":true,\"critic_steps_identical\":true,",
        "\"target_optimizer_executions\":5,\"candidate_training_continued\":false}}\n"),
        seed,target,added_steps,history_steps,eval_steps,compared_files)).unwrap();
    println!("PAIRED ACTION CREDIT COMPLETE {seed}");
}

#[test]
fn identical_first_actions_give_identical_credit_paths() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"),203);
    let actor = s.snapshot(); let critic = value_snapshot(&s);
    let sample = Sample { state:[0.8,0.4,0.1,-0.2], observation:[0.801,0.402,0.101,-0.199], latent:0.2 };
    let mut trace = Vec::new();
    let a = credit_path(&s,&actor,&critic,sample,0.2,77,Some(&mut trace));
    let b = credit_path(&s,&actor,&critic,sample,0.2,77,None);
    assert_eq!(a,b); assert_eq!(a.estimate-b.estimate,0.0); assert!(!trace.is_empty());
}
#[test]
fn baseline_keys_are_independent_of_recorded_actions_and_unique() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"),203); let actor=s.snapshot();
    let mut keys = std::collections::BTreeSet::new();
    for seed in [41004,41006] { for row in 0..1024 { for draw in 0..DRAWS {
        assert!(keys.insert(credit_key(seed,target_update(seed),row,draw)));
    } } }
    let key=credit_key(41006,4140,0,0); let observation=[0.0;4];
    let first=baseline_action(&actor,observation,key);
    for recorded in [-3.0,0.0,3.0] {
        let sample=Sample{state:[0.0;4],observation,latent:recorded};
        assert_eq!(first,baseline_action(&actor,sample.observation,key));
    }
}
#[test]
fn terminal_credit_has_no_bootstrap_and_cutoff_retains_it() {
    let s=PpoTrainerSession::new_seeded(learning_config("gamma995"),203);
    let actor=s.snapshot(); let critic=value_snapshot(&s);
    let sample=Sample{state:[2.5,0.0,0.0,0.0],observation:[2.5,0.0,0.0,0.0],latent:0.0};
    let result=credit_path(&s,&actor,&critic,sample,0.0,12,None);
    assert!(result.terminal); assert_eq!(result.steps,1); assert_eq!(result.bootstrap,0.0);
    assert_eq!(result.estimate,result.discounted);
    let gamma=f64::from(0.995_f32); let value=7.0;
    assert!((gamma.powi(1024)*value-0.041298557).abs()<1e-6);
}
#[test]
fn capture_and_optimizer_restore_preserve_original_and_next_update() {
    let mut s=PpoTrainerSession::new_seeded(learning_config("gamma995"),204);
    update_support(&mut s,204,1,1024,None); let incoming=clone_session(&s);
    let mut untouched=clone_session(&s); let scope=Scope::enter();
    update_support(&mut s,204,4097,1024,None); let captured=scope.finish();
    update_support(&mut untouched,204,4097,1024,None); assert_eq!(fingerprint(&s),fingerprint(&untouched));
    let batch=captured.batch.unwrap(); assert_eq!(captured.samples.len(),1024);
    let mut sham=restored_optimizer(&incoming,&s); sham.config.ppo.mini_batch_size=256; sham.optimize(&batch); sham.config.ppo.mini_batch_size=128;
    assert_eq!(fingerprint(&sham),fingerprint(&s));
    update_support(&mut s,204,4098,1024,None); update_support(&mut sham,204,4098,1024,None);
    assert_eq!(fingerprint(&s),fingerprint(&sham));
}
