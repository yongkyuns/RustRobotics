//! A diagnostic witness, never a training-policy acceptance or rollback rule.
use super::*;
use crate::trainer::late_nominal_adam;

const LATE_SEED: u64 = 41004;
const WINDOW_FIRST: usize = 4128;
const WINDOW_LAST: usize = 4224;
const CONFIRM_DOMAIN: u64 = 0x7400_0000;
const PAIR_XOR: u64 = 0x2000_0000_0000_0000;

struct LateWitness {
    update: usize,
    drop: f64,
    incoming: PpoTrainerSession,
    original: PpoTrainerSession,
    batch: RolloutBatch,
    samples: Vec<Sample>,
}

fn replace_witness(previous: Option<(usize, f64)>, update: usize, drop: f64) -> bool {
    assert!((WINDOW_FIRST + 1..=WINDOW_LAST).contains(&update) && drop.is_finite());
    if drop <= 0.0 { return false; }
    previous.is_none_or(|(old_update, old_drop)| drop > old_drop || (drop == old_drop && update < old_update))
}

fn window_score(s: &PpoTrainerSession, cp: usize, w: &mut dyn Write) -> (f64, usize) {
    let before = fingerprint(s);
    let actor = s.snapshot();
    let mut total = 0.0;
    let mut steps = 0;
    for panel in ["reset-det", "reset-stoch", "outward-stoch"] {
        for rep in 0..64 {
            let r = eval_one(&actor, LATE_SEED + ONLINE_EVAL, panel, rep, None);
            writeln!(w, "{LATE_SEED},extra-fresh,{cp},{panel},{rep},{},2048,{},{},{},{},{},{},{},{}",
                r.key, r.steps, r.ending, r.total, r.discounted, r.max_position,
                r.max_angle, r.centered, r.force_rms).unwrap();
            if panel == "reset-det" { total += r.discounted; }
            steps += r.steps;
        }
    }
    assert_eq!(fingerprint(s), before);
    (total / 64.0, steps)
}

fn exact_window_anchors(current: &Path, prior: &Path) {
    let left = fs::read_to_string(current).unwrap();
    let right = fs::read_to_string(prior).unwrap();
    for cp in [WINDOW_FIRST, WINDOW_LAST] {
        let prefix = format!("{LATE_SEED},extra-fresh,{cp},");
        let select = |text: &str| -> Vec<String> {
            text.lines().filter(|line| {
                let parts: Vec<_> = line.split(',').collect();
                line.starts_with(&prefix) && ["reset-det", "reset-stoch", "outward-stoch"].contains(&parts[3])
                    && parts[4].parse::<usize>().unwrap() < 64
            }).map(str::to_owned).collect()
        };
        let a = select(&left); let b = select(&right);
        assert_eq!(a.len(), 192); assert_eq!(a, b, "window endpoint differs from original pilot");
    }
}

fn selected_credit(witness: &LateWitness, out: &Path) -> (Vec<f32>, usize) {
    let s = &witness.incoming;
    let before = fingerprint(s);
    let actor = s.snapshot(); let critic = value_snapshot(s);
    let mut states = BufWriter::new(fs::File::create(out.join("fit-states.csv")).unwrap());
    writeln!(states, "row,x,v,theta,omega,o0,o1,o2,o3,latent").unwrap();
    let mut log = BufWriter::new(fs::File::create(out.join("paired-credit.csv")).unwrap());
    write_credit_header(&mut log);
    let mut baselines = Vec::new(); let mut steps = 0;
    for (index, sample) in witness.samples.iter().copied().enumerate() {
        assert_eq!(sample.observation, witness.batch.observations[index]);
        assert_eq!(sample.latent, witness.batch.latent_actions[index]);
        writeln!(states, "{index},{},{},{},{},{},{},{},{},{}", sample.state[0], sample.state[1], sample.state[2], sample.state[3], sample.observation[0], sample.observation[1], sample.observation[2], sample.observation[3], sample.latent).unwrap();
        let mut sum = 0.0;
        for draw in 0..DRAWS {
            let key = credit_key(LATE_SEED, witness.update, index, draw) ^ PAIR_XOR;
            let alternate = baseline_action(&actor, sample.observation, key);
            let trace = draw == 0 && TRACE_ROWS.contains(&index);
            let mut qt = trace.then(|| BufWriter::new(fs::File::create(out.join(format!("credit-{index}-q.csv"))).unwrap()));
            let mut vt = trace.then(|| BufWriter::new(fs::File::create(out.join(format!("credit-{index}-v.csv"))).unwrap()));
            let q = credit_path(s, &actor, &critic, sample, sample.latent, key, qt.as_mut().map(|v| v as &mut dyn Write));
            let v = credit_path(s, &actor, &critic, sample, alternate, key, vt.as_mut().map(|v| v as &mut dyn Write));
            let difference = q.estimate - v.estimate;
            sum += v.estimate; steps += q.steps + v.steps;
            writeln!(log, "{index},{draw},{key},{},{alternate},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{difference}", sample.latent,
                q.steps,q.terminal,q.total,q.discounted,q.bootstrap,q.coefficient,q.estimate,
                v.steps,v.terminal,v.total,v.discounted,v.bootstrap,v.coefficient,v.estimate).unwrap();
        }
        baselines.push((sum / DRAWS as f64) as f32);
        if (index + 1).is_multiple_of(64) { log.flush().unwrap(); println!("LATE CREDIT ROWS {}", index + 1); }
    }
    states.flush().unwrap(); log.flush().unwrap();
    assert_eq!(fingerprint(s), before);
    assert_eq!(baselines.len(), 1024);
    (baselines, steps)
}

fn causal_replay(witness: &LateWitness, root: &Path) -> (usize, usize) {
    let out = root.join("selected"); fs::create_dir_all(&out).unwrap();
    let ordinary = root.join(format!("update-{}/ordinary/optimizer", witness.update));
    save_state(&witness.incoming, &out, 0);
    save_state(&witness.original, &out, 1);
    late_nominal_adam::start(out.join("sham-adam"));
    let sham = fit_from(&witness.incoming, &witness.original, &witness.batch, &out.join("sham-optimizer"));
    late_nominal_adam::stop();
    assert_eq!(fingerprint(&sham), fingerprint(&witness.original));
    assert_eq!(compare_tree(&out.join("sham-optimizer"), &ordinary), 48);
    let mut a = clone_session(&sham); let mut b = clone_session(&witness.original);
    a.config.ppo.mini_batch_size = 256; b.config.ppo.mini_batch_size = 256;
    a.optimize(&witness.batch); b.optimize(&witness.batch);
    assert_eq!(fingerprint(&a), fingerprint(&b));
    let (baselines, credit_steps) = selected_credit(witness, &out);
    fs::write(out.join("baseline-values.bin"), baselines.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let raw: Vec<f32> = witness.batch.returns.iter().zip(&baselines).map(|(r, v)| r - v).collect();
    let mut batch = witness.batch.clone(); batch.advantages = normalize(&raw);
    save_union(&out.join("baseline-only.json"), &batch, &raw);
    late_nominal_adam::start(out.join("candidate-adam"));
    let candidate = fit_from(&witness.incoming, &witness.original, &batch, &out.join("candidate-optimizer"));
    late_nominal_adam::stop();
    for step in 1..=16 {
        let name = format!("critic-{step}.bin");
        assert_eq!(fs::read(out.join("candidate-optimizer").join(&name)).unwrap(), fs::read(ordinary.join(&name)).unwrap());
    }
    assert_eq!(cursor(&candidate.update_rng), cursor(&witness.original.update_rng));
    let mut log = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut log); let mut eval_steps = 0;
    for (name, cp, actor) in [("incoming", witness.update-1, witness.incoming.snapshot()), ("original", witness.update, witness.original.snapshot()), ("baseline-only", witness.update, candidate.snapshot())] {
        save(&out.join(format!("actor-{name}.bin")), &actor);
        eval_steps += evaluate_policy(&actor, LATE_SEED, CONFIRM_DOMAIN, name, cp, 512, Some(&out), None, &mut log).steps;
    }
    log.flush().unwrap();
    (credit_steps, eval_steps)
}

#[test]
#[ignore = "fixed late-window localization and baseline-only diagnostic; not qualification"]
fn emit_late_nominal() {
    let root = PathBuf::from(std::env::var("LATE_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("LATE_PRIOR").unwrap());
    fs::create_dir_all(&root).unwrap();
    let history = prior.join("history"); let online = prior.join("online/extra-fresh");
    let text = fs::read_to_string(history.join("updates.csv")).unwrap();
    let historical: Vec<_> = text.lines().skip(1).collect(); assert_eq!(historical.len(), 4224);
    let expected = fs::read_to_string(online.join("updates.csv")).unwrap();
    let expected_rows: Vec<_> = expected.lines().skip(1).collect(); assert_eq!(expected_rows.len(), 128);
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), LATE_SEED);
    compare_weights(&s, &root, &history, 0);
    let mut prefix_cost = Costs::default();
    for update in 1..=ONLINE_PREFIX {
        let cost = update_support(&mut s, LATE_SEED, update, 1024, None);
        prefix_cost.add(cost);
        assert_eq!(update_row(&s, update, cost), historical[update-1]);
        if [1024, ONLINE_PREFIX].contains(&update) { compare_weights(&s, &root, &history, update); }
        if update.is_multiple_of(512) { println!("LATE PREFIX {update}"); }
    }
    fs::write(root.join("prefix-costs.json"), cost_json(prefix_cost)).unwrap();
    let mut rng = StdRng::seed_from_u64(0x4f4e_4c43_5249_5455 ^ LATE_SEED);
    let mut window = BufWriter::new(fs::File::create(root.join("window.csv")).unwrap()); eval_header(&mut window);
    let mut curve = BufWriter::new(fs::File::create(root.join("curve.csv")).unwrap());
    writeln!(curve, "update,nominal_det_mean,previous_minus_current").unwrap();
    let mut log = BufWriter::new(fs::File::create(root.join("updates.csv")).unwrap());
    writeln!(log, "{}", expected.lines().next().unwrap()).unwrap();
    let mut best: Option<LateWitness> = None; let mut previous = None;
    let mut window_steps = 0; let mut ordinary_cost = Costs::default();
    for update in ONLINE_PREFIX+1..=WINDOW_LAST {
        let incoming = clone_session(&s);
        let dir = root.join(format!("update-{update}")); fs::create_dir_all(&dir).unwrap();
        let detailed = update >= WINDOW_FIRST || update == 4097;
        let ordinary = detailed.then(|| dir.join("ordinary"));
        let scope = Scope::enter();
        let cost = update_support(&mut s, LATE_SEED, update, 1024, ordinary.as_deref());
        let captured = scope.finish(); let batch = captured.batch.unwrap(); ordinary_cost.add(cost);
        let original = clone_session(&s);
        if let Some(ref path) = ordinary {
            let old = online.join(format!("update-{update}/ordinary"));
            if old.is_dir() { assert_eq!(compare_tree(path, &old), 72); }
        }
        let fresh = collect_online_data(&s, stream_id(LATE_SEED, update) ^ 0x8000_0000, true, Some(&dir.join("fresh")));
        assert_eq!(compare_tree(&dir.join("fresh"), &online.join(format!("update-{update}/fresh"))), 8);
        fresh.save(&dir.join("fit-data.bin"));
        assert_eq!(fs::read(dir.join("fit-data.bin")).unwrap(), fs::read(online.join(format!("update-{update}/fit-data.bin"))).unwrap());
        let (pre, post) = fit_online_value(&mut s, &fresh, &mut rng, 48, Some(&dir.join("extra-optimizer.jsonl")));
        assert_eq!(fs::read(dir.join("extra-optimizer.jsonl")).unwrap(), fs::read(online.join(format!("update-{update}/extra-optimizer.jsonl"))).unwrap());
        assert_eq!(s.snapshot(), original.snapshot(), "extra critic phase changed actor");
        let record = format!("{update},{},{},{},16,16,12288,48,12288,4096,{pre},{post},{},{}", cost.primary,cost.tails,cost.supplement,s.metrics.last_policy_loss,s.metrics.last_value_loss);
        assert_eq!(record, expected_rows[update-4097]); writeln!(log, "{record}").unwrap();
        if [4097, WINDOW_FIRST, 4137, WINDOW_LAST].contains(&update) { compare_weights(&s, &root, &online, update); }
        if update >= WINDOW_FIRST {
            save_state(&incoming, &dir, 0); save_state(&s, &dir, 1);
            let (score, steps) = window_score(&s, update, &mut window); window_steps += steps;
            let drop = previous.map_or(0.0, |v: f64| v - score);
            writeln!(curve, "{update},{score},{drop}").unwrap();
            if update > WINDOW_FIRST && replace_witness(best.as_ref().map(|v| (v.update, v.drop)), update, drop) {
                best = Some(LateWitness { update, drop, incoming, original, batch, samples: captured.samples });
            }
            previous = Some(score); window.flush().unwrap(); curve.flush().unwrap();
            println!("LATE WINDOW {update} {score} {drop}");
        }
        assert_eq!(s.metrics.total_updates, update); assert_eq!(s.metrics.total_env_steps, update*512);
    }
    log.flush().unwrap(); window.flush().unwrap(); curve.flush().unwrap();
    exact_window_anchors(&root.join("window.csv"), &prior.join("online/evaluation.csv"));
    fs::write(root.join("ordinary-costs.json"), cost_json(ordinary_cost)).unwrap();
    if let Some(witness) = best {
        fs::write(root.join("selection.json"), format!("{{\"update\":{},\"drop\":{},\"selection_domain\":\"0x72000000\",\"confirmation_domain\":\"0x74000000\",\"diagnostic_selection_only\":true}}\n", witness.update, witness.drop)).unwrap();
        let (credit_steps, confirmation_steps) = causal_replay(&witness, &root);
        fs::write(root.join("complete.json"), format!("{{\"seed\":41004,\"history_exact\":true,\"continuation_updates\":128,\"window_records\":18624,\"window_steps\":{window_steps},\"selected_update\":{},\"pair_records\":8192,\"credit_steps\":{credit_steps},\"confirmation_records\":4608,\"confirmation_steps\":{confirmation_steps},\"sham_exact\":true,\"candidate_continued\":false}}\n", witness.update)).unwrap();
    } else {
        fs::write(root.join("complete.json"), format!("{{\"seed\":41004,\"history_exact\":true,\"window_records\":18624,\"window_steps\":{window_steps},\"selected_update\":null,\"reason\":\"no_positive_drop\"}}\n")).unwrap();
    }
    println!("LATE NOMINAL COMPLETE 41004");
}

#[test]
fn selection_is_positive_largest_drop_and_earliest_tie() {
    assert!(!replace_witness(None, 4129, 0.0));
    assert!(!replace_witness(None, 4129, -1.0));
    assert!(replace_witness(None, 4129, 0.01));
    assert!(replace_witness(Some((4140, 0.1)), 4141, 0.2));
    assert!(!replace_witness(Some((4140, 0.1)), 4141, 0.1));
    assert!(replace_witness(Some((4140, 0.1)), 4139, 0.1));
}

#[test]
fn selection_rejects_invalid_update_and_nonfinite_scores() {
    assert!(std::panic::catch_unwind(|| replace_witness(None, 4128, 1.0)).is_err());
    assert!(std::panic::catch_unwind(|| replace_witness(None, 4225, 1.0)).is_err());
    assert!(std::panic::catch_unwind(|| replace_witness(None, 4129, f64::NAN)).is_err());
}

#[test]
fn observed_update_replays_exactly_with_live_moments() {
    let path = std::env::temp_dir().join(format!("late-nominal-sham-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), 203);
    s.train_updates(1);
    let incoming = clone_session(&s);
    let scope = Scope::enter();
    late_nominal_adam::start(path.join("moments"));
    update_support(&mut s, 203, 2, 1024, Some(&path.join("ordinary")));
    late_nominal_adam::stop();
    let captured = scope.finish();
    let batch = captured.batch.unwrap();
    let sham = fit_from(&incoming, &s, &batch, &path.join("sham"));
    assert_eq!(fingerprint(&sham), fingerprint(&s));
    assert_eq!(compare_tree(&path.join("sham"), &path.join("ordinary/optimizer")), 48);
    let mut a = clone_session(&sham); let mut b = clone_session(&s);
    a.config.ppo.mini_batch_size = 256; b.config.ppo.mini_batch_size = 256;
    a.optimize(&batch); b.optimize(&batch);
    assert_eq!(fingerprint(&a), fingerprint(&b));
    fs::remove_dir_all(path).unwrap();
}
