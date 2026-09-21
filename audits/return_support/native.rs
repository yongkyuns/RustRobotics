//! From-scratch gamma .995 comparison; only sampled return support differs.
use super::*;
use crate::trainer::support_window_audit::Window;

const SUPPORT_ARMS: [(&str, usize); 2] = [("support512", 512), ("support1024", 1024)];
const SUPPORT_DOMAIN: u64 = 0x1000_0000;
fn support_length(arm: &str) -> usize {
    SUPPORT_ARMS.iter().find(|(name, _)| *name == arm).expect("unknown support arm").1
}
fn update_support(s: &mut PpoTrainerSession, seed: u64, u: usize, n: usize, out: Option<&Path>) -> Costs {
    let _window = Window::enter(n);
    one_update(s, seed, u, u > 4096, out)
}
fn evaluate_support(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, out: &Path, log: &mut dyn Write) -> usize {
    let before = fingerprint(s);
    let policy = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == TOTAL_UPDATES { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut steps = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        for rep in 0..64 {
            let mut trace = (rep == 0 && [4096, TOTAL_UPDATES].contains(&cp)).then(||
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(&policy, seed + SUPPORT_DOMAIN, panel, rep, trace.as_mut().map(|t| t as &mut dyn Write));
            writeln!(log, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key, r.steps, r.ending, r.total, r.discounted, r.max_position, r.max_angle, r.centered, r.force_rms).unwrap();
            steps += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s), "evaluation changed learner");
    steps
}
fn compare_checkpoint(out: &Path, prior: &Path, cp: usize) -> usize {
    for kind in ["actor", "critic"] {
        let name = format!("{kind}-{cp}.bin");
        assert_eq!(fs::read(out.join(&name)).unwrap(), fs::read(prior.join(&name)).unwrap(),
            "historical short-support checkpoint mismatch: {name}");
    }
    2
}
#[test]
#[ignore = "registered support-length learning experiment; not production qualification"]
fn emit_return_support() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((41001..=41008).contains(&seed));
    let arm = std::env::var("SUPPORT_ARM").unwrap();
    let n = support_length(&arm);
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), seed);
    assert_eq!(s.config.ppo.gamma, 0.995);
    let old_updates = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old_rows: Vec<_> = old_updates.lines().skip(1).collect();
    assert_eq!(old_rows.len(), TOTAL_UPDATES);
    fs::write(out.join("config.json"), format!(concat!(
        "{{\"seed\":{},\"arm\":\"{}\",\"training_gamma\":0.995,\"evaluation_gamma\":0.99,",
        "\"lambda\":1,\"updates\":4608,\"ordinary_reset_updates\":4096,\"outward_mix_updates\":512,",
        "\"checkpoints\":[0,1024,4096,4128,4224,4608],\"detailed_updates\":[1,4097,4608],",
        "\"primary_rows\":512,\"supplement_rows\":512,\"supplement_streams\":8,",
        "\"stream_steps\":{},\"selected_per_stream\":64,\"tail_draws\":4,\"tail_steps\":{},",
        "\"minibatch\":256,\"epochs\":4,\"lr\":0.0003,\"epsilon\":1e-5,",
        "\"normalization\":\"global\",\"repetitions\":64,\"evaluation_seed\":{},",
        "\"from_scratch\":true,\"cohort\":\"exposed-development\"}}\n"),
        seed, arm, 64+n, n, seed+SUPPORT_DOMAIN)).unwrap();
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    save_state(&s, &out, 0);
    let mut comparisons = if n == 512 { compare_checkpoint(&out, &prior, 0) } else { 0 };
    let mut evaluation_steps = evaluate_support(&s, seed, &arm, 0, &out, &mut eval);
    eval.flush().unwrap();
    let mut log = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(log, "arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut costs = Costs::default();
    let mut update_comparisons = 0;
    for u in 1..=TOTAL_UPDATES {
        let path = [1,4097,TOTAL_UPDATES].contains(&u).then(||out.join(format!("update-{u}")));
        if let Some(path) = &path {
            fs::create_dir_all(path).unwrap(); save_state(&s, path, 0);
            fs::write(path.join("training-gamma.txt"), "0.995\n").unwrap();
            fs::write(path.join("support-steps.txt"), format!("{n}\n")).unwrap();
        }
        let c = update_support(&mut s, seed, u, n, path.as_deref());
        assert_eq!(crate::trainer::support_window_audit::steps(512), 512);
        assert_eq!(s.config.ppo.gamma, 0.995);
        assert_eq!(s.metrics.total_env_steps, u*512);
        assert_eq!(s.metrics.total_updates, u);
        assert_eq!(c.supplement, 8*(64+n));
        assert!(c.tails <= 4*n);
        costs.add(c);
        let row = format!("{arm},{u},{u},{},{},{},{},{},{},{},{}", c.primary, c.tails, c.supplement,
            c.actor_steps, c.sample_visits, s.metrics.total_episodes, s.metrics.last_policy_loss, s.metrics.last_value_loss);
        if n == 512 {
            assert_eq!(row.split_once(',').unwrap().1, old_rows[u-1].split_once(',').unwrap().1,
                "historical update record mismatch at {u}");
            update_comparisons += 1;
        }
        writeln!(log, "{row}").unwrap();
        if u % 256 == 0 { log.flush().unwrap(); println!("SUPPORT PROGRESS {seed} {arm} {u}"); }
        if LEARNING_CHECKPOINTS.contains(&u) {
            save_state(&s, &out, u);
            if n == 512 { comparisons += compare_checkpoint(&out, &prior, u); }
            evaluation_steps += evaluate_support(&s, seed, &arm, u, &out, &mut eval);
            eval.flush().unwrap(); fs::write(out.join("costs.json"), cost_json(costs)).unwrap();
        }
    }
    log.flush().unwrap();
    assert_eq!(costs.primary, 2359296);
    assert_eq!(costs.supplement, TOTAL_UPDATES*8*(64+n));
    assert_eq!(costs.actor_steps, 73728);
    assert_eq!(costs.sample_visits, 18874368);
    assert_eq!(comparisons, if n == 512 {12} else {0});
    assert_eq!(update_comparisons, if n == 512 {4608} else {0});
    if n == 512 { assert_eq!(fs::read(out.join("costs.json")).unwrap(), fs::read(prior.join("costs.json")).unwrap()); }
    fs::write(out.join("complete.json"), format!(concat!(
        "{{\"seed\":{},\"arm\":\"{}\",\"execution\":\"complete\",\"from_scratch\":true,",
        "\"updates\":4608,\"primary_steps\":{},\"tail_steps\":{},\"supplement_steps\":{},",
        "\"actor_steps\":{},\"critic_steps\":{},\"sample_visits_each\":{},",
        "\"historical_weight_comparisons\":{},\"historical_update_comparisons\":{},",
        "\"evaluation_steps\":{},\"evaluation_records\":1344}}\n"),
        seed, arm, costs.primary, costs.tails, costs.supplement, costs.actor_steps, costs.actor_steps,
        costs.sample_visits, comparisons, update_comparisons, evaluation_steps)).unwrap();
    println!("RETURN SUPPORT COMPLETE {seed} {arm}");
}
#[test]
fn default_scope_reproduces_original_updates_and_following_optimizer_history() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 201);
    let mut b = clone_session(&a);
    for u in [1, 2, 4097] {
        let ca = update_support(&mut a, 201, u, 512, None);
        let cb = one_update(&mut b, 201, u, u>4096, None);
        assert_eq!(ca, cb); assert_eq!(fingerprint(&a), fingerprint(&b));
    }
    a.train_updates(1); b.train_updates(1);
    assert_eq!(fingerprint(&a), fingerprint(&b));
}
#[test]
fn longer_support_preserves_main_and_selected_supplement_paths() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 202);
    let mut b = clone_session(&a);
    let (ba, ra, ea) = { let _n = Window::enter(512); capture(&mut a) };
    let (bb, rb, eb) = { let _n = Window::enter(1024); capture(&mut b) };
    assert_eq!(ea, eb); assert_eq!(ba.observations, bb.observations);
    assert_eq!(ba.latent_actions, bb.latent_actions); assert_eq!(ba.returns, bb.returns);
    assert_eq!(ra.iter().map(|r|r.reward).collect::<Vec<_>>(), rb.iter().map(|r|r.reward).collect::<Vec<_>>());
    for mixed in [false, true] {
        let key = stream_id(202, if mixed {4097} else {1});
        let x = { let _mix = Mode::enter(mixed); let _n = Window::enter(512); supplement(&a, key, false, None) };
        let y = { let _mix = Mode::enter(mixed); let _n = Window::enter(1024); supplement(&a, key, false, None) };
        assert_eq!(x.steps, 4608); assert_eq!(y.steps, 8704);
        assert_eq!(x.batch.observations, y.batch.observations);
        assert_eq!(x.batch.latent_actions, y.batch.latent_actions);
        assert_eq!(x.batch.old_log_probs, y.batch.old_log_probs);
        assert_eq!(x.raw.len(), 512); assert_eq!(y.raw.len(), 512);
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }
}
#[test]
fn tails_retain_the_prefix_and_use_only_the_requested_cutoff() {
    let s = PpoTrainerSession::new_seeded(learning_config("gamma995"), 203);
    for d in 0..4 {
        let x = { let _n = Window::enter(512); draw_tail(&s, 203, d) };
        let y = { let _n = Window::enter(1024); draw_tail(&s, 203, d) };
        let k = x.rewards.len();
        assert_eq!(x.rewards, y.rewards[..k]);
        assert_eq!(x.observations, y.observations[..k]);
        assert_eq!(x.latents, y.latents[..k]);
        assert_eq!(x.next_states, y.next_states[..k]);
        assert!(x.terminal || k == 512);
        assert!(y.terminal || y.rewards.len() == 1024);
        if x.terminal { assert_eq!(k, y.rewards.len()); assert_eq!(x.estimate, y.estimate); }
        assert_eq!(y.estimate, discounted_tail(&y.rewards, y.bootstrap, 0.995));
    }
}
#[test]
fn support_controls_bootstrap_weight_not_true_terminal_credit() {
    let g = 0.995_f32;
    for n in [512,1024] {
        let r = vec![0.0; n]; let v = vec![0.0; n];
        let mut ended = vec![false; n]; ended[n-1] = true;
        let (terminal, _) = compute_gae(&r, &v, &ended, 1.0, g, 1.0);
        let (cutoff, _) = compute_gae(&r, &v, &vec![false;n], 1.0, g, 1.0);
        assert_eq!(terminal[0], 0.0);
        let expected = (0..n).fold(1.0_f32, |x,_| g*x);
        assert_eq!(cutoff[0], expected);
    }
    assert!(g.powi(1024) < 0.006 && g.powi(512) > 0.076);
}
#[test]
fn extended_support_runs_reproducibly_with_unchanged_optimizer_work() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 204);
    let mut b = clone_session(&a);
    for u in [1,4097] {
        let ca = update_support(&mut a, 204, u, 1024, None);
        let cb = update_support(&mut b, 204, u, 1024, None);
        assert_eq!(ca, cb); assert_eq!(ca.actor_steps, 16); assert_eq!(ca.sample_visits, 4096);
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }
    a.train_updates(1); b.train_updates(1); assert_eq!(fingerprint(&a), fingerprint(&b));
}
