//! Observation-only replay of two exposed learning histories, not a new learner.
use super::*;
use std::collections::BTreeMap;

const FIRST: usize = 4128;
const LAST: usize = 4224;
const PANELS: [&str; 3] = ["reset-det", "reset-stoch", "outward-stoch"];
const FRESH_DOMAIN: u64 = 0x2000_0000;
type Historical = BTreeMap<(usize, String, usize), String>;

fn row(seed: u64, arm: &str, cp: usize, panel: &str, rep: usize, r: &Eval) -> String {
    let (_, _, cap, _) = panel_spec(panel);
    format!("{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
        r.key, r.steps, r.ending, r.total, r.discounted, r.max_position,
        r.max_angle, r.centered, r.force_rms)
}
fn historical(text: &str, seed: u64) -> Historical {
    let mut result = BTreeMap::new();
    for line in text.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[0].parse::<u64>().unwrap(), seed);
        assert_eq!(fields[1], "support1024");
        let key = (fields[2].parse().unwrap(), fields[3].to_owned(), fields[4].parse().unwrap());
        assert!(result.insert(key, line.to_owned()).is_none(), "duplicate historical case");
    }
    assert_eq!(result.len(), 1344);
    result
}
struct Evaluation { counts: [usize; 3], steps: usize, compared: usize }
fn evaluate_policy(
    policy: &PolicySnapshot, seed: u64, domain: u64, arm: &str, cp: usize,
    reps: usize, trace_dir: Option<&Path>, old: Option<&Historical>, log: &mut dyn Write,
) -> Evaluation {
    let mut result = Evaluation { counts: [0; 3], steps: 0, compared: 0 };
    for (index, panel) in PANELS.into_iter().enumerate() {
        for rep in 0..reps {
            let mut trace = trace_dir.filter(|_| rep == 0).map(|path|
                BufWriter::new(fs::File::create(path.join(format!("trace-{arm}-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(policy, seed + domain, panel, rep, trace.as_mut().map(|v| v as &mut dyn Write));
            let record = row(seed, arm, cp, panel, rep, &r);
            if let Some(reference) = old {
                assert_eq!(&record, reference.get(&(cp, panel.to_owned(), rep)).expect("missing historical case"),
                    "historical evaluation mismatch");
                result.compared += 1;
            }
            result.counts[index] += usize::from(r.ending == "timeout");
            result.steps += r.steps;
            writeln!(log, "{record}").unwrap();
        }
    }
    result
}
#[derive(Debug)]
struct Selection { update: usize, drop: i32, before: PolicySnapshot, after: PolicySnapshot }
fn replace_selection(current: Option<(usize, i32)>, update: usize, drop: i32) -> bool {
    match current { None => true, Some((previous, best)) => { assert!(update > previous); drop > best } }
}
fn update_row(s: &PpoTrainerSession, u: usize, c: Costs) -> String {
    format!("support1024,{u},{u},{},{},{},{},{},{},{},{}", c.primary, c.tails, c.supplement,
        c.actor_steps, c.sample_visits, s.metrics.total_episodes,
        s.metrics.last_policy_loss, s.metrics.last_value_loss)
}
fn compare_weights(s: &PpoTrainerSession, out: &Path, prior: &Path, cp: usize) {
    save_state(s, out, cp);
    for kind in ["actor", "critic"] {
        let name = format!("{kind}-{cp}.bin");
        assert_eq!(fs::read(out.join(&name)).unwrap(), fs::read(prior.join(&name)).unwrap(),
            "history checkpoint mismatch at {cp}: {kind}");
    }
}
#[test]
#[ignore = "explicit historical-window measurement, never a policy acceptance gate"]
fn emit_regression_window() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!([41004, 41006].contains(&seed));
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let old_text = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old_rows: Vec<_> = old_text.lines().skip(1).collect();
    assert_eq!(old_rows.len(), 4608);
    let old_eval = historical(&fs::read_to_string(prior.join("evaluation.csv")).unwrap(), seed);
    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), seed);
    compare_weights(&s, &out, &prior, 0);
    let mut metrics = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(metrics, "arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut costs = Costs::default();
    let mut weight_checks = 2;
    for u in 1..=FIRST {
        let c = update_support(&mut s, seed, u, 1024, None);
        costs.add(c);
        let record = update_row(&s, u, c);
        assert_eq!(record, old_rows[u - 1], "prefix update mismatch at {u}");
        writeln!(metrics, "{record}").unwrap();
        if [1024, 4096, FIRST].contains(&u) { compare_weights(&s, &out, &prior, u); weight_checks += 2; }
        if u % 256 == 0 { metrics.flush().unwrap(); println!("WINDOW PREFIX {seed} {u}"); }
    }
    let mut evaluations = BufWriter::new(fs::File::create(out.join("window-evaluation.csv")).unwrap());
    eval_header(&mut evaluations);
    let unchanged = fingerprint(&s);
    let initial = evaluate_policy(&s.snapshot(), seed, SUPPORT_DOMAIN, "support1024", FIRST, 64,
        Some(&out), Some(&old_eval), &mut evaluations);
    assert_eq!(unchanged, fingerprint(&s), "initial evaluation mutated learner");
    let mut previous = initial.counts;
    let mut evaluation_steps = initial.steps;
    let mut historical_evaluations = initial.compared;
    let mut selected: Option<Selection> = None;
    let mut deltas = BufWriter::new(fs::File::create(out.join("window-deltas.csv")).unwrap());
    writeln!(deltas, "update,nominal_det_before,nominal_det_after,nominal_stoch_before,nominal_stoch_after,outward_before,outward_after,outward_drop").unwrap();
    for u in FIRST + 1..=LAST {
        let before_policy = s.snapshot();
        let path = out.join(format!("update-{u}"));
        fs::create_dir_all(&path).unwrap();
        save_state(&s, &path, 0);
        fs::write(path.join("training-gamma.txt"), "0.995\n").unwrap();
        fs::write(path.join("support-steps.txt"), "1024\n").unwrap();
        let c = update_support(&mut s, seed, u, 1024, Some(&path));
        costs.add(c);
        let record = update_row(&s, u, c);
        assert_eq!(record, old_rows[u - 1], "instrumented update mismatch at {u}");
        assert_eq!(s.metrics.total_updates, u);
        assert_eq!(s.metrics.total_env_steps, u * 512);
        assert_eq!(c.actor_steps, 16);
        assert_eq!(c.sample_visits, 4096);
        writeln!(metrics, "{record}").unwrap();
        save_state(&s, &out, u);
        let actor = s.snapshot();
        let unchanged = fingerprint(&s);
        let measurement = evaluate_policy(&actor, seed, SUPPORT_DOMAIN, "support1024", u, 64,
            (u == LAST).then_some(out.as_path()), (u == LAST).then_some(&old_eval), &mut evaluations);
        assert_eq!(unchanged, fingerprint(&s), "window evaluation mutated learner");
        evaluation_steps += measurement.steps;
        historical_evaluations += measurement.compared;
        let next = measurement.counts;
        let drop = previous[2] as i32 - next[2] as i32;
        writeln!(deltas, "{u},{},{},{},{},{},{},{drop}", previous[0], next[0], previous[1], next[1], previous[2], next[2]).unwrap();
        if replace_selection(selected.as_ref().map(|v| (v.update, v.drop)), u, drop) {
            selected = Some(Selection { update: u, drop, before: before_policy, after: actor });
        }
        previous = next;
        evaluations.flush().unwrap(); metrics.flush().unwrap(); deltas.flush().unwrap();
        println!("WINDOW UPDATE {seed} {u} {} {} {}", next[0], next[1], next[2]);
    }
    compare_weights(&s, &out, &prior, LAST); weight_checks += 2;
    let selection = selected.unwrap();
    fs::write(out.join("selected.json"), format!(concat!(
        "{{\"seed\":{},\"update\":{},\"outward_completion_drop\":{},",
        "\"rule\":\"largest consecutive outward count drop; earliest tie\",",
        "\"historical_domain\":{},\"independent_domain\":{},\"selection_is_not_policy_acceptance\":true}}\n"),
        seed, selection.update, selection.drop, SUPPORT_DOMAIN, FRESH_DOMAIN)).unwrap();
    let independent = out.join("independent"); fs::create_dir_all(&independent).unwrap();
    save(&independent.join("actor-before.bin"), &selection.before);
    save(&independent.join("actor-after.bin"), &selection.after);
    let mut log = BufWriter::new(fs::File::create(independent.join("evaluation.csv")).unwrap());
    eval_header(&mut log);
    let unchanged = fingerprint(&s);
    let mut independent_steps = 0;
    for (name, cp, actor) in [("before", selection.update - 1, &selection.before), ("after", selection.update, &selection.after)] {
        independent_steps += evaluate_policy(actor, seed, FRESH_DOMAIN, name, cp, 256, Some(&independent), None, &mut log).steps;
    }
    log.flush().unwrap();
    assert_eq!(unchanged, fingerprint(&s), "independent evaluation mutated learner");
    assert_eq!(historical_evaluations, 384);
    assert_eq!(weight_checks, 10);
    assert_eq!(costs.primary, LAST * 512);
    assert_eq!(costs.supplement, LAST * 8 * 1088);
    assert_eq!(costs.actor_steps, LAST * 16);
    assert_eq!(costs.sample_visits, LAST * 4096);
    fs::write(out.join("costs.json"), cost_json(costs)).unwrap();
    fs::write(out.join("complete.json"), format!(concat!(
        "{{\"seed\":{},\"execution\":\"complete\",\"first\":4128,\"last\":4224,",
        "\"historical_updates_exact\":4224,\"historical_weights_exact\":10,\"historical_evaluations_exact\":384,",
        "\"instrumented_updates\":96,\"window_evaluation_records\":18624,\"independent_evaluation_records\":1536,",
        "\"window_evaluation_steps\":{},\"independent_evaluation_steps\":{},\"selected_update\":{},",
        "\"selected_outward_drop\":{},\"candidate_training\":false,\"portable_optimizer_resume_exported\":false}}\n"),
        seed, evaluation_steps, independent_steps, selection.update, selection.drop)).unwrap();
    println!("REGRESSION WINDOW COMPLETE {seed}");
}

#[test]
fn maximum_drop_selection_is_fixed_including_ties_and_improvements() {
    assert!(replace_selection(None, 4129, -3));
    assert!(!replace_selection(Some((4129, -3)), 4130, -3));
    assert!(replace_selection(Some((4129, -3)), 4130, -2));
    assert!(!replace_selection(Some((4130, 4)), 4131, 2));
    assert!(replace_selection(Some((4130, 4)), 4131, 5));
}
#[test]
fn full_trace_observer_preserves_next_real_optimizer_update() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 203);
    let mut b = clone_session(&a);
    let path = std::env::temp_dir().join(format!("rr-window-observer-{}", std::process::id()));
    assert!(!path.exists()); fs::create_dir(&path).unwrap();
    let x = update_support(&mut a, 203, 1, 1024, Some(&path));
    let y = update_support(&mut b, 203, 1, 1024, None);
    assert_eq!(x, y); assert_eq!(fingerprint(&a), fingerprint(&b));
    let x = update_support(&mut a, 203, 4097, 1024, None);
    let y = update_support(&mut b, 203, 4097, 1024, None);
    assert_eq!(x, y); assert_eq!(fingerprint(&a), fingerprint(&b));
    fs::remove_dir_all(path).unwrap();
}
#[test]
fn frozen_panel_readout_does_not_change_next_training() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma995"), 204);
    update_support(&mut a, 204, 1, 1024, None);
    let mut b = clone_session(&a); let mut log = Vec::new();
    let before = fingerprint(&a);
    let r = evaluate_policy(&a.snapshot(), 41004, FRESH_DOMAIN, "before", 4128, 2, None, None, &mut log);
    assert!(r.steps > 0); assert_eq!(r.compared, 0); assert_eq!(before, fingerprint(&a));
    update_support(&mut a, 204, 4097, 1024, None);
    update_support(&mut b, 204, 4097, 1024, None);
    assert_eq!(fingerprint(&a), fingerprint(&b));
}
