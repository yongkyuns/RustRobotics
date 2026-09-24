//! Read-only reconstruction of already-recorded loss witnesses.
//! Saved snapshots are NOT optimizer-resume checkpoints.
use super::*;

fn decode_actor(bytes: &[u8], template: &PolicySnapshot) -> PolicySnapshot {
    assert_eq!(bytes.len(), flat(template).len() * 4, "snapshot byte length");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()), "nonfinite snapshot");
    let mut actor = template.clone();
    let mut offset = 0;
    for layer in [&mut actor.input, &mut actor.hidden, &mut actor.output] {
        for target in [&mut layer.weight, &mut layer.bias] {
            let end = offset + target.len();
            target.copy_from_slice(&values[offset..end]);
            offset = end;
        }
    }
    assert_eq!(offset, values.len());
    assert_eq!(flat(&actor), values);
    actor
}

fn replay_panel(panel: &str) -> (&str, bool) {
    match panel {
        "stress-det" => ("reset-det", true),
        "stress-stoch" => ("reset-stoch", true),
        "reset-det" | "reset-stoch" | "outward-stoch" | "long-det" | "long-stoch"
        | "outward-long" => (panel, false),
        _ => panic!("unknown replay panel"),
    }
}

#[test]
fn replay_snapshot_roundtrip_preserves_every_weight_and_inference() {
    let template = PpoTrainerSession::new_seeded(learning_config("gamma995"), 41002).snapshot();
    let bytes: Vec<u8> = flat(&template).into_iter().flat_map(f32::to_le_bytes).collect();
    let actor = decode_actor(&bytes, &template);
    assert_eq!(actor, template);
    for observation in [[0.0; 4], [0.1, -0.3, 0.2, -0.4], [-1.8, 3.0, -0.4, 1.0]] {
        assert_eq!(mean(&actor, observation), mean(&template, observation));
    }
    for panel in ["reset-det", "outward-stoch", "stress-stoch"] {
        let (native, stress) = replay_panel(panel);
        let _scope = stress.then(hooks::Evaluation::enter);
        let seed = 41002 + COVERAGE_EVAL + if stress { STRESS_OFFSET } else { 0 };
        let mut a = Vec::new();
        let mut b = Vec::new();
        assert_eq!(eval_one(&actor, seed, native, 7, Some(&mut a)),
                   eval_one(&template, seed, native, 7, Some(&mut b)));
        assert_eq!(a, b);
    }
}

#[test]
#[should_panic(expected = "snapshot byte length")]
fn replay_snapshot_rejects_wrong_length() {
    let template = PpoTrainerSession::new_seeded(learning_config("gamma995"), 41002).snapshot();
    decode_actor(&[], &template);
}

#[test]
#[should_panic(expected = "nonfinite snapshot")]
fn replay_snapshot_rejects_nonfinite_weights() {
    let template = PpoTrainerSession::new_seeded(learning_config("gamma995"), 41002).snapshot();
    let mut bytes: Vec<u8> = flat(&template).into_iter().flat_map(f32::to_le_bytes).collect();
    bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    decode_actor(&bytes, &template);
}

#[test]
#[ignore = "exposed-case evaluation replay; never a training or production gate"]
fn emit_reset_failure_replay() {
    let root = PathBuf::from(std::env::var("REPLAY_INPUT").unwrap());
    let out = PathBuf::from(std::env::var("REPLAY_OUT").unwrap());
    let requests = fs::read_to_string(out.join("requests.csv")).unwrap();
    let mut lines = requests.lines();
    assert_eq!(lines.next(), Some("seed,arm,checkpoint,panel,rep"));
    // Use native configuration only for shape/metadata. No train call or state import.
    let template = PpoTrainerSession::new_seeded(learning_config("gamma995"), 41001).snapshot();
    assert_eq!(template.action_std, 2.0);
    assert_eq!(template.action_limit, 20.0);
    let mut log = BufWriter::new(fs::File::create(out.join("replayed.csv")).unwrap());
    eval_header(&mut log);
    let mut identities = std::collections::BTreeSet::new();
    let mut steps = 0;
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 5);
        let seed: u64 = fields[0].parse().unwrap();
        let arm = fields[1];
        let checkpoint: usize = fields[2].parse().unwrap();
        let panel = fields[3];
        let rep: usize = fields[4].parse().unwrap();
        assert!(COVERAGE_SEEDS.contains(&seed) && COVERAGE_ARMS.contains(&arm));
        assert!([4096, 4608].contains(&checkpoint));
        let (native, stress) = replay_panel(panel);
        let (_, _, cap, _) = panel_spec(native);
        let n = if cap > 2048 { 32 } else if checkpoint == 4608 { 256 } else { 64 };
        assert!(rep < n && (cap <= 2048 || checkpoint == 4608));
        let id = format!("{seed}_{arm}_{checkpoint}_{panel}_{rep}");
        assert!(identities.insert(id.clone()), "duplicate request");
        let path = root.join(format!("reset-coverage-run-{seed}-{arm}"))
            .join(format!("actor-{checkpoint}.bin"));
        let actor = decode_actor(&fs::read(path).unwrap(), &template);
        let mut trace = BufWriter::new(fs::File::create(out.join(format!("trace-{id}.csv"))).unwrap());
        let _scope = stress.then(hooks::Evaluation::enter);
        let r = eval_one(&actor, seed + COVERAGE_EVAL + if stress { STRESS_OFFSET } else { 0 },
                         native, rep, Some(&mut trace));
        trace.flush().unwrap();
        writeln!(log, "{seed},{arm},{checkpoint},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                 r.key, r.steps, r.ending, r.total, r.discounted, r.max_position,
                 r.max_angle, r.centered, r.force_rms).unwrap();
        steps += r.steps;
    }
    log.flush().unwrap();
    assert!(!identities.is_empty());
    fs::write(out.join("native-complete.json"), format!(
        "{{\"cases\":{},\"simulation_steps\":{steps},\"training_updates\":0}}\n", identities.len())).unwrap();
    println!("RESET FAILURE REPLAY COMPLETE {} cases {} steps; zero training updates", identities.len(), steps);
}
