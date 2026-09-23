//! Full native live-Adam replay for two independently learned value baselines.
//!
//! The historical actor/data/minibatch order/Adam history and critic targets are
//! unchanged. Only actor advantages use fixed predictions from critics fitted in
//! the already-completed frozen-policy generalization pilot.
use super::*;

const SEED: u64 = 41006;
const TARGET: usize = 4140;
const EVAL_DOMAIN: u64 = 0x6100_0000;

fn read_fixed_values(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len(), 1024 * 4, "learned baseline byte count");
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "learned baseline bytes not f32 aligned");
    let values = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 1024);
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

/// Identical, borrowed inputs shared by both baseline-only interventions.
struct CandidateReplay<'a> {
    incoming: &'a PpoTrainerSession,
    original: &'a PpoTrainerSession,
    batch: &'a RolloutBatch,
    current_values: &'a [f32],
    current_raw: &'a [f32],
    current_norm: &'a [f32],
    original_dir: &'a Path,
    out: &'a Path,
}

fn run_candidate(
    label: &str,
    values: &[f32],
    replay: &CandidateReplay<'_>,
) -> PpoTrainerSession {
    let CandidateReplay {
        incoming,
        original,
        batch,
        current_values,
        current_raw,
        current_norm,
        original_dir,
        out,
    } = *replay;
    assert_eq!(values.len(), 1024);
    let raw = batch
        .returns
        .iter()
        .zip(values)
        .map(|(ret, value)| *ret - *value)
        .collect::<Vec<_>>();
    let learned_norm = normalize(&raw);

    let mut diagnostic = BufWriter::new(
        fs::File::create(out.join(format!("baseline-{label}.csv"))).unwrap(),
    );
    writeln!(
        diagnostic,
        "row,return,current_value,learned_value,return_minus_current_value,learned_raw,current_norm,learned_norm"
    )
    .unwrap();
    for row in 0..1024 {
        writeln!(
            diagnostic,
            "{row},{},{},{},{},{},{},{}",
            batch.returns[row],
            current_values[row],
            values[row],
            current_raw[row],
            raw[row],
            current_norm[row],
            learned_norm[row]
        )
        .unwrap();
    }
    diagnostic.flush().unwrap();

    let mut candidate_batch = batch.clone();
    candidate_batch.advantages = learned_norm;
    save_union(
        &out.join(format!("{label}-baseline.json")),
        &candidate_batch,
        &raw,
    );

    let optimizer_dir = out.join(format!("{label}-optimizer"));
    let candidate = fit_from(incoming, original, &candidate_batch, &optimizer_dir);

    for step in 1..=16 {
        let file = format!("critic-{step}.bin");
        assert_eq!(
            fs::read(optimizer_dir.join(&file)).unwrap(),
            fs::read(original_dir.join("optimizer").join(&file)).unwrap(),
            "{label} critic differs at optimizer step {step}"
        );
    }
    assert_eq!(
        candidate.update_rng.clone().gen::<[u64; 4]>(),
        original.update_rng.clone().gen::<[u64; 4]>(),
        "{label} consumed a different optimizer shuffle stream"
    );
    candidate
}

#[test]
#[ignore = "explicit learned-baseline live-Adam replay; never controller qualification"]
fn emit_learned_baseline_live_step() {
    let out = PathBuf::from(std::env::var("LEARNED_BASELINE_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("LEARNED_BASELINE_PRIOR").unwrap());
    let uniform_path =
        PathBuf::from(std::env::var("LEARNED_BASELINE_UNIFORM_VALUES").unwrap());
    let balanced_path =
        PathBuf::from(std::env::var("LEARNED_BASELINE_BALANCED_VALUES").unwrap());
    fs::create_dir_all(&out).unwrap();

    let old_text = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old_rows: Vec<_> = old_text.lines().skip(1).collect();
    assert_eq!(old_rows.len(), 4224);

    let mut session = PpoTrainerSession::new_seeded(learning_config("gamma995"), SEED);
    compare_weights(&session, &out, &prior, 0);
    for update in 1..TARGET {
        let costs = update_support(&mut session, SEED, update, 1024, None);
        assert_eq!(
            update_row(&session, update, costs),
            old_rows[update - 1],
            "prefix differs at {update}"
        );
        if [1024, 4096, 4128, TARGET - 1].contains(&update) {
            compare_weights(&session, &out, &prior, update);
        }
        if update % 512 == 0 {
            println!("LEARNED BASELINE PREFIX {SEED} {update}");
        }
    }

    let incoming = clone_session(&session);
    let incoming_actor = incoming.snapshot();
    let incoming_fingerprint = fingerprint(&incoming);

    let original_dir = out.join("original");
    fs::create_dir_all(&original_dir).unwrap();
    let scope = Scope::enter();
    let costs = update_support(
        &mut session,
        SEED,
        TARGET,
        1024,
        Some(&original_dir),
    );
    let captured = scope.finish();
    let batch = captured.batch.expect("captured fitting batch");
    assert_eq!(captured.samples.len(), 1024);
    assert_eq!(batch.observations.len(), 1024);
    assert_eq!(update_row(&session, TARGET, costs), old_rows[TARGET - 1]);
    compare_weights(&session, &out, &prior, TARGET);
    let original = clone_session(&session);

    // Bind the actual collection/target/optimizer records, including union.json's
    // raw and normalized advantages, to the immutable historical transaction.
    let compared = compare_tree(&original_dir, &prior.join(format!("update-{TARGET}")));
    assert!(compared > 0, "no historical transaction files compared");
    fs::write(out.join("historical-files-compared.txt"), format!("{compared}\n")).unwrap();
    println!("LEARNED BASELINE HISTORICAL FILES IDENTICAL {compared}");

    let current_values = batch
        .observations
        .iter()
        .map(|observation| predict_value(&incoming, *observation))
        .collect::<Vec<_>>();
    // Diagnostic only: R = fl(A + V), so fl(R - V) need not recover A's bits.
    // Keep the collector's original normalized advantages for the sham and logs.
    // Each registered candidate still uses normalize(original R - learned V).
    let current_raw = batch
        .returns
        .iter()
        .zip(&current_values)
        .map(|(ret, value)| *ret - *value)
        .collect::<Vec<_>>();
    let current_norm = batch.advantages.clone();

    let sham = fit_from(&incoming, &original, &batch, &out.join("sham-optimizer"));
    assert_eq!(
        fingerprint(&sham),
        fingerprint(&original),
        "sham must reproduce historical update4140"
    );

    let uniform_values = read_fixed_values(&uniform_path);
    let balanced_values = read_fixed_values(&balanced_path);
    let replay = CandidateReplay {
        incoming: &incoming,
        original: &original,
        batch: &batch,
        current_values: &current_values,
        current_raw: &current_raw,
        current_norm: &current_norm,
        original_dir: &original_dir,
        out: &out,
    };
    let uniform = run_candidate("learned-uniform", &uniform_values, &replay);
    let balanced = run_candidate("learned-early-balanced", &balanced_values, &replay);
    assert_eq!(
        fingerprint(&incoming),
        incoming_fingerprint,
        "diagnostic mutated incoming session"
    );

    let policies = [
        ("incoming", TARGET - 1, incoming_actor),
        ("original", TARGET, original.snapshot()),
        ("learned-uniform", TARGET, uniform.snapshot()),
        ("learned-early-balanced", TARGET, balanced.snapshot()),
    ];
    let mut evaluation =
        BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut evaluation);
    let mut evaluation_steps = 0usize;
    for (name, checkpoint, actor) in &policies {
        save(&out.join(format!("actor-{name}.bin")), actor);
        evaluation_steps += evaluate_policy(
            actor,
            SEED,
            EVAL_DOMAIN,
            name,
            *checkpoint,
            512,
            Some(&out),
            None,
            &mut evaluation,
        )
        .steps;
    }
    evaluation.flush().unwrap();

    fs::write(
        out.join("complete.json"),
        format!(
            concat!(
                "{{\"seed\":{SEED},\"target_update\":{TARGET},",
                "\"training_updates\":0,\"fitting_rows\":1024,",
                "\"learned_candidates\":2,\"actor_advantages_only_changed\":true,",
                "\"critic_targets_unchanged\":true,\"critic_transactions_identical\":true,",
                "\"sham_exact\":true,\"evaluation_domain\":\"0x61000000\",",
                "\"evaluation_records\":6144,\"evaluation_steps\":{evaluation_steps},",
                "\"candidate_training_continued\":false}}\n"
            ),
            SEED = SEED,
            TARGET = TARGET,
            evaluation_steps = evaluation_steps,
        ),
    )
    .unwrap();
    println!("LEARNED BASELINE LIVE STEP COMPLETE {SEED} {TARGET}");
}

#[test]
fn fixed_value_parser_rejects_wrong_size() {
    let path = std::env::temp_dir().join("rustrobotics-learned-baseline-short.bin");
    fs::write(&path, [0_u8; 8]).unwrap();
    assert!(std::panic::catch_unwind(|| read_fixed_values(&path)).is_err());
    let _ = fs::remove_file(path);
}

#[test]
fn rounded_return_is_not_an_exact_advantage_archive() {
    // Exact row-0 bits from the original update4140 artifact.
    let advantage = f32::from_bits(0x4010_0f8d);
    let value = f32::from_bits(0x4343_d10a);
    let ret = advantage + value;
    assert_eq!(ret.to_bits(), 0x4346_1148);
    assert_ne!((ret - value).to_bits(), advantage.to_bits());
    assert_eq!((ret - value) + value, ret);
}

#[test]
fn historical_file_binding_rejects_changed_bytes() {
    let root = std::env::temp_dir().join(format!(
        "rustrobotics-learned-historical-binding-{}",
        std::process::id()
    ));
    let actual = root.join("actual");
    let expected = root.join("expected");
    fs::create_dir_all(&actual).unwrap();
    fs::create_dir_all(&expected).unwrap();
    fs::write(actual.join("union.json"), b"captured normalized advantages").unwrap();
    fs::write(expected.join("union.json"), b"captured normalized advantages").unwrap();
    assert_eq!(compare_tree(&actual, &expected), 1);
    fs::write(actual.join("union.json"), b"recomputed rounded advantages").unwrap();
    assert!(std::panic::catch_unwind(|| compare_tree(&actual, &expected)).is_err());
    fs::remove_dir_all(root).unwrap();
}
