//! Causal diagnostic for seed41006/update4140.
//!
//! The actor/data/minibatch order/Adam history and critic targets are unchanged.
//! Only the actor baseline is replaced by precomputed Monte-Carlo state values
//! from the already-completed paired-action-credit diagnostic.
use super::*;

const SEED: u64 = 41006;
const TARGET: usize = 4140;
const EVAL_DOMAIN: u64 = 0x6000_0000;

fn read_oracle_values(path: &Path) -> Vec<f32> {
    let text = fs::read_to_string(path).unwrap();
    let values = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 1024, "oracle value count");
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

#[test]
#[ignore = "explicit critic-baseline causality diagnostic; never controller qualification"]
fn emit_critic_baseline_causality() {
    let out = PathBuf::from(std::env::var("CRITIC_BASELINE_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("CRITIC_BASELINE_PRIOR").unwrap());
    let oracle_path = PathBuf::from(std::env::var("CRITIC_BASELINE_VALUES").unwrap());
    fs::create_dir_all(&out).unwrap();

    let old_text = fs::read_to_string(prior.join("updates.csv")).unwrap();
    let old_rows: Vec<_> = old_text.lines().skip(1).collect();
    assert_eq!(old_rows.len(), 4224);

    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), SEED);
    compare_weights(&s, &out, &prior, 0);
    for u in 1..TARGET {
        let c = update_support(&mut s, SEED, u, 1024, None);
        assert_eq!(update_row(&s, u, c), old_rows[u - 1], "prefix differs at {u}");
        if [1024, 4096, 4128, TARGET - 1].contains(&u) {
            compare_weights(&s, &out, &prior, u);
        }
        if u % 512 == 0 {
            println!("CRITIC BASELINE PREFIX {SEED} {u}");
        }
    }

    let incoming = clone_session(&s);
    let incoming_actor = incoming.snapshot();
    let incoming_fp = fingerprint(&incoming);

    let original_dir = out.join("original");
    fs::create_dir_all(&original_dir).unwrap();
    let scope = Scope::enter();
    let c = update_support(&mut s, SEED, TARGET, 1024, Some(&original_dir));
    let captured = scope.finish();
    let batch = captured.batch.expect("captured fitting batch");
    assert_eq!(captured.samples.len(), 1024);
    assert_eq!(batch.observations.len(), 1024);
    assert_eq!(update_row(&s, TARGET, c), old_rows[TARGET - 1]);
    compare_weights(&s, &out, &prior, TARGET);
    let original = clone_session(&s);

    let oracle_values = read_oracle_values(&oracle_path);
    let mut raw = Vec::with_capacity(1024);
    let mut diagnostic = BufWriter::new(fs::File::create(out.join("baseline.csv")).unwrap());
    writeln!(
        diagnostic,
        "row,return,current_value,oracle_value,current_raw,oracle_raw,current_norm,oracle_norm"
    )
    .unwrap();

    let current_values = batch
        .observations
        .iter()
        .map(|observation| predict_value(&incoming, *observation))
        .collect::<Vec<_>>();
    let current_raw = batch
        .returns
        .iter()
        .zip(&current_values)
        .map(|(ret, value)| *ret - *value)
        .collect::<Vec<_>>();
    let current_norm = normalize(&current_raw);
    assert_eq!(current_norm, batch.advantages);

    for (ret, value) in batch.returns.iter().zip(&oracle_values) {
        raw.push(*ret - *value);
    }
    let oracle_norm = normalize(&raw);

    for row in 0..1024 {
        writeln!(
            diagnostic,
            "{row},{},{},{},{},{},{},{}",
            batch.returns[row],
            current_values[row],
            oracle_values[row],
            current_raw[row],
            raw[row],
            current_norm[row],
            oracle_norm[row]
        )
        .unwrap();
    }
    diagnostic.flush().unwrap();

    let mut candidate_batch = batch.clone();
    candidate_batch.advantages = oracle_norm;
    save_union(&out.join("oracle-baseline.json"), &candidate_batch, &raw);

    // Reuse exact incoming modules/parameter IDs/Adam records/shuffle cursor.
    // Keep critic targets unchanged; only actor advantages differ.
    let candidate = fit_from(
        &incoming,
        &original,
        &candidate_batch,
        &out.join("oracle-optimizer"),
    );
    let sham = fit_from(&incoming, &original, &batch, &out.join("sham-optimizer"));
    assert_eq!(fingerprint(&sham), fingerprint(&original), "sham must reproduce original");

    // Critic inputs/targets/minibatch order are unchanged, so every critic
    // transaction must remain byte-identical.
    for step in 1..=16 {
        let file = format!("critic-{step}.bin");
        assert_eq!(
            fs::read(out.join("oracle-optimizer").join(&file)).unwrap(),
            fs::read(original_dir.join("optimizer").join(&file)).unwrap(),
            "critic differs at optimizer step {step}"
        );
    }
    assert_eq!(
        candidate.update_rng.clone().gen::<[u64; 4]>(),
        original.update_rng.clone().gen::<[u64; 4]>(),
        "candidate consumed a different optimizer shuffle stream"
    );
    assert_eq!(fingerprint(&incoming), incoming_fp, "diagnostic mutated incoming session");

    let candidate_actor = candidate.snapshot();
    save(&out.join("actor-incoming.bin"), &incoming_actor);
    save(&out.join("actor-original.bin"), &original.snapshot());
    save(&out.join("actor-oracle-baseline.bin"), &candidate_actor);

    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    let mut evaluation_steps = 0usize;
    for (name, cp, actor) in [
        ("incoming", TARGET - 1, &incoming_actor),
        ("original", TARGET, &original.snapshot()),
        ("oracle-baseline", TARGET, &candidate_actor),
    ] {
        evaluation_steps += evaluate_policy(
            actor,
            SEED,
            EVAL_DOMAIN,
            name,
            cp,
            512,
            Some(&out),
            None,
            &mut eval,
        )
        .steps;
    }
    eval.flush().unwrap();

    fs::write(
        out.join("complete.json"),
        format!(
            concat!(
                "{{\"seed\":{SEED},\"target_update\":{TARGET},\"training_updates\":0,",
                "\"fitting_rows\":1024,\"oracle_values\":1024,",
                "\"actor_advantages_only_changed\":true,\"critic_targets_unchanged\":true,",
                "\"critic_transactions_identical\":true,\"sham_exact\":true,",
                "\"evaluation_records\":4608,\"evaluation_steps\":{evaluation_steps},",
                "\"candidate_training_continued\":false}}\n"
            )
        ),
    )
    .unwrap();
    println!("CRITIC BASELINE CAUSALITY COMPLETE {SEED} {TARGET}");
}

#[test]
fn oracle_value_parser_rejects_wrong_count() {
    let path = std::env::temp_dir().join("rustrobotics-critic-baseline-short.txt");
    fs::write(&path, "1\n2\n").unwrap();
    assert!(std::panic::catch_unwind(|| read_oracle_values(&path)).is_err());
    let _ = fs::remove_file(path);
}
