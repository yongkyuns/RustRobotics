//! One retrospectively selected historical case; no training or policy changes.
use super::*;

fn historical_row<'a>(records: &'a [Vec<&'a str>], mode: &str) -> &'a Vec<&'a str> {
    let matching: Vec<_> = records.iter().filter(|r|
        r[0] == "41003" && r[1] == mode && r[2] == "long-stoch" && r[3] == "45"
    ).collect();
    assert_eq!(matching.len(), 1, "ambiguous historical outcome");
    matching[0]
}
fn require_same(row: &[&str], result: &Outcome) {
    // The inherited verifier expects the earlier checkpointed CSV layout.
    let adapted = [row[0], row[1], "512", row[2], row[3], row[4], row[5], row[6],
        row[7], row[8], row[9], row[10], row[11], row[12], row[13]];
    verify(&adapted, result);
    for i in 0..4 { assert_eq!(row[14+i].parse::<f32>().unwrap(), result.initial[i]); }
    assert_eq!(row[18].parse::<f32>().unwrap(), result.max_velocity);
}
#[test]
#[ignore = "explicit selected-case reward audit, not a new reliability trial"]
fn emit_reward_alignment() {
    let input = PathBuf::from(std::env::var("ALIGN_INPUT").unwrap());
    let out = PathBuf::from(std::env::var("ALIGN_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    let weights = fs::read(input.join("actor.bin")).unwrap();
    let policy = decode(&weights);
    let history = fs::read_to_string(input.join("evaluation.csv")).unwrap();
    let records: Vec<Vec<&str>> = history.lines().skip(1).map(|line|line.split(',').collect()).collect();
    assert_eq!(records.len(), 576);
    assert!(records.iter().all(|r|r.len()==19));
    let mut output = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    writeln!(output, "{}", history.lines().next().unwrap()).unwrap();
    let mut steps = 0;
    for (mode, name) in MODES.into_iter().enumerate() {
        let row = historical_row(&records, name);
        let mut trace = BufWriter::new(fs::File::create(out.join(format!("trace-long-stoch-{name}.csv"))).unwrap());
        let result = episode(&policy, mode, 41003, "long-stoch", 45, Some(&mut trace));
        trace.flush().unwrap();
        require_same(row, &result);
        // Copy only AFTER every field was verified against the new replay.
        writeln!(output, "{}", row.join(",")).unwrap();
        steps += result.steps;
        println!("REWARD ALIGNMENT CASE {name}: {} steps, {}", result.steps, result.ending);
    }
    output.flush().unwrap();
    assert_eq!(fs::read(input.join("actor.bin")).unwrap(), weights);
    fs::write(out.join("actor.bin"), weights).unwrap();
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":41003,\"panel\":\"long-stoch\",\"rep\":45,",
        "\"episodes\":3,\"historical_exact\":true,\"training_steps\":0,",
        "\"replay_steps\":{},\"selected_retrospectively\":true}}\n"),steps)).unwrap();
    println!("REWARD ALIGNMENT REPLAY COMPLETE");
}
#[test]
fn reference_matching_rejects_duplicates_and_preserves_layout() {
    let example = "41003,original,long-stoch,45,1,30000,120,position,4,3,2.41,0.2,false,1,0,0,0,0,2";
    let row: Vec<_> = example.split(',').collect();
    let records = vec![row.clone()];
    assert_eq!(historical_row(&records, "original"), &row);
    let duplicate = vec![row.clone(), row];
    assert!(std::panic::catch_unwind(||historical_row(&duplicate, "original")).is_err());
}
