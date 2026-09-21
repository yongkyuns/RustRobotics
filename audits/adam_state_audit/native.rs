//! Exact live replay of seed41006 through update4140 with gradient/Adam capture.
//! This is a correctness audit only; the learner recipe is unchanged.
use super::*;

const TARGET: usize = 4140;
const SEED: u64 = 41006;

#[test]
#[ignore = "explicit historical optimizer-state replay, not controller qualification"]
fn emit_adam_state_audit() {
    let out = PathBuf::from(std::env::var("ADAM_AUDIT_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("ADAM_AUDIT_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();

    let mut s = PpoTrainerSession::new_seeded(learning_config("gamma995"), SEED);
    for u in 1..TARGET {
        update_support(&mut s, SEED, u, 1024, None);
        if u % 512 == 0 {
            println!("ADAM AUDIT PREFIX {SEED} {u}");
        }
    }
    assert_eq!(s.metrics.total_updates, TARGET - 1);
    compare_weights(&s, &out, &prior, TARGET - 1);

    crate::trainer::adam_state_audit_hooks::start(out.join("hooks"));
    let detailed = out.join(format!("update-{TARGET}"));
    fs::create_dir_all(&detailed).unwrap();
    update_support(&mut s, SEED, TARGET, 1024, Some(&detailed));
    crate::trainer::adam_state_audit_hooks::stop();

    assert_eq!(s.metrics.total_updates, TARGET);
    compare_weights(&s, &out, &prior, TARGET);

    fs::write(
        out.join("complete.json"),
        format!(
            "{{\"seed\":{SEED},\"target_update\":{TARGET},\"historical_before_exact\":true,\"historical_after_exact\":true,\"actor_steps\":16,\"critic_steps\":16,\"candidate_training\":false}}\n"
        ),
    )
    .unwrap();
    println!("ADAM STATE AUDIT COMPLETE {SEED} {TARGET}");
}
