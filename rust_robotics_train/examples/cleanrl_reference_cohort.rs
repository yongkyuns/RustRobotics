//! Full fixed-budget native PPO diagnostic. Manual opt-in only.
//!
//! cargo run --release -p rust_robotics_train --features cleanrl-reference \
//!   --example cleanrl_reference_cohort -- 201 /tmp/cleanrl-native-201.csv
//!
//! The output is create-only. Runs all checkpoints without best-checkpoint
//! selection and retains every episode, including failures and late losses.
//! Pass --shared-heldout to append common reset-key panels (same across seeds
//! 201..204) to the original per-seed development panels. Evaluation is
//! read-only and never feeds observations or rewards to the optimizer.
//! Results are evidence, NOT automatic production acceptance on issue #33.
#[cfg(not(feature = "cleanrl-reference"))]
fn main() {
    eprintln!("Enable --features cleanrl-reference to run this diagnostic");
    std::process::exit(2);
}

#[cfg(feature = "cleanrl-reference")]
mod cohort {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rust_robotics_train::{
        cleanrl_reference::{
            CleanRlReferenceConfig, CleanRlReferenceTrainer, ReferenceEvaluationMode,
        },
        PendulumEnv,
    };
    use std::{
        env,
        fs::OpenOptions,
        io::{BufWriter, Write},
    };

    // Each whole-cohort checkpoint is fixed BEFORE any new policy outcome.
    const UPDATES: [usize; 7] = [0, 64, 128, 256, 512, 768, 976];
    const PANEL_SIZE: usize = 64;
    const EVAL_TICKS: usize = 1500;

    fn initial_state(config: &CleanRlReferenceConfig, key: u64) -> [f32; 4] {
        let mut root = StdRng::seed_from_u64(key);
        let mut environment_rng = StdRng::seed_from_u64(root.gen());
        let mut eval_config = config.environment;
        eval_config.max_steps = EVAL_TICKS;
        let env =
            PendulumEnv::new_with_rng(config.plant.model(), eval_config, &mut environment_rng);
        let s = env.state();
        [s[0], s[1], s[2], s[3]]
    }

    fn fixed_panels(config: &CleanRlReferenceConfig, seed: u64) -> (Vec<u64>, Vec<u64>) {
        let ordinary_start = 2_600_100_000_u64 + seed * 100_000;
        let difficult_start = 3_300_100_000_u64 + seed * 100_000;
        let ordinary = (ordinary_start..ordinary_start + PANEL_SIZE as u64).collect::<Vec<_>>();
        let mut difficult = Vec::with_capacity(PANEL_SIZE);
        for key in difficult_start..difficult_start + 20_000 {
            let s = initial_state(config, key);
            // Predeclared physics-only difficult reset predicate; do not use
            // reward, policy, actual trajectory or future outcome for selection.
            if s[2].abs() >= 0.15 && s[3].abs() >= 0.25 && s[2] * s[3] > 0.0 {
                difficult.push(key);
                if difficult.len() == PANEL_SIZE {
                    break;
                }
            }
        }
        assert_eq!(
            difficult.len(),
            PANEL_SIZE,
            "insufficient difficult reset keys"
        );
        (ordinary, difficult)
    }

    fn shared_heldout_panels(config: &CleanRlReferenceConfig) -> (Vec<u64>, Vec<u64>) {
        // Predeclared, common across training seeds and never fed to the optimizer.
        // Disjoint key namespaces from the seed-specific development panels.
        let ordinary_start = 4_000_100_000_u64;
        let difficult_start = 4_100_100_000_u64;
        let ordinary = (ordinary_start..ordinary_start + PANEL_SIZE as u64).collect::<Vec<_>>();
        let mut difficult = Vec::with_capacity(PANEL_SIZE);
        for key in difficult_start..difficult_start + 20_000 {
            let s = initial_state(config, key);
            // Same physics-only outward-motion predicate as development.
            if s[2].abs() >= 0.15 && s[3].abs() >= 0.25 && s[2] * s[3] > 0.0 {
                difficult.push(key);
                if difficult.len() == PANEL_SIZE {
                    break;
                }
            }
        }
        assert_eq!(
            difficult.len(),
            PANEL_SIZE,
            "insufficient common held-out reset keys"
        );
        (ordinary, difficult)
    }

    pub fn run() {
        let args = env::args().collect::<Vec<_>>();
        assert!(
            args.len() == 3 || (args.len() == 4 && args[3] == "--shared-heldout"),
            "usage: cleanrl_reference_cohort SEED CREATE_ONLY_CSV_PATH [--shared-heldout]"
        );
        let seed: u64 = args[1].parse().expect("seed must be u64");
        assert!(
            (201..=204).contains(&seed),
            "fixed development seeds 201..204 only"
        );
        let path = &args[2];
        let out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("result path already exists or cannot be created");
        let mut out = BufWriter::new(out);
        writeln!(
            out,
            "seed,updates,train_steps,panel,mode,key,initial_x,initial_xdot,initial_theta,initial_omega,\
             initial_value,steps,completed,terminated,raw_return,min_cart_clearance_m,force_switches"
        ).unwrap();

        let config = CleanRlReferenceConfig {
            seed,
            ..CleanRlReferenceConfig::default()
        };
        let (ordinary, difficult) = fixed_panels(&config, seed);
        let mut panels = vec![("ordinary", ordinary), ("difficult", difficult)];
        if args.len() == 4 {
            let (heldout_ordinary, heldout_difficult) = shared_heldout_panels(&config);
            panels.push(("heldout_ordinary", heldout_ordinary));
            panels.push(("heldout_difficult", heldout_difficult));
        }
        let mut trainer = CleanRlReferenceTrainer::new(config);
        let mut done_updates = 0;
        for &checkpoint in &UPDATES {
            // Critically, Adam state and the linear learning-rate schedule are
            // NEVER restarted when moving between checkpoints.
            trainer.train_updates(checkpoint - done_updates);
            done_updates = checkpoint;
            assert_eq!(trainer.metrics().updates, checkpoint);
            assert_eq!(trainer.metrics().environment_steps, checkpoint * 512);
            for (panel, keys) in &panels {
                for (mode_name, mode) in [
                    ("deterministic", ReferenceEvaluationMode::Deterministic),
                    ("stochastic", ReferenceEvaluationMode::Stochastic),
                ] {
                    let rows = trainer.evaluate_seeded(keys, EVAL_TICKS, mode);
                    assert_eq!(rows.len(), PANEL_SIZE);
                    let success = rows.iter().filter(|x| x.completed).count();
                    for row in rows {
                        let s = row.initial_state;
                        writeln!(
                            out,
                            "{seed},{checkpoint},{},{panel},{mode_name},{},{},{},{},{},{},{},{},{},{},{},{}",
                            checkpoint * 512, row.key,
                            s[0], s[1], s[2], s[3], row.initial_value, row.steps,
                            row.completed, row.terminated, row.raw_return,
                            row.min_cart_clearance_m, row.force_switches
                        ).unwrap();
                    }
                    println!(
                        "seed={seed} updates={checkpoint} panel={panel} mode={mode_name} completed={success}/{PANEL_SIZE}"
                    );
                }
            }
            out.flush().expect("persist all checkpoint records");
        }
        assert_eq!(done_updates, 976);
    }
}

#[cfg(feature = "cleanrl-reference")]
fn main() {
    cohort::run();
}
