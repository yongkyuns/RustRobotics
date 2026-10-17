//! Frozen 3-by-3 absolute test-gain crossover of trained Gaussian policies.
//! This uses the SAME fixed recipe in every arm. Development seeds only.
//!
//! cargo run --release -p rust_robotics_train --features cleanrl-reference \
//!   --example cleanrl_gain_crossover -- gain_1p5 201 /tmp/crossover-201.csv
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
            CleanRlReferenceConfig, CleanRlReferenceTrainer, ReferenceActionDistribution,
            ReferenceEvaluationMode,
        },
        PendulumEnv,
    };
    use std::{
        env,
        fs::OpenOptions,
        io::{BufWriter, Write},
    };

    // Each whole-cohort checkpoint is fixed BEFORE any new policy outcome.
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
        assert_eq!(
            args.len(),
            4,
            "usage: cleanrl_gain_crossover gain_1p0|gain_1p5|gain_2p0 SEED CREATE_ONLY_CSV_PATH"
        );
        let train_gain_label = args[1].as_str();
        let train_gain = match train_gain_label {
            "gain_1p0" => 1.0_f32,
            "gain_1p5" => 1.5_f32,
            "gain_2p0" => 2.0_f32,
            _ => panic!("unknown frozen PPO training gain: {train_gain_label}"),
        };
        let seed: u64 = args[2].parse().expect("seed must be u64");
        assert!(
            (201..=204).contains(&seed),
            "exposed development training seeds 201..204 only"
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])
            .expect("output CSV must not already exist");
        let mut out = BufWriter::new(file);
        writeln!(
            out,
            "train_gain,eval_gain,seed,updates,train_steps,panel,mode,key,initial_x,initial_xdot,initial_theta,initial_omega,initial_value,steps,completed,terminated,raw_return,min_cart_clearance_m,force_switches"
        )
        .unwrap();

        let config = CleanRlReferenceConfig {
            seed,
            action_distribution: ReferenceActionDistribution::SquashedGaussian,
            continuous_force_gain: train_gain,
            ..CleanRlReferenceConfig::default()
        };
        let (ordinary, difficult) = fixed_panels(&config, seed);
        let (heldout_ordinary, heldout_difficult) = shared_heldout_panels(&config);
        let panels = [
            ("ordinary", ordinary),
            ("difficult", difficult),
            ("heldout_ordinary", heldout_ordinary),
            ("heldout_difficult", heldout_difficult),
        ];
        let mut trainer = CleanRlReferenceTrainer::new(config);
        trainer.train_updates(976);
        assert_eq!(trainer.metrics().updates, 976);
        assert_eq!(trainer.metrics().environment_steps, 499_712);

        // These are ABSOLUTE physical-force gain maps, not multipliers on
        // the already trained force gain. Training weights are fixed.
        for (eval_gain_label, eval_gain) in [
            ("gain_1p0", 1.0_f32),
            ("gain_1p5", 1.5_f32),
            ("gain_2p0", 2.0_f32),
        ] {
            for (panel, keys) in &panels {
                for (mode_name, mode) in [
                    ("deterministic", ReferenceEvaluationMode::Deterministic),
                    ("stochastic", ReferenceEvaluationMode::Stochastic),
                ] {
                    let rows = trainer.evaluate_seeded_at_gain(
                        keys, EVAL_TICKS, mode, eval_gain,
                    );
                    assert_eq!(rows.len(), PANEL_SIZE);
                    let success = rows.iter().filter(|row| row.completed).count();
                    for row in rows {
                        let s = row.initial_state;
                        let fields = [
                            train_gain_label.to_string(),
                            eval_gain_label.to_string(),
                            seed.to_string(),
                            "976".to_string(),
                            "499712".to_string(),
                            (*panel).to_string(),
                            mode_name.to_string(),
                            row.key.to_string(),
                            s[0].to_string(),
                            s[1].to_string(),
                            s[2].to_string(),
                            s[3].to_string(),
                            row.initial_value.to_string(),
                            row.steps.to_string(),
                            row.completed.to_string(),
                            row.terminated.to_string(),
                            row.raw_return.to_string(),
                            row.min_cart_clearance_m.to_string(),
                            row.force_switches.to_string(),
                        ];
                        writeln!(out, "{}", fields.join(",")).unwrap();
                    }
                    println!(
                        "train_gain={train_gain_label} eval_gain={eval_gain_label} seed={seed} panel={panel} mode={mode_name} completed={success}/{PANEL_SIZE}"
                    );
                }
            }
            out.flush().expect("retain every evaluation gain matrix slice");
        }
    }
}

#[cfg(feature = "cleanrl-reference")]
fn main() {
    cohort::run();
}
