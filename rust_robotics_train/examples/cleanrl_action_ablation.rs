//! Matched action-distribution PPO diagnostic. Manual development seeds only.
//! Same actor/critic network shape, PPO optimizer, GAE and fixed training
//! budget; only action distribution, density, and entropy semantics differ.
//!
//! cargo run --release -p rust_robotics_train --features cleanrl-reference \
//!   --example cleanrl_action_ablation -- squashed_gaussian 201 /tmp/ppo-gaussian-201.csv
//!
//! The output is create-only. Runs all checkpoints without best-checkpoint
//! selection and retains every episode, including failures and late losses.
//! Always evaluates both development and common held-out panels. An optional
//! fourth argument creates per-tick traces for four already-exposed difficult
//! witnesses at checkpoint 976. This is diagnostic ONLY, not confirmation.
//! Read-only evaluation never feeds trajectories to the optimizer.
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
            ReferenceActuation, ReferenceEvaluationMode,
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
    // All keys were selected only AFTER examining exposed development results.
    // First three are Gaussian failures and categorical successes; last is
    // a hard reset that both arms frequently fail.
    const TRACE_KEYS: [u64; 4] = [4_100_100_015, 4_100_100_021, 4_100_100_170, 4_100_100_100];

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
            args.len() == 4 || args.len() == 5,
            "usage: cleanrl_action_ablation categorical|squashed_gaussian SEED CREATE_ONLY_CSV_PATH [TRACE_CSV_PATH]"
        );
        let distribution_name = args[1].as_str();
        let action_distribution = match distribution_name {
            "categorical" => ReferenceActionDistribution::Categorical,
            "squashed_gaussian" => ReferenceActionDistribution::SquashedGaussian,
            _ => panic!("unknown action distribution: {distribution_name}"),
        };
        let seed: u64 = args[2].parse().expect("seed must be u64");
        assert!(
            (201..=204).contains(&seed),
            "fixed development seeds 201..204 only"
        );
        let path = &args[3];
        let out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("result path already exists or cannot be created");
        let mut out = BufWriter::new(out);
        writeln!(
            out,
            "distribution,seed,updates,train_steps,panel,mode,key,initial_x,initial_xdot,initial_theta,initial_omega,\
             initial_value,steps,completed,terminated,raw_return,min_cart_clearance_m,force_switches"
        )
        .unwrap();
        let mut trace_out = if args.len() == 5 {
            let trace_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args[4])
                .expect("trace output must not already exist");
            let mut writer = BufWriter::new(trace_file);
            writeln!(
                writer,
                "distribution,seed,mode,key,tick,obs_x,obs_xdot,obs_theta,obs_omega,                 before_x,before_xdot,before_theta,before_omega,desired_force_n,                 applied_force_n,after_x,after_xdot,after_theta,after_omega,                 rail_breached,angle_breached,terminated,truncated"
            )
            .unwrap();
            Some(writer)
        } else {
            None
        };

        let config = CleanRlReferenceConfig {
            seed,
            action_distribution,
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
                    if checkpoint == 976 && *panel == "heldout_difficult" {
                        if let Some(writer) = trace_out.as_mut() {
                            for witness in TRACE_KEYS {
                                let (audited, frames) = trainer
                                    .evaluate_seeded_with_actuator_trace(
                                        keys,
                                        EVAL_TICKS,
                                        mode,
                                        ReferenceActuation::Ideal,
                                        Some(witness),
                                    );
                                assert_eq!(
                                    audited, rows,
                                    "trace instrumentation changed the original evaluator"
                                );
                                let expected_steps = rows
                                    .iter()
                                    .find(|r| r.key == witness)
                                    .expect("trace witness missing from common difficult panel")
                                    .steps;
                                assert_eq!(frames.len(), expected_steps);
                                for frame in frames {
                                    let fields = [
                                        distribution_name.to_string(),
                                        seed.to_string(),
                                        mode_name.to_string(),
                                        witness.to_string(),
                                        frame.tick.to_string(),
                                        frame.observation_before[0].to_string(),
                                        frame.observation_before[1].to_string(),
                                        frame.observation_before[2].to_string(),
                                        frame.observation_before[3].to_string(),
                                        frame.true_state_before[0].to_string(),
                                        frame.true_state_before[1].to_string(),
                                        frame.true_state_before[2].to_string(),
                                        frame.true_state_before[3].to_string(),
                                        frame.desired_force_n.to_string(),
                                        frame.applied_force_n.to_string(),
                                        frame.true_state_after[0].to_string(),
                                        frame.true_state_after[1].to_string(),
                                        frame.true_state_after[2].to_string(),
                                        frame.true_state_after[3].to_string(),
                                        frame.rail_breached.to_string(),
                                        frame.angle_breached.to_string(),
                                        frame.terminated.to_string(),
                                        frame.truncated.to_string(),
                                    ];
                                    writeln!(writer, "{}", fields.join(",")).unwrap();
                                }
                            }
                        }
                    }
                    for row in rows {
                        let s = row.initial_state;
                        writeln!(
                            out,
                            "{distribution_name},{seed},{checkpoint},{},{panel},{mode_name},{},{},{},{},{},{},{},{},{},{},{},{}",
                            checkpoint * 512, row.key,
                            s[0], s[1], s[2], s[3], row.initial_value, row.steps,
                            row.completed, row.terminated, row.raw_return,
                            row.min_cart_clearance_m, row.force_switches
                        ).unwrap();
                    }
                    println!(
                        "distribution={distribution_name} seed={seed} updates={checkpoint} panel={panel} mode={mode_name} completed={success}/{PANEL_SIZE}"
                    );
                }
            }
            out.flush().expect("persist all checkpoint records");
            if let Some(writer) = trace_out.as_mut() {
                writer.flush().expect("persist trace records");
            }
        }
        assert_eq!(done_updates, 976);
    }
}

#[cfg(feature = "cleanrl-reference")]
fn main() {
    cohort::run();
}
