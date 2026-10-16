//! Frozen-Gaussian evaluation-time corrective-force intervention, issue #42.
//! Four EXPOSED development policies are trained with unchanged PPO, then
//! tested across all common reset keys under predetermined force maps.
//! NOT fresh confirmation and NOT a production trainer/default change.
#![allow(clippy::result_large_err)]

#[cfg(not(feature = "cleanrl-reference"))]
fn main() {
    eprintln!("This diagnostic requires --features cleanrl-reference");
    std::process::exit(2);
}

#[cfg(feature = "cleanrl-reference")]
mod audit {
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
        fs::{create_dir_all, OpenOptions},
        io::{BufWriter, Write},
        path::Path,
    };

    const WITNESS_KEY: u64 = 4_100_100_015;
    const EVAL_TICKS: usize = 1500;
    const PANEL_SIZE: usize = 64;

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

    fn panels(config: &CleanRlReferenceConfig) -> (Vec<u64>, Vec<u64>) {
        let ordinary = (4_000_100_000..4_000_100_000 + PANEL_SIZE as u64).collect();
        let mut difficult = Vec::with_capacity(PANEL_SIZE);
        for key in 4_100_100_000..4_100_100_000 + 20_000 {
            let s = initial_state(config, key);
            if s[2].abs() >= 0.15 && s[3].abs() >= 0.25 && s[2] * s[3] > 0.0 {
                difficult.push(key);
                if difficult.len() == PANEL_SIZE {
                    break;
                }
            }
        }
        assert_eq!(difficult.len(), PANEL_SIZE);
        assert!(
            difficult.contains(&WITNESS_KEY),
            "the frozen failure key must stay in the panel"
        );
        (ordinary, difficult)
    }

    fn csv_line(file: &mut impl Write, fields: Vec<String>) {
        writeln!(file, "{}", fields.join(",")).expect("write diagnostic row");
    }

    fn create_csv(path: &Path) -> BufWriter<std::fs::File> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create-only diagnostic file must not already exist");
        BufWriter::new(file)
    }

    pub fn run() {
        let args = env::args().collect::<Vec<_>>();
        assert_eq!(
            args.len(),
            2,
            "usage: cleanrl_gaussian_force_gain CREATE_ONLY_OUTPUT_DIRECTORY"
        );
        let folder = Path::new(&args[1]);
        create_dir_all(folder).expect("create diagnostic output directory");
        let mut episodes = create_csv(&folder.join("episodes.csv"));
        let mut traces = create_csv(&folder.join("witness-traces.csv"));
        let mut protocol = create_csv(&folder.join("provenance.txt"));
        writeln!(
            protocol,
            "seeds=201,202,203,204\nupdates_per_seed=976\ntraining_steps_per_seed=499712\n\
             evaluation_horizon=1500\nphysics_dt_seconds=0.01\n\
             common_witness_key=4100100015\nactuator_modes=ideal,boost_1p5,boost_2p0,bang_bang_sign\n             distribution=squashed_gaussian\ncohort=exposed_development_only"
        ).unwrap();
        writeln!(
            episodes,
            "seed,updates,actuator,mode,panel,key,steps,completed,terminated,raw_return,\
             min_cart_clearance_m,policy_force_switches,initial_x,initial_xdot,initial_theta,initial_omega"
        ).unwrap();
        writeln!(
            traces,
            "seed,actuator,mode,key,tick,obs_x,obs_xdot,obs_theta,obs_omega,\
             before_x,before_xdot,before_theta,before_omega,desired_force_n,\
             actuator_force_n,applied_force_n,reward,after_x,after_xdot,\
             after_theta,after_omega,rail_breached,angle_breached,terminated,truncated"
        )
        .unwrap();

        let actuators = [
            ReferenceActuation::Ideal,
            ReferenceActuation::BoostOneAndHalf,
            ReferenceActuation::BoostTwo,
            ReferenceActuation::BangBangSign,
        ];
        let modes = [
            ("deterministic", ReferenceEvaluationMode::Deterministic),
            ("stochastic", ReferenceEvaluationMode::Stochastic),
        ];
        for seed in 201..=204_u64 {
            let config = CleanRlReferenceConfig {
                seed,
                action_distribution: ReferenceActionDistribution::SquashedGaussian,
                ..CleanRlReferenceConfig::default()
            };
            let (ordinary, difficult) = panels(&config);
            let mut trainer = CleanRlReferenceTrainer::new(config);
            trainer.train_updates(976);
            assert_eq!(trainer.metrics().updates, 976);
            assert_eq!(trainer.metrics().environment_steps, 499_712);
            writeln!(protocol, "seed_{seed}_complete=yes").unwrap();
            for actuator in actuators {
                for (mode_name, mode) in modes {
                    for (panel_name, keys) in [("ordinary", &ordinary), ("difficult", &difficult)] {
                        // Keep all 64 lanes for inference-shape equivalence to
                        // original cohort, but trace only the known failure key.
                        let witness = if panel_name == "difficult" {
                            Some(WITNESS_KEY)
                        } else {
                            None
                        };
                        let (rows, frame_traces) = trainer.evaluate_seeded_with_actuator_trace(
                            keys, EVAL_TICKS, mode, actuator, witness,
                        );
                        assert_eq!(rows.len(), PANEL_SIZE);
                        let success = rows.iter().filter(|r| r.completed).count();
                        for row in rows {
                            csv_line(
                                &mut episodes,
                                vec![
                                    seed.to_string(),
                                    "976".to_string(),
                                    actuator.name().to_string(),
                                    mode_name.to_string(),
                                    panel_name.to_string(),
                                    row.key.to_string(),
                                    row.steps.to_string(),
                                    row.completed.to_string(),
                                    row.terminated.to_string(),
                                    row.raw_return.to_string(),
                                    row.min_cart_clearance_m.to_string(),
                                    row.force_switches.to_string(),
                                    row.initial_state[0].to_string(),
                                    row.initial_state[1].to_string(),
                                    row.initial_state[2].to_string(),
                                    row.initial_state[3].to_string(),
                                ],
                            );
                        }
                        for step in frame_traces {
                            csv_line(
                                &mut traces,
                                vec![
                                    seed.to_string(),
                                    actuator.name().to_string(),
                                    mode_name.to_string(),
                                    step.key.to_string(),
                                    step.tick.to_string(),
                                    step.observation_before[0].to_string(),
                                    step.observation_before[1].to_string(),
                                    step.observation_before[2].to_string(),
                                    step.observation_before[3].to_string(),
                                    step.true_state_before[0].to_string(),
                                    step.true_state_before[1].to_string(),
                                    step.true_state_before[2].to_string(),
                                    step.true_state_before[3].to_string(),
                                    step.desired_force_n.to_string(),
                                    step.actuator_force_n.to_string(),
                                    step.applied_force_n.to_string(),
                                    step.reward.to_string(),
                                    step.true_state_after[0].to_string(),
                                    step.true_state_after[1].to_string(),
                                    step.true_state_after[2].to_string(),
                                    step.true_state_after[3].to_string(),
                                    step.rail_breached.to_string(),
                                    step.angle_breached.to_string(),
                                    step.terminated.to_string(),
                                    step.truncated.to_string(),
                                ],
                            );
                        }
                        println!(
                            "seed={seed} actuator={} mode={mode_name} panel={panel_name} completed={success}/{PANEL_SIZE}",
                            actuator.name()
                        );
                    }
                }
            }
            episodes.flush().unwrap();
            traces.flush().unwrap();
            protocol.flush().unwrap();
        }
    }
}

#[cfg(feature = "cleanrl-reference")]
fn main() {
    audit::run();
}
