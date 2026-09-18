//! Permanent training/deployment contract tests, not learning-quality thresholds.
use super::domain::{InvertedPendulum, NoiseConfig, PENDULUM_FIXED_DT};
use crate::simulator::Simulate;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_robotics_algo::{cart_pole::CartPoleParameters, Vector4};
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig, PpoTrainerConfig, PpoTrainerSession};

#[test]
fn frozen_policy_live_and_training_paths_match_noise_resets_and_custom_physics() {
    for (index, plant) in [
        CartPoleParameters::default(),
        CartPoleParameters {
            length_m: 0.6,
            cart_mass_kg: 0.4,
            pole_mass_kg: 0.2,
        },
        CartPoleParameters {
            length_m: 3.1,
            cart_mass_kg: 1.7,
            pole_mass_kg: 2.3,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let config = PpoTrainerConfig {
            plant,
            env: PendulumEnvConfig {
                max_steps: 9,
                ..Default::default()
            },
            ..Default::default()
        };
        let policy = PpoTrainerSession::new_seeded(config.clone(), 901 + index as u64).snapshot();
        // Deliberately near a true-terminal boundary, followed by fresh resets.
        let state = Vector4::new(0.2, -0.1, 0.599, 1.0);
        let mut env = PendulumEnv::from_state(plant.model(), config.env, state, 0);
        let mut sim = InvertedPendulum {
            state,
            model: plant.model(),
            trainer_config: config.clone(),
            ..Default::default()
        };
        sim.set_policy_controller(&policy);
        sim.state = state;
        sim.policy_observation = None;
        let mut a = StdRng::seed_from_u64(711 + index as u64);
        let mut b = a.clone();
        let mut observation = env.observation_with_rng(&mut a);
        let mut steps = 0;
        let mut timeouts = 0;
        let mut terminals = 0;
        for _ in 0..300 {
            let result = env.step_with_rng(policy.act(observation), &mut a);
            sim.step_with_rng(
                PENDULUM_FIXED_DT,
                NoiseConfig {
                    enabled: true,
                    scale: 1.0,
                },
                &mut b,
            );
            steps += 1;
            if result.done {
                timeouts += usize::from(result.truncated);
                terminals += usize::from(result.terminated());
                observation = env.reset_with_rng(&mut a);
                steps = 0;
            } else {
                observation = result.observation;
            }
            assert_eq!(sim.state, env.state());
            assert_eq!(sim.policy_observation, Some(observation));
            assert_eq!(sim.visual_episode_steps, steps);
            assert_eq!(a.clone().gen::<u64>(), b.clone().gen::<u64>());
        }
        assert!(
            terminals > 0 && timeouts > 0,
            "both episode boundaries must be exercised"
        );
    }
}

#[test]
fn changing_physics_stops_training_and_blocks_stale_policy_until_restart() {
    let mut sim = InvertedPendulum::default();
    sim.start_training();
    assert!(sim.trainer_backend.is_initialized());
    sim.model.l_bar = 0.7;
    sim.tick_training();
    assert!(!sim.training_active && !sim.trainer_backend.is_initialized());
    assert!(sim.policy_environment_stale);
    let state = sim.state;
    sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut StdRng::seed_from_u64(9));
    assert_eq!(
        state, sim.state,
        "a stale snapshot must not drive a changed plant"
    );
    sim.start_training();
    assert!(sim.training_active && !sim.policy_environment_stale);
    assert_eq!(sim.trainer_config.plant.length_m, 0.7);
    assert_eq!(sim.active_training_environment.unwrap().0.length_m, 0.7);
}

#[test]
fn noise_controls_apply_to_both_paths_and_invalidate_existing_training() {
    let mut sim = InvertedPendulum::default();
    sim.start_training();
    sim.configure_training_noise(NoiseConfig {
        enabled: false,
        scale: 1.0,
    });
    assert!(sim.policy_environment_stale && !sim.training_active);
    let cfg = sim.trainer_config.env;
    assert_eq!(cfg.action_noise_force_n, 0.0);
    assert_eq!(cfg.observation_angle_noise_rad, 0.0);
    assert_eq!(cfg.disturbance_probability_per_step, 0.0);
    sim.start_training();
    assert_eq!(sim.active_training_environment.unwrap().1, cfg);
    sim.configure_training_noise(NoiseConfig {
        enabled: true,
        scale: 2.0,
    });
    assert!(sim.policy_environment_stale);
    assert_eq!(
        sim.trainer_config.env.action_noise_force_n,
        2.0 * PendulumEnvConfig::default().action_noise_force_n
    );
}

#[test]
fn stopped_policy_still_uses_the_training_time_limit_and_reset_distribution() {
    let mut sim = InvertedPendulum::default();
    sim.trainer_config.env.max_steps = 1;
    sim.start_training();
    sim.stop_training();
    let mut rng = StdRng::seed_from_u64(123);
    sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut rng);
    assert_eq!(sim.visual_episode_steps, 0);
    assert!(sim.state[0].abs() <= sim.trainer_config.env.reset_position_range_m);
    assert!(sim.state[1].abs() <= sim.trainer_config.env.reset_velocity_range_mps);
    assert!(sim.state[2].abs() <= sim.trainer_config.env.reset_angle_range_rad);
    assert!(sim.state[3].abs() <= sim.trainer_config.env.reset_angular_velocity_range_radps);
}

#[test]
fn old_json_config_defaults_physics_and_custom_physics_roundtrips() {
    let mut json = serde_json::to_value(PpoTrainerConfig::default()).unwrap();
    json.as_object_mut().unwrap().remove("plant");
    let old: PpoTrainerConfig = serde_json::from_value(json).unwrap();
    assert_eq!(old.plant, CartPoleParameters::default());
    let custom = PpoTrainerConfig {
        plant: CartPoleParameters {
            length_m: 0.9,
            cart_mass_kg: 1.6,
            pole_mass_kg: 0.4,
        },
        ..Default::default()
    };
    let decoded: PpoTrainerConfig =
        serde_json::from_str(&serde_json::to_string(&custom).unwrap()).unwrap();
    assert_eq!(custom.plant, decoded.plant);
}

#[test]
fn copied_visible_state_discards_previous_noisy_policy_observation() {
    let mut a = InvertedPendulum::default();
    let b = InvertedPendulum::default();
    a.policy_observation = Some([100.0; 4]);
    a.match_state_with(&b);
    assert!(a.policy_observation.is_none());
    assert_eq!(a.state, b.state);
}

#[test]
fn activating_policy_resets_but_publishing_new_weights_does_not() {
    let mut sim = InvertedPendulum {
        state: Vector4::new(3.0, 5.0, 0.8, 6.0),
        visual_episode_steps: 17,
        ..Default::default()
    };
    let policy = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 17).snapshot();
    sim.set_policy_controller(&policy);
    assert_eq!(
        sim.controller_selection,
        super::domain::ControllerKind::Policy
    );
    assert_eq!(sim.visual_episode_steps, 0);
    assert!(sim.state[0].abs() <= sim.trainer_config.env.reset_position_range_m);
    assert!(sim.state[2].abs() <= sim.trainer_config.env.reset_angle_range_rad);
    let state = sim.state;
    let observation = sim.policy_observation;
    sim.set_policy_controller(&policy);
    assert_eq!(sim.state, state);
    assert_eq!(sim.policy_observation, observation);
}

#[test]
fn pending_replacement_snapshot_does_not_run_old_policy() {
    let mut sim = InvertedPendulum::default();
    sim.start_training();
    sim.trainer_backend.destroy();
    // The environment contract remains active but there is no published policy:
    // this models the browser worker creation interval without mocking physics.
    let state = sim.state;
    sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut StdRng::seed_from_u64(33));
    assert_eq!(sim.state, state);
    assert!(sim.last_control_error().unwrap().contains("Waiting"));
}
