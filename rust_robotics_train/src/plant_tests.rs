use super::*;
use rust_robotics_algo::{cart_pole::CartPoleParameters, control::StateSpace, Vector4};

#[test]
fn every_pooled_stream_uses_the_requested_physical_parameters() {
    let plant = CartPoleParameters {
        length_m: 0.8,
        cart_mass_kg: 1.9,
        pole_mass_kg: 0.3,
    };
    let cfg = PpoTrainerConfig {
        plant,
        ..Default::default()
    };
    let mut session = PpoTrainerSession::new_seeded_with_environments(cfg, 203, 4);
    assert_eq!(session.env.model(), plant.model());
    for env in &session.additional_environments {
        assert_eq!(env.env.model(), plant.model());
    }
    session.train_updates(2);
    assert_eq!(session.metrics.total_env_steps, 4 * 512 * 2);
    assert_eq!(session.env.model(), plant.model());
    for env in &session.additional_environments {
        assert_eq!(env.env.model(), plant.model());
    }
}

#[test]
fn training_transition_is_shared_nonlinear_rk4_not_linear_euler() {
    let plant = CartPoleParameters {
        length_m: 1.3,
        cart_mass_kg: 0.7,
        pole_mass_kg: 0.4,
    };
    let cfg = PendulumEnvConfig {
        action_noise_force_n: 0.0,
        disturbance_force_n: 0.0,
        disturbance_probability_per_step: 0.0,
        ..Default::default()
    };
    let state = Vector4::from_columns([[0.1, -0.2, 0.5, 0.8]]);
    let mut env = PendulumEnv::from_state(plant.model(), cfg, state, 0);
    env.step_with_rng(1.2, &mut StdRng::seed_from_u64(99));
    assert_eq!(env.state(), plant.step(state, 1.2, cfg.dt));
    let (a, b) = plant.model().model(cfg.dt);
    assert!((env.state() - (a * state + b * 1.2)).norm() > 1e-4);
    assert_eq!(env.last_applied_force(), 1.2);
}

#[test]
fn nondefault_physics_changes_trajectories_without_changing_initial_networks() {
    let default = PpoTrainerConfig::default();
    let other = PpoTrainerConfig {
        plant: CartPoleParameters {
            length_m: 0.5,
            ..Default::default()
        },
        ..default.clone()
    };
    let mut a = PpoTrainerSession::new_seeded(default, 203);
    let mut b = PpoTrainerSession::new_seeded(other, 203);
    assert_eq!(a.snapshot(), b.snapshot());
    assert_eq!(a.current_observation, b.current_observation);
    assert_ne!(
        a.collect_rollout().observations,
        b.collect_rollout().observations
    );
}
