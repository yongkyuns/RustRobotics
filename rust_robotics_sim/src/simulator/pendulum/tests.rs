use super::domain::ControllerKind;
use super::*;
use crate::simulator::Simulate;
use rust_robotics_algo::inverted_pendulum::Model;
use rust_robotics_algo::nalgebra;
use rust_robotics_algo::vector;
use rust_robotics_core::LinearSnapshot;
use rust_robotics_core::PolicySnapshot;
use serde_json;

fn linear_snapshot(in_dim: usize, out_dim: usize, weight: &[f32], bias: &[f32]) -> LinearSnapshot {
    LinearSnapshot {
        in_dim,
        out_dim,
        weight: weight.to_vec(),
        bias: bias.to_vec(),
    }
}

fn policy_snapshot(output_gain: f32) -> PolicySnapshot {
    PolicySnapshot {
        input: linear_snapshot(4, 1, &[1.0, 0.0, 0.0, 0.0], &[0.0]),
        hidden: linear_snapshot(1, 1, &[1.0], &[0.0]),
        output: linear_snapshot(1, 1, &[output_gain], &[0.0]),
        action_limit: 1.0,
        action_std: 0.25,
    }
}

#[test]
fn disabled_noise_is_a_noop() {
    let config = NoiseConfig {
        enabled: false,
        scale: 1.0,
    };
    let initial_state = vector![0.1, -0.2, 0.05, 0.3];
    let mut sim_with_noise_api = InvertedPendulum {
        state: initial_state,
        ..Default::default()
    };
    let mut sim_plain = InvertedPendulum {
        state: initial_state,
        ..Default::default()
    };

    sim_with_noise_api.step_with_noise(0.01, config);
    sim_plain.step(0.01);

    assert_eq!(sim_with_noise_api.state, sim_plain.state);
}

#[test]
fn noise_profile_uses_signal_specific_units() {
    let profile = NoiseConfig {
        enabled: true,
        scale: 1.0,
    }
    .profile();

    assert_eq!(profile.position_m, 0.005);
    assert_eq!(profile.velocity_mps, 0.02);
    assert_eq!(profile.angle_rad, 0.004);
    assert_eq!(profile.angular_velocity_radps, 0.02);
    assert_eq!(profile.force_n, 0.2);
}

#[test]
fn selecting_policy_initializes_policy_controller() {
    let mut sim = InvertedPendulum::default();

    sim.select_controller_kind(ControllerKind::Policy);

    assert_eq!(sim.controller_selection, ControllerKind::Policy);
    assert_eq!(sim.controller.kind(), ControllerKind::Policy);
    assert!(sim.trainer_backend.is_initialized());
}

#[test]
fn policy_snapshot_roundtrip_preserves_action() {
    let snapshot = policy_snapshot(0.75);
    let observation = [1.25, -0.5, 0.1, 0.0];
    let encoded = serde_json::to_string(&snapshot).expect("serialize policy snapshot");
    let decoded: PolicySnapshot =
        serde_json::from_str(&encoded).expect("deserialize policy snapshot");

    assert_eq!(decoded, snapshot);
    assert!((decoded.act(observation) - snapshot.act(observation)).abs() < 1e-6);
}

#[test]
fn policy_controller_uses_snapshot_action() {
    let snapshot = policy_snapshot(0.5);
    let state = vector![2.0, 0.0, 0.0, 0.0];
    let expected = snapshot.act([2.0, 0.0, 0.0, 0.0]);
    let mut controller = Controller::policy(snapshot);

    let action = controller.control(state, PENDULUM_FIXED_DT);

    assert!((action - expected).abs() < 1e-6);
}

#[test]
fn sync_policy_controller_updates_policy_behavior() {
    let mut sim = InvertedPendulum::default();
    let first = policy_snapshot(0.25);
    let second = policy_snapshot(1.5);
    let state = vector![1.0, 0.0, 0.0, 0.0];

    sim.set_policy_controller(&first);
    let first_action = sim.controller.control(state, PENDULUM_FIXED_DT);

    sim.sync_policy_controller(&second);
    let second_action = sim.controller.control(state, PENDULUM_FIXED_DT);

    assert_eq!(sim.controller.kind(), ControllerKind::Policy);
    assert!((first_action - first.act([1.0, 0.0, 0.0, 0.0])).abs() < 1e-6);
    assert!((second_action - second.act([1.0, 0.0, 0.0, 0.0])).abs() < 1e-6);
    assert!(second_action > first_action);
}

#[test]
fn lqr_controller_reduces_known_initial_error_over_fixed_rollout() {
    let mut sim = InvertedPendulum {
        state: vector![0.3, 0.0, 0.2, 0.0],
        controller: Controller::lqr(Model::default()),
        ..Default::default()
    };
    let initial_angle = sim.state[2].abs();

    for _ in 0..400 {
        sim.step(PENDULUM_FIXED_DT);
    }

    assert!(
        sim.state.iter().all(|value| value.is_finite()),
        "final_state={:?}",
        sim.state
    );
    assert!(
        sim.state[2].abs() < initial_angle,
        "final_state={:?}",
        sim.state
    );
    assert!(sim.state[2].abs() < 0.05, "final_state={:?}", sim.state);
    assert!(sim.state[3].abs() < 0.02, "final_state={:?}", sim.state);
}

#[test]
fn nonlinear_plant_falls_farther_from_upright_without_control() {
    let mut sim = InvertedPendulum {
        state: vector![0.0, 0.0, 1.0, 0.0],
        controller: Controller::policy(policy_snapshot(0.0)),
        ..Default::default()
    };

    let initial_angle = sim.state[2];
    sim.step(PENDULUM_FIXED_DT);

    assert!(sim.state[2] > initial_angle, "final_state={:?}", sim.state);
    assert!(sim.state[3] > 0.0, "final_state={:?}", sim.state);
}

#[test]
fn pendulum_runtime_prepares_and_reuses_lqr_configuration() {
    let model = Model::default();
    let mut sim = InvertedPendulum {
        state: vector![0.1, 0.0, 0.05, 0.0],
        controller: Controller::lqr(model),
        ..Default::default()
    };

    assert!(!sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));

    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));
}

#[test]
fn lqr_runtime_reprepares_when_model_changes() {
    let original = Model::default();
    let mut sim = InvertedPendulum {
        controller: Controller::lqr(original),
        ..Default::default()
    };
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(original, PENDULUM_FIXED_DT));

    let mut updated = original;
    updated.Q[10] *= 1.5;
    sim.controller = Controller::lqr(updated);
    sim.step(PENDULUM_FIXED_DT);

    assert!(!sim.lqr_cache_matches(original, PENDULUM_FIXED_DT));
    assert!(sim.lqr_cache_matches(updated, PENDULUM_FIXED_DT));
}

#[test]
fn successful_mpc_runtime_clears_control_error() {
    let mut sim = InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::mpc(Model::default()),
        last_control_error: Some("stale error".to_owned()),
        ..Default::default()
    };

    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.last_control_error().is_none());
}

fn default_mpc_model() -> Model {
    match Controller::mpc(Model::default()) {
        Controller::MPC(model) => model,
        _ => unreachable!("MPC constructor must select MPC"),
    }
}

fn mpc_sim(model: Model) -> InvertedPendulum {
    InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::MPC(model),
        controller_selection: ControllerKind::Mpc,
        ..Default::default()
    }
}

#[test]
fn pendulum_runtime_prepares_and_reuses_mpc_configuration() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);

    assert!(!sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 0);
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 1);
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(
        sim.mpc_preparations, 1,
        "unchanged configuration must be reused"
    );
    assert!(sim.last_control_error().is_none());
}

#[test]
fn mpc_runtime_reprepares_when_model_changes() {
    let original = default_mpc_model();
    let mut sim = mpc_sim(original);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);

    let mut updated = original;
    updated.Q[10] *= 1.5;
    sim.controller = Controller::MPC(updated);
    sim.step(PENDULUM_FIXED_DT);

    assert!(!sim.mpc_cache_matches(original, PENDULUM_FIXED_DT));
    assert!(sim.mpc_cache_matches(updated, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 2);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn mpc_runtime_reprepares_when_timestep_changes() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);

    let updated_dt = PENDULUM_FIXED_DT * 0.5;
    sim.step(updated_dt);
    assert!(!sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert!(sim.mpc_cache_matches(model, updated_dt));
    assert_eq!(sim.mpc_preparations, 2);
    sim.step(updated_dt);
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn switching_away_from_mpc_discards_its_prepared_configuration() {
    let model = default_mpc_model();
    for controller in [
        Controller::lqr(Model::default()),
        Controller::pid(),
        Controller::policy(policy_snapshot(0.0)),
    ] {
        let mut sim = mpc_sim(model);
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
        assert_eq!(sim.mpc_preparations, 1);

        sim.controller = controller;
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache.is_none());
        assert_eq!(sim.mpc_preparations, 1);

        sim.controller = Controller::MPC(model);
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
        assert!(sim.lqr_cache.is_none());
        assert_eq!(sim.mpc_preparations, 2);
        assert!(sim.last_control_error().is_none());
    }
}

#[test]
fn resetting_mpc_clears_prepared_configuration_and_error() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);
    sim.last_control_error = Some("stale failure".to_owned());

    sim.reset_state();
    assert!(sim.mpc_cache.is_none());
    assert!(sim.lqr_cache.is_none());
    assert!(sim.last_control_error().is_none());

    // Keep the rollout deterministic despite reset's randomized starting angle.
    sim.state = vector![0.0, 0.0, 0.05, 0.0];
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn cached_mpc_rollout_matches_fresh_preparation_each_step() {
    let model = default_mpc_model();
    let mut cached = mpc_sim(model);
    let mut fresh = mpc_sim(model);
    for _ in 0..12 {
        fresh.mpc_cache = None;
        cached.step(PENDULUM_FIXED_DT);
        fresh.step(PENDULUM_FIXED_DT);
        assert!(cached.last_control_error().is_none());
        assert!(fresh.last_control_error().is_none());
        assert!((cached.state - fresh.state).norm() < 1e-6);
    }
    assert_eq!(cached.mpc_preparations, 1);
    assert_eq!(fresh.mpc_preparations, 12);
}
