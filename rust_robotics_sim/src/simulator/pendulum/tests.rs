use super::domain::ControllerKind;
use super::*;
use crate::simulator::Simulate;
use rust_robotics_algo::control::StateSpace;
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
    let mut sim = InvertedPendulum::default();
    sim.state = vector![0.1, -0.2, 0.05, 0.3];
    let expected = {
        let (a, b) = sim.model.model(0.01);
        let u = sim.controller.control(sim.state, 0.01);
        a * sim.state + b * u
    };

    sim.step_with_noise(0.01, config);

    assert_eq!(sim.state, expected);
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
    let mut sim = InvertedPendulum::default();
    sim.state = vector![0.3, 0.0, 0.2, 0.0];
    sim.controller = Controller::lqr(Model::default());
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
