//! Inverted-pendulum dynamics and controller-domain logic.
//!
//! The pendulum mode is intentionally a compact teaching environment. It uses a
//! shared four-state representation:
//!
//! `x = [cart_position, cart_velocity, rod_angle, rod_angular_velocity]`
//!
//! and advances the plant with the nonlinear cart-pole equations integrated by
//! fourth-order Runge-Kutta (RK4).
//!
//! The classical controllers still use the linearized upright model from
//! `rust_robotics_algo::control::inverted_pendulum`, but the live simulation
//! itself is stepped with the nonlinear plant so large-angle motion remains
//! physically plausible when the rod falls away from the equilibrium.
//!
//! Optional noise can perturb:
//!
//! - the measured state seen by the controller
//! - the control force actually applied to the plant
//!
//! which makes it easy to compare classical and learned controllers under the
//! same disturbance model.
#[cfg(target_arch = "wasm32")]
use super::super::ppo_trainer::PpoReplicaStatus;
use super::super::ppo_trainer::PpoTrainerCoordinator;
use super::super::{Draw, Simulate};
use crate::data::{IntoValues, TimeTable};
use crate::prelude::draw_cart;

use egui::Ui;
use egui_plot::{Line, PlotUi};
use rand::Rng;
use rb::inverted_pendulum::*;
use rb::prelude::*;
use rust_robotics_algo as rb;
use rust_robotics_algo::cart_pole::CartPoleParameters;
use rust_robotics_core::PolicySnapshot;
#[cfg(target_arch = "wasm32")]
use rust_robotics_core::PpoMetrics;
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig, PpoTrainerConfig};
use serde::{Deserialize, Serialize};

/// Pendulum state vector with ordering `[x, x_dot, theta, theta_dot]`.
pub type State = rb::Vector4;
/// Fixed simulator step used by both visualization and PPO training.
pub const PENDULUM_FIXED_DT: f32 = 0.01;

/// High-level toggle for stochastic perturbations in the pendulum demo.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NoiseConfig {
    pub enabled: bool,
    pub scale: f32,
}

/// Concrete noise amplitudes derived from the dimensionless `NoiseConfig`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoiseProfile {
    pub(crate) position_m: f32,
    pub(crate) velocity_mps: f32,
    pub(crate) angle_rad: f32,
    pub(crate) angular_velocity_radps: f32,
    pub(crate) force_n: f32,
}

impl NoiseConfig {
    /// Returns whether noise should currently be injected.
    pub(super) fn is_active(self) -> bool {
        self.enabled && self.scale > 0.0
    }

    /// Converts the user-facing noise scale into per-signal standard amplitudes.
    ///
    /// The constants were chosen as practical demo values rather than from a
    /// particular hardware sensor model.
    pub(super) fn profile(self) -> NoiseProfile {
        let scale = self.scale.max(0.0);
        NoiseProfile {
            position_m: 0.005 * scale,
            velocity_mps: 0.02 * scale,
            angle_rad: 0.004 * scale,
            angular_velocity_radps: 0.02 * scale,
            force_n: 0.2 * scale,
        }
    }
}

/// Controller variants supported by the pendulum demo.
///
/// All controllers consume the same state vector and emit the same scalar force
/// command, which makes side-by-side comparison straightforward.
#[derive(Debug, PartialEq, Clone)]
pub enum Controller {
    LQR(Model),
    PID(PID),
    MPC(Model),
    Policy(PolicySnapshot),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControllerKind {
    Lqr,
    Pid,
    Mpc,
    Policy,
}

impl ControllerKind {
    /// Human-readable label used in the UI controller selector.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Lqr => "LQR",
            Self::Pid => "PID",
            Self::Mpc => "MPC",
            Self::Policy => "PPO Policy",
        }
    }
}

/// User-facing pendulum plant parameters exposed in the DOM card.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PendulumParamsSnapshot {
    pub(crate) beam_length: f32,
    pub(crate) cart_mass: f32,
    pub(crate) ball_mass: f32,
}

/// Controller-specific parameters exposed for the currently selected controller.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg(target_arch = "wasm32")]
pub(crate) enum ControllerParamsSnapshot {
    Lqr {
        beam_length: f32,
        cart_mass: f32,
        ball_mass: f32,
        q_position: f32,
        q_velocity: f32,
        q_angle: f32,
        q_angular_velocity: f32,
        r_input: f32,
    },
    Pid {
        kp: f32,
        ki: f32,
        kd: f32,
    },
    Mpc {
        beam_length: f32,
        cart_mass: f32,
        ball_mass: f32,
        q_position: f32,
        q_velocity: f32,
        q_angle: f32,
        q_angular_velocity: f32,
        r_input: f32,
        horizon: usize,
    },
    Policy {
        ready: bool,
        action_std: Option<f32>,
    },
}

/// UI-facing snapshot of PPO trainer controls and health for one pendulum.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PolicyTrainerSnapshot {
    pub(crate) training_active: bool,
    pub(crate) initialized: bool,
    pub(crate) busy: bool,
    pub(crate) snapshot_ready: bool,
    pub(crate) total_replicas: usize,
    pub(crate) ready_replicas: usize,
    pub(crate) busy_replicas: usize,
    pub(crate) parallel_trainers: usize,
    pub(crate) training_updates_per_tick: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) epochs_per_update: usize,
    pub(crate) learning_rate: f64,
    pub(crate) action_std: f32,
    pub(crate) metrics: Option<PpoMetrics>,
    pub(crate) last_error: Option<String>,
}

/// Serialized state for one pendulum DOM card in the focused web embed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PendulumCardState {
    pub(crate) id: usize,
    pub(crate) controller: ControllerKind,
    pub(crate) pendulum: PendulumParamsSnapshot,
    pub(crate) controller_params: ControllerParamsSnapshot,
    pub(crate) policy_trainer: PolicyTrainerSnapshot,
    pub(crate) control_error: Option<String>,
}

/// Patch payload applied from the focused web embed to a pendulum instance.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PendulumPatch {
    pub(crate) controller: Option<ControllerKind>,
    pub(crate) beam_length: Option<f32>,
    pub(crate) cart_mass: Option<f32>,
    pub(crate) ball_mass: Option<f32>,
    pub(crate) lqr: Option<LqrPatch>,
    pub(crate) pid: Option<PidPatch>,
    pub(crate) mpc: Option<MpcPatch>,
    pub(crate) policy: Option<PolicyPatch>,
    pub(crate) trainer_action: Option<PendulumTrainerAction>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct LqrPatch {
    pub(crate) beam_length: Option<f32>,
    pub(crate) cart_mass: Option<f32>,
    pub(crate) ball_mass: Option<f32>,
    pub(crate) q_position: Option<f32>,
    pub(crate) q_velocity: Option<f32>,
    pub(crate) q_angle: Option<f32>,
    pub(crate) q_angular_velocity: Option<f32>,
    pub(crate) r_input: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PidPatch {
    pub(crate) kp: Option<f32>,
    pub(crate) ki: Option<f32>,
    pub(crate) kd: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct MpcPatch {
    pub(crate) beam_length: Option<f32>,
    pub(crate) cart_mass: Option<f32>,
    pub(crate) ball_mass: Option<f32>,
    pub(crate) q_position: Option<f32>,
    pub(crate) q_velocity: Option<f32>,
    pub(crate) q_angle: Option<f32>,
    pub(crate) q_angular_velocity: Option<f32>,
    pub(crate) r_input: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct PolicyPatch {
    pub(crate) parallel_trainers: Option<usize>,
    pub(crate) training_updates_per_tick: Option<usize>,
    pub(crate) rollout_steps: Option<usize>,
    pub(crate) epochs_per_update: Option<usize>,
    pub(crate) learning_rate: Option<f64>,
    pub(crate) action_std: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg(target_arch = "wasm32")]
pub(crate) enum PendulumTrainerAction {
    Start,
    Stop,
    Reset,
    Use,
}

#[cfg(target_arch = "wasm32")]
fn clamp_beam_length(value: f32) -> f32 {
    value.clamp(0.1, 10.0)
}

#[cfg(target_arch = "wasm32")]
fn clamp_cart_mass(value: f32) -> f32 {
    value.clamp(0.1, 3.0)
}

#[cfg(target_arch = "wasm32")]
fn clamp_ball_mass(value: f32) -> f32 {
    value.clamp(0.1, 10.0)
}

#[cfg(target_arch = "wasm32")]
fn clamp_weight(value: f32) -> f32 {
    value.clamp(0.0, 100.0)
}

#[cfg(target_arch = "wasm32")]
fn clamp_pid_gain(value: f32) -> f32 {
    value.clamp(0.01, 10000.0)
}

#[cfg(target_arch = "wasm32")]
fn clamp_learning_rate(value: f64) -> f64 {
    value.clamp(1.0e-5, 1.0e-2)
}

#[cfg(target_arch = "wasm32")]
fn clamp_action_std(value: f32) -> f32 {
    value.clamp(0.05, 10.0)
}

#[derive(Debug, Clone)]
pub(crate) struct LqrRuntimeCache {
    model: Model,
    dt: f32,
    prepared: rb::control::lqr::PreparedLqr<4, 1>,
}

pub(crate) struct MpcRuntimeCache {
    model: Model,
    dt: f32,
    prepared: PreparedMpc,
}

fn mpc_stage_cost_model(mut model: Model) -> Model {
    model.Q[0] = 1.0;
    model.Q[5] = 1.0;
    model.Q[10] = 10.0;
    model.Q[15] = 1.0;
    model
}

impl Controller {
    /// Computes the control command for the current pendulum state.
    ///
    /// - LQR and MPC operate on the model state directly
    /// - PID treats rod angle as the primary error signal
    /// - PPO runs deterministic actor inference from the exported snapshot
    pub fn control(&mut self, x: State, dt: f32) -> f32 {
        match self {
            Self::LQR(model) => model.control(x, dt)[0],
            Self::PID(pid) => pid.control(0.0 - x[2], dt),
            Self::MPC(model) => {
                #[cfg(target_arch = "wasm32")]
                let start = web_sys::window().unwrap().performance().unwrap().now();

                let u = mpc_control(x, *model, dt);

                #[cfg(target_arch = "wasm32")]
                {
                    let elapsed = web_sys::window().unwrap().performance().unwrap().now() - start;
                    web_sys::console::log_1(&format!("MPC took {:.2}ms", elapsed).into());
                }

                u
            }
            Self::Policy(policy) => policy.act([x[0], x[1], x[2], x[3]]),
        }
    }

    /// Builds an LQR-backed controller.
    pub fn lqr(model: Model) -> Self {
        Self::LQR(model)
    }

    /// Builds the default angle-based PID controller.
    pub fn pid() -> Self {
        Self::PID(PID::with_gains(25.0, 3.0, 3.0))
    }

    /// Builds an MPC-backed controller.
    pub fn mpc(model: Model) -> Self {
        Self::MPC(mpc_stage_cost_model(model))
    }

    /// Builds a controller from a trained PPO actor snapshot.
    pub fn policy(snapshot: PolicySnapshot) -> Self {
        Self::Policy(snapshot)
    }

    /// Returns the logical controller kind for UI/state synchronization.
    pub(crate) fn kind(&self) -> ControllerKind {
        match self {
            Self::LQR(_) => ControllerKind::Lqr,
            Self::PID(_) => ControllerKind::Pid,
            Self::MPC(_) => ControllerKind::Mpc,
            Self::Policy(_) => ControllerKind::Policy,
        }
    }

    /// Replaces the controller implementation while preserving the requested
    /// semantic kind.
    pub(crate) fn set_kind(
        &mut self,
        kind: ControllerKind,
        model: Model,
        available_policy: Option<&PolicySnapshot>,
    ) {
        match kind {
            ControllerKind::Lqr => *self = Self::lqr(model),
            ControllerKind::Pid => *self = Self::pid(),
            ControllerKind::Mpc => *self = Self::mpc(model),
            ControllerKind::Policy => {
                if let Some(policy) = available_policy {
                    *self = Self::policy(policy.clone());
                }
            }
        }
    }

    /// Updates the embedded PPO policy in-place.
    pub fn sync_policy(&mut self, snapshot: &PolicySnapshot) {
        if let Self::Policy(policy) = self {
            *policy = snapshot.clone();
        }
    }

    /// Resets controller-internal dynamic state while preserving configuration.
    pub fn reset_state(&mut self) {
        match self {
            Self::LQR(_) => (),
            Self::PID(pid) => pid.reset_state(),
            Self::MPC(_) => (),
            Self::Policy(_) => (),
        }
    }

    /// Restores controller defaults.
    pub fn reset_all(&mut self) {
        match self {
            Self::LQR(_) => *self = Self::lqr(Model::default()),
            Self::PID(_) => *self = Self::pid(),
            Self::MPC(_) => *self = Self::mpc(Model::default()),
            Self::Policy(policy) => *self = Self::policy(policy.clone()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn params_snapshot(&self) -> ControllerParamsSnapshot {
        match self {
            Self::LQR(model) => ControllerParamsSnapshot::Lqr {
                beam_length: model.l_bar,
                cart_mass: model.m_cart,
                ball_mass: model.m_ball,
                q_position: model.Q[0],
                q_velocity: model.Q[5],
                q_angle: model.Q[10],
                q_angular_velocity: model.Q[15],
                r_input: model.R[0],
            },
            Self::PID(pid) => ControllerParamsSnapshot::Pid {
                kp: pid.P,
                ki: pid.I,
                kd: pid.D,
            },
            Self::MPC(model) => ControllerParamsSnapshot::Mpc {
                beam_length: model.l_bar,
                cart_mass: model.m_cart,
                ball_mass: model.m_ball,
                q_position: model.Q[0],
                q_velocity: model.Q[5],
                q_angle: model.Q[10],
                q_angular_velocity: model.Q[15],
                r_input: model.R[0],
                horizon: 12,
            },
            Self::Policy(policy) => ControllerParamsSnapshot::Policy {
                ready: true,
                action_std: Some(policy.action_std()),
            },
        }
    }
}

/// One pendulum simulation instance shown in the UI.
///
/// Each instance owns:
/// - current plant state
/// - selected controller implementation
/// - time-series data for plotting
/// - PPO trainer state and configuration
/// - local UI metadata such as `id`
pub struct InvertedPendulum {
    pub(crate) state: State,
    pub(crate) controller: Controller,
    pub(crate) controller_selection: ControllerKind,
    pub(crate) model: Model,
    pub(crate) id: usize,
    pub(crate) data: TimeTable,
    pub(crate) time_init: f32,
    pub(crate) visual_episode_steps: usize,
    pub(crate) policy_observation: Option<[f32; 4]>,
    pub(crate) active_training_environment: Option<(CartPoleParameters, PendulumEnvConfig)>,
    pub(crate) policy_environment_stale: bool,
    pub(crate) trainer_config: PpoTrainerConfig,
    pub(crate) trainer_backend: PpoTrainerCoordinator,
    pub(crate) training_active: bool,
    pub(crate) training_updates_per_tick: usize,
    pub(crate) parallel_trainers: usize,
    pub(crate) lqr_cache: Option<LqrRuntimeCache>,
    pub(crate) mpc_cache: Option<MpcRuntimeCache>,
    #[cfg(test)]
    pub(crate) mpc_preparations: usize,
    pub(crate) last_control_error: Option<String>,
}

impl Default for InvertedPendulum {
    fn default() -> Self {
        let state = vector![0., 0., rand(0.4), 0.];
        let data = TimeTable::init_with_names(vec![
            "Lateral Position",
            "Lateral Velocity",
            "Rod Angle",
            "Rod Angular Velocity",
            "Control Input",
        ]);
        let mut trainer_config = PpoTrainerConfig::default();
        trainer_config.env.dt = PENDULUM_FIXED_DT;

        Self {
            state,
            controller: Controller::lqr(Model::default()),
            controller_selection: ControllerKind::Lqr,
            model: Model::default(),
            id: 1,
            time_init: 0.0,
            data,
            visual_episode_steps: 0,
            policy_observation: None,
            active_training_environment: None,
            policy_environment_stale: false,
            trainer_config,
            trainer_backend: PpoTrainerCoordinator::default(),
            training_active: false,
            training_updates_per_tick: 1,
            parallel_trainers: 1,
            lqr_cache: None,
            mpc_cache: None,
            #[cfg(test)]
            mpc_preparations: 0,
            last_control_error: None,
        }
    }
}

impl InvertedPendulum {
    /// Labels for the five signals stored in the time-series table.
    pub const SIGNAL_LABELS: [&'static str; 5] = [
        "Lateral Position",
        "Lateral Velocity",
        "Rod Angle",
        "Rod Angular Velocity",
        "Control Input",
    ];

    /// Creates a new pendulum instance anchored to the given simulation time.
    pub fn new(id: usize, time: f32) -> Self {
        Self {
            id,
            time_init: time,
            ..Default::default()
        }
    }

    /// Stable identifier used to label cards and plots.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the cart position in meters.
    pub fn x_position(&self) -> f32 {
        self.state[0]
    }

    /// Returns the rod angle in radians.
    pub fn rod_angle(&self) -> f32 {
        self.state[2]
    }

    /// Exposes the underlying pendulum model for UI and tests.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Classical controllers and PPO use the same physical RK4 plant.
    fn integrate_plant_rk4(&self, state: State, control: f32, dt: f32) -> State {
        CartPoleParameters::from(self.model).step(state, control, dt)
    }

    #[cfg(target_arch = "wasm32")]
    fn policy_trainer_snapshot(&self) -> PolicyTrainerSnapshot {
        let PpoReplicaStatus { total, ready, busy } = self.trainer_backend.status();

        PolicyTrainerSnapshot {
            training_active: self.training_active,
            initialized: self.trainer_backend.is_initialized(),
            busy: self.trainer_backend.busy(),
            snapshot_ready: self.trainer_backend.snapshot().is_some(),
            total_replicas: total,
            ready_replicas: ready,
            busy_replicas: busy,
            parallel_trainers: self.parallel_trainers,
            training_updates_per_tick: self.training_updates_per_tick,
            rollout_steps: self.trainer_config.ppo.rollout_steps,
            epochs_per_update: self.trainer_config.ppo.epochs_per_update,
            learning_rate: self.trainer_config.ppo.learning_rate,
            action_std: self.trainer_config.action_std,
            metrics: self.trainer_backend.metrics().cloned(),
            last_error: if self.policy_environment_stale {
                Some(
                    "Plant/noise changed: restart PPO training before using this policy."
                        .to_owned(),
                )
            } else {
                self.trainer_backend.last_error().map(str::to_owned)
            },
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn card_state(&self) -> PendulumCardState {
        let controller_params = if self.controller_selection == ControllerKind::Policy
            && self.controller.kind() != ControllerKind::Policy
        {
            ControllerParamsSnapshot::Policy {
                ready: false,
                action_std: None,
            }
        } else {
            self.controller.params_snapshot()
        };

        PendulumCardState {
            id: self.id,
            controller: self.controller_selection,
            pendulum: PendulumParamsSnapshot {
                beam_length: self.model.l_bar,
                cart_mass: self.model.m_cart,
                ball_mass: self.model.m_ball,
            },
            controller_params,
            policy_trainer: self.policy_trainer_snapshot(),
            control_error: self.last_control_error.clone(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn apply_patch(&mut self, patch: PendulumPatch) {
        if let Some(controller) = patch.controller {
            self.select_controller_kind(controller);
        }

        if let Some(beam_length) = patch.beam_length {
            self.model.l_bar = clamp_beam_length(beam_length);
        }
        if let Some(cart_mass) = patch.cart_mass {
            self.model.m_cart = clamp_cart_mass(cart_mass);
        }
        if let Some(ball_mass) = patch.ball_mass {
            self.model.m_ball = clamp_ball_mass(ball_mass);
        }

        if let Some(patch) = patch.lqr {
            if let Controller::LQR(model) = &mut self.controller {
                if let Some(beam_length) = patch.beam_length {
                    model.l_bar = clamp_beam_length(beam_length);
                }
                if let Some(cart_mass) = patch.cart_mass {
                    model.m_cart = clamp_cart_mass(cart_mass);
                }
                if let Some(ball_mass) = patch.ball_mass {
                    model.m_ball = clamp_ball_mass(ball_mass);
                }
                if let Some(q_position) = patch.q_position {
                    model.Q[0] = clamp_weight(q_position);
                }
                if let Some(q_velocity) = patch.q_velocity {
                    model.Q[5] = clamp_weight(q_velocity);
                }
                if let Some(q_angle) = patch.q_angle {
                    model.Q[10] = clamp_weight(q_angle);
                }
                if let Some(q_angular_velocity) = patch.q_angular_velocity {
                    model.Q[15] = clamp_weight(q_angular_velocity);
                }
                if let Some(r_input) = patch.r_input {
                    model.R[0] = clamp_weight(r_input);
                }
            }
        }

        if let Some(patch) = patch.pid {
            if let Controller::PID(pid) = &mut self.controller {
                if let Some(kp) = patch.kp {
                    pid.P = clamp_pid_gain(kp);
                }
                if let Some(ki) = patch.ki {
                    pid.I = clamp_pid_gain(ki);
                }
                if let Some(kd) = patch.kd {
                    pid.D = clamp_pid_gain(kd);
                }
            }
        }

        if let Some(patch) = patch.mpc {
            if let Controller::MPC(model) = &mut self.controller {
                if let Some(beam_length) = patch.beam_length {
                    model.l_bar = clamp_beam_length(beam_length);
                }
                if let Some(cart_mass) = patch.cart_mass {
                    model.m_cart = clamp_cart_mass(cart_mass);
                }
                if let Some(ball_mass) = patch.ball_mass {
                    model.m_ball = clamp_ball_mass(ball_mass);
                }
                if let Some(q_position) = patch.q_position {
                    model.Q[0] = clamp_weight(q_position);
                }
                if let Some(q_velocity) = patch.q_velocity {
                    model.Q[5] = clamp_weight(q_velocity);
                }
                if let Some(q_angle) = patch.q_angle {
                    model.Q[10] = clamp_weight(q_angle);
                }
                if let Some(q_angular_velocity) = patch.q_angular_velocity {
                    model.Q[15] = clamp_weight(q_angular_velocity);
                }
                if let Some(r_input) = patch.r_input {
                    model.R[0] = clamp_weight(r_input);
                }
            }
        }

        if let Some(patch) = patch.policy {
            if let Some(parallel_trainers) = patch.parallel_trainers {
                self.parallel_trainers = parallel_trainers.clamp(1, 32);
            }
            if let Some(training_updates_per_tick) = patch.training_updates_per_tick {
                self.training_updates_per_tick = training_updates_per_tick.clamp(1, 32);
            }
            if let Some(rollout_steps) = patch.rollout_steps {
                self.trainer_config.ppo.rollout_steps = rollout_steps.clamp(32, 8192);
            }
            if let Some(epochs_per_update) = patch.epochs_per_update {
                self.trainer_config.ppo.epochs_per_update = epochs_per_update.clamp(1, 16);
            }
            if let Some(learning_rate) = patch.learning_rate {
                self.trainer_config.ppo.learning_rate = clamp_learning_rate(learning_rate);
            }
            if let Some(action_std) = patch.action_std {
                self.trainer_config.action_std = clamp_action_std(action_std);
            }
        }

        self.validate_training_environment();
        if let Some(action) = patch.trainer_action {
            match action {
                PendulumTrainerAction::Start => self.start_training(),
                PendulumTrainerAction::Stop => self.stop_training(),
                PendulumTrainerAction::Reset => {
                    let was_training = self.training_active;
                    self.reset_trainer();
                    if was_training {
                        self.start_training();
                    }
                }
                PendulumTrainerAction::Use => {
                    if let Some(snapshot) = self.trainer_backend.snapshot().cloned() {
                        self.set_policy_controller(&snapshot);
                        self.controller_selection = ControllerKind::Policy;
                    }
                }
            }
        }
    }

    /// Plots one recorded signal for this pendulum instance.
    pub fn plot_signal(&self, plot_ui: &mut PlotUi<'_>, signal_index: usize) {
        if signal_index >= Self::SIGNAL_LABELS.len() {
            return;
        }

        let line_name = format!("{}_{}", Self::SIGNAL_LABELS[signal_index], self.id);
        if let Some(values) = self.data.values_shifted(signal_index, self.time_init, 0.0) {
            plot_ui.line(Line::new(&line_name, values));
        }
    }

    /// Advances the pendulum by one fixed step with optional injected noise.
    ///
    /// The update follows three stages:
    ///
    /// 1. derive the measured state seen by the controller
    /// 2. compute the controller command from that measured state
    /// 3. perturb the applied force if actuation noise is enabled and advance the
    ///    true nonlinear plant with RK4
    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        self.step_with_rng(dt, noise, &mut rand::thread_rng());
    }

    pub(crate) fn step_with_rng<R: Rng + ?Sized>(
        &mut self,
        dt: f32,
        noise: NoiseConfig,
        rng: &mut R,
    ) {
        self.configure_training_noise(noise);
        self.validate_training_environment();
        if self.controller.kind() == ControllerKind::Policy {
            self.step_policy_with_rng(dt, rng);
            return;
        }
        let mut measured_state = self.state;
        if noise.is_active() {
            let profile = noise.profile();
            measured_state[0] += rand(profile.position_m);
            measured_state[1] += rand(profile.velocity_mps);
            measured_state[2] += rand(profile.angle_rad);
            measured_state[3] += rand(profile.angular_velocity_radps);
        }

        let control_command = match &mut self.controller {
            Controller::LQR(model) => {
                self.mpc_cache = None;
                let needs_prepare = self.lqr_cache.as_ref().is_none_or(|cache| {
                    cache.model != *model || cache.dt.to_bits() != dt.to_bits()
                });
                if needs_prepare {
                    self.lqr_cache = Some(LqrRuntimeCache {
                        model: *model,
                        dt,
                        prepared: model.prepare(dt),
                    });
                }
                self.last_control_error = None;
                self.lqr_cache
                    .as_ref()
                    .expect("LQR cache is prepared above")
                    .prepared
                    .control(measured_state)[0]
            }
            Controller::PID(pid) => {
                self.lqr_cache = None;
                self.mpc_cache = None;
                self.last_control_error = None;
                pid.control(-measured_state[2], dt)
            }
            Controller::MPC(model) => {
                self.lqr_cache = None;
                let needs_prepare = self.mpc_cache.as_ref().is_none_or(|cache| {
                    cache.model != *model || cache.dt.to_bits() != dt.to_bits()
                });
                if needs_prepare {
                    self.mpc_cache = Some(MpcRuntimeCache {
                        model: *model,
                        dt,
                        prepared: PreparedMpc::new(*model, dt),
                    });
                    #[cfg(test)]
                    {
                        self.mpc_preparations += 1;
                    }
                }
                match self
                    .mpc_cache
                    .as_ref()
                    .expect("MPC cache is prepared above")
                    .prepared
                    .try_control(measured_state)
                {
                    Ok(control) => {
                        self.last_control_error = None;
                        control
                    }
                    Err(error) => {
                        self.last_control_error = Some(error.to_string());
                        0.0
                    }
                }
            }
            Controller::Policy(policy) => {
                self.lqr_cache = None;
                self.mpc_cache = None;
                self.last_control_error = None;
                policy.act([
                    measured_state[0],
                    measured_state[1],
                    measured_state[2],
                    measured_state[3],
                ])
            }
        };
        let applied_control = if noise.is_active() {
            control_command + rand(noise.profile().force_n)
        } else {
            control_command
        };

        self.state = self.integrate_plant_rk4(self.state, applied_control, dt);
        self.visual_episode_steps += 1;
        self.data.add(
            self.data.time_last() + dt,
            vec![
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                applied_control,
            ],
        );

        if self.training_active && self.visual_episode_done() {
            self.reset_state();
        }
    }

    /// The exact PPO environment path, including pre-reset observations and
    /// termination after Stop. The caller owns noise draws; no GUI clock is used.
    pub(crate) fn step_policy_with_rng<R: Rng + ?Sized>(&mut self, dt: f32, rng: &mut R) {
        self.lqr_cache = None;
        self.mpc_cache = None;
        self.validate_training_environment();
        if self.policy_environment_stale {
            self.last_control_error = Some("Plant/noise changed: restart PPO training.".to_owned());
            return;
        }
        if self.active_training_environment.is_some() && self.trainer_backend.snapshot().is_none() {
            // A web reset may not have published its new snapshot yet. Never
            // execute retained old weights on the newly configured environment.
            self.last_control_error =
                Some("Waiting for the matching PPO policy snapshot.".to_owned());
            return;
        }
        assert_eq!(
            dt, self.trainer_config.env.dt,
            "PPO live timestep must match training"
        );
        let mut env = PendulumEnv::from_state(
            self.model,
            self.trainer_config.env,
            self.state,
            self.visual_episode_steps,
        );
        let observation = self
            .policy_observation
            .take()
            .unwrap_or_else(|| env.observation_with_rng(rng));
        let Controller::Policy(policy) = &self.controller else {
            unreachable!("PPO path only");
        };
        let result = env.step_with_rng(policy.act(observation), rng);
        self.state = env.state();
        self.visual_episode_steps += 1;
        self.policy_observation = Some(result.observation);
        self.last_control_error = None;
        self.data.add(
            self.data.time_last() + dt,
            vec![
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                env.last_applied_force(),
            ],
        );
        if result.done {
            self.policy_observation = Some(env.reset_with_rng(rng));
            self.state = env.state();
            self.visual_episode_steps = 0;
            self.time_init = 0.0;
            self.data.clear();
        }
    }

    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        let entering_policy = self.controller.kind() != ControllerKind::Policy;
        self.controller = Controller::policy(snapshot.clone());
        self.controller_selection = ControllerKind::Policy;
        if entering_policy {
            self.reset_state();
        }
    }

    pub fn sync_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        self.controller.sync_policy(snapshot);
    }

    #[cfg(test)]
    pub(crate) fn lqr_cache_matches(&self, model: Model, dt: f32) -> bool {
        self.lqr_cache
            .as_ref()
            .is_some_and(|cache| cache.model == model && cache.dt.to_bits() == dt.to_bits())
    }

    #[cfg(test)]
    pub(crate) fn mpc_cache_matches(&self, model: Model, dt: f32) -> bool {
        self.mpc_cache
            .as_ref()
            .is_some_and(|cache| cache.model == model && cache.dt.to_bits() == dt.to_bits())
    }

    pub fn last_control_error(&self) -> Option<&str> {
        self.last_control_error.as_deref()
    }

    pub(crate) fn visual_episode_done(&self) -> bool {
        let env = &self.trainer_config.env;
        self.state[2].abs() > env.max_angle_rad
            || self.state[0].abs() > env.max_position_m
            || self.visual_episode_steps >= env.max_steps
    }
}

impl Simulate for InvertedPendulum {
    fn get_state(&self) -> &dyn std::any::Any {
        &self.state
    }

    fn match_state_with(&mut self, other: &dyn Simulate) {
        if let Some(data) = other.get_state().downcast_ref::<State>() {
            self.state.clone_from(data);
            self.policy_observation = None;
        }
    }

    fn step(&mut self, dt: f32) {
        self.step_with_noise(dt, NoiseConfig::default());
    }

    fn reset_state(&mut self) {
        if self.controller.kind() == ControllerKind::Policy {
            let mut rng = rand::thread_rng();
            let mut env =
                PendulumEnv::from_state(self.model, self.trainer_config.env, self.state, 0);
            self.policy_observation = Some(env.reset_with_rng(&mut rng));
            self.state = env.state();
        } else {
            self.state = vector![0., 0., rand(0.4), 0.];
            self.policy_observation = None;
        }
        self.time_init = 0.0;
        self.visual_episode_steps = 0;
        self.controller.reset_state();
        self.lqr_cache = None;
        self.mpc_cache = None;
        self.last_control_error = None;
        self.data.clear();
    }

    fn reset_all(&mut self) {
        self.trainer_backend.destroy();
        *self = Self::default();
    }
}

impl Draw for InvertedPendulum {
    fn plot(&self, plot_ui: &mut PlotUi<'_>) {
        let names: Vec<String> = self
            .data
            .names()
            .iter()
            .map(|name| format!("{}_{}", name, self.id))
            .collect();

        (0..self.data.ncols()).for_each(|i| {
            if let Some(values) = self.data.values_shifted(i, self.time_init, 0.0) {
                plot_ui.line(Line::new(&names[i], values));
            }
        });
    }

    fn scene(&self, plot_ui: &mut PlotUi<'_>) {
        draw_cart(
            plot_ui,
            self.x_position(),
            self.rod_angle(),
            &self.model,
            &format!("Cart {}", self.id),
        );
    }

    fn options(&mut self, ui: &mut Ui) -> bool {
        self.options_with_policy(ui)
    }
}

pub(super) fn rand(max: f32) -> f32 {
    rand::thread_rng().gen_range(-max..max)
}
