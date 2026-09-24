//! Scoped test-only batch augmentation; no production trainer owns this state.
use super::*;
use std::cell::RefCell;

pub(super) struct Receipt {
    pub(super) original: RolloutBatch,
    pub(super) original_raw: Vec<f32>,
    pub(super) training: RolloutBatch,
    pub(super) raw: Vec<f32>,
}
struct Injection {
    extra: Option<(RolloutBatch, Vec<f32>)>,
    receipt: Option<Receipt>,
}
thread_local! {
    static INJECTION: RefCell<Option<Injection>> = const { RefCell::new(None) };
    static STRESS_EVALUATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) struct Scope { active: bool }
impl Scope {
    pub(super) fn enter(extra: Option<(RolloutBatch, Vec<f32>)>) -> Self {
        INJECTION.with(|state| {
            let mut state = state.borrow_mut();
            assert!(state.is_none(), "nested coverage injection");
            if let Some((batch, raw)) = &extra {
                validate(batch, raw);
                assert_eq!(raw.len(), 512);
            }
            *state = Some(Injection { extra, receipt: None });
        });
        Self { active: true }
    }
    pub(super) fn finish(mut self) -> Receipt {
        let item = INJECTION.with(|state| state.borrow_mut().take().unwrap());
        self.active = false;
        item.receipt.expect("coverage hook was not called")
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if self.active { INJECTION.with(|state| { state.borrow_mut().take(); }); }
    }
}
fn validate(batch: &RolloutBatch, raw: &[f32]) {
    let n = raw.len();
    assert!(n > 0);
    assert_eq!(batch.observations.len(), n);
    for field in [&batch.latent_actions, &batch.old_log_probs, &batch.returns, &batch.advantages] {
        assert_eq!(field.len(), n);
        assert!(field.iter().all(|v| v.is_finite()));
    }
    assert!(raw.iter().all(|v| v.is_finite()));
    assert!(batch.observations.iter().flatten().all(|v| v.is_finite()));
}
/// Inactive is exact identity, including already-normalized advantage bits.
pub(super) fn augment(mut batch: RolloutBatch, mut raw: Vec<f32>) -> (RolloutBatch, Vec<f32>) {
    INJECTION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(injection) = slot.as_mut() else { return (batch, raw); };
        assert!(injection.receipt.is_none(), "coverage hook called twice");
        validate(&batch, &raw);
        assert_eq!(raw.len(), 1024);
        let original = batch.clone();
        let original_raw = raw.clone();
        if let Some((extra, extra_raw)) = &injection.extra {
            batch.observations.extend_from_slice(&extra.observations);
            batch.latent_actions.extend_from_slice(&extra.latent_actions);
            batch.old_log_probs.extend_from_slice(&extra.old_log_probs);
            batch.returns.extend_from_slice(&extra.returns);
            raw.extend_from_slice(extra_raw);
            batch.advantages = normalize(&raw);
            assert_eq!(raw.len(), 1536);
        }
        validate(&batch, &raw);
        injection.receipt = Some(Receipt { original, original_raw, training: batch.clone(), raw: raw.clone() });
        (batch, raw)
    })
}

/// The same predefined conditional reset law is used for training and testing,
/// but initial-state and future-noise domains are kept separate.
pub(super) fn stress_state(cfg: PendulumEnvConfig, rng: &mut StdRng) -> [f32; 4] {
    let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    [rng.gen_range(-cfg.reset_position_range_m..cfg.reset_position_range_m),
     rng.gen_range(-cfg.reset_velocity_range_mps..cfg.reset_velocity_range_mps),
     sign * rng.gen_range(0.5 * cfg.reset_angle_range_rad..cfg.reset_angle_range_rad),
     sign * rng.gen_range(0.5 * cfg.reset_angular_velocity_range_radps..cfg.reset_angular_velocity_range_radps)]
}
pub(super) struct Evaluation;
impl Evaluation {
    pub(super) fn enter() -> Self {
        STRESS_EVALUATION.with(|v| { assert!(!v.get()); v.set(true); });
        Self
    }
}
impl Drop for Evaluation {
    fn drop(&mut self) { STRESS_EVALUATION.with(|v| v.set(false)); }
}
/// Called after the original initializer; never advances its future-noise RNG.
pub(super) fn evaluation_start(env: PendulumEnv, obs: [f32; 4], key: u64) -> (PendulumEnv, [f32; 4]) {
    if !STRESS_EVALUATION.with(|v| v.get()) { return (env, obs); }
    let mut rng = StdRng::seed_from_u64(key ^ 0x5354_5245_5353_0001);
    let state = stress_state(env.config(), &mut rng);
    let changed = PendulumEnv::from_state(env.model(), env.config(), rust_robotics_algo::Vector4::from_column_slice(&state), 0);
    let observation = changed.observation_with_rng(&mut rng);
    (changed, observation)
}
