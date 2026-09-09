//! Temporary native audit ABI. Single calling thread; no production API change.
//! No raw pointers, alternative dynamics, noiseless observations, or new trainer.
use crate::{PendulumEnv, PendulumEnvConfig, PpoTrainerConfig, PpoTrainerSession};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CStep {
    observation: [f32; 4],
    state: [f32; 4],
    reward: f32,
    force: f32,
    terminated: u32,
    truncated: u32,
    steps: u32,
    status: u32,
}

#[repr(C)]
pub struct CWeights {
    words: [f32; 4545],
    status: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct CMetrics {
    updates: u64,
    steps: u64,
    episodes: u64,
    policy_loss: f32,
    value_loss: f32,
    status: u32,
}

struct Environment {
    env: PendulumEnv,
    rng: StdRng,
    first_reset: bool,
    last: CStep,
}

thread_local! {
    static ENVS: RefCell<Vec<Option<Environment>>> = const { RefCell::new(Vec::new()) };
    static TRAINERS: RefCell<Vec<Option<PpoTrainerSession>>> = const { RefCell::new(Vec::new()) };
}

fn environment_rng(seed: u64, training: bool) -> StdRng {
    if training {
        let mut root = StdRng::seed_from_u64(seed);
        let _: u64 = root.gen(); // Actor initialization child.
        let _: u64 = root.gen(); // Critic initialization child.
        StdRng::seed_from_u64(root.gen())
    } else {
        StdRng::seed_from_u64(seed)
    }
}

fn output(env: &PendulumEnv, observation: [f32; 4]) -> CStep {
    let s = env.state();
    CStep { observation, state: [s[0], s[1], s[2], s[3]], ..Default::default() }
}
fn bad_step() -> CStep { CStep { status: 1, ..Default::default() } }

#[no_mangle]
pub extern "C" fn rr_env_create(seed: u64, max_steps: u32, training: u32) -> u64 {
    if max_steps == 0 || training > 1 { return 0; }
    let mut rng = environment_rng(seed, training == 1);
    let env = PendulumEnv::new_with_rng(Default::default(), PendulumEnvConfig {
        max_steps: max_steps as usize, ..Default::default()
    }, &mut rng);
    // Match the actual trainer's initialization, including this observation draw.
    let last = output(&env, env.observation_with_rng(&mut rng));
    ENVS.with(|v| {
        let mut v = v.borrow_mut();
        v.push(Some(Environment { env, rng, first_reset: true, last }));
        v.len() as u64
    })
}

#[no_mangle]
pub extern "C" fn rr_env_reset(handle: u64) -> CStep {
    ENVS.with(|v| {
        let mut v = v.borrow_mut();
        let Some(Some(e)) = v.get_mut(handle.wrapping_sub(1) as usize) else { return bad_step(); };
        if e.first_reset { e.first_reset = false; } else {
            let obs = e.env.reset_with_rng(&mut e.rng);
            e.last = output(&e.env, obs);
        }
        e.last
    })
}

#[no_mangle]
pub extern "C" fn rr_env_step(handle: u64, action: f32, transform: u32) -> CStep {
    if !action.is_finite() || transform > 2 { return bad_step(); }
    ENVS.with(|v| {
        let mut v = v.borrow_mut();
        let Some(Some(e)) = v.get_mut(handle.wrapping_sub(1) as usize) else { return bad_step(); };
        if e.first_reset || e.last.terminated + e.last.truncated > 0 { return bad_step(); }
        let force = match transform {
            0 => action,
            1 => 20.0 * action.tanh(),
            _ => 20.0 * action.clamp(-1.0, 1.0),
        };
        let step = e.env.step_with_rng(force, &mut e.rng);
        e.last = CStep {
            reward: step.reward, force: force.clamp(-20.0, 20.0),
            terminated: u32::from(step.terminated()), truncated: u32::from(step.truncated),
            steps: e.last.steps + 1, ..output(&e.env, step.observation)
        };
        e.last
    })
}

#[no_mangle]
pub extern "C" fn rr_env_free(handle: u64) -> u32 {
    ENVS.with(|v| match v.borrow_mut().get_mut(handle.wrapping_sub(1) as usize) {
        Some(slot) if slot.is_some() => { *slot = None; 0 }, _ => 1
    })
}

#[no_mangle]
pub extern "C" fn rr_trainer_create(seed: u64) -> u64 {
    TRAINERS.with(|v| {
        let mut v = v.borrow_mut();
        v.push(Some(PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed)));
        v.len() as u64
    })
}

#[no_mangle]
pub extern "C" fn rr_trainer_update(handle: u64, updates: u32) -> CMetrics {
    TRAINERS.with(|v| {
        let mut v = v.borrow_mut();
        let Some(Some(t)) = v.get_mut(handle.wrapping_sub(1) as usize) else {
            return CMetrics { status: 1, ..Default::default() };
        };
        t.train_updates(updates as usize);
        let m = t.metrics();
        CMetrics { updates: m.total_updates as u64, steps: m.total_env_steps as u64,
            episodes: m.total_episodes as u64, policy_loss: m.last_policy_loss,
            value_loss: m.last_value_loss, status: 0 }
    })
}

#[no_mangle]
pub extern "C" fn rr_trainer_weights(handle: u64, critic: u32) -> CWeights {
    TRAINERS.with(|v| {
        let v = v.borrow();
        let Some(Some(t)) = v.get(handle.wrapping_sub(1) as usize) else {
            return CWeights { words: [0.0; 4545], status: 1 };
        };
        if critic > 1 { return CWeights { words: [0.0; 4545], status: 1 }; }
        let s = t.shared_state();
        let layers = if critic == 1 { [s.value.input, s.value.hidden, s.value.output] }
            else { [s.policy.input, s.policy.hidden, s.policy.output] };
        let mut words = Vec::with_capacity(4545);
        for layer in layers { words.extend(layer.weight); words.extend(layer.bias); }
        CWeights { words: words.try_into().expect("default network shape"), status: 0 }
    })
}

#[no_mangle]
pub extern "C" fn rr_trainer_act(handle: u64, x: f32, v: f32, a: f32, w: f32) -> f32 {
    TRAINERS.with(|t| {
        let t = t.borrow();
        let Some(Some(t)) = t.get(handle.wrapping_sub(1) as usize) else { return f32::NAN; };
        t.snapshot().act([x, v, a, w])
    })
}

#[no_mangle]
pub extern "C" fn rr_trainer_free(handle: u64) -> u32 {
    TRAINERS.with(|v| match v.borrow_mut().get_mut(handle.wrapping_sub(1) as usize) {
        Some(slot) if slot.is_some() => { *slot = None; 0 }, _ => 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_matches_direct_noisy_rust_trajectories_resets_and_boundaries() {
        for training in [0, 1] {
            for cap in [1, 7, 5000] {
                let seed = 201;
                let handle = rr_env_create(seed, cap, training);
                let mut rng = environment_rng(seed, training == 1);
                let mut env = PendulumEnv::new_with_rng(Default::default(), PendulumEnvConfig {
                    max_steps: cap as usize, ..Default::default()
                }, &mut rng);
                let mut expected = output(&env, env.observation_with_rng(&mut rng));
                assert_eq!(rr_env_reset(handle), expected);
                for i in 0..2000 {
                    let action = ((i % 41) as f32 - 20.0) / 10.0;
                    let force = 20.0 * action.tanh();
                    let r = env.step_with_rng(force, &mut rng);
                    expected = CStep { reward: r.reward, force, terminated: u32::from(r.terminated()),
                        truncated: u32::from(r.truncated), steps: expected.steps + 1,
                        ..output(&env, r.observation) };
                    assert_eq!(rr_env_step(handle, action, 1), expected);
                    if r.done {
                        assert_ne!(rr_env_step(handle, action, 1).status, 0);
                        let obs = env.reset_with_rng(&mut rng);
                        expected = output(&env, obs);
                        assert_eq!(rr_env_reset(handle), expected);
                    }
                }
                assert_eq!(rr_env_free(handle), 0);
            }
        }
    }
    #[test]
    fn abi_invalid_actions_and_handles_are_atomic() {
        let a = rr_env_create(204, 5000, 0); let b = rr_env_create(204, 5000, 0);
        assert_eq!(rr_env_reset(a), rr_env_reset(b));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_ne!(rr_env_step(a, value, 0).status, 0);
        }
        assert_ne!(rr_env_step(a, 0.0, 99).status, 0);
        assert_ne!(rr_env_step(0, 0.0, 0).status, 0);
        assert_eq!(rr_env_step(a, 3.0, 0), rr_env_step(b, 3.0, 0));
        assert_eq!(rr_env_free(a), 0); assert_eq!(rr_env_free(a), 1);
        assert_eq!(rr_env_free(b), 0);
    }
    #[test]
    fn abi_normalized_force_mapping_matches_direct_force() {
        for x in [-3.0_f32, -1.0, -0.3, 0.0, 0.8, 1.0, 5.0] {
            let a = rr_env_create(202, 7, 0); let b = rr_env_create(202, 7, 0);
            assert_eq!(rr_env_reset(a), rr_env_reset(b));
            assert_eq!(rr_env_step(a, x, 2), rr_env_step(b, 20.0*x.clamp(-1.0,1.0), 0));
            rr_env_free(a); rr_env_free(b);
        }
    }
    #[test]
    fn abi_observation_and_weight_reads_do_not_change_training() {
        let h = rr_trainer_create(201);
        let initial = rr_trainer_weights(h, 0).words;
        assert_eq!(rr_trainer_update(h, 0).steps, 0);
        assert_eq!(rr_trainer_weights(h, 0).words, initial);
        let mut direct = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 201);
        for _ in 0..2 {
            assert!(rr_trainer_act(h, 0.1, 0.2, -0.1, 0.3).is_finite());
            rr_trainer_update(h, 1); direct.train_updates(1);
            let p = direct.snapshot();
            let mut raw = Vec::new();
            for layer in [p.input,p.hidden,p.output] { raw.extend(layer.weight); raw.extend(layer.bias); }
            assert_eq!(rr_trainer_weights(h, 0).words.as_slice(), raw);
        }
        assert_eq!(rr_trainer_free(h), 0);
    }
}
