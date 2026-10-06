// Appended to the archived audit bridge only; production sources stay unchanged.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn rr_credit_from_state(
    x: f32, v: f32, a: f32, w: f32,
    ox: f32, ov: f32, oa: f32, ow: f32,
    seed: u64, horizon: u32,
) -> u64 {
    if horizon == 0 || ![x, v, a, w, ox, ov, oa, ow].iter().all(|q| q.is_finite()) {
        return 0;
    }
    let state = rust_robotics_algo::Vector4::from_columns([[x, v, a, w]]);
    let env = PendulumEnv::from_state(Default::default(), PendulumEnvConfig {
        max_steps: horizon as usize, ..Default::default()
    }, state, 0);
    let last = output(&env, [ox, ov, oa, ow]);
    ENVS.with(|slots| {
        let mut slots = slots.borrow_mut();
        slots.push(Some(Environment { env, rng: StdRng::seed_from_u64(seed), first_reset: false, last }));
        slots.len() as u64
    })
}

#[no_mangle]
pub extern "C" fn rr_credit_peek(handle: u64) -> CStep {
    ENVS.with(|slots| match slots.borrow().get(handle.wrapping_sub(1) as usize) {
        Some(Some(e)) => e.last,
        _ => bad_step(),
    })
}

#[no_mangle]
pub extern "C" fn rr_credit_clone(handle: u64) -> u64 {
    ENVS.with(|slots| {
        let mut slots = slots.borrow_mut();
        let Some(Some(e)) = slots.get(handle.wrapping_sub(1) as usize) else { return 0; };
        let copy = Environment { env: e.env.clone(), rng: e.rng.clone(), first_reset: e.first_reset, last: e.last };
        slots.push(Some(copy));
        slots.len() as u64
    })
}

#[no_mangle]
pub extern "C" fn rr_credit_reseed(handle: u64, seed: u64) -> u32 {
    ENVS.with(|slots| {
        let mut slots = slots.borrow_mut();
        let Some(Some(e)) = slots.get_mut(handle.wrapping_sub(1) as usize) else { return 1; };
        e.rng = StdRng::seed_from_u64(seed);
        0
    })
}
