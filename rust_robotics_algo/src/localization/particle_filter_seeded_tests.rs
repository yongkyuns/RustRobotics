//! Reproducible coverage of every particle-filter random draw path.
use super::*;
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn check_distribution(px: &PX, pw: &PW) {
    assert!(px.iter().all(|x| x.is_finite()), "particle state must be finite");
    assert!(pw.iter().all(|w| w.is_finite() && *w >= 0.0));
    // Normalizing and summing NP f32 weights introduces O(NP * epsilon) error.
    assert!((pw.sum() - 1.0).abs() <= 4.0 * NP as f32 * f32::EPSILON);
}

#[test]
fn random_sources_are_seeded_and_replayable() {
    let mut a = StdRng::seed_from_u64(0x5046_0001);
    let mut b = StdRng::seed_from_u64(0x5046_0001);
    let mut c = StdRng::seed_from_u64(0x5046_0002);
    let mut changed = false;
    for _ in 0..128 {
        let x = rand_with_rng(&mut a);
        assert_eq!(x.to_bits(), rand_with_rng(&mut b).to_bits());
        assert!((-1.0..1.0).contains(&x));
        changed |= x.to_bits() != rand_with_rng(&mut c).to_bits();
        let u = uniform_with_rng(0.0, 0.01, &mut a);
        assert_eq!(u.to_bits(), uniform_with_rng(0.0, 0.01, &mut b).to_bits());
        assert!((0.0..0.01).contains(&u));
        let _ = uniform_with_rng(0.0, 0.01, &mut c);
    }
    assert!(changed, "changing the seed must change the noise stream");
}

#[test]
fn observation_entry_points_consume_the_same_seeded_stream() {
    let mut a = StdRng::seed_from_u64(0x5046_0003);
    let mut b = a.clone();
    let mut truth = Vector4::zeros();
    let mut dr_a = Vector4::zeros();
    let mut dr_b = dr_a;
    let landmarks = [Vector2::new(3.0, 4.0), Vector2::new(200.0, 0.0)];
    let u = calc_input();
    let (z_a, u_a) = observation_with_rng(&mut truth, &mut dr_a, u, &landmarks, (0.1, MAX_RANGE), &mut a);
    let (z_b, u_b) = observation_from_state_with_rng(&truth, &mut dr_b, u, &landmarks, (0.1, MAX_RANGE), &mut b);
    assert_eq!(z_a.len(), 1);
    assert_eq!(z_a, z_b);
    assert_eq!(u_a, u_b);
    assert_eq!(dr_a, dr_b);
    assert_eq!(a.next_u64(), b.next_u64(), "observation draw counts must agree");
}

fn rollout(seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let landmarks = [Vector2::new(10.0, 0.0), Vector2::new(10.0, 10.0), Vector2::new(0.0, 15.0), Vector2::new(-5.0, 20.0)];
    let (mut estimated, mut truth, mut dr) = (Vector4::zeros(), Vector4::zeros(), Vector4::zeros());
    let mut px = PX::zeros();
    let mut pw = PW::repeat(1.0 / NP as f32);
    let mut history = Vec::new();
    // Retains the old 50-second smoke rollout; now every step has assertions.
    let mut time = 0.0_f32;
    while time < 50.0 {
        time += 0.1;
        let (z, u) = observation_with_rng(&mut truth, &mut dr, calc_input(), &landmarks, (0.1, MAX_RANGE), &mut rng);
        // The public one-shot entry point also creates default recovery state
        // per invocation. Do not silently change that behavior in this fixture.
        let covariance = pf_localization_with_rng(&mut estimated, &mut px, &mut pw, z, (u, 0.1), (&mut PFState::default(), &PFNoiseParams::default()), &mut rng);
        check_distribution(&px, &pw);
        assert!(estimated.iter().chain(truth.iter()).chain(dr.iter()).chain(covariance.iter()).all(|x| x.is_finite()));
        let scale = covariance.amax().max(1.0);
        assert!((covariance - covariance.transpose()).amax() <= 2e-5 * scale);
        assert!(covariance.symmetric_eigen().eigenvalues.min() >= -2e-5 * scale);
        history.extend(estimated.iter().chain(px.iter()).chain(pw.iter()).chain(covariance.iter()).map(|x| x.to_bits()));
    }
    history
}

#[test]
fn complete_rollout_replays_every_particle_weight_and_covariance() {
    let first = rollout(0x5046_0004);
    let other = rollout(0x5046_0005);
    let replay = rollout(0x5046_0004);
    assert_eq!(first, replay, "same seed must replay the entire filter rollout");
    assert_ne!(first, other, "rollout must actually depend on seeded draws");
}

#[test]
fn recovery_uses_the_callers_rng_for_every_reset_draw() {
    let mut rng = StdRng::seed_from_u64(0x5046_0006);
    let mut reference_rng = rng.clone();
    let mut px = PX::zeros();
    let mut pw = PW::zeros(); // Forces the existing weight-collapse recovery path.
    let mut estimate = Vector4::new(1.0, 2.0, 0.2, 0.5);
    let before = estimate;
    let z = vec![Vector3::new(4.0, 2.0, -1.0)];
    let mut state = PFState { no_obs_count: 12, recovery_count: 0 };
    let covariance = pf_localization_with_rng(&mut estimate, &mut px, &mut pw, z.clone(), (calc_input(), 0.1), (&mut state, &PFNoiseParams::default()), &mut rng);
    // Each of NP particles consumes two motion draws before recovery. The
    // reference then invokes the reset with exactly the remaining caller stream.
    for _ in 0..2 * NP { let _ = rand_with_rng(&mut reference_rng); }
    let mut expected = PX::zeros();
    let mut expected_weights = PW::zeros();
    reset_particles_with_rng(&mut expected, &mut expected_weights, &z, &before, &mut reference_rng);
    assert_eq!(px, expected, "recovery must use the injected random stream");
    assert_eq!(pw, expected_weights);
    assert_eq!(rng.next_u64(), reference_rng.next_u64(), "all recovery draws must be accounted for");
    assert_eq!(state.no_obs_count, 0);
    assert_eq!(state.recovery_count, 1);
    assert!(covariance.iter().all(|x| x.is_finite()));
    check_distribution(&px, &pw);
    for particle in px.column_iter() {
        let radius = ((particle[0] - 2.0).powi(2) + (particle[1] + 1.0).powi(2)).sqrt();
        assert!((radius - 4.0).abs() <= RESET_SPREAD + 1e-5);
        assert!((particle[2] - before[2]).abs() <= 0.3 + 1e-6);
        assert_eq!(particle[3], before[3]);
    }
}

#[test]
fn resampling_replays_and_selects_only_supported_particles() {
    let original = PX::from_fn(|row, col| (row * NP + col) as f32);
    let mut weights = PW::from_fn(|_, col| if col.is_multiple_of(3) { 1.0 } else { 0.0 });
    weights /= weights.sum();
    let (mut a, mut b) = (original, original);
    let (mut wa, mut wb) = (weights, weights);
    let mut rng_a = StdRng::seed_from_u64(0x5046_0007);
    let mut rng_b = rng_a.clone();
    re_sampling_with_rng(&mut a, &mut wa, &mut rng_a);
    re_sampling_with_rng(&mut b, &mut wb, &mut rng_b);
    assert_eq!(a, b, "resampling must use the injected random stream");
    assert_eq!(wa, wb);
    assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    check_distribution(&a, &wa);
    for column in a.column_iter() {
        let index = column[0] as usize;
        assert!(index < NP && index.is_multiple_of(3));
        assert_eq!(column, original.column(index));
    }
    assert!(wa.iter().all(|w| *w == 1.0 / NP as f32));
}
