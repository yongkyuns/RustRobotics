//! Checks the RNG seam without replacing production observation or step logic.
use super::*;
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn observations_equal(a: &[(usize, Observation)], b: &[(usize, Observation)]) {
    assert_eq!(a.len(), b.len());
    for ((id_a, a), (id_b, b)) in a.iter().zip(b) {
        assert_eq!(id_a, id_b);
        assert_eq!(a.range.to_bits(), b.range.to_bits());
        assert_eq!(a.bearing.to_bits(), b.bearing.to_bits());
    }
}

#[test]
fn seeded_observations_preserve_noise_formula_and_draw_order() {
    let mut rng = StdRng::seed_from_u64(0xe4f1_0001);
    let mut reference = rng.clone();
    let pose = Vector3::zeros();
    let landmarks = [
        Vector2::new(3.0, 4.0),
        Vector2::new(100.0, 100.0),
        Vector2::new(-4.0, 3.0),
    ];
    let config = EkfSlamConfig::default();
    let observations = generate_observations_with_rng(&pose, &landmarks, &config, true, &mut rng);
    assert_eq!(observations.len(), 2);
    for ((id, obs), expected_id) in observations.iter().zip([0, 2]) {
        assert_eq!(*id, expected_id);
        // Preserve the existing uniform noise law and two draws per visible
        // landmark. This is deliberately not a Gaussian-noise retuning.
        let range = 5.0 + rand(&mut reference) * config.observation_noise[(0, 0)].sqrt() * 0.5;
        let bearing = normalize_angle(
            landmarks[*id][1].atan2(landmarks[*id][0])
                + rand(&mut reference) * config.observation_noise[(1, 1)].sqrt() * 0.5,
        );
        assert_eq!(
            obs.range.to_bits(),
            range.to_bits(),
            "seeded observation range"
        );
        assert_eq!(
            obs.bearing.to_bits(),
            bearing.to_bits(),
            "seeded observation bearing"
        );
    }
    assert_eq!(
        rng.next_u64(),
        reference.next_u64(),
        "invisible landmarks must not consume noise draws"
    );
}

#[test]
fn noise_free_observations_match_public_api_without_consuming_rng() {
    let mut rng = StdRng::seed_from_u64(0xe4f1_0002);
    let mut untouched = rng.clone();
    let config = EkfSlamConfig::default();
    let pose = Vector3::new(1.0, 2.0, 0.3);
    let landmarks = [Vector2::new(4.0, 6.0)];
    let seeded = generate_observations_with_rng(&pose, &landmarks, &config, false, &mut rng);
    let public = generate_observations(&pose, &landmarks, &config, false);
    observations_equal(&seeded, &public);
    assert_eq!(rng.next_u64(), untouched.next_u64());
}

#[test]
fn seeded_step_replays_the_explicit_prediction_observation_update_sequence() {
    let mut rng_a = StdRng::seed_from_u64(0xe4f1_0003);
    let mut rng_b = rng_a.clone();
    let config = EkfSlamConfig::default();
    let landmarks = [Vector2::new(5.0, 5.0), Vector2::new(10.0, 0.0)];
    let (mut a, mut b) = (EkfSlamState::new(), EkfSlamState::new());
    let mut truth = Vector3::zeros();
    for _ in 0..50 {
        truth = motion_model(&truth, 1.0, 0.1, 0.1);
        step_with_rng(
            &mut a,
            &config,
            &truth,
            &landmarks,
            (1.0, 0.1, 0.1),
            &mut rng_a,
        );
        predict(&mut b, &config, 1.0, 0.1, 0.1);
        let observations =
            generate_observations_with_rng(&truth, &landmarks, &config, true, &mut rng_b);
        update(&mut b, &config, &observations);
        assert_eq!(a.mu, b.mu);
        assert_eq!(a.sigma, b.sigma);
        assert_eq!(a.n_landmarks, b.n_landmarks);
    }
    assert!(a.n_landmarks > 0);
    assert_eq!(rng_a.next_u64(), rng_b.next_u64());
}
