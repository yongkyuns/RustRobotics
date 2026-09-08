//! Independent numerical oracles for EKF-SLAM (issue #4).
//!
//! Motion expectations integrate the continuous velocity equations in f64; they
//! do not reuse production arc/sinc algebra. Covariance checks differentiate the
//! independent models and retain all robot/landmark cross-covariances.

use super::*;

fn wrap64(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(actual.is_finite() && expected.is_finite(), "{context}: non-finite value");
    let bound = tolerance * expected.abs().max(1.0);
    assert!((actual - expected).abs() <= bound,
        "{context}: actual={actual}, expected={expected}, bound={bound}");
}

fn pose_close(actual: Vector3<f32>, expected: Vector3<f64>, context: &str) {
    for axis in 0..2 {
        close(f64::from(actual[axis]), expected[axis], 2e-6, context);
    }
    close(wrap64(f64::from(actual[2]) - expected[2]), 0.0, 2e-6, context);
}

fn integrate(pose: Vector3<f64>, v: f64, w: f64, dt: f64, intervals: usize) -> Vector3<f64> {
    assert_eq!(intervals % 2, 0);
    let mut displacement = Vector2::<f64>::zeros();
    // Composite Simpson integration of [v cos(theta+w*t), v sin(theta+w*t)].
    // Test turns are at most 1.5 radians. At 256 intervals the O(N^-4) error is
    // far below f32 rounding; the first test also checks 256 vs 512 intervals.
    for i in 0..=intervals {
        let weight = if i == 0 || i == intervals { 1.0 } else if i % 2 == 0 { 2.0 } else { 4.0 };
        let t = dt * i as f64 / intervals as f64;
        let angle = pose[2] + w * t;
        displacement[0] += weight * v * angle.cos();
        displacement[1] += weight * v * angle.sin();
    }
    displacement *= dt / (3.0 * intervals as f64);
    Vector3::new(pose[0] + displacement[0], pose[1] + displacement[1], wrap64(pose[2] + w * dt))
}

fn motion_cases() -> Vec<(Vector3<f32>, f32, f32, f32)> {
    let mut cases = Vec::new();
    for theta in [0.0, 0.7, -1.2, std::f32::consts::PI - 0.01] {
        for v in [0.0, 1.0, -2.0] {
            for w in [0.0, 5e-7, -5e-7, 0.99e-6, 1.01e-6, -1.01e-6, 1e-5, -1e-4, 0.02, -0.2, 1.5] {
                for dt in [0.0, 0.01, 0.1, 1.0] {
                    cases.push((Vector3::new(0.4, -0.3, theta), v, w, dt));
                }
            }
        }
    }
    cases
}

#[test]
fn motion_matches_independent_velocity_integral() {
    for (pose, v, w, dt) in motion_cases() {
        let expected = integrate(pose.map(f64::from), v.into(), w.into(), dt.into(), 256);
        let finer = integrate(pose.map(f64::from), v.into(), w.into(), dt.into(), 512);
        for axis in 0..3 {
            close(expected[axis], finer[axis], 1e-9, "quadrature refinement");
        }
        // 2e-6 mixed absolute/relative slack covers f32 arithmetic/trigonometry,
        // not the loss of displacement caused by subtracting nearby sines.
        pose_close(motion_model(&pose, v, w, dt), expected, "motion integral");
    }
}

fn numeric_motion_jacobian(pose: Vector3<f32>, v: f32, w: f32, dt: f32) -> Matrix3<f64> {
    let pose = pose.map(f64::from);
    let mut jacobian = Matrix3::zeros();
    // h=1e-5 balances f64 cancellation against O(h^2) central-difference error.
    // Angular differences are on S1, including when outputs straddle +/-pi.
    let h = 1e-5;
    for col in 0..3 {
        let mut plus = pose;
        let mut minus = pose;
        plus[col] += h;
        minus[col] -= h;
        let a = integrate(plus, v.into(), w.into(), dt.into(), 256);
        let b = integrate(minus, v.into(), w.into(), dt.into(), 256);
        for row in 0..3 {
            let difference = if row == 2 { wrap64(a[row] - b[row]) } else { a[row] - b[row] };
            jacobian[(row, col)] = difference / (2.0 * h);
        }
    }
    jacobian
}

#[test]
fn motion_jacobian_matches_independent_finite_differences() {
    for (pose, v, w, dt) in motion_cases() {
        let expected = numeric_motion_jacobian(pose, v, w, dt);
        let actual = motion_jacobian(&pose, v, w, dt);
        for row in 0..3 {
            for col in 0..3 {
                close(f64::from(actual[(row, col)]), expected[(row, col)], 2e-6, "motion Jacobian");
            }
        }
    }
}

#[test]
fn tiny_angular_rate_preserves_accumulated_turn() {
    // Small rate is not the same as small total turn. The old |w| threshold
    // suppressed both heading and curvature for this finite, well-scaled step.
    let pose = Vector3::new(0.0, 0.0, 0.7);
    let (v, w, dt) = (1e-6_f32, 5e-7_f32, 1e6_f32);
    let expected = integrate(pose.map(f64::from), v.into(), w.into(), dt.into(), 256);
    pose_close(motion_model(&pose, v, w, dt), expected, "accumulated turn");
}

#[test]
fn motion_composes_and_reverses_across_small_turns() {
    for w in [0.0, 0.99e-6, 1.01e-6, -1.01e-6, 0.02, -0.2, 1.5] {
        let pose = Vector3::new(0.4, -0.3, 0.7);
        let full = motion_model(&pose, 1.0, w, 0.1);
        let half = motion_model(&pose, 1.0, w, 0.05);
        let composed = motion_model(&half, 1.0, w, 0.05);
        pose_close(composed, full.map(f64::from), "two half steps");
        let reversed = motion_model(&full, -1.0, -w, 0.1);
        pose_close(reversed, pose.map(f64::from), "inverse motion");
    }
}

fn joint_state() -> EkfSlamState {
    let mu = DVector::from_vec(vec![0.4, -0.3, 0.7, 2.3, 1.4, -1.5, 2.1]);
    // Deterministic, non-diagonal SPD covariance with robot/landmark coupling.
    let basis = DMatrix::from_fn(7, 7, |row, col| ((row * 7 + col * 3) % 11) as f32 * 0.02 - 0.1);
    let sigma = &basis * basis.transpose() + DMatrix::identity(7, 7) * 0.05;
    EkfSlamState { mu, sigma, n_landmarks: 2 }
}

#[test]
fn prediction_covariance_matches_independent_linearization() {
    for w in [0.0, 1.01e-6, -1.01e-6, 0.2] {
        let mut state = joint_state();
        let before = state.clone();
        let config = EkfSlamConfig::default();
        let (v, dt) = (1.0_f32, 0.1_f32);
        let g = numeric_motion_jacobian(before.robot_pose(), v, w, dt);
        let mut full_g = DMatrix::<f64>::identity(7, 7);
        full_g.slice_mut((0, 0), (3, 3)).copy_from(&g);
        let mut expected = &full_g * before.sigma.map(f64::from) * full_g.transpose();
        // Preserve the existing configured noise law; this PR does not change
        // process-noise tuning or how its diagonal entries are scaled.
        let distance = (f64::from(v) * f64::from(dt)).abs();
        let angle = (f64::from(w) * f64::from(dt)).abs();
        expected[(0, 0)] += 0.002 * f64::from(dt) + f64::from(config.motion_noise[(0, 0)]) * distance;
        expected[(1, 1)] += 0.002 * f64::from(dt) + f64::from(config.motion_noise[(1, 1)]) * distance;
        expected[(2, 2)] += 0.005 * f64::from(dt) + f64::from(config.motion_noise[(2, 2)]) * (angle + 0.02);
        predict(&mut state, &config, v, w, dt);
        for row in 0..7 {
            for col in 0..7 {
                close(f64::from(state.sigma[(row, col)]), expected[(row, col)], 2e-6, "prediction covariance");
            }
        }
        for index in 3..7 {
            assert_eq!(state.mu[index], before.mu[index], "prediction changed a landmark mean");
        }
    }
}

fn observe64(input: &[f64; 5]) -> Vector2<f64> {
    let dx = input[3] - input[0];
    let dy = input[4] - input[1];
    Vector2::new(dx.hypot(dy), wrap64(dy.atan2(dx) - input[2]))
}

#[test]
fn observation_jacobians_match_independent_finite_differences() {
    for (dx, dy) in [(2.0_f32, 1.5_f32), (-2.0, 1.5), (-2.0, -1.5), (0.2, -0.1), (20.0, 15.0)] {
        for theta in [0.0, 0.7, -2.9, std::f32::consts::PI] {
            let pose = Vector3::new(0.4, -0.3, theta);
            let landmark = Vector2::new(pose[0] + dx, pose[1] + dy);
            let input = [f64::from(pose[0]), f64::from(pose[1]), f64::from(theta), f64::from(landmark[0]), f64::from(landmark[1])];
            let expected = observe64(&input);
            let actual = observation_model(&pose, &landmark);
            close(f64::from(actual.range), expected[0], 2e-6, "observation range");
            close(wrap64(f64::from(actual.bearing) - expected[1]), 0.0, 2e-6, "observation bearing");
            let (robot, lm) = observation_jacobian(&pose, &landmark);
            let h = 1e-5;
            for col in 0..5 {
                let mut plus = input;
                let mut minus = input;
                plus[col] += h;
                minus[col] -= h;
                let a = observe64(&plus);
                let b = observe64(&minus);
                for row in 0..2 {
                    let difference = if row == 1 { wrap64(a[row] - b[row]) } else { a[row] - b[row] };
                    let actual = if col < 3 { robot[(row, col)] } else { lm[(row, col - 3)] };
                    close(f64::from(actual), difference / (2.0 * h), 3e-6, "observation Jacobian");
                }
            }
        }
    }
}

fn initialize64(input: &[f64; 5]) -> Vector2<f64> {
    Vector2::new(input[0] + input[3] * (input[2] + input[4]).cos(), input[1] + input[3] * (input[2] + input[4]).sin())
}

#[test]
fn landmark_augmentation_covariance_matches_finite_differences() {
    let mut state = joint_state();
    let before = state.clone();
    let mut config = EkfSlamConfig::default();
    config.observation_noise = Matrix2::new(0.04, 0.003, 0.003, 0.0025);
    let obs = Observation { range: 3.0, bearing: -0.8 };
    let input = [f64::from(state.mu[0]), f64::from(state.mu[1]), f64::from(state.mu[2]), f64::from(obs.range), f64::from(obs.bearing)];
    let mut derivative = DMatrix::<f64>::zeros(2, 5);
    let h = 1e-5;
    for col in 0..5 {
        let mut plus = input;
        let mut minus = input;
        plus[col] += h;
        minus[col] -= h;
        derivative.set_column(col, &((initialize64(&plus) - initialize64(&minus)) / (2.0 * h)));
    }
    let mut state_jacobian = DMatrix::<f64>::zeros(2, 7);
    state_jacobian.slice_mut((0, 0), (2, 3)).copy_from(&derivative.columns(0, 3));
    let noise_jacobian = derivative.columns(3, 2);
    let old_sigma = before.sigma.map(f64::from);
    let cross = &state_jacobian * &old_sigma;
    let new_covariance = &cross * state_jacobian.transpose()
        + noise_jacobian * config.observation_noise.map(f64::from) * noise_jacobian.transpose();
    let mut expected = DMatrix::<f64>::zeros(9, 9);
    expected.slice_mut((0, 0), (7, 7)).copy_from(&old_sigma);
    expected.slice_mut((7, 0), (2, 7)).copy_from(&cross);
    expected.slice_mut((0, 7), (7, 2)).copy_from(&cross.transpose());
    expected.slice_mut((7, 7), (2, 2)).copy_from(&new_covariance);
    add_landmark(&mut state, &obs, &config);
    assert_eq!(state.n_landmarks, 3);
    assert_eq!(state.sigma.shape(), (9, 9));
    let expected_mean = initialize64(&input);
    for axis in 0..2 {
        close(f64::from(state.mu[7 + axis]), expected_mean[axis], 2e-6, "augmented mean");
    }
    for row in 0..9 {
        for col in 0..9 {
            close(f64::from(state.sigma[(row, col)]), expected[(row, col)], 2e-6, "augmented covariance");
        }
    }
}

#[test]
fn motion_at_zero_time_is_identity_and_wraps_heading() {
    let pose = Vector3::new(0.4, -0.3, 0.7);
    for w in [0.0, 1.01e-6, 0.2] {
        assert_eq!(motion_model(&pose, 1.0, w, 0.0), pose);
        assert_eq!(motion_jacobian(&pose, 1.0, w, 0.0), Matrix3::identity());
    }
    let pose = Vector3::new(0.0, 0.0, std::f32::consts::PI - 0.01);
    let expected = integrate(pose.map(f64::from), 0.0, 0.2, 0.1, 256);
    pose_close(motion_model(&pose, 0.0, 0.2, 0.1), expected, "heading wrap");
}
