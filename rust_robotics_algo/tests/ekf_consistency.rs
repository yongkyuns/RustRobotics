//! Seeded, calibrated EKF-SLAM ensemble regressions (issue #4).
//!
//! Ground truth uses f64 quadrature and independent process/observation draws.
//! Statistics pool independent trials at ONE terminal time, never correlated
//! timesteps. No observations/trials are discarded from the statistics. The
//! production association and innovation gates remain enabled.
//!
//! Bounds come from Gaussian chi-square concentration, not observed outputs.
//! Nonlinear EKF/gating makes them regression envelopes, not a formal guarantee
//! of consistency for arbitrary noise, trajectories, or the default simulator.

use nalgebra::{DMatrix, DVector};
use rust_robotics_algo::slam::{predict, update, EkfSlamConfig, EkfSlamState, Observation};

const TRIALS: usize = 1024;
const STEPS: usize = 12;
const FAMILY_CHECKS: f64 = 16.0;
const ALPHA: f64 = 1e-6;

struct Noise {
    state: u64,
    spare: Option<f64>,
}

impl Noise {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn word(&mut self) -> u64 {
        // SplitMix64: explicit wrapping arithmetic fixes the stream independently
        // of the rand crate, pointer width, and randomized hash iteration.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        // Open interval (0,1), with exactly representable 52-bit midpoints.
        ((self.word() >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        let (s, c) = angle.sin_cos();
        self.spare = Some(radius * s);
        radius * c
    }

    fn vector(&mut self, n: usize) -> DVector<f64> {
        DVector::from_iterator(n, (0..n).map(|_| self.normal()))
    }
}

fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

fn initial_state() -> EkfSlamState {
    let mu = DVector::from_vec(vec![0.4_f32, -0.3, 0.25, 5.0, 2.0, -3.0, 4.5]);
    // Shared global translation/rotation uncertainty plus independent local
    // uncertainty. This is a proper anchored joint Gaussian, not a singular
    // gauge-free prior; its cross-covariances are essential to the regression.
    let mut gauge = DMatrix::<f64>::zeros(7, 3);
    for row in [0, 3, 5] {
        gauge[(row, 0)] = 1.0;
        gauge[(row + 1, 1)] = 1.0;
        gauge[(row, 2)] = -f64::from(mu[row + 1]);
        gauge[(row + 1, 2)] = f64::from(mu[row]);
    }
    gauge[(2, 2)] = 1.0;
    let global = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01, 0.0009]));
    let local = DMatrix::from_diagonal(&DVector::from_vec(vec![
        0.002, 0.002, 0.0001, 0.0025, 0.0025, 0.0025, 0.0025,
    ]));
    let sigma = (&gauge * global * gauge.transpose() + local).map(|x| x as f32);
    EkfSlamState {
        mu,
        sigma,
        n_landmarks: 2,
    }
}

fn config() -> EkfSlamConfig {
    EkfSlamConfig {
        motion_noise: nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(
            0.002, 0.003, 0.001,
        )),
        // Correlation 0.55 between range and bearing; independent across sensors
        // and time. This noise is sampled, not silently diagonalized or clipped.
        observation_noise: nalgebra::Matrix2::new(0.01, 0.0011, 0.0011, 0.0004),
        ..EkfSlamConfig::default()
    }
}

fn controls(step: usize, direction: f32, turn: f32) -> (f32, f32, f32) {
    let v = direction * (0.8 + 0.1 * (step % 3) as f32);
    let w = if step < STEPS / 2 { turn } else { -turn * 0.5 };
    (v, w, 0.1)
}

fn advance_truth(
    truth: &mut DVector<f64>,
    config: &EkfSlamConfig,
    u: (f32, f32, f32),
    rng: &mut Noise,
) {
    let (v, w, dt) = (f64::from(u.0), f64::from(u.1), f64::from(u.2));
    // Composite Simpson integration of x_dot=v*cos(theta), y_dot=v*sin(theta).
    // |w*dt| <= 0.016 in these fixtures; 16 intervals make quadrature error
    // negligible relative to f32. No production motion helper is called.
    let mut displacement = [0.0; 2];
    for i in 0..=16 {
        let weight = if i == 0 || i == 16 {
            1.0
        } else if i & 1 == 0 {
            2.0
        } else {
            4.0
        };
        let (s, c) = (truth[2] + w * dt * i as f64 / 16.0).sin_cos();
        displacement[0] += weight * c;
        displacement[1] += weight * s;
    }
    truth[0] += v * dt * displacement[0] / 48.0;
    truth[1] += v * dt * displacement[1] / 48.0;
    truth[2] = wrap(truth[2] + w * dt);
    // Calibrated additive world-pose noise law specified by predict(), evaluated
    // independently in f64. This tests implementation/model agreement, not the
    // real-world suitability of that law or ignored off-diagonal motion noise.
    let variance = [
        0.002 * dt + f64::from(config.motion_noise[(0, 0)]) * (v * dt).abs(),
        0.002 * dt + f64::from(config.motion_noise[(1, 1)]) * (v * dt).abs(),
        0.005 * dt + f64::from(config.motion_noise[(2, 2)]) * ((w * dt).abs() + 0.02),
    ];
    for i in 0..3 {
        truth[i] += variance[i].sqrt() * rng.normal();
    }
    truth[2] = wrap(truth[2]);
}

fn expected_observations(mu: &DVector<f64>, ids: &[usize]) -> DVector<f64> {
    let mut z = DVector::zeros(2 * ids.len());
    for (row, &id) in ids.iter().enumerate() {
        let dx = mu[3 + 2 * id] - mu[0];
        let dy = mu[4 + 2 * id] - mu[1];
        z[2 * row] = dx.hypot(dy);
        z[2 * row + 1] = wrap(dy.atan2(dx) - mu[2]);
    }
    z
}

fn quadratic(error: &DVector<f64>, covariance: &DMatrix<f64>) -> f64 {
    assert!(error.iter().chain(covariance.iter()).all(|x| x.is_finite()));
    // Check symmetry before using a factorization that reads only one triangle.
    let scale = covariance.iter().fold(1.0_f64, |s, x| s.max(x.abs()));
    assert!((covariance - covariance.transpose()).amax() <= 1e-5 * scale);
    let symmetric = (covariance + covariance.transpose()) * 0.5;
    let factor = symmetric
        .cholesky()
        .expect("covariance must be positive definite");
    let value = error.dot(&factor.solve(error));
    assert!(value.is_finite() && value >= 0.0);
    value
}

fn predictive_nis(
    state: &EkfSlamState,
    config: &EkfSlamConfig,
    observations: &[(usize, Observation)],
) -> f64 {
    let ids: Vec<_> = observations.iter().map(|(id, _)| *id).collect();
    let mean = state.mu.map(f64::from);
    let expected = expected_observations(&mean, &ids);
    let m = expected.len();
    let mut residual = DVector::zeros(m);
    let mut noise = DMatrix::zeros(m, m);
    for (i, (_, obs)) in observations.iter().enumerate() {
        residual[2 * i] = f64::from(obs.range) - expected[2 * i];
        residual[2 * i + 1] = wrap(f64::from(obs.bearing) - expected[2 * i + 1]);
        for row in 0..2 {
            for col in 0..2 {
                noise[(2 * i + row, 2 * i + col)] = f64::from(config.observation_noise[(row, col)]);
            }
        }
    }
    let mut h = DMatrix::zeros(m, mean.len());
    // Independent f64 central differences, away from zero range. Angular
    // differences are wrapped so crossing +/-pi is not a spurious derivative.
    let step = 1e-5;
    for col in 0..mean.len() {
        let mut plus = mean.clone();
        let mut minus = mean.clone();
        plus[col] += step;
        minus[col] -= step;
        let mut difference =
            expected_observations(&plus, &ids) - expected_observations(&minus, &ids);
        for row in (1..m).step_by(2) {
            difference[row] = wrap(difference[row]);
        }
        h.set_column(col, &(difference / (2.0 * step)));
    }
    let covariance = &h * state.sigma.map(f64::from) * h.transpose() + noise;
    quadratic(&residual, &covariance)
}

fn check_average(total: f64, dof: usize, trials: usize, name: &str) {
    // For ideal independent Gaussian trials, summed quadratic errors have
    // nu=N*d degrees of freedom. Laurent-Massart bounds and a union bound over
    // <=16 comparisons give this envelope with nominal family alpha=1e-6.
    // Nonlinearity and gates mean no exact false-alarm probability is claimed.
    let t = (2.0 * FAMILY_CHECKS / ALPHA).ln();
    let nu = (trials * dof) as f64;
    let width = 2.0 * (nu * t).sqrt();
    let lower = (nu - width).max(0.0) / trials as f64;
    let upper = (nu + width + 2.0 * t) / trials as f64;
    let average = total / trials as f64;
    println!("{name}: N={trials}, dof={dof}, mean={average:.8}, bounds=[{lower:.8}, {upper:.8}]");
    assert!(
        average.is_finite() && average >= lower && average <= upper,
        "{name}: consistency envelope: mean={average}, bounds=[{lower}, {upper}]"
    );
}

fn ensemble(observation_count: usize, direction: f32, turn: f32, seed: u64) {
    let config = config();
    let prior = initial_state();
    let prior_mean = prior.mu.map(f64::from);
    let prior_factor = prior.sigma.map(f64::from).cholesky().unwrap().l();
    let observation_factor = config
        .observation_noise
        .map(f64::from)
        .cholesky()
        .unwrap()
        .l();
    let (mut joint_nees, mut robot_nees, mut nis) = (0.0, 0.0, 0.0);
    for trial in 0..TRIALS {
        let mut rng = Noise::new(seed ^ trial as u64);
        let mut truth = &prior_mean + &prior_factor * rng.vector(7);
        let mut state = prior.clone();
        for step in 0..STEPS {
            let u = controls(step, direction, turn);
            advance_truth(&mut truth, &config, u, &mut rng);
            predict(&mut state, &config, u.0, u.1, u.2);
            if observation_count == 0 {
                continue;
            }
            let ids = if observation_count == 1 {
                vec![step % 2]
            } else {
                vec![0, 1]
            };
            let expected = expected_observations(&truth, &ids);
            let mut observations = Vec::new();
            for (i, &id) in ids.iter().enumerate() {
                let noise = observation_factor * nalgebra::Vector2::new(rng.normal(), rng.normal());
                observations.push((
                    id,
                    Observation {
                        range: (expected[2 * i] + noise[0]) as f32,
                        bearing: wrap(expected[2 * i + 1] + noise[1]) as f32,
                    },
                ));
            }
            if step == STEPS - 1 {
                // BEFORE public update/association/gating; no tail truncation.
                nis += predictive_nis(&state, &config, &observations);
            }
            update(&mut state, &config, &observations);
            assert_eq!(
                state.n_landmarks, 2,
                "well-separated existing map must not grow"
            );
            // Keep this fixture above the production variance floors so floor
            // inflation cannot silently change the intended calibration regime.
            for (i, floor) in [0.001_f32, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
                .iter()
                .enumerate()
            {
                assert!(
                    state.sigma[(i, i)] > *floor,
                    "fixture reached a covariance floor"
                );
            }
        }
        let mut error = &truth - state.mu.map(f64::from);
        error[2] = wrap(error[2]);
        let covariance = state.sigma.map(f64::from);
        joint_nees += quadratic(&error, &covariance);
        robot_nees += quadratic(
            &error.rows(0, 3).into_owned(),
            &covariance.slice((0, 0), (3, 3)).into_owned(),
        );
    }
    check_average(robot_nees, 3, TRIALS, "terminal robot NEES");
    if observation_count > 0 {
        check_average(joint_nees, 7, TRIALS, "terminal joint NEES");
        check_average(nis, 2 * observation_count, TRIALS, "pre-update NIS");
    }
}

#[test]
fn seeded_prediction_covariance_matches_errors() {
    for (index, (direction, turn)) in [(1.0, 0.0), (1.0, 0.16), (-1.0, -0.12)]
        .into_iter()
        .enumerate()
    {
        ensemble(
            0,
            direction,
            turn,
            0x5820_6701_1000_0000 + ((index as u64) << 32),
        );
    }
}

#[test]
fn seeded_single_observation_consistency() {
    ensemble(1, 1.0, 0.12, 0x82a1_1050_0000_0000);
}

#[test]
fn seeded_batch_observation_consistency() {
    ensemble(2, 1.0, 0.12, 0x65d2_4200_0000_0000);
}

#[test]
fn seeded_noise_reference_is_reproducible_and_calibrated() {
    let mut a = Noise::new(0);
    let mut b = Noise::new(0);
    // Pin the integer stream; floating-point transcendental results need not be
    // bit-identical across platforms. The stochastic tests use error envelopes.
    assert_eq!(a.word(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(b.word(), 0xe220_a839_7b1d_cdaf);
    let n = 32_768;
    let (mut sum, mut squares) = (0.0, 0.0);
    for _ in 0..n {
        let value = a.normal();
        assert_eq!(value, b.normal());
        assert!(value.is_finite());
        sum += value;
        squares += value * value;
    }
    check_average(squares, 1, n, "reference normal second moment");
    // Gaussian sample mean has variance 1/N; two-sided normal Chernoff bound.
    let bound = (2.0 * (2.0 * FAMILY_CHECKS / ALPHA).ln() / n as f64).sqrt();
    assert!((sum / n as f64).abs() < bound, "reference normal mean");
}
