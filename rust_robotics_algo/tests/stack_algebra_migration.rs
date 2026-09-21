//! Independent parity checks for the fixed-size backend migration.
use nalgebra as na;
use rust_robotics_algo::{
    control::{StateSpace, LQR},
    diag, eye,
    inverted_pendulum::Model,
    localization::particle_filter::{self as pf, NP, PW, PX},
    matrix, ones, vector, zeros, Mat, Vector2, Vector4,
};

#[test]
fn literal_shapes_and_layout_keep_the_robotics_contract() {
    const STATE: Vector4 = vector![1.0, 2.0, 3.0, 4.0];
    let row: Mat<1, 4> = ones!(4);
    let empty_row: Mat<1, 4> = zeros!(4);
    let scalar: Mat<1, 1> = diag![0.01];
    let rectangular: Mat<2, 3> = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
    let zero_f64: Mat<2, 3, f64> = zeros!(2, 3, f64);
    assert_eq!(STATE.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!((row * STATE)[0], 10.0);
    assert_eq!(empty_row.as_slice(), &[0.0; 4]);
    assert_eq!(scalar[(0, 0)], 0.01);
    assert_eq!(rectangular.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(zero_f64.as_slice(), &[0.0; 6]);
    assert_eq!(eye!(4) * STATE, STATE);
    let _: stack_algebra::Matrix<4, 1, f32> = STATE;
}

#[test]
fn particle_motion_matches_the_previous_matrix_formulation() {
    for heading in [-3.0_f32, -0.1, 0.0, 0.7, 3.0] {
        for dt in [0.0_f32, 0.01, 0.1] {
            let state = vector![2.0, -1.0, heading, 4.0];
            let input: Vector2 = vector![3.0, -0.2];
            let f = na::Matrix4::from_diagonal(&na::Vector4::new(1.0, 1.0, 1.0, 0.0));
            let b = na::SMatrix::<f32, 4, 2>::from_row_slice(&[
                heading.cos() * dt,
                0.0,
                heading.sin() * dt,
                0.0,
                0.0,
                dt,
                1.0,
                0.0,
            ]);
            let expected = f * na::Vector4::from_column_slice(state.as_slice())
                + b * na::Vector2::from_column_slice(input.as_slice());
            let actual = pf::motion_model(state, input, dt);
            for (a, b) in actual.iter().zip(expected.iter()) {
                assert!((a - b).abs() <= 2.0e-6 * (1.0 + b.abs()));
            }
        }
    }
}

#[test]
fn particle_covariance_matches_independent_nalgebra_reference() {
    let particles =
        PX::from_fn(|row, column| (column as f32 * 0.07 + row as f32 * 0.3).sin() + row as f32);
    let mut weights = PW::from_fn(|_, column| (column + 1) as f32);
    weights /= weights.iter().sum::<f32>();
    let estimate = vector![0.1, -0.2, 0.4, 1.0];
    let actual = pf::calc_covariance(&estimate, &particles, &weights);
    let old_particles = na::SMatrix::<f32, 4, NP>::from_column_slice(particles.as_slice());
    let old_weights = na::SMatrix::<f32, 1, NP>::from_column_slice(weights.as_slice());
    let old_estimate = na::Vector4::from_column_slice(estimate.as_slice());
    let mut expected = na::Matrix3::<f32>::zeros();
    for index in 0..NP {
        let delta = old_particles.column(index) - old_estimate;
        let delta = delta.fixed_rows::<3>(0);
        expected += old_weights[index] * (delta * delta.transpose());
    }
    expected /= 1.0 - (old_weights * old_weights.transpose())[0];
    for (a, b) in actual.iter().zip(expected.iter()) {
        assert!((a - b).abs() <= 2.0e-5 * (1.0 + b.abs()));
    }
}

fn reference_gain(model: Model, dt: f32) -> na::SMatrix<f32, 1, 4> {
    let (a, b) = model.model(dt);
    let a = na::Matrix4::from_column_slice(a.as_slice());
    let b = na::Vector4::from_column_slice(b.as_slice());
    let q = na::Matrix4::from_column_slice(model.Q.as_slice());
    let r = na::Matrix1::from_column_slice(model.R.as_slice());
    let mut p = q;
    for _ in 0..model.max_iter {
        let inverse = (r + b.transpose() * p * b)
            .pseudo_inverse(model.eps)
            .unwrap();
        let next =
            a.transpose() * p * a - a.transpose() * p * b * inverse * b.transpose() * p * a + q;
        let converged = (next - p).abs().amax() < model.eps;
        p = next;
        if converged {
            break;
        }
    }
    (b.transpose() * p * b + r)
        .pseudo_inverse(model.eps)
        .unwrap()
        * (b.transpose() * p * a)
}

#[test]
fn prepared_pendulum_lqr_matches_historical_absolute_cutoff() {
    for model in [
        Model::default(),
        Model {
            m_cart: 2.0,
            m_ball: 0.7,
            l_bar: 1.2,
            ..Model::default()
        },
    ] {
        for dt in [0.01, 0.05, 0.1] {
            let prepared = model.prepare(dt);
            let expected = reference_gain(model, dt);
            for (a, b) in prepared.gain().iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() <= 5.0e-4 * (1.0 + b.abs()),
                    "dt={dt}, actual={a}, reference={b}"
                );
            }
            let state = vector![0.2, -0.1, 0.05, 0.3];
            let actual = prepared.control(state)[0];
            let expected = (-expected * na::Vector4::from_column_slice(state.as_slice()))[0];
            assert!((actual - expected).abs() <= 5.0e-4 * (1.0 + expected.abs()));
        }
    }
}

struct ScaledTwoInput;
impl StateSpace<2, 2> for ScaledTwoInput {
    fn model(&self, _dt: f32) -> (Mat<2, 2>, Mat<2, 2>) {
        (eye!(2) * 0.5, eye!(2))
    }
}
impl LQR<2, 2> for ScaledTwoInput {
    fn Q(&self) -> Mat<2, 2> {
        diag![1.0e6, 1.0e-3]
    }
    fn R(&self) -> Mat<2, 2> {
        diag![1.0e6, 1.0e-3]
    }
    fn epsilon(&self) -> f32 {
        1.0e-6
    }
    fn max_iter(&self) -> u32 {
        200
    }
}

#[test]
fn generic_lqr_does_not_discard_the_small_input_direction() {
    let gain = ScaledTwoInput.prepare(0.1);
    // Independent scalar DARE: p^2 - a^2*p - 1 = 0, k = a*p/(1+p).
    let p = (0.25_f64 + (0.25_f64.powi(2) + 4.0).sqrt()) / 2.0;
    let expected = (0.5 * p / (1.0 + p)) as f32;
    for index in 0..2 {
        assert!((gain.gain()[(index, index)] - expected).abs() < 2.0e-5);
    }
    assert!(gain.gain()[(0, 1)].abs() < 1.0e-6);
    assert!(gain.gain()[(1, 0)].abs() < 1.0e-6);
}
