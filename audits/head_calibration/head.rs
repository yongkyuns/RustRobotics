//! Independent output-layer calibration. No tensor backend or optimizer state.
//!
//! The caller owns a frozen encoder snapshot and supplies its 64 float32 features.
//! Returned coefficients belong to that snapshot, not to an Adam-managed head.

pub const FEATURES: usize = 64;
pub const COEFFICIENTS: usize = FEATURES + 1;
pub const RIDGE: f64 = 0.001;
pub const SCALE_FLOOR: f64 = 1e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    Empty,
    Length,
    NonFinite,
    NotPositiveDefinite,
    InvalidSolution,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FittedHead {
    pub coefficients: [f32; COEFFICIENTS],
    pub delta: [f64; COEFFICIENTS],
    pub feature_mean: [f64; FEATURES],
    pub feature_scale: [f64; FEATURES],
    pub relative_residual: f64,
    pub objective_before: f64,
    pub objective_after: f64,
}

impl FittedHead {
    /// Scalar float32 prediction on features from the same frozen encoder.
    pub fn predict(&self, features: &[f32; FEATURES]) -> Result<f32, FitError> {
        if !features.iter().all(|x| x.is_finite()) {
            return Err(FitError::NonFinite);
        }
        let value = features
            .iter()
            .zip(&self.coefficients[..FEATURES])
            .map(|(x, w)| *x * *w)
            .sum::<f32>()
            + self.coefficients[FEATURES];
        if value.is_finite() {
            Ok(value)
        } else {
            Err(FitError::NonFinite)
        }
    }
}

fn solve_spd(
    matrix: &[[f64; COEFFICIENTS]],
    rhs: &[f64; COEFFICIENTS],
) -> Result<[f64; COEFFICIENTS], FitError> {
    let mut lower = vec![[0.0; COEFFICIENTS]; COEFFICIENTS];
    for (i, matrix_row) in matrix.iter().enumerate() {
        let (previous, current) = lower.split_at_mut(i);
        let row = &mut current[0];
        for (j, other) in previous.iter().enumerate() {
            let inner: f64 = row[..j].iter().zip(&other[..j]).map(|(a, b)| a * b).sum();
            row[j] = (matrix_row[j] - inner) / other[j];
        }
        let diagonal = matrix_row[i] - row[..i].iter().map(|x| x * x).sum::<f64>();
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(FitError::NotPositiveDefinite);
        }
        row[i] = diagonal.sqrt();
    }
    let mut y = [0.0; COEFFICIENTS];
    for (i, row) in lower.iter().enumerate() {
        let inner: f64 = row[..i].iter().zip(&y[..i]).map(|(a, b)| a * b).sum();
        y[i] = (rhs[i] - inner) / row[i];
    }
    let mut x = [0.0; COEFFICIENTS];
    for (i, row) in lower.iter().enumerate().rev() {
        let inner: f64 = lower[i + 1..]
            .iter()
            .zip(&x[i + 1..])
            .map(|(other, value)| other[i] * value)
            .sum();
        x[i] = (y[i] - inner) / row[i];
    }
    if x.iter().all(|x| x.is_finite()) {
        Ok(x)
    } else {
        Err(FitError::InvalidSolution)
    }
}

fn standardized(
    row: &[f32; FEATURES],
    mean: &[f64; FEATURES],
    scale: &[f64; FEATURES],
) -> [f64; COEFFICIENTS] {
    let mut result = [1.0; COEFFICIENTS];
    for (i, x) in row.iter().enumerate() {
        result[i] = (f64::from(*x) - mean[i]) / scale[i];
    }
    result
}

/// One ridge solve around `incoming`, with an unpenalized intercept correction.
/// Fit data and incoming coefficients cannot be mutated by this API.
pub fn fit(
    features: &[[f32; FEATURES]],
    targets: &[f32],
    incoming: &[f32; COEFFICIENTS],
) -> Result<FittedHead, FitError> {
    if features.is_empty() {
        return Err(FitError::Empty);
    }
    if features.len() != targets.len() {
        return Err(FitError::Length);
    }
    if !features.iter().flatten().all(|x| x.is_finite())
        || !targets.iter().all(|x| x.is_finite())
        || !incoming.iter().all(|x| x.is_finite())
    {
        return Err(FitError::NonFinite);
    }
    let count = features.len() as f64;
    let mut mean = [0.0; FEATURES];
    for row in features {
        for (sum, x) in mean.iter_mut().zip(row) {
            *sum += f64::from(*x) / count;
        }
    }
    let mut scale = [0.0; FEATURES];
    for row in features {
        for (i, variance) in scale.iter_mut().enumerate() {
            let d = f64::from(row[i]) - mean[i];
            *variance += d * d / count;
        }
    }
    for s in &mut scale {
        let deviation = s.sqrt();
        *s = if deviation > SCALE_FLOOR { deviation } else { 1.0 };
    }
    let mut matrix = vec![[0.0; COEFFICIENTS]; COEFFICIENTS];
    let mut rhs = [0.0; COEFFICIENTS];
    let mut residuals = Vec::with_capacity(features.len());
    for (row, target) in features.iter().zip(targets) {
        let old_value = row
            .iter()
            .zip(&incoming[..FEATURES])
            .map(|(x, w)| f64::from(*x) * f64::from(*w))
            .sum::<f64>()
            + f64::from(incoming[FEATURES]);
        let residual = f64::from(*target) - old_value;
        residuals.push(residual);
        let z = standardized(row, &mean, &scale);
        for (i, matrix_row) in matrix.iter_mut().enumerate() {
            rhs[i] += z[i] * residual / count;
            for (cell, zj) in matrix_row.iter_mut().zip(z) {
                *cell += z[i] * zj / count;
            }
        }
    }
    for (i, row) in matrix.iter_mut().take(FEATURES).enumerate() {
        row[i] += RIDGE;
    }
    let delta = solve_spd(&matrix, &rhs)?;
    let mut coefficients = *incoming;
    let mut offset = 0.0;
    for (i, value) in coefficients.iter_mut().take(FEATURES).enumerate() {
        let slope = delta[i] / scale[i];
        *value = (f64::from(incoming[i]) + slope) as f32;
        offset += mean[i] * slope;
    }
    coefficients[FEATURES] = (f64::from(incoming[FEATURES]) + delta[FEATURES] - offset) as f32;
    let residual_norm = matrix
        .iter()
        .zip(rhs)
        .map(|(row, target)| {
            let d = row.iter().zip(delta).map(|(a, b)| a * b).sum::<f64>() - target;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    let relative_residual = residual_norm / rhs.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    let objective_before = residuals.iter().map(|x| x * x / count).sum::<f64>();
    let mut objective_after = RIDGE * delta[..FEATURES].iter().map(|x| x * x).sum::<f64>();
    for (row, residual) in features.iter().zip(residuals) {
        let z = standardized(row, &mean, &scale);
        let error = z.iter().zip(delta).map(|(a, b)| a * b).sum::<f64>() - residual;
        objective_after += error * error / count;
    }
    if !coefficients.iter().all(|x| x.is_finite())
        || !relative_residual.is_finite()
        || relative_residual > 1e-9
        || !objective_after.is_finite()
        || objective_after > objective_before + 1e-9 * objective_before.max(1.0)
    {
        return Err(FitError::InvalidSolution);
    }
    Ok(FittedHead {
        coefficients,
        delta,
        feature_mean: mean,
        feature_scale: scale,
        relative_residual,
        objective_before,
        objective_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varied() -> Vec<[f32; FEATURES]> {
        (0..32)
            .map(|i| {
                let mut h = [2.0; FEATURES];
                h[0] = i as f32 / 8.0;
                h[1] = h[0] * h[0];
                h
            })
            .collect()
    }

    #[test]
    fn constant_features_fit_unpenalized_intercept() {
        let mut old = [0.0; COEFFICIENTS];
        old[FEATURES] = 5.0;
        let result = fit(&[[3.0; FEATURES]; 8], &[7.0; 8], &old).unwrap();
        assert_eq!(result.coefficients[FEATURES], 7.0);
        assert!(result.coefficients[..FEATURES].iter().all(|x| *x == 0.0));
        assert_eq!(result.predict(&[3.0; FEATURES]).unwrap(), 7.0);
    }

    #[test]
    fn one_row_is_well_defined() {
        let result = fit(&[[0.0; FEATURES]], &[3.0], &[0.0; COEFFICIENTS]).unwrap();
        assert_eq!(result.coefficients[FEATURES], 3.0);
    }

    #[test]
    fn zero_residual_keeps_head() {
        let old = [0.0; COEFFICIENTS];
        assert_eq!(fit(&varied(), &[0.0; 32], &old).unwrap().coefficients, old);
    }

    #[test]
    fn rank_deficient_features_are_regularized() {
        let x = varied();
        let y: Vec<_> = x.iter().map(|h| 3.0 * h[0] + 4.0).collect();
        let result = fit(&x, &y, &[0.0; COEFFICIENTS]).unwrap();
        assert!(result.objective_after < result.objective_before);
        assert!(result.relative_residual < 1e-12);
    }

    #[test]
    fn rejected_inputs() {
        let old = [0.0; COEFFICIENTS];
        assert_eq!(fit(&[], &[], &old), Err(FitError::Empty));
        assert_eq!(fit(&varied(), &[0.0], &old), Err(FitError::Length));
        assert_eq!(fit(&[[f32::NAN; FEATURES]], &[0.0], &old), Err(FitError::NonFinite));
        assert_eq!(fit(&[[0.0; FEATURES]], &[f32::INFINITY], &old), Err(FitError::NonFinite));
        assert_eq!(fit(&[[0.0; FEATURES]], &[0.0], &[f32::NAN; COEFFICIENTS]), Err(FitError::NonFinite));
    }

    #[test]
    fn inputs_immutable_and_repeat_exact() {
        let x = varied();
        let y = vec![7.0; x.len()];
        let old = [0.0; COEFFICIENTS];
        let before = (x.clone(), y.clone(), old);
        let first = fit(&x, &y, &old).unwrap();
        let second = fit(&x, &y, &old).unwrap();
        assert_eq!(first, second);
        assert_eq!((x, y, old), before);
    }

    #[test]
    fn different_labels_do_not_reuse_previous_fit() {
        let x = varied();
        let old = [0.0; COEFFICIENTS];
        let first = fit(&x, &[3.0; 32], &old).unwrap();
        let second = fit(&x, &[8.0; 32], &old).unwrap();
        assert_eq!(first.coefficients[FEATURES], 3.0);
        assert_eq!(second.coefficients[FEATURES], 8.0);
    }

    #[test]
    fn row_permutation_is_numerically_stable() {
        let x = varied();
        let y: Vec<_> = x.iter().map(|h| h[0] * 2.0 + 1.0).collect();
        let old = [0.0; COEFFICIENTS];
        let first = fit(&x, &y, &old).unwrap();
        let mut reverse_x = x.clone();
        let mut reverse_y = y.clone();
        reverse_x.reverse();
        reverse_y.reverse();
        let second = fit(&reverse_x, &reverse_y, &old).unwrap();
        for (a, b) in first.coefficients.iter().zip(second.coefficients) {
            assert!((*a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn nonfinite_prediction_rejected() {
        let fitted = fit(&[[0.0; FEATURES]], &[0.0], &[0.0; COEFFICIENTS]).unwrap();
        assert_eq!(fitted.predict(&[f32::NAN; FEATURES]), Err(FitError::NonFinite));
    }
}
