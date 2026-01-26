//! Robust kernel functions for outlier-resistant optimization
//!
//! Robust kernels reduce the influence of outliers by re-weighting residuals.
//! Instead of minimizing sum(r^2), we minimize sum(rho(r)) where rho is a
//! robust loss function.
//!
//! ## Common Kernels
//! - **Huber**: Linear outside threshold, quadratic inside (default in slam_toolbox)
//! - **Cauchy**: Heavy-tailed, very outlier-resistant
//! - **Tukey**: Complete outlier rejection beyond threshold
//!
//! ## References
//! - [Robust Estimation](https://en.wikipedia.org/wiki/Robust_statistics)
//! - [M-estimators](https://en.wikipedia.org/wiki/M-estimator)

/// Trait for robust kernel functions
///
/// A robust kernel transforms a squared residual r^2 to reduce outlier influence.
/// The key function is `weight(r)` which returns a multiplicative weight for
/// iteratively reweighted least squares (IRLS).
pub trait RobustKernel {
    /// Compute the weight for a given residual magnitude
    ///
    /// For IRLS: minimize sum(w_i * r_i^2) where w_i = weight(|r_i|)
    fn weight(&self, residual_abs: f64) -> f64;

    /// Compute the robust cost rho(r) for a given residual
    fn cost(&self, residual_abs: f64) -> f64;

    /// Compute the first derivative rho'(r) / r (influence function)
    fn influence(&self, residual_abs: f64) -> f64;
}

/// Trivial kernel - standard least squares (no robustification)
#[derive(Debug, Clone, Copy, Default)]
pub struct TrivialKernel;

impl RobustKernel for TrivialKernel {
    fn weight(&self, _residual_abs: f64) -> f64 {
        1.0
    }

    fn cost(&self, residual_abs: f64) -> f64 {
        0.5 * residual_abs * residual_abs
    }

    fn influence(&self, residual_abs: f64) -> f64 {
        residual_abs
    }
}

/// Huber robust kernel
///
/// Quadratic for small residuals, linear for large ones.
/// This is the default in slam_toolbox and g2o.
///
/// rho(r) = 0.5 * r^2           if |r| <= k
///        = k * |r| - 0.5 * k^2  if |r| > k
///
/// weight(r) = 1                if |r| <= k
///           = k / |r|          if |r| > k
#[derive(Debug, Clone, Copy)]
pub struct HuberKernel {
    /// Threshold parameter (default: 1.345 for 95% efficiency at normal distribution)
    pub k: f64,
}

impl Default for HuberKernel {
    fn default() -> Self {
        Self { k: 1.345 }
    }
}

impl HuberKernel {
    pub fn new(k: f64) -> Self {
        Self { k: k.abs().max(1e-6) }
    }

    /// Create with automatic scaling based on MAD (Median Absolute Deviation)
    pub fn from_mad(mad: f64) -> Self {
        // k = 1.345 * sigma, where sigma ≈ 1.4826 * MAD for normal distribution
        Self::new(1.345 * 1.4826 * mad)
    }
}

impl RobustKernel for HuberKernel {
    fn weight(&self, residual_abs: f64) -> f64 {
        if residual_abs <= self.k {
            1.0
        } else {
            self.k / residual_abs
        }
    }

    fn cost(&self, residual_abs: f64) -> f64 {
        if residual_abs <= self.k {
            0.5 * residual_abs * residual_abs
        } else {
            self.k * residual_abs - 0.5 * self.k * self.k
        }
    }

    fn influence(&self, residual_abs: f64) -> f64 {
        if residual_abs <= self.k {
            residual_abs
        } else {
            self.k * residual_abs.signum()
        }
    }
}

/// Cauchy (Lorentzian) robust kernel
///
/// Very outlier-resistant with heavy tails. Good for severe outliers.
///
/// rho(r) = (c^2 / 2) * log(1 + (r/c)^2)
/// weight(r) = 1 / (1 + (r/c)^2)
#[derive(Debug, Clone, Copy)]
pub struct CauchyKernel {
    /// Scale parameter (default: 2.3849 for 95% efficiency)
    pub c: f64,
}

impl Default for CauchyKernel {
    fn default() -> Self {
        Self { c: 2.3849 }
    }
}

impl CauchyKernel {
    pub fn new(c: f64) -> Self {
        Self { c: c.abs().max(1e-6) }
    }

    /// Create with automatic scaling based on MAD
    pub fn from_mad(mad: f64) -> Self {
        Self::new(2.3849 * 1.4826 * mad)
    }
}

impl RobustKernel for CauchyKernel {
    fn weight(&self, residual_abs: f64) -> f64 {
        let ratio = residual_abs / self.c;
        1.0 / (1.0 + ratio * ratio)
    }

    fn cost(&self, residual_abs: f64) -> f64 {
        let ratio = residual_abs / self.c;
        0.5 * self.c * self.c * (1.0 + ratio * ratio).ln()
    }

    fn influence(&self, residual_abs: f64) -> f64 {
        let ratio = residual_abs / self.c;
        residual_abs / (1.0 + ratio * ratio)
    }
}

/// Tukey's biweight (bisquare) kernel
///
/// Completely rejects outliers beyond threshold. Most aggressive outlier rejection.
///
/// rho(r) = (c^2/6) * (1 - (1 - (r/c)^2)^3)  if |r| <= c
///        = c^2/6                             if |r| > c
///
/// weight(r) = (1 - (r/c)^2)^2  if |r| <= c
///           = 0                if |r| > c
#[derive(Debug, Clone, Copy)]
pub struct TukeyKernel {
    /// Cutoff parameter (default: 4.685 for 95% efficiency)
    pub c: f64,
}

impl Default for TukeyKernel {
    fn default() -> Self {
        Self { c: 4.685 }
    }
}

impl TukeyKernel {
    pub fn new(c: f64) -> Self {
        Self { c: c.abs().max(1e-6) }
    }

    /// Create with automatic scaling based on MAD
    pub fn from_mad(mad: f64) -> Self {
        Self::new(4.685 * 1.4826 * mad)
    }
}

impl RobustKernel for TukeyKernel {
    fn weight(&self, residual_abs: f64) -> f64 {
        if residual_abs <= self.c {
            let ratio = residual_abs / self.c;
            let term = 1.0 - ratio * ratio;
            term * term
        } else {
            0.0
        }
    }

    fn cost(&self, residual_abs: f64) -> f64 {
        let c2_6 = self.c * self.c / 6.0;
        if residual_abs <= self.c {
            let ratio = residual_abs / self.c;
            let term = 1.0 - ratio * ratio;
            c2_6 * (1.0 - term * term * term)
        } else {
            c2_6
        }
    }

    fn influence(&self, residual_abs: f64) -> f64 {
        if residual_abs <= self.c {
            let ratio = residual_abs / self.c;
            let term = 1.0 - ratio * ratio;
            residual_abs * term * term
        } else {
            0.0
        }
    }
}

/// Type of robust kernel to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RobustKernelType {
    /// No robustification (standard least squares)
    None,
    /// Huber kernel (default, good balance)
    #[default]
    Huber,
    /// Cauchy kernel (heavy outliers)
    Cauchy,
    /// Tukey kernel (complete rejection)
    Tukey,
}

/// Boxed robust kernel for dynamic dispatch
pub struct DynamicKernel {
    kernel: Box<dyn RobustKernel + Send + Sync>,
}

impl DynamicKernel {
    pub fn new(kernel_type: RobustKernelType, scale: Option<f64>) -> Self {
        let kernel: Box<dyn RobustKernel + Send + Sync> = match kernel_type {
            RobustKernelType::None => Box::new(TrivialKernel),
            RobustKernelType::Huber => {
                Box::new(scale.map(HuberKernel::new).unwrap_or_default())
            }
            RobustKernelType::Cauchy => {
                Box::new(scale.map(CauchyKernel::new).unwrap_or_default())
            }
            RobustKernelType::Tukey => {
                Box::new(scale.map(TukeyKernel::new).unwrap_or_default())
            }
        };
        Self { kernel }
    }

    pub fn weight(&self, residual_abs: f64) -> f64 {
        self.kernel.weight(residual_abs)
    }

    pub fn cost(&self, residual_abs: f64) -> f64 {
        self.kernel.cost(residual_abs)
    }
}

/// Chi-squared thresholds for outlier detection
///
/// These are the critical values of chi-squared distribution for various DOFs
/// at the 95% confidence level.
pub mod chi_squared {
    /// Chi-squared threshold for 2 DOF (landmark observation) at 95%
    pub const DOF_2_95: f64 = 5.991;

    /// Chi-squared threshold for 3 DOF (pose constraint) at 95%
    pub const DOF_3_95: f64 = 7.815;

    /// Chi-squared threshold for 2 DOF at 99%
    pub const DOF_2_99: f64 = 9.210;

    /// Chi-squared threshold for 3 DOF at 99%
    pub const DOF_3_99: f64 = 11.345;

    /// Check if a Mahalanobis distance squared indicates an outlier
    ///
    /// d2 = r^T * Omega * r where Omega is the information matrix
    pub fn is_outlier(mahalanobis_sq: f64, dof: usize, confidence: f64) -> bool {
        let threshold = match (dof, (confidence * 100.0) as u32) {
            (2, 95) => DOF_2_95,
            (3, 95) => DOF_3_95,
            (2, 99) => DOF_2_99,
            (3, 99) => DOF_3_99,
            _ => {
                // Approximate using Wilson-Hilferty transformation
                let p = 1.0 - confidence;
                let z = if p > 0.5 {
                    -normal_quantile(p)
                } else {
                    normal_quantile(1.0 - p)
                };
                let dof_f = dof as f64;
                let term = 1.0 - 2.0 / (9.0 * dof_f) + z * (2.0 / (9.0 * dof_f)).sqrt();
                dof_f * term * term * term
            }
        };
        mahalanobis_sq > threshold
    }

    /// Approximate normal quantile (inverse CDF) using Beasley-Springer-Moro algorithm
    fn normal_quantile(p: f64) -> f64 {
        // Rational approximation for 0.5 < p < 1
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 {
            -result
        } else {
            result
        }
    }
}

/// Compute the Median Absolute Deviation (MAD) of residuals
///
/// MAD is a robust measure of scale: MAD = median(|r_i - median(r)|)
/// For Gaussian data, sigma ≈ 1.4826 * MAD
pub fn compute_mad(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 1.0;
    }

    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());

    let median_abs = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1].abs() + sorted[mid].abs()) / 2.0
    } else {
        sorted[sorted.len() / 2].abs()
    };

    // MAD of absolute residuals (assuming zero median for residuals)
    median_abs.max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_weight() {
        let kernel = HuberKernel::new(1.0);

        // Inside threshold: weight = 1
        assert!((kernel.weight(0.5) - 1.0).abs() < 1e-6);

        // At threshold: weight = 1
        assert!((kernel.weight(1.0) - 1.0).abs() < 1e-6);

        // Outside threshold: weight = k/r
        assert!((kernel.weight(2.0) - 0.5).abs() < 1e-6);
        assert!((kernel.weight(4.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_cauchy_weight() {
        let kernel = CauchyKernel::new(1.0);

        // At zero: weight = 1
        assert!((kernel.weight(0.0) - 1.0).abs() < 1e-6);

        // At c: weight = 0.5
        assert!((kernel.weight(1.0) - 0.5).abs() < 1e-6);

        // Large residuals get small weights
        assert!(kernel.weight(10.0) < 0.01);
    }

    #[test]
    fn test_tukey_weight() {
        let kernel = TukeyKernel::new(1.0);

        // At zero: weight = 1
        assert!((kernel.weight(0.0) - 1.0).abs() < 1e-6);

        // Beyond threshold: weight = 0
        assert!((kernel.weight(1.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared_outlier() {
        // 2 DOF at 95%: threshold = 5.991
        assert!(!chi_squared::is_outlier(5.0, 2, 0.95));
        assert!(chi_squared::is_outlier(6.5, 2, 0.95));

        // 3 DOF at 95%: threshold = 7.815
        assert!(!chi_squared::is_outlier(7.0, 3, 0.95));
        assert!(chi_squared::is_outlier(8.0, 3, 0.95));
    }

    #[test]
    fn test_compute_mad() {
        let residuals = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier at 100
        let mad = compute_mad(&residuals);

        // MAD should be robust to the outlier
        // median of [1, 2, 3, 4, 100] is 3
        // But we compute median of absolute values which is also 3
        assert!((mad - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_kernel() {
        let huber = DynamicKernel::new(RobustKernelType::Huber, Some(1.0));
        assert!((huber.weight(0.5) - 1.0).abs() < 1e-6);
        assert!((huber.weight(2.0) - 0.5).abs() < 1e-6);

        let none = DynamicKernel::new(RobustKernelType::None, None);
        assert!((none.weight(100.0) - 1.0).abs() < 1e-6);
    }
}
