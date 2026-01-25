//! Noise generation utilities for simulation.

/// Generate random noise in range [-1, 1] (uniform distribution).
#[inline]
pub fn rand_noise() -> f32 {
    2.0 * (rand::random::<f32>() - 0.5)
}

/// Generate Gaussian noise with the specified standard deviation.
///
/// Uses the Box-Muller transform to generate normally distributed values.
pub fn gaussian_noise(std_dev: f32) -> f32 {
    // Box-Muller transform
    let u1: f32 = rand::random();
    let u2: f32 = rand::random();

    // Avoid log(0)
    let u1 = u1.max(1e-10);

    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    z0 * std_dev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_noise_range() {
        for _ in 0..1000 {
            let n = rand_noise();
            assert!(n >= -1.0 && n <= 1.0);
        }
    }

    #[test]
    fn test_gaussian_noise_distribution() {
        let std_dev = 1.0;
        let n_samples = 10000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n_samples {
            let n = gaussian_noise(std_dev);
            sum += n;
            sum_sq += n * n;
        }

        let mean = sum / n_samples as f32;
        let variance = sum_sq / n_samples as f32 - mean * mean;

        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
        // Variance should be close to std_dev^2
        assert!((variance - std_dev * std_dev).abs() < 0.2);
    }
}
