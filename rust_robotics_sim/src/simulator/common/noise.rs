//! Noise generation utilities for simulation.

/// Generate random noise in range [-1, 1] (uniform distribution).
#[inline]
pub fn rand_noise() -> f32 {
    rand_noise_with_rng(&mut rand::thread_rng())
}

fn rand_noise_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> f32 {
    2.0 * (rng.gen::<f32>() - 0.5)
}

/// Generate Gaussian noise with the specified standard deviation.
///
/// Uses the Box-Muller transform to generate normally distributed values.
pub fn gaussian_noise(std_dev: f32) -> f32 {
    gaussian_noise_with_rng(std_dev, &mut rand::thread_rng())
}

fn gaussian_noise_with_rng<R: rand::Rng + ?Sized>(std_dev: f32, rng: &mut R) -> f32 {
    // Box-Muller transform
    let u1: f32 = rng.gen();
    let u2: f32 = rng.gen();

    // Avoid log(0)
    let u1 = u1.max(1e-10);

    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    z0 * std_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_rand_noise_range() {
        let mut rng = StdRng::seed_from_u64(0x5015_0001);
        for _ in 0..1000 {
            let n = rand_noise_with_rng(&mut rng);
            assert!((-1.0..=1.0).contains(&n));
        }
    }

    #[test]
    fn test_gaussian_noise_distribution() {
        let mut rng = StdRng::seed_from_u64(0x5015_0002);
        let std_dev = 1.0;
        let n_samples = 10000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n_samples {
            let n = gaussian_noise_with_rng(std_dev, &mut rng);
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
