use super::*;
use nalgebra::{ArrayStorage, Matrix, U1, U100, U4};

/// Maximum observation range
pub const MAX_RANGE: f32 = 60.0;

pub const NP: usize = 100;
pub const NTh: f32 = NP as f32 / 2.0;

pub type PX = Matrix<f32, U4, U100, ArrayStorage<f32, 4, 100>>;
pub type PW = Matrix<f32, U1, U100, ArrayStorage<f32, 1, 100>>;

/// Generate random float between [-1.0, 1.0]
pub fn rand() -> f32 {
    2.0 * (rand::random::<f32>() - 0.5)
}

pub fn rand_unifrom(low: f32, high: f32) -> f32 {
    use rand::Rng;
    let lim = rand::distributions::Uniform::new(low, high);
    let mut rng = rand::thread_rng();
    rng.sample(lim)
}

pub fn calc_input() -> Vector2 {
    let v = 1.0;
    let yaw_rate = 0.1;
    vector![v, yaw_rate]
}

pub fn observation(
    x_true: &mut Vector4,
    xd: &mut Vector4,
    u: Vector2,
    rf_id: &[Vector2],
    dt: f32,
    max_range: f32,
) -> (Vec<Vector3>, Vector2) {
    *x_true = motion_model(*x_true, u, dt);

    let Q_sim = diag![0.2];
    let R_sim = diag![1.0, (30_f32).to_radians()];
    let mut z = Vec::new();

    for rf_id in rf_id.iter() {
        let dx = x_true.x() - rf_id.x();
        let dy = x_true.y() - rf_id.y();
        let d = hypot(dx, dy);
        if d <= max_range {
            let dn = d + rand() * sqrt(Q_sim[0]);
            let zi = Vector3::new(dn, rf_id.x(), rf_id.y());
            z.push(zi);
        }
    }
    let ud1 = u.x() + rand() * sqrt(R_sim.get_diagonal(0));
    let ud2 = u.y() + rand() * sqrt(R_sim.get_diagonal(1));
    let ud = Vector2::new(ud1, ud2);

    *xd = motion_model(*xd, ud, dt);

    (z, ud)
}

/// Generate observations from an externally-controlled state
///
/// Unlike `observation`, this function does not update x_true using motion_model.
/// Use this when x_true is controlled by an external dynamics model (e.g., bicycle model).
pub fn observation_from_state(
    x_true: &Vector4,
    xd: &mut Vector4,
    u: Vector2,
    rf_id: &[Vector2],
    dt: f32,
    max_range: f32,
) -> (Vec<Vector3>, Vector2) {
    let Q_sim = diag![0.2];
    let R_sim = diag![1.0, (30_f32).to_radians()];
    let mut z = Vec::new();

    for rf_id in rf_id.iter() {
        let dx = x_true.x() - rf_id.x();
        let dy = x_true.y() - rf_id.y();
        let d = hypot(dx, dy);
        if d <= max_range {
            let dn = d + rand() * sqrt(Q_sim[0]);
            let zi = Vector3::new(dn, rf_id.x(), rf_id.y());
            z.push(zi);
        }
    }
    let ud1 = u.x() + rand() * sqrt(R_sim.get_diagonal(0));
    let ud2 = u.y() + rand() * sqrt(R_sim.get_diagonal(1));
    let ud = Vector2::new(ud1, ud2);

    *xd = motion_model(*xd, ud, dt);

    (z, ud)
}

pub fn motion_model(x: Vector4, u: Vector2, dt: f32) -> Vector4 {
    let F = diag![1., 1., 1., 0.];

    let B = matrix![cos(x.phi()) * dt, 0.;
                    sin(x.phi()) * dt, 0.;
                    0., dt;
                    1., 0.];

    F * x + B * u
}

pub fn gauss_likelihood(x: f32, sigma: f32) -> f32 {
    1.0 / sqrt(2.0 * PI * sigma.powi(2)) * exp(-(x.powi(2)) / (2.0 * sigma.powi(2)))
}

pub fn calc_covariance(x_est: &Vector4, px: &PX, pw: &PW) -> Matrix3 {
    let mut cov = zeros!(3, 3);
    for i in 0..NP {
        let dx = px.column(i) - x_est;
        let dx = dx.rows(0, 3);
        cov += pw[i] * (dx * dx.transpose());
    }
    cov *= 1. / (vector![1_f32] - pw * pw.transpose())[0];
    cov
}

/// Minimum weight sum threshold - below this, particles have diverged
const DIVERGENCE_THRESHOLD: f32 = 1e-10;
/// Spread for resetting particles around observations
const RESET_SPREAD: f32 = 2.0;
/// Base smoothing factor for estimate updates (0 = no smoothing, 1 = full smoothing)
const ESTIMATE_SMOOTHING_BASE: f32 = 0.5;
/// High smoothing factor used during recovery from no observations
const ESTIMATE_SMOOTHING_RECOVERY: f32 = 0.9;
/// Weight for uniform component in likelihood (prevents complete weight collapse)
const LIKELIHOOD_UNIFORM_WEIGHT: f32 = 0.01;

/// Compute likelihood with uniform mixture to prevent weight collapse
fn robust_likelihood(dz: f32, sigma: f32) -> f32 {
    let gauss = gauss_likelihood(dz, sigma);
    // Mix with small uniform probability to prevent complete collapse
    (1.0 - LIKELIHOOD_UNIFORM_WEIGHT) * gauss + LIKELIHOOD_UNIFORM_WEIGHT * 0.01
}

pub fn pf_localization(
    x_est: &mut Vector4,
    px: &mut PX,
    pw: &mut PW,
    z: Vec<Vector3>,
    u: Vector2,
    dt: f32,
) -> Matrix3 {
    pf_localization_with_state(x_est, px, pw, z, u, dt, &mut PFState::default())
}

/// Particle filter state for tracking recovery phases
#[derive(Default)]
pub struct PFState {
    /// Number of consecutive steps without observations
    pub no_obs_count: u32,
    /// Number of steps since observations resumed (for recovery smoothing)
    pub recovery_count: u32,
}

/// Extended particle filter localization with state tracking
pub fn pf_localization_with_state(
    x_est: &mut Vector4,
    px: &mut PX,
    pw: &mut PW,
    z: Vec<Vector3>,
    u: Vector2,
    dt: f32,
    state: &mut PFState,
) -> Matrix3 {
    // Softer observation noise for more robust likelihood
    let Q = diag![1.0];  // Increased from 0.2 for softer likelihood
    let R = diag![2.0, (40_f32).to_radians()];

    // Track observation state
    let has_observations = !z.is_empty();
    if !has_observations {
        state.no_obs_count = state.no_obs_count.saturating_add(1);
        state.recovery_count = 0;
    } else if state.no_obs_count > 10 {
        // Was without observations for a while, now in recovery
        state.recovery_count = state.recovery_count.saturating_add(1);
        state.no_obs_count = 0;
    } else {
        state.no_obs_count = 0;
        state.recovery_count = state.recovery_count.saturating_add(1);
    }

    for ip in 0..NP {
        let x = px.column(ip).into_owned();
        let mut w = pw[(0, ip)];

        let ud1 = u[0] + rand() * sqrt(R[(0, 0)]);
        let ud2 = u[1] + rand() * sqrt(R[(1, 1)]);
        let ud = vector![ud1, ud2];
        let x = motion_model(x, ud, dt);

        for zi in z.iter() {
            let dx = x[0] - zi.x();
            let dy = x[1] - zi.y();
            let pre_z = hypot(dx, dy);
            let dz = pre_z - zi.d();
            w = w * robust_likelihood(dz, sqrt(Q[(0, 0)]));
        }
        px.set_column(ip, &x);
        pw[ip] = w;
    }

    let w_sum = pw.sum();

    // Check for divergence: if weight sum is too small, reset particles
    if w_sum < DIVERGENCE_THRESHOLD && has_observations {
        // Reset particles around observed landmarks
        reset_particles_around_observations(px, pw, &z, x_est);
        state.recovery_count = 1; // Start recovery phase
    } else if w_sum > 0.0 {
        *pw /= w_sum;
    } else {
        // No observations and weights collapsed - reset to uniform
        *pw = ones!(1, NP) * (1. / NP as f32);
    }

    // Compute new estimate from particles
    let x_new = *px * pw.transpose();

    // Adaptive smoothing: use higher smoothing during recovery phase
    let smoothing = if state.recovery_count > 0 && state.recovery_count < 50 {
        // Gradually reduce smoothing from recovery to base over 50 steps
        let t = state.recovery_count as f32 / 50.0;
        ESTIMATE_SMOOTHING_RECOVERY * (1.0 - t) + ESTIMATE_SMOOTHING_BASE * t
    } else {
        ESTIMATE_SMOOTHING_BASE
    };

    // Smooth the estimate to avoid sudden jumps
    *x_est = *x_est * smoothing + x_new * (1.0 - smoothing);

    let p_est = calc_covariance(x_est, px, pw);

    let N_eff = 1. / (*pw * pw.transpose())[0];
    if N_eff < NTh {
        re_sampling(px, pw);
    }
    p_est
}

/// Reset particles around observed landmarks when filter has diverged
fn reset_particles_around_observations(px: &mut PX, pw: &mut PW, z: &[Vector3], x_est: &Vector4) {
    for ip in 0..NP {
        // Pick a random observation to reset around
        let zi = &z[ip % z.len()];
        // The observation gives distance to landmark at (zi.x(), zi.y())
        // Place particle at random angle from landmark at observed distance
        let angle = rand() * 2.0 * PI; // Full circle
        let dist = zi.d() + rand() * RESET_SPREAD;
        let new_particle = vector![
            zi.x() + dist * cos(angle),
            zi.y() + dist * sin(angle),
            x_est[2] + rand() * 0.3,  // Keep approximate heading with small noise
            x_est[3]                   // Keep velocity
        ];
        px.set_column(ip, &new_particle);
    }
    *pw = ones!(1, NP) * (1. / NP as f32);
}

pub fn re_sampling(px: &mut PX, pw: &mut PW) {
    let w_cum: Vec<f32> = pw
        .as_slice()
        .iter()
        .scan(0.0, |acc, x| {
            *acc += *x;
            Some(*acc)
        })
        .collect();

    let base = (0..NP).map(|x| x as f32 / NP as f32);
    let resample_id: Vec<f32> = base.map(|x| x + rand_unifrom(0., 1. / NP as f32)).collect();

    let mut ind = 0;
    let mut px_new = zeros!(4, NP);
    for ip in 0..NP {
        while resample_id[ip] > w_cum[ind] && ind < NP - 1 {
            ind += 1;
        }
        px_new.set_column(ip, &px.column(ind));
    }
    *px = px_new;
    *pw = ones!(1, NP) * (1. / NP as f32);
}

/// Getter methods for particle
pub trait Observation {
    fn d(&self) -> f32;
    fn x(&self) -> f32;
    fn y(&self) -> f32;
}

impl Observation for Vector3 {
    fn d(&self) -> f32 {
        *self.get(0).expect("Cannot get 1st element of Vector3")
    }
    fn x(&self) -> f32 {
        *self.get(1).expect("Cannot get 2nd element of Vector3")
    }
    fn y(&self) -> f32 {
        *self.get(2).expect("Cannot get 3rd element of Vector3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // #[test]
    // fn check_heapless() {
    //     let mut v: Vec<f32, 3> = Vec::new();
    //     v.push(0.1).unwrap();
    //     v.push(0.2).unwrap();
    //     v.push(0.3).unwrap();
    //     assert_eq!(v.get(1), Some(&0.2));
    // }
    // #[test]
    // fn check_oorandom() {
    //     let seed = 4;
    //     let mut rng = oorandom::Rand32::new(seed);
    //     for _ in 0..10 {
    //         std::println!("{}", rng.rand_float());
    //     }
    // }

    #[test]
    fn check_rand() {
        for _ in 0..100 {
            let r = rand::random::<f32>();
            println!("{r}");
        }
    }

    #[test]
    fn test_main() {
        let mut time = 0_f32;
        let rf_id = vec![
            vector![10.0_f32, 0.0_f32],
            vector![10.0, 10.0],
            vector![0.0, 15.0],
            vector![-5.0, 20.0],
        ];

        let mut x_est = zeros!(4, 1);
        let mut x_true = zeros!(4, 1);

        let mut px = zeros!(4, NP);
        let mut pw = ones!(1, NP) * (1. / NP as f32);
        let mut x_dr = zeros!(4, 1);

        let mut h_x_est = vec![x_est];
        let mut h_x_true = vec![x_true];
        let mut h_x_dr = vec![x_true];

        let dt = 0.1;

        while 50.0 > time {
            time += dt;
            let u = calc_input();

            let (z, ud) = observation(&mut x_true, &mut x_dr, u, &rf_id, dt, MAX_RANGE);
            let _PEst = pf_localization(&mut x_est, &mut px, &mut pw, z, ud, dt);

            h_x_est.push(x_est);
            h_x_true.push(x_true);
            h_x_dr.push(x_dr);

            println!("{time}");
            dbg!(&x_est);
        }
    }
}
