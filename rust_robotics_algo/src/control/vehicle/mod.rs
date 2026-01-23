//! Vehicle dynamics module implementing single-track (bicycle) model
//!
//! This module provides a bicycle model with:
//! - Nonlinear tire dynamics using Pacejka Magic Formula
//! - Linear tire option for comparison
//! - Simple longitudinal dynamics (acceleration-based)
//!
//! # State Vector
//! The lateral state vector is: [yaw_rate (r), yaw (ψ), lateral_position (y), lateral_velocity (vy)]
//!
//! # Inputs
//! - Steering angle (δ) for lateral control
//! - Acceleration (ax) for longitudinal control

use crate::prelude::*;

/// Pacejka Magic Formula tire parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TireParams {
    /// B - Stiffness factor (higher = sharper response, earlier peak)
    pub b: f32,
    /// C - Shape factor (typically 1.0-2.0)
    pub c: f32,
    /// D - Peak value (maximum force) [N]
    pub d: f32,
    /// E - Curvature factor (-1 to 1)
    pub e: f32,
}

impl Default for TireParams {
    fn default() -> Self {
        Self {
            b: 12.0,      // Stiffness factor (higher for sharper peak)
            c: 1.9,       // Shape factor for lateral
            d: 4500.0,    // Peak force ~4500N per tire
            e: 0.97,      // Curvature factor
        }
    }
}

impl TireParams {
    /// Create tire params with specified peak force
    pub fn with_peak_force(d: f32) -> Self {
        Self { d, ..Default::default() }
    }

    /// Compute lateral force using Pacejka Magic Formula
    ///
    /// F = D * sin(C * atan(B * α - E * (B * α - atan(B * α))))
    ///
    /// α: slip angle in radians
    /// Returns: lateral force in Newtons
    pub fn magic_formula(&self, slip_angle: f32) -> f32 {
        let b_alpha = self.b * slip_angle;
        let inner = b_alpha - self.e * (b_alpha - b_alpha.atan());
        self.d * (self.c * inner.atan()).sin()
    }
}

/// Number of lateral states: [yaw_rate, yaw, lateral_position, lateral_velocity]
pub const NX_LAT: usize = 4;
/// Number of lateral inputs: [steering_angle]
pub const NU_LAT: usize = 1;

/// Convenience types for lateral dynamics matrices
pub type AMatLat = Mat<NX_LAT, NX_LAT>;
pub type BMatLat = Mat<NX_LAT, NU_LAT>;
pub type StateLat = Vector<NX_LAT>;

/// Vehicle parameters for bicycle model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleParams {
    /// Vehicle mass [kg]
    pub mass: f32,
    /// Yaw moment of inertia [kg*m^2]
    pub iz: f32,
    /// Distance from CG to front axle [m]
    pub lf: f32,
    /// Distance from CG to rear axle [m]
    pub lr: f32,
    /// Front tire cornering stiffness [N/rad]
    pub cf: f32,
    /// Rear tire cornering stiffness [N/rad]
    pub cr: f32,
}

impl Default for VehicleParams {
    fn default() -> Self {
        Self {
            mass: 1500.0,      // kg
            iz: 3000.0,        // kg*m^2
            lf: 1.2,           // m
            lr: 1.4,           // m
            cf: 80000.0,       // N/rad
            cr: 80000.0,       // N/rad
        }
    }
}

impl VehicleParams {
    /// Get wheelbase (total length between axles)
    pub fn wheelbase(&self) -> f32 {
        self.lf + self.lr
    }
}

/// Tire model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TireModel {
    /// Linear tire model using cornering stiffness (simpler, no slip-out)
    #[default]
    Linear,
    /// Nonlinear Pacejka Magic Formula tire model (realistic, allows drifting)
    Pacejka,
}

/// Bicycle model for vehicle lateral dynamics
///
/// Supports both linear and Pacejka tire models.
/// State: [yaw_rate (r), yaw (ψ), lateral_position (y), lateral_velocity (vy)]
/// Input: [steering_angle (δ)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicycleModel {
    /// Vehicle parameters
    pub params: VehicleParams,
    /// Longitudinal velocity [m/s] (assumed constant for lateral dynamics)
    pub vx: f32,
    /// Tire model to use
    pub tire_model: TireModel,
    /// Front tire Pacejka parameters
    pub tire_front: TireParams,
    /// Rear tire Pacejka parameters (typically less grip for oversteer)
    pub tire_rear: TireParams,
}

impl Default for BicycleModel {
    fn default() -> Self {
        Self {
            params: VehicleParams::default(),
            vx: 10.0, // 10 m/s default speed
            tire_model: TireModel::default(),
            tire_front: TireParams::default(),
            tire_rear: TireParams::with_peak_force(3600.0), // 80% of front for oversteer
        }
    }
}

impl BicycleModel {
    /// Create a new bicycle model with given parameters and velocity
    pub fn new(params: VehicleParams, vx: f32) -> Self {
        Self {
            params,
            vx,
            tire_model: TireModel::default(),
            tire_front: TireParams::default(),
            tire_rear: TireParams::with_peak_force(3600.0), // 80% of front for oversteer
        }
    }

    /// Create a bicycle model with custom tire parameters
    pub fn with_tires(params: VehicleParams, vx: f32, tire_front: TireParams, tire_rear: TireParams) -> Self {
        Self {
            params,
            vx,
            tire_model: TireModel::Pacejka, // Custom tires implies Pacejka
            tire_front,
            tire_rear,
        }
    }

    /// Set the tire model
    pub fn with_tire_model(mut self, tire_model: TireModel) -> Self {
        self.tire_model = tire_model;
        self
    }

    /// Velocity threshold for transitioning from kinematic to dynamic model [m/s]
    /// Below this speed, kinematic model is used (no lateral dynamics)
    /// Above this speed, dynamic model with tire forces is used
    pub const KINEMATIC_THRESHOLD: f32 = 3.0;

    /// Blending zone width [m/s] for smooth transition between models
    pub const BLEND_ZONE: f32 = 1.0;

    /// Maximum yaw rate [rad/s] for stability
    pub const MAX_YAW_RATE: f32 = 2.0;

    /// Maximum lateral velocity [m/s] for drifting
    /// Higher value allows larger slip angles
    pub const MAX_VY: f32 = 15.0;

    /// Get the continuous-time state-space matrices (A, B) for lateral dynamics
    ///
    /// The linearized bicycle model equations:
    /// ```text
    /// dr/dt = -(Cf*lf^2 + Cr*lr^2)/(Iz*vx) * r + (Cr*lr - Cf*lf)/(Iz*vx) * vy + Cf*lf/Iz * δ
    /// dψ/dt = r
    /// dy/dt = vx*sin(ψ) + vy*cos(ψ) ≈ vx*ψ + vy (for small angles)
    /// dvy/dt = (Cr*lr - Cf*lf)/(m*vx) * r - (Cf + Cr)/(m*vx) * vy + Cf/m * δ
    /// ```
    ///
    /// State: x = [r, ψ, y, vy]^T
    /// Input: u = [δ]
    pub fn continuous_model(&self) -> (AMatLat, BMatLat) {
        let VehicleParams { mass: m, iz, lf, lr, cf, cr } = self.params;
        // Use threshold velocity for A matrix to prevent singularity
        // The blending in step() handles the transition smoothly
        let vx = self.vx.max(Self::KINEMATIC_THRESHOLD);

        // Precompute common terms
        let a11 = -(cf * lf * lf + cr * lr * lr) / (iz * vx);
        let a14 = (cr * lr - cf * lf) / (iz * vx);
        let a41 = (cr * lr - cf * lf) / (m * vx);
        let a44 = -(cf + cr) / (m * vx);

        // Continuous-time A matrix
        // State order: [r, ψ, y, vy]
        let A = matrix![
            a11,  0.,  0.,  a14;
            1.,   0.,  0.,  0.;
            0.,   vx,  0.,  1.;
            a41,  0.,  0.,  a44
        ];

        // Continuous-time B matrix
        let b1 = cf * lf / iz;
        let b4 = cf / m;
        let B = vector![b1, 0., 0., b4];

        (A, B)
    }

    /// Get discrete-time state-space matrices using Euler discretization
    ///
    /// Ad = I + A*dt
    /// Bd = B*dt
    pub fn discrete_model(&self, dt: f32) -> (AMatLat, BMatLat) {
        let (A, B) = self.continuous_model();
        let Ad = eye!(NX_LAT) + A * dt;
        let Bd = B * dt;
        (Ad, Bd)
    }

    /// Step the lateral dynamics forward by dt
    ///
    /// x_new = Ad * x + Bd * u
    pub fn step_lateral(&self, state: StateLat, steering_angle: f32, dt: f32) -> StateLat {
        let (Ad, Bd) = self.discrete_model(dt);
        Ad * state + Bd * steering_angle
    }

    /// Compute yaw rate using kinematic bicycle model (valid at low speeds)
    ///
    /// At low speeds, tire slip is negligible, so:
    /// yaw_rate = vx * tan(steering) / wheelbase
    pub fn kinematic_yaw_rate(&self, steering: f32) -> f32 {
        let wheelbase = self.params.wheelbase();
        self.vx * steering.tan() / wheelbase
    }

    /// Compute front tire slip angle (ISO convention)
    ///
    /// α_f = atan((vy + lf * r) / vx) - δ
    ///
    /// Positive slip = velocity pointing more left than wheel heading
    /// This generates a restoring force pushing right (negative Fy after negation)
    pub fn front_slip_angle(&self, steering: f32, vy: f32, yaw_rate: f32) -> f32 {
        let vx = self.vx.max(0.1); // Prevent division by zero
        ((vy + self.params.lf * yaw_rate) / vx).atan() - steering
    }

    /// Compute rear tire slip angle (ISO convention)
    ///
    /// α_r = atan((vy - lr * r) / vx)
    ///
    /// Rear wheels don't steer, so wheel heading is 0
    pub fn rear_slip_angle(&self, vy: f32, yaw_rate: f32) -> f32 {
        let vx = self.vx.max(0.1); // Prevent division by zero
        ((vy - self.params.lr * yaw_rate) / vx).atan()
    }

    /// Yaw damping coefficient [N*m*s/rad]
    /// Represents aerodynamic drag, drivetrain friction, etc.
    pub const YAW_DAMPING: f32 = 300.0;

    /// Lateral velocity damping coefficient [N*s/m]
    /// Represents aerodynamic side drag
    pub const LATERAL_DAMPING: f32 = 100.0;

    /// Apply friction circle constraint to tire forces
    ///
    /// If combined force exceeds grip limit, scale both forces proportionally
    /// Returns (limited_fx, limited_fy)
    fn apply_friction_circle(fx: f32, fy: f32, max_force: f32) -> (f32, f32) {
        let combined = (fx * fx + fy * fy).sqrt();
        if combined > max_force {
            let scale = max_force / combined;
            (fx * scale, fy * scale)
        } else {
            (fx, fy)
        }
    }

    /// Step dynamics using nonlinear Pacejka tire model with friction circle
    ///
    /// Equations of motion (body frame, SAE convention):
    /// m * dvx/dt = Fx_f + Fx_r (limited by friction circle)
    /// m * dvy/dt = Fy_f + Fy_r - m * vx * r - C_lat * vy
    /// Iz * dr/dt = lf * Fy_f - lr * Fy_r - C_yaw * r
    ///
    /// The friction circle limits combined lateral and longitudinal forces:
    /// sqrt(Fx² + Fy²) ≤ D (peak tire force)
    ///
    /// Returns (new_yaw_rate, new_vy, actual_ax)
    pub fn step_nonlinear(
        &self,
        yaw_rate: f32,
        vy: f32,
        steering: f32,
        requested_ax: f32,
        dt: f32,
    ) -> (f32, f32, f32) {
        let VehicleParams { mass: m, iz, lf, lr, .. } = self.params;
        let vx = self.vx.max(0.1);

        // Compute slip angles
        let alpha_f = self.front_slip_angle(steering, vy, yaw_rate);
        let alpha_r = self.rear_slip_angle(vy, yaw_rate);

        // Compute lateral tire forces using Magic Formula
        // Negate because tire force opposes slip (restoring force)
        let fy_f_raw = -self.tire_front.magic_formula(alpha_f);
        let fy_r_raw = -self.tire_rear.magic_formula(alpha_r);

        // Requested longitudinal force (split between front and rear)
        // Assume 50/50 brake distribution
        let fx_requested = requested_ax * m / 2.0;

        // Apply friction circle to each tire
        // The peak force D represents the maximum grip available
        let (fx_f, fy_f) = Self::apply_friction_circle(fx_requested, fy_f_raw, self.tire_front.d);
        let (fx_r, fy_r) = Self::apply_friction_circle(fx_requested, fy_r_raw, self.tire_rear.d);

        // Tire scrub drag: when tires slip laterally, friction opposes forward motion
        // Fx_scrub = -|Fy| * sin(|α|) for each tire
        let scrub_f = -fy_f.abs() * alpha_f.sin().abs();
        let scrub_r = -fy_r.abs() * alpha_r.sin().abs();
        let scrub_drag = (scrub_f + scrub_r) / m;

        // Actual longitudinal acceleration from tire forces plus scrub drag
        let actual_ax = (fx_f + fx_r) / m + scrub_drag;

        // Equations of motion (Euler integration)
        // dvy/dt = (Fy_f + Fy_r) / m - vx * r - C_lat * vy / m
        let dvy = (fy_f + fy_r) / m - vx * yaw_rate - Self::LATERAL_DAMPING * vy / m;
        // dr/dt = (lf * Fy_f - lr * Fy_r - C_yaw * r) / Iz
        let dr = (lf * fy_f - lr * fy_r - Self::YAW_DAMPING * yaw_rate) / iz;

        let new_vy = vy + dvy * dt;
        let new_yaw_rate = yaw_rate + dr * dt;

        (new_yaw_rate, new_vy, actual_ax)
    }

    /// Step dynamics using linear tire model (cornering stiffness)
    ///
    /// Uses linearized tire forces: Fy = -Cf * α
    /// This model doesn't allow slip-out but provides stable, predictable handling.
    ///
    /// Returns (new_yaw_rate, new_vy, actual_ax)
    pub fn step_linear(
        &self,
        yaw_rate: f32,
        vy: f32,
        steering: f32,
        requested_ax: f32,
        dt: f32,
    ) -> (f32, f32, f32) {
        let VehicleParams { mass: m, iz, lf, lr, cf, cr } = self.params;
        let vx = self.vx.max(0.1);

        // Compute slip angles
        let alpha_f = self.front_slip_angle(steering, vy, yaw_rate);
        let alpha_r = self.rear_slip_angle(vy, yaw_rate);

        // Linear tire forces: Fy = -C * α (negative because force opposes slip)
        let fy_f = -cf * alpha_f;
        let fy_r = -cr * alpha_r;

        // Equations of motion (Euler integration)
        // dvy/dt = (Fy_f + Fy_r) / m - vx * r
        let dvy = (fy_f + fy_r) / m - vx * yaw_rate;
        // dr/dt = (lf * Fy_f - lr * Fy_r) / Iz
        let dr = (lf * fy_f - lr * fy_r) / iz;

        let new_vy = vy + dvy * dt;
        let new_yaw_rate = yaw_rate + dr * dt;

        // Linear model passes through full acceleration (no friction circle)
        (new_yaw_rate, new_vy, requested_ax)
    }

    /// Compute blending factor between kinematic (0.0) and dynamic (1.0) models
    fn dynamic_blend_factor(&self) -> f32 {
        let low = Self::KINEMATIC_THRESHOLD - Self::BLEND_ZONE * 0.5;
        let high = Self::KINEMATIC_THRESHOLD + Self::BLEND_ZONE * 0.5;

        if self.vx <= low {
            0.0 // Pure kinematic
        } else if self.vx >= high {
            1.0 // Pure dynamic
        } else {
            // Smooth blend using smoothstep
            let t = (self.vx - low) / (high - low);
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Update longitudinal velocity based on acceleration
    ///
    /// vx_new = vx + ax * dt
    pub fn step_longitudinal(&mut self, acceleration: f32, dt: f32) {
        self.vx = (self.vx + acceleration * dt).max(0.0); // Allow zero velocity
    }
}

/// Full vehicle state including position and heading in global frame
#[derive(Debug, Clone, Copy, Default)]
pub struct VehicleState {
    /// X position in global frame [m]
    pub x: f32,
    /// Y position in global frame [m]
    pub y: f32,
    /// Heading angle (yaw) [rad]
    pub yaw: f32,
    /// Yaw rate [rad/s]
    pub yaw_rate: f32,
    /// Longitudinal velocity [m/s]
    pub vx: f32,
    /// Lateral velocity [m/s]
    pub vy: f32,
}

impl VehicleState {
    /// Create a new vehicle state at the origin
    pub fn new() -> Self {
        Self {
            vx: 1.0, // Start with some velocity
            ..Default::default()
        }
    }

    /// Convert to Vector4 format [x, y, yaw, vx] for compatibility with existing code
    pub fn to_vector4(&self) -> Vector4 {
        vector![self.x, self.y, self.yaw, self.vx]
    }

    /// Step the vehicle state using bicycle model dynamics
    ///
    /// Uses kinematic model at low speeds and dynamic model at high speeds,
    /// with smooth blending in between to avoid discontinuities.
    ///
    /// For Pacejka tire model: friction circle limits combined braking and steering forces.
    /// For Linear tire model: stable handling without slip-out.
    pub fn step(&mut self, model: &mut BicycleModel, steering: f32, acceleration: f32, dt: f32) {
        // Get blending factor: 0.0 = kinematic, 1.0 = dynamic
        let alpha = model.dynamic_blend_factor();

        // Kinematic model: yaw_rate from steering geometry, no lateral velocity
        let kinematic_yaw_rate = model.kinematic_yaw_rate(steering);
        let kinematic_vy = 0.0;

        // Dynamic model: full lateral dynamics with selected tire model
        let (dyn_yaw_rate, dyn_vy, actual_ax) = if alpha > 0.0 {
            let (new_yaw_rate, new_vy, ax) = match model.tire_model {
                TireModel::Pacejka => {
                    model.step_nonlinear(self.yaw_rate, self.vy, steering, acceleration, dt)
                }
                TireModel::Linear => {
                    model.step_linear(self.yaw_rate, self.vy, steering, acceleration, dt)
                }
            };

            // Check for NaN/Inf
            if new_yaw_rate.is_finite() && new_vy.is_finite() && ax.is_finite() {
                (new_yaw_rate, new_vy, ax)
            } else {
                (kinematic_yaw_rate, 0.0, acceleration) // Fall back to kinematic if unstable
            }
        } else {
            (kinematic_yaw_rate, kinematic_vy, acceleration)
        };

        // Blend acceleration between kinematic (full) and dynamic (friction-limited)
        let blended_ax = acceleration * (1.0 - alpha) + actual_ax * alpha;

        // Update longitudinal velocity with friction-limited acceleration
        self.vx = (self.vx + blended_ax * dt).max(0.5); // Keep minimum speed
        model.vx = self.vx;

        // Blend between kinematic and dynamic
        let blended_yaw_rate = kinematic_yaw_rate * (1.0 - alpha) + dyn_yaw_rate * alpha;
        let blended_vy = kinematic_vy * (1.0 - alpha) + dyn_vy * alpha;

        // Apply saturation limits
        self.yaw_rate = blended_yaw_rate.clamp(-BicycleModel::MAX_YAW_RATE, BicycleModel::MAX_YAW_RATE);
        self.vy = blended_vy.clamp(-BicycleModel::MAX_VY, BicycleModel::MAX_VY);

        // Update heading
        self.yaw += self.yaw_rate * dt;

        // Update global position
        // Transform velocities from body to global frame
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();
        self.x += (self.vx * cos_yaw - self.vy * sin_yaw) * dt;
        self.y += (self.vx * sin_yaw + self.vy * cos_yaw) * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bicycle_model_creation() {
        let model = BicycleModel::default();
        assert!(model.vx > 0.0);
        assert!(model.params.mass > 0.0);
    }

    #[test]
    fn test_discrete_model() {
        let model = BicycleModel::default();
        let dt = 0.01;
        let (Ad, Bd) = model.discrete_model(dt);

        // Check dimensions
        assert_eq!(Ad.nrows(), NX_LAT);
        assert_eq!(Ad.ncols(), NX_LAT);
        assert_eq!(Bd.nrows(), NX_LAT);
        assert_eq!(Bd.ncols(), NU_LAT);

        println!("Ad = {:?}", Ad);
        println!("Bd = {:?}", Bd);
    }

    #[test]
    fn test_vehicle_state_step() {
        let mut model = BicycleModel::default();
        model.vx = 5.0;

        let mut state = VehicleState::new();
        state.vx = 5.0;

        let dt = 0.01;
        let steering = 0.1; // 0.1 rad steering
        let acceleration = 0.0;

        // Step a few times
        for _ in 0..100 {
            state.step(&mut model, steering, acceleration, dt);
        }

        println!("Final state: x={:.2}, y={:.2}, yaw={:.2}°",
                 state.x, state.y, state.yaw.to_degrees());

        // With positive steering, vehicle should turn
        assert!(state.yaw.abs() > 0.0);
    }

    #[test]
    fn test_pacejka_magic_formula() {
        let tire = TireParams::default();

        // Force should be zero at zero slip
        let force_zero = tire.magic_formula(0.0);
        assert!(force_zero.abs() < 1e-5);

        // Force should increase with slip angle initially
        let force_small = tire.magic_formula(0.05);
        assert!(force_small > 0.0);

        // Force should be opposite for negative slip
        let force_neg = tire.magic_formula(-0.05);
        assert!(force_neg < 0.0);
        assert!((force_neg.abs() - force_small.abs()).abs() < 1e-5);

        // Force should saturate at large slip angles (not exceed peak D)
        let force_large = tire.magic_formula(0.5);
        assert!(force_large.abs() <= tire.d * 1.1); // Allow small overshoot

        // Force should decrease after peak (slip-out behavior)
        let force_very_large = tire.magic_formula(1.0);
        println!("Force at 0 slip: {:.2} N", force_zero);
        println!("Force at 0.05 rad: {:.2} N", force_small);
        println!("Force at 0.5 rad: {:.2} N", force_large);
        println!("Force at 1.0 rad: {:.2} N (should be less than peak)", force_very_large);
    }

    #[test]
    fn test_vehicle_state_step_pacejka() {
        let params = VehicleParams::default();
        let tire_front = TireParams::default();
        let tire_rear = TireParams::with_peak_force(3600.0); // Less rear grip for oversteer
        let mut model = BicycleModel::with_tires(params, 15.0, tire_front, tire_rear);

        let mut state = VehicleState::new();
        state.vx = 15.0; // High speed for drifting

        let dt = 0.01;
        let steering = 0.3; // Large steering angle
        let acceleration = 0.0;

        // Step a few times
        for _ in 0..100 {
            state.step(&mut model, steering, acceleration, dt);
        }

        println!("Pacejka Final state: x={:.2}, y={:.2}, yaw={:.2}°, vy={:.2}",
                 state.x, state.y, state.yaw.to_degrees(), state.vy);

        // With positive steering, vehicle should turn
        assert!(state.yaw.abs() > 0.0);
        // With Pacejka and large steering, lateral velocity should develop (drifting)
        assert!(state.vy.abs() > 0.0);
    }

    #[test]
    fn test_vehicle_state_step_linear() {
        let mut model = BicycleModel::default().with_tire_model(TireModel::Linear);
        model.vx = 15.0;

        let mut state = VehicleState::new();
        state.vx = 15.0; // High speed

        let dt = 0.01;
        let steering = 0.3; // Large steering angle
        let acceleration = 0.0;

        // Step a few times
        for _ in 0..100 {
            state.step(&mut model, steering, acceleration, dt);
        }

        println!("Linear Final state: x={:.2}, y={:.2}, yaw={:.2}°, vy={:.2}",
                 state.x, state.y, state.yaw.to_degrees(), state.vy);

        // With positive steering, vehicle should turn
        assert!(state.yaw.abs() > 0.0);
        // Linear model still develops lateral velocity but should remain stable
        // (no slip-out behavior like Pacejka)
    }
}
