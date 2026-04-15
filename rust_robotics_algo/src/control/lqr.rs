//! Generic discrete-time Linear Quadratic Regulator support.
//!
//! This file is the mathematical core behind the repo's model-based pendulum
//! stabilizer. The problem it solves is:
//!
//! - dynamics: `x_(k+1) = A x_k + B u_k`
//! - cost: `J = sum_k (x_k^T Q x_k + u_k^T R u_k)`
//!
//! Intuition:
//!
//! - `Q` tells the controller which state errors are expensive
//! - `R` tells the controller how expensive control effort is
//! - the optimal infinite-horizon controller is a linear feedback law
//!   `u_k = -K x_k`
//!
//! The gain `K` is computed from the solution `P` of the discrete algebraic
//! Riccati equation (DARE):
//!
//! `P = A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A + Q`
//!
//! Once `P` is known, the optimal gain is:
//!
//! `K = (R + B^T P B)^(-1) B^T P A`
//!
//! In this repository the implementation uses an iterative fixed-point solve
//! for the DARE and a pseudo-inverse for numerical robustness.
use crate::control::StateSpace;
use crate::prelude::*;

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimSub, ToTypenum};

/// Trait for systems that can provide an LQR controller.
///
/// Implementors must provide:
///
/// - a discrete-time model through [`StateSpace`]
/// - state and control cost matrices `Q` and `R`
/// - numerical tolerances for the Riccati solve
///
/// This keeps the trait generic enough to work for any small fixed-size linear
/// system in the workspace, not just the inverted pendulum.
pub trait LQR<const N: usize, const M: usize, S = f32>: StateSpace<N, M>
where
    Const<N>: DimSub<Const<1_usize>>,
    Const<N>: ToTypenum,
    DefaultAllocator: Allocator<f32, Const<N>, <Const<N> as DimSub<Const<1_usize>>>::Output>,
    DefaultAllocator: Allocator<f32, <Const<N> as DimSub<Const<1_usize>>>::Output>,
    Const<M>: DimMin<Const<M>>,
    Const<M>: ToTypenum,
    <Const<M> as DimMin<Const<M>>>::Output: DimSub<Const<1_usize>>,
    DefaultAllocator:
        Allocator<f32, <<Const<M> as DimMin<Const<M>>>::Output as DimSub<Const<1_usize>>>::Output>,
    DefaultAllocator: Allocator<f32, <Const<M> as DimMin<Const<M>>>::Output, Const<M>>,
    DefaultAllocator: Allocator<f32, Const<M>, <Const<M> as DimMin<Const<M>>>::Output>,
    DefaultAllocator: Allocator<f32, <Const<M> as DimMin<Const<M>>>::Output>,
{
    /// Returns the state cost matrix in the quadratic objective.
    fn Q(&self) -> Mat<N, N>;
    /// Returns the control cost matrix in the quadratic objective.
    fn R(&self) -> Mat<M, M>;
    /// Returns the numerical tolerance used for pseudo-inverses and
    /// Riccati-iteration convergence.
    fn epsilon(&self) -> f32;
    /// Returns the maximum number of Riccati iterations.
    fn max_iter(&self) -> u32;

    /// Computes the LQR control input for the current state.
    ///
    /// This method:
    ///
    /// 1. discretizes the model at `dt`
    /// 2. solves the DARE
    /// 3. forms the optimal feedback gain `K`
    /// 4. returns `u = -Kx`
    ///
    /// For small fixed-size systems this is convenient and readable, though it
    /// recomputes `K` each call instead of caching it.
    fn control(&self, x: Vector<N>, dt: f32) -> Vector<M> {
        let (Ad, Bd) = self.model(dt);
        let K = self.dlqr(Ad, Bd);
        let u = -K * x;
        u
    }

    /// Computes the optimal discrete-time feedback gain `K`.
    ///
    /// After solving for the Riccati matrix `P`, the discrete-time LQR gain is:
    ///
    /// `K = (R + B^T P B)^(-1) B^T P A`
    fn dlqr(&self, A: Mat<N, N>, B: Mat<N, M>) -> Mat<M, N> {
        let P = self.solve_DARE(A, B);
        let R = self.R();

        // compute the LQR gain
        let BT = B.transpose();
        let inv = (BT * P * B + R)
            .pseudo_inverse(self.epsilon())
            .expect("Matrix inverse failed for DARE");
        let K = inv * (BT * P * A);

        let _eigen_vals = (A - B * K).eigenvalues();

        K
    }

    /// Solves the discrete algebraic Riccati equation by fixed-point iteration.
    ///
    /// The target problem is:
    ///
    /// `x[k+1] = A x[k] + B u[k]`
    ///
    /// `cost = sum x[k]^T Q x[k] + u[k]^T R u[k]`
    ///
    /// Starting from `P_0 = Q`, the implementation repeatedly applies the DARE
    /// update until the maximum absolute coefficient change falls below
    /// `epsilon`, or until `max_iter` is reached.
    ///
    /// # ref Bertsekas, p.151
    fn solve_DARE(&self, A: Mat<N, N>, B: Mat<N, M>) -> Mat<N, N> {
        let max_iter = self.max_iter();
        let eps = self.epsilon();
        let Q = self.Q();
        let R = self.R();
        let mut P = self.Q();
        let AT = A.transpose();
        let BT = B.transpose();

        for _ in 0..max_iter {
            let inv = (R + BT * P * B)
                .pseudo_inverse(eps)
                .expect("Matrix inverse failed for DARE");

            let Pn = (AT * P * A) - (AT * P * B) * inv * (BT * P * A) + Q;
            if (Pn - P).abs().amax() < eps {
                return Pn;
            }

            P = Pn;
        }
        P
    }
}
