pub mod inverted_pendulum;
pub mod vehicle;

use crate::prelude::*;

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimSub, ToTypenum};

/// Trait for providing a discrete-time state-space model
pub trait StateSpace<const N: usize, const M: usize, S = f32> {
    fn model(&self, dt: S) -> (Mat<N, N, S>, Mat<N, M, S>);
}

/// Trait for providing LQR implementation
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
    fn Q(&self) -> Mat<N, N>;
    fn R(&self) -> Mat<M, M>;
    fn epsilon(&self) -> f32;
    fn max_iter(&self) -> u32;
    fn control(&self, x: Vector<N>, dt: f32) -> Vector<M> {
        let (Ad, Bd) = self.model(dt);
        let K = self.dlqr(Ad, Bd);
        let u = -K * x;
        u
    }

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

    /// Solve the discrete time LQR controller.
    ///
    /// x[k+1] = A x[k] + B u[k]
    /// cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
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
        return P;
    }
}
