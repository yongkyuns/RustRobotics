//! Fixed-size robotics math and small scalar helpers.
//!
//! Controllers and particle filtering use stack-algebra directly. Runtime-sized
//! SLAM matrices remain explicitly owned by the SLAM modules during migration.

pub use core::f32::consts::{PI, TAU};

pub type Mat<const M: usize, const N: usize, T = f32> = stack_algebra::Matrix<M, N, T>;
pub type Vector<const M: usize, T = f32> = stack_algebra::Vector<M, T>;
pub type RowVector<const M: usize, T = f32> = stack_algebra::RowVector<M, T>;
pub type Matrix1 = Mat<1, 1>;
pub type Matrix2 = Mat<2, 2>;
pub type Matrix3 = Mat<3, 3>;
pub type Matrix4 = Mat<4, 4>;
pub type Matrix5 = Mat<5, 5>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Vector4 = Vector<4>;
pub type RowVector4 = RowVector<4>;

pub trait MatrixUtil {
    fn get_diagonal(&self, index: usize) -> f32;
}

impl MatrixUtil for Matrix2 {
    fn get_diagonal(&self, index: usize) -> f32 {
        self[(index, index)]
    }
}

pub trait Point {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
}

impl Point for Vector2 {
    fn x(&self) -> f32 {
        self[0]
    }
    fn y(&self) -> f32 {
        self[1]
    }
}

/// Builds a diagonal matrix, including one-element costs and empty matrices.
pub fn diagonal<const N: usize, T: Copy + num_traits::Zero>(values: [T; N]) -> Mat<N, N, T> {
    Mat::from_fn(|row, column| {
        if row == column {
            values[row]
        } else {
            T::zero()
        }
    })
}

/// Largest coefficient magnitude; used for the existing Riccati stopping rule.
pub fn max_abs<const M: usize, const N: usize>(matrix: &Mat<M, N>) -> f32 {
    matrix.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

pub fn exp(u: f32) -> f32 {
    u.exp()
}

pub fn sqrt(u: f32) -> f32 {
    u.sqrt()
}

pub fn hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

pub fn sin(u: f32) -> f32 {
    u.sin()
}

pub fn cos(u: f32) -> f32 {
    u.cos()
}

/// A robotics state/control vector is always a column, including comma literals.
/// This deliberately preserves RustRobotics' public convention; stack-algebra's
/// own comma-separated `vector!` constructs a row instead.
#[macro_export]
macro_rules! vector {
    ($($value:expr),* $(,)?) => {
        $crate::stack_algebra::Matrix::from_columns([[$($value),*]])
    };
}

#[macro_export]
macro_rules! diag {
    ($($value:expr),* $(,)?) => { $crate::util::diagonal([$($value),*]) };
}

#[macro_export]
macro_rules! eye {
    ($size:expr) => {
        $crate::Mat::<{ $size }, { $size }>::eye()
    };
}

#[macro_export]
macro_rules! zeros {
    ($columns:expr) => {
        $crate::Mat::<1, { $columns }>::zeros()
    };
    ($rows:expr, $columns:expr) => {
        $crate::Mat::<{ $rows }, { $columns }>::zeros()
    };
    ($rows:expr, $columns:expr, $ty:ty) => {
        $crate::Mat::<{ $rows }, { $columns }, $ty>::zeros()
    };
}

#[macro_export]
macro_rules! ones {
    ($columns:expr) => {
        $crate::Mat::<1, { $columns }>::ones()
    };
    ($rows:expr, $columns:expr) => {
        $crate::Mat::<{ $rows }, { $columns }>::ones()
    };
    ($rows:expr, $columns:expr, $ty:ty) => {
        $crate::Mat::<{ $rows }, { $columns }, $ty>::ones()
    };
}

#[macro_export]
macro_rules! join {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {{
        let mut out = $first.to_string();
        let second = $second.to_string();
        out.push_str(&second);
        $(out.push_str(&join!($rest));)*
        out
    }};
    ($last:expr $(,)? ) => {{
        let out = $last.to_string();
        out
    }};
}

#[macro_export]
macro_rules! disp {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {
        println!("{}, {}", $first, $second);
        disp!($($rest),*);
    };
    ($single:expr $(,)? ) => {
        println!("{}", $single);
    };
    () => {};
}
