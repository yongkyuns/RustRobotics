#![macro_use]

use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::dimension::{DimAdd, DimSum};
use nalgebra::storage::{Storage, StorageMut};
use nalgebra::{ArrayStorage, Const};
use nalgebra::{DefaultAllocator, OMatrix};
use nalgebra::{Dim, Matrix, Scalar};
use num_traits::Zero;

pub use core::f32::consts::PI;
pub use core::f32::consts::TAU;

pub type Matrix1 = nalgebra::Matrix1<f32>;
pub type Matrix2 = nalgebra::Matrix2<f32>;
pub type Matrix3 = nalgebra::Matrix3<f32>;
pub type Matrix4 = nalgebra::Matrix4<f32>;
pub type Matrix5 = nalgebra::Matrix5<f32>;
pub type Vector2 = nalgebra::Vector2<f32>;
pub type Vector3 = nalgebra::Vector3<f32>;
pub type Vector4 = nalgebra::Vector4<f32>;
pub type RowVector4 = nalgebra::RowVector4<f32>;

pub type Mat<const M: usize, const N: usize, S = f32> =
    Matrix<S, Const<M>, Const<N>, ArrayStorage<S, M, N>>;

pub type Vector<const M: usize, S = f32> = Matrix<S, Const<M>, Const<1>, ArrayStorage<S, M, 1>>;

pub type RowVector<const M: usize, S = f32> = Matrix<S, Const<1>, Const<M>, ArrayStorage<S, 1, M>>;

pub trait MatrixUtil {
    fn get_diagonal(&self, index: usize) -> f32;
}

impl MatrixUtil for Matrix2 {
    fn get_diagonal(&self, index: usize) -> f32 {
        *self.diagonal().get(index).unwrap()
    }
}

pub trait Point {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
}

impl Point for Vector2 {
    fn x(&self) -> f32 {
        *self.get(0).expect("Cannot get 1st element of Vector2")
    }
    fn y(&self) -> f32 {
        *self.get(1).expect("Cannot get 2nd element of Vector2")
    }
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

pub trait Block<T> {
    type Rows: Dim;
    type Cols: Dim;

    fn shape(&self) -> (Self::Rows, Self::Cols);

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>;
}

impl<T, R, C, S> Block<T> for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    type Rows = R;
    type Cols = C;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        self.data.shape()
    }

    fn populate<S2>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S2>)
    where
        T: Scalar,
        S2: StorageMut<T, Self::Rows, Self::Cols>,
    {
        output.copy_from(self)
    }
}

pub struct Horizontal<X>(pub X);
pub struct Vertical<X>(pub X);
pub struct Diagonal<X>(pub X);

impl<T, B> Block<T> for Horizontal<(B,)>
where
    B: Block<T>,
{
    type Rows = B::Rows;
    type Cols = B::Cols;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        self.0 .0.shape()
    }

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>,
    {
        self.0 .0.populate(output);
    }
}

impl<T, B1, B2> Block<T> for Horizontal<(B1, B2)>
where
    B1: Block<T>,
    B2: Block<T>,
    B1::Cols: DimAdd<B2::Cols>,
    ShapeConstraint: SameNumberOfRows<B1::Rows, B2::Rows>,
{
    type Rows = <ShapeConstraint as SameNumberOfRows<B1::Rows, B2::Rows>>::Representative;
    type Cols = DimSum<B1::Cols, B2::Cols>;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        let (r1, c1) = self.0 .0.shape();
        let (_, c2) = self.0 .1.shape();
        let r = <Self::Rows as Dim>::from_usize(r1.value());
        let c = c1.add(c2);
        (r, c)
    }

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>,
    {
        assert_eq!(self.0 .0.shape().0.value(), self.0 .1.shape().0.value());

        let mut output_0 = output.generic_slice_mut((0, 0), self.0 .0.shape());
        self.0 .0.populate(&mut output_0);

        let offset = self.0 .0.shape().1.value();
        let mut output_1 = output.generic_slice_mut((0, offset), self.0 .1.shape());
        self.0 .1.populate(&mut output_1);
    }
}

impl<T, B1, B2> Block<T> for Vertical<(B1, B2)>
where
    B1: Block<T>,
    B2: Block<T>,
    B1::Rows: DimAdd<B2::Rows>,
    ShapeConstraint: SameNumberOfColumns<B1::Cols, B2::Cols>,
{
    type Rows = DimSum<B1::Rows, B2::Rows>;
    type Cols = <ShapeConstraint as SameNumberOfColumns<B1::Cols, B2::Cols>>::Representative;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        let (r1, c1) = self.0 .0.shape();
        let (r2, _) = self.0 .1.shape();
        let r = r1.add(r2);
        let c = <Self::Cols as Dim>::from_usize(c1.value());
        (r, c)
    }

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>,
    {
        assert_eq!(self.0 .0.shape().1.value(), self.0 .1.shape().1.value());

        let mut output_0 = output.generic_slice_mut((0, 0), self.0 .0.shape());
        self.0 .0.populate(&mut output_0);

        let offset = self.0 .0.shape().0.value();
        let mut output_1 = output.generic_slice_mut((offset, 0), self.0 .1.shape());
        self.0 .1.populate(&mut output_1);
    }
}

impl<T, B> Block<T> for Diagonal<(B,)>
where
    B: Block<T>,
{
    type Rows = B::Rows;
    type Cols = B::Cols;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        self.0 .0.shape()
    }

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>,
    {
        self.0 .0.populate(output);
    }
}

impl<T, B1, B2> Block<T> for Diagonal<(B1, B2)>
where
    B1: Block<T>,
    B2: Block<T>,
    B1::Cols: DimAdd<B2::Cols>,
    B1::Rows: DimAdd<B2::Rows>,
{
    type Rows = DimSum<B1::Rows, B2::Rows>;
    type Cols = DimSum<B1::Cols, B2::Cols>;

    fn shape(&self) -> (Self::Rows, Self::Cols) {
        let (r1, c1) = self.0 .0.shape();
        let (r2, c2) = self.0 .1.shape();
        let r = r1.add(r2);
        let c = c1.add(c2);
        (r, c)
    }

    fn populate<S>(&self, output: &mut Matrix<T, Self::Rows, Self::Cols, S>)
    where
        T: Scalar,
        S: StorageMut<T, Self::Rows, Self::Cols>,
    {
        let mut output_0 = output.generic_slice_mut((0, 0), self.0 .0.shape());
        self.0 .0.populate(&mut output_0);

        let offset = (self.0 .0.shape().0.value(), self.0 .0.shape().1.value());
        let mut output_1 = output.generic_slice_mut(offset, self.0 .1.shape());
        self.0 .1.populate(&mut output_1);
    }
}

pub fn allocate_block_output<T, B>(block: &B) -> OMatrix<T, B::Rows, B::Cols>
where
    T: Scalar + Zero,
    B: Block<T>,
    DefaultAllocator: Allocator<T, B::Rows, B::Cols>,
{
    let (rows, cols) = block.shape();
    OMatrix::zeros_generic(rows, cols)
}

#[macro_export]
macro_rules! block {
    ($( $( $x: expr ),*);*) => {
        {
			use crate::matrix::{Horizontal, Vertical, allocate_block_output, Block};

            let block_expression = Vertical(($(Horizontal(($($x),*,))),*));
            let mut output = allocate_block_output(&block_expression);
            block_expression.populate(&mut output);
            output
        }
    }
}

#[macro_export]
macro_rules! hstack_pair {
    ($x:expr, $y:expr) => {{
        use crate::util::{allocate_block_output, Block, Horizontal};

        let block_expression = Horizontal(($x, $y));
        let mut output = allocate_block_output(&block_expression);
        block_expression.populate(&mut output);
        output
    }};
}

#[macro_export]
macro_rules! hstack {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {{
        let head = hstack_pair!($first, $second);
        $(let head = hstack_pair!(head, hstack!($rest));)*
        head
    }};
    ($tail:expr $(,)? ) => {
        $tail
    };
}

#[macro_export]
macro_rules! vstack_pair {
    ($x:expr, $y:expr) => {{
        use crate::util::{allocate_block_output, Block, Vertical};

        let block_expression = Vertical(($x, $y));
        let mut output = allocate_block_output(&block_expression);
        block_expression.populate(&mut output);
        output
    }};
}

#[macro_export]
macro_rules! vstack {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {{
        let head = vstack_pair!($first, $second);
        $(let head = vstack_pair!(head, vstack!($rest));)*
        head
    }};
    ($tail:expr $(,)? ) => {
        $tail
    };
}

#[macro_export]
macro_rules! block_diag_pair {
    ($x:expr, $y:expr) => {{
        use crate::util::{allocate_block_output, Block, Diagonal};

        let block_expression = Diagonal(($x, $y));
        let mut output = allocate_block_output(&block_expression);
        block_expression.populate(&mut output);
        output
    }};
}

#[macro_export]
macro_rules! block_diag {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {{
        let head = block_diag_pair!($first, $second);
        $(let head = block_diag_pair!(head, block_diag!($rest));)*
        head
    }};
    ($tail:expr $(,)? ) => {
        $tail
    };
}

#[macro_export]
macro_rules! diag {
    (
        $(
            $x:expr
        ),*
    )=>{{
        use nalgebra::{ SMatrix, vector };
        SMatrix::from_diagonal(&vector![$($x),*])
    }};
}

#[macro_export]
macro_rules! eye {
    (
        $size:expr
    ) => {{
        use nalgebra::{SMatrix, SVector};
        SMatrix::from_diagonal(&SVector::from([1.0; $size]))
    }};
    (
        $size:expr,
        $off_diag:expr
    ) => {{
        // use crate::S;
        let mut out = zeros!($size, $size);
        let offset = (($off_diag as f32).abs() as usize);
        if $off_diag > 0 {
            for i in offset..$size {
                out[(i - offset, i)] = 1.0;
            }
        } else {
            for i in 0..$size - offset {
                out[(i + offset, i)] = 1.0;
            }
        }
        out
    }};
}

#[macro_export]
macro_rules! dot {
    ($mat:expr, $vec:expr) => {{
        let mut out = ($vec).clone_owned();
        let v = ($vec).transpose();
        for i in 0..out.nrows() {
            out[i] = ($mat).row(i).dot(&v);
        }
        out
    }};
}

#[macro_export]
macro_rules! kron {
    ($a:expr, $b:expr) => {{
        ($a).kronecker(&$b)
    }};
}

#[macro_export]
macro_rules! zeros {
    (
        $cols:expr
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<f32, 1, $cols>::zeros()
    }};
    (
        $rows:expr,
        $cols:expr
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<f32, $rows, $cols>::zeros()
    }};
    (
        $rows:expr,
        $cols:expr,
        $ty:ty
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<$ty, $rows, $cols>::zeros()
    }};
}

#[macro_export]
macro_rules! ones {
    (
        $cols:expr
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<f32, 1, $cols>::from_element(1.0)
    }};
    (
        $rows:expr,
        $cols:expr
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<f32, $rows, $cols>::from_element(1.0)
    }};
    (
        $rows:expr,
        $cols:expr,
        $ty:ty
    ) => {{
        use nalgebra::SMatrix;
        SMatrix::<$ty, $rows, $cols>::from_element(1.0 as $ty)
    }};
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
