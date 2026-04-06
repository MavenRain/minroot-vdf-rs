//! Test vector generation via [`Stream`].
//!
//! Produces streams of `MinRoot` test cases for hardware verification.
//! Each vector specifies initial `(x, y)` coordinates and an iteration
//! count.  The reference model in [`crate::verify`] computes expected
//! outputs for comparison with hdl-cat simulation results.
//!
//! [`Stream`]: comp_cat_rs::effect::stream::Stream

use comp_cat_rs::effect::stream::Stream;
use minroot_core::field::{Curve, FieldElement};

/// A single `MinRoot` test case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestVector {
    /// Initial `x` coordinate.
    pub x: FieldElement,
    /// Initial `y` coordinate.
    pub y: FieldElement,
    /// Number of iterations to run.
    pub iterations: u64,
}

impl TestVector {
    /// Constructs a test vector.
    #[must_use]
    pub fn new(x: FieldElement, y: FieldElement, iterations: u64) -> Self {
        debug_assert_eq!(x.curve(), y.curve());
        Self { x, y, iterations }
    }

    /// Returns the curve for this test.
    #[must_use]
    pub fn curve(&self) -> Curve {
        self.x.curve()
    }
}

/// Produces a stream of test vectors with small, incrementing seeds.
///
/// Generates `count` vectors with `x = seed + 1`, `y = seed + 2`,
/// all sharing the same `iterations` count.
#[must_use]
pub fn small_seeds(
    curve: Curve,
    count: usize,
    iterations: u64,
) -> Stream<core::convert::Infallible, TestVector> {
    let vectors: Vec<TestVector> = (0u64..)
        .take(count)
        .map(|s| {
            TestVector::new(
                FieldElement::from_u64(s + 1, curve),
                FieldElement::from_u64(s + 2, curve),
                iterations,
            )
        })
        .collect();
    Stream::from_vec(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infallible<T>(r: Result<T, core::convert::Infallible>) -> T {
        r.unwrap_or_else(|e| match e {})
    }

    #[test]
    fn small_seeds_produces_count() {
        let stream = small_seeds(Curve::Pallas, 5, 10);
        let vecs = infallible(stream.collect().run());
        assert_eq!(vecs.len(), 5);
    }

    #[test]
    fn small_seeds_iterations_correct() {
        let stream = small_seeds(Curve::Pallas, 3, 42);
        let vecs = infallible(stream.collect().run());
        assert!(vecs.iter().all(|v| v.iterations == 42));
    }

    #[test]
    fn small_seeds_on_correct_curve() {
        let stream = small_seeds(Curve::Vesta, 2, 5);
        let vecs = infallible(stream.collect().run());
        assert!(vecs.iter().all(|v| v.curve() == Curve::Vesta));
    }
}
