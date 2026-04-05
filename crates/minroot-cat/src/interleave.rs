//! N-way interleaving as a product functor.
//!
//! The `MinRoot` hardware interleaves `N` independent fifth-root
//! computations around a shared pipeline ring.  On each clock cycle,
//! data from a different computation advances through the stages.
//!
//! Categorically, interleaving is a **product functor** `F(X) = X^N`:
//! it lifts each pipeline object to an N-tuple and each morphism to
//! a component-wise application with rotation.
//!
//! The rotation is a **natural transformation** `σ: F => F` that
//! cyclically shifts which computation is active.

use comp_cat_rs::foundation::kind::Kind;

/// The product functor `F(X) = [X; N]` for interleaving.
///
/// Each element of the array corresponds to one independent
/// fifth-root computation occupying a slot in the ring.
pub struct InterleaveK<const N: usize>;

impl<const N: usize> Kind for InterleaveK<N> {
    type F<A> = [A; N];
}

/// Rotation: cyclically shifts the active slot.
///
/// On each clock cycle, the pipeline advances data for the next
/// computation.  After `N` cycles, all computations have advanced
/// by one pipeline stage.
///
/// The rotation is a natural transformation (`[A; N] -> [A; N]`
/// for all `A: Copy`).  It commutes with any per-element operation,
/// so applying a pipeline stage then rotating is the same as rotating
/// then applying.
///
/// Note: this is not implemented via [`comp_cat_rs::foundation::nat_trans::NatTrans`]
/// because that trait's `transform<A>` has no bounds on `A`, and we
/// require `Copy` (moving individual array elements without `Copy` is
/// not possible in safe Rust without allocation).  The naturality
/// property holds for all `A: Copy`, which covers all hardware signal
/// types.
pub struct Rotate<const N: usize>;

impl<const N: usize> Rotate<N> {
    /// Rotate the array left by one position.
    ///
    /// `[a, b, c, d]` becomes `[b, c, d, a]`.
    #[must_use]
    pub fn apply<A: Copy>(arr: [A; N]) -> [A; N] {
        rotate_left(arr)
    }
}

/// Rotates an array left by one position.
///
/// `[a, b, c, d]` becomes `[b, c, d, a]`.
///
/// Requires `Copy` because we index into the source array
/// at `(i + 1) % N`, which is always in bounds.
fn rotate_left<A: Copy, const N: usize>(arr: [A; N]) -> [A; N] {
    core::array::from_fn(|i| arr[(i + 1) % N])
}

/// The interleave depth used by the default `MinRoot` hardware.
///
/// The original design uses `N = 3` datapaths.  `N` must divide
/// [`FIFTH_ROOT_CYCLES`](super::traced::FIFTH_ROOT_CYCLES) for
/// synchronized completion (258 / 3 = 86 exactly).
pub const DEFAULT_INTERLEAVE: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_identity_for_single() {
        let arr = [42];
        assert_eq!(Rotate::<1>::apply(arr), [42]);
    }

    #[test]
    fn rotate_left_by_one() {
        let arr = [1, 2, 3, 4];
        assert_eq!(Rotate::<4>::apply(arr), [2, 3, 4, 1]);
    }

    #[test]
    fn rotate_n_times_is_identity() {
        let arr = [10, 20, 30];
        let rotated = (0..3).fold(arr, |a, _| Rotate::<3>::apply(a));
        assert_eq!(rotated, arr);
    }

    #[test]
    fn default_interleave_divides_cycles() {
        assert_eq!(
            super::super::traced::FIFTH_ROOT_CYCLES % DEFAULT_INTERLEAVE,
            0
        );
    }
}
