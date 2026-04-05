//! Polynomial squaring (optimized self-product).
//!
//! Squaring is a special case of multiplication where both operands
//! are the same polynomial.  In the reference model, we delegate to
//! [`PolyMul`] for correctness.  A hardware-optimized version would
//! exploit the symmetry of self-products (upper triangle + doubled
//! off-diagonal terms).

use crate::circuit::Combinational;
use crate::poly_mul::{PolyMul, UnreducedProduct};
use crate::types::PolySignal;

/// Polynomial squarer: `a^2` in coefficient form.
///
/// Delegates to [`PolyMul`] with identical operands for correctness.
/// A synthesis-optimized version would exploit the symmetric partial
/// product matrix.
pub struct PolySqr;

impl Combinational for PolySqr {
    type Input = PolySignal;
    type Output = UnreducedProduct;

    fn eval(a: Self::Input) -> Self::Output {
        PolyMul::eval((a, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HW_COEFF_BITS;
    use rhdl::bits::Bits;

    #[test]
    fn sqr_matches_mul_self() {
        let coeffs = core::array::from_fn(|i| {
            if i < 3 {
                Bits::<{ HW_COEFF_BITS }>::from(u128::try_from(i + 1).unwrap_or(0))
            } else {
                Bits::from(0u128)
            }
        });
        let a = PolySignal::from_coeffs(coeffs);

        let sqr_result = PolySqr::eval(a);
        let mul_result = PolyMul::eval((a, a));

        sqr_result
            .coeffs()
            .iter()
            .zip(mul_result.coeffs().iter())
            .enumerate()
            .for_each(|(i, (s, m))| {
                assert_eq!(
                    crate::bits_ext::to_u128(*s),
                    crate::bits_ext::to_u128(*m),
                    "mismatch at coefficient {i}"
                );
            });
    }
}
