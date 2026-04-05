//! Polynomial multiplication (schoolbook) via partial product matrix.
//!
//! Computes the product of two polynomial-form field elements using
//! schoolbook multiplication.  Each pair of coefficients `(c_i, c_j)`
//! produces a partial product that contributes to output coefficient
//! `k = i + j`.
//!
//! The partial products are summed using the Wallace tree compressor
//! from [`crate::tree`].  The result is a double-width polynomial
//! (up to `2 * NUM_COEFFS - 1` coefficients) that must be reduced
//! by [`crate::poly_reduce`].

use rhdl::bits::Bits;

use crate::bits_ext;
use crate::circuit::Combinational;
use crate::tree;
use crate::types::{HW_NUM_COEFFS, PartialProduct, PolySignal, zero_pp};

/// Number of output coefficients in the unreduced product.
pub const PRODUCT_COEFFS: usize = 2 * HW_NUM_COEFFS - 1;

/// The unreduced product of two polynomial elements.
///
/// Has `2 * NUM_COEFFS - 1 = 33` coefficients, each up to
/// `2 * COEFF_BITS = 34` bits wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnreducedProduct {
    coeffs: [PartialProduct; PRODUCT_COEFFS],
}

impl Default for UnreducedProduct {
    fn default() -> Self {
        Self {
            coeffs: [zero_pp(); PRODUCT_COEFFS],
        }
    }
}

impl UnreducedProduct {
    /// Constructs from a coefficient array.
    #[must_use]
    pub fn from_coeffs(coeffs: [PartialProduct; PRODUCT_COEFFS]) -> Self {
        Self { coeffs }
    }

    /// Returns the coefficient array.
    #[must_use]
    pub fn coeffs(&self) -> &[PartialProduct; PRODUCT_COEFFS] {
        &self.coeffs
    }

    /// Returns a single coefficient, or [`zero_pp()`] if out of bounds.
    #[must_use]
    pub fn coeff(&self, i: usize) -> PartialProduct {
        self.coeffs.get(i).copied().unwrap_or(zero_pp())
    }
}

/// Polynomial multiplier: `a * b` in coefficient form.
///
/// Uses schoolbook multiplication with Wallace tree compression
/// of partial products.
pub struct PolyMul;

impl Combinational for PolyMul {
    type Input = (PolySignal, PolySignal);
    type Output = UnreducedProduct;

    fn eval((a, b): Self::Input) -> Self::Output {
        let coeffs: [PartialProduct; PRODUCT_COEFFS] = core::array::from_fn(|k| {
            // Collect partial products c_i * c_j where i + j == k
            let partials: Vec<PartialProduct> = (0..HW_NUM_COEFFS)
                .filter(|&i| k >= i && (k - i) < HW_NUM_COEFFS)
                .map(|i| {
                    let j = k - i;
                    widen_mul(a.coeff(i), b.coeff(j))
                })
                .collect();

            if partials.is_empty() {
                zero_pp()
            } else {
                tree::compress(&partials).resolve()
            }
        });

        UnreducedProduct::from_coeffs(coeffs)
    }
}

/// Multiplies two coefficients, widening the result.
///
/// `Bits<17> * Bits<17> -> Bits<34>`
fn widen_mul(a: crate::types::Coeff, b: crate::types::Coeff) -> PartialProduct {
    Bits::from(bits_ext::to_u128(a) * bits_ext::to_u128(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_by_zero_is_zero() {
        let a_coeffs = core::array::from_fn(|i| {
            Bits::<{ crate::types::HW_COEFF_BITS }>::from(u128::try_from(i + 1).unwrap_or(0))
        });
        let a = PolySignal::from_coeffs(a_coeffs);
        let zero = PolySignal::default();
        let product = PolyMul::eval((a, zero));
        product
            .coeffs()
            .iter()
            .for_each(|c| assert_eq!(bits_ext::to_u128(*c), 0));
    }

    #[test]
    fn multiply_small_values() {
        // Multiply (3) * (5) in polynomial form.
        let a_coeffs = core::array::from_fn(|i| {
            if i == 0 { Bits::from(3u128) } else { Bits::from(0u128) }
        });
        let b_coeffs = core::array::from_fn(|i| {
            if i == 0 { Bits::from(5u128) } else { Bits::from(0u128) }
        });

        let a = PolySignal::from_coeffs(a_coeffs);
        let b = PolySignal::from_coeffs(b_coeffs);
        let product = PolyMul::eval((a, b));

        assert_eq!(bits_ext::to_u128(product.coeff(0)), 15);
        // All other coefficients should be 0
        (1..PRODUCT_COEFFS).for_each(|i| {
            assert_eq!(bits_ext::to_u128(product.coeff(i)), 0);
        });
    }

    #[test]
    fn product_coefficient_count() {
        assert_eq!(PRODUCT_COEFFS, 33);
    }
}
