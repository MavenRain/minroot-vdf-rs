//! Polynomial reduction: Montgomery + upper modular reduction.
//!
//! After multiplication, the product has `2 * NUM_COEFFS - 1 = 33`
//! coefficients.  Reduction brings this back to `NUM_COEFFS = 17`
//! coefficients representing a value less than the modulus.
//!
//! This reference implementation converts to integer limbs, reduces
//! mod `p` via shift-and-subtract, and converts back to polynomial form.

use crate::bits_ext;
use crate::poly_mul::UnreducedProduct;
use crate::types::{PolySignal, HW_WORD_BITS};
use minroot_core::field::{Curve, FieldElement};
use minroot_core::polynomial::PolyElement;

/// Polynomial reducer: brings an [`UnreducedProduct`] back to
/// a [`PolySignal`] with 17 coefficients, modulo `p`.
///
/// This is parameterized by the curve since reduction is modular.
pub struct PolyReduce {
    curve: Curve,
}

impl PolyReduce {
    /// Creates a reducer for the given curve.
    #[must_use]
    pub fn new(curve: Curve) -> Self {
        Self { curve }
    }

    /// Reduces an unreduced product to a polynomial-form field element.
    #[must_use]
    pub fn reduce(&self, product: &UnreducedProduct) -> PolySignal {
        let wide = product_to_limbs(product);
        let reduced = reduce_wide_limbs(&wide, self.curve);
        FieldElement::from_limbs(reduced, self.curve)
            .map(|fe| PolySignal::from_poly_element(&PolyElement::from_field(fe)))
            .unwrap_or_default()
    }
}

/// Converts an unreduced product to 8 little-endian u64 limbs (512 bits).
///
/// Each product coefficient contributes at bit offset `k * WORD_BITS`.
/// For each limb `i`, we sum the contributions from all coefficients
/// whose bit window overlaps that limb.
#[allow(clippy::cast_possible_truncation)]
fn product_to_limbs(product: &UnreducedProduct) -> [u64; 8] {
    core::array::from_fn(|i| {
        product
            .coeffs()
            .iter()
            .enumerate()
            .fold(0u64, |acc, (k, coeff)| {
                let coeff_val = bits_ext::to_u128(*coeff);
                let bit_offset = k * HW_WORD_BITS;
                let limb_idx = bit_offset / 64;
                let bit_idx = bit_offset % 64;

                // The coefficient's low portion lands in `limb_idx`,
                // and (if it straddles) the high portion in `limb_idx + 1`.
                if limb_idx == i {
                    acc.wrapping_add((coeff_val << bit_idx) as u64)
                } else if limb_idx + 1 == i && bit_idx + HW_WORD_BITS > 64 {
                    acc.wrapping_add((coeff_val >> (64 - bit_idx)) as u64)
                } else {
                    acc
                }
            })
    })
}

/// Reduces 512-bit limbs modulo `p` to 256-bit limbs.
///
/// Uses shift-and-subtract.
fn reduce_wide_limbs(wide: &[u64; 8], curve: Curve) -> [u64; 4] {
    let modulus = curve.modulus();
    let total_bits = 8 * 64;

    (0..total_bits).rev().fold([0u64; 4], |acc, bit| {
        let shifted = shift_left_one_4(&acc);
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        let incoming = wide
            .get(limb_idx)
            .map_or(0, |limb| (limb >> bit_idx) & 1);
        let with_bit = [
            shifted[0] | incoming,
            shifted[1],
            shifted[2],
            shifted[3],
        ];
        if gte_modulus(&with_bit, &modulus) {
            sub_4(&with_bit, &modulus)
        } else {
            with_bit
        }
    })
}

/// Shifts 4 limbs left by one bit.
fn shift_left_one_4(a: &[u64; 4]) -> [u64; 4] {
    core::array::from_fn(|i| {
        let current = a.get(i).map_or(0, |v| v << 1);
        let from_below = if i > 0 {
            a.get(i - 1).map_or(0, |v| v >> 63)
        } else {
            0
        };
        current | from_below
    })
}

/// Returns `true` if `a >= b` (4-limb comparison).
fn gte_modulus(a: &[u64; 4], b: &[u64; 4]) -> bool {
    a.iter()
        .zip(b.iter())
        .rev()
        .fold(core::cmp::Ordering::Equal, |ord, (&ai, &bi)| match ord {
            core::cmp::Ordering::Equal => ai.cmp(&bi),
            other => other,
        })
        != core::cmp::Ordering::Less
}

/// Subtracts two 4-limb numbers (assumes a >= b).
///
/// Uses a fold that accumulates both the propagating borrow and
/// the sequence of diff limbs.
#[allow(clippy::cast_possible_truncation)]
fn sub_4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (_, diffs) = a.iter().zip(b.iter()).fold(
        (0u128, Vec::<u64>::with_capacity(4)),
        |(borrow, diffs), (&ai, &bi)| {
            let d = u128::from(ai)
                .wrapping_sub(u128::from(bi))
                .wrapping_sub(borrow);
            let new_borrow = u128::from(d >> 127 != 0);
            let new_diffs = diffs.into_iter().chain(core::iter::once(d as u64)).collect();
            (new_borrow, new_diffs)
        },
    );
    core::array::from_fn(|i| diffs.get(i).copied().unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_mul::PolyMul;
    use crate::circuit::Combinational;
    use hdl_cat::bits::Bits;

    #[test]
    fn reduce_zero_product_is_zero() {
        let product = UnreducedProduct::default();
        let reducer = PolyReduce::new(Curve::Pallas);
        let result = reducer.reduce(&product);
        assert_eq!(result, PolySignal::default());
    }

    #[test]
    fn reduce_small_product() {
        // 3 * 5 = 15, which is < p, so reduction is identity
        let a = PolySignal::from_coeffs(core::array::from_fn(|i| {
            if i == 0 { Bits::new_wrapping(3u128) } else { Bits::new_wrapping(0u128) }
        }));
        let b = PolySignal::from_coeffs(core::array::from_fn(|i| {
            if i == 0 { Bits::new_wrapping(5u128) } else { Bits::new_wrapping(0u128) }
        }));

        let product = PolyMul::eval((a, b));
        let reducer = PolyReduce::new(Curve::Pallas);
        let result = reducer.reduce(&product);

        assert_eq!(bits_ext::to_u128(result.coeff(0)), 15);
    }

    #[test]
    fn reduce_matches_field_mul() {
        let x = FieldElement::from_u64(12345, Curve::Pallas);
        let y = FieldElement::from_u64(67890, Curve::Pallas);
        let expected = x * y;

        let px = PolyElement::from_field(x);
        let py = PolyElement::from_field(y);
        let sx = PolySignal::from_poly_element(&px);
        let sy = PolySignal::from_poly_element(&py);

        let product = PolyMul::eval((sx, sy));
        let reducer = PolyReduce::new(Curve::Pallas);
        let result = reducer.reduce(&product);

        let expected_poly = PolyElement::from_field(expected);
        let expected_signal = PolySignal::from_poly_element(&expected_poly);
        assert_eq!(result, expected_signal);
    }
}
