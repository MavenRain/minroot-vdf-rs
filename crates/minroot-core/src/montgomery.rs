//! Montgomery form conversion for field elements.
//!
//! The hardware operates in Montgomery domain with `R = 2^128`.
//! This module provides conversion between standard and Montgomery
//! representations.
//!
//! In Montgomery form, an element `a` is represented as `aR mod p`.
//! Montgomery multiplication: `MontMul(aR, bR) = abR mod p`.
//!
//! This reference implementation stores the **standard** value
//! internally and performs standard arithmetic for correctness.
//! The Montgomery representation is available via [`MontgomeryElement::to_mont_repr`]
//! for comparison with hardware outputs.

use crate::error::Error;
use crate::field::{Curve, FieldElement};
use core::ops;

/// Number of bits in the Montgomery constant R.
///
/// Matches the hardware parameter `LowerTriBits = 128`.
const MONT_BITS: usize = 128;

/// A field element that tracks both standard and Montgomery forms.
///
/// Internally stores the standard value.  Arithmetic is performed
/// in standard form for correctness.  The Montgomery representation
/// (`aR mod p`) is available for hardware comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MontgomeryElement {
    /// The standard (non-Montgomery) value.
    standard: FieldElement,
}

impl MontgomeryElement {
    /// Wraps a standard field element for Montgomery-domain operations.
    #[must_use]
    pub fn from_field(a: FieldElement) -> Self {
        Self { standard: a }
    }

    /// Returns the standard (non-Montgomery) field element.
    #[must_use]
    pub fn to_field(self) -> FieldElement {
        self.standard
    }

    /// Returns the Montgomery representation `aR mod p`.
    ///
    /// This is what the hardware stores internally.
    #[must_use]
    pub fn to_mont_repr(self) -> FieldElement {
        // Compute a * 2^128 mod p by repeated doubling.
        (0..MONT_BITS).fold(self.standard, |acc, _| acc + acc)
    }

    /// Constructs from a Montgomery representation `aR mod p`.
    ///
    /// Recovers the standard value `a` by halving 128 times
    /// (each halving computes `x * 2^{-1} mod p`).
    #[must_use]
    pub fn from_mont_repr(mont_repr: FieldElement) -> Self {
        let standard =
            (0..MONT_BITS).fold(mont_repr, |acc, _| halve_mod_p(acc));
        Self { standard }
    }

    /// Montgomery squaring.
    #[must_use]
    pub fn sqr(self) -> Self {
        self * self
    }

    /// Returns the curve.
    #[must_use]
    pub fn curve(&self) -> Curve {
        self.standard.curve()
    }

    /// The zero element.
    #[must_use]
    pub fn zero(curve: Curve) -> Self {
        Self {
            standard: FieldElement::zero(curve),
        }
    }

    /// The multiplicative identity.
    #[must_use]
    pub fn one(curve: Curve) -> Self {
        Self {
            standard: FieldElement::one(curve),
        }
    }

    /// Modular exponentiation.
    #[must_use]
    pub fn pow(self, exp: &[u64; 4], num_bits: usize) -> Self {
        Self {
            standard: self.standard.pow(exp, num_bits),
        }
    }

    /// Computes the fifth root.
    #[must_use]
    pub fn fifth_root(self) -> Self {
        Self {
            standard: self.standard.fifth_root(),
        }
    }

    /// Constructs from raw limbs in Montgomery representation.
    ///
    /// The limbs are interpreted as `aR mod p` and converted to
    /// the standard value internally.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] if the limbs are not less than the modulus.
    pub fn from_raw_mont_limbs(
        limbs: [u64; 4],
        curve: Curve,
    ) -> Result<Self, Error> {
        FieldElement::from_limbs(limbs, curve).map(Self::from_mont_repr)
    }
}

impl ops::Mul for MontgomeryElement {
    type Output = Self;

    /// Montgomery multiplication: produces `a * b` in the Montgomery domain.
    fn mul(self, rhs: Self) -> Self {
        debug_assert_eq!(self.standard.curve(), rhs.standard.curve());
        Self {
            standard: self.standard * rhs.standard,
        }
    }
}

/// Halves a field element modulo p.
///
/// Computes `a / 2 mod p`.  If `a` is odd, adds `p` first to make
/// it even, then shifts right.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn halve_mod_p(a: FieldElement) -> FieldElement {
    let limbs = a.limbs();
    let is_odd = limbs[0] & 1 == 1;
    let modulus = a.curve().modulus();

    // If odd, compute (a + p) / 2; if even, compute a / 2.
    // Since p is odd, a + p is even when a is odd.
    let (words, high_bit) = if is_odd {
        let mut result = [0u64; 4];
        let carry =
            limbs
                .iter()
                .zip(modulus.iter())
                .enumerate()
                .fold(0u128, |carry, (i, (&ai, &mi))| {
                    let sum = u128::from(ai) + u128::from(mi) + carry;
                    result[i] = sum as u64;
                    sum >> 64
                });
        (result, carry as u64)
    } else {
        (*limbs, 0u64)
    };

    // Shift right by 1
    let shifted: [u64; 4] = core::array::from_fn(|i| {
        let current = words[i] >> 1;
        let from_above = if i + 1 < 4 {
            words[i + 1] << 63
        } else {
            high_bit << 63
        };
        current | from_above
    });

    // Result is guaranteed < p since we started with a < p.
    FieldElement::from_limbs(shifted, a.curve())
        .unwrap_or_else(|_| FieldElement::zero(a.curve()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_to_from_field() {
        let x = FieldElement::from_u64(42, Curve::Pallas);
        let m = MontgomeryElement::from_field(x);
        let back = m.to_field();
        assert_eq!(back, x);
    }

    #[test]
    fn mont_repr_roundtrip() {
        let x = FieldElement::from_u64(42, Curve::Pallas);
        let m = MontgomeryElement::from_field(x);
        let repr = m.to_mont_repr();
        let recovered = MontgomeryElement::from_mont_repr(repr);
        assert_eq!(recovered.to_field(), x);
    }

    #[test]
    fn mont_mul_matches_field_mul() {
        let a = FieldElement::from_u64(123, Curve::Pallas);
        let b = FieldElement::from_u64(456, Curve::Pallas);
        let expected = a * b;

        let ma = MontgomeryElement::from_field(a);
        let mb = MontgomeryElement::from_field(b);
        let result = (ma * mb).to_field();
        assert_eq!(result, expected);
    }

    #[test]
    fn mont_fifth_root_roundtrip() {
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let mx = MontgomeryElement::from_field(x);
        let root = mx.fifth_root();
        let root5 = root * root * root * root * root;
        assert_eq!(root5.to_field(), x);
    }

    #[test]
    fn mont_repr_of_zero_is_zero() {
        let z = MontgomeryElement::zero(Curve::Pallas);
        assert_eq!(z.to_mont_repr(), FieldElement::zero(Curve::Pallas));
    }

    #[test]
    fn halve_double_roundtrip() {
        let x = FieldElement::from_u64(99, Curve::Pallas);
        let doubled = x + x;
        let halved = halve_mod_p(doubled);
        assert_eq!(halved, x);
    }

    #[test]
    fn halve_odd_value() {
        // 7 / 2 mod p = (7 + p) / 2
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let half = halve_mod_p(x);
        // half + half = 7 mod p
        assert_eq!(half + half, x);
    }
}
