//! Prime field arithmetic for the Pasta curves (Pallas and Vesta).
//!
//! Field elements are 256-bit integers stored as four 64-bit limbs
//! in little-endian order.  All arithmetic is modular with respect
//! to the chosen curve's prime modulus.
//!
//! # Moduli
//!
//! - **Pallas**: `0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001`
//! - **Vesta**:  `0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001`

use crate::error::Error;
use core::ops;

/// Number of 64-bit limbs in a field element.
const LIMBS: usize = 4;

/// The Pallas curve base field modulus, little-endian limbs.
const PALLAS_MODULUS: [u64; LIMBS] = [
    0x992d_30ed_0000_0001,
    0x2246_98fc_094c_f91b,
    0x0000_0000_0000_0000,
    0x4000_0000_0000_0000,
];

/// The Vesta curve base field modulus, little-endian limbs.
const VESTA_MODULUS: [u64; LIMBS] = [
    0x8c46_eb21_0000_0001,
    0x2246_98fc_0994_a8dd,
    0x0000_0000_0000_0000,
    0x4000_0000_0000_0000,
];

/// Identifies which Pasta curve modulus to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Curve {
    /// The Pallas curve base field.
    Pallas,
    /// The Vesta curve base field.
    Vesta,
}

impl Curve {
    /// Returns the modulus limbs for this curve.
    #[must_use]
    pub fn modulus(self) -> [u64; LIMBS] {
        match self {
            Self::Pallas => PALLAS_MODULUS,
            Self::Vesta => VESTA_MODULUS,
        }
    }

    /// Returns the fifth-root exponent `(4p - 3) / 5` for this curve,
    /// as little-endian limbs.
    #[must_use]
    pub fn fifth_root_exponent(self) -> [u64; LIMBS] {
        match self {
            Self::Pallas => PALLAS_FIFTH_ROOT_EXP,
            Self::Vesta => VESTA_FIFTH_ROOT_EXP,
        }
    }

    /// Number of significant bits in the fifth-root exponent.
    #[must_use]
    pub fn exponent_bits(self) -> usize {
        // Both Pallas and Vesta exponents are 254 bits.
        254
    }
}

/// Fifth-root exponent for Pallas: `(4p - 3) / 5`, little-endian limbs.
///
/// `0x333333333333333333333333333333334e9ee0c9a10a60e2e0f0f3f0cccccccd`
const PALLAS_FIFTH_ROOT_EXP: [u64; LIMBS] = [
    0xe0f0_f3f0_cccc_cccd,
    0x4e9e_e0c9_a10a_60e2,
    0x3333_3333_3333_3333,
    0x3333_3333_3333_3333,
];

/// Fifth-root exponent for Vesta: `(4p - 3) / 5`, little-endian limbs.
///
/// `0x333333333333333333333333333333334e9ee0c9a143ba4ad69f2280cccccccd`
const VESTA_FIFTH_ROOT_EXP: [u64; LIMBS] = [
    0xd69f_2280_cccc_cccd,
    0x4e9e_e0c9_a143_ba4a,
    0x3333_3333_3333_3333,
    0x3333_3333_3333_3333,
];

/// A 256-bit prime field element stored as four little-endian 64-bit limbs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldElement {
    limbs: [u64; LIMBS],
    curve: Curve,
}

impl core::fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "FieldElement({:?}, 0x{:016x}{:016x}{:016x}{:016x})",
            self.curve, self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0],
        )
    }
}

impl FieldElement {
    /// The additive identity (zero) for the given curve.
    #[must_use]
    pub fn zero(curve: Curve) -> Self {
        Self {
            limbs: [0; LIMBS],
            curve,
        }
    }

    /// The multiplicative identity (one) for the given curve.
    #[must_use]
    pub fn one(curve: Curve) -> Self {
        Self {
            limbs: [1, 0, 0, 0],
            curve,
        }
    }

    /// Constructs a field element from little-endian limbs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] if the value is not less than the modulus.
    pub fn from_limbs(limbs: [u64; LIMBS], curve: Curve) -> Result<Self, Error> {
        let modulus = curve.modulus();
        if gte_modulus(&limbs, &modulus) {
            Err(Error::OutOfRange {
                context: "from_limbs",
            })
        } else {
            Ok(Self { limbs, curve })
        }
    }

    /// Constructs a field element from a single `u64`, placed in the
    /// lowest limb.
    #[must_use]
    pub fn from_u64(val: u64, curve: Curve) -> Self {
        Self {
            limbs: [val, 0, 0, 0],
            curve,
        }
    }

    /// Returns the little-endian limb representation.
    #[must_use]
    pub fn limbs(&self) -> &[u64; LIMBS] {
        &self.limbs
    }

    /// Returns the curve this element belongs to.
    #[must_use]
    pub fn curve(&self) -> Curve {
        self.curve
    }

    /// Returns `true` if this element is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// Modular squaring: `self * self mod p`.
    #[must_use]
    pub fn sqr(self) -> Self {
        self * self
    }

    /// Modular exponentiation via square-and-multiply.
    ///
    /// The exponent is given as little-endian limbs with `num_bits`
    /// significant bits.
    #[must_use]
    pub fn pow(self, exp: &[u64; LIMBS], num_bits: usize) -> Self {
        (0..num_bits).rev().fold(Self::one(self.curve), |acc, i| {
            let squared = acc.sqr();
            let limb_idx = i / 64;
            let bit_idx = i % 64;
            if (exp[limb_idx] >> bit_idx) & 1 == 1 {
                squared * self
            } else {
                squared
            }
        })
    }

    /// Computes the fifth root: `self^((4p-3)/5) mod p`.
    #[must_use]
    pub fn fifth_root(self) -> Self {
        let exp = self.curve.fifth_root_exponent();
        let bits = self.curve.exponent_bits();
        self.pow(&exp, bits)
    }

    /// Extracts bit `i` from the element (bit 0 is LSB).
    #[must_use]
    pub fn bit(&self, i: usize) -> bool {
        let limb_idx = i / 64;
        let bit_idx = i % 64;
        if limb_idx < LIMBS {
            (self.limbs[limb_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }
}

impl ops::Add for FieldElement {
    type Output = Self;

    /// Modular addition: `self + rhs mod p`.
    fn add(self, rhs: Self) -> Self {
        debug_assert_eq!(self.curve, rhs.curve);
        let modulus = self.curve.modulus();
        let (sum, carry) = add_limbs(&self.limbs, &rhs.limbs);
        let result = if carry || gte_modulus(&sum, &modulus) {
            sub_limbs(&sum, &modulus).0
        } else {
            sum
        };
        Self {
            limbs: result,
            curve: self.curve,
        }
    }
}

impl ops::Sub for FieldElement {
    type Output = Self;

    /// Modular subtraction: `self - rhs mod p`.
    fn sub(self, rhs: Self) -> Self {
        debug_assert_eq!(self.curve, rhs.curve);
        let modulus = self.curve.modulus();
        let (diff, borrow) = sub_limbs(&self.limbs, &rhs.limbs);
        let result = if borrow {
            add_limbs(&diff, &modulus).0
        } else {
            diff
        };
        Self {
            limbs: result,
            curve: self.curve,
        }
    }
}

impl ops::Mul for FieldElement {
    type Output = Self;

    /// Modular multiplication: `self * rhs mod p`.
    ///
    /// Uses schoolbook multiplication followed by shift-and-subtract reduction.
    fn mul(self, rhs: Self) -> Self {
        debug_assert_eq!(self.curve, rhs.curve);
        let wide = mul_wide(&self.limbs, &rhs.limbs);
        let reduced = reduce_wide(&wide, &self.curve.modulus());
        Self {
            limbs: reduced,
            curve: self.curve,
        }
    }
}

// ── Multi-limb arithmetic helpers ──────────────────────────────────

/// Adds two 4-limb numbers, returning (result, carry).
#[allow(clippy::cast_possible_truncation)]
fn add_limbs(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> ([u64; LIMBS], bool) {
    let mut result = [0u64; LIMBS];
    let carry = a.iter().zip(b.iter()).enumerate().fold(
        0u128,
        |carry, (i, (&ai, &bi))| {
            let sum = u128::from(ai) + u128::from(bi) + carry;
            result[i] = sum as u64;
            sum >> 64
        },
    );
    (result, carry != 0)
}

/// Subtracts two 4-limb numbers, returning (result, borrow).
#[allow(clippy::cast_possible_truncation)]
fn sub_limbs(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> ([u64; LIMBS], bool) {
    let mut result = [0u64; LIMBS];
    let borrow = a.iter().zip(b.iter()).enumerate().fold(
        0u128,
        |borrow, (i, (&ai, &bi))| {
            let diff = u128::from(ai).wrapping_sub(u128::from(bi)).wrapping_sub(borrow);
            result[i] = diff as u64;
            u128::from(diff >> 127 != 0)
        },
    );
    (result, borrow != 0)
}

/// Returns `true` if `a >= modulus`.
fn gte_modulus(a: &[u64; LIMBS], modulus: &[u64; LIMBS]) -> bool {
    a.iter()
        .zip(modulus.iter())
        .rev()
        .fold(core::cmp::Ordering::Equal, |ord, (&ai, &mi)| match ord {
            core::cmp::Ordering::Equal => ai.cmp(&mi),
            other => other,
        })
        != core::cmp::Ordering::Less
}

/// Schoolbook multiplication producing an 8-limb (512-bit) result.
#[allow(clippy::cast_possible_truncation)]
fn mul_wide(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS * 2] {
    let mut result = [0u64; LIMBS * 2];
    a.iter().enumerate().for_each(|(i, &ai)| {
        let carry = b.iter().enumerate().fold(0u128, |carry, (j, &bj)| {
            let prod =
                u128::from(ai) * u128::from(bj) + u128::from(result[i + j]) + carry;
            result[i + j] = prod as u64;
            prod >> 64
        });
        result[i + LIMBS] = carry as u64;
    });
    result
}

/// Reduces a 512-bit product modulo `p` via shift-and-subtract.
fn reduce_wide(wide: &[u64; LIMBS * 2], modulus: &[u64; LIMBS]) -> [u64; LIMBS] {
    let total_bits = LIMBS * 2 * 64;
    (0..total_bits).rev().fold([0u64; LIMBS], |acc, bit| {
        // Shift accumulator left by 1
        let shifted = shift_left_one(&acc);
        // Bring in the current bit from the wide product
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        let incoming = (wide[limb_idx] >> bit_idx) & 1;
        let with_bit = [
            shifted[0] | incoming,
            shifted[1],
            shifted[2],
            shifted[3],
        ];
        // Conditional subtract
        if gte_modulus(&with_bit, modulus) {
            sub_limbs(&with_bit, modulus).0
        } else {
            with_bit
        }
    })
}

/// Shifts a 4-limb number left by one bit.
fn shift_left_one(a: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut result = [0u64; LIMBS];
    (0..LIMBS).rev().for_each(|i| {
        result[i] = a[i] << 1;
        if i > 0 {
            result[i] |= a[i - 1] >> 63;
        }
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_add_identity() {
        let a = FieldElement::from_u64(42, Curve::Pallas);
        let z = FieldElement::zero(Curve::Pallas);
        assert_eq!(a + z, a);
        assert_eq!(z + a, a);
    }

    #[test]
    fn one_mul_identity() {
        let a = FieldElement::from_u64(12345, Curve::Pallas);
        let one = FieldElement::one(Curve::Pallas);
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = FieldElement::from_u64(100, Curve::Pallas);
        let b = FieldElement::from_u64(200, Curve::Pallas);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn sqr_equals_mul_self() {
        let a = FieldElement::from_u64(9999, Curve::Pallas);
        assert_eq!(a.sqr(), a * a);
    }

    #[test]
    fn fifth_root_roundtrip() {
        // x^5 should be the inverse of fifth_root for nonzero elements.
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let r = x.fifth_root();
        let r5 = r * r * r * r * r;
        assert_eq!(r5, x);
    }

    #[test]
    fn fifth_root_roundtrip_vesta() {
        let x = FieldElement::from_u64(13, Curve::Vesta);
        let r = x.fifth_root();
        let r5 = r * r * r * r * r;
        assert_eq!(r5, x);
    }

    #[test]
    fn from_limbs_rejects_modulus() {
        let result = FieldElement::from_limbs(PALLAS_MODULUS, Curve::Pallas);
        assert!(result.is_err());
    }

    #[test]
    fn modulus_minus_one_is_valid() {
        let mut limbs = PALLAS_MODULUS;
        limbs[0] -= 1;
        let result = FieldElement::from_limbs(limbs, Curve::Pallas);
        assert!(result.is_ok());
    }
}
