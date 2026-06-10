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
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from multiplication (unreachable).
    pub fn sqr(self) -> Result<Self, Error> {
        self.try_mul(self)
    }

    /// Modular exponentiation via square-and-multiply.
    ///
    /// The exponent is given as little-endian limbs with `num_bits`
    /// significant bits.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from multiplication (unreachable).
    pub fn pow(self, exp: &[u64; LIMBS], num_bits: usize) -> Result<Self, Error> {
        (0..num_bits).rev().try_fold(Self::one(self.curve), |acc, i| {
            let squared = acc.sqr()?;
            let limb_idx = i / 64;
            let bit_idx = i % 64;
            if (exp[limb_idx] >> bit_idx) & 1 == 1 {
                squared.try_mul(self)
            } else {
                Ok(squared)
            }
        })
    }

    /// Computes the fifth root: `self^((4p-3)/5) mod p`.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from exponentiation (unreachable).
    pub fn fifth_root(self) -> Result<Self, Error> {
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

impl FieldElement {
    /// Modular addition: `self + rhs mod p`.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from limb addition or subtraction
    /// (unreachable).
    pub fn try_add(self, rhs: Self) -> Result<Self, Error> {
        debug_assert_eq!(self.curve, rhs.curve);
        Ok(Self {
            limbs: mod_add(&self.limbs, &rhs.limbs, &self.curve.modulus())?,
            curve: self.curve,
        })
    }

    /// Modular subtraction: `self - rhs mod p`.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from limb subtraction (unreachable).
    pub fn try_sub(self, rhs: Self) -> Result<Self, Error> {
        debug_assert_eq!(self.curve, rhs.curve);
        Ok(Self {
            limbs: mod_sub(&self.limbs, &rhs.limbs, &self.curve.modulus())?,
            curve: self.curve,
        })
    }

    /// Modular multiplication: `self * rhs mod p`.
    ///
    /// Uses schoolbook multiplication followed by shift-and-subtract reduction.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Truncation`] from wide reduction (unreachable).
    pub fn try_mul(self, rhs: Self) -> Result<Self, Error> {
        debug_assert_eq!(self.curve, rhs.curve);
        let wide = mul_wide(&self.limbs, &rhs.limbs);
        let reduced = reduce_wide(&wide, &self.curve.modulus())?;
        Ok(Self {
            limbs: reduced,
            curve: self.curve,
        })
    }
}

// ── Multi-limb arithmetic helpers ──────────────────────────────────

/// Adds two 4-limb numbers, returning (result, carry).
///
/// Ripple-carry addition, unrolled across the four little-endian limbs.
/// Each `s*` is the `u128` column sum (two 64-bit limbs plus the carry-in
/// `s_{i-1} >> 64`, which is 0 or 1), computed with `wrapping_add`; every
/// column stays within `2^65`, so the wrap never triggers and `>> 64`
/// recovers the exact carry.  Each limb is masked to 64 bits before the
/// checked `u64` conversion, so the `TryFrom` is total in practice.
///
/// # Errors
///
/// Returns [`Error::Truncation`] if a masked limb fails to fit in `u64`
/// (unreachable; the mask guarantees the value is in range).
fn add_limbs(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> Result<([u64; LIMBS], bool), Error> {
    let s0 = u128::from(a[0]).wrapping_add(u128::from(b[0]));
    let s1 = u128::from(a[1]).wrapping_add(u128::from(b[1])).wrapping_add(s0 >> 64);
    let s2 = u128::from(a[2]).wrapping_add(u128::from(b[2])).wrapping_add(s1 >> 64);
    let s3 = u128::from(a[3]).wrapping_add(u128::from(b[3])).wrapping_add(s2 >> 64);
    let mask = u128::from(u64::MAX);
    let l0 = u64::try_from(s0 & mask).map_err(|_| Error::Truncation)?;
    let l1 = u64::try_from(s1 & mask).map_err(|_| Error::Truncation)?;
    let l2 = u64::try_from(s2 & mask).map_err(|_| Error::Truncation)?;
    let l3 = u64::try_from(s3 & mask).map_err(|_| Error::Truncation)?;
    Ok(([l0, l1, l2, l3], s3 >> 64 != 0))
}

/// Subtracts two 4-limb numbers, returning (result, borrow).
///
/// Ripple-borrow subtraction, unrolled across the four little-endian
/// limbs.  Each `bo*` is the borrow out of the limb below it (0 or 1),
/// detected via bit 127 of the wrapping `u128` difference.  Each limb is
/// masked to 64 bits before the checked `u64` conversion, so the
/// `TryFrom` is total in practice.
///
/// # Errors
///
/// Returns [`Error::Truncation`] if a masked limb fails to fit in `u64`
/// (unreachable; the mask guarantees the value is in range).
fn sub_limbs(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> Result<([u64; LIMBS], bool), Error> {
    let d0 = u128::from(a[0]).wrapping_sub(u128::from(b[0]));
    let bo0 = u128::from(d0 >> 127 != 0);
    let d1 = u128::from(a[1]).wrapping_sub(u128::from(b[1])).wrapping_sub(bo0);
    let bo1 = u128::from(d1 >> 127 != 0);
    let d2 = u128::from(a[2]).wrapping_sub(u128::from(b[2])).wrapping_sub(bo1);
    let bo2 = u128::from(d2 >> 127 != 0);
    let d3 = u128::from(a[3]).wrapping_sub(u128::from(b[3])).wrapping_sub(bo2);
    let mask = u128::from(u64::MAX);
    let l0 = u64::try_from(d0 & mask).map_err(|_| Error::Truncation)?;
    let l1 = u64::try_from(d1 & mask).map_err(|_| Error::Truncation)?;
    let l2 = u64::try_from(d2 & mask).map_err(|_| Error::Truncation)?;
    let l3 = u64::try_from(d3 & mask).map_err(|_| Error::Truncation)?;
    Ok(([l0, l1, l2, l3], d3 >> 127 != 0))
}

/// Returns `true` if `a >= modulus`.
///
/// Lexicographic comparison from the most-significant limb down.
fn gte_modulus(a: &[u64; LIMBS], modulus: &[u64; LIMBS]) -> bool {
    if a[3] != modulus[3] {
        a[3] > modulus[3]
    } else if a[2] != modulus[2] {
        a[2] > modulus[2]
    } else if a[1] != modulus[1] {
        a[1] > modulus[1]
    } else {
        a[0] >= modulus[0]
    }
}

/// Modular addition of two reduced 4-limb values: `(a + b) mod modulus`.
///
/// Adds the limbs, then conditionally subtracts `modulus` once if the sum
/// carried out of 256 bits or is already `>= modulus`.  For inputs below
/// `modulus` (with `modulus < 2^255`) the sum is below `2·modulus`, so a
/// single conditional subtraction restores the canonical residue.
///
/// # Errors
///
/// Propagates [`Error::Truncation`] from limb addition or subtraction
/// (unreachable).
fn mod_add(
    a: &[u64; LIMBS],
    b: &[u64; LIMBS],
    modulus: &[u64; LIMBS],
) -> Result<[u64; LIMBS], Error> {
    let (sum, carry) = add_limbs(a, b)?;
    let result = if carry || gte_modulus(&sum, modulus) {
        sub_limbs(&sum, modulus)?.0
    } else {
        sum
    };
    Ok(result)
}

/// Modular subtraction of two reduced 4-limb values: `(a - b) mod modulus`.
///
/// Subtracts the limbs, then adds `modulus` back if the subtraction
/// borrowed (i.e. `a < b`), yielding the canonical residue.
///
/// # Errors
///
/// Propagates [`Error::Truncation`] from limb subtraction (unreachable).
fn mod_sub(
    a: &[u64; LIMBS],
    b: &[u64; LIMBS],
    modulus: &[u64; LIMBS],
) -> Result<[u64; LIMBS], Error> {
    let (diff, borrow) = sub_limbs(a, b)?;
    let result = if borrow {
        add_limbs(&diff, modulus)?.0
    } else {
        diff
    };
    Ok(result)
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
///
/// Processes the wide product one bit at a time from the most
/// significant (bit 511) down to bit 0, shifting the accumulator left,
/// bringing in the next bit, and conditionally subtracting the modulus.
fn reduce_wide(
    wide: &[u64; LIMBS * 2],
    modulus: &[u64; LIMBS],
) -> Result<[u64; LIMBS], Error> {
    reduce_wide_rec(wide, modulus, LIMBS * 2 * 64, [0u64; LIMBS])
}

/// Tail-recursive core of [`reduce_wide`].
///
/// `remaining` counts the bits still to process; the bit handled in this
/// step is `remaining - 1`, so the recursion walks bits 511 → 0.
///
/// # Errors
///
/// Propagates [`Error::Truncation`] from limb subtraction (unreachable).
fn reduce_wide_rec(
    wide: &[u64; LIMBS * 2],
    modulus: &[u64; LIMBS],
    remaining: usize,
    acc: [u64; LIMBS],
) -> Result<[u64; LIMBS], Error> {
    if remaining == 0 {
        Ok(acc)
    } else {
        let bit = remaining - 1;
        let shifted = shift_left_one(&acc);
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        let incoming = (wide[limb_idx] >> bit_idx) & 1;
        let with_bit = [shifted[0] | incoming, shifted[1], shifted[2], shifted[3]];
        let next = if gte_modulus(&with_bit, modulus) {
            sub_limbs(&with_bit, modulus)?.0
        } else {
            with_bit
        };
        reduce_wide_rec(wide, modulus, bit, next)
    }
}

/// Shifts a 4-limb number left by one bit.
///
/// Each limb takes its own `<< 1` plus the top bit carried up from the
/// limb below.  The bit shifted out of the top limb is discarded.
fn shift_left_one(a: &[u64; LIMBS]) -> [u64; LIMBS] {
    [
        a[0] << 1,
        (a[1] << 1) | (a[0] >> 63),
        (a[2] << 1) | (a[1] >> 63),
        (a[3] << 1) | (a[2] >> 63),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_add_identity() -> Result<(), Error> {
        let a = FieldElement::from_u64(42, Curve::Pallas);
        let z = FieldElement::zero(Curve::Pallas);
        assert_eq!(a.try_add(z)?, a);
        assert_eq!(z.try_add(a)?, a);
        Ok(())
    }

    #[test]
    fn one_mul_identity() -> Result<(), Error> {
        let a = FieldElement::from_u64(12345, Curve::Pallas);
        let one = FieldElement::one(Curve::Pallas);
        assert_eq!(a.try_mul(one)?, a);
        assert_eq!(one.try_mul(a)?, a);
        Ok(())
    }

    #[test]
    fn add_sub_roundtrip() -> Result<(), Error> {
        let a = FieldElement::from_u64(100, Curve::Pallas);
        let b = FieldElement::from_u64(200, Curve::Pallas);
        assert_eq!((a.try_add(b)?).try_sub(b)?, a);
        Ok(())
    }

    #[test]
    fn sqr_equals_mul_self() -> Result<(), Error> {
        let a = FieldElement::from_u64(9999, Curve::Pallas);
        assert_eq!(a.sqr()?, a.try_mul(a)?);
        Ok(())
    }

    #[test]
    fn fifth_root_roundtrip() -> Result<(), Error> {
        // x^5 should be the inverse of fifth_root for nonzero elements.
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let r = x.fifth_root()?;
        let r5 = r.try_mul(r)?.try_mul(r)?.try_mul(r)?.try_mul(r)?;
        assert_eq!(r5, x);
        Ok(())
    }

    #[test]
    fn fifth_root_roundtrip_vesta() -> Result<(), Error> {
        let x = FieldElement::from_u64(13, Curve::Vesta);
        let r = x.fifth_root()?;
        let r5 = r.try_mul(r)?.try_mul(r)?.try_mul(r)?.try_mul(r)?;
        assert_eq!(r5, x);
        Ok(())
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
