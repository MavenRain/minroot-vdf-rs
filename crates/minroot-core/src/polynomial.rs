//! Redundant polynomial representation for hardware arithmetic.
//!
//! The hardware represents 256-bit field elements as polynomials with
//! [`NUM_COEFFS`] coefficients of [`COEFF_BITS`] bits each.  This
//! redundant representation avoids long carry chains in the critical
//! path, enabling higher clock frequencies.
//!
//! # Parameters (matching the `SystemVerilog` `mrt_pkg`)
//!
//! - `TargetBits = 256`
//! - `WordBits = 16`
//! - `RedundantBits = 1`
//! - `ExtraCoeffs = 1`
//! - `NumCoeffs = ceil(256 / 16) + 1 = 17`
//! - `CoeffBits = 16 + 1 = 17`

use crate::error::Error;
use crate::field::{Curve, FieldElement};

/// Number of data bits per coefficient word.
pub const WORD_BITS: usize = 16;

/// Number of redundant bits per coefficient (carry absorption).
pub const REDUNDANT_BITS: usize = 1;

/// Total bits per coefficient.
pub const COEFF_BITS: usize = WORD_BITS + REDUNDANT_BITS;

/// Extra coefficients beyond `ceil(target_bits / word_bits)`.
pub const EXTRA_COEFFS: usize = 1;

/// Total number of coefficients per polynomial.
pub const NUM_COEFFS: usize = 256_usize.div_ceil(WORD_BITS) + EXTRA_COEFFS;

/// Mask for a single coefficient value.
const COEFF_MASK: u32 = (1 << COEFF_BITS) - 1;

/// A field element in redundant polynomial form.
///
/// Each coefficient `c[i]` holds up to [`COEFF_BITS`] bits.
/// The integer value is `sum(c[i] * 2^(i * WORD_BITS))` for `i` in `0..NUM_COEFFS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolyElement {
    coeffs: [u32; NUM_COEFFS],
    curve: Curve,
}

impl PolyElement {
    /// Constructs a polynomial element from an array of coefficients.
    ///
    /// Each coefficient must fit in [`COEFF_BITS`] bits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] if any coefficient exceeds the bit width.
    pub fn from_coeffs(
        coeffs: [u32; NUM_COEFFS],
        curve: Curve,
    ) -> Result<Self, Error> {
        if coeffs.iter().all(|&c| c <= COEFF_MASK) {
            Ok(Self { coeffs, curve })
        } else {
            Err(Error::OutOfRange {
                context: "polynomial coefficient exceeds COEFF_BITS",
            })
        }
    }

    /// Converts a [`FieldElement`] into polynomial form.
    ///
    /// Extracts [`WORD_BITS`]-wide chunks from the integer representation.
    #[must_use]
    pub fn from_field(fe: FieldElement) -> Self {
        let limbs = fe.limbs();
        let word_mask = (1u64 << WORD_BITS) - 1;
        let coeffs = core::array::from_fn(|i| {
            let bit_offset = i * WORD_BITS;
            let limb_idx = bit_offset / 64;
            let bit_idx = bit_offset % 64;
            if limb_idx < 4 {
                let val = limbs[limb_idx] >> bit_idx;
                // Handle crossing a limb boundary
                let combined = if bit_idx + WORD_BITS > 64 && limb_idx + 1 < 4 {
                    val | (limbs[limb_idx + 1] << (64 - bit_idx))
                } else {
                    val
                };
                #[allow(clippy::cast_possible_truncation)]
                { (combined & word_mask) as u32 }
            } else {
                0
            }
        });
        Self {
            coeffs,
            curve: fe.curve(),
        }
    }

    /// Converts back to a [`FieldElement`].
    ///
    /// Performs carry propagation and modular reduction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] if the polynomial's value exceeds the modulus.
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_field(self) -> Result<FieldElement, Error> {
        // Accumulate into a 512-bit intermediate to handle overflow,
        // then reduce mod p.
        let mut accum = [0u128; 5];

        self.coeffs.iter().enumerate().for_each(|(i, &c)| {
            let bit_offset = i * WORD_BITS;
            let chunk_idx = bit_offset / 128;
            let chunk_bit = bit_offset % 128;
            accum[chunk_idx] += u128::from(c) << chunk_bit;
        });

        // Propagate carries across chunks into 64-bit limbs
        let mut limbs = [0u64; 4];
        let _ = accum.iter().enumerate().fold(0u128, |carry, (i, &val)| {
            let total = val + carry;
            let base = i * 2;
            if base < 4 {
                limbs[base] = total as u64;
                if base + 1 < 4 {
                    limbs[base + 1] = (total >> 64) as u64;
                    0
                } else {
                    total >> 64
                }
            } else {
                total
            }
        });

        FieldElement::from_limbs(limbs, self.curve).map_err(|_| Error::OutOfRange {
            context: "polynomial to_field: value exceeds modulus after carry propagation",
        })
    }

    /// Returns the coefficient array.
    #[must_use]
    pub fn coeffs(&self) -> &[u32; NUM_COEFFS] {
        &self.coeffs
    }

    /// Returns the curve.
    #[must_use]
    pub fn curve(&self) -> Curve {
        self.curve
    }

    /// Coefficient-wise addition without carry propagation.
    ///
    /// This is the hardware-friendly operation: coefficients may
    /// temporarily exceed [`WORD_BITS`], using the redundant bit.
    #[must_use]
    pub fn add_no_reduce(self, rhs: Self) -> Self {
        debug_assert_eq!(self.curve, rhs.curve);
        let coeffs = core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]);
        Self {
            coeffs,
            curve: self.curve,
        }
    }

    /// Zero polynomial.
    #[must_use]
    pub fn zero(curve: Curve) -> Self {
        Self {
            coeffs: [0; NUM_COEFFS],
            curve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small_value() {
        let fe = FieldElement::from_u64(0xDEAD_BEEF, Curve::Pallas);
        let poly = PolyElement::from_field(fe);
        let back = poly.to_field();
        assert_eq!(back, Ok(fe));
    }

    #[test]
    fn roundtrip_large_value() {
        // Use a value near the modulus but below it
        let fe = FieldElement::from_limbs(
            [
                0x992d_30ec_ffff_ffff,
                0x2246_98fc_094c_f91a,
                0,
                0x3fff_ffff_ffff_ffff,
            ],
            Curve::Pallas,
        );
        assert!(
            fe.iter().all(|fe| {
                let poly = PolyElement::from_field(*fe);
                poly.to_field() == Ok(*fe)
            })
        );
    }

    #[test]
    fn zero_roundtrip() {
        let fe = FieldElement::zero(Curve::Pallas);
        let poly = PolyElement::from_field(fe);
        assert_eq!(poly.to_field(), Ok(fe));
    }

    #[test]
    fn coefficient_count() {
        assert_eq!(NUM_COEFFS, 17);
    }

    #[test]
    fn coeff_bits_correct() {
        assert_eq!(COEFF_BITS, 17);
    }
}
