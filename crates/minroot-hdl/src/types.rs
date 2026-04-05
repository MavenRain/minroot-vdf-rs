//! Hardware signal types for the `MinRoot` pipeline.
//!
//! All types use [`rhdl::bits::Bits`] for fixed-width signals.
//! The polynomial representation uses 17 coefficients of 17 bits each,
//! matching the `SystemVerilog` `mrt_pkg` parameters.

use rhdl::bits::Bits;

use crate::bits_ext;
use minroot_core::polynomial::{COEFF_BITS, NUM_COEFFS, WORD_BITS};

/// A single polynomial coefficient: [`COEFF_BITS`] = 17 bits.
pub type Coeff = Bits<{ COEFF_BITS }>;

/// Returns a zero coefficient.
#[must_use]
pub fn zero_coeff() -> Coeff {
    bits_ext::zero()
}

/// A field element in redundant polynomial form.
///
/// 17 coefficients of 17 bits each.  This is the fundamental
/// data type flowing through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolySignal {
    coeffs: [Coeff; NUM_COEFFS],
}

impl Default for PolySignal {
    fn default() -> Self {
        Self {
            coeffs: [Bits::from(0u128); NUM_COEFFS],
        }
    }
}

impl PolySignal {
    /// Constructs from a coefficient array.
    #[must_use]
    pub fn from_coeffs(coeffs: [Coeff; NUM_COEFFS]) -> Self {
        Self { coeffs }
    }

    /// Returns the coefficient array.
    #[must_use]
    pub fn coeffs(&self) -> &[Coeff; NUM_COEFFS] {
        &self.coeffs
    }

    /// Returns a single coefficient by index, or [`zero_coeff()`] if out of bounds.
    #[must_use]
    pub fn coeff(&self, i: usize) -> Coeff {
        self.coeffs.get(i).copied().unwrap_or(zero_coeff())
    }

    /// Converts from a [`minroot_core::polynomial::PolyElement`].
    #[must_use]
    pub fn from_poly_element(pe: &minroot_core::polynomial::PolyElement) -> Self {
        let coeffs = core::array::from_fn(|i| {
            pe.coeffs()
                .get(i)
                .map_or(zero_coeff(), |&c| Bits::from(u128::from(c)))
        });
        Self { coeffs }
    }

}

impl core::ops::Add for PolySignal {
    type Output = Self;

    /// Coefficient-wise addition (no carry propagation).
    ///
    /// Each coefficient is added independently.  The result may
    /// have coefficients exceeding `WORD_BITS`, using the redundant bit.
    fn add(self, rhs: Self) -> Self {
        let coeffs = core::array::from_fn(|i| self.coeff(i) + rhs.coeff(i));
        Self { coeffs }
    }
}

/// A partial product from multiplying two coefficients.
///
/// Two [`COEFF_BITS`]-wide values multiplied produce a result up to
/// `2 * COEFF_BITS = 34` bits wide.
pub type PartialProduct = Bits<{ COEFF_BITS * 2 }>;

/// Returns a zero partial product.
#[must_use]
pub fn zero_pp() -> PartialProduct {
    bits_ext::zero()
}

/// Control signal for the multiply stage.
///
/// Tells the pipeline whether to multiply by the base value
/// (exponent bit = 1) or bypass (exponent bit = 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MulControl {
    /// Bypass: pass the squared value through unchanged.
    #[default]
    Bypass,
    /// Multiply: multiply the squared value by the base.
    Multiply,
}

/// Pipeline stage state: tracks which exponent bit is being processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PipelineState {
    /// The current accumulator (running result of square-and-multiply).
    accum: PolySignal,
    /// The base value being exponentiated.
    base: PolySignal,
    /// Current bit position in the exponent (counts down).
    bit_position: u16,
    /// Whether the pipeline is actively computing.
    active: bool,
}

impl PipelineState {
    /// Constructs a new pipeline state for a fifth-root computation.
    #[must_use]
    pub fn new(base: PolySignal, num_bits: u16) -> Self {
        Self {
            accum: PolySignal::default(),
            base,
            bit_position: num_bits,
            active: true,
        }
    }

    /// Returns the current accumulator.
    #[must_use]
    pub fn accum(&self) -> &PolySignal {
        &self.accum
    }

    /// Returns the base value.
    #[must_use]
    pub fn base(&self) -> &PolySignal {
        &self.base
    }

    /// Returns the current bit position.
    #[must_use]
    pub fn bit_position(&self) -> u16 {
        self.bit_position
    }

    /// Returns whether the pipeline is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Number of bits in a word (re-exported for convenience).
pub const HW_WORD_BITS: usize = WORD_BITS;

/// Number of coefficients (re-exported for convenience).
pub const HW_NUM_COEFFS: usize = NUM_COEFFS;

/// Number of coefficient bits (re-exported for convenience).
pub const HW_COEFF_BITS: usize = COEFF_BITS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_signal_default_is_zero() {
        let ps = PolySignal::default();
        ps.coeffs().iter().for_each(|c| assert_eq!(*c, zero_coeff()));
    }

    #[test]
    fn poly_signal_add_coeffwise() {
        let a_coeffs: [Coeff; NUM_COEFFS] =
            core::array::from_fn(|i| Bits::from(u128::try_from(i).unwrap_or(0)));
        let b_coeffs: [Coeff; NUM_COEFFS] =
            core::array::from_fn(|i| Bits::from(u128::try_from(i * 2).unwrap_or(0)));
        let a = PolySignal::from_coeffs(a_coeffs);
        let b = PolySignal::from_coeffs(b_coeffs);
        let sum = a + b;
        sum.coeffs().iter().enumerate().for_each(|(i, c)| {
            let expected = Bits::<{ COEFF_BITS }>::from(u128::try_from(i * 3).unwrap_or(0));
            assert_eq!(*c, expected);
        });
    }

    #[test]
    fn coeff_out_of_bounds_returns_zero() {
        let ps = PolySignal::default();
        assert_eq!(ps.coeff(999), zero_coeff());
    }

    #[test]
    fn pipeline_state_tracks_bit_position() {
        let state = PipelineState::new(PolySignal::default(), 254);
        assert_eq!(state.bit_position(), 254);
        assert!(state.is_active());
    }
}
