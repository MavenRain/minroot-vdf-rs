//! Hardware signal types for the `MinRoot` pipeline.
//!
//! All types use [`hdl_cat::bits::Bits`] for fixed-width signals.
//! The polynomial representation uses 17 coefficients of 17 bits each,
//! matching the `SystemVerilog` `mrt_pkg` parameters.

use hdl_cat::bits::Bits;

use crate::bits_ext;
use minroot_cat::schedule::RoundControl;
use minroot_core::polynomial::{COEFF_BITS, NUM_COEFFS, WORD_BITS};

/// A single polynomial coefficient: [`COEFF_BITS`] = 17 bits.
pub type Coeff = Bits<{ COEFF_BITS }>;

/// Returns a zero coefficient.
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
            coeffs: [Bits::ZERO; NUM_COEFFS],
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
    pub fn coeffs(&self) -> &[Coeff; NUM_COEFFS] {
        &self.coeffs
    }

    /// Returns a single coefficient by index, or [`zero_coeff()`] if out of bounds.
    pub fn coeff(&self, i: usize) -> Coeff {
        self.coeffs.get(i).copied().unwrap_or(zero_coeff())
    }

    /// Converts from a [`minroot_core::polynomial::PolyElement`].
    #[must_use]
    pub fn from_poly_element(pe: &minroot_core::polynomial::PolyElement) -> Self {
        let coeffs = core::array::from_fn(|i| {
            pe.coeffs()
                .get(i)
                .map_or(zero_coeff(), |&c| Bits::new_wrapping(u128::from(c)))
        });
        Self { coeffs }
    }

    /// Converts back to a [`minroot_core::polynomial::PolyElement`].
    ///
    /// Each [`Bits<17>`](Coeff) coefficient is narrowed to `u32` and packed
    /// into the polynomial form.  This is the inverse of
    /// [`from_poly_element`](Self::from_poly_element) for values that
    /// originated from a field element.
    ///
    /// # Errors
    ///
    /// Returns [`minroot_core::error::Error::OutOfRange`] if any
    /// coefficient exceeds [`COEFF_BITS`] bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use minroot_core::field::{Curve, FieldElement};
    /// use minroot_core::polynomial::PolyElement;
    /// use minroot_hdl::types::PolySignal;
    ///
    /// # fn main() -> Result<(), minroot_core::error::Error> {
    /// let fe = FieldElement::from_u64(42, Curve::Pallas);
    /// let signal = PolySignal::from_poly_element(&PolyElement::from_field(fe));
    /// let roundtrip = signal.to_poly_element(Curve::Pallas)?.to_field()?;
    /// assert_eq!(roundtrip, fe);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_poly_element(
        &self,
        curve: minroot_core::field::Curve,
    ) -> Result<minroot_core::polynomial::PolyElement, minroot_core::error::Error> {
        let coeffs = core::array::from_fn(|i| {
            bits_ext::to_u128(self.coeff(i)) as u32
        });
        minroot_core::polynomial::PolyElement::from_coeffs(coeffs, curve)
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

impl From<RoundControl> for MulControl {
    /// Converts a categorical [`RoundControl`] to a hardware [`MulControl`].
    ///
    /// [`RoundControl::SquareOnly`] maps to [`MulControl::Bypass`] and
    /// [`RoundControl::SquareAndMultiply`] maps to [`MulControl::Multiply`].
    fn from(ctrl: RoundControl) -> Self {
        match ctrl {
            RoundControl::SquareOnly => Self::Bypass,
            RoundControl::SquareAndMultiply => Self::Multiply,
        }
    }
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

    /// Constructs a pipeline state with explicit control over all fields.
    ///
    /// Used by the engine's tick logic to advance the computation
    /// while preserving the accumulator value and active flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use minroot_hdl::types::{PipelineState, PolySignal};
    ///
    /// let base = PolySignal::default();
    /// let accum = PolySignal::default();
    /// let state = PipelineState::advance(base, accum, 100, true);
    /// assert_eq!(state.bit_position(), 100);
    /// assert!(state.is_active());
    /// ```
    #[must_use]
    pub fn advance(
        base: PolySignal,
        accum: PolySignal,
        bit_position: u16,
        active: bool,
    ) -> Self {
        Self { accum, base, bit_position, active }
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
            core::array::from_fn(|i| Bits::new_wrapping(u128::try_from(i).unwrap_or(0)));
        let b_coeffs: [Coeff; NUM_COEFFS] =
            core::array::from_fn(|i| Bits::new_wrapping(u128::try_from(i * 2).unwrap_or(0)));
        let a = PolySignal::from_coeffs(a_coeffs);
        let b = PolySignal::from_coeffs(b_coeffs);
        let sum = a + b;
        sum.coeffs().iter().enumerate().for_each(|(i, c)| {
            let expected = Bits::<{ COEFF_BITS }>::new_wrapping(u128::try_from(i * 3).unwrap_or(0));
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

    #[test]
    fn pipeline_state_advance_preserves_accum() {
        let base = PolySignal::default();
        let accum_coeffs = core::array::from_fn(|i| {
            Bits::<{ COEFF_BITS }>::new_wrapping(u128::try_from(i + 1).unwrap_or(0))
        });
        let accum = PolySignal::from_coeffs(accum_coeffs);
        let state = PipelineState::advance(base, accum, 42, true);
        assert_eq!(*state.accum(), accum);
        assert_eq!(state.bit_position(), 42);
        assert!(state.is_active());
    }

    #[test]
    fn pipeline_state_advance_inactive() {
        let state = PipelineState::advance(
            PolySignal::default(),
            PolySignal::default(),
            0,
            false,
        );
        assert!(!state.is_active());
    }

    #[test]
    fn poly_signal_to_poly_element_roundtrip() {
        let fe = minroot_core::field::FieldElement::from_u64(12345, minroot_core::field::Curve::Pallas);
        let pe = minroot_core::polynomial::PolyElement::from_field(fe);
        let signal = PolySignal::from_poly_element(&pe);
        let back = signal.to_poly_element(minroot_core::field::Curve::Pallas);
        assert!(back.iter().all(|p| p.to_field() == Ok(fe)));
    }

    #[test]
    fn mul_control_from_round_control() {
        use minroot_cat::schedule::RoundControl;
        assert_eq!(
            MulControl::from(RoundControl::SquareOnly),
            MulControl::Bypass,
        );
        assert_eq!(
            MulControl::from(RoundControl::SquareAndMultiply),
            MulControl::Multiply,
        );
    }
}
