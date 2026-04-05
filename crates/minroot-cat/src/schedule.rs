//! Exponent scanning as a catamorphism over [`Stream`].
//!
//! The fifth-root computation `x^((4p-3)/5)` processes the exponent
//! bit-by-bit via square-and-multiply.  Each bit produces a control
//! signal telling the pipeline whether to square only or square-and-multiply.
//!
//! This is a **catamorphism** (fold) over the exponent bits, producing
//! a [`Stream`] of [`RoundControl`] signals.  Equivalently, it is an
//! **anamorphism** (unfold) that generates the control stream from the
//! exponent state.
//!
//! [`Stream`]: comp_cat_rs::effect::stream::Stream

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use minroot_core::field::Curve;

/// Control signal for a single pipeline round.
///
/// Tells the multiply stage whether to multiply by the base or bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundControl {
    /// Square only (exponent bit is 0): bypass the multiplier.
    SquareOnly,
    /// Square and multiply (exponent bit is 1): multiply by base.
    SquareAndMultiply,
}

impl RoundControl {
    /// Returns `true` if this round includes a multiplication.
    #[must_use]
    pub fn is_multiply(self) -> bool {
        self == Self::SquareAndMultiply
    }
}

/// State for the exponent scanner unfold.
///
/// Tracks the current bit position as we scan the exponent
/// from MSB to LSB.
#[derive(Debug, Clone)]
struct ExponentState {
    /// The exponent limbs (little-endian).
    limbs: [u64; 4],
    /// Current bit position (counts down from `num_bits - 1` to 0).
    /// `None` means we have finished scanning.
    position: Option<usize>,
}

/// Generates the control signal stream for a fifth-root computation.
///
/// Scans the exponent `(4p - 3) / 5` bit-by-bit from MSB to LSB,
/// producing a [`Stream`] of [`RoundControl`] values.  This is the
/// anamorphism (unfold) that drives the pipeline FSM.
///
/// The stream produces exactly `curve.exponent_bits()` elements.
#[must_use]
pub fn exponent_schedule(curve: Curve) -> Stream<core::convert::Infallible, RoundControl> {
    let limbs = curve.fifth_root_exponent();
    let num_bits = curve.exponent_bits();
    let init = ExponentState {
        limbs,
        position: num_bits.checked_sub(1),
    };

    Stream::unfold(
        init,
        Arc::new(|state: ExponentState| {
            Io::pure(state.position.map(|pos| {
                let limb_idx = pos / 64;
                let bit_idx = pos % 64;
                let bit = (state.limbs[limb_idx] >> bit_idx) & 1;
                let control = if bit == 1 {
                    RoundControl::SquareAndMultiply
                } else {
                    RoundControl::SquareOnly
                };
                let next_pos = pos.checked_sub(1);
                let next_state = ExponentState {
                    limbs: state.limbs,
                    position: next_pos,
                };
                (control, next_state)
            }))
        }),
    )
}

/// Counts the number of multiply rounds in the exponent schedule.
///
/// This equals the Hamming weight of the exponent, which determines
/// the average pipeline utilization (multiplies are more expensive
/// than square-only rounds in terms of power consumption).
#[must_use]
pub fn hamming_weight(curve: Curve) -> Io<core::convert::Infallible, usize> {
    exponent_schedule(curve).fold(
        0usize,
        Arc::new(|count, ctrl| {
            if ctrl.is_multiply() {
                count + 1
            } else {
                count
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwraps an infallible result via `into_ok`.
    fn infallible_ok<T>(r: Result<T, core::convert::Infallible>) -> T {
        r.unwrap_or_else(|e| match e {})
    }

    #[test]
    fn schedule_length_matches_exponent_bits() {
        let count = infallible_ok(
            exponent_schedule(Curve::Pallas)
                .fold(0usize, Arc::new(|n, _| n + 1))
                .run(),
        );
        assert_eq!(count, 254);
    }

    #[test]
    fn first_bit_is_square_and_multiply_for_pallas() {
        // MSB of Pallas exponent: 0x33... = 0b0011_0011...
        // Bit 253 (MSB of 254-bit number) is 1.
        let first = infallible_ok(
            exponent_schedule(Curve::Pallas)
                .take(1)
                .collect()
                .run(),
        );
        assert_eq!(first[0], RoundControl::SquareAndMultiply);
    }

    #[test]
    fn hamming_weight_is_positive() {
        let w = infallible_ok(hamming_weight(Curve::Pallas).run());
        assert!(w > 0);
    }
}
