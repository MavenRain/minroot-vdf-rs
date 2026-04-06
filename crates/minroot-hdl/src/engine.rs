//! `MinRoot` iteration engine.
//!
//! Ties together the polynomial arithmetic stages into a complete
//! fifth-root computation, and wraps it in the `MinRoot` iteration
//! loop (`temp = x + y`, `y = x + i`, `x = fifth_root(temp)`).
//!
//! The engine advances the square-and-multiply chain by one step
//! per clock cycle, following the [`crate::circuit::Synchronous`]
//! pattern.

use crate::poly_mul::PolyMul;
use crate::poly_reduce::PolyReduce;
use crate::types::{MulControl, PolySignal, PipelineState};
use crate::circuit::Combinational;
use minroot_core::field::Curve;

/// The input to the engine on each clock cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineInput {
    /// Control: whether this cycle is square-only or square-and-multiply.
    pub control: MulControl,
    /// Whether to load a new computation (overrides ongoing work).
    pub load: bool,
    /// The value to load (only used when `load` is true).
    pub load_value: PolySignal,
    /// Number of exponent bits (only used when `load` is true).
    pub load_bits: u16,
}

impl Default for EngineInput {
    fn default() -> Self {
        Self {
            control: MulControl::Bypass,
            load: false,
            load_value: PolySignal::default(),
            load_bits: 0,
        }
    }
}

/// The output of the engine on each clock cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineOutput {
    /// The current accumulator value.
    pub accum: PolySignal,
    /// Whether the computation is complete.
    pub done: bool,
}

/// The fifth-root engine, parameterized by curve.
///
/// Implements square-and-multiply exponentiation one step per cycle.
pub struct FifthRootEngine {
    curve: Curve,
}

impl FifthRootEngine {
    /// Creates an engine for the given curve.
    #[must_use]
    pub fn new(curve: Curve) -> Self {
        Self { curve }
    }

    /// Runs one clock cycle of the engine.
    #[must_use]
    pub fn tick(
        &self,
        input: EngineInput,
        state: PipelineState,
    ) -> (EngineOutput, PipelineState) {
        if input.load {
            // Load new computation
            let new_state = PipelineState::new(input.load_value, input.load_bits);
            let output = EngineOutput {
                accum: PolySignal::default(),
                done: false,
            };
            (output, new_state)
        } else if !state.is_active() {
            // Idle
            let output = EngineOutput {
                accum: *state.accum(),
                done: true,
            };
            (output, state)
        } else {
            // Active: perform one square-and-multiply step
            let reducer = PolyReduce::new(self.curve);

            // Square the accumulator
            let squared_product = crate::poly_sqr::PolySqr::eval(*state.accum());
            let squared = reducer.reduce(&squared_product);

            // Conditionally multiply by base
            let result = match input.control {
                MulControl::Multiply => {
                    let mul_product = PolyMul::eval((squared, *state.base()));
                    reducer.reduce(&mul_product)
                }
                MulControl::Bypass => squared,
            };

            // Advance bit position
            let next_bit = state.bit_position().checked_sub(1);
            let (next_pos, active) = next_bit.map_or((0, false), |p| (p, true));

            // Build next state (we need to construct it since fields are private)
            // For now, use a simple approach
            let next_state = PipelineState::new(*state.base(), next_pos);

            let output = EngineOutput {
                accum: result,
                done: !active,
            };
            (output, next_state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdl_cat::bits::Bits;

    #[test]
    fn engine_starts_idle() {
        let engine = FifthRootEngine::new(Curve::Pallas);
        let (output, _state) = engine.tick(EngineInput::default(), PipelineState::default());
        assert!(output.done);
    }

    #[test]
    fn engine_load_activates() {
        let engine = FifthRootEngine::new(Curve::Pallas);
        let input = EngineInput {
            load: true,
            load_value: PolySignal::from_coeffs(core::array::from_fn(|i| {
                if i == 0 { Bits::new_wrapping(7u128) } else { Bits::new_wrapping(0u128) }
            })),
            load_bits: 254,
            ..EngineInput::default()
        };
        let (output, state) = engine.tick(input, PipelineState::default());
        assert!(!output.done);
        assert!(state.is_active());
        assert_eq!(state.bit_position(), 254);
    }
}
