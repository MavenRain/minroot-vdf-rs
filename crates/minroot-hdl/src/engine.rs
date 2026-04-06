//! `MinRoot` iteration engine.
//!
//! Ties together the polynomial arithmetic stages into a complete
//! fifth-root computation, and wraps it in the `MinRoot` iteration
//! loop (`temp = x + y`, `y = x + i`, `x = fifth_root(temp)`).
//!
//! The engine advances the square-and-multiply chain by one step
//! per clock cycle, following the [`crate::circuit::Synchronous`]
//! pattern.
//!
//! # Behavioral Simulation
//!
//! [`PallasEngine`] and [`VestaEngine`] implement
//! [`Synchronous`](crate::circuit::Synchronous), enabling
//! cycle-accurate behavioral simulation via
//! [`Synchronous::simulate`](crate::circuit::Synchronous::simulate).
//!
//! ```
//! # fn main() -> Result<(), minroot_core::error::Error> {
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_core::polynomial::PolyElement;
//! use minroot_hdl::circuit::Synchronous;
//! use minroot_hdl::engine::{PallasEngine, EngineInput};
//! use minroot_hdl::types::{MulControl, PolySignal};
//!
//! let x = FieldElement::from_u64(7, Curve::Pallas);
//! let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));
//!
//! // Compute x^3: exponent 3 = binary 11 (2 bits)
//! let inputs: Vec<EngineInput> = vec![
//!     EngineInput::load(signal, 2),
//!     EngineInput::round(MulControl::Multiply), // MSB = 1
//!     EngineInput::round(MulControl::Multiply), // LSB = 1
//! ];
//!
//! let (outputs, final_state) = PallasEngine::simulate(inputs);
//! assert!(outputs.last().is_some_and(|o| o.done));
//!
//! // Verify against field arithmetic
//! let result = final_state.accum().to_poly_element(Curve::Pallas)?.to_field()?;
//! assert_eq!(result, x * x * x);
//! # Ok(())
//! # }
//! ```

use crate::poly_mul::PolyMul;
use crate::poly_reduce::PolyReduce;
use crate::types::{MulControl, PolySignal, PipelineState};
use crate::circuit::{Combinational, Synchronous};
use minroot_core::field::{Curve, FieldElement};
use minroot_core::polynomial::PolyElement;

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

impl EngineInput {
    /// Constructs a load input that begins a new fifth-root computation.
    ///
    /// The engine initializes its accumulator to the polynomial form of
    /// the multiplicative identity and stores `value` as the base for
    /// subsequent square-and-multiply cycles.
    ///
    /// # Examples
    ///
    /// ```
    /// use minroot_hdl::engine::EngineInput;
    /// use minroot_hdl::types::PolySignal;
    ///
    /// let signal = PolySignal::default();
    /// let input = EngineInput::load(signal, 254);
    /// assert!(input.load);
    /// assert_eq!(input.load_bits, 254);
    /// ```
    #[must_use]
    pub fn load(value: PolySignal, num_bits: u16) -> Self {
        Self {
            control: MulControl::Bypass,
            load: true,
            load_value: value,
            load_bits: num_bits,
        }
    }

    /// Constructs a round input driven by a [`MulControl`] signal.
    ///
    /// This is the input for each exponent-bit-processing cycle.
    /// Use [`MulControl::from`] to convert from
    /// [`minroot_cat::schedule::RoundControl`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minroot_hdl::engine::EngineInput;
    /// use minroot_hdl::types::MulControl;
    /// use minroot_cat::schedule::RoundControl;
    ///
    /// let input = EngineInput::round(RoundControl::SquareAndMultiply.into());
    /// assert!(!input.load);
    /// assert_eq!(input.control, MulControl::Multiply);
    /// ```
    #[must_use]
    pub fn round(control: MulControl) -> Self {
        Self {
            control,
            load: false,
            load_value: PolySignal::default(),
            load_bits: 0,
        }
    }
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
    ///
    /// On a **load** cycle, the engine initializes the accumulator to
    /// the polynomial form of `1` (the multiplicative identity) and
    /// stores the input value as the base for exponentiation.
    ///
    /// On an **active** cycle, the engine performs one square-and-multiply
    /// step: squares the accumulator, conditionally multiplies by the base
    /// (per the [`MulControl`] signal), reduces, and decrements the bit
    /// position.
    ///
    /// When **idle**, the engine returns the final accumulator with
    /// `done = true`.
    ///
    /// # Examples
    ///
    /// ```
    /// use minroot_core::field::Curve;
    /// use minroot_hdl::engine::{FifthRootEngine, EngineInput};
    /// use minroot_hdl::types::{PipelineState, PolySignal};
    ///
    /// let engine = FifthRootEngine::new(Curve::Pallas);
    /// let (output, _) = engine.tick(EngineInput::default(), PipelineState::default());
    /// assert!(output.done);
    /// ```
    #[must_use]
    pub fn tick(
        &self,
        input: EngineInput,
        state: PipelineState,
    ) -> (EngineOutput, PipelineState) {
        if input.load {
            // Initialize accumulator to the multiplicative identity (one)
            let one_signal = PolySignal::from_poly_element(
                &PolyElement::from_field(FieldElement::one(self.curve)),
            );
            let new_state = PipelineState::advance(
                input.load_value,
                one_signal,
                input.load_bits,
                true,
            );
            let output = EngineOutput {
                accum: one_signal,
                done: false,
            };
            (output, new_state)
        } else if !state.is_active() {
            // Idle: computation complete
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

            // Advance bit position: bit_position counts remaining
            // rounds.  When it reaches 1, this is the final active
            // round.
            let remaining = state.bit_position().saturating_sub(1);
            let active = remaining > 0;

            let next_state = PipelineState::advance(
                *state.base(),
                result,
                remaining,
                active,
            );

            let output = EngineOutput {
                accum: result,
                done: !active,
            };
            (output, next_state)
        }
    }
}

/// The fifth-root engine specialized to the **Pallas** curve.
///
/// Implements [`Synchronous`] for cycle-accurate behavioral simulation
/// of the square-and-multiply exponentiation used in the `MinRoot`
/// fifth-root computation.
///
/// # End-to-End Simulation
///
/// ```
/// # fn main() -> Result<(), minroot_core::error::Error> {
/// use minroot_core::field::{Curve, FieldElement};
/// use minroot_core::polynomial::PolyElement;
/// use minroot_hdl::circuit::Synchronous;
/// use minroot_hdl::engine::{PallasEngine, EngineInput};
/// use minroot_hdl::types::{MulControl, PolySignal};
///
/// let x = FieldElement::from_u64(7, Curve::Pallas);
/// let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));
///
/// // Compute x^5: exponent 5 = binary 101 (3 bits)
/// let inputs: Vec<EngineInput> = vec![
///     EngineInput::load(signal, 3),
///     EngineInput::round(MulControl::Multiply), // bit 2 = 1
///     EngineInput::round(MulControl::Bypass),    // bit 1 = 0
///     EngineInput::round(MulControl::Multiply), // bit 0 = 1
/// ];
///
/// let (outputs, final_state) = PallasEngine::simulate(inputs);
/// assert!(outputs.last().is_some_and(|o| o.done));
///
/// let result = final_state.accum().to_poly_element(Curve::Pallas)?.to_field()?;
/// assert_eq!(result, x * x * x * x * x);
/// # Ok(())
/// # }
/// ```
pub struct PallasEngine;

impl Synchronous for PallasEngine {
    type Input = EngineInput;
    type Output = EngineOutput;
    type State = PipelineState;

    fn kernel(input: Self::Input, state: Self::State) -> (Self::Output, Self::State) {
        FifthRootEngine::new(Curve::Pallas).tick(input, state)
    }
}

/// The fifth-root engine specialized to the **Vesta** curve.
///
/// Identical to [`PallasEngine`] but uses the Vesta field modulus.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), minroot_core::error::Error> {
/// use minroot_core::field::{Curve, FieldElement};
/// use minroot_core::polynomial::PolyElement;
/// use minroot_hdl::circuit::Synchronous;
/// use minroot_hdl::engine::{VestaEngine, EngineInput};
/// use minroot_hdl::types::{MulControl, PolySignal};
///
/// let x = FieldElement::from_u64(13, Curve::Vesta);
/// let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));
///
/// // Compute x^3: exponent 3 = binary 11 (2 bits)
/// let inputs: Vec<EngineInput> = vec![
///     EngineInput::load(signal, 2),
///     EngineInput::round(MulControl::Multiply),
///     EngineInput::round(MulControl::Multiply),
/// ];
///
/// let (outputs, final_state) = VestaEngine::simulate(inputs);
/// assert!(outputs.last().is_some_and(|o| o.done));
///
/// let result = final_state.accum().to_poly_element(Curve::Vesta)?.to_field()?;
/// assert_eq!(result, x * x * x);
/// # Ok(())
/// # }
/// ```
pub struct VestaEngine;

impl Synchronous for VestaEngine {
    type Input = EngineInput;
    type Output = EngineOutput;
    type State = PipelineState;

    fn kernel(input: Self::Input, state: Self::State) -> (Self::Output, Self::State) {
        FifthRootEngine::new(Curve::Vesta).tick(input, state)
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
        let input = EngineInput::load(
            PolySignal::from_coeffs(core::array::from_fn(|i| {
                if i == 0 { Bits::new_wrapping(7u128) } else { Bits::new_wrapping(0u128) }
            })),
            254,
        );
        let (output, state) = engine.tick(input, PipelineState::default());
        assert!(!output.done);
        assert!(state.is_active());
        assert_eq!(state.bit_position(), 254);
    }

    #[test]
    fn engine_load_initializes_accum_to_one() {
        let engine = FifthRootEngine::new(Curve::Pallas);
        let input = EngineInput::load(PolySignal::default(), 254);
        let (_, state) = engine.tick(input, PipelineState::default());
        let one = PolySignal::from_poly_element(
            &PolyElement::from_field(FieldElement::one(Curve::Pallas)),
        );
        assert_eq!(*state.accum(), one);
    }

    #[test]
    fn engine_round_constructor() {
        let input = EngineInput::round(MulControl::Multiply);
        assert!(!input.load);
        assert_eq!(input.control, MulControl::Multiply);
    }

    /// Computes `x^3` via a 2-bit exponent (binary `11`) and verifies
    /// against field arithmetic.  Uses a short exponent to stay within
    /// the 34-bit partial-product range of the current polynomial pipeline.
    #[test]
    fn pallas_engine_cubed() -> Result<(), minroot_core::error::Error> {
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));

        // Exponent 3 = binary 11 (2 bits, both set)
        let inputs: Vec<EngineInput> = vec![
            EngineInput::load(signal, 2),
            EngineInput::round(MulControl::Multiply), // MSB = 1
            EngineInput::round(MulControl::Multiply), // LSB = 1
        ];

        let (outputs, final_state) = PallasEngine::simulate(inputs);
        assert!(outputs.last().is_some_and(|o| o.done));

        let result = final_state.accum().to_poly_element(Curve::Pallas)?.to_field()?;
        assert_eq!(result, x * x * x);
        Ok(())
    }

    /// Computes `x^5` via a 3-bit exponent (binary `101`) and verifies
    /// the fifth-power identity: `fifth_root(x)^5 == x`.
    #[test]
    fn vesta_engine_fifth_power() -> Result<(), minroot_core::error::Error> {
        let x = FieldElement::from_u64(13, Curve::Vesta);
        let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));

        // Exponent 5 = binary 101 (3 bits)
        let inputs: Vec<EngineInput> = vec![
            EngineInput::load(signal, 3),
            EngineInput::round(MulControl::Multiply), // bit 2 = 1
            EngineInput::round(MulControl::Bypass),    // bit 1 = 0
            EngineInput::round(MulControl::Multiply), // bit 0 = 1
        ];

        let (outputs, final_state) = VestaEngine::simulate(inputs);
        assert!(outputs.last().is_some_and(|o| o.done));

        let result = final_state.accum().to_poly_element(Curve::Vesta)?.to_field()?;
        assert_eq!(result, x * x * x * x * x);
        Ok(())
    }
}
