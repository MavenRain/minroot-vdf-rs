//! Circuit abstraction traits modeling hdl-cat conventions.
//!
//! These traits capture the synchronous circuit semantics that
//! hdl-cat's [`Sync`] and [`CircuitArrow`] model structurally.
//! A synchronous circuit has:
//!
//! - **Input** (`I`): signals driven from outside
//! - **Output** (`O`): signals driven by this circuit
//! - **State** (`S`): registered state (updated on clock edge)
//!
//! The kernel function computes `(output, next_state)` from
//! `(input, current_state)` purely, with no side effects.
//!
//! [`Sync`]: hdl_cat::sync::Sync
//! [`CircuitArrow`]: hdl_cat::circuit::CircuitArrow

/// A synchronous (single-clock-domain) circuit.
///
/// Models a clocked hardware block: on each rising edge, the kernel
/// reads the current input and state, produces output and next state.
///
/// This matches hdl-cat's Mealy machine pattern:
/// ```text
/// fn kernel(input: I, state: S) -> (O, S)
/// ```
pub trait Synchronous: Sized {
    /// The input signal bundle.
    type Input: Copy;

    /// The output signal bundle.
    type Output: Copy;

    /// The registered state.
    type State: Copy + Default;

    /// Pure combinational logic: compute output and next state.
    ///
    /// This function must be **purely functional** (no side effects,
    /// no allocation, no I/O).  It represents the combinational
    /// logic between register stages.
    fn kernel(input: Self::Input, state: Self::State) -> (Self::Output, Self::State);

    /// Simulate one clock cycle.
    ///
    /// Applies the kernel to produce output and advance state.
    fn step(input: Self::Input, state: Self::State) -> (Self::Output, Self::State) {
        Self::kernel(input, state)
    }

    /// Simulate multiple clock cycles from an input iterator.
    ///
    /// Returns the sequence of outputs and the final state.
    fn simulate<I: IntoIterator<Item = Self::Input>>(
        inputs: I,
    ) -> (Vec<Self::Output>, Self::State) {
        inputs.into_iter().fold(
            (Vec::new(), Self::State::default()),
            |(outputs, state), input| {
                let (output, next_state) = Self::step(input, state);
                let next_outputs = outputs
                    .into_iter()
                    .chain(core::iter::once(output))
                    .collect();
                (next_outputs, next_state)
            },
        )
    }
}

/// A combinational (stateless) circuit.
///
/// Pure function from input to output, no clock, no state.
pub trait Combinational: Sized {
    /// The input signal bundle.
    type Input: Copy;

    /// The output signal bundle.
    type Output: Copy;

    /// Pure combinational logic.
    fn eval(input: Self::Input) -> Self::Output;
}
