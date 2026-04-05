//! The `MinRoot` Verifiable Delay Function.
//!
//! `MinRoot` is a sequential function designed for Ethereum's RANDAO:
//!
//! ```text
//! for i in 0..num_iterations:
//!     temp = x + y
//!     y    = x + i
//!     x    = fifth_root(temp)
//! ```
//!
//! The fifth root is the computationally expensive step, requiring
//! modular exponentiation by `(4p - 3) / 5`.  The function is
//! inherently sequential: each iteration depends on the previous one.

use crate::error::Error;
use crate::field::{Curve, FieldElement};

/// Input/output state of a `MinRoot` computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinRootState {
    x: FieldElement,
    y: FieldElement,
    iteration: u64,
}

impl MinRootState {
    /// Constructs an initial state from `x`, `y` coordinates.
    #[must_use]
    pub fn new(x: FieldElement, y: FieldElement) -> Self {
        debug_assert_eq!(x.curve(), y.curve());
        Self {
            x,
            y,
            iteration: 0,
        }
    }

    /// Returns the current `x` coordinate.
    #[must_use]
    pub fn x(&self) -> FieldElement {
        self.x
    }

    /// Returns the current `y` coordinate.
    #[must_use]
    pub fn y(&self) -> FieldElement {
        self.y
    }

    /// Returns the current iteration count.
    #[must_use]
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Returns the curve.
    #[must_use]
    pub fn curve(&self) -> Curve {
        self.x.curve()
    }
}

/// Performs a single `MinRoot` iteration.
///
/// ```text
/// temp = x + y
/// y'   = x + i
/// x'   = fifth_root(temp)
/// ```
#[must_use]
pub fn step(state: MinRootState) -> MinRootState {
    let i_field = FieldElement::from_u64(state.iteration, state.curve());
    let temp = state.x + state.y;
    let new_y = state.x + i_field;
    let new_x = temp.fifth_root();
    MinRootState {
        x: new_x,
        y: new_y,
        iteration: state.iteration + 1,
    }
}

/// Runs the `MinRoot` VDF for `num_iterations` steps.
///
/// Returns the final state.
///
/// # Errors
///
/// Returns [`Error::ZeroIterations`] if `num_iterations` is zero.
pub fn evaluate(
    x: FieldElement,
    y: FieldElement,
    num_iterations: u64,
) -> Result<MinRootState, Error> {
    if num_iterations == 0 {
        Err(Error::ZeroIterations)
    } else {
        let init = MinRootState::new(x, y);
        Ok((0..num_iterations).fold(init, |state, _| step(state)))
    }
}

/// Runs the `MinRoot` VDF, collecting all intermediate states.
///
/// Returns a vector of `num_iterations + 1` states (including the initial state).
///
/// # Errors
///
/// Returns [`Error::ZeroIterations`] if `num_iterations` is zero.
pub fn evaluate_trace(
    x: FieldElement,
    y: FieldElement,
    num_iterations: u64,
) -> Result<Vec<MinRootState>, Error> {
    if num_iterations == 0 {
        Err(Error::ZeroIterations)
    } else {
        let init = MinRootState::new(x, y);
        Ok((0..num_iterations)
            .fold(vec![init], |mut trace, _| {
                // The last element is always the current state to step from.
                let current = trace[trace.len() - 1];
                trace.push(step(current));
                trace
            }))
    }
}

/// Verifies a `MinRoot` VDF evaluation by re-executing from the claimed inputs.
///
/// Returns `true` if the claimed output matches the re-computed result.
///
/// # Errors
///
/// Returns [`Error::ZeroIterations`] if `num_iterations` is zero.
pub fn verify(
    x: FieldElement,
    y: FieldElement,
    num_iterations: u64,
    claimed_x: FieldElement,
    claimed_y: FieldElement,
) -> Result<bool, Error> {
    evaluate(x, y, num_iterations)
        .map(|result| result.x == claimed_x && result.y == claimed_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_deterministic() {
        let x = FieldElement::from_u64(3, Curve::Pallas);
        let y = FieldElement::from_u64(5, Curve::Pallas);
        let s1 = step(MinRootState::new(x, y));
        let s2 = step(MinRootState::new(x, y));
        assert_eq!(s1, s2);
    }

    #[test]
    fn step_modifies_state() {
        let x = FieldElement::from_u64(3, Curve::Pallas);
        let y = FieldElement::from_u64(5, Curve::Pallas);
        let init = MinRootState::new(x, y);
        let after = step(init);
        // x should change (fifth root of 8 is not 3)
        assert_ne!(after.x(), init.x());
        // y should be x + 0 = 3
        assert_eq!(after.y(), x);
        assert_eq!(after.iteration(), 1);
    }

    #[test]
    fn fifth_root_consistency() {
        // After one step: x' = fifth_root(x + y), so x'^5 = x + y
        let x = FieldElement::from_u64(3, Curve::Pallas);
        let y = FieldElement::from_u64(5, Curve::Pallas);
        let after = step(MinRootState::new(x, y));
        let x_prime = after.x();
        let x5 = x_prime * x_prime * x_prime * x_prime * x_prime;
        assert_eq!(x5, x + y);
    }

    #[test]
    fn evaluate_matches_iterated_step() {
        let x = FieldElement::from_u64(10, Curve::Pallas);
        let y = FieldElement::from_u64(20, Curve::Pallas);
        let n = 3;
        let eval_result = evaluate(x, y, n);
        let step_result = (0..n).fold(MinRootState::new(x, y), |s, _| step(s));
        assert_eq!(
            eval_result.map(|r| (r.x(), r.y())),
            Ok((step_result.x(), step_result.y()))
        );
    }

    #[test]
    fn verify_accepts_correct_result() {
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let y = FieldElement::from_u64(11, Curve::Pallas);
        let n = 2;
        let result = evaluate(x, y, n);
        assert!(
            result
                .iter()
                .all(|r| verify(x, y, n, r.x(), r.y()) == Ok(true))
        );
    }

    #[test]
    fn verify_rejects_wrong_result() {
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let y = FieldElement::from_u64(11, Curve::Pallas);
        let wrong = FieldElement::from_u64(999, Curve::Pallas);
        assert_eq!(verify(x, y, 2, wrong, wrong), Ok(false));
    }

    #[test]
    fn zero_iterations_is_error() {
        let x = FieldElement::from_u64(1, Curve::Pallas);
        let y = FieldElement::from_u64(2, Curve::Pallas);
        assert!(evaluate(x, y, 0).is_err());
    }

    #[test]
    fn trace_has_correct_length() {
        let x = FieldElement::from_u64(1, Curve::Pallas);
        let y = FieldElement::from_u64(2, Curve::Pallas);
        let trace = evaluate_trace(x, y, 3);
        assert!(trace.iter().all(|t| t.len() == 4));
    }
}
