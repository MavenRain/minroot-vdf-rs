//! `Io`-based simulation harness.
//!
//! Drives a simulation for each test vector in a stream, comparing
//! outputs against the [`minroot_core`] reference model.  The entire
//! pipeline stays inside [`Io`] and [`Stream`] combinators; effects
//! execute only when [`Io::run`] is called at the boundary.
//!
//! # Reference Model Simulation
//!
//! [`run_reference_only`] processes a [`Stream`] of test vectors
//! through the pure-Rust reference model, collecting pass/fail results.
//!
//! ```
//! use minroot_core::field::Curve;
//! use minroot_sim::harness::run_reference_only;
//! use minroot_sim::vectors::small_seeds;
//!
//! let vectors = small_seeds(Curve::Pallas, 5, 2);
//! let summary = run_reference_only(vectors).run().unwrap_or_default();
//! assert_eq!(summary.total(), 5);
//! assert!(summary.all_passed());
//! ```
//!
//! # Engine Behavioral Simulation
//!
//! [`run_engine_cubed`] drives the behavioral
//! [`FifthRootEngine`](minroot_hdl::engine::FifthRootEngine) through a
//! short exponentiation (`x^3`) and verifies the result against
//! field arithmetic.
//!
//! ```
//! # fn main() -> Result<(), minroot_core::error::Error> {
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_sim::harness::run_engine_cubed;
//!
//! let x = FieldElement::from_u64(42, Curve::Pallas);
//! let result = run_engine_cubed(x)?;
//! assert!(result.matched());
//! # Ok(())
//! # }
//! ```
//!
//! [`Io`]: comp_cat_rs::effect::io::Io
//! [`Stream`]: comp_cat_rs::effect::stream::Stream

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use minroot_core::field::{Curve, FieldElement};
use minroot_core::minroot;
use minroot_core::polynomial::PolyElement;

use minroot_hdl::circuit::Synchronous;
use minroot_hdl::engine::{EngineInput, PallasEngine, VestaEngine};
use minroot_hdl::types::PolySignal;

use crate::vectors::TestVector;
use crate::verify::VerificationResult;

/// A summary of a simulation run over many test vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimSummary {
    /// Total number of vectors processed.
    total: usize,
    /// Number of vectors where the output matched the reference.
    passed: usize,
}

impl SimSummary {
    /// Returns the total count.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Returns the number of passing vectors.
    #[must_use]
    pub fn passed(&self) -> usize {
        self.passed
    }

    /// Returns the number of failing vectors.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.total.saturating_sub(self.passed)
    }

    /// Returns `true` if all vectors passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.total == self.passed
    }

    /// Folds a new verification result into the summary.
    #[must_use]
    pub fn record(self, result: &VerificationResult) -> Self {
        Self {
            total: self.total + 1,
            passed: self.passed + usize::from(result.matched()),
        }
    }
}

/// Runs the reference-model-only simulation (no hardware required).
///
/// For each test vector, computes the expected output via
/// [`minroot::evaluate`].  Any vector that fails to evaluate (e.g.,
/// zero iterations) is recorded as a failed vector in the summary.
///
/// Stays inside [`Io`] until the caller invokes `.run()`.
#[must_use]
pub fn run_reference_only(
    vectors: Stream<core::convert::Infallible, TestVector>,
) -> Io<core::convert::Infallible, SimSummary> {
    vectors.fold(
        SimSummary::default(),
        Arc::new(|summary, tv| {
            let result = minroot::evaluate(tv.x, tv.y, tv.iterations).map_or_else(
                |_| {
                    // Synthesize a failed verification with a sentinel state.
                    let sentinel = minroot::MinRootState::new(tv.x, tv.y);
                    VerificationResult::new(sentinel, false)
                },
                |state| VerificationResult::new(state, true),
            );
            summary.record(&result)
        }),
    )
}

/// Runs a behavioral cube computation (`x^3`) through the engine and
/// verifies the result against the reference model.
///
/// Drives the appropriate engine ([`PallasEngine`] or [`VestaEngine`])
/// through a 2-round square-and-multiply sequence for the exponent 3
/// (binary `11`), then compares the final accumulator against
/// `x * x * x`.
///
/// This function demonstrates the full simulation flow:
///
/// 1. Convert the input [`FieldElement`] to a [`PolySignal`]
/// 2. Build the round inputs (exponent 3 = two multiply rounds)
/// 3. Run [`Synchronous::simulate`] on the engine
/// 4. Convert the result back to a [`FieldElement`]
/// 5. Compare against the reference
///
/// # Errors
///
/// Returns [`minroot_core::error::Error`] if the polynomial-to-field
/// conversion fails (should not happen for well-formed inputs).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), minroot_core::error::Error> {
/// use minroot_core::field::{Curve, FieldElement};
/// use minroot_sim::harness::run_engine_cubed;
///
/// let result = run_engine_cubed(FieldElement::from_u64(7, Curve::Pallas))?;
/// assert!(result.matched());
///
/// let all_pass = (1u64..=5).try_fold(true, |pass, v| {
///     run_engine_cubed(FieldElement::from_u64(v, Curve::Pallas))
///         .map(|r| pass && r.matched())
/// })?;
/// assert!(all_pass);
/// # Ok(())
/// # }
/// ```
///
/// On **Vesta**:
///
/// ```
/// # fn main() -> Result<(), minroot_core::error::Error> {
/// use minroot_core::field::{Curve, FieldElement};
/// use minroot_sim::harness::run_engine_cubed;
///
/// let result = run_engine_cubed(FieldElement::from_u64(13, Curve::Vesta))?;
/// assert!(result.matched());
/// # Ok(())
/// # }
/// ```
pub fn run_engine_cubed(
    x: FieldElement,
) -> Result<VerificationResult, minroot_core::error::Error> {
    use minroot_hdl::types::MulControl;

    let curve = x.curve();
    let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));

    // Exponent 3 = binary 11 (2 bits, both set)
    let inputs: Vec<EngineInput> = vec![
        EngineInput::load(signal, 2),
        EngineInput::round(MulControl::Multiply), // MSB = 1
        EngineInput::round(MulControl::Multiply), // LSB = 1
    ];

    // Run the curve-specific engine
    let final_state = match curve {
        Curve::Pallas => PallasEngine::simulate(inputs).1,
        Curve::Vesta => VestaEngine::simulate(inputs).1,
    };

    // Convert result back to field element
    let hw_result = final_state
        .accum()
        .to_poly_element(curve)?
        .to_field()?;

    let expected = x * x * x;
    let expected_state = minroot::MinRootState::new(expected, FieldElement::zero(curve));
    let matched = hw_result == expected;
    Ok(VerificationResult::new(expected_state, matched))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::small_seeds;

    #[test]
    fn summary_record_counts_pass() {
        let state = minroot::MinRootState::new(
            minroot_core::field::FieldElement::zero(Curve::Pallas),
            minroot_core::field::FieldElement::zero(Curve::Pallas),
        );
        let result = VerificationResult::new(state, true);
        let summary = SimSummary::default().record(&result);
        assert_eq!(summary.total(), 1);
        assert_eq!(summary.passed(), 1);
    }

    #[test]
    fn summary_all_passed_when_no_failures() {
        let s = SimSummary::default();
        assert!(s.all_passed());
    }

    fn infallible<T>(r: Result<T, core::convert::Infallible>) -> T {
        r.unwrap_or_else(|e| match e {})
    }

    #[test]
    fn run_reference_produces_summary() {
        let vectors = small_seeds(Curve::Pallas, 3, 2);
        let summary = infallible(run_reference_only(vectors).run());
        assert_eq!(summary.total(), 3);
        assert!(summary.all_passed());
    }

    #[test]
    fn engine_cubed_pallas() -> Result<(), minroot_core::error::Error> {
        let x = FieldElement::from_u64(7, Curve::Pallas);
        let result = run_engine_cubed(x)?;
        assert!(result.matched());
        Ok(())
    }

    #[test]
    fn engine_cubed_vesta() -> Result<(), minroot_core::error::Error> {
        let x = FieldElement::from_u64(13, Curve::Vesta);
        let result = run_engine_cubed(x)?;
        assert!(result.matched());
        Ok(())
    }

    #[test]
    fn engine_cubed_multiple_values() -> Result<(), minroot_core::error::Error> {
        let all_pass = (1u64..=5).try_fold(true, |pass, v| {
            run_engine_cubed(FieldElement::from_u64(v, Curve::Pallas))
                .map(|r| pass && r.matched())
        })?;
        assert!(all_pass);
        Ok(())
    }
}
