//! `Io`-based simulation harness.
//!
//! Drives a simulation for each test vector in a stream, comparing
//! outputs against the [`minroot_core`] reference model.  The entire
//! pipeline stays inside [`Io`] and [`Stream`] combinators; effects
//! execute only when [`Io::run`] is called at the boundary.
//!
//! [`Io`]: comp_cat_rs::effect::io::Io
//! [`Stream`]: comp_cat_rs::effect::stream::Stream

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use minroot_core::minroot;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::small_seeds;
    use minroot_core::field::Curve;

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
}
