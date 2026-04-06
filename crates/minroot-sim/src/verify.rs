//! Reference model verification.
//!
//! Compares hdl-cat simulation outputs against the pure-Rust `MinRoot`
//! reference implementation in [`minroot_core`].  A test passes if
//! the hardware result matches the software result bit-for-bit.

use minroot_core::error::Error;
use minroot_core::field::FieldElement;
use minroot_core::minroot::{self, MinRootState};

use crate::vectors::TestVector;

/// The result of verifying a single test vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationResult {
    /// The expected final state (computed by the reference model).
    expected: MinRootState,
    /// Whether the hardware output matched expected.
    matched: bool,
}

impl VerificationResult {
    /// Constructs a verification result directly.
    #[must_use]
    pub fn new(expected: MinRootState, matched: bool) -> Self {
        Self { expected, matched }
    }

    /// Returns the expected state.
    #[must_use]
    pub fn expected(&self) -> MinRootState {
        self.expected
    }

    /// Returns whether the comparison succeeded.
    #[must_use]
    pub fn matched(&self) -> bool {
        self.matched
    }
}

/// Computes the expected output for a test vector via the reference model.
///
/// # Errors
///
/// Returns [`Error`] if the reference evaluation fails
/// (e.g., zero iterations).
pub fn expected_output(vector: &TestVector) -> Result<MinRootState, Error> {
    minroot::evaluate(vector.x, vector.y, vector.iterations)
}

/// Verifies a claimed hardware output against the reference model.
///
/// # Errors
///
/// Returns [`Error`] if the reference evaluation fails.
pub fn verify_vector(
    vector: &TestVector,
    hw_x: FieldElement,
    hw_y: FieldElement,
) -> Result<VerificationResult, Error> {
    expected_output(vector).map(|expected| VerificationResult {
        expected,
        matched: hw_x == expected.x() && hw_y == expected.y(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minroot_core::field::Curve;

    #[test]
    fn expected_matches_reference() -> Result<(), Error> {
        let tv = TestVector::new(
            FieldElement::from_u64(3, Curve::Pallas),
            FieldElement::from_u64(5, Curve::Pallas),
            4,
        );
        let expected = expected_output(&tv)?;
        let direct = minroot::evaluate(tv.x, tv.y, tv.iterations)?;
        assert_eq!(expected, direct);
        Ok(())
    }

    #[test]
    fn verify_accepts_correct() -> Result<(), Error> {
        let tv = TestVector::new(
            FieldElement::from_u64(2, Curve::Pallas),
            FieldElement::from_u64(3, Curve::Pallas),
            2,
        );
        let expected = expected_output(&tv)?;
        let result = verify_vector(&tv, expected.x(), expected.y())?;
        assert!(result.matched());
        Ok(())
    }

    #[test]
    fn verify_rejects_wrong() -> Result<(), Error> {
        let tv = TestVector::new(
            FieldElement::from_u64(2, Curve::Pallas),
            FieldElement::from_u64(3, Curve::Pallas),
            2,
        );
        let wrong = FieldElement::from_u64(999, Curve::Pallas);
        let result = verify_vector(&tv, wrong, wrong)?;
        assert!(!result.matched());
        Ok(())
    }
}
