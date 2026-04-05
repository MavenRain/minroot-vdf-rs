//! Project-wide error type for `minroot-core`.

use core::fmt;

/// Errors arising from field arithmetic and `MinRoot` computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A value exceeds the field modulus.
    OutOfRange {
        /// The limb index or context where overflow was detected.
        context: &'static str,
    },
    /// Polynomial coefficient count does not match the expected width.
    CoefficientCountMismatch {
        /// Number of coefficients provided.
        got: usize,
        /// Number of coefficients expected.
        expected: usize,
    },
    /// Attempted to invert zero in the field.
    DivisionByZero,
    /// Iteration count must be positive.
    ZeroIterations,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { context } => {
                write!(f, "value out of field range in {context}")
            }
            Self::CoefficientCountMismatch { got, expected } => {
                write!(
                    f,
                    "coefficient count mismatch: got {got}, expected {expected}"
                )
            }
            Self::DivisionByZero => write!(f, "division by zero in field"),
            Self::ZeroIterations => {
                write!(f, "iteration count must be positive")
            }
        }
    }
}

impl core::error::Error for Error {}
