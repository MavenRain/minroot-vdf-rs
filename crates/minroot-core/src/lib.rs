//! Pure Rust reference implementation of `MinRoot` VDF field arithmetic.
//!
//! Provides software-level implementations of the Pallas and Vesta
//! prime field arithmetic, Montgomery form, redundant polynomial
//! representation, and the `MinRoot` VDF algorithm itself.
//!
//! This crate has zero dependencies and serves two purposes:
//!
//! 1. **Reference model** for verifying RHDL hardware simulations.
//! 2. **Domain types** shared across the categorical and HDL layers.
//!
//! # Examples
//!
//! Evaluate the `MinRoot` VDF for 10 iterations on Pallas:
//!
//! ```
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_core::minroot;
//!
//! let x = FieldElement::from_u64(3, Curve::Pallas);
//! let y = FieldElement::from_u64(5, Curve::Pallas);
//! let result = minroot::evaluate(x, y, 10);
//! assert!(result.is_ok());
//! ```

pub mod error;
pub mod field;
pub mod minroot;
pub mod montgomery;
pub mod polynomial;
