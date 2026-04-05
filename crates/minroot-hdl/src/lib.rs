//! RHDL hardware blocks for the `MinRoot` VDF.
//!
//! Concrete hardware implementations interpreted from the categorical
//! pipeline specifications in [`minroot_cat`].  Uses [`rhdl::bits::Bits`]
//! for fixed-width signal types.
//!
//! # Examples
//!
//! Multiply two polynomial field elements through the hardware multiplier:
//!
//! ```
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_core::polynomial::PolyElement;
//! use minroot_hdl::circuit::Combinational;
//! use minroot_hdl::poly_mul::PolyMul;
//! use minroot_hdl::types::PolySignal;
//!
//! let x = FieldElement::from_u64(7, Curve::Pallas);
//! let y = FieldElement::from_u64(11, Curve::Pallas);
//! let px = PolySignal::from_poly_element(&PolyElement::from_field(x));
//! let py = PolySignal::from_poly_element(&PolyElement::from_field(y));
//! let product = PolyMul::eval((px, py));
//! // Coefficient 0 of 7 * 11 = 77
//! assert_eq!(minroot_hdl::bits_ext::to_u128(product.coeff(0)), 77);
//! ```

pub mod bits_ext;
pub mod circuit;
pub mod clock_gate;
pub mod engine;
pub mod poly_mul;
pub mod poly_reduce;
pub mod poly_sqr;
pub mod tree;
pub mod types;
