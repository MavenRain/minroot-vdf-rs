//! Hardware blocks for the `MinRoot` VDF.
//!
//! Concrete hardware implementations interpreted from the categorical
//! pipeline specifications in [`minroot_cat`].  Uses [`hdl_cat::bits::Bits`]
//! for fixed-width signal types.
//!
//! # Examples
//!
//! ## Polynomial multiplication
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
//! assert_eq!(minroot_hdl::bits_ext::to_u128(product.coeff(0)), 77);
//! ```
//!
//! ## Behavioral engine simulation
//!
//! ```
//! # fn main() -> Result<(), minroot_core::error::Error> {
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_core::polynomial::PolyElement;
//! use minroot_hdl::circuit::Synchronous;
//! use minroot_hdl::engine::{PallasEngine, EngineInput};
//! use minroot_hdl::types::{MulControl, PolySignal};
//!
//! let x = FieldElement::from_u64(42, Curve::Pallas);
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
//! assert!(outputs.last().map_or(false, |o| o.done));
//!
//! let result = final_state.accum().to_poly_element(Curve::Pallas)?.to_field()?;
//! assert_eq!(result, x * x * x);
//! # Ok(())
//! # }
//! ```
//!
//! ## Verilog emission
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use minroot_hdl::synthesis::{coeff_adder, full_adder, emit_verilog, emit_sync_verilog};
//!
//! // Combinational circuit
//! let adder = coeff_adder()?;
//! let adder_v = emit_verilog(&adder, "coeff_add").run()?;
//! assert!(adder_v.contains("coeff_add"));
//!
//! // Composed circuit (single-bit full adder from XOR/AND/OR)
//! let fa = full_adder()?;
//! let fa_v = emit_verilog(&fa, "full_adder").run()?;
//! assert!(fa_v.contains("full_adder"));
//!
//! // Synchronous machine
//! let counter = hdl_cat::std_lib::counter::<8>()?;
//! let counter_v = emit_sync_verilog(&counter, "counter8").run()?;
//! assert!(counter_v.contains("counter8"));
//! # Ok(())
//! # }
//! ```

pub mod bits_ext;
pub mod circuit;
pub mod clock_gate;
pub mod engine;
pub mod poly_mul;
pub mod poly_reduce;
pub mod poly_sqr;
pub mod synthesis;
pub mod tree;
pub mod types;
