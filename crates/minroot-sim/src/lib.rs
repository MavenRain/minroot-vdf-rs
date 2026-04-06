//! Simulation and verification harness for `MinRoot` VDF hardware.
//!
//! Uses [`comp_cat_rs::effect::io::Io`] for simulation orchestration,
//! [`comp_cat_rs::effect::stream::Stream`] for test vector generation,
//! and compares hdl-cat hardware outputs against the
//! [`minroot_core`] reference model.
//!
//! # Reference Model Verification
//!
//! The [`harness`] module runs test vectors through the pure-Rust
//! reference implementation.  The [`verify`] module compares hardware
//! outputs against expected results.
//!
//! # Engine Behavioral Simulation
//!
//! The [`harness::run_engine_cubed`] function drives the
//! [`FifthRootEngine`](minroot_hdl::engine::FifthRootEngine) through a
//! short exponentiation and verifies the result against the reference
//! model.  This exercises the full pipeline: polynomial conversion,
//! square-and-multiply, and field-element round-trip.
//!
//! # hdl-cat Testbench Simulation
//!
//! The [`testbench`] module provides cycle-accurate simulation of
//! hdl-cat circuits via [`Testbench`], with VCD waveform output for
//! debugging.
//!
//! # Examples
//!
//! ## Reference model over test vectors
//!
//! ```
//! use minroot_core::field::Curve;
//! use minroot_sim::harness::run_reference_only;
//! use minroot_sim::vectors::small_seeds;
//!
//! let vectors = small_seeds(Curve::Pallas, 3, 2);
//! let summary = run_reference_only(vectors).run().unwrap_or_default();
//! assert_eq!(summary.total(), 3);
//! assert!(summary.all_passed());
//! ```
//!
//! ## Engine behavioral verification
//!
//! ```
//! # fn main() -> Result<(), minroot_core::error::Error> {
//! use minroot_core::field::{Curve, FieldElement};
//! use minroot_sim::harness::run_engine_cubed;
//!
//! let result = run_engine_cubed(FieldElement::from_u64(42, Curve::Pallas))?;
//! assert!(result.matched());
//! # Ok(())
//! # }
//! ```
//!
//! ## Simulate a combinational circuit with the hdl-cat testbench
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use hdl_cat::bits::Bits;
//! use hdl_cat::kind::{BitSeq, Hw};
//! use minroot_hdl::synthesis::coeff_adder;
//! use minroot_sim::testbench::simulate_combinational;
//!
//! let arrow = coeff_adder()?;
//! let a = Bits::<17>::new_wrapping(10u128);
//! let b = Bits::<17>::new_wrapping(20u128);
//! let input = a.to_bits_seq().concat(b.to_bits_seq());
//! let samples = simulate_combinational(arrow, vec![input]).run()?;
//! assert_eq!(samples.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Generate a VCD waveform trace
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use hdl_cat::bits::Bits;
//! use hdl_cat::kind::{BitSeq, Hw};
//! use minroot_hdl::synthesis::coeff_adder;
//! use minroot_sim::testbench::trace_combinational_vcd;
//!
//! let arrow = coeff_adder()?;
//! let a = Bits::<17>::new_wrapping(7u128);
//! let b = Bits::<17>::new_wrapping(11u128);
//! let input = a.to_bits_seq().concat(b.to_bits_seq());
//! let vcd = trace_combinational_vcd(arrow, vec![input])?;
//! assert!(vcd.contains("$end"));
//! // Write to file for GTKWave: std::fs::write("trace.vcd", &vcd)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Simulate a synchronous counter over multiple cycles
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use hdl_cat::kind::BitSeq;
//! use minroot_sim::testbench::{simulate_sync, trace_sync_vcd};
//!
//! let counter = hdl_cat::std_lib::counter::<4>()?;
//! let inputs = vec![BitSeq::new(); 8];
//! let samples = simulate_sync(counter, inputs).run()?;
//! assert_eq!(samples.len(), 8);
//! # Ok(())
//! # }
//! ```
//!
//! [`Testbench`]: hdl_cat::sim::Testbench

pub mod harness;
pub mod testbench;
pub mod vectors;
pub mod verify;
