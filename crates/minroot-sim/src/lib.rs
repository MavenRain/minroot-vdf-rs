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
//! # hdl-cat Testbench Simulation
//!
//! The [`testbench`] module provides cycle-accurate simulation of
//! hdl-cat circuits via [`Testbench`], with VCD waveform output for
//! debugging.
//!
//! # Examples
//!
//! Run the reference-model-only simulation over three test vectors:
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
//! Simulate a combinational circuit with the hdl-cat testbench:
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
//! [`Testbench`]: hdl_cat::sim::Testbench

pub mod harness;
pub mod testbench;
pub mod vectors;
pub mod verify;
