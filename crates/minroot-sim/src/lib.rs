//! Simulation and verification harness for `MinRoot` VDF hardware.
//!
//! Uses [`comp_cat_rs::effect::io::Io`] for simulation orchestration,
//! [`comp_cat_rs::effect::stream::Stream`] for test vector generation,
//! and compares RHDL hardware outputs against the
//! [`minroot_core`] reference model.
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

pub mod harness;
pub mod vectors;
pub mod verify;
