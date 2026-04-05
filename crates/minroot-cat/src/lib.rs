//! Categorical pipeline abstractions for `MinRoot` VDF hardware.
//!
//! Models the `MinRoot` pipeline ring using structures from
//! [`comp_cat_rs`]:
//!
//! - **Pipeline stages** as morphisms in a [`Category`](comp_cat_rs::foundation::category::Category)
//! - **Ring topology** via a [`Traced`](traced::Traced) monoidal category (feedback loops)
//! - **N-way interleaving** as a product [`Functor`](comp_cat_rs::foundation::functor::Functor)
//! - **Exponent scanning** as a catamorphism over [`Stream`](comp_cat_rs::effect::stream::Stream)
//! - **FPGA/ASIC targeting** via the [`Target`](target::Target) trait
//!
//! # Examples
//!
//! Build the single-round pipeline path and inspect its structure:
//!
//! ```
//! use minroot_cat::pipeline::{PipelineVertex, single_round_path};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let path = single_round_path()?;
//! assert_eq!(path.source(), PipelineVertex::PreSquare.to_vertex());
//! assert_eq!(path.target(), PipelineVertex::PostReduce.to_vertex());
//! assert_eq!(path.len(), 3); // SQR -> MUL -> RED
//! # Ok(())
//! # }
//! # run().unwrap_or(());
//! ```

pub mod interleave;
pub mod pipeline;
pub mod schedule;
pub mod target;
pub mod traced;
