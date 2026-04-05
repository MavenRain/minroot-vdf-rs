//! Traced monoidal categories for modeling feedback loops.
//!
//! A traced monoidal category extends a symmetric monoidal category
//! with a **trace** operator that models feedback: given a morphism
//! `f: A ⊗ U -> B ⊗ U`, the trace produces `Tr(f): A -> B` by
//! feeding the `U` output back to the `U` input.
//!
//! In the `MinRoot` pipeline, the trace models the ring topology:
//!
//! ```text
//!   ┌──────────────────────────────────────────────────┐
//!   │                                                  │
//!   │   PreSqr ──[SQR]──> PostSqr ──[MUL]──> PostMul  │
//!   │     ▲                                     │      │
//!   │     │                                     │      │
//!   │     └──────── PostRed <──[RED]────────────┘      │
//!   │                                                  │
//!   └──────────────────────────────────────────────────┘
//! ```
//!
//! The feedback variable `U` is the pipeline state (`PolyElement`).
//! The trace introduces registers to hold the state across clock cycles,
//! turning the combinational loop into a clocked pipeline.
//!
//! # Extending `comp-cat-rs`
//!
//! `comp-cat-rs` does not include a `Traced` trait (it models
//! categories, monoidal structure, and braiding, but not traced
//! structure).  This module extends the hierarchy.

use comp_cat_rs::foundation::monoidal::Symmetric;

/// A traced symmetric monoidal category.
///
/// Adds the trace operator to a [`Symmetric`] monoidal category.
///
/// # Laws (from Joyal, Street, Verity; verified conceptually via Lean 4)
///
/// - **Naturality in A**: `Tr(f . (g ⊗ id_U)) = Tr(f) . g`
/// - **Naturality in B**: `Tr((h ⊗ id_U) . f) = h . Tr(f)`
/// - **Dinaturality in U**: (sliding) `Tr((id_B ⊗ g) . f) = Tr(f . (id_A ⊗ g))`
/// - **Vanishing I**: `Tr_{I}(f) = f` (trace over the unit is identity)
/// - **Vanishing II**: `Tr_{U⊗V}(f) = Tr_U(Tr_V(f'))` (iterated trace)
/// - **Superposing**: `Tr(f) ⊗ g = Tr(f ⊗ g)` (external operations pass through)
/// - **Yanking**: `Tr(σ_{U,U}) = id_U` (braiding traced is identity)
pub trait Traced: Symmetric {
    /// The trace operator.
    ///
    /// Given a morphism `f: A ⊗ U -> B ⊗ U`, produces `Tr(f): A -> B`
    /// by feeding the `U` component back.
    ///
    /// In hardware: introduces a register bank holding `U` and connects
    /// the output back to the input, creating a clocked pipeline stage.
    fn trace<A, B, U>(
        f: Self::Hom<Self::Tensor<A, U>, Self::Tensor<B, U>>,
    ) -> Self::Hom<A, B>
    where
        A: Into<Self>,
        B: Into<Self>,
        U: Into<Self>;
}

/// A pipeline ring built from the trace of a single-round morphism.
///
/// The ring takes a morphism `round: State ⊗ Control -> State ⊗ Control`
/// (one round of SQR ; MUL ; RED with exponent-bit control) and traces
/// out the `State` component to produce the iterated pipeline.
///
/// This is a concrete model, not a trait implementation, because Rust's
/// type system cannot fully express the traced monoidal laws at the
/// type level.  The structure is verified by the categorical semantics.
#[derive(Debug, Clone)]
pub struct PipelineRing<RoundMorphism> {
    round: RoundMorphism,
    num_rounds: usize,
}

impl<R> PipelineRing<R> {
    /// Constructs a pipeline ring from a single-round morphism.
    ///
    /// `num_rounds` is the number of exponent bits to process (typically 258
    /// for the `MinRoot` fifth root: 254 exponent bits + 4 overhead cycles).
    #[must_use]
    pub fn new(round: R, num_rounds: usize) -> Self {
        Self { round, num_rounds }
    }

    /// Returns the single-round morphism.
    #[must_use]
    pub fn round(&self) -> &R {
        &self.round
    }

    /// Returns the number of rounds.
    #[must_use]
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

/// Number of clock cycles for one fifth-root computation.
///
/// 254 exponent bits + 4 cycles overhead (init, finalize, etc.) = 258.
pub const FIFTH_ROOT_CYCLES: usize = 258;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_ring_stores_round_count() {
        let ring = PipelineRing::new("mock_round", FIFTH_ROOT_CYCLES);
        assert_eq!(ring.num_rounds(), FIFTH_ROOT_CYCLES);
        assert_eq!(*ring.round(), "mock_round");
    }
}
