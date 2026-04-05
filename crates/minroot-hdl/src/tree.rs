//! Wallace tree compressor for summing partial products.
//!
//! A carry-save adder (CSA) tree reduces `N` partial products to
//! two values (sum, carry) without propagating carries.  This avoids
//! the long carry chain that would result from sequential addition.
//!
//! The hardware uses behavioral tree compression (not structural
//! instantiation of specific full-adder cells), making it
//! target-agnostic for both FPGA and ASIC.

use crate::circuit::Combinational;
use crate::types::{PartialProduct, zero_pp};
#[cfg(test)]
use crate::bits_ext;

/// Carry-save pair: sum and carry vectors.
///
/// The true value is `sum + (carry << 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CarrySavePair {
    sum: PartialProduct,
    carry: PartialProduct,
}

impl CarrySavePair {
    /// Constructs a carry-save pair.
    #[must_use]
    pub fn new(sum: PartialProduct, carry: PartialProduct) -> Self {
        Self { sum, carry }
    }

    /// Returns the sum component.
    #[must_use]
    pub fn sum(&self) -> PartialProduct {
        self.sum
    }

    /// Returns the carry component.
    #[must_use]
    pub fn carry(&self) -> PartialProduct {
        self.carry
    }

    /// Resolves the carry-save pair to a single value.
    ///
    /// Computes `sum + (carry << 1)`.  This introduces a carry chain
    /// and should only be used at the final output, not in the
    /// pipeline critical path.
    #[must_use]
    pub fn resolve(self) -> PartialProduct {
        self.sum + (self.carry << 1)
    }
}

/// A 3-to-2 compressor (full adder tree building block).
///
/// Takes three inputs and produces a carry-save pair (sum, carry).
pub struct Compressor3to2;

impl Combinational for Compressor3to2 {
    type Input = (PartialProduct, PartialProduct, PartialProduct);
    type Output = CarrySavePair;

    fn eval((a, b, c): Self::Input) -> Self::Output {
        // Full adder: sum = a ^ b ^ c, carry = (a & b) | (b & c) | (a & c)
        let sum = a ^ b ^ c;
        let carry = (a & b) | (b & c) | (a & c);
        CarrySavePair::new(sum, carry)
    }
}

/// Reduces a slice of partial products to a carry-save pair
/// using recursive 3-to-2 compression.
///
/// This is the behavioral Wallace tree: it groups inputs into
/// triples, compresses each triple, and recurses on the results.
#[must_use]
pub fn compress(inputs: &[PartialProduct]) -> CarrySavePair {
    match inputs.len() {
        0 => CarrySavePair::default(),
        1 => CarrySavePair::new(
            inputs.first().copied().unwrap_or(zero_pp()),
            zero_pp(),
        ),
        2 => {
            // Half adder: sum = a XOR b, carry = a AND b
            let a = inputs.first().copied().unwrap_or(zero_pp());
            let b = inputs.get(1).copied().unwrap_or(zero_pp());
            CarrySavePair::new(a ^ b, a & b)
        }
        _ => {
            // Group into triples, compress each, collect results for next level
            let next_level: Vec<PartialProduct> = inputs
                .chunks(3)
                .flat_map(|chunk| {
                    let a = chunk.first().copied().unwrap_or(zero_pp());
                    let b = chunk.get(1).copied().unwrap_or(zero_pp());
                    match chunk.get(2).copied() {
                        Some(c) => {
                            let csp = Compressor3to2::eval((a, b, c));
                            vec![csp.sum(), csp.carry() << 1]
                        }
                        None => chunk.to_vec(),
                    }
                })
                .collect();
            compress(&next_level)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_3to2_basic() {
        let a: PartialProduct = 0b101u128.into();
        let b: PartialProduct = 0b011u128.into();
        let c: PartialProduct = 0b110u128.into();
        let csp = Compressor3to2::eval((a, b, c));
        let resolved = csp.resolve();
        assert_eq!(
            bits_ext::to_u128(resolved),
            bits_ext::to_u128(a) + bits_ext::to_u128(b) + bits_ext::to_u128(c)
        );
    }

    #[test]
    fn compress_empty() {
        let csp = compress(&[]);
        assert_eq!(bits_ext::to_u128(csp.resolve()), 0);
    }

    #[test]
    fn compress_single() {
        let pp: PartialProduct = 42u128.into();
        let csp = compress(&[pp]);
        assert_eq!(csp.sum(), pp);
    }

    #[test]
    fn compress_pair() {
        let a: PartialProduct = 10u128.into();
        let b: PartialProduct = 20u128.into();
        let csp = compress(&[a, b]);
        assert_eq!(bits_ext::to_u128(csp.resolve()), 10 + 20);
    }

    #[test]
    fn compress_multiple() {
        let inputs: Vec<PartialProduct> = (1u128..=5)
            .map(Into::into)
            .collect();
        let csp = compress(&inputs);
        assert_eq!(bits_ext::to_u128(csp.resolve()), 1 + 2 + 3 + 4 + 5);
    }
}
