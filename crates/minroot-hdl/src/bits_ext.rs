//! Extension utilities for [`rhdl::bits::Bits`].
//!
//! Thin wrappers for constructing and extracting values from
//! `Bits<N>` that avoid direct field access.

use rhdl::bits::{BitWidth, Bits, W};

/// Extracts the underlying `u128` value from a `Bits<N>`.
#[must_use]
pub fn to_u128<const N: usize>(bits: Bits<N>) -> u128
where
    W<N>: BitWidth,
{
    bits.raw()
}

/// Creates a zero `Bits<N>` value.
#[must_use]
pub fn zero<const N: usize>() -> Bits<N>
where
    W<N>: BitWidth,
{
    Bits::from(0u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small() {
        let val: Bits<8> = Bits::from(42u128);
        assert_eq!(to_u128(val), 42);
    }

    #[test]
    fn roundtrip_zero() {
        let val: Bits<17> = zero();
        assert_eq!(to_u128(val), 0);
    }

    #[test]
    fn roundtrip_17bit() {
        let val: Bits<17> = Bits::from(0x1_FFFFu128);
        assert_eq!(to_u128(val), 0x1_FFFF);
    }
}
