//! Extension utilities for [`hdl_cat::bits::Bits`].
//!
//! Thin wrappers for constructing and extracting values from
//! `Bits<N>` that avoid direct field access.

use hdl_cat::bits::Bits;

/// Extracts the underlying `u128` value from a `Bits<N>`.
#[must_use]
pub fn to_u128<const N: usize>(bits: Bits<N>) -> u128 {
    bits.to_u128()
}

/// Creates a zero `Bits<N>` value.
pub fn zero<const N: usize>() -> Bits<N> {
    Bits::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small() {
        let val: Bits<8> = Bits::new_wrapping(42u128);
        assert_eq!(to_u128(val), 42);
    }

    #[test]
    fn roundtrip_zero() {
        let val: Bits<17> = zero();
        assert_eq!(to_u128(val), 0);
    }

    #[test]
    fn roundtrip_17bit() {
        let val: Bits<17> = Bits::new_wrapping(0x1_FFFFu128);
        assert_eq!(to_u128(val), 0x1_FFFF);
    }
}
