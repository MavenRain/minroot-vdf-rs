//! Clock gating abstraction for FPGA and ASIC targets.
//!
//! The `MinRoot` hardware uses clock gating to reduce power
//! consumption when the engine is idle.  FPGA and ASIC targets
//! implement this differently:
//!
//! - **FPGA**: clock enable (CE) on flip-flops
//! - **ASIC**: integrated clock gating (ICG) cell
//!
//! This module provides the behavioral model; the actual cell
//! selection is driven by [`minroot_cat::target::Target`].

use crate::circuit::Combinational;

/// Clock gate input: the enable signal and whether the clock is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ClockGateInput {
    /// Whether the gated clock domain should be active.
    pub enable: bool,
}

/// Clock gate output: the gated clock enable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ClockGateOutput {
    /// Whether the downstream logic should capture new data.
    pub clock_enable: bool,
}

/// Behavioral clock gate model.
///
/// In simulation, this is just a pass-through of the enable signal.
/// In synthesis, the target-specific implementation is selected.
pub struct ClockGate;

impl Combinational for ClockGate {
    type Input = ClockGateInput;
    type Output = ClockGateOutput;

    fn eval(input: Self::Input) -> Self::Output {
        ClockGateOutput {
            clock_enable: input.enable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_passes_enable() {
        let on = ClockGate::eval(ClockGateInput { enable: true });
        assert!(on.clock_enable);

        let off = ClockGate::eval(ClockGateInput { enable: false });
        assert!(!off.clock_enable);
    }
}
