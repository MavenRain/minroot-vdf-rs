//! FPGA/ASIC target abstraction.
//!
//! The `MinRoot` hardware generates target-agnostic Verilog via RHDL.
//! Technology-specific cells (clock gating, multiplier hints) are
//! parameterized by the [`Target`] trait, allowing a single design
//! to produce correct output for both FPGA prototyping and ASIC
//! fabrication.
//!
//! Categorically, this is an **adjunction** between the specification
//! category (target-independent pipeline) and the implementation
//! category (target-specific hardware).  The left adjoint maps
//! abstract pipeline stages to concrete cells; the right adjoint
//! forgets implementation details back to the spec.

/// A synthesis target with technology-specific cell selection.
///
/// Implementations provide the concrete cell choices for clock gating,
/// multiplier inference, and other technology-dependent features.
pub trait Target {
    /// Human-readable name for this target (e.g., `"Xilinx UltraScale+"`, `"TSMC 12nm"`).
    fn name(&self) -> &str;

    /// Clock gating strategy for this target.
    fn clock_gating(&self) -> ClockGating;

    /// Multiplier implementation strategy.
    fn multiplier_strategy(&self) -> MultiplierStrategy;
}

/// Clock gating approach.
///
/// FPGA and ASIC handle clock gating differently at the cell level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockGating {
    /// Use a clock enable signal (FPGA style).
    ///
    /// The clock runs continuously; a CE pin on the flip-flop
    /// controls whether it captures new data.
    ClockEnable,

    /// Use an integrated clock gating cell (ASIC style).
    ///
    /// A dedicated ICG cell gates the clock tree, reducing
    /// dynamic power consumption.
    IntegratedClockGate,
}

/// Multiplier implementation strategy.
///
/// FPGAs have dedicated DSP blocks; ASICs use standard cell logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiplierStrategy {
    /// Infer DSP blocks (FPGA).
    ///
    /// The synthesis tool maps multiplications to hard DSP slices.
    DspInference,

    /// Use standard cell logic (ASIC).
    ///
    /// Multiplications are synthesized from standard gates.
    StandardCell,

    /// Behavioral (let the synthesis tool decide).
    ///
    /// No technology-specific hints; the tool chooses.
    Behavioral,
}

/// FPGA target configuration.
#[derive(Debug, Clone)]
pub struct Fpga {
    family: &'static str,
}

impl Fpga {
    /// Creates a new FPGA target for the given device family.
    #[must_use]
    pub fn new(family: &'static str) -> Self {
        Self { family }
    }

    /// Returns the device family name.
    #[must_use]
    pub fn family(&self) -> &str {
        self.family
    }
}

impl Target for Fpga {
    fn name(&self) -> &str {
        self.family
    }

    fn clock_gating(&self) -> ClockGating {
        ClockGating::ClockEnable
    }

    fn multiplier_strategy(&self) -> MultiplierStrategy {
        MultiplierStrategy::DspInference
    }
}

/// ASIC target configuration.
#[derive(Debug, Clone)]
pub struct Asic {
    process_node: &'static str,
}

impl Asic {
    /// Creates a new ASIC target for the given process node.
    #[must_use]
    pub fn new(process_node: &'static str) -> Self {
        Self { process_node }
    }

    /// Returns the process node name.
    #[must_use]
    pub fn process_node(&self) -> &str {
        self.process_node
    }
}

impl Target for Asic {
    fn name(&self) -> &str {
        self.process_node
    }

    fn clock_gating(&self) -> ClockGating {
        ClockGating::IntegratedClockGate
    }

    fn multiplier_strategy(&self) -> MultiplierStrategy {
        MultiplierStrategy::StandardCell
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fpga_uses_clock_enable() {
        let fpga = Fpga::new("Xilinx UltraScale+");
        assert_eq!(fpga.clock_gating(), ClockGating::ClockEnable);
        assert_eq!(fpga.multiplier_strategy(), MultiplierStrategy::DspInference);
    }

    #[test]
    fn asic_uses_icg() {
        let asic = Asic::new("TSMC 12nm");
        assert_eq!(asic.clock_gating(), ClockGating::IntegratedClockGate);
        assert_eq!(asic.multiplier_strategy(), MultiplierStrategy::StandardCell);
    }
}
