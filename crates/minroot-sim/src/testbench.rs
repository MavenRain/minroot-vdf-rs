//! hdl-cat [`Testbench`] simulation for MinRoot circuits.
//!
//! Provides helpers to simulate MinRoot circuit blocks using hdl-cat's
//! cycle-accurate [`Testbench`] and generate VCD waveform traces for
//! debugging and verification.
//!
//! # Combinational Simulation
//!
//! [`simulate_combinational`] wraps a stateless [`CircuitArrow`] in a
//! [`Sync`] machine and runs it through the testbench.  Each [`BitSeq`]
//! in the input vector drives one simulation cycle.
//!
//! # VCD Trace Generation
//!
//! [`trace_combinational_vcd`] and [`trace_sync_vcd`] produce Value
//! Change Dump (VCD) output suitable for viewing in GTKWave or
//! similar waveform viewers.
//!
//! # Examples
//!
//! Simulate a coefficient adder:
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
//! Generate a VCD trace for waveform viewing:
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
//! # Ok(())
//! # }
//! ```
//!
//! [`CircuitArrow`]: hdl_cat::circuit::CircuitArrow
//! [`Sync`]: hdl_cat::sync::Sync
//! [`Testbench`]: hdl_cat::sim::Testbench
//! [`BitSeq`]: hdl_cat::kind::BitSeq

use hdl_cat::circuit::{CircuitArrow, Object};
use hdl_cat::kind::BitSeq;
use hdl_cat::sim::{Testbench, TimedSample};
use hdl_cat::sync::Sync;
use hdl_cat::Error;
use comp_cat_rs::effect::io::Io;

/// Simulates a combinational [`CircuitArrow`] over a sequence of inputs.
///
/// Wraps the arrow in a stateless [`Sync`] machine via
/// [`Sync::lift_comb`] and runs it through hdl-cat's [`Testbench`].
/// Each [`BitSeq`] in `inputs` drives one clock cycle.
///
/// The returned [`Io`] produces the simulation results when run.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use hdl_cat::bits::Bits;
/// use hdl_cat::kind::{BitSeq, Hw};
/// use minroot_hdl::synthesis::coeff_adder;
/// use minroot_sim::testbench::simulate_combinational;
///
/// let arrow = coeff_adder()?;
/// let a = Bits::<17>::new_wrapping(3u128);
/// let b = Bits::<17>::new_wrapping(5u128);
/// let input = a.to_bits_seq().concat(b.to_bits_seq());
/// let samples = simulate_combinational(arrow, vec![input]).run()?;
/// assert_eq!(samples.len(), 1);
/// # Ok(())
/// # }
/// ```
///
/// [`Sync::lift_comb`]: hdl_cat::sync::Sync::lift_comb
/// [`Testbench`]: hdl_cat::sim::Testbench
pub fn simulate_combinational<I, O>(
    arrow: CircuitArrow<I, O>,
    inputs: Vec<BitSeq>,
) -> Io<Error, Vec<TimedSample<BitSeq>>>
where
    I: Object + 'static,
    O: Object + 'static,
{
    Testbench::new(Sync::lift_comb(arrow)).run(inputs)
}

/// Generates a VCD waveform trace for a combinational circuit.
///
/// Wraps the [`CircuitArrow`] in a stateless [`Sync`] machine and
/// produces a VCD (Value Change Dump) string for viewing in
/// waveform viewers such as GTKWave.
///
/// # Errors
///
/// Returns [`Error`] if simulation or trace generation fails.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use hdl_cat::bits::Bits;
/// use hdl_cat::kind::{BitSeq, Hw};
/// use minroot_hdl::synthesis::coeff_adder;
/// use minroot_sim::testbench::trace_combinational_vcd;
///
/// let arrow = coeff_adder()?;
/// let a = Bits::<17>::new_wrapping(7u128);
/// let b = Bits::<17>::new_wrapping(11u128);
/// let input = a.to_bits_seq().concat(b.to_bits_seq());
/// let vcd = trace_combinational_vcd(arrow, vec![input])?;
/// assert!(vcd.contains("$end"));
/// # Ok(())
/// # }
/// ```
pub fn trace_combinational_vcd<I, O>(
    arrow: CircuitArrow<I, O>,
    inputs: Vec<BitSeq>,
) -> Result<String, Error>
where
    I: Object + 'static,
    O: Object + 'static,
{
    let machine = Sync::lift_comb(arrow);
    hdl_cat::sim::trace_to_string(&machine, inputs)
}

/// Simulates a synchronous [`Sync`] machine over a sequence of inputs.
///
/// Each [`BitSeq`] in `inputs` drives one clock cycle.  The returned
/// [`Io`] produces the timed output samples when run.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use hdl_cat::kind::BitSeq;
/// use minroot_sim::testbench::simulate_sync;
///
/// let counter = hdl_cat::std_lib::counter::<4>()?;
/// let inputs = vec![BitSeq::new(); 8];
/// let samples = simulate_sync(counter, inputs).run()?;
/// assert_eq!(samples.len(), 8);
/// # Ok(())
/// # }
/// ```
pub fn simulate_sync<S, I, O>(
    machine: Sync<S, I, O>,
    inputs: Vec<BitSeq>,
) -> Io<Error, Vec<TimedSample<BitSeq>>>
where
    S: 'static,
    I: 'static,
    O: 'static,
{
    Testbench::new(machine).run(inputs)
}

/// Generates a VCD waveform trace for a synchronous machine.
///
/// # Errors
///
/// Returns [`Error`] if simulation or trace generation fails.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use hdl_cat::kind::BitSeq;
/// use minroot_sim::testbench::trace_sync_vcd;
///
/// let counter = hdl_cat::std_lib::counter::<4>()?;
/// let inputs = vec![BitSeq::new(); 4];
/// let vcd = trace_sync_vcd(&counter, inputs)?;
/// assert!(vcd.contains("$end"));
/// # Ok(())
/// # }
/// ```
pub fn trace_sync_vcd<S, I, O>(
    machine: &Sync<S, I, O>,
    inputs: Vec<BitSeq>,
) -> Result<String, Error> {
    hdl_cat::sim::trace_to_string(machine, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdl_cat::bits::Bits;
    use hdl_cat::kind::Hw;

    #[test]
    fn simulate_coeff_adder() -> Result<(), Error> {
        let arrow = minroot_hdl::synthesis::coeff_adder()?;
        let a = Bits::<17>::new_wrapping(10u128);
        let b = Bits::<17>::new_wrapping(20u128);
        let input = a.to_bits_seq().concat(b.to_bits_seq());
        let samples = simulate_combinational(arrow, vec![input]).run()?;
        assert_eq!(samples.len(), 1);
        samples.iter().try_for_each(|sample| {
            Bits::<17>::from_bits_seq(sample.value()).map(|result| {
                assert_eq!(result.to_u128(), 30);
            })
        })
    }

    #[test]
    fn trace_coeff_adder_generates_vcd() -> Result<(), Error> {
        let arrow = minroot_hdl::synthesis::coeff_adder()?;
        let a = Bits::<17>::new_wrapping(5u128);
        let b = Bits::<17>::new_wrapping(3u128);
        let input = a.to_bits_seq().concat(b.to_bits_seq());
        let vcd = trace_combinational_vcd(arrow, vec![input])?;
        assert!(vcd.contains("$end"));
        Ok(())
    }

    #[test]
    fn simulate_sync_counter() -> Result<(), Error> {
        let counter = hdl_cat::std_lib::counter::<4>()?;
        let inputs = vec![BitSeq::new(); 8];
        let samples = simulate_sync(counter, inputs).run()?;
        assert_eq!(samples.len(), 8);
        Ok(())
    }
}
