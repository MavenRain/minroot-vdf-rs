//! Synthesis circuit builders and Verilog emission for `MinRoot` hardware blocks.
//!
//! Provides [`CircuitArrow`]-based circuit builders for the fundamental
//! arithmetic operations in the `MinRoot` polynomial pipeline, and generic
//! helpers for emitting synthesizable Verilog from any hdl-cat circuit.
//!
//! # Circuit Builders
//!
//! | Builder | Width | Operation | Role |
//! |---|---|---|---|
//! | [`coeff_adder`] | 17-bit | `a + b` | Polynomial coefficient addition |
//! | [`pp_xor`] | 34-bit | `a ^ b` | Carry-save sum component |
//! | [`pp_and`] | 34-bit | `a & b` | Carry-save carry component |
//! | [`pp_or`] | 34-bit | `a \| b` | Carry-save carry combiner |
//! | [`full_adder`] | 1-bit | full adder | Composed XOR/AND/OR circuit |
//!
//! # Verilog Emission
//!
//! [`emit_verilog`] emits any combinational [`CircuitArrow`] as a
//! Verilog module.  [`emit_sync_verilog`] handles sequential [`Sync`]
//! machines with state registers.  [`emit_verilog_to_file`] writes
//! the Verilog directly to disk.
//!
//! # Examples
//!
//! Build a coefficient adder and emit its Verilog:
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use minroot_hdl::synthesis::{coeff_adder, emit_verilog};
//!
//! let arrow = coeff_adder()?;
//! let verilog_text = emit_verilog(&arrow, "coeff_add").run()?;
//! assert!(verilog_text.contains("coeff_add"));
//! # Ok(())
//! # }
//! ```
//!
//! Use the composed full-adder circuit and emit its Verilog:
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use minroot_hdl::synthesis::{full_adder, emit_verilog};
//!
//! let fa = full_adder()?;
//! let verilog_text = emit_verilog(&fa, "full_adder").run()?;
//! assert!(verilog_text.contains("full_adder"));
//! # Ok(())
//! # }
//! ```
//!
//! Emit a synchronous counter as Verilog:
//!
//! ```
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use minroot_hdl::synthesis::emit_sync_verilog;
//!
//! let counter = hdl_cat::std_lib::counter::<8>()?;
//! let verilog_text = emit_sync_verilog(&counter, "counter8").run()?;
//! assert!(verilog_text.contains("module counter8"));
//! # Ok(())
//! # }
//! ```
//!
//! Write Verilog to a file:
//!
//! ```no_run
//! # fn main() -> Result<(), hdl_cat::Error> {
//! use minroot_hdl::synthesis::{coeff_adder, emit_verilog_to_file};
//!
//! let arrow = coeff_adder()?;
//! emit_verilog_to_file(&arrow, "coeff_add", "build/coeff_add.v").run()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Writing Custom Kernels
//!
//! For more complex combinational logic, hdl-cat's `#[kernel]` macro
//! lifts a pure Rust function into a [`CircuitArrow`] builder.  The
//! macro rewrites the function into a nullary builder returning
//! `Result<CircuitArrow<..>, Error>`:
//!
//! ```text
//! use hdl_cat::kernel;
//! use hdl_cat::bits::Bits;
//!
//! #[kernel]
//! fn my_adder(a: Bits<8>, b: Bits<8>) -> Bits<8> {
//!     a + b
//! }
//!
//! // After expansion:
//! //   my_adder() -> Result<CircuitArrow<CircuitTensor<Obj<Bits<8>>, Obj<Bits<8>>>, Obj<Bits<8>>>, Error>
//! let arrow = my_adder()?;
//! let verilog = emit_verilog(&arrow, "my_adder").run()?;
//! ```
//!
//! [`CircuitArrow`]: hdl_cat::circuit::CircuitArrow
//! [`Sync`]: hdl_cat::sync::Sync

use hdl_cat::bits::Bits;
use hdl_cat::circuit::{CircuitArrow, CircuitTensor, Obj, gates};
use hdl_cat::sync::Sync;
use hdl_cat::verilog;
use hdl_cat::Error;
use comp_cat_rs::effect::io::Io;

/// A binary arithmetic circuit on 17-bit polynomial coefficients.
pub type CoeffBinArrow = CircuitArrow<
    CircuitTensor<Obj<Bits<17>>, Obj<Bits<17>>>,
    Obj<Bits<17>>,
>;

/// A binary arithmetic circuit on 34-bit partial products.
pub type PpBinArrow = CircuitArrow<
    CircuitTensor<Obj<Bits<34>>, Obj<Bits<34>>>,
    Obj<Bits<34>>,
>;

/// Builds a 17-bit coefficient adder circuit.
///
/// This is the fundamental arithmetic building block for polynomial
/// coefficient-wise addition in the `MinRoot` pipeline.
///
/// # Errors
///
/// Returns [`Error`] if the circuit IR cannot be constructed.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// let arrow = minroot_hdl::synthesis::coeff_adder()?;
/// assert!(!arrow.graph().wires().is_empty());
/// # Ok(())
/// # }
/// ```
pub fn coeff_adder() -> Result<CoeffBinArrow, Error> {
    gates::add_bits::<17>()
}

/// Builds a 34-bit XOR circuit for carry-save addition.
///
/// In a 3-to-2 compressor (Wallace tree building block), the sum
/// output is `a ^ b ^ c`.  This builds a single 2-input XOR gate;
/// compose two instances to build the full 3-input CSA sum.
///
/// # Errors
///
/// Returns [`Error`] if the circuit IR cannot be constructed.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// let arrow = minroot_hdl::synthesis::pp_xor()?;
/// assert!(!arrow.graph().wires().is_empty());
/// # Ok(())
/// # }
/// ```
pub fn pp_xor() -> Result<PpBinArrow, Error> {
    gates::xor_bits::<34>()
}

/// Builds a 34-bit AND circuit for carry-save carry generation.
///
/// Computes one of the three terms in the carry equation:
/// `carry = (a & b) | (b & c) | (a & c)`.
///
/// # Errors
///
/// Returns [`Error`] if the circuit IR cannot be constructed.
pub fn pp_and() -> Result<PpBinArrow, Error> {
    gates::and_bits::<34>()
}

/// Builds a 34-bit OR circuit for combining carry-save carry terms.
///
/// Combines the three AND terms in the carry equation:
/// `carry = (a & b) | (b & c) | (a & c)`.
///
/// # Errors
///
/// Returns [`Error`] if the circuit IR cannot be constructed.
pub fn pp_or() -> Result<PpBinArrow, Error> {
    gates::or_bits::<34>()
}

/// Emits a combinational [`CircuitArrow`] as a Verilog module.
///
/// The returned [`Io`] produces the full Verilog source text when
/// executed via [`.run()`](comp_cat_rs::effect::io::Io::run).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use minroot_hdl::synthesis::{coeff_adder, emit_verilog};
///
/// let arrow = coeff_adder()?;
/// let verilog_text = emit_verilog(&arrow, "coeff_add").run()?;
/// assert!(verilog_text.contains("coeff_add"));
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn emit_verilog<A, B>(
    arrow: &CircuitArrow<A, B>,
    name: &str,
) -> Io<Error, String> {
    verilog::emit_graph(arrow.graph(), name, arrow.inputs(), arrow.outputs())
        .flat_map(|module| module.render())
}

/// Emits a synchronous [`Sync`] machine as a Verilog module.
///
/// Produces a clocked Verilog module with state registers, input
/// ports, and output ports.  The returned [`Io`] produces the full
/// Verilog source text when executed.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use minroot_hdl::synthesis::emit_sync_verilog;
///
/// let counter = hdl_cat::std_lib::counter::<8>()?;
/// let verilog_text = emit_sync_verilog(&counter, "counter8").run()?;
/// assert!(verilog_text.contains("counter8"));
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn emit_sync_verilog<S, I, O>(
    machine: &Sync<S, I, O>,
    name: &str,
) -> Io<Error, String> {
    verilog::emit_sync_graph(
        machine.graph(),
        name,
        machine.state_wire_count(),
        machine.input_wires(),
        machine.output_wires(),
        machine.initial_state(),
    )
    .flat_map(|module| module.render())
}

/// Builds a single-bit full-adder circuit.
///
/// Wraps [`hdl_cat::std_lib::full_adder`], which composes XOR, AND,
/// and OR gates at the IR level:
///
/// ```text
/// sum  = a ^ b ^ cin
/// cout = (a & b) | (cin & (a ^ b))
/// ```
///
/// The resulting circuit takes three 1-bit inputs `((a, b), cin)` and
/// produces two 1-bit outputs `(sum, cout)`.  This is a good example
/// of a composed combinational block suitable for Verilog emission.
///
/// # Errors
///
/// Returns [`Error`] if the IR builder rejects an instruction.
///
/// # Examples
///
/// Build the full adder and emit its Verilog:
///
/// ```
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use minroot_hdl::synthesis::{full_adder, emit_verilog};
///
/// let fa = full_adder()?;
/// let verilog_text = emit_verilog(&fa, "full_adder").run()?;
/// assert!(verilog_text.contains("full_adder"));
/// # Ok(())
/// # }
/// ```
pub fn full_adder() -> Result<hdl_cat::std_lib::FullAdderArrow, Error> {
    hdl_cat::std_lib::full_adder()
}

/// Emits a combinational circuit as Verilog and writes it to a file.
///
/// Combines [`emit_verilog`] with an `Io`-based file write, keeping
/// the entire pipeline lazy until [`.run()`](comp_cat_rs::effect::io::Io::run)
/// is called.
///
/// # Errors
///
/// Returns [`Error`] if Verilog generation or file writing fails.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use minroot_hdl::synthesis::{coeff_adder, emit_verilog_to_file};
///
/// let arrow = coeff_adder()?;
/// emit_verilog_to_file(&arrow, "coeff_add", "build/coeff_add.v").run()?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn emit_verilog_to_file<A, B>(
    arrow: &CircuitArrow<A, B>,
    name: &str,
    path: &str,
) -> Io<Error, ()>
where
    A: 'static,
    B: 'static,
{
    let path = path.to_owned();
    emit_verilog(arrow, name).flat_map(move |text| {
        Io::suspend(move || {
            std::fs::write(&path, &text)
                .map_err(Error::Io)
        })
    })
}

/// Emits a synchronous machine as Verilog and writes it to a file.
///
/// Combines [`emit_sync_verilog`] with an `Io`-based file write.
///
/// # Errors
///
/// Returns [`Error`] if Verilog generation or file writing fails.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> Result<(), hdl_cat::Error> {
/// use minroot_hdl::synthesis::emit_sync_verilog_to_file;
///
/// let counter = hdl_cat::std_lib::counter::<8>()?;
/// emit_sync_verilog_to_file(&counter, "counter8", "build/counter8.v").run()?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn emit_sync_verilog_to_file<S, I, O>(
    machine: &Sync<S, I, O>,
    name: &str,
    path: &str,
) -> Io<Error, ()> {
    let path = path.to_owned();
    emit_sync_verilog(machine, name).flat_map(move |text| {
        Io::suspend(move || {
            std::fs::write(&path, &text)
                .map_err(Error::Io)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeff_adder_builds() -> Result<(), Error> {
        let arrow = coeff_adder()?;
        assert!(!arrow.graph().wires().is_empty());
        Ok(())
    }

    #[test]
    fn pp_xor_builds() -> Result<(), Error> {
        let arrow = pp_xor()?;
        assert!(!arrow.graph().wires().is_empty());
        Ok(())
    }

    #[test]
    fn pp_and_builds() -> Result<(), Error> {
        let arrow = pp_and()?;
        assert!(!arrow.graph().wires().is_empty());
        Ok(())
    }

    #[test]
    fn pp_or_builds() -> Result<(), Error> {
        let arrow = pp_or()?;
        assert!(!arrow.graph().wires().is_empty());
        Ok(())
    }

    #[test]
    fn emit_coeff_adder_produces_verilog() -> Result<(), Error> {
        let arrow = coeff_adder()?;
        let text = emit_verilog(&arrow, "coeff_add").run()?;
        assert!(text.contains("coeff_add"));
        Ok(())
    }

    #[test]
    fn emit_pp_xor_produces_verilog() -> Result<(), Error> {
        let arrow = pp_xor()?;
        let text = emit_verilog(&arrow, "pp_xor").run()?;
        assert!(text.contains("pp_xor"));
        Ok(())
    }

    #[test]
    fn emit_sync_counter_produces_verilog() -> Result<(), Error> {
        let counter = hdl_cat::std_lib::counter::<8>()?;
        let text = emit_sync_verilog(&counter, "bit_counter").run()?;
        assert!(text.contains("bit_counter"));
        Ok(())
    }

    #[test]
    fn full_adder_builds() -> Result<(), Error> {
        let fa = full_adder()?;
        assert!(!fa.graph().wires().is_empty());
        Ok(())
    }

    #[test]
    fn emit_full_adder_produces_verilog() -> Result<(), Error> {
        let fa = full_adder()?;
        let text = emit_verilog(&fa, "full_adder").run()?;
        assert!(text.contains("full_adder"));
        Ok(())
    }

    #[test]
    fn emit_multiple_circuits_to_verilog() -> Result<(), Error> {
        let adder = coeff_adder()?;
        let xor = pp_xor()?;
        let fa = full_adder()?;

        let add_v = emit_verilog(&adder, "coeff_add").run()?;
        let xor_v = emit_verilog(&xor, "pp_xor_gate").run()?;
        let fa_v = emit_verilog(&fa, "full_adder_circuit").run()?;

        assert!(add_v.contains("coeff_add"));
        assert!(xor_v.contains("pp_xor_gate"));
        assert!(fa_v.contains("full_adder_circuit"));
        Ok(())
    }
}
