# minroot-vdf-rs

A Rust implementation of the [MinRoot](https://eprint.iacr.org/2022/1626) Verifiable Delay Function (VDF) hardware, targeting hdl-cat and built on the comp-cat-rs categorical effect system.

This is a port of [supranational/minroot_hardware](https://github.com/supranational/minroot_hardware) (SystemVerilog) to Rust, with categorical abstractions modeling the pipeline ring as a traced monoidal category.

## Crates

| Crate | Role |
|---|---|
| [`minroot-core`](crates/minroot-core) | Pure Rust reference model: Pallas/Vesta field arithmetic, Montgomery form, redundant polynomial representation, MinRoot VDF algorithm |
| [`minroot-cat`](crates/minroot-cat) | Categorical pipeline abstractions: `PipelineGraph` (free category), `Traced` monoidal category (ring feedback), `InterleaveK<N>` product functor, `exponent_schedule` catamorphism, `Target` trait for FPGA/ASIC |
| [`minroot-hdl`](crates/minroot-hdl) | hdl-cat hardware blocks: `PolySignal`, Wallace tree CSA compressor, `PolyMul`, `PolySqr`, `PolyReduce`, `FifthRootEngine`, clock gating |
| [`minroot-sim`](crates/minroot-sim) | `Io`/`Stream`-based simulation harness with reference-model verification |

## Architecture

```
comp-cat-rs (v0.5)
    |
    v
minroot-core        -- Pure field arithmetic, reference MinRoot
    |
    v
minroot-cat         -- Categorical pipeline (PipelineCat, Traced, Interleave)
    |
    v
minroot-hdl         -- hdl-cat hardware blocks interpreted from categorical specs
    |
    v
minroot-sim         -- Io/Stream simulation harness
```

### Categorical Model

The MinRoot fifth-root pipeline decomposes into:

- **Pipeline stages** (`SQR`, `MUL`, `RED`) as morphisms in `PipelineGraph`
- **Ring topology** via the `Traced` monoidal category trait (extending comp-cat-rs's `Symmetric`)
- **N-way interleaving** as the product functor `F(X) = [X; N]` with rotation as a natural transformation
- **Exponent bit scanning** as a catamorphism: `Stream::unfold` over 254 bits producing `RoundControl` signals
- **FPGA/ASIC targeting** as target-specific cell selection via the `Target` trait

The free category's universal property bridges the abstract pipeline spec and its concrete hdl-cat implementation: any `GraphMorphism` from `PipelineGraph` extends uniquely to a functor into the category of hdl-cat circuits.

## Simulation

Three levels of simulation, from fastest to most detailed:

### Reference Model

Run the pure-Rust `MinRoot` VDF against test vectors.  No hardware involved, used as the ground truth.

```rust,ignore
use minroot_core::field::{Curve, FieldElement};
use minroot_sim::harness::run_reference_only;
use minroot_sim::vectors::small_seeds;

let vectors = small_seeds(Curve::Pallas, 5, 2);
let summary = run_reference_only(vectors).run().unwrap_or_default();
assert!(summary.all_passed());
```

### Behavioral Engine

Drive the `FifthRootEngine` cycle-by-cycle through square-and-multiply steps, then verify against field arithmetic.  This exercises the full polynomial arithmetic pipeline (PolySqr, PolyMul, PolyReduce) without lowering to hdl-cat gates.

```rust,ignore
use minroot_core::field::{Curve, FieldElement};
use minroot_sim::harness::run_engine_cubed;

let result = run_engine_cubed(FieldElement::from_u64(42, Curve::Pallas))?;
assert!(result.matched());
```

Or directly via the `Synchronous` trait for full control:

```rust,ignore
use minroot_core::field::{Curve, FieldElement};
use minroot_core::polynomial::PolyElement;
use minroot_hdl::circuit::Synchronous;
use minroot_hdl::engine::{PallasEngine, EngineInput};
use minroot_hdl::types::{MulControl, PolySignal};

let x = FieldElement::from_u64(7, Curve::Pallas);
let signal = PolySignal::from_poly_element(&PolyElement::from_field(x));

// Compute x^3: exponent 3 = binary 11 (2 bits)
let inputs: Vec<EngineInput> = vec![
    EngineInput::load(signal, 2),
    EngineInput::round(MulControl::Multiply), // MSB = 1
    EngineInput::round(MulControl::Multiply), // LSB = 1
];

let (outputs, final_state) = PallasEngine::simulate(inputs);
assert!(outputs.last().map_or(false, |o| o.done));

let result = final_state.accum().to_poly_element(Curve::Pallas)?.to_field()?;
assert_eq!(result, x * x * x);
```

### hdl-cat Testbench

Cycle-accurate simulation of hdl-cat `CircuitArrow` and `Sync` machines with optional VCD waveform output for GTKWave.

```rust,ignore
use hdl_cat::bits::Bits;
use hdl_cat::kind::{BitSeq, Hw};
use minroot_hdl::synthesis::coeff_adder;
use minroot_sim::testbench::{simulate_combinational, trace_combinational_vcd};

let arrow = coeff_adder()?;
let a = Bits::<17>::new_wrapping(10u128);
let b = Bits::<17>::new_wrapping(20u128);
let input = a.to_bits_seq().concat(b.to_bits_seq());

// Simulate
let samples = simulate_combinational(arrow.clone(), vec![input.clone()]).run()?;
assert_eq!(samples.len(), 1);

// Generate VCD waveform
let vcd = trace_combinational_vcd(arrow, vec![input])?;
std::fs::write("trace.vcd", &vcd)?;  // Open in GTKWave
```

## Verilog Emission

hdl-cat generates synthesizable Verilog from Rust circuit descriptions.

### Combinational Circuits

```rust,ignore
use minroot_hdl::synthesis::{coeff_adder, full_adder, emit_verilog};

// Single gate
let adder = coeff_adder()?;
let adder_v = emit_verilog(&adder, "coeff_add").run()?;

// Composed circuit (single-bit full adder from XOR/AND/OR)
let fa = full_adder()?;
let fa_v = emit_verilog(&fa, "full_adder").run()?;
```

### Synchronous Machines

```rust,ignore
use minroot_hdl::synthesis::emit_sync_verilog;

let counter = hdl_cat::std_lib::counter::<8>()?;
let verilog = emit_sync_verilog(&counter, "counter8").run()?;
```

### Writing Verilog to Files

```rust,ignore
use minroot_hdl::synthesis::{coeff_adder, full_adder, emit_verilog_to_file, emit_sync_verilog_to_file};

// Combinational
let adder = coeff_adder()?;
emit_verilog_to_file(&adder, "coeff_add", "build/coeff_add.v").run()?;

// Composed
let fa = full_adder()?;
emit_verilog_to_file(&fa, "full_adder", "build/full_adder.v").run()?;

// Synchronous
let counter = hdl_cat::std_lib::counter::<8>()?;
emit_sync_verilog_to_file(&counter, "counter8", "build/counter8.v").run()?;
```

## Build and Test

```sh
cargo build --workspace
cargo test --workspace
cargo test --doc --workspace
RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets
```

## Documentation

Docs auto-publish to GitHub Pages via [`.github/workflows/docs.yml`](.github/workflows/docs.yml) on push to `main`.

Build and view locally:

```sh
cargo doc --no-deps --document-private-items --workspace --open
```

## Dual FPGA/ASIC Targeting

hdl-cat generates target-agnostic Verilog.  Technology-specific cells (clock gating, multiplier inference) are abstracted by the `minroot_cat::target::Target` trait:

```rust,ignore
use minroot_cat::target::{Fpga, Asic, Target};

let fpga = Fpga::new("Xilinx UltraScale+");
let asic = Asic::new("TSMC 12nm");
// Same pipeline, different cell selections per target.
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
