# minroot-vdf-rs

A Rust implementation of the [MinRoot](https://eprint.iacr.org/2022/1626) Verifiable Delay Function (VDF) hardware, targeting RHDL and built on the comp-cat-rs categorical effect system.

This is a port of [supranational/minroot_hardware](https://github.com/supranational/minroot_hardware) (SystemVerilog) to Rust, with categorical abstractions modeling the pipeline ring as a traced monoidal category.

## Crates

| Crate | Role |
|---|---|
| [`minroot-core`](crates/minroot-core) | Pure Rust reference model: Pallas/Vesta field arithmetic, Montgomery form, redundant polynomial representation, MinRoot VDF algorithm |
| [`minroot-cat`](crates/minroot-cat) | Categorical pipeline abstractions: `PipelineGraph` (free category), `Traced` monoidal category (ring feedback), `InterleaveK<N>` product functor, `exponent_schedule` catamorphism, `Target` trait for FPGA/ASIC |
| [`minroot-hdl`](crates/minroot-hdl) | RHDL hardware blocks: `PolySignal`, Wallace tree CSA compressor, `PolyMul`, `PolySqr`, `PolyReduce`, `FifthRootEngine`, clock gating |
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
minroot-hdl         -- RHDL hardware blocks interpreted from categorical specs
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

The free category's universal property bridges the abstract pipeline spec and its concrete RHDL implementation: any `GraphMorphism` from `PipelineGraph` extends uniquely to a functor into the category of RHDL circuits.

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

RHDL generates target-agnostic Verilog.  Technology-specific cells (clock gating, multiplier inference) are abstracted by the `minroot_cat::target::Target` trait:

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
