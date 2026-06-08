# CLAUDE.md — minroot-vdf-rs Project Conventions

## Philosophy

This project follows **functional**, **type-driven**, and **domain-driven** design principles.  The type system is the primary tool for correctness.  If a state is illegal, it should be unrepresentable.

Every Rust repo uses **comp-cat-rs** as its core building block/framework.

## Architecture

- **Modules by domain context**, not by technical layer.
- Each bounded context gets its own module with its own types.
- Prefer thin `main.rs`/`lib.rs` that wires contexts together.
- **comp-cat-rs** provides the categorical foundation (Category, Functor, Monad, Io, Stream, Resource, Fiber).
- **hdl-cat** provides the hardware description layer (Bits, Hw, CircuitArrow, Sync, #[kernel]).
- The categorical layer (minroot-cat) bridges comp-cat-rs abstractions to hdl-cat hardware blocks.

### Crate Hierarchy

```
comp-cat-rs (v0.5)
    |
    v
minroot-core          -- Pure field arithmetic, reference MinRoot, no HDL dependency
    |
    v
minroot-cat           -- Categorical pipeline abstractions (PipelineCat, Traced, Interleave)
    |
    v
minroot-hdl           -- hdl-cat hardware blocks interpreted from categorical specs
    |
    v
minroot-sim           -- Io/Stream simulation harness, Verilog testbench generation
```

## Types

- **Newtypes** for all domain primitives.  Never pass raw `String`, `u64`, `f64`, etc. across function boundaries when they carry domain meaning.
- **Sum types (enums)** to model domain variants and state machines.
- **Phantom types** where compile-time state tracking is warranted.
- **No public struct fields**.  All fields must be private.  Expose access through getter methods and construct through associated functions or builder patterns.
- `#[must_use]` on pure functions and types whose values should not be silently dropped.

## Error Handling

- A single project-wide `Error` enum defined in a dedicated `error` module.
- Each variant wraps an underlying error type from a dependency or domain context.
- Implement `From<UnderlyingError> for Error` for every variant to enable `?` ergonomics.
- Implement `std::fmt::Display` and `std::error::Error` by hand.  No `thiserror`, no `anyhow`.
- Domain logic returns `Result<T, Error>`.  Never panic in library code.

## Style

- **Prefer `match` over `if`/`else`** except when branching on `bool`.
- **No `return` keyword**.  Every function body is a single expression.
- **No `mut`**.  All bindings and parameters must be immutable.
- **Combinators** (`map`, `and_then`, `filter`, `fold`, etc.) over imperative loops.
- **Prefer `and_then` over nested pattern-matching**.
- **Pure functions** at the core; side effects at the boundaries via `Io`.
- **No `unwrap()`/`expect()`**.  Prohibited everywhere, including tests, docs, and `main`.
- **No `loop` or `for`**.  Use iterator combinators or recursion.
- Prefer **iterators** over indexed access.
- Stay inside `Io`/`Stream`/etc. via combinators; call `run` only at the boundary.

## hdl-cat-Specific Conventions

- All hardware types implement `Hw` (the hardware-representable trait).
- Pipeline stages use `CircuitArrow` for combinational blocks and `Sync` for stateful machines.
- Clock domains use hdl-cat's typed domain markers (`Red`, `Blue`, etc.).
- FSM states are Rust enums.
- `#[kernel]` functions use the synthesizable Rust subset.

## Testing

- **Property-based tests** via `proptest` where types suggest invariants.
- Unit tests live in the same file as the code they test.
- Integration tests go in `tests/` organized by domain context.
- Reference model comparison: every hdl-cat circuit is tested against its minroot-core software equivalent.

## Dependencies

- Minimize dependencies.  comp-cat-rs and hdl-cat are the two pillars; evaluate everything else.
- No `thiserror`, no `anyhow`.

## Documentation

- Doc comments on all public items.
- Include `# Examples` sections with runnable code blocks.
- Docs must never use `unwrap`, `expect`, or `unreachable`.
