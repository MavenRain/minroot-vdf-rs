# Proof Recipe — Lean 4 / Aeneas spec for the 256-bit modular reduction (`reduce_wide`)

Toolchain `leanprover/lean4:v4.28.0-rc1`. NEW `step` framework; `progress` / `@[progress]` /
`@[pspec]` are deprecated aliases for `step` / `@[step]`. WP triple
`f args ⦃ ret => post ⦄` = `Aeneas.Std.WP.spec (f args) (fun ret => post)`
(`aeneas/backends/lean/Aeneas/Std/WP.lean:34`; notation `:202-227`).
Bridges: `WP.spec_equiv_exists` (`WP.lean:174`), `WP.spec_ok` (`WP.lean:67`).

## GATING TASK — model the 4 `core::result::Result` combinators (DONE)
`FunsExternal.lean` shipped `try_from` / `map_err` / `branch` / `from_residual` as bare
`axiom`s. As axioms you CANNOT `step` through `sub_limbs` / `reduce_wide_rec` (no equation).
Replaced with concrete computable `def`s + (TODO) `@[step]` specs. `TryFromIntError` given a
concrete 1-ctor type. On `reduce_wide_rec`'s path `try_from` cannot fail (each
`dN &&& mask < 2^64`), so `sub_limbs` always returns `.Ok` and the `Break`/`from_residual`
arms are dead code.

## (1) PARTIAL_FIXPOINT
`reduce_wide_rec` ends in `partial_fixpoint` because it recurses on `bit = remaining - 1`
where `bit` comes from a *monadic* (fail-able) `Usize` subtraction, so the syntactic
decreasing-arg heuristic can't see `bit < remaining`. This is the SAME mechanism as Aeneas's
ordinary `_loop` translation (`aeneas/documentation/aeneas-overview.md:154`, `zero_loop`).

- `partial_fixpoint` does NOT block unfolding. One-step unfold: **`unfold field.reduce_wide_rec`**
  (also `rw [field.reduce_wide_rec]` / `simp only [field.reduce_wide_rec]` /
  `rw [field.reduce_wide_rec.eq_def]`).
- DO NOT use `partial_fixpoint_induct` (banned: needs explicit motive + usually-`sorry`'d
  `admissible` side-goal).
- PATTERN: re-prove termination INSIDE the spec proof on the clean `Nat` measure you control.
  The `@[step]` spec self-references on the recursive call:

```lean
@[step] theorem myrec.spec (remaining : Std.Usize) (acc : Std.U64)
    (hbound : acc.val + remaining.val ≤ Std.U64.max) :
    myrec remaining acc ⦃ r => r.val = acc.val + remaining.val ⦄ := by
  unfold myrec
  by_cases h : remaining = 0#usize
  · simp [h]
  · simp only [h, reduceIte]
    step as ⟨bit, h_bit⟩      -- remaining - 1, gives y.val ≤ x.val so 1 ≤ remaining.val
    step as ⟨acc', h_acc'⟩
    step                       -- recursive call: auto-applies myrec.spec (self)
    · scalar_tac
    · simp only [*]; scalar_tac
termination_by remaining.val
decreasing_by scalar_decr_tac
```
`Usize.sub_spec` (`aeneas/.../Scalar/Ops/Sub.lean:154`): `x - y ⦃ z => z.val = x.val - y.val ∧ y.val ≤ x.val ⦄`
gives `1 ≤ remaining.val` on the else branch; `scalar_decr_tac` closes the decrease.

## (2) TACTICS (this checkout)
- `bv_tac N` = `bvify N` → `simp_all` → `bv_decide`. `bv_tac 64` (U64), `bv_tac 128` (the U128 `dN`/`mask`).
- `bvify N` / `natify`: Nat↔BitVec lifting.
- `scalar_tac` / `scalar_tac +nonLin`: scalar bounds/arith. `omega`/`linarith`/`nlinarith` BANNED.
- `zmodify` / `zmodify to (2^256)` / `zmodify [..] at h`: lift to `ZMod n` (ring).
- `scalar_decr_tac`: for `decreasing_by`.
- `split_conjs`, `simp_scalar`, `ring_eq_nf` helpers.

Mapping:
- (a) `(d &&& mask).val = d.val % 2^64` over U128 → `bv_tac 128` (or `UScalar.val_and`, `Bitwise.lean:279`).
- (b) per-limb `<<<`/`>>>`/`|||` U64 facts → `bv_tac 64`; assembly `val = Σ limb·2^(64i)` → `simp_scalar`/`ring_eq_nf`; `% 2^256` → `zmodify to (2^256)`.
- (c) `(2·acc + bit) % p`: modular → `zmodify [to p]; ring/simp`; bounds → `scalar_tac`.

## (3) Worked repo examples
- bv mask+sub+shift+modular battery: `aeneas/.../Tactic/Solver/BvTac/BvTac.lean:153-259`
  (collect each op's `.bv`/`.val` eq as a hyp via `step`, then one `bv_tac N` closes masking+shift+modular AND the `< p` bound `:244-259`).
- Leaf step specs in `aeneas/.../Std/Scalar/Bitwise.lean`: shift specs (`ShiftLeft_*_spec :216`,
  `ShiftRight_*_spec :185`), `UScalar.or_spec :231` (wrapped in `lift`, matches `lift (i3 ||| i4)`),
  `UScalar.and_spec :225`, bv/val identities `:270-281`. Also `Array.index_usize_spec`
  (`Std/Array/Array.lean:103`), `Usize.div_spec`/`rem_spec` (`Ops/Div.lean:418`, `Ops/Rem.lean:120`),
  `Array.repeat_val` (`Array.lean:99` ⇒ `val4 (repeat 4 0) = 0`).

## (4) Decomposition (new `Spec.lean`, pure functions not relations)
- `val4 a = Σ a[i].val·2^(64i)`, `val8 w` (8 limbs), `wideBit w k = (w[k/64] >>> (k%64)) &&& 1`.
- `shift_left_one.spec`: `val4 r = (2 * val4 a) % 2^256`.
- `gte_modulus.spec`: `b = decide (val4 modulus ≤ val4 a)`.
- `sub_limbs.spec` (precond `val4 modulus ≤ val4 a`):
  `∃ out borrow, r = .Ok (out, borrow) ∧ val4 out = val4 a - val4 modulus ∧ borrow = false`.
- `reduce_wide_rec.spec` (invariant): preconds `0 < val4 modulus`, `val4 modulus ≤ 2^255`,
  `remaining.val ≤ 512`, `val4 acc < val4 modulus`;
  post `∃ out, r = .Ok out ∧ val4 out < val4 modulus ∧
  (val4 out : ZMod p) = (val4 acc)·2^remaining.val + Σ_{k<remaining.val} wideBit w k · 2^k`.
- `reduce_wide.spec` (top-level): preconds `0 < val4 modulus`, `val4 modulus ≤ 2^255`;
  post `∃ out, r = .Ok out ∧ val4 out = val8 wide % val4 modulus`
  (instantiate rec.spec at `remaining = 512`, `acc = 0`; `Σ_{k<512}` collapses to `val8 wide`).

Required hypotheses: `0 < val4 modulus` (well-defined `% p`, nontrivial `ZMod p`) and
`val4 modulus ≤ 2^255` (so `2·acc+1 < 2^256`, one conditional subtract restores `< p`).
Pallas/Vesta `p ≈ 2^254`, well under `2^255`.

ORDER: (0) model 4 combinators [DONE] → (2) `Spec.lean` defs → (3) `shift_left_one` →
(4) `gte_modulus` → (5) `sub_limbs` → (6) `reduce_wide_rec` → (7) `reduce_wide`.
Keep `maxHeartbeats 1000000`. Never `omega`/`linarith`/`nlinarith`/`partial_fixpoint_induct`/`all_goals`.

## (5b) `add_limbs` [DONE] — and the extraction-widening workflow

`add_limbs` is the additive twin of `sub_limbs`; `add_limbs_spec` (post
`bv256 out = bv256 a + bv256 b`, carry left existential like `sub`'s borrow)
proves by the SAME skeleton: `unfold; simp only [lift]; step*` → first `simp`
(try_from/map_err/branch/`mask_val_le`) → `refine ⟨_, _, rfl, ?_⟩` → second
`simp` → `exact add_carry_concat _ _ _ _ _ _ _ _`. The bv lemma
`add_carry_concat` is `sub_borrow_concat` with `+`/`>>> 64` (carry) replacing
`-`/`if … >>> 127` (borrow); `bv_decide (timeout := 60)`.

TWO GOTCHAS that gate this:
- **Iterators don't extract to provable code.** The original `add_limbs`/`mul_wide`
  used `.iter().zip().enumerate().fold(...)` / `.for_each(...)`. Aeneas has no
  Lean model for `Iterator::{fold,for_each,zip}` (it warns + emits OPAQUE calls →
  no equation → unprovable). FIX: rewrite the Rust iterator-free in the unrolled
  /explicit-recursion style of `sub_limbs`/`reduce_wide_rec` (CLAUDE.md permits
  recursion as the alt to combinators). Behavior-preserving; `cargo test` still 28/28.
- **Use `wrapping_add`, not `+`.** Plain `u128 +` extracts as a FALLIBLE add, so
  `step*` leaves three `case hmax : ↑i_ + ↑i_ ≤ U128.max` overflow side-goals AND
  gives only `.val` posts (no `.bv`). `.wrapping_add(...)` (like `sub`'s
  `wrapping_sub`) is TOTAL: zero side-goals, and `core.num.U128.wrapping_add_bv_eq`
  (`@[simp,bvify]`) gives the `.bv`. Columns are `< 2^65` so the wrap never fires —
  `>> 64` still recovers the exact carry. (If forced to keep `+`: discharge each
  `hmax` with `scalar_tac`, then bridge each `s_i.bv` from its `.val` post.)

RE-EXTRACTION (widening past the `reduce_wide` subtree): the prior `.llbc` was
scoped via `charon cargo --preset=aeneas --start-from crate::field::reduce_wide`.
Add targets, e.g. `… --start-from crate::field::add_limbs`, `--dest-file …/minroot_core.llbc`
(charon at `aeneas/charon/bin`; `--preset=aeneas` is MANDATORY or aeneas rejects the
llbc). Then `aeneas -backend lean -split-files -subdir MinrootCore/Code -dest verification
minroot_core.llbc`. It REGENERATES `Funs/Types/*_Template` but NOT the hand-edited
`FunsExternal/TypesExternal`; the `reduce_wide` subtree comes out byte-identical
(deterministic), so existing proofs survive. **Restart the LSP (`lean_build`) after
re-extraction** — a running server holds the stale `Funs.olean` and lowercase
identifiers like `add_limbs` will autobind as `Sort u` implicits.

NEXT: `add_limbs` carry characterization (for `try_add`), then `mul_wide` (rewrite
4×4 schoolbook iterator-free first; `bv_decide` won't scale to a 512-bit multiply →
needs Nat schoolbook decomposition), then `try_mul`/`try_add`/`try_sub` compositions.
