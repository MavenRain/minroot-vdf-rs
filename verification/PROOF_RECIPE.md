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

`add_limbs` is the additive twin of `sub_limbs`; `add_limbs_spec` is the FULL
ADDER: post `bv256 out = bv256 a + bv256 b ∧ carry = decide (2^256 ≤
(bv256 a).toNat + (bv256 b).toNat)`. Skeleton: `unfold; simp only [lift]; step*`
→ first `simp` (try_from/map_err/branch/`mask_val_le`) → `refine ⟨_, _, rfl, ?_, ?_⟩`.
- Output conjunct: second `simp` → `exact add_carry_concat _ _ _ _ _ _ _ _`. The bv
  lemma `add_carry_concat` is `sub_borrow_concat` with `+`/`>>> 64` (carry) replacing
  `-`/`if … >>> 127` (borrow); `bv_decide (timeout := 60)`.
- Carry conjunct (the hard one — Aeneas returns the carry as a U128 `bne`):
  `rw [← uaddOverflow_eq_decide, BitVec.uaddOverflow_eq, Bool.eq_iff_iff]` then
  `simp only [bv256, wrapping_add_bv_eq, from_bv_eq, i9_post2, i15_post2, i21_post2,
  i_post…i18_post, Std.U64.bv, Std.U128.bv, bne_iff_ne, ne_eq, Std.U128.eq_equiv_bv_eq,
  show (0#u128).bv = 0#128 from rfl, show ((64#i32).toNat) = 64 from rfl]` →
  `exact add_carry_bit _ _ _ _ _ _ _ _`. Pieces that make it work:
  - `uaddOverflow_eq_decide (x y : BitVec 256) : x.uaddOverflow y = decide (2^256 ≤
    x.toNat + y.toNat)` — bridge proven via `uaddOverflow_eq` + `msb_eq_decide` +
    `toNat_setWidth_of_le` + `toNat_add_of_lt` (the `2^256+2^256=2^257` step =
    `by rw [← two_mul, mul_comm, ← pow_succ]`).
  - `Bool.eq_iff_iff` turns the `Bool = Bool` goal into a Prop `↔` so `bne_iff_ne`
    fires; the Aeneas U128 `BEq` is `beq a b := a.bv = b.bv`, so `eq_equiv_bv_eq`
    + `(0#u128).bv = 0#128` reduce `(w#uscalar != 0#u128)` to a BitVec `≠`.
  - `BitVec.uaddOverflow_eq` is MANDATORY: this `bv_decide` does NOT bit-blast
    `BitVec.uaddOverflow` directly (abstracts it + its args) — rewrite it to the
    `.msb` form first.
  - CRITICAL: the post-simp goal is a clean pure-bv `↔`, but `bv_decide` STILL
    abstracts it because the atoms are array accesses `(↑a)[k]!.bv` used in mixed
    128-bit `setWidth`/256-bit `++` shapes. Same fix as `add_carry_concat`: state a
    FRESH-LIMB lemma `add_carry_bit (a0..b3 : BitVec 64)` (carry chain `.ushiftRight 64
    = 0#128 ↔ (setWidth 257 (a3++…) + setWidth 257 (b3++…)).msb = true`, `bv_decide`)
    and `exact` it. Don't run `bv_decide` on the array-atom goal directly.

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

## (5c) `mod_add` [DONE] — verified modular field addition; and the method-extraction wall

**AENEAS DROPS INHERENT IMPL METHODS.** `FieldElement::try_add`/`try_sub` (methods) are
NOT translated even though charon puts them in the `.llbc` (the method root yields empty
aeneas output; free-fn roots like `reduce_wide`/`add_limbs` extract fine). Also the charon
pattern for a method is `crate::field::FieldElement::try_add` (NOT `{impl …}` forms, and
NOT `crate::field::try_add` which is a free-fn path). FIX: factor the modular core into
FREE functions `mod_add`/`mod_sub` (`fn mod_add(a, b, modulus) -> Result<[u64;4], Error>`),
have the methods call them; verify the free fns. (Behavior-preserving; `cargo test` 28/28.)

`mod_add_spec` (preconds `val4 a < val4 modulus`, `val4 b < val4 modulus`,
`2 * val4 modulus ≤ 2^256`; post `val4 out = (val4 a + val4 b) % val4 modulus`) composes
the verified leaves: `unfold mod_add; step*` (through `add_limbs`, giving the full-adder
posts) → `have hab : … < 2^256 := by unfold val4 at ha hb hmod; omega` (note: `unfold … at`
inside a `have` is LOCAL, outer ha/hb stay `val4`-form for later `omega`s) → `have hval_r :
val4 r = val4 a + val4 b` (via `r_post2`+`BitVec.toNat_add`+`Nat.mod_eq_of_lt hab`) → drive
the control flow with `simp only [r_post1, r_post3, hnc, decide_false, …, branch, bind_tc_ok]`
(carry is false) → `step` (gte_modulus, auto-names `b1`/`b1_post`, do NOT `rename_i`) →
`simp only [b1_post, decide_eq_true_eq]; by_cases hge` → subtract branch: `step` (sub_limbs,
gives `r1`/`r1_post1`/`r1_post2`), `simp only [r1_post1, bind_tc_ok]`, `refine ⟨r1, rfl, ?_⟩`,
`BitVec.le_def.mpr hge` for `bv256 modulus ≤ bv256 r`, then `BitVec.toNat_sub_of_le` +
`Nat.mod_eq_sub_mod`/`Nat.mod_eq_of_lt` (side goals all `by omega` from the val4-level facts).
`omega` IS available here (the existing reduce_wide_spec uses it).

## (5d) `sub_limbs` borrow char + `mod_sub` [DONE] — verified modular field subtraction

`sub_limbs_spec` strengthened with the borrow conjunct `bo = decide (val4 a < val4 b)`.
EASIER than the add carry char: BitVec `<` (`ult`) is bv_decide-native, so NO `msb` /
`usubOverflow` detour. Borrow conjunct proof: `rw [Bool.eq_iff_iff]` then the SAME simp as the
output conjunct (`wrapping_sub_bv_eq`, `from_bv_eq`, `i4/i10/i16_post2`, `i_post…`, `Std.U64.bv`,
`Std.U128.bv`, `FromU128Bool.from`, `apply_ite Std.UScalar.bv`, `bne_iff_ne`, `ne_eq`,
`eq_equiv_bv_eq`) PLUS `decide_eq_true_eq`, `← BitVec.lt_def`, `show (0#u128).bv = 0#128`,
`show (1#u128).bv = 1#128`, `show (127#i32).toNat = 127` → `exact sub_borrow_bit _ … _`.
`sub_borrow_bit` = fresh-limb lemma (borrow chain `.ushiftRight 127 = 0#128 ↔ (a3++…) <
(b3++…)`, BitVec ult RHS — `.toNat < .toNat` was OUT of bv_decide's fragment, use BitVec `<`);
`bv_decide (timeout := 120)`.

CAUTION REALIZED: strengthening `sub_limbs_spec` shifted `reduce_wide_rec`'s `rename_i rsub xb2
xb1 hrp1 hrsub xst` → fixed by inserting the new borrow hyp before the do-marker:
`rename_i rsub xb2 xb1 hrp1 hrsub hborrow xst` (the post3 lands between post2=`hrsub` and the
do-marker=`xst`; `mod_add`'s use of `sub_limbs` survived unchanged since it only `simp`s `r1_post1`).

`mod_sub_spec` (post `val4 out = (val4 a + val4 modulus - val4 b) % val4 modulus`) mirrors
`mod_add`: `step*` (sub_limbs, now giving the borrow `r_post3`) → `simp [r_post1, r_post3,
branch, bind_tc_ok, decide_eq_true_eq]` → `by_cases hlt : val4 a < val4 b`. No-borrow: `if_neg`,
`val4 r = val4 a - val4 b` (`toNat_sub_of_le`), then `Nat.add_mod_left` + `Nat.mod_eq_of_lt`.
Borrow: `if_pos`, `step` add_limbs, `bv256 r1 = bv256 a + bv256 modulus - bv256 b` (`rw [r1_post2,
r_post2]; ring`), then `BitVec.toNat_add`/`toNat_sub_of_le` to get `val4 r1 = a+modulus-b`
(finish the `val4`↔`.toNat` defeq gaps with `simp only [val4] at *; omega`).

## (5e) `mul_wide` [DONE] — verified exact 512-bit schoolbook product

`mul_wide` rewritten iterator-free (fully-unrolled 4×4 row-wise schoolbook, 16 `wrapping_mul`
MACs, `wrapping_add`/`& mask`/`>>> 64`, `try_from` per limb) + re-extracted (`--start-from
crate::field::mul_wide` alongside the others) + `try_mul` call-site gets `?`. Behavior-preserving.
**GOTCHA: the generated `Funs.lean` hits `maxRecDepth` (default 512) on the deep nesting — aeneas
has NO `-max-rec-depth` flag, so POST-EXTRACTION add `set_option maxRecDepth 4000` to `Funs.lean`
(re-add after EVERY re-run of aeneas; it is dropped each time).**

`mul_wide_spec` post `val8 out = val4 a * val4 b`. Axioms = `[propext, Classical.choice,
Quot.sound]` (only 3 — cleaner than the bv_decide specs' 5, because the final bridge is `linarith`,
not bv_decide). Skeleton: `unfold; simp only [lift]; step*`; first simp (try_from/branch) needs
`mask_val_le` + `hr7 : r7.val ≤ U64.max := by scalar_tac` (top carry `r7` is the only UNMASKED
limb); `refine ⟨_, rfl, ?_⟩`.

KEY LESSONS (the OOM/runaway saga, now resolved):
- **bv_decide does NOT scale** to the carry-save assembly (only 2×2/256-bit is instant; 4×4/512-bit
  times out the SAT solver). Stay at the Nat level.
- **`step*` materializes the `>>> 64` carries but INLINES the `& mask` forward-fed intermediates**,
  so the assembled goal is astronomically large. FIX: `set` the nine forward masks (`R*`) and seven
  output masks (`M*`) innermost-first to opaque vars (+ `clear_value`), collapsing the goal to a
  small Horner form. `comba_recombine` (the pre-existing LINEAR omega lemma) supplies the column
  identity `h : ↑M0 + ↑M1·2⁶⁴ + … + ↑r7·2⁴⁴⁸ = a·b` (expanded form). 17 column `omega`s feed it.
- **THE FINAL-STEP FIX (this is what was outstanding):** after the masks are set, the goal is the
  Horner form `((…·2⁶⁴ + ↑M0%2⁶⁴)…) = product` with a `% 2⁶⁴` on every output limb. A single
  `omega` here DOES NOT TERMINATE (observed: 96 min single-core, then killed — the eight residues ×
  `2⁴⁴⁸`-scale coefficients explode omega's search; without `hr7'` it instead fast-failed "could
  not prove", which is what masked the non-termination). RESOLUTION: each masked limb is `< 2⁶⁴`
  (`have bk : Mk.val < 2^64 := by rw [hMkv]; exact Nat.mod_lt _ (by norm_num)`), and `hr7'` bounds
  the unmasked top, so `rw [Nat.mod_eq_of_lt b0, …, Nat.mod_eq_of_lt hr7']` deletes all eight mods;
  the residue-free goal is then `h` up to Horner↔expanded regrouping, closed by `linarith [h]`.
  **Use `linarith`, NOT `omega`, for this linear bridge: `linarith` fast-fails, `omega` thrashes.**
- PROBE TECHNIQUE that cracked it: the file's heavyweight proof makes the LSP `lean_goal` time out,
  but `lean_multi_attempt {line, snippets:["skip","exact h",…]}` dumps the goal once (the `exact h`
  type-mismatch printed h vs the expected Horner goal, revealing the exact `% 2⁶⁴` + Horner gap).

## (5f) `mod_mul` [DONE] — verified modular multiply (the headline, behind `try_mul`)

Aeneas drops inherent impl methods, so (as with `mod_add`/`mod_sub`) factor the multiply core into
a FREE fn `mod_mul(a, b, modulus) = reduce_wide(mul_wide(a,b)?, modulus)` and have `try_mul`
delegate; verify `mod_mul`. `mod_mul_spec` post `val4 out < val4 modulus ∧ val4 out =
(val4 a * val4 b) % val4 modulus` (preconds `0 < val4 modulus`, `2·val4 modulus ≤ 2^256`). Axioms =
standard 5 (the bv_decide pair flows in via `reduce_wide`), no sorryAx.

**COMPUTE-REDUCER PRACTICE: put `mod_mul_spec` in its OWN file `MinrootCore/Properties/ModMul.lean`
(importing `ReduceWide`), NOT in `ReduceWide.lean`.** Co-locating it with `mul_wide_spec` would make
every `mod_mul_spec` iteration re-elaborate the ~30-min `mul_wide` proof; the separate file reuses
the cached `ReduceWide.olean`, so iterating costs seconds. (Re-extracting `Funs.lean` still forces
ONE `ReduceWide.olean` rebuild, since lake keys on the dependency olean's hash.)

Proof: `unfold mod_mul; step*` runs `mul_wide_spec` (it is `@[step]`), binding the wide product `r`
with `r_post1 : x✝ = .Ok r` and `r_post2 : val8 r = val4 a * val4 b`, but LEAVES the `?` (branch)
acting on the raw inner result `x✝` (does NOT substitute). So `rw [r_post1]` then `simp only
[…CoreOpsTry…branch, bind_tc_ok]` reduces the branch+match to `reduce_wide r modulus`; then
`apply WP.spec_mono (reduce_wide_spec r modulus hp0 hp)` (mirrors how `reduce_wide_spec` consumes
`reduce_wide_rec_spec`); `rintro res ⟨out, hres, hlt, heq⟩`; close the residue conjunct with
`exact ⟨out, hres, hlt, by rw [heq, r_post2]⟩` (`heq : val4 out = val8 r % modulus`, `r_post2`
rewrites `val8 r` to the product).
