/- Functional-correctness spec for `mod_mul`, the modular multiply behind
   `FieldElement::try_mul`.

   `mod_mul` composes `mul_wide` (the exact, unreduced 512-bit schoolbook
   product) with `reduce_wide` (the shift-and-subtract reduction modulo `p`),
   so its correctness is just the composition of `mul_wide_spec` and
   `reduce_wide_spec`: the result is `(a * b) mod p` in canonical form.

   This spec lives in its own file (importing the heavyweight `ReduceWide`
   rather than extending it) so that iterating on this proof reuses the
   cached `ReduceWide.olean` instead of re-elaborating `mul_wide_spec`. -/
import MinrootCore.Properties.ReduceWide

open Aeneas Aeneas.Std Result

namespace minroot_core.field.spec

set_option maxHeartbeats 1000000
set_option maxRecDepth 4000

/-- `mod_mul a b modulus` returns the canonical modular product
    `(a * b) % modulus` for `0 < modulus` with `2·modulus ≤ 2^256` (so the
    reduction yields the residue `< modulus`).  Composes the verified
    `mul_wide` (exact 512-bit product) and `reduce_wide` (modular reduction). -/
@[step] theorem mod_mul_spec (a b modulus : Array Std.U64 4#usize)
    (hp0 : 0 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256) :
    mod_mul a b modulus ⦃ r => ∃ out, r = core.result.Result.Ok out ∧
      val4 out < val4 modulus ∧
      val4 out = (val4 a * val4 b) % val4 modulus ⦄ := by
  unfold mod_mul
  step*
  -- `step*` runs `mul_wide_spec`, binding the wide product `r` with
  -- `r_post1 : x✝ = .Ok r` and `r_post2 : val8 r = val4 a * val4 b`, but leaves the
  -- `?` (branch) acting on the raw result `x✝`.  Rewrite `x✝ := .Ok r`, reduce the
  -- branch to `reduce_wide r modulus`, then compose with `reduce_wide_spec`.
  rw [r_post1]
  simp only [core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch,
    bind_tc_ok]
  apply WP.spec_mono (reduce_wide_spec r modulus hp0 hp)
  rintro res ⟨out, hres, hlt, heq⟩
  exact ⟨out, hres, hlt, by rw [heq, r_post2]⟩

end minroot_core.field.spec
