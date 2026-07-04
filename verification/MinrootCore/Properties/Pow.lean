/- Functional-correctness spec for `mod_pow`, the square-and-multiply modular
   exponentiation behind `FieldElement::pow` and `FieldElement::fifth_root`.

   `mod_pow base exp num_bits modulus` walks the exponent bits from the most
   significant (`num_bits - 1`) down to bit 0, squaring the accumulator each
   step (via the verified `mod_mul`) and multiplying `base` back in whenever
   the current exponent bit is set.  Its correctness is therefore the
   square-and-multiply loop invariant, composed on top of `mod_mul_spec`:

     val4 out = (val4 base) ^ (low `num_bits` bits of val4 exp)  (mod p)

   where "low `n` bits of E" is exactly `E % 2^n`.  The bit-doubling identity
   `a^(2E+b) = (a^E)^2 · a^b` turns each MSB-first step into a `Nat.ModEq`
   congruence; no field theory or Fermat is needed for this layer.

   This spec lives in its own file (importing `ModMul`, which pulls
   `mod_mul_spec` and, transitively, the heavyweight `ReduceWide`) so that
   iterating here reuses the cached `ReduceWide.olean` / `ModMul.olean`. -/
import MinrootCore.Properties.ModMul

open Aeneas Aeneas.Std Result

namespace minroot_core.field.spec

set_option maxHeartbeats 1000000
set_option maxRecDepth 4000

-- ── Pure bit / exponent helpers ──────────────────────────────────────

/-- `getLsbD` as a 0/1 natural: the `i`-th bit of `x.toNat`.  (Local copy;
    the identical lemma in `ReduceWide` is `private`.) -/
private theorem getLsbD_toNat {n : Nat} (x : BitVec n) (i : Nat) :
    (x.getLsbD i).toNat = x.toNat / 2 ^ i % 2 := by
  rw [BitVec.getLsbD, Nat.testBit_eq_decide_div_mod_eq]
  rcases Nat.mod_two_eq_zero_or_one (x.toNat / 2 ^ i) with h | h <;> simp [h]

/-- Bit `k` of the 256-bit value is bit `k % 64` of limb `k / 64`.  The 4-limb
    analogue of `ReduceWide`'s `getLsbD_bv512`. -/
private theorem getLsbD_bv256 (a : Array Std.U64 4#usize) (k : Nat) (hk : k < 256) :
    (bv256 a).getLsbD k = (a.val[k / 64]!.bv).getLsbD (k % 64) := by
  unfold bv256
  simp only [BitVec.getLsbD_append]
  split_ifs <;> (congr 1 <;> first | omega | (congr 2 <;> omega))

/-- MSB-first square-and-multiply Horner step when the processed bit is `1`:
    the accumulator was squared *and* multiplied by `base`. -/
private theorem sqmul_bit1 (acc base E n : Nat) (hE : E / 2 ^ n % 2 = 1) :
    (acc ^ 2 * base) ^ 2 ^ n * base ^ (E % 2 ^ n)
      = acc ^ 2 ^ (n + 1) * base ^ (E % 2 ^ (n + 1)) := by
  have hexp : E % 2 ^ (n + 1) = 2 ^ n + E % 2 ^ n := by
    rw [pow_succ, Nat.mod_mul, hE]; ring
  rw [hexp, pow_add, mul_pow, ← pow_mul, ← pow_succ' 2 n]
  ring

/-- MSB-first square-and-multiply Horner step when the processed bit is `0`:
    the accumulator was only squared. -/
private theorem sqmul_bit0 (acc base E n : Nat) (hE : E / 2 ^ n % 2 = 0) :
    (acc ^ 2) ^ 2 ^ n * base ^ (E % 2 ^ n)
      = acc ^ 2 ^ (n + 1) * base ^ (E % 2 ^ (n + 1)) := by
  have hexp : E % 2 ^ (n + 1) = E % 2 ^ n := by
    rw [pow_succ, Nat.mod_mul, hE]; ring
  have hacc : (acc ^ 2) ^ 2 ^ n = acc ^ 2 ^ (n + 1) := by
    rw [← pow_mul, ← pow_succ' 2 n]
  rw [hacc, hexp]

-- ── The loop invariant and top-level specs ──────────────────────────

/-- Loop invariant for `mod_pow_rec` (twin of `reduce_wide_rec_spec`).  From
    state `(remaining, acc)` the recursion folds exponent bits
    `remaining-1 .. 0` into `acc`, yielding

      val4 out = (val4 acc ^ 2^remaining · val4 base ^ (val4 exp % 2^remaining)) % p

    and keeps the result the canonical residue `< p`.  The bound
    `remaining ≤ 256` guarantees each `exp[·]` limb access is in range. -/
theorem mod_pow_rec_spec (base exp modulus acc : Array Std.U64 4#usize)
    (remaining : Std.Usize)
    (hp0 : 0 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256)
    (hrem : remaining.val ≤ 256) (hacc : val4 acc < val4 modulus) :
    field.mod_pow_rec base exp modulus remaining acc ⦃ r => ∃ out,
      r = core.result.Result.Ok out ∧ val4 out < val4 modulus ∧
      val4 out = (val4 acc ^ (2 ^ remaining.val)
        * val4 base ^ (val4 exp % 2 ^ remaining.val)) % val4 modulus ⦄ := by
  unfold field.mod_pow_rec
  by_cases h : remaining = 0#usize
  · -- BASE: remaining = 0.  out = acc; 2^0 = 1, exp % 2^0 = 0, base^0 = 1.
    simp only [h, reduceIte]
    simp only [WP.spec_ok]
    exact ⟨acc, rfl, hacc, by simp [Nat.mod_one, Nat.mod_eq_of_lt hacc]⟩
  · -- STEP: remaining = bit + 1, bit = remaining - 1 (the bit processed now).
    simp only [h, reduceIte]
    -- Drive: `bit ← remaining - 1`, the squaring `mod_mul acc acc modulus`
    -- (mod_mul_spec @[step], preconds discharged by assumption), the branch,
    -- the div/rem for limb_idx/bit_idx and the exp-limb bit test.
    step*
    rw [r_post1]
    simp only [core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch, bind_tc_ok]
    -- `r` is the squared accumulator here; capture its value before the next
    -- `step*` shadows the name with the recursion result.
    have hsqval : val4 r = val4 acc * val4 acc % val4 modulus := r_post3
    step*
    · -- MULTIPLY branch (exp bit = 1): acc' = mod_mul (acc²) base, then recurse.
      -- Capture the multiply's value before the next `step*` shadows `r1`.
      have hmulval : val4 r1 = val4 r * val4 base % val4 modulus := r1_post3
      rw [r1_post1]
      simp only [core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch, bind_tc_ok]
      step*
      refine ⟨r, r_post1, r_post2, ?_⟩
      have hrem_eq : remaining.val = bit.val + 1 := by omega
      have hb256 : bit.val < 256 := by omega
      have e1 : i2.val = i.val / 2 ^ bit_idx.val % 2 := by
        rw [i2_post1, Std.UScalar.val_and, i1_post1]
        simp only [show (1#u64).val = 1 from rfl, Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod]
      have key := getLsbD_bv256 exp bit.val hb256
      rw [← limb_idx_post, ← i_post, ← bit_idx_post] at key
      have key2 := congrArg Bool.toNat key
      rw [getLsbD_toNat, getLsbD_toNat, U64.bv_toNat] at key2
      have hbridge : i2.val = val4 exp / 2 ^ bit.val % 2 := by rw [e1, val4]; exact key2.symm
      have heq1 : i2.val = 1 := congrArg Std.UScalar.val ‹i2 = 1#u64›
      have hbit1 : val4 exp / 2 ^ bit.val % 2 = 1 := by rw [← hbridge]; exact heq1
      rw [r_post3, hrem_eq, hmulval, hsqval]
      have e2 : val4 acc * val4 acc % val4 modulus * val4 base % val4 modulus
          ≡ val4 acc ^ 2 * val4 base [MOD val4 modulus] :=
        calc val4 acc * val4 acc % val4 modulus * val4 base % val4 modulus
            ≡ val4 acc * val4 acc % val4 modulus * val4 base [MOD val4 modulus] := Nat.mod_modEq _ _
          _ ≡ val4 acc * val4 acc * val4 base [MOD val4 modulus] := (Nat.mod_modEq _ _).mul_right _
          _ = val4 acc ^ 2 * val4 base := by rw [pow_two]
      calc (val4 acc * val4 acc % val4 modulus * val4 base % val4 modulus) ^ 2 ^ bit.val
              * val4 base ^ (val4 exp % 2 ^ bit.val)
          ≡ (val4 acc ^ 2 * val4 base) ^ 2 ^ bit.val * val4 base ^ (val4 exp % 2 ^ bit.val)
            [MOD val4 modulus] := (e2.pow _).mul_right _
        _ = val4 acc ^ 2 ^ (bit.val + 1) * val4 base ^ (val4 exp % 2 ^ (bit.val + 1)) :=
            sqmul_bit1 _ _ _ _ hbit1
    · -- SQUARE-only branch (exp bit = 0): acc' = acc², then recurse.
      refine ⟨r, r_post1, r_post2, ?_⟩
      have hrem_eq : remaining.val = bit.val + 1 := by omega
      have hb256 : bit.val < 256 := by omega
      -- Bridge the extracted exp-limb bit test to `val4 exp / 2^bit % 2`.
      have e1 : i2.val = i.val / 2 ^ bit_idx.val % 2 := by
        rw [i2_post1, Std.UScalar.val_and, i1_post1]
        simp only [show (1#u64).val = 1 from rfl, Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod]
      have key := getLsbD_bv256 exp bit.val hb256
      rw [← limb_idx_post, ← i_post, ← bit_idx_post] at key
      have key2 := congrArg Bool.toNat key
      rw [getLsbD_toNat, getLsbD_toNat, U64.bv_toNat] at key2
      have hbridge : i2.val = val4 exp / 2 ^ bit.val % 2 := by rw [e1, val4]; exact key2.symm
      have hle : i2.val ≤ 1 := by
        rw [i2_post1, Std.UScalar.val_and, show (1#u64).val = 1 from rfl]; exact Nat.and_le_right
      have hne : i2.val ≠ 1 := by simpa using ‹¬i2 = 1#u64›
      have hbit0 : val4 exp / 2 ^ bit.val % 2 = 0 := by rw [← hbridge]; omega
      -- Push the accumulator's `% p` through the power tower, then apply the
      -- bit-0 Horner identity.
      rw [r_post3, hrem_eq, hsqval]
      have e2 : val4 acc * val4 acc % val4 modulus ≡ val4 acc ^ 2 [MOD val4 modulus] := by
        rw [pow_two]; exact Nat.mod_modEq _ _
      calc (val4 acc * val4 acc % val4 modulus) ^ 2 ^ bit.val
              * val4 base ^ (val4 exp % 2 ^ bit.val)
          ≡ (val4 acc ^ 2) ^ 2 ^ bit.val * val4 base ^ (val4 exp % 2 ^ bit.val)
            [MOD val4 modulus] := (e2.pow _).mul_right _
        _ = val4 acc ^ 2 ^ (bit.val + 1) * val4 base ^ (val4 exp % 2 ^ (bit.val + 1)) :=
            sqmul_bit0 _ _ _ _ hbit0
termination_by remaining.val
decreasing_by all_goals scalar_tac

/-- Top level: `mod_pow base exp num_bits modulus = base ^ (low num_bits bits of exp) % modulus`.
    Instantiates the loop invariant at `acc = one = [1,0,0,0]` (`val4 one = 1`,
    `1 ^ (2^n) = 1`) and `remaining = num_bits`.  Needs `1 < modulus`. -/
theorem mod_pow_spec (base exp modulus : Array Std.U64 4#usize) (num_bits : Std.Usize)
    (hp1 : 1 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256)
    (hnb : num_bits.val ≤ 256) :
    field.mod_pow base exp num_bits modulus ⦃ r => ∃ out,
      r = core.result.Result.Ok out ∧ val4 out < val4 modulus ∧
      val4 out = val4 base ^ (val4 exp % 2 ^ num_bits.val) % val4 modulus ⦄ := by
  unfold field.mod_pow
  have hone : val4 (Array.make 4#usize [1#u64, 0#u64, 0#u64, 0#u64] (by simp)) = 1 := by
    simp [val4, bv256, Array.make]
  have IH := mod_pow_rec_spec base exp modulus
    (Array.make 4#usize [1#u64, 0#u64, 0#u64, 0#u64] (by simp)) num_bits
    (by omega) hp hnb (by rw [hone]; exact hp1)
  apply WP.spec_mono IH
  rintro res ⟨out, hres, hlt, heq⟩
  exact ⟨out, hres, hlt, by rw [heq, hone, one_pow, one_mul]⟩

/-- `FieldElement::fifth_root a = pow a ((4p-3)/5) 254`, i.e.
    `mod_pow a exp 254 modulus`.  A direct instance of `mod_pow_spec`: the result
    is `a ^ (exp mod 2^254) % modulus`.  For the concrete Pasta
    `fifth_root_exponent` constant `exp < 2^254`, so the `% 2^254` is the identity
    and this is the genuine modular fifth root `a^((4p-3)/5) mod p`. -/
theorem fifth_root_spec (a exp modulus : Array Std.U64 4#usize)
    (hp1 : 1 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256) :
    field.mod_pow a exp (254#usize) modulus ⦃ r => ∃ out,
      r = core.result.Result.Ok out ∧ val4 out < val4 modulus ∧
      val4 out = val4 a ^ (val4 exp % 2 ^ 254) % val4 modulus ⦄ :=
  mod_pow_spec a exp modulus (254#usize) hp1 hp (by decide)

end minroot_core.field.spec
