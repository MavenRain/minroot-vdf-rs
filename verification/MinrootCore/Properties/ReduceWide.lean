/- Functional-correctness spec for the `reduce_wide` shift-and-subtract
   modular reduction chain extracted from `minroot-core::field`.

   The reduction is a textbook binary modular reduction: starting from
   `acc = 0`, for each of the 512 bits of the wide product from the most
   significant down, set `acc := (2*acc + bit) mod p` (realized as a left
   shift, bit injection, and conditional subtraction).  The invariant
   `acc < p` is maintained throughout, so the final `acc` equals
   `val(wide) mod p`.  No deep number theory is required — only the
   bignum↔Nat bridge and a loop-invariant induction. -/
import MinrootCore.Code.Funs

open Aeneas Aeneas.Std Result

namespace minroot_core.field.spec

set_option maxHeartbeats 1000000
set_option maxRecDepth 4000

/-- The 256-bit bit-vector of a 4-limb element, high limb first.  Its `toNat`
    is exactly the little-endian value `a[0] + a[1]·2^64 + a[2]·2^128 + a[3]·2^192`.
    Working through this single-width view lets per-limb bit operations be
    discharged by `bv_decide`. -/
def bv256 (a : Array Std.U64 4#usize) : BitVec 256 :=
  (a.val[3]!).bv ++ (a.val[2]!).bv ++ (a.val[1]!).bv ++ (a.val[0]!).bv

/-- The 512-bit bit-vector of an 8-limb wide product, high limb first. -/
def bv512 (w : Array Std.U64 8#usize) : BitVec 512 :=
  (w.val[7]!).bv ++ (w.val[6]!).bv ++ (w.val[5]!).bv ++ (w.val[4]!).bv
    ++ (w.val[3]!).bv ++ (w.val[2]!).bv ++ (w.val[1]!).bv ++ (w.val[0]!).bv

/-- Little-endian value of a 4-limb (256-bit) field element. -/
def val4 (a : Array Std.U64 4#usize) : Nat := (bv256 a).toNat

/-- Little-endian value of an 8-limb (512-bit) wide product. -/
def val8 (w : Array Std.U64 8#usize) : Nat := (bv512 w).toNat

/-- `toNat` of a bit-vector concatenation in additive form: the high part is
    weighted by `2^m` and disjointly added to the (bounded) low part.  Lets a
    nested `++` collapse to a positional little-endian sum that `ring`/`omega`
    can manipulate. -/
private theorem toNat_append_add {n m : Nat} (x : BitVec n) (y : BitVec m) :
    (x ++ y).toNat = x.toNat * 2 ^ m + y.toNat := by
  rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt y.isLt, Nat.shiftLeft_eq]

/-- The 4×4 schoolbook (comba) reassembly as a *purely linear* fact over the concrete base
    `2^64`: given the carry/limb decomposition of each column (`column = limb + 2^64·carry`),
    the little-endian recombination of the eight output limbs equals the positional sum of the
    sixteen partial products.  The column relations are supplied as linear hypotheses, so `omega`
    closes it with no `div`/`mod` reasoning at all (and therefore quickly). -/
private theorem comba_recombine
    (P00 P01 P02 P03 P10 P11 P12 P13 P20 P21 P22 P23 P30 P31 P32 P33 : Nat)
    (c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 : Nat)
    (mA1 mA2 mA3 mB1 mB2 mB3 mC2 mC3 mC4 : Nat)
    (L0 L1 L2 L3 L4 L5 L6 L7 : Nat)
    (dA0 : P00 = L0 + 2 ^ 64 * c0)
    (dA1 : P01 + c0 = mA1 + 2 ^ 64 * c1)
    (dA2 : P02 + c1 = mA2 + 2 ^ 64 * c2)
    (dA3 : P03 + c2 = mA3 + 2 ^ 64 * c3)
    (dB0 : P10 + mA1 = L1 + 2 ^ 64 * c4)
    (dB1 : P11 + mA2 + c4 = mB1 + 2 ^ 64 * c5)
    (dB2 : P12 + mA3 + c5 = mB2 + 2 ^ 64 * c6)
    (dB3 : P13 + c3 + c6 = mB3 + 2 ^ 64 * c7)
    (dC0 : P20 + mB1 = L2 + 2 ^ 64 * c8)
    (dC1 : P21 + mB2 + c8 = mC2 + 2 ^ 64 * c9)
    (dC2 : P22 + mB3 + c9 = mC3 + 2 ^ 64 * c10)
    (dC3 : P23 + c7 + c10 = mC4 + 2 ^ 64 * c11)
    (dD0 : P30 + mC2 = L3 + 2 ^ 64 * c12)
    (dD1 : P31 + mC3 + c12 = L4 + 2 ^ 64 * c13)
    (dD2 : P32 + mC4 + c13 = L5 + 2 ^ 64 * c14)
    (dD3 : P33 + c11 + c14 = L6 + 2 ^ 64 * c15)
    (dTop : L7 = c15) :
    L0 + L1 * 2 ^ 64 + L2 * 2 ^ 128 + L3 * 2 ^ 192 + L4 * 2 ^ 256 + L5 * 2 ^ 320
      + L6 * 2 ^ 384 + L7 * 2 ^ 448
    = P00 + (P01 + P10) * 2 ^ 64 + (P02 + P11 + P20) * 2 ^ 128
      + (P03 + P12 + P21 + P30) * 2 ^ 192 + (P13 + P22 + P31) * 2 ^ 256
      + (P23 + P32) * 2 ^ 320 + P33 * 2 ^ 384 := by
  omega

/-- The `k`-th bit (`0 ≤ k < 512`) of the wide product, as a `Nat` (0 or 1). -/
def wideBit (w : Array Std.U64 8#usize) (k : Nat) : Nat :=
  ((w.val[k / 64]!).val >>> (k % 64)) &&& 1

/-- The per-limb left-shift recombination realizes a single 256-bit left shift:
    each limb takes `<< 1` plus the carried-out top bit of the limb below.
    A self-contained bit-vector identity, discharged by `bv_decide`. -/
private theorem shl_concat (b0 b1 b2 b3 : BitVec 64) :
    (b3 <<< 1 ||| b2 >>> 63) ++ (b2 <<< 1 ||| b1 >>> 63)
      ++ (b1 <<< 1 ||| b0 >>> 63) ++ b0 <<< 1
    = (b3 ++ b2 ++ b1 ++ b0) <<< 1 := by
  bv_decide

/-- `shift_left_one` is exactly a 256-bit left shift by one (the bit-vector form,
    which makes the value form `val4 r = 2 * val4 a % 2^256` immediate). -/
@[step] theorem shift_left_one_spec (a : Array Std.U64 4#usize) :
    shift_left_one a ⦃ r => bv256 r = bv256 a <<< 1 ⦄ := by
  unfold shift_left_one
  step*
  simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
    i_post, i2_post, i6_post, i10_post, i1_post2, i3_post2, i4_post2, i5_post2,
    i7_post2, i8_post2, i9_post2, i11_post2, i12_post2, i13_post2]
  exact shl_concat _ _ _ _

/-- Lexicographic comparison lemmas (high limb first).  Each captures one branch
    of `gte_modulus`'s limb-by-limb comparison as a self-contained 256-bit
    bit-vector identity, discharged by `bv_decide`.  Stating them generically (over
    fresh `BitVec 64`) sidesteps `bv_decide`'s sensitivity to the concrete array
    access terms; the per-branch goals are closed by instantiation. -/
private theorem lex0 (a0 a1 a2 a3 m0 m1 m2 m3 : BitVec 64) (h : ¬ a3 = m3) :
    m3.ult a3 = (m3 ++ m2 ++ m1 ++ m0).ule (a3 ++ a2 ++ a1 ++ a0) := by bv_decide

private theorem lex1 (a0 a1 a2 a3 m0 m1 m2 m3 : BitVec 64)
    (h3 : a3 = m3) (h : ¬ a2 = m2) :
    m2.ult a2 = (m3 ++ m2 ++ m1 ++ m0).ule (a3 ++ a2 ++ a1 ++ a0) := by bv_decide

private theorem lex2 (a0 a1 a2 a3 m0 m1 m2 m3 : BitVec 64)
    (h3 : a3 = m3) (h2 : a2 = m2) (h : ¬ a1 = m1) :
    m1.ult a1 = (m3 ++ m2 ++ m1 ++ m0).ule (a3 ++ a2 ++ a1 ++ a0) := by bv_decide

private theorem lex3 (a0 a1 a2 a3 m0 m1 m2 m3 : BitVec 64)
    (h3 : a3 = m3) (h2 : a2 = m2) (h1 : a1 = m1) :
    m0.ule a0 = (m3 ++ m2 ++ m1 ++ m0).ule (a3 ++ a2 ++ a1 ++ a0) := by bv_decide

/-- `gte_modulus a m` decides `m ≤ a` on the 256-bit values. -/
@[step] theorem gte_modulus_spec (a modulus : Array Std.U64 4#usize) :
    gte_modulus a modulus ⦃ b => b = decide (val4 modulus ≤ val4 a) ⦄ := by
  unfold gte_modulus
  step* <;>
    simp only [val4, bv256, gt_iff_lt, ge_iff_le, bne_iff_ne, ne_eq, not_not,
      UScalar.lt_equiv, UScalar.le_equiv, UScalar.eq_equiv, ← U64.bv_toNat,
      BitVec.toNat_inj, ← BitVec.ult_eq_decide, ← BitVec.ule_eq_decide, *] at *
  · exact lex0 _ _ _ _ _ _ _ _ (by assumption)
  · exact lex1 _ _ _ _ _ _ _ _ (by assumption) (by assumption)
  · exact lex2 _ _ _ _ _ _ _ _ (by assumption) (by assumption) (by assumption)
  · exact lex3 _ _ _ _ _ _ _ _ (by assumption) (by assumption) (by assumption)

/-- Masking a `u128` with `u64::MAX` always fits in a `u64`, so the checked
    `try_from` in `sub_limbs` never fails. -/
private theorem mask_val_le (d : Std.U128) :
    (d &&& core.convert.num.FromU128U64.from core.num.U64.MAX).val ≤ Std.U64.max := by
  simp only [Std.UScalar.val_and, core.convert.num.FromU128U64.from_val_eq]
  exact le_trans Nat.and_le_right (by rw [Std.U64.max_eq]; exact Std.U64.le_max _)

/-- The ripple-borrow `u128` chain reassembles into a single 256-bit subtraction.
    Self-contained bit-vector identity over fresh limbs (`bv_decide`); used by
    instantiation to dodge `bv_decide`'s sensitivity to the array-access terms. -/
private theorem sub_borrow_concat (a0 a1 a2 a3 b0 b1 b2 b3 : BitVec 64) :
    BitVec.setWidth 64 ((BitVec.setWidth 128 a3 - BitVec.setWidth 128 b3 -
          if ¬(BitVec.setWidth 128 a2 - BitVec.setWidth 128 b2 -
                if ¬(BitVec.setWidth 128 a1 - BitVec.setWidth 128 b1 -
                      if ¬(BitVec.setWidth 128 a0 - BitVec.setWidth 128 b0) >>> 127 = 0#128
                      then 1#128 else 0#128) >>> 127 = 0#128
                  then 1#128 else 0#128) >>> 127 = 0#128
            then 1#128 else 0#128) &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 ((BitVec.setWidth 128 a2 - BitVec.setWidth 128 b2 -
          if ¬(BitVec.setWidth 128 a1 - BitVec.setWidth 128 b1 -
                if ¬(BitVec.setWidth 128 a0 - BitVec.setWidth 128 b0) >>> 127 = 0#128
                then 1#128 else 0#128) >>> 127 = 0#128
            then 1#128 else 0#128) &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 ((BitVec.setWidth 128 a1 - BitVec.setWidth 128 b1 -
          if ¬(BitVec.setWidth 128 a0 - BitVec.setWidth 128 b0) >>> 127 = 0#128
          then 1#128 else 0#128) &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 (BitVec.setWidth 128 a0 - BitVec.setWidth 128 b0
          &&& BitVec.setWidth 128 18446744073709551615#64)
    = (a3 ++ a2 ++ a1 ++ a0) - (b3 ++ b2 ++ b1 ++ b0) := by
  bv_decide (config := { timeout := 60 })

/-- The top borrow-out of the ripple chain (`d₃ >> 127`) is nonzero exactly when
    the 256-bit subtraction underflows (`a < b`).  Fresh-limb bit-vector identity
    (`bv_decide`); the borrow analogue of `add_carry_bit`. -/
private theorem sub_borrow_bit (a0 a1 a2 a3 b0 b1 b2 b3 : BitVec 64) :
    ¬((BitVec.setWidth 128 a3 - BitVec.setWidth 128 b3 -
            if ¬(BitVec.setWidth 128 a2 - BitVec.setWidth 128 b2 -
                    if ¬(BitVec.setWidth 128 a1 - BitVec.setWidth 128 b1 -
                            if ¬(BitVec.setWidth 128 a0 - BitVec.setWidth 128 b0) >>> 127 = 0#128
                            then 1#128 else 0#128) >>> 127 = 0#128
                      then 1#128 else 0#128) >>> 127 = 0#128
              then 1#128 else 0#128).ushiftRight 127 = 0#128)
      ↔ (a3 ++ a2 ++ a1 ++ a0) < (b3 ++ b2 ++ b1 ++ b0) := by
  bv_decide (config := { timeout := 120 })

/-- `sub_limbs a b` returns the 256-bit wrapping difference (and a borrow flag),
    never failing.  The four limb subtractions form a ripple-borrow chain via
    `u128`; the masked low-64 casts reassemble into the 256-bit difference, and
    the top borrow-out is exactly `a < b`. -/
@[step] theorem sub_limbs_spec (a b : Array Std.U64 4#usize) :
    sub_limbs a b ⦃ r => ∃ out bo,
      r = core.result.Result.Ok (out, bo) ∧ bv256 out = bv256 a - bv256 b ∧
      bo = decide (val4 a < val4 b) ⦄ := by
  unfold sub_limbs
  simp only [lift]
  step*
  -- The do-block has collapsed to the `try_from`/`map_err`/`branch` chain.
  -- Evaluate it: the mask guarantees each `try_from` succeeds (`mask_val_le`).
  simp only [U64.Insts.CoreConvertTryFromU128TryFromIntError.try_from,
    core.result.Result.map_err,
    core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch,
    mask_val_le, if_true, bind_tc_ok]
  refine ⟨_, _, rfl, ?_, ?_⟩
  · simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
      Std.U64.bv, Std.U128.bv, Std.UScalar.cast_bv_eq,
      core.convert.num.FromU128U64.from_bv_eq, core.num.U128.wrapping_sub_bv_eq,
      Std.UScalar.bv_and, core.convert.num.FromU128Bool.from, apply_ite Std.UScalar.bv,
      bne_iff_ne, ne_eq, Std.U128.eq_equiv_bv_eq, core.num.U64.MAX, Std.U64.ofNat_bv,
      Std.U64.rMax, Std.UScalarTy.U64_numBits_eq,
      show (0#u128).bv = 0#128 from rfl, show (1#u128).bv = 1#128 from rfl,
      i_post, i2_post, i5_post, i7_post,
      i11_post, i13_post, i17_post, i19_post, i4_post2, i10_post2, i16_post2]
    exact sub_borrow_concat _ _ _ _ _ _ _ _
  · rw [Bool.eq_iff_iff]
    simp only [val4, bv256, core.num.U128.wrapping_sub_bv_eq,
      core.convert.num.FromU128U64.from_bv_eq, i4_post2, i10_post2, i16_post2,
      i_post, i2_post, i5_post, i7_post, i11_post, i13_post, i17_post, i19_post,
      Std.U64.bv, Std.U128.bv, core.convert.num.FromU128Bool.from, apply_ite Std.UScalar.bv,
      bne_iff_ne, ne_eq, Std.U128.eq_equiv_bv_eq, decide_eq_true_eq, ← BitVec.lt_def,
      show (0#u128).bv = 0#128 from rfl, show (1#u128).bv = 1#128 from rfl,
      show ((127#i32).toNat) = 127 from rfl]
    exact sub_borrow_bit _ _ _ _ _ _ _ _

/-- The ripple-carry `u128` chain reassembles into a single 256-bit addition.
    Self-contained bit-vector identity over fresh limbs (`bv_decide`); the
    additive companion to `sub_borrow_concat`. -/
private theorem add_carry_concat (a0 a1 a2 a3 b0 b1 b2 b3 : BitVec 64) :
    BitVec.setWidth 64 (BitVec.setWidth 128 a3 + BitVec.setWidth 128 b3 +
            (BitVec.setWidth 128 a2 + BitVec.setWidth 128 b2 +
                (BitVec.setWidth 128 a1 + BitVec.setWidth 128 b1 +
                    (BitVec.setWidth 128 a0 + BitVec.setWidth 128 b0) >>> 64) >>> 64) >>> 64
          &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 (BitVec.setWidth 128 a2 + BitVec.setWidth 128 b2 +
            (BitVec.setWidth 128 a1 + BitVec.setWidth 128 b1 +
                (BitVec.setWidth 128 a0 + BitVec.setWidth 128 b0) >>> 64) >>> 64
          &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 (BitVec.setWidth 128 a1 + BitVec.setWidth 128 b1 +
            (BitVec.setWidth 128 a0 + BitVec.setWidth 128 b0) >>> 64
          &&& BitVec.setWidth 128 18446744073709551615#64)
      ++ BitVec.setWidth 64 (BitVec.setWidth 128 a0 + BitVec.setWidth 128 b0
          &&& BitVec.setWidth 128 18446744073709551615#64)
    = (a3 ++ a2 ++ a1 ++ a0) + (b3 ++ b2 ++ b1 ++ b0) := by
  bv_decide (config := { timeout := 60 })

/-- The top carry-out of the ripple chain (`s₃ >> 64`) is nonzero exactly when
    the full 256-bit addition overflows (`msb` of the 257-bit sum).  Fresh-limb
    bit-vector identity (`bv_decide`); stated over fresh `BitVec 64` to dodge
    `bv_decide`'s sensitivity to the array-access atoms `(↑a)[k]!.bv`. -/
private theorem add_carry_bit (a0 a1 a2 a3 b0 b1 b2 b3 : BitVec 64) :
    ¬((BitVec.setWidth 128 a3 + BitVec.setWidth 128 b3 +
            (BitVec.setWidth 128 a2 + BitVec.setWidth 128 b2 +
                (BitVec.setWidth 128 a1 + BitVec.setWidth 128 b1 +
                    (BitVec.setWidth 128 a0 + BitVec.setWidth 128 b0) >>> 64) >>> 64) >>> 64).ushiftRight
          64 = 0#128)
      ↔ (BitVec.setWidth 257 (a3 ++ a2 ++ a1 ++ a0) +
          BitVec.setWidth 257 (b3 ++ b2 ++ b1 ++ b0)).msb = true := by
  bv_decide (config := { timeout := 60 })

/-- The unsigned-add overflow flag is exactly the `2^w ≤ a + b` predicate on
    the underlying naturals.  Bridges `BitVec.uaddOverflow` (bit-blastable, so
    `bv_decide` can reason about it) to the `decide (2^256 ≤ …)` carry spec. -/
private theorem uaddOverflow_eq_decide (x y : BitVec 256) :
    x.uaddOverflow y = decide (2 ^ 256 ≤ x.toNat + y.toNat) := by
  have hx : (BitVec.setWidth 257 x).toNat = x.toNat := BitVec.toNat_setWidth_of_le (by norm_num)
  have hy : (BitVec.setWidth 257 y).toNat = y.toNat := BitVec.toNat_setWidth_of_le (by norm_num)
  have hlt : (BitVec.setWidth 257 x).toNat + (BitVec.setWidth 257 y).toNat < 2 ^ 257 := by
    rw [hx, hy]
    calc x.toNat + y.toNat < 2 ^ 256 + 2 ^ 256 := Nat.add_lt_add x.isLt y.isLt
      _ = 2 ^ 257 := by rw [← two_mul, mul_comm, ← pow_succ]
  rw [BitVec.uaddOverflow_eq, BitVec.msb_eq_decide, BitVec.toNat_add_of_lt hlt, hx, hy]

/-- `add_limbs a b` returns the 256-bit wrapping sum together with a carry
    flag that is exactly the overflow bit `2^256 ≤ val a + val b`.  Companion
    to `sub_limbs_spec`: the four limb additions form a ripple-carry chain via
    `u128`; the masked low-64 casts reassemble into the 256-bit sum and the top
    `>> 64` is the carry-out. -/
@[step] theorem add_limbs_spec (a b : Array Std.U64 4#usize) :
    add_limbs a b ⦃ r => ∃ out carry,
      r = core.result.Result.Ok (out, carry) ∧ bv256 out = bv256 a + bv256 b ∧
      carry = decide (2 ^ 256 ≤ (bv256 a).toNat + (bv256 b).toNat) ⦄ := by
  unfold add_limbs
  simp only [lift]
  step*
  simp only [U64.Insts.CoreConvertTryFromU128TryFromIntError.try_from,
    core.result.Result.map_err,
    core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch,
    mask_val_le, if_true, bind_tc_ok]
  refine ⟨_, _, rfl, ?_, ?_⟩
  · simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
      Std.U64.bv, Std.U128.bv, Std.UScalar.cast_bv_eq,
      core.convert.num.FromU128U64.from_bv_eq, core.num.U128.wrapping_add_bv_eq,
      Std.UScalar.bv_and, core.num.U64.MAX, Std.U64.ofNat_bv,
      Std.U64.rMax, Std.UScalarTy.U64_numBits_eq,
      i9_post2, i15_post2, i21_post2,
      i_post, i2_post, i4_post, i6_post, i10_post, i12_post, i16_post, i18_post]
    exact add_carry_concat _ _ _ _ _ _ _ _
  · rw [← uaddOverflow_eq_decide, BitVec.uaddOverflow_eq, Bool.eq_iff_iff]
    simp only [bv256, core.num.U128.wrapping_add_bv_eq, core.convert.num.FromU128U64.from_bv_eq,
      i9_post2, i15_post2, i21_post2, i_post, i2_post, i4_post, i6_post, i10_post, i12_post,
      i16_post, i18_post, Std.U64.bv, Std.U128.bv, bne_iff_ne, ne_eq, Std.U128.eq_equiv_bv_eq,
      show (0#u128).bv = 0#128 from rfl, show ((64#i32).toNat) = 64 from rfl]
    exact add_carry_bit _ _ _ _ _ _ _ _

/-- Injecting the incoming bit into the low limb is an `or` against the
    zero-extended bit (the high limbs are unaffected). -/
private theorem or_distrib (a0 a1 a2 a3 z : BitVec 64) :
    a3 ++ a2 ++ a1 ++ (a0 ||| z) = (a3 ++ a2 ++ a1 ++ a0) ||| BitVec.setWidth 256 z := by
  bv_decide

/-- A 256-bit left shift by one followed by injecting a single bit into bit 0
    equals `2·B + bit` as naturals (no wraparound, given `2·B < 2^256`). -/
private theorem shl_or_toNat (B : BitVec 256) (w : BitVec 64)
    (hB : 2 * B.toNat < 2 ^ 256) :
    ((B <<< 1) ||| BitVec.setWidth 256 (w &&& 1)).toNat
      = 2 * B.toNat + (w &&& 1).toNat := by
  have hbit : (w &&& 1).toNat ≤ 1 := by
    rcases (by bv_decide : w &&& 1 = 0 ∨ w &&& 1 = 1) with h | h <;> rw [h] <;> decide
  have heq : (B <<< 1) ||| BitVec.setWidth 256 (w &&& 1)
      = (B <<< 1) + BitVec.setWidth 256 (w &&& 1) := by bv_decide
  have h1 : (B <<< 1).toNat = 2 * B.toNat := by
    rw [BitVec.toNat_shiftLeft, Nat.shiftLeft_eq, pow_one, Nat.mod_eq_of_lt (by omega),
      Nat.mul_comm]
  have h2 : (BitVec.setWidth 256 (w &&& 1)).toNat = (w &&& 1).toNat := by
    rw [BitVec.toNat_setWidth, Nat.mod_eq_of_lt (lt_of_le_of_lt hbit (by norm_num))]
  rw [heq, BitVec.toNat_add, h1, h2, Nat.mod_eq_of_lt (by omega)]

/-- `getLsbD` as a 0/1 natural: the `i`-th bit of `x.toNat`. -/
private theorem getLsbD_toNat {n : Nat} (x : BitVec n) (i : Nat) :
    (x.getLsbD i).toNat = x.toNat / 2 ^ i % 2 := by
  rw [BitVec.getLsbD, Nat.testBit_eq_decide_div_mod_eq]
  rcases Nat.mod_two_eq_zero_or_one (x.toNat / 2 ^ i) with h | h <;> simp [h]

/-- Bit `k` of the 512-bit wide value is bit `k % 64` of limb `k / 64`. -/
private theorem getLsbD_bv512 (w : Array Std.U64 8#usize) (k : Nat) (hk : k < 512) :
    (bv512 w).getLsbD k = (w.val[k / 64]!.bv).getLsbD (k % 64) := by
  unfold bv512
  simp only [BitVec.getLsbD_append]
  split_ifs <;> (congr 1 <;> first | omega | (congr 2 <;> omega))

/-- Loop invariant for the shift-and-subtract reduction.  At each step the
    accumulator equals `(acc·2^remaining + low `remaining` bits of wide) mod p`,
    and stays below `p`.  Hypotheses: `p > 0`, `2p ≤ 2^256` (so one conditional
    subtraction restores `< p`), `remaining ≤ 512`, and the entry invariant
    `acc < p`. -/
theorem reduce_wide_rec_spec (wide : Array Std.U64 8#usize)
    (modulus : Array Std.U64 4#usize) (remaining : Std.Usize)
    (acc : Array Std.U64 4#usize)
    (hp0 : 0 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256)
    (hrem : remaining.val ≤ 512) (hacc : val4 acc < val4 modulus) :
    field.reduce_wide_rec wide modulus remaining acc ⦃ r => ∃ out,
      r = core.result.Result.Ok out ∧ val4 out < val4 modulus ∧
      val4 out = (val4 acc * 2 ^ remaining.val + val8 wide % 2 ^ remaining.val)
        % val4 modulus ⦄ := by
  unfold field.reduce_wide_rec
  by_cases h : remaining = 0#usize
  · simp only [h, reduceIte]
    simp only [WP.spec_ok]
    exact ⟨acc, rfl, hacc, by simp [Nat.mod_one, Nat.mod_eq_of_lt hacc]⟩
  · simp only [h, reduceIte]
    step*
    · -- subtract branch (with_bit ≥ modulus)
      rw [r_post1]
      simp only [core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch,
        bind_tc_ok]
      -- the recursive call (self-spec) is applied here in a *light* context;
      -- the heavy bit-vector facts are established afterwards, per goal.
      step
      · -- precondition `val4 r < val4 modulus` for the recursive call
        have h2acc : 2 * val4 acc < 2 ^ 256 := by omega
        have hwb : val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
            = 2 * val4 acc + incoming.val := by
          have hbv : bv256 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
              = bv256 shifted ||| BitVec.setWidth 256 incoming.bv := by
            simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
              i2_post, i3_post2, i4_post, i5_post, i6_post]
            exact or_distrib _ _ _ _ _
          simp only [val4, hbv, shifted_post, incoming_post2, ← U64.bv_toNat,
            show (1#u64).bv = 1#64 from rfl]
          exact shl_or_toNat (bv256 acc) i1.bv h2acc
        have hwbge : val4 modulus ≤ val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp)) := by
          have hbt : b = true := ‹b = true›
          rw [b_post] at hbt; exact of_decide_eq_true hbt
        have hinc1 : incoming.val ≤ 1 := by
          rw [incoming_post1, Std.UScalar.val_and, show (1#u64).val = 1 from rfl]
          exact Nat.and_le_right
        have hsub : val4 r
            = val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp)) - val4 modulus := by
          unfold val4; rw [r_post2, BitVec.toNat_sub_of_le hwbge]
        omega
      · -- the post, from the IH `r_post3`
        rename_i rsub xb2 xb1 hrp1 hrsub hborrow xst
        have h2acc : 2 * val4 acc < 2 ^ 256 := by omega
        have hwb : val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
            = 2 * val4 acc + incoming.val := by
          have hbv : bv256 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
              = bv256 shifted ||| BitVec.setWidth 256 incoming.bv := by
            simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
              i2_post, i3_post2, i4_post, i5_post, i6_post]
            exact or_distrib _ _ _ _ _
          simp only [val4, hbv, shifted_post, incoming_post2, ← U64.bv_toNat,
            show (1#u64).bv = 1#64 from rfl]
          exact shl_or_toNat (bv256 acc) i1.bv h2acc
        have hwbge : val4 modulus ≤ val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp)) := by
          have hbt : b = true := ‹b = true›
          rw [b_post] at hbt; exact of_decide_eq_true hbt
        have hinc1 : incoming.val ≤ 1 := by
          rw [incoming_post1, Std.UScalar.val_and, show (1#u64).val = 1 from rfl]
          exact Nat.and_le_right
        have hsub : val4 rsub
            = val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp)) - val4 modulus := by
          unfold val4; rw [hrsub, BitVec.toNat_sub_of_le hwbge]
        have hremeq : remaining.val = bit.val + 1 := by omega
        have hinc : incoming.val = val8 wide / 2 ^ bit.val % 2 := by
          have hb : bit.val < 512 := by omega
          have e1 : incoming.val = i.val / 2 ^ bit_idx.val % 2 := by
            rw [incoming_post1, Std.UScalar.val_and, i1_post1]
            simp only [show (1#u64).val = 1 from rfl, Nat.shiftRight_eq_div_pow,
              Nat.and_one_is_mod]
          have key := getLsbD_bv512 wide bit.val hb
          rw [← limb_idx_post, ← i_post, ← bit_idx_post] at key
          have key2 := congrArg Bool.toNat key
          rw [getLsbD_toNat, getLsbD_toNat, U64.bv_toNat] at key2
          rw [e1, val8]; exact key2.symm
        have hmod : val8 wide % 2 ^ remaining.val
            = (val8 wide / 2 ^ bit.val % 2) * 2 ^ bit.val + val8 wide % 2 ^ bit.val := by
          rw [hremeq, pow_succ, Nat.mod_mul]; ring
        have hcong : (val4 rsub * 2 ^ bit.val + val8 wide % 2 ^ bit.val) % val4 modulus
            = ((2 * val4 acc + incoming.val) * 2 ^ bit.val + val8 wide % 2 ^ bit.val)
              % val4 modulus := by
          have hval : 2 * val4 acc + incoming.val = val4 rsub + val4 modulus := by omega
          have hmeq : val4 rsub ≡ 2 * val4 acc + incoming.val [MOD val4 modulus] := by
            rw [hval]; exact (Nat.add_mod_right (val4 rsub) (val4 modulus)).symm
          exact (hmeq.mul_right _).add_right _
        refine ⟨r, r_post1, r_post2, ?_⟩
        rw [r_post3, hcong, hmod, hinc, hremeq]
        congr 1
        ring
    · -- no-subtract branch (with_bit < modulus): result is the IH directly
      refine ⟨r, r_post1, r_post2, ?_⟩
      have h2acc : 2 * val4 acc < 2 ^ 256 := by omega
      -- H1: the bit-injected, shifted accumulator has value `2*acc + incoming`.
      have hwb : val4 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
          = 2 * val4 acc + incoming.val := by
        have hbv : bv256 (Array.make 4#usize [i3, i4, i5, i6] (by simp))
            = bv256 shifted ||| BitVec.setWidth 256 incoming.bv := by
          simp only [bv256, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
            i2_post, i3_post2, i4_post, i5_post, i6_post]
          exact or_distrib _ _ _ _ _
        simp only [val4, hbv, shifted_post, incoming_post2, ← U64.bv_toNat,
          show (1#u64).bv = 1#64 from rfl]
        exact shl_or_toNat (bv256 acc) i1.bv h2acc
      rw [r_post3, hwb]
      have hrem : remaining.val = bit.val + 1 := by omega
      -- H2a: incoming bit equals bit `bit` of the wide value (via the limb bridge).
      have hinc : incoming.val = val8 wide / 2 ^ bit.val % 2 := by
        have hb : bit.val < 512 := by omega
        -- incoming bit, in terms of limb `i` and bit position `bit_idx`
        have e1 : incoming.val = i.val / 2 ^ bit_idx.val % 2 := by
          rw [incoming_post1, Std.UScalar.val_and, i1_post1]
          simp only [show (1#u64).val = 1 from rfl, Nat.shiftRight_eq_div_pow,
            Nat.and_one_is_mod]
        -- bit `bit` of the wide value is bit `bit_idx` of limb `i`
        have key := getLsbD_bv512 wide bit.val hb
        rw [← limb_idx_post, ← i_post, ← bit_idx_post] at key
        have key2 := congrArg Bool.toNat key
        rw [getLsbD_toNat, getLsbD_toNat, U64.bv_toNat] at key2
        rw [e1, val8]; exact key2.symm
      -- H2b: split off the top bit of the low `remaining` bits.
      have hmod : val8 wide % 2 ^ remaining.val
          = (val8 wide / 2 ^ bit.val % 2) * 2 ^ bit.val + val8 wide % 2 ^ bit.val := by
        rw [hrem, pow_succ, Nat.mod_mul]; ring
      rw [hmod, hinc, hrem]
      congr 1
      ring
termination_by remaining.val
decreasing_by all_goals scalar_tac

/-- Top-level correctness of `reduce_wide`: the result is the wide product reduced
    modulo `p`, in canonical form (`< p`).  Instantiates the loop invariant at
    `remaining = 512`, `acc = 0`; since `val8 wide < 2^512`, the `% 2^512`
    collapses and the accumulator term vanishes. -/
theorem reduce_wide_spec (wide : Array Std.U64 8#usize)
    (modulus : Array Std.U64 4#usize)
    (hp0 : 0 < val4 modulus) (hp : 2 * val4 modulus ≤ 2 ^ 256) :
    field.reduce_wide wide modulus ⦃ r => ∃ out,
      r = core.result.Result.Ok out ∧ val4 out < val4 modulus ∧
      val4 out = val8 wide % val4 modulus ⦄ := by
  unfold field.reduce_wide
  simp only [field.LIMBS]
  step*
  have hi1 : i1.val = 512 := by scalar_tac
  have hrepeat : val4 (Array.repeat 4#usize 0#u64) = 0 := by
    simp [val4, bv256, Array.repeat]
  have hval8 : val8 wide < 2 ^ 512 := (bv512 wide).isLt
  have IH := reduce_wide_rec_spec wide modulus i1 (Array.repeat 4#usize 0#u64) hp0 hp
    (by omega) (by rw [hrepeat]; exact hp0)
  apply WP.spec_mono IH
  rintro res ⟨out, hres, hlt, heq⟩
  refine ⟨out, hres, hlt, ?_⟩
  rw [heq, hrepeat, Nat.zero_mul, Nat.zero_add, hi1, Nat.mod_eq_of_lt hval8]

/-- `mod_add a b modulus` computes the canonical modular sum `(a + b) % modulus`
    for reduced inputs `a, b < modulus`, when `2·modulus ≤ 2^256` (so the sum stays
    below `2^256` and a single conditional subtraction restores the residue).
    Composes `add_limbs` (full adder), `gte_modulus`, and `sub_limbs`. -/
@[step] theorem mod_add_spec (a b modulus : Array Std.U64 4#usize)
    (ha : val4 a < val4 modulus) (hb : val4 b < val4 modulus)
    (hmod : 2 * val4 modulus ≤ 2 ^ 256) :
    mod_add a b modulus ⦃ r => ∃ out, r = core.result.Result.Ok out ∧
      val4 out = (val4 a + val4 b) % val4 modulus ⦄ := by
  unfold mod_add
  step*
  have hab : (bv256 a).toNat + (bv256 b).toNat < 2 ^ 256 := by
    unfold val4 at ha hb hmod; omega
  have hval_r : val4 r = val4 a + val4 b := by
    unfold val4; rw [r_post2, BitVec.toNat_add, Nat.mod_eq_of_lt hab]
  have hnc : ¬ (2 ^ 256 ≤ (bv256 a).toNat + (bv256 b).toNat) := by omega
  simp only [r_post1, r_post3, hnc, decide_false, Bool.false_eq_true, if_false,
    core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch, bind_tc_ok]
  step
  simp only [b1_post, decide_eq_true_eq]
  by_cases hge : val4 modulus ≤ val4 r
  · rw [if_pos hge]
    step
    simp only [r1_post1, bind_tc_ok]
    refine ⟨r1, rfl, ?_⟩
    have hle : bv256 modulus ≤ bv256 r := BitVec.le_def.mpr hge
    have hr1 : val4 r1 = val4 r - val4 modulus := by
      unfold val4; rw [r1_post2, BitVec.toNat_sub_of_le hle]
    rw [hr1, hval_r, Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
  · rw [if_neg hge]
    refine ⟨r, rfl, ?_⟩
    rw [hval_r, Nat.mod_eq_of_lt (by omega)]

/-- `mod_sub a b modulus` computes the canonical modular difference
    `(a + modulus - b) % modulus` for reduced inputs `a, b < modulus`, when
    `2·modulus ≤ 2^256`.  Composes `sub_limbs` (with its borrow characterization)
    and `add_limbs`. -/
@[step] theorem mod_sub_spec (a b modulus : Array Std.U64 4#usize)
    (ha : val4 a < val4 modulus) (hb : val4 b < val4 modulus)
    (hmod : 2 * val4 modulus ≤ 2 ^ 256) :
    mod_sub a b modulus ⦃ r => ∃ out, r = core.result.Result.Ok out ∧
      val4 out = (val4 a + val4 modulus - val4 b) % val4 modulus ⦄ := by
  unfold mod_sub
  step*
  simp only [r_post1, r_post3,
    core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch, bind_tc_ok,
    decide_eq_true_eq]
  by_cases hlt : val4 a < val4 b
  · rw [if_pos hlt]
    step
    simp only [r1_post1, bind_tc_ok]
    refine ⟨_, rfl, ?_⟩
    have hXlt : val4 a + val4 modulus < 2 ^ 256 := by omega
    have add_bv : bv256 r1 = bv256 a + bv256 modulus - bv256 b := by
      rw [r1_post2, r_post2]; ring
    have hXval : (bv256 a + bv256 modulus).toNat = val4 a + val4 modulus := by
      rw [BitVec.toNat_add]; exact Nat.mod_eq_of_lt hXlt
    have hbleX : bv256 b ≤ bv256 a + bv256 modulus := by
      rw [BitVec.le_def, hXval]; show val4 b ≤ val4 a + val4 modulus; omega
    have hsum : val4 r1 = val4 a + val4 modulus - val4 b := by
      have h1 : (bv256 r1).toNat = (bv256 a + bv256 modulus).toNat - (bv256 b).toNat := by
        rw [add_bv, BitVec.toNat_sub_of_le hbleX]
      simp only [val4] at *
      omega
    rw [hsum, Nat.mod_eq_of_lt (by omega)]
  · rw [if_neg hlt]
    refine ⟨r, rfl, ?_⟩
    have hge : val4 b ≤ val4 a := by omega
    have hle : bv256 b ≤ bv256 a := BitVec.le_def.mpr hge
    have hdiff : val4 r = val4 a - val4 b := by
      unfold val4; rw [r_post2, BitVec.toNat_sub_of_le hle]
    rw [hdiff, show val4 a + val4 modulus - val4 b = val4 modulus + (val4 a - val4 b) from by omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]

set_option maxHeartbeats 10000000 in
set_option maxRecDepth 100000 in
set_option exponentiation.threshold 600 in
/-- `mul_wide a b` returns the exact 512-bit product `a * b` (no reduction).
    The unrolled 4x4 schoolbook reassembles to the full product. -/
@[step] theorem mul_wide_spec (a b : Array Std.U64 4#usize) :
    mul_wide a b ⦃ r => ∃ out, r = core.result.Result.Ok out ∧
      val8 out = val4 a * val4 b ⦄ := by
  unfold mul_wide
  simp only [lift]
  step*
  have hr7 : (r7 : Std.U128).val ≤ Std.U64.max := by scalar_tac
  simp only [U64.Insts.CoreConvertTryFromU128TryFromIntError.try_from,
    core.result.Result.map_err,
    core.result.Result.Insts.CoreOpsTry_traitTryTResultInfallibleE.branch,
    mask_val_le, hr7, if_true, bind_tc_ok]
  refine ⟨_, rfl, ?_⟩
  -- Aeneas `step*` materializes the `>>> 64` carries as variables but INLINES the `&&& mask`
  -- intermediates, so each output limb's term recursively embeds every prior column and the
  -- assembled goal is too large for any single tactic.  Materialize the nine forward-fed masks
  -- (`R*`) and seven output masks (`M*`) as opaque vars: `set` rewrites the carry hypotheses too,
  -- and `clear_value` stops `omega` from re-unfolding them.  Innermost first so each stays small.
  set R1 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i) (core.convert.num.FromU128U64.from i5)) i8) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR1
  set R2 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i) (core.convert.num.FromU128U64.from i10)) i13) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR2
  set R3 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i) (core.convert.num.FromU128U64.from i15)) i18) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR3
  set R21 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i19) (core.convert.num.FromU128U64.from i5)) R2) i27) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR21
  set R31 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i19) (core.convert.num.FromU128U64.from i10)) R3) i32) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR31
  set R41 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i19) (core.convert.num.FromU128U64.from i15)) r4) i37) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR41
  set R32 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i38) (core.convert.num.FromU128U64.from i5)) R31) i46) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR32
  set R42 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i38) (core.convert.num.FromU128U64.from i10)) R41) i51) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR42
  set R51 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i38) (core.convert.num.FromU128U64.from i15)) r5) i56) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hR51
  set M0 := ((core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i) (core.convert.num.FromU128U64.from i2)) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM0
  set M1 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i19) (core.convert.num.FromU128U64.from i2)) R1) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM1
  set M2 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i38) (core.convert.num.FromU128U64.from i2)) R21) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM2
  set M3 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i57) (core.convert.num.FromU128U64.from i2)) R32) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM3
  set M4 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i57) (core.convert.num.FromU128U64.from i5)) R42) i65) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM4
  set M5 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i57) (core.convert.num.FromU128U64.from i10)) R51) i70) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM5
  set M6 := ((core.num.U128.wrapping_add (core.num.U128.wrapping_add (core.num.U128.wrapping_mul (core.convert.num.FromU128U64.from i57) (core.convert.num.FromU128U64.from i15)) r6) i75) &&& core.convert.num.FromU128U64.from core.num.U64.MAX) with hM6
  clear_value R1 R2 R3 R21 R31 R41 R32 R42 R51 M0 M1 M2 M3 M4 M5 M6
  have hmax : (core.num.U64.MAX).val = 2 ^ 64 - 1 := rfl
  have hva : val4 a = i.val + i19.val * 2 ^ 64 + i38.val * 2 ^ 128 + i57.val * 2 ^ 192 := by
    simp only [val4, bv256, ← i_post, ← i19_post, ← i38_post, ← i57_post,
      toNat_append_add, ← Std.U64.bv_toNat]; ring
  have hvb : val4 b = i2.val + i5.val * 2 ^ 64 + i10.val * 2 ^ 128 + i15.val * 2 ^ 192 := by
    simp only [val4, bv256, ← i2_post, ← i5_post, ← i10_post, ← i15_post,
      toNat_append_add, ← Std.U64.bv_toNat]; ring
  have hrhs : val4 a * val4 b =
      i.val * i2.val
      + (i.val * i5.val + i19.val * i2.val) * 2 ^ 64
      + (i.val * i10.val + i19.val * i5.val + i38.val * i2.val) * 2 ^ 128
      + (i.val * i15.val + i19.val * i10.val + i38.val * i5.val + i57.val * i2.val) * 2 ^ 192
      + (i19.val * i15.val + i38.val * i10.val + i57.val * i5.val) * 2 ^ 256
      + (i38.val * i15.val + i57.val * i10.val) * 2 ^ 320
      + (i57.val * i15.val) * 2 ^ 384 := by
    rw [hva, hvb]; ring
  rw [hrhs]
  -- Each 64×64 product is `≤ (2^64−1)^2`, so every intermediate `wrapping_*` (mod 2^128) stays
  -- below `2^128` and the truncations are exact.  These bounds let the carry chain be expressed
  -- with `Nat.div`/`Nat.mod` by `2^64` only.
  have hbnd : ∀ x y : Std.U64, x.val * y.val ≤ (2 ^ 64 - 1) * (2 ^ 64 - 1) := fun x y =>
    Nat.mul_le_mul (by scalar_tac) (by scalar_tac)
  have b00 := hbnd i i2;   have b01 := hbnd i i5;   have b02 := hbnd i i10;   have b03 := hbnd i i15
  have b10 := hbnd i19 i2; have b11 := hbnd i19 i5; have b12 := hbnd i19 i10; have b13 := hbnd i19 i15
  have b20 := hbnd i38 i2; have b21 := hbnd i38 i5; have b22 := hbnd i38 i10; have b23 := hbnd i38 i15
  have b30 := hbnd i57 i2; have b31 := hbnd i57 i5; have b32 := hbnd i57 i10; have b33 := hbnd i57 i15
  -- Project the `.val` of each materialized mask, then reduce every carry post AND every mask
  -- value to a clean `Nat` (products and lower carries kept as atoms) in one `simp`.
  have hM0v := congrArg Std.UScalar.val hM0
  have hM1v := congrArg Std.UScalar.val hM1
  have hM2v := congrArg Std.UScalar.val hM2
  have hM3v := congrArg Std.UScalar.val hM3
  have hM4v := congrArg Std.UScalar.val hM4
  have hM5v := congrArg Std.UScalar.val hM5
  have hM6v := congrArg Std.UScalar.val hM6
  have hR1v := congrArg Std.UScalar.val hR1
  have hR2v := congrArg Std.UScalar.val hR2
  have hR3v := congrArg Std.UScalar.val hR3
  have hR21v := congrArg Std.UScalar.val hR21
  have hR31v := congrArg Std.UScalar.val hR31
  have hR41v := congrArg Std.UScalar.val hR41
  have hR32v := congrArg Std.UScalar.val hR32
  have hR42v := congrArg Std.UScalar.val hR42
  have hR51v := congrArg Std.UScalar.val hR51
  simp only [core.num.U128.wrapping_mul_val_eq, core.num.U128.wrapping_add_val_eq,
    Std.UScalar.val_and, core.convert.num.FromU128U64.from_val_eq, UScalar.size, UScalarTy.numBits,
    hmax, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow] at i8_post1 i13_post1 i18_post1 r4_post1 i27_post1 i32_post1 i37_post1 r5_post1 i46_post1 i51_post1 i56_post1 r6_post1 i65_post1 i70_post1 i75_post1 r7_post1 hM0v hM1v hM2v hM3v hM4v hM5v hM6v hR1v hR2v hR3v hR21v hR31v hR41v hR32v hR42v hR51v
  simp only [val8, bv512, Array.make, List.getElem!_cons_zero, List.getElem!_cons_succ,
    toNat_append_add, Std.U64.bv_toNat,
    UScalar.cast_val_eq, UScalarTy.numBits, Std.UScalar.val_and,
    core.convert.num.FromU128U64.from_val_eq, core.num.U128.wrapping_mul_val_eq,
    core.num.U128.wrapping_add_val_eq, UScalar.size, hmax,
    Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]
  clear hva hvb hrhs hmax hbnd hR1 hR2 hR3 hR21 hR31 hR41 hR32 hR42 hR51 hM0 hM1 hM2 hM3 hM4 hM5 hM6
  have h := comba_recombine
    (P00 := i.val*i2.val) (P01 := i.val*i5.val) (P02 := i.val*i10.val) (P03 := i.val*i15.val)
    (P10 := i19.val*i2.val) (P11 := i19.val*i5.val) (P12 := i19.val*i10.val) (P13 := i19.val*i15.val)
    (P20 := i38.val*i2.val) (P21 := i38.val*i5.val) (P22 := i38.val*i10.val) (P23 := i38.val*i15.val)
    (P30 := i57.val*i2.val) (P31 := i57.val*i5.val) (P32 := i57.val*i10.val) (P33 := i57.val*i15.val)
    (c0 := i8.val) (c1 := i13.val) (c2 := i18.val) (c3 := r4.val) (c4 := i27.val) (c5 := i32.val)
    (c6 := i37.val) (c7 := r5.val) (c8 := i46.val) (c9 := i51.val) (c10 := i56.val) (c11 := r6.val)
    (c12 := i65.val) (c13 := i70.val) (c14 := i75.val) (c15 := r7.val)
    (mA1 := R1.val) (mA2 := R2.val) (mA3 := R3.val)
    (mB1 := R21.val) (mB2 := R31.val) (mB3 := R41.val)
    (mC2 := R32.val) (mC3 := R42.val) (mC4 := R51.val)
    (L0 := M0.val) (L1 := M1.val) (L2 := M2.val) (L3 := M3.val)
    (L4 := M4.val) (L5 := M5.val) (L6 := M6.val) (L7 := r7.val)
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
    (by omega)
  -- The top limb is `r7.val` directly (no mask); omega needs it numerically `< 2⁶⁴`.
  have hr7' : (r7 : Std.U128).val < 2 ^ 64 := by scalar_tac
  -- Goal is now the small Horner form `Σ Mₖ.val % 2⁶⁴ · 2⁶⁴ᵏ = a·b`; `h` gives the same sum
  -- without the masks (each `Mₖ.val < 2⁶⁴` via `hM*v`), so omega bridges it directly.  Clear the
  -- `% 2¹²⁸`-laden carry / R-value / bound hypotheses first so omega's model stays small.
  clear i8_post1 i13_post1 i18_post1 r4_post1 i27_post1 i32_post1 i37_post1 r5_post1
    i46_post1 i51_post1 i56_post1 r6_post1 i65_post1 i70_post1 i75_post1 r7_post1
    hR1v hR2v hR3v hR21v hR31v hR41v hR32v hR42v hR51v
    b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33
  -- Each masked output limb is `< 2^64`, so the `& mask` (`% 2^64`) on the seven low
  -- limbs and the bound `hr7'` on the unmasked top limb make every `% 2^64` the identity.
  -- Removing the masks is ESSENTIAL: `omega` over the `% 2^64`-laden Horner form does not
  -- terminate (the eight residues × `2^448`-scale coefficients explode its search).  With
  -- the masks gone the goal is `comba_recombine`'s identity `h` up to Horner/expanded
  -- regrouping, which `linarith` discharges.
  have b0 : M0.val < 2 ^ 64 := by rw [hM0v]; exact Nat.mod_lt _ (by norm_num)
  have b1 : M1.val < 2 ^ 64 := by rw [hM1v]; exact Nat.mod_lt _ (by norm_num)
  have b2 : M2.val < 2 ^ 64 := by rw [hM2v]; exact Nat.mod_lt _ (by norm_num)
  have b3 : M3.val < 2 ^ 64 := by rw [hM3v]; exact Nat.mod_lt _ (by norm_num)
  have b4 : M4.val < 2 ^ 64 := by rw [hM4v]; exact Nat.mod_lt _ (by norm_num)
  have b5 : M5.val < 2 ^ 64 := by rw [hM5v]; exact Nat.mod_lt _ (by norm_num)
  have b6 : M6.val < 2 ^ 64 := by rw [hM6v]; exact Nat.mod_lt _ (by norm_num)
  rw [Nat.mod_eq_of_lt b0, Nat.mod_eq_of_lt b1, Nat.mod_eq_of_lt b2, Nat.mod_eq_of_lt b3,
    Nat.mod_eq_of_lt b4, Nat.mod_eq_of_lt b5, Nat.mod_eq_of_lt b6, Nat.mod_eq_of_lt hr7']
  linarith [h]

end minroot_core.field.spec
