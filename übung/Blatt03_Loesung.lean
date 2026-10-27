/-
Übungsblatt 3 -- Mathematik in Lean
Ersetzen Sie jedes `sorry` durch einen Beweis. Die Datei ist fertig, wenn Lean
keine Warnung `declaration uses 'sorry'` mehr meldet.
-/
--<<nurLoesung>>
-- Loesungsfassung. Die Aufgabenfassung lean/Blatt03.lean erzeugt erzeuge.pl daraus.
--<<nurLoesungEnd>>
import Mathlib

namespace Uebung03

/-! ## B1: Mengen sind Praedikate (Formalisierung von A1) -/

--<<b1>>
/-- Aufgabe B1 (a): Distributivitaet. -/
theorem distributiv {α : Type*} (s t u : Set α) :
    s ∩ (t ∪ u) = (s ∩ t) ∪ (s ∩ u) := by
  -- `ext x`, `constructor`, dann zerlegen, etwa mit `rintro ⟨hs, ht | hu⟩`
  --<<loesung>>
  ext x
  constructor
  · rintro ⟨hs, ht | hu⟩
    · exact Or.inl ⟨hs, ht⟩
    · exact Or.inr ⟨hs, hu⟩
  · rintro (⟨hs, ht⟩ | ⟨hs, hu⟩)
    · exact ⟨hs, Or.inl ht⟩
    · exact ⟨hs, Or.inr hu⟩
  --<<loesungEnd>>

/-- Aufgabe B1 (b), im Termmodus: Komplementbildung kehrt Inklusionen um. -/
theorem kompl_antiton {α : Type*} (s t : Set α) (h : s ⊆ t) : tᶜ ⊆ sᶜ :=
  --<<loesung>>
  fun _ hx hs => hx (h hs)
  --<<loesungEnd>>

/-- Aufgabe B1 (c): De Morgan fuer den Durchschnitt, eine Inklusion nach der
anderen. Fuer eine der beiden brauchen Sie `Classical.em`. -/
theorem de_morgan_1 {α : Type*} (s t : Set α) : sᶜ ∪ tᶜ ⊆ (s ∩ t)ᶜ := by
  --<<loesung>>
  rintro x (hs | ht) ⟨h1, h2⟩
  · exact hs h1
  · exact ht h2
  --<<loesungEnd>>

theorem de_morgan_2 {α : Type*} (s t : Set α) : (s ∩ t)ᶜ ⊆ sᶜ ∪ tᶜ := by
  --<<loesung>>
  intro x hx
  rcases Classical.em (x ∈ s) with hs | hs
  · exact Or.inr (fun ht => hx ⟨hs, ht⟩)
  · exact Or.inl hs
  --<<loesungEnd>>

theorem de_morgan {α : Type*} (s t : Set α) : (s ∩ t)ᶜ = sᶜ ∪ tᶜ :=
  -- aus den beiden Inklusionen, mit `Set.Subset.antisymm`
  --<<loesung>>
  Set.Subset.antisymm (de_morgan_2 s t) (de_morgan_1 s t)
  --<<loesungEnd>>

/- Worauf beruhen Ihre beiden Beweise? Achten Sie auf `Classical.choice`. -/
#print axioms de_morgan_1
#print axioms de_morgan_2
--<<b1End>>

--<<nurLoesung>>
/- `#print axioms` meldet oben: Der Beweis von `de_morgan_1` beruht auf keinem
   Axiom, der von `de_morgan_2` ueber `Classical.em` auf `propext`,
   `Classical.choice` und `Quot.sound`. Das zeigt, worauf diese Beweise
   beruhen, nicht, was noetig ist.
   Gegenprobe: die Mathlib-Fassungen derselben Aussagen. -/
example {α : Type*} (s t u : Set α) : s ∩ (t ∪ u) = (s ∩ t) ∪ (s ∩ u) :=
  Set.inter_union_distrib_left s t u
example {α : Type*} (s t : Set α) (h : s ⊆ t) : tᶜ ⊆ sᶜ :=
  Set.compl_subset_compl.mpr h
example {α : Type*} (s t : Set α) : (s ∩ t)ᶜ = sᶜ ∪ tᶜ :=
  Set.compl_inter s t
--<<nurLoesungEnd>>

/-! ## B2: Rechnen mit Matrizen (Formalisierung von A2) -/

section Matrizen

--<<b2>>
/- `!![a, b; c, d]` ist eine 2×2-Matrix, `![x, y]` ein Vektor in ℝ²; nach
   `open Matrix` heisst Matrix mal Vektor `M *ᵥ v`. In ℤ rechnet der Kern
   Produkte, Determinanten und Potenzen mit `decide` selbst aus. In ℝ geht
   das nicht; dort vergleicht man Eintraege. -/
open Matrix

/-- Aufgabe B2 (a): die Rechnungen aus A2 (a). -/
example : !![1, 2; 3, 4] * !![0, 1; 1, 0] = (!![2, 1; 4, 3] : Matrix (Fin 2) (Fin 2) ℤ) := by
  --<<loesung>>
  decide
  --<<loesungEnd>>

example : (!![0, 1; 1, 0] * !![1, 2; 3, 4] : Matrix (Fin 2) (Fin 2) ℤ) ≠
    !![1, 2; 3, 4] * !![0, 1; 1, 0] := by
  --<<loesung>>
  decide
  --<<loesungEnd>>

example : (!![1, 2; 3, 4] : Matrix (Fin 2) (Fin 2) ℤ).det = -2 := by
  --<<loesung>>
  decide
  --<<loesungEnd>>

/-- Aufgabe B2 (b): die Formel fuer die Inverse aus A2 (b). -/
theorem inverse_2x2 (a b c d : ℝ) (h : a * d - b * c ≠ 0) :
    !![a, b; c, d] * ((a * d - b * c)⁻¹ • !![d, -b; -c, a]) = 1 := by
  -- wie in A2 (b): erst `have` mit M * !![d, -b; -c, a] = (a * d - b * c) • 1
  --<<loesung>>
  have hM : !![a, b; c, d] * !![d, -b; -c, a] =
      (a * d - b * c) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
    rw [Matrix.mul_fin_two]
    ext i j
    fin_cases i <;> fin_cases j <;> simp <;> ring
  rw [Matrix.mul_smul, hM, smul_smul, inv_mul_cancel₀ h, one_smul]
  --<<loesungEnd>>

/-- Aufgabe B2 (c): `v ↦ M v` ist linear (A2 (c)). -/
theorem mulVec_linear (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ) (c : ℝ) :
    M *ᵥ (v + w) = M *ᵥ v + M *ᵥ w ∧ M *ᵥ (c • v) = c • (M *ᵥ v) := by
  --<<loesung>>
  constructor
  · ext i
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    ring
  · ext i
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    ring
  --<<loesungEnd>>

/-- Aufgabe B2 (d): die Drehung `R` aus A2 (d), viermal hintereinander. -/
example : (!![0, -1; 1, 0] : Matrix (Fin 2) (Fin 2) ℤ) ^ 4 = 1 := by
  --<<loesung>>
  decide
  --<<loesungEnd>>
--<<b2End>>

--<<nurLoesung>>
/- Gegenprobe: die Mathlib-Fassungen derselben Aussagen. -/
example (a b c d : ℝ) : !![a, b; c, d].det = a * d - b * c :=
  Matrix.det_fin_two_of a b c d
example (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ) :
    M *ᵥ (v + w) = M *ᵥ v + M *ᵥ w :=
  Matrix.mulVec_add M v w
example (M : Matrix (Fin 2) (Fin 2) ℝ) (c : ℝ) (v : Fin 2 → ℝ) :
    M *ᵥ (c • v) = c • (M *ᵥ v) :=
  Matrix.mulVec_smul M c v
--<<nurLoesungEnd>>

end Matrizen

/-! ## B3: Lineare Abbildungen und ihr Kern (Formalisierung von A3) -/

--<<b3>>
/- In Mathlib waere `f` ein Element von `(Fin n → ℝ) →ₗ[ℝ] (Fin m → ℝ)`. Hier
   ist `f` eine gewoehnliche Funktion von ℝ^n nach ℝ^m; die Linearitaet steht
   als Voraussetzungen `hadd` und `hsmul` daneben. -/

/-- Aufgabe B3 (a): `f` bildet `0` auf `0` ab ... -/
theorem lin_null {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ))
    (hadd : ∀ v w, f (v + w) = f v + f w) : f 0 = 0 := by
  -- wie in A3 (a): erst `have` mit f 0 + f 0 = f 0 + 0, dann `add_left_cancel`
  --<<loesung>>
  have h : f 0 + f 0 = f 0 + 0 := by rw [← hadd, add_zero, add_zero]
  exact add_left_cancel h
  --<<loesungEnd>>

/-- ... und `-v` auf `-f v`. -/
theorem lin_neg {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ))
    (hsmul : ∀ (c : ℝ) v, f (c • v) = c • f v) (v : Fin n → ℝ) : f (-v) = -f v := by
  --<<loesung>>
  rw [← neg_one_smul ℝ v, hsmul, neg_one_smul]
  --<<loesungEnd>>

/-- Aufgabe B3 (b): Der Kern als Unterraum, gebuendelt: Traegermenge und
drei Beweise. `v ∈ {v | f v = 0}` ist definitorisch `f v = 0`;
`rw [Set.mem_setOf_eq] at hv` macht das fuer `rw` sichtbar. -/
def kern {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ))
    (hadd : ∀ v w, f (v + w) = f v + f w)
    (hsmul : ∀ (c : ℝ) v, f (c • v) = c • f v) : Submodule ℝ (Fin n → ℝ) where
  carrier := {v | f v = 0}
  zero_mem' := by
    --<<loesung>>
    exact lin_null f hadd
    --<<loesungEnd>>
  add_mem' := by
    --<<loesung>>
    intro v w hv hw
    rw [Set.mem_setOf_eq] at hv hw ⊢
    rw [hadd, hv, hw, add_zero]
    --<<loesungEnd>>
  smul_mem' := by
    --<<loesung>>
    intro c v hv
    rw [Set.mem_setOf_eq] at hv ⊢
    rw [hsmul, hv, smul_zero]
    --<<loesungEnd>>

/-- Aufgabe B3 (c): Die Verschiebung aus A3 (d) ist nicht additiv. -/
theorem verschiebung_nicht_linear : ∃ g : ℝ → ℝ, ¬ ∀ x y, g (x + y) = g x + g y :=
  --<<loesung>>
  ⟨fun x => x + 1, fun h => by have h0 := h 0 0; norm_num at h0⟩
  --<<loesungEnd>>
--<<b3End>>

--<<nurLoesung>>
/- Probe: Mitgliedschaft im Kern ist eine gewoehnliche Aussage. -/
example {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ)) (hadd : ∀ v w, f (v + w) = f v + f w)
    (hsmul : ∀ (c : ℝ) v, f (c • v) = c • f v) (v : Fin n → ℝ) :
    v ∈ kern f hadd hsmul ↔ f v = 0 :=
  Iff.rfl

/- Mathlib buendelt `f` mit `hadd` und `hsmul` zu einer linearen Abbildung;
   ihr Kern heisst `LinearMap.ker`. -/
example {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ)) (hadd : ∀ v w, f (v + w) = f v + f w)
    (hsmul : ∀ (c : ℝ) v, f (c • v) = c • f v) : (Fin n → ℝ) →ₗ[ℝ] (Fin m → ℝ) :=
  { toFun := f, map_add' := hadd, map_smul' := hsmul }
example {n m : ℕ} (f : (Fin n → ℝ) →ₗ[ℝ] (Fin m → ℝ)) : Submodule ℝ (Fin n → ℝ) :=
  LinearMap.ker f
--<<nurLoesungEnd>>

/-! ## B4: Induktion und Rechnen (Formalisierung von A4 und A5 (c)) -/

--<<b4>>
/-- Aufgabe B4 (a): die Summe der dritten Potenzen. -/
theorem summe_kuben (n : ℕ) :
    4 * (∑ i ∈ Finset.range (n + 1), i ^ 3) = n ^ 2 * (n + 1) ^ 2 := by
  --<<loesung>>
  induction n with
  | zero => simp
  | succ k ih =>
      rw [Finset.sum_range_succ, Nat.mul_add, ih]
      ring
  --<<loesungEnd>>

/- Aufgabe B4 (b): die Probe fuer `n = 10`. `#eval` rechnet nur aus; ein
   Beweis, den der Kern durch Rechnen prueft, steht darunter. -/
#eval 4 * (∑ i ∈ Finset.range 11, i ^ 3)

example : 4 * (∑ i ∈ Finset.range 11, i ^ 3) = 10 ^ 2 * 11 ^ 2 := by
  --<<loesung>>
  decide
  --<<loesungEnd>>

/-- Aufgabe B4 (c): die Ungleichung aus A4 (b). -/
theorem zweierpotenz (n : ℕ) : n + 1 ≤ 2 ^ n := by
  --<<loesung>>
  induction n with
  | zero => simp
  | succ k ih =>
      rw [pow_succ]
      omega
  --<<loesungEnd>>

/- Aufgabe B4 (d), zu A5 (c). `Nat.add` rekursiert ueber das zweite Argument.
   Versuchen Sie in beiden Faellen zuerst `rfl`. -/
example : (fun x : ℕ => x + 0) = (fun x => x) :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : (fun x : ℕ => 0 + x) = (fun x => x) :=
  --<<loesung>>
  funext Nat.zero_add
  --<<loesungEnd>>

/- Nicht definitorisch gleich: `#check_failure` erwartet, dass das scheitert. -/
#check_failure (rfl : (fun x : ℕ => 0 + x) = (fun x => x))
--<<b4End>>

--<<nurLoesung>>
/- Gegenprobe zu B4 (c): In Mathlib steht die Ungleichung als `n < 2 ^ n`, in `ℕ`
   dasselbe. -/
example (n : ℕ) : n + 1 ≤ 2 ^ n := Nat.lt_two_pow_self
--<<nurLoesungEnd>>

end Uebung03
