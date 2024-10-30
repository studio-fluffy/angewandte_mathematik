import Mathlib.Topology.Instances.Real
import Mathlib.Data.Set.Basic



-- Lean-Code: Eine Funktion, die eine natürliche Zahl in eine reelle Zahl umwandelt
example : ℕ → ℝ :=
fun n => (n : ℝ) + 0.5  -- Gebe das Ergebnis `n + 0.5` zurück, was ein reeller Wert ist


-- Beispiel für `refl` mit natürlichen Zahlen
example : 5 = 5 := by
  rfl  -- Beweise, dass 5 gleich 5 ist

example : 2 + 3 = 5 := by
  rfl  -- Beweise, dass 5 gleich 5 ist


-- Beispiel für `exact` mit natürlichen Zahlen
example (a b : ℕ) (h : a = b) : b = a := by
  exact h.symm  -- Verwende die Symmetrie der Gleichheit


-- Beispiel für `simp` mit natürlichen Zahlen
example : 2 + 2 = 4 := by
  simp  -- Vereinfache den Ausdruck

-- Beispiel für `apply` mit natürlichen Zahlen
example (a b c : ℕ) (h : a = b) (h2 : b = c) : a = c := by
  apply Eq.trans h  -- Wende die Transitivität der Gleichheit an
  exact h2          -- Beweise, dass `b = c`


-- Beispiel für `cases` mit natürlichen Zahlen
example (n : ℕ) : n = 0 ∨ n ≠ 0 := by
  cases n with
  | zero => left; rfl  -- Fall `n = 0`
  | succ k => right; intro h; contradiction  -- Fall `n = succ k` (n ist der Nachfolger von k)



-- Beispiel für `intro` mit natürlichen Zahlen
example (a b : ℕ) : a = b → b = a := by
  intro h  -- Führe die Annahme `h : a = b` ein
  exact h.symm  -- Verwende die Symmetrie der Gleichheit

-- Beispiel für `rw` mit natürlichen Zahlen
example (a b : ℕ) (h : a = b) : a + 1 = b + 1 := by
  rw [h]  -- Ersetze `a` durch `b` in der Gleichung


open Set -- Öffne den `Set`-Namespace für eine einfachere Syntax




variable {α : Type*}  -- Definiere eine Typvariable `α`



-- Intro
example {α : Type*} (A B : Set α) : A ∩ B ⊆ A := by
  intro x  -- Führe `x` ein
  intro hx -- Führe die Annahme `hx : x ∈ A ∩ B` ein
  exact hx.1 -- Zeige, dass `x ∈ A` gilt


-- Exact, rewrite, ext
example {α : Type*} (A B : Set α) : A ∩ B = B ∩ A := by
  ext x  -- Verwende Extensionalität
  rw [mem_inter_iff, mem_inter_iff]  -- Wende die Definition der Schnittmenge an
  exact and_comm  -- Verwende die Kommutativität der Konjunktion

def B : Set ℕ := { x | x < 5 }

-- simp
example : 3 ∈ B := by
  unfold B  -- Wandle `A` in seine Definition um
  simp      -- Zeige, dass 3 < 5 gilt

-- apply
example {α : Type*} (A B : Set α) (h : A ⊆ B) : A ∩ A ⊆ B := by
  intro x
  intro hx
  apply h  -- Wende die Hypothese `h` an
  exact hx.1

--cases
example {α : Type*} (A B : Set α) (x : α) (h : x ∈ A ∪ B) : x ∈ A ∨ x ∈ B := by
  cases h with
  | inl ha => left; exact ha  -- Fall `x ∈ A`
  | inr hb => right; exact hb -- Fall `x ∈ B`

example (x y z : ℝ) (h₀ : x ≤ y) (h₁ : y ≤ z) : x ≤ z := by
  apply le_trans
  apply h₀
  apply h₁
