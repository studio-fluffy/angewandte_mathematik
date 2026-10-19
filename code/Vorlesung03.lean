/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 3
  (Folien: folien/vorlesung03.tex)
-/
import Mathlib

namespace Skript.Mengenlehre

/-! ## Mengen in Lean (Skript 3.6): `Set α` ist ein Praedikat -/

-- Folie: "Mengen sind Praedikate"
/- In Lean ist `Set α` *definitorisch* dasselbe wie ein Praedikat `α → Prop`.
   Eine Menge ist also immer Teilmenge eines bereits vorhandenen Typs. -/
example (α : Type) : Set α = (α → Prop) := rfl

/- `x ∈ s` bedeutet schlicht: das Praedikat `s` trifft auf `x` zu. -/
example (α : Type) (s : Set α) (x : α) : (x ∈ s) = s x := rfl

/- Komprehension `{x | p x}` ist genau das Praedikat `p`. -/
example (α : Type) (p : α → Prop) (x : α) : (x ∈ {y | p y}) = p x := rfl

section MengenTaktiken

/-! Jede Mengenoperation ist punktweise eine logische Verknuepfung. Deshalb sind
die Taktiken fuer `∧`, `∨`, `¬`, `→`, `∀` und `∃` zugleich die Taktiken fuer
Mengen. Die Zeile `variable` legt fuer alle Beispiele bis `end MengenTaktiken`
fest: `α` und `β` sind Typen, `s t u` Mengen in `α`. -/

-- Folie: "Mengenoperationen sind Logik"
variable {α β : Type} (s t u : Set α)

/- Alle Zeilen gehen mit `Iff.rfl` durch: Beide Seiten sind definitorisch gleich. -/
example (x : α) : x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t := Iff.rfl
example (x : α) : x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t := Iff.rfl
example (x : α) : x ∈ sᶜ ↔ ¬ x ∈ s := Iff.rfl
example (x : α) : x ∈ (∅ : Set α) ↔ False := Iff.rfl
example : s ⊆ t ↔ ∀ x, x ∈ s → x ∈ t := Iff.rfl
example (f : α → β) (y : β) : y ∈ f '' s ↔ ∃ x, x ∈ s ∧ f x = y := Iff.rfl

/-! ## Taktiken am Beispiel der Mengen (Skript 3.6, 3.7) -/

-- Folie: "Teilmengen: Annahmen einfuehren und anwenden"
/- `s ⊆ t` heisst `∀ x, x ∈ s → x ∈ t`: Ein Beweis beginnt mit `intro`. -/
example : s ∩ t ⊆ s := by
  intro x hx               -- hx : x ∈ s ∩ t,  Ziel: x ∈ s
  exact hx.1               -- hx ist ein Paar, hx.1 : x ∈ s
example (h : s ⊆ t) (x : α) (hx : x ∈ s) : x ∈ t := by
  exact h hx               -- h angewandt auf hx; das x ergaenzt Lean selbst
example (h₁ : s ⊆ t) (h₂ : t ⊆ u) : s ⊆ u := by
  intro x hx               -- Ziel: x ∈ u
  apply h₂                 -- Ziel: x ∈ t
  apply h₁                 -- Ziel: x ∈ s
  exact hx

-- Folie: "Schnitt: zerlegen und zusammenbauen"
/- Ein Beweis von `x ∈ s ∩ t` ist ein Paar: `obtain` zerlegt es,
   `constructor` baut eines. -/
example : s ∩ t ⊆ t ∩ s := by
  intro x hx
  obtain ⟨hs, ht⟩ := hx    -- hs : x ∈ s,  ht : x ∈ t
  constructor              -- zwei Ziele: x ∈ t  und  x ∈ s
  · exact ht
  · exact hs
example : s ∩ t ⊆ t ∩ s := by
  rintro x ⟨hs, ht⟩        -- intro und obtain in einem Schritt
  exact ⟨ht, hs⟩           -- das Paar direkt angeben

-- Folie: "Vereinigung: Faelle unterscheiden"
/- `x ∈ s ∪ t` heisst `x ∈ s ∨ x ∈ t`: `rcases` unterscheidet die beiden
   Faelle, `left` und `right` waehlen eine Seite. -/
example : s ∪ t ⊆ t ∪ s := by
  intro x hx
  rcases hx with hs | ht   -- Fall 1: hs : x ∈ s,  Fall 2: ht : x ∈ t
  · right                  -- Ziel: x ∈ s
    exact hs
  · left                   -- Ziel: x ∈ t
    exact ht
example : s ∪ t ⊆ t ∪ s := by
  rintro x (hs | ht)       -- intro und rcases in einem Schritt
  · exact Or.inr hs
  · exact Or.inl ht

-- Folie: "Gleichheit von Mengen: Extensionalitaet"
/- `ext x` fuehrt `s = t` auf `x ∈ s ↔ x ∈ t` zurueck, `constructor`
   zerlegt das `↔` in zwei Richtungen. -/
example : s ∩ t = t ∩ s := by
  ext x                    -- Ziel: x ∈ s ∩ t ↔ x ∈ t ∩ s
  constructor
  · rintro ⟨hs, ht⟩        -- Richtung →
    exact ⟨ht, hs⟩
  · rintro ⟨ht, hs⟩        -- Richtung ←
    exact ⟨hs, ht⟩
/- Oder zwei Inklusionen, zusammengesetzt mit `Set.Subset.antisymm`. -/
example (h₁ : s ⊆ t) (h₂ : t ⊆ s) : s = t := Set.Subset.antisymm h₁ h₂

-- Folie: "Komplement und leere Menge"
/- `x ∈ sᶜ` heisst `x ∈ s → False`. Eine solche Annahme wendet man an,
   ein solches Ziel beginnt mit `intro`. -/
example : s ∩ sᶜ ⊆ ∅ := by
  rintro x ⟨hs, hns⟩       -- hs : x ∈ s,  hns : x ∈ sᶜ,  Ziel: x ∈ ∅
  exact hns hs             -- hns hs : False, und x ∈ ∅ heisst False
example (x : α) (hs : x ∈ s) : x ∉ sᶜ := by
  intro hns                -- hns : x ∈ sᶜ,  Ziel: False
  exact hns hs

-- Folie: "Bilder: Zeugen angeben und auspacken"
/- Das Bild `f '' s` ist `{y | ∃ x, x ∈ s ∧ f x = y}`. -/
example (f : α → β) (x : α) (hx : x ∈ s) : f x ∈ f '' s := by
  use x                    -- Zeuge x; x ∈ s und f x = f x findet use selbst
example (f : α → β) (h : s ⊆ t) : f '' s ⊆ f '' t := by
  rintro y ⟨x, hx, rfl⟩    -- Zeuge x, hx : x ∈ s, und y wird zu f x
  exact ⟨x, h hx, rfl⟩     -- Zeuge x, Nachweis x ∈ t, und f x = f x

-- Folie: "Saetze finden"
/- Mathlib benennt Saetze nach ihrer Aussage. -/
#check @Set.inter_subset_left      -- s ∩ t ⊆ s
#check @Set.subset_union_left      -- s ⊆ s ∪ t
#check @Set.mem_inter_iff          -- x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b
#check @Set.mem_of_mem_of_subset   -- aus x ∈ s und s ⊆ t folgt x ∈ t
#check @Set.Subset.antisymm        -- aus a ⊆ b und b ⊆ a folgt a = b

-- Folie: "Umschreiben mit Gleichungen"
/- `rw [h]` schreibt mit einer Gleichung um: aus einer Annahme oder aus der
   Bibliothek. -/
example (h : s = t) : s ∩ u = t ∩ u := by
  rw [h]
example : s ∩ t ⊆ t := by
  rw [Set.inter_comm]      -- Ziel: t ∩ s ⊆ t
  exact Set.inter_subset_left

-- Folie: "Vereinfachen mit Rechenregeln"
/- `simp` rechnet mit den bekannten Regeln, `simp at h` in einer Annahme. -/
example : (s ∪ t)ᶜ = sᶜ ∩ tᶜ := by
  ext x                    -- Ziel: x ∈ (s ∪ t)ᶜ ↔ x ∈ sᶜ ∩ tᶜ
  simp                     -- macht daraus Logik und erledigt sie
example : s ∩ ∅ = ∅ := by
  simp
example (x : α) (hx : x ∈ s ∩ ∅) : False := by
  simp at hx               -- aus hx wird False, damit ist alles bewiesen

-- Folie: "Ketten von Inklusionen"
/- `calc`: eine Kette von Inklusionen, jeder Schritt mit Begruendung.
   `=` und `⊆` duerfen sich mischen. -/
example (h : s ⊆ t) : s ∩ u ⊆ t ∪ u :=
  calc s ∩ u ⊆ s := Set.inter_subset_left
    _ ⊆ t := h
    _ ⊆ t ∪ u := Set.subset_union_left
example : s ∩ t ⊆ t :=
  calc s ∩ t = t ∩ s := Set.inter_comm s t
    _ ⊆ t := Set.inter_subset_left

-- Folie: "Ziel und Annahme: zwei Seiten einer Aussage"
/- Dieselbe Aussage kann Ziel sein (rechts von ⊢: beweisen) oder Annahme
   (links von ⊢: benutzen). Hier steht `P ∧ Q` bald als Annahme da und
   `Q ∧ P` als Ziel. -/
example (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h             -- Ziel →, Annahme einfuehren:  h : P ∧ Q ⊢ Q ∧ P
  obtain ⟨p, q⟩ := h  -- Annahme ∧ zerlegen:  p : P, q : Q ⊢ Q ∧ P
  constructor         -- Ziel ∧, zwei Ziele:  ⊢ Q  und  ⊢ P
  · exact q           -- das Ziel Q steht als Annahme da
  · exact p           -- ebenso P

/- Dasselbe mit Mengen: `x ∈ s ∩ t` ist `x ∈ s ∧ x ∈ t`. -/
example (x : α) : x ∈ s ∩ t → x ∈ t ∩ s := by
  intro h             -- h : x ∈ s ∩ t ⊢ x ∈ t ∩ s
  obtain ⟨hs, ht⟩ := h
  exact ⟨ht, hs⟩      -- das Paar: erst x ∈ t, dann x ∈ s

-- Folie: "Eine Annahme P → Q benutzen: vorwaerts und rueckwaerts"
/- `h : P → Q` ist eine Funktion. Vorwaerts: Aus `p : P` wird `h p : Q`.
   Rueckwaerts: Ist das Ziel `Q`, macht `apply h` daraus das Ziel `P`. -/
example (P Q R : Prop) : (P → Q) → (Q → R) → P → R := by
  intro f g p         -- f : P → Q, g : Q → R, p : P ⊢ R
  apply g             -- statt R genuegt Q:  ⊢ Q
  apply f             -- statt Q genuegt P:  ⊢ P
  exact p             -- P steht als Annahme da

/- Dasselbe vorwaerts, in einem Schritt: erst `f` auf `p` anwenden, dann `g`. -/
example (P Q R : Prop) : (P → Q) → (Q → R) → P → R :=
  fun f g p => g (f p)

-- Folie: "Welche Taktik wann?"
/-! Die Tabelle in Lean: zu jeder Zeile ein Beispiel. "Ziel" heisst, die Form
steht rechts von `⊢` und wird bewiesen; "Annahme" heisst, sie steht als `h`
links davon und wird benutzt. -/
section Tabelle

variable (P Q : Prop)

/- P → Q -/
example : P → P := by                  -- Ziel: intro
  intro p                              -- p : P,  Ziel: P
  exact p
example (h : P → Q) (p : P) : Q := by  -- Annahme, rueckwaerts: apply
  apply h                              -- Ziel Q wird P
  exact p
example (h : P → Q) (p : P) : Q :=     -- Annahme, vorwaerts: h p : Q
  h p

/- ∀ x, P x -/
example : ∀ n : ℕ, n + 0 = n := by     -- Ziel: intro
  intro n                              -- n einfuehren,  Ziel: n + 0 = n
  rfl
example (h : ∀ n : ℕ, n ≤ n * n) : 3 ≤ 3 * 3 :=   -- Annahme: einsetzen
  h 3

/- s ⊆ t -/
example : s ∩ t ⊆ s := by              -- Ziel: intro
  intro x hx                           -- hx : x ∈ s ∩ t,  Ziel: x ∈ s
  exact hx.1
example (h : s ⊆ t) (x : α) (hx : x ∈ s) : x ∈ t :=   -- Annahme: h hx : x ∈ t
  h hx

/- P ∧ Q,  x ∈ s ∩ t -/
example (p : P) (q : Q) : P ∧ Q := by  -- Ziel: constructor
  constructor                          -- zwei Ziele: P und Q
  · exact p
  · exact q
example (x : α) (hs : x ∈ s) (ht : x ∈ t) : x ∈ s ∩ t :=   -- Ziel: das Paar
  ⟨hs, ht⟩
example (h : P ∧ Q) : Q ∧ P := by      -- Annahme: obtain
  obtain ⟨p, q⟩ := h                   -- p : P,  q : Q
  exact ⟨q, p⟩

/- P ∨ Q,  x ∈ s ∪ t -/
example (x : α) (ht : x ∈ t) : x ∈ s ∪ t := by   -- Ziel: left oder right
  right                                -- Ziel: x ∈ t
  exact ht
example (h : P ∨ Q) : Q ∨ P := by      -- Annahme: rcases
  rcases h with p | q                  -- zwei Faelle
  · right                              -- Fall p : P
    exact p
  · left                               -- Fall q : Q
    exact q

/- ¬P,  x ∈ sᶜ -/
example : ¬(P ∧ ¬P) := by              -- Ziel: intro
  intro h                              -- h : P ∧ ¬P,  Ziel: False
  exact h.2 h.1
example (h : ¬P) (p : P) : Q := by     -- Annahme: h p : False
  have hf : False := h p
  exact hf.elim                        -- aus False folgt alles
example (x : α) (h : x ∈ sᶜ) : x ∈ (s ∩ t)ᶜ := by   -- beides bei Mengen
  intro hst                            -- hst : x ∈ s ∩ t,  Ziel: False
  exact h hst.1                        -- h angewandt auf x ∈ s

/- ∃ x, P x -/
example : ∃ n : ℕ, n * n = 9 := by     -- Ziel: use
  use 3                                -- Zeuge 3; 3 * 3 = 9 rechnet use nach
example (h : ∃ x, x ∈ s ∩ t) : ∃ x, x ∈ s := by   -- Annahme: obtain
  obtain ⟨x, hx⟩ := h                  -- Zeuge x,  hx : x ∈ s ∩ t
  use x
  exact hx.1

/- y ∈ f '' s -/
example (f : α → β) (x : α) (hx : x ∈ s) : f x ∈ f '' s := by   -- Ziel: use
  use x
example (f : α → β) (h : s ⊆ t) (y : β) (hy : y ∈ f '' s) : y ∈ f '' t := by
  obtain ⟨x, hx, rfl⟩ := hy            -- Annahme: Zeuge x, und y wird f x
  exact ⟨x, h hx, rfl⟩

/- a = b,  s = t -/
example : 2 + 2 = 4 := rfl             -- Ziel: Reflexivitaet
example (a : ℕ) : a + 0 = a := by simp -- Ziel: simp
example (a b c : ℕ) (h₁ : a = b) (h₂ : b = c) : a = c :=   -- Ziel: calc
  calc a = b := h₁
    _ = c := h₂
example : s ∩ t = t ∩ s := by          -- Ziel: ext
  ext x                                -- Ziel: x ∈ s ∩ t ↔ x ∈ t ∩ s
  constructor
  · intro h
    exact ⟨h.2, h.1⟩
  · intro h
    exact ⟨h.2, h.1⟩
example (a b : ℕ) (h : a = b) : a + 1 = b + 1 := by   -- Annahme: rw
  rw [h]                               -- a durch b ersetzen
example (a b : ℕ) (h : a = b) : b = a := by           -- Annahme: simp
  simp [h]

end Tabelle

/-! ### Weitere Taktiken zum Nachschlagen

Zwischenschritte, Fallunterscheidung, Widerspruch, Vereinfachen und
Automatik. -/

/-! #### Zwischenschritte: `have`, `specialize`, `refine` -/

/- `have` haelt ein Zwischenergebnis fest und gibt ihm einen Namen. -/
example (h₁ : s ⊆ t) (h₂ : t ⊆ u) (x : α) (hx : x ∈ s) : x ∈ u := by
  have hxt : x ∈ t := h₁ hx  -- neue Annahme hxt : x ∈ t
  exact h₂ hxt

/- `specialize h a` setzt in eine Annahme `h : ∀ x, ...` ein festes `a` ein. -/
example (h : ∀ x, x ∈ s → x ∈ t) (a : α) (ha : a ∈ s) : a ∈ t := by
  specialize h a             -- h : a ∈ s → a ∈ t
  exact h ha

/- `refine` baut einen Beweis mit Luecken `?_`. Jede Luecke wird ein Ziel. -/
example (h : s ⊆ t) : s ∩ u ⊆ t ∩ u := by
  intro x hx
  refine ⟨?_, ?_⟩            -- zwei Ziele: x ∈ t  und  x ∈ u
  · exact h hx.1
  · exact hx.2

/-! #### Fallunterscheidung: `by_cases`, `rcases` mit Mustern -/

/- `by_cases h : P` unterscheidet zwei Faelle: `P` gilt, oder `P` gilt nicht.
   Das benutzt das ausgeschlossene Dritte. -/
example (x : α) : x ∈ s ∪ sᶜ := by
  by_cases h : x ∈ s
  · exact Or.inl h           -- Fall h : x ∈ s
  · exact Or.inr h           -- Fall h : x ∉ s, und das heisst x ∈ sᶜ

/- `rcases` zerlegt auch verschachtelte Annahmen in einem Schritt:
   `⟨_, _⟩` fuer `∧` und `∃`, `_ | _` fuer `∨`. -/
example (x : α) (h : x ∈ s ∩ (t ∪ u)) : x ∈ (s ∩ t) ∪ (s ∩ u) := by
  rcases h with ⟨hs, ht | hu⟩  -- erst das ∧, darin das ∨
  · left
    exact ⟨hs, ht⟩
  · right
    exact ⟨hs, hu⟩

/-! #### Widerspruch: `exfalso`, `contradiction`, `by_contra`, `push_neg` -/

/- `exfalso` macht `False` zum Ziel; aus einem Widerspruch folgt alles. -/
example (x : α) (hs : x ∈ s) (hns : x ∈ sᶜ) : x ∈ t := by
  exfalso                    -- Ziel: False
  exact hns hs

/- `contradiction` findet zwei Annahmen, die sich widersprechen, selbst. -/
example (x : α) (hs : x ∈ s) (hns : x ∉ s) : x ∈ t := by
  contradiction

/- `by_contra h` nimmt das Gegenteil des Ziels als Annahme `h`; zu zeigen
   bleibt `False`. Das ist der klassische Widerspruchsbeweis. Hier ist er
   noetig: Ohne klassische Logik folgt `x ∈ s` nicht aus `x ∈ sᶜᶜ`. -/
example (x : α) (h : x ∈ sᶜᶜ) : x ∈ s := by
  by_contra hns              -- hns : x ∉ s,  Ziel: False
  exact h hns                -- x ∉ s heisst x ∈ sᶜ, und h : x ∈ sᶜ → False

/- `push_neg` schiebt eine Verneinung nach innen: Aus `¬ ∀` wird `∃ ¬`,
   aus `¬ (A → B)` wird `A ∧ ¬ B`. -/
example (h : ¬ ∀ x, x ∈ s → x ∈ t) : ∃ x, x ∈ s ∧ x ∉ t := by
  push_neg at h              -- h : ∃ x, x ∈ s ∧ x ∉ t
  exact h

/-! #### Vereinfachen und Automatik: `simp only`, `tauto`, `aesop` -/

/- `simp only [...]` rechnet nur mit den angegebenen Regeln. So bleibt
   sichtbar, was passiert. -/
example (x : α) : x ∈ s ∩ t ↔ x ∈ t ∩ s := by
  simp only [Set.mem_inter_iff]  -- Ziel: x ∈ s ∧ x ∈ t ↔ x ∈ t ∧ x ∈ s
  exact and_comm

/- `tauto` erledigt jede Aufgabe der Aussagenlogik. Nach `ext` und
   `simp only` ist eine Mengengleichung genau so eine Aufgabe. -/
example : s ∩ (t ∪ u) = (s ∩ t) ∪ (s ∩ u) := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_union]
  tauto

/- `aesop` sucht selbst einen Beweis: Es fuehrt `intro` aus, zerlegt
   Annahmen, vereinfacht und probiert die Annahmen durch. -/
example (h : ∀ x, x ∈ s → x ∈ t) : ∀ x, x ∈ s ∩ u → x ∈ t ∩ u := by
  aesop

end MengenTaktiken

end Skript.Mengenlehre

namespace Skript.Typentheorie

universe u v

/-! ## Zahlen und Rechnen (Skript 3.8) -/

-- Folie: "Am Anfang steht die Arithmetik"
/- Die Zahlbereiche sind gewoehnliche Typen. `ℕ` ist induktiv definiert,
   alles Weitere wird darauf aufgebaut. -/
#check @Nat
#check @Int
#check @Rat
#check @Real

/- Konkrete Zahlen: der Kern rechnet selbst. -/
example : 2 + 2 = 4 := rfl
example : 17 * 23 = 391 := by norm_num
example : (5 : ℤ) - 8 = -3 := by decide
#eval (2 : ℕ) ^ 10

-- Folie: "Aussagen ueber alle Zahlen: Induktion"
/- Gesetze fuer *alle* Zahlen sind kein Rechnen mehr, sondern Induktion.
   Man sieht das an der Asymmetrie: `n + 0` reduziert, `0 + n` nicht. -/
example (n : ℕ) : n + 0 = n := rfl
example (n : ℕ) : 0 + n = n := Nat.zero_add n

/-- Dasselbe von Hand, durch Induktion ueber `n`. Die Addition rekursiert
ueber das zweite Argument: `0 + (k + 1)` ist nach Definition `(0 + k) + 1`. -/
theorem null_plus (n : ℕ) : 0 + n = n := by
  induction n with
  | zero => rfl                              -- Anfang: 0 + 0 = 0, gerechnet
  | succ k ih =>                             -- Schritt, mit ih : 0 + k = k
      calc 0 + (k + 1) = (0 + k) + 1 := rfl  -- Definition der Addition
        _ = k + 1 := by rw [ih]              -- Induktionsannahme

-- Folie: "Drei Taktiken fuer die Praxis"
/- Drei Taktiken decken das meiste ab:
     ring     -- Gleichungen in kommutativen Ringen
     omega    -- lineare Arithmetik ueber ℕ und ℤ, mit Ungleichungen
     norm_num -- konkrete Zahlen, auch in ℚ und ℝ                      -/
example (a b : ℤ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by ring
example (n : ℕ) (h : 3 < n) : 2 < n := by omega
example : (1 : ℝ) / 3 + 1 / 6 = 1 / 2 := by norm_num

-- Folie: "Ein Induktionsbeweis: die Summe der ungeraden Zahlen"
/-- Die Summe der ersten `n` ungeraden Zahlen `1 + 3 + ... + (2n - 1)`,
rekursiv definiert: Bei `n + 1` kommt die naechste ungerade Zahl `2n + 1` dazu. -/
def summeUngerade : ℕ → ℕ
  | 0 => 0
  | n + 1 => summeUngerade n + (2 * n + 1)

#eval summeUngerade 4                               -- 1 + 3 + 5 + 7 = 16

/-- Die Summe der ersten `n` ungeraden Zahlen ist `n ^ 2`. Der Schritt hat
dasselbe Muster wie `null_plus`: erst die Definition, dann die
Induktionsannahme; den Rest rechnet `ring`. -/
theorem ungerade_quadrat (n : ℕ) : summeUngerade n = n ^ 2 := by
  induction n with
  | zero => rfl                                     -- Anfang: 0 = 0 ^ 2
  | succ k ih =>                                    -- ih : summeUngerade k = k ^ 2
      rw [summeUngerade, ih]                        -- Ziel: k ^ 2 + (2 * k + 1) = (k + 1) ^ 2
      ring                                          -- binomische Formel

/-- Dasselbe mit Mathlibs Summenzeichen (so auch auf Blatt 3): An die Stelle
der Definition tritt `Finset.sum_range_succ`, es spaltet den letzten Summanden ab. -/
theorem ungerade_quadrat' (n : ℕ) : ∑ i ∈ Finset.range n, (2 * i + 1) = n ^ 2 := by
  induction n with
  | zero => rfl
  | succ k ih =>
      rw [Finset.sum_range_succ, ih]
      ring

-- Folie: "ℝ rechnet nicht" (im zweiten Teil, bei den Strukturen der Mathematik)
/- `ℝ` ist nicht berechenbar -- `#eval` gibt es dort nicht, und `decide`
   fuehrt zu nichts. Stattdessen `norm_num`. -/
#check (inferInstance : LinearOrder ℝ)

/-! ## Strukturen und Typklassen (Skript 3.2) -/

-- Folie: "Strukturen: Daten buendeln"
/-- Eine Struktur buendelt Daten, hier zwei ganze Zahlen. -/
structure Punkt where
  x : ℤ
  y : ℤ

def punktA : Punkt := ⟨3, 4⟩                  -- anonymer Konstruktor
def punktB : Punkt := { x := 1, y := -2 }     -- mit Feldnamen
#eval punktA.x + punktB.y                     -- Projektionen: 3 + (-2)

/-- Felder duerfen auch Beweise sein: Zahl und Eigenschaft in einem Objekt. -/
structure PositiveZahl where
  wert : ℕ
  pos : 0 < wert

def drei : PositiveZahl := ⟨3, by norm_num⟩

-- Folie: "Eine Typklasse und eine Instanz"
/-- Typklassen sind Strukturen plus automatische Instanzsuche.
Mathlibs gesamte algebraische Hierarchie ist so gebaut. -/
class MeinMonoid (α : Type) where
  op : α → α → α
  e : α
  assoc : ∀ a b c : α, op (op a b) c = op a (op b c)
  e_op : ∀ a : α, op e a = a
  op_e : ∀ a : α, op a e = a

instance : MeinMonoid ℕ where
  op := (· + ·)
  e := 0
  assoc := Nat.add_assoc
  e_op := Nat.zero_add
  op_e := Nat.add_zero

-- Folie: "Instanzsuche"
/-- Ein Satz, der fuer *jedes* Monoid gilt -- Instanzsuche liefert die Struktur. -/
theorem op_e_op {α : Type} [M : MeinMonoid α] (a : α) :
    M.op (M.op a M.e) M.e = a := by
  rw [M.op_e, M.op_e]

/- So findet Lean die echten Mathlib-Strukturen: -/
#check (inferInstance : AddGroup ℤ)
#check (inferInstance : CommRing ℤ)
#check (inferInstance : Field ℝ)
/- Auch `/` gehoert zu einer Typklasse. Welche Instanz gilt, entscheidet der Typ: -/
example : (7 : ℕ) / 2 = 3 := by decide        -- Division mit Rest
example : (7 : ℚ) / 2 = 3.5 := by norm_num    -- Bruchrechnung

-- Folie: "Ein Satz, viele Monoide"
/- Dieselbe Typklasse, weitere Instanzen: Listen mit Anhaengen und
   Funktionen mit Hintereinanderausfuehrung. -/
instance (α : Type) : MeinMonoid (List α) where
  op := (· ++ ·)
  e := []
  assoc := List.append_assoc
  e_op := List.nil_append
  op_e := List.append_nil

instance (α : Type) : MeinMonoid (α → α) where
  op f g := f ∘ g
  e := id
  assoc _ _ _ := rfl
  e_op _ := rfl
  op_e _ := rfl

/- `op_e_op` gilt fuer jede Instanz, ohne neuen Beweis. -/
example (l : List ℕ) : (l ++ []) ++ [] = l := op_e_op l

-- Folie: "Ein Beispiel aus der Informatik: Zusammenfassen in Teilen"
/-- Zusammenfassen (in Programmiersprachen `fold` oder `reduce`): alle
Elemente einer Liste mit `op` verknuepfen; die leere Liste ergibt `e`. Eine
Definition fuer jedes Monoid. -/
def zusammenfassen {α : Type} [M : MeinMonoid α] : List α → α
  | [] => M.e
  | x :: xs => M.op x (zusammenfassen xs)

/-- Man darf die Liste beliebig teilen, die Teile getrennt zusammenfassen und
die Teilergebnisse verknuepfen. Darauf beruht MapReduce: Jeder Rechner fasst
seinen Teil zusammen, am Ende werden die Ergebnisse verknuepft. -/
theorem zusammenfassen_teilen {α : Type} [M : MeinMonoid α] (xs ys : List α) :
    zusammenfassen (xs ++ ys) =
      M.op (zusammenfassen xs) (zusammenfassen ys) := by
  induction xs with
  | nil => simp only [List.nil_append, zusammenfassen, M.e_op]    -- e neutral
  | cons x xs ih =>
      simp only [List.cons_append, zusammenfassen, ih, M.assoc]   -- assoziativ

/- Eine neue Instanz: Zeichenketten mit Aneinanderfuegen. Der Satz gilt
   sofort auch fuer sie. -/
instance : MeinMonoid String where
  op := (· ++ ·)
  e := ""
  assoc _ _ _ := String.append_assoc
  e_op _ := String.empty_append
  op_e _ := String.append_empty

#eval zusammenfassen ["Map", "Reduce"]   -- "MapReduce"
#eval zusammenfassen [3, 5, 2]           -- 10

/- Zwei Rechner, zwei Haelften, dasselbe Ergebnis: -/
example : zusammenfassen [3, 5, 2, 7] =
    zusammenfassen [3, 5] + zusammenfassen [2, 7] :=
  zusammenfassen_teilen [3, 5] [2, 7]

-- Folie: "Erweitern: vom Monoid zur Gruppe"
/-- `extends`: Eine Gruppe ist ein Monoid mit Inversen. Die Felder des Monoids
kommen mit, und jede Gruppe ist von selbst ein Monoid. -/
class MeineGruppe (α : Type) extends MeinMonoid α where
  inv : α → α
  inv_op : ∀ a : α, op (inv a) a = e

instance : MeineGruppe ℤ where
  op := (· + ·)
  e := 0
  assoc := Int.add_assoc
  e_op := Int.zero_add
  op_e := Int.add_zero
  inv := (- ·)
  inv_op a := by omega

/- Saetze ueber Monoide gelten damit fuer jede Gruppe. -/
example (a : ℤ) : (a + 0) + 0 = a := op_e_op a

-- Folie: "Ein einfaches Beispiel aus der Mathematik: Bits mit XOR"
/- Bits mit XOR, also Addition modulo 2: eine Gruppe, in der jedes Bit sein
   eigenes Inverses ist. -/
instance : MeineGruppe Bool where
  op := xor
  e := false
  assoc := Bool.xor_assoc
  e_op := Bool.false_xor
  op_e := Bool.xor_false
  inv := id                       -- jedes Bit ist sein eigenes Inverses
  inv_op := Bool.xor_self

/-- In jeder Gruppe macht `inv b` die Verknuepfung mit `b` rueckgaengig. -/
theorem inv_op_op {α : Type} [G : MeineGruppe α] (a b : α) :
    G.op (G.inv b) (G.op b a) = a := by
  rw [← G.assoc, G.inv_op, G.e_op]       -- assoziativ, invers, neutral

/- Fuer Bits: Verschluesseln mit dem Schluessel k ist x ↦ k xor x, und
   Entschluesseln dasselbe noch einmal (One-Time-Pad). Ein Fall des Satzes,
   ohne neuen Beweis: -/
example (k x : Bool) : xor k (xor k x) = x := inv_op_op x k

/-! ### Noch ein Beispiel fuer `extends`: ein Stapel mit Spezifikation -/

/-- `extends`: Die Schnittstelle `Stapel` legt nur die Operationen fest. Die
Erweiterung fuegt die Gesetze hinzu, also die Spezifikation. Die Felder von
`Stapel` kommen mit, und jeder korrekte Stapel ist von selbst ein Stapel. -/
class Stapel (S : Type) where
  leer : S
  push : ℕ → S → S
  pop : S → S
  top : S → Option ℕ

class KorrekterStapel (S : Type) extends Stapel S where
  pop_push : ∀ x s, pop (push x s) = s
  top_push : ∀ x s, top (push x s) = some x

/-- Eine Implementierung: eine Liste, das oberste Element vorn. -/
instance : KorrekterStapel (List ℕ) where
  leer := []
  push := List.cons
  pop := List.tail
  top := List.head?
  pop_push _ _ := rfl
  top_push _ _ := rfl

/-- Ein Satz fuer *jede* korrekte Implementierung. -/
theorem top_pop_push {S : Type} [K : KorrekterStapel S] (x y : ℕ) (s : S) :
    K.top (K.pop (K.push y (K.push x s))) = some x := by
  rw [K.pop_push, K.top_push]

#eval Stapel.top (Stapel.pop (Stapel.push 2 (Stapel.push 1 ([] : List ℕ))))  -- some 1

/-! ## Die Strukturen der Mathematik in Lean (Skript 3.8) -/

-- Folie: "Die algebraische Hierarchie" (dort auch der Satz fuer alle Gruppen)
/- Die algebraische Hierarchie. Jede Stufe erweitert die vorige,
   aufgebaut mit `class ... extends ...`. -/
#check @Semigroup
#check @Monoid
#check @Group
#check @CommRing
#check @Field

/-- Ein Satz fuer *alle* Gruppen: die Struktur kommt als Instanzargument
`[Group G]` herein, nicht als Datenargument. -/
theorem inverse_von_rechts {G : Type*} [Group G] (a : G) : a * a⁻¹ = 1 :=
  mul_inv_cancel a

/- Damit gilt er sofort fuer jede konkrete Gruppe -- Lean sucht die Instanz. -/
example (a : Equiv.Perm (Fin 3)) : a * a⁻¹ = 1 := inverse_von_rechts a

-- Folie: "Halbring, Ring, Koerper: was die Stufen verlangen"
/- Die Stufen mit zwei Verknuepfungen + und *, zu jeder ein Beispiel. Die
   Instanzsuche findet die Struktur selbst. -/
example : Semiring ℕ := inferInstance                       -- Halbring
example : Ring (Matrix (Fin 2) (Fin 2) ℤ) := inferInstance  -- Ring
example : CommRing ℤ := inferInstance                       -- kommutativer Ring
example : CommRing (ZMod (2^64)) := inferInstance           -- 64-Bit-Zahlen
example : Field ℚ := inferInstance                          -- Koerper
example : Field (ZMod 2) := inferInstance                   -- die Bits

/- Halbring: ℕ hat keine Negativen. Deshalb schneidet Lean die Subtraktion
   bei 0 ab. -/
example : ¬ ∃ n : ℕ, 1 + n = 0 := by
  rintro ⟨n, h⟩
  omega
#eval (3 - 5 : ℕ)                                  -- 0

/- Ring: Bei Matrizen kommt es auf die Reihenfolge an. Eine Matrix schreibt man
   Zeile fuer Zeile: `!![0, 1; 0, 0]` hat die Zeilen (0, 1) und (0, 0). Kommas
   trennen die Eintraege einer Zeile, Semikolons die Zeilen. -/
#eval !![0, 1; 0, 0] * !![0, 0; 1, 0]              -- !![1, 0; 0, 0]
#eval !![0, 0; 1, 0] * !![0, 1; 0, 0]              -- !![0, 0; 0, 1]

/- Eine Matrix ist eine Funktion: `A i j` ist der Eintrag in Zeile `i` und
   Spalte `j`, gezaehlt ab 0. -/
example : (!![0, 1; 0, 0] : Matrix (Fin 2) (Fin 2) ℕ) 0 1 = 1 := rfl

/- Kommutativer Ring: 64-Bit-Zahlen (in Lean `UInt64`) rechnen mit Ueberlauf,
   also modulo 2^64. Kein Koerper: 2 * 2^63 = 0, also hat 2 kein Inverses. -/
#eval (2 : UInt64) * 2^63                          -- 0
example : (2 : ZMod (2^64)) * 2^63 = 0 := by decide

/- Koerper: In F_2 = `ZMod 2` ist die Addition XOR und die Multiplikation AND. -/
example : ∀ a b : ZMod 2, (a + b = 1 ↔ a ≠ b) := by decide          -- XOR
example : ∀ a b : ZMod 2, (a * b = 1 ↔ a = 1 ∧ b = 1) := by decide  -- AND

/-! ### Lineare Algebra, selbst gebaut

Wie bei den Monoiden: erst die Begriffe so, wie man sie aus der Linearen
Algebra kennt, dann der Abgleich mit Mathlib. -/

namespace LineareAlgebra

section Selbstgebaut

-- Folie: "Vektorraeume, wie man sie kennt"
/-- Ein Vektorraum ueber dem Koerper `K`, wie in der Linearen Algebra: `(V, +)`
ist eine abelsche Gruppe, `a • v` eine Skalarmultiplikation, und es gelten vier
Gesetze. Die Klasse enthaelt nur die Gesetze; `+` und `•` kommen aus
`[AddCommGroup V]` und `[SMul K V]`. -/
class Vektorraum (K V : Type*) [Field K] [AddCommGroup V] [SMul K V] : Prop where
  smul_add : ∀ (a : K) (u v : V), a • (u + v) = a • u + a • v
  add_smul : ∀ (a b : K) (v : V), (a + b) • v = a • v + b • v
  mul_smul : ∀ (a b : K) (v : V), (a * b) • v = a • (b • v)
  one_smul : ∀ v : V, (1 : K) • v = v

/- "Sei K ein Koerper und V ein K-Vektorraum": -/
variable {K V : Type*} [Field K] [AddCommGroup V] [SMul K V] [Vektorraum K V]

/-- Das Standardbeispiel `Kⁿ`. Ein Vektor ist eine Funktion `v : Fin n → K`;
`v i` ist seine `i`-te Komponente, `+` und `•` rechnen komponentenweise. -/
instance (n : ℕ) : Vektorraum K (Fin n → K) where
  smul_add a u v := by funext i; simp [mul_add]
  add_smul a b v := by funext i; simp [add_mul]
  mul_smul a b v := by funext i; simp [mul_assoc]
  one_smul v := by funext i; simp

/-- Die erste Rechnung der Linearen Algebra: `0 • v = 0`. Denn
`0 • v + 0 • v = (0 + 0) • v = 0 • v`; Kuerzen laesst `0 • v = 0`. -/
theorem null_smul (v : V) : (0 : K) • v = 0 := by
  have h : (0 : K) • v + (0 : K) • v = (0 : K) • v + 0 := by
    rw [← Vektorraum.add_smul, add_zero, add_zero]
  exact add_left_cancel h

/-- Ebenso `a • 0 = 0`, mit dem anderen Distributivgesetz. -/
theorem smul_null (a : K) : a • (0 : V) = 0 := by
  have h : a • (0 : V) + a • (0 : V) = a • (0 : V) + 0 := by
    rw [← Vektorraum.smul_add, add_zero, add_zero]
  exact add_left_cancel h

-- Folie: "Lineare Abbildungen und Matrizen"
/- "Sei W ein weiterer K-Vektorraum": -/
variable {W : Type*} [AddCommGroup W] [SMul K W] [Vektorraum K W]

variable (K) in
/-- `f : V → W` heisst linear, wenn es mit `+` und mit `•` vertraeglich ist. -/
structure IstLinear (f : V → W) : Prop where
  additiv : ∀ u v : V, f (u + v) = f u + f v
  homogen : ∀ (a : K) (v : V), f (a • v) = a • f v

/-- Lineare Abbildungen bilden `0` auf `0` ab, mit `0 • v = 0` von oben. -/
theorem IstLinear.null {f : V → W} (hf : IstLinear K f) : f 0 = 0 :=
  calc f 0 = f ((0 : K) • (0 : V)) := by rw [null_smul]
    _ = (0 : K) • f 0 := hf.homogen 0 0
    _ = 0 := null_smul (f 0)

/-- Matrix mal Vektor, wie in der Linearen Algebra: `(A v) i = ∑ j, A i j * v j`. -/
def matVek {m n : ℕ} (A : Matrix (Fin m) (Fin n) K) (v : Fin n → K) : Fin m → K :=
  fun i => ∑ j, A i j * v j

/-- Matrix mal Vektor ist linear. -/
theorem matVek_linear {m n : ℕ} (A : Matrix (Fin m) (Fin n) K) :
    IstLinear K (matVek A) where
  additiv u v := by funext i; simp [matVek, mul_add, Finset.sum_add_distrib]
  homogen a v := by funext i; simp [matVek, Finset.mul_sum, mul_left_comm]

/-- Beispiel: die Drehung um 90 Grad in der Ebene. -/
def drehung90 : Matrix (Fin 2) (Fin 2) ℚ := !![0, -1; 1, 0]

/- Mit einem `!` statt zwei schreibt man Vektoren: `![1, 0]` ist der Vektor
   (1, 0), also eine Funktion `Fin 2 → ℚ`. -/
example : matVek drehung90 ![1, 0] = ![0, 1] := by    -- (1, 0) ↦ (0, 1)
  ext i; fin_cases i <;> simp [matVek, drehung90]
example : matVek drehung90 ![0, 1] = ![-1, 0] := by   -- (0, 1) ↦ (-1, 0)
  ext i; fin_cases i <;> simp [matVek, drehung90]

/-- Der Kern von `f`: alle Vektoren, die `f` auf `0` abbildet. Eine Menge,
also ein Praedikat -- wie im ersten Teil. -/
def Kern (f : V → W) : Set V := {v | f v = 0}

variable (K) in
/-- `U` ist ein Untervektorraum: Er enthaelt `0` und ist abgeschlossen unter
`+` und `•`. -/
structure IstUnterraum (U : Set V) : Prop where
  enthaelt_null : (0 : V) ∈ U
  abgeschlossen_add : ∀ u ∈ U, ∀ v ∈ U, u + v ∈ U
  abgeschlossen_smul : ∀ (a : K), ∀ v ∈ U, a • v ∈ U

/-- Der Kern einer linearen Abbildung ist ein Untervektorraum. -/
theorem kern_unterraum {f : V → W} (hf : IstLinear K f) :
    IstUnterraum K (Kern f) where
  enthaelt_null := hf.null
  abgeschlossen_add u (hu : f u = 0) v (hv : f v = 0) := by
    change f (u + v) = 0
    rw [hf.additiv, hu, hv, add_zero]
  abgeschlossen_smul a v (hv : f v = 0) := by
    change f (a • v) = 0
    rw [hf.homogen, hv, smul_null]

-- Folie: "Basis, Dimension und Mathlib"
variable (K) in
/-- Die Vektoren `v 0, ..., v (n-1)` sind linear unabhaengig: Nur die triviale
Linearkombination ergibt `0`. -/
def LinearUnabhaengig {n : ℕ} (v : Fin n → V) : Prop :=
  ∀ c : Fin n → K, ∑ i, c i • v i = 0 → ∀ i, c i = 0

/-- Der Einheitsvektor `e i` in `Kⁿ`: eine `1` an der Stelle `i`, sonst `0`. -/
def e {n : ℕ} (i : Fin n) : Fin n → K := fun j => if j = i then 1 else 0

/-- Die Einheitsvektoren sind linear unabhaengig: Die Komponente `j` von
`∑ i, c i • e i` ist `c j`. Ist die Summe `0`, so ist also jedes `c j = 0`. -/
theorem e_unabhaengig (n : ℕ) :
    LinearUnabhaengig K (fun i : Fin n => (e i : Fin n → K)) := by
  intro c h j
  have hj := congrFun h j                    -- Komponente j der Gleichung
  simpa [e, Finset.sum_apply] using hj

end Selbstgebaut

end LineareAlgebra

-- Folie: "Basis, Dimension und Mathlib"
/- Lineare Algebra: ein Vektorraum ist ein Modul ueber einem Koerper.
   Einen eigenen Typ `VectorSpace` gibt es nicht -- er waere ueberfluessig. -/
#check @Module
#check @Module.Basis
#check @Module.finrank
#check @LinearMap

example {K V : Type*} [Field K] [AddCommGroup V] [Module K V] (a : K) (v : V) : V :=
  a • v

namespace LineareAlgebra

/-- Jedes Modul ueber einem Koerper ist ein Vektorraum in unserem Sinn: Die
vier Gesetze sind Saetze aus Mathlib. -/
instance modul_ist_vektorraum {K V : Type*} [Field K] [AddCommGroup V] [Module K V] :
    Vektorraum K V :=
  ⟨smul_add, add_smul, mul_smul, one_smul K⟩

/-- Umgekehrt ist jeder Vektorraum in unserem Sinn ein Modul. Mathlib verlangt
dafuer noch `0 • v = 0` und `a • 0 = 0` -- die zwei Rechnungen von oben. -/
def vektorraum_ist_modul {K V : Type*} [Field K] [AddCommGroup V] [SMul K V]
    [Vektorraum K V] : Module K V where
  one_smul := Vektorraum.one_smul
  mul_smul := Vektorraum.mul_smul
  smul_zero := smul_null
  smul_add := Vektorraum.smul_add
  add_smul := Vektorraum.add_smul
  zero_smul := null_smul

section Abgleich

/- "Seien V und W Vektorraeume ueber dem Koerper K" -- so schreibt es Mathlib: -/
variable {K V W : Type*} [Field K] [AddCommGroup V] [Module K V]
  [AddCommGroup W] [Module K W]

/- Mathlibs lineare Abbildungen `V →ₗ[K] W` sind genau unsere: die Abbildung
   samt den zwei Eigenschaften, gebuendelt. -/
example (g : V →ₗ[K] W) : IstLinear K g := ⟨map_add g, map_smul g⟩

example (f : V → W) (hf : IstLinear K f) : V →ₗ[K] W where
  toFun := f
  map_add' := hf.additiv
  map_smul' := hf.homogen

/- Kern, Matrix mal Vektor, lineare Unabhaengigkeit, Standardbasis und
   Dimension heissen in Mathlib so: -/
example (g : V →ₗ[K] W) : Kern g = (LinearMap.ker g : Set V) := by
  ext v
  simp [Kern]

example {m n : ℕ} (A : Matrix (Fin m) (Fin n) K) (v : Fin n → K) :
    matVek A v = A.mulVec v := rfl

example {n : ℕ} (v : Fin n → V) : LinearUnabhaengig K v ↔ LinearIndependent K v :=
  Fintype.linearIndependent_iff.symm

example {n : ℕ} (i : Fin n) : Pi.basisFun K (Fin n) i = e i := by
  ext j
  simp [e, Pi.single_apply]

example (n : ℕ) : Module.finrank K (Fin n → K) = n := Module.finrank_fin_fun K

end Abgleich

end LineareAlgebra

-- Folie: "Ordnungen"
/- Ordnungen sind ebenso gestuft. -/
#check @Preorder
#check @PartialOrder
#check @LinearOrder
#check @Lattice
#check @CompleteLattice

/- Die Teilmengen eines Typs bilden einen vollstaendigen Verband. -/
example : CompleteLattice (Set ℕ) := inferInstance

-- Folie: "Mixins: kombinieren statt vererben"
/- **Mixins**: Struktur wird kombiniert, nicht vererbt. Ein geordnetes
   Monoid ist ein Monoid *und* eine Ordnung *und* die Vertraeglichkeit
   beider: aus a <= b folgt c*a <= c*b. Das sind drei getrennte
   Instanzargumente. `IsOrderedMonoid` traegt selbst keine Daten, es ist
   eine `Prop`. -/
#check @IsOrderedMonoid

example {M : Type*} [CommMonoid M] [PartialOrder M] [IsOrderedMonoid M]
    (a b c : M) (h : a ≤ b) : c * a ≤ c * b := by gcongr

-- Folie: "Unterobjekte und Morphismen sind gebuendelt"
/- Unterobjekte sind **gebuendelt**: `Subgroup G` ist ein eigener Typ und
   kein Praedikat auf `Set G`. Traegermenge und Abgeschlossenheitsbeweise
   stecken zusammen in einem Objekt. -/
#check @Subgroup
#check @Submodule

example {G : Type*} [Group G] (H : Subgroup G) (a : G) : Prop := a ∈ H

/- Morphismen ebenso: eigene Typen mit Koerzierung zur Funktion.
   `MonoidHom` schreibt man `G →* H`, dazu gibt es `LinearMap` und
   `ContinuousLinearMap`. -/
#check @MonoidHom
#check @LinearMap
#check @ContinuousLinearMap

example {G H : Type*} [Group G] [Group H] (f : G →* H) (a b : G) :
    f (a * b) = f a * f b := map_mul f a b

end Skript.Typentheorie
