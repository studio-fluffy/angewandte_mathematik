/-
Übungsblatt 2 -- Typentheorie: Terme, Typen, Regeln
Ersetzen Sie jedes `sorry` durch einen Term: einen Beweis oder, bei `def`, eine
Definition. Die Datei ist fertig, wenn Lean keine Warnung
`declaration uses 'sorry'` mehr meldet.

Alles geht im Termmodus: Sie schreiben den Term direkt hin, mit `fun`,
Anwendung, `⟨_, _⟩` und den Namen aus den Hinweisen. Den Taktikmodus (`by ...`)
aus Vorlesung 2 brauchen Sie hier nicht; einzige Ausnahme ist `by decide` in B1.

Sonderzeichen tippt man mit einem Backslash: `\<` und `\>` ergeben `⟨` und `⟩`,
`\2` ergibt den Index in `congrArg₂`.
-/
--<<nurLoesung>>
-- Loesungsfassung. Die Aufgabenfassung lean/Blatt02.lean erzeugt erzeuge.pl daraus.
--<<nurLoesungEnd>>
import Mathlib

namespace Uebung02

/-! ## Aufgabe B1: Formalisierung von A2 -/

--<<b1>>
/-- Das Urteil aus A2 (a). Lean prueft genau
`A : Prop, B : Prop, C : Prop ⊢ t : (A → B) → (B → C) → A → C`. -/
theorem kette (A B C : Prop) : (A → B) → (B → C) → A → C :=
  --<<loesung>>
  fun f g x => g (f x)
  --<<loesungEnd>>

/-- Derselbe Typ fuer Daten: erst `f` anwenden, dann `g`. -/
def nacheinander (α β γ : Type) : (α → β) → (β → γ) → α → γ :=
  --<<loesung>>
  fun f g x => g (f x)
  --<<loesungEnd>>

/- Die Terme aus A2 (c), fuer jedes Universum `Sort u`. -/
universe u

def null (α : Sort u) : (α → α) → α → α := fun _ x => x

def eins (α : Sort u) : (α → α) → α → α :=
  --<<loesung>>
  fun f x => f x
  --<<loesungEnd>>

def zwei (α : Sort u) : (α → α) → α → α :=
  --<<loesung>>
  fun f x => f (f x)
  --<<loesungEnd>>

/- Als Programme, mit α := ℕ: `eins` und `zwei` zaehlen bis 1 bzw. 2 ... -/
example : eins ℕ Nat.succ 0 = 1 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : zwei ℕ Nat.succ 0 = 2 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

/- ... und `null ℕ` ist ein anderes Programm als `zwei ℕ`.
   `congrFun h a` macht aus `h : f = g` die Gleichung `f a = g a`. -/
example : null ℕ ≠ zwei ℕ :=
  --<<loesung>>
  fun h => absurd (congrFun (congrFun h Nat.succ) 0) (by decide)
  --<<loesungEnd>>

/- Als Beweise, mit einer Aussage A fuer α: -/
example (A : Prop) : null A = zwei A :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>
--<<b1End>>

/-! ## Aufgabe B2: Formalisierung von A3 -/

--<<b2>>
/-- (a) Eine Implikation, angewandt auf die linke Komponente -/
theorem und_imp_links (A B C : Prop) : (A → B) → A ∧ C → B ∧ C :=
  --<<loesung>>
  fun f h => ⟨f h.1, h.2⟩
  --<<loesungEnd>>

/-- (b) De Morgan, in beiden Richtungen konstruktiv -/
theorem de_morgan (A B : Prop) : ¬(A ∨ B) ↔ ¬A ∧ ¬B :=
  --<<loesung>>
  ⟨fun h => ⟨fun a => h (Or.inl a), fun b => h (Or.inr b)⟩,
   fun k h => h.elim k.1 k.2⟩
  --<<loesungEnd>>

/-- (c) Dreifache Negation -/
theorem dreifach (A : Prop) : ¬¬¬A → ¬A :=
  --<<loesung>>
  fun h a => h (fun na => na a)
  --<<loesungEnd>>

/-- (d) Die andere De-Morgan-Regel, mit `Classical.em A : A ∨ ¬A` -/
theorem klassisch (A B : Prop) : ¬(A ∧ B) → ¬A ∨ ¬B :=
  --<<loesung>>
  fun h => (Classical.em A).elim (fun a => Or.inr (fun b => h ⟨a, b⟩)) Or.inl
  --<<loesungEnd>>

/- Worauf beruhen die beiden letzten Saetze? Ueber die Typentheorie hinaus
   braucht Mathlib nur `propext`, `Classical.choice` und `Quot.sound`. -/
#print axioms dreifach
#print axioms klassisch
--<<b2End>>

/-! ## Aufgabe B3: Formalisierung von A4 -/

--<<b3>>
def zweimal : (ℕ → ℕ) → ℕ → ℕ := fun f x => f (f x)
def doppelt : ℕ → ℕ := fun n => n + n

/- Die Rechnungen aus A4 (a) und (b) -/
example : zweimal doppelt 3 = 12 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : (let d := doppelt; d (d 1)) = 4 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : Nat.rec (motive := fun _ => ℕ) 1 (fun _ ih => 2 * ih) 3 = 8 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

/- Die Gleichungen aus A4 (c): `rfl`, wo sie definitorisch gelten, sonst
   einer der Saetze `Nat.add_zero`, `Nat.zero_add`, `Nat.one_add`,
   `Nat.mul_zero`, `Nat.zero_mul`, `Nat.one_mul` (nicht jeder passt). -/
example (n : ℕ) : n + 1 = Nat.succ n :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example (n : ℕ) : 1 + n = Nat.succ n :=
  --<<loesung>>
  Nat.one_add n
  --<<loesungEnd>>

example (n : ℕ) : 0 * n = 0 :=
  --<<loesung>>
  Nat.zero_mul n
  --<<loesungEnd>>

example (n : ℕ) : n * 1 = n :=
  --<<loesung>>
  Nat.zero_add n    -- n * 1 rechnet zu 0 + n (Konversionsregel)
  -- ebenso ginge Nat.mul_one n
  --<<loesungEnd>>

example (n : ℕ) : n * 1 = 0 + n :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>
--<<b3End>>

/-! ## Aufgabe B4: Formalisierung von A5 -/

--<<b4>>
inductive Baum where
  | blatt : Baum
  | knoten : Baum → Baum → Baum

namespace Baum

/-- Anzahl der Blaetter: ein Fall pro Konstruktor, wie bei `MeinNat.add`. -/
def blaetter : Baum → ℕ :=
  fun
  | .blatt => 1
  | .knoten l r => blaetter l + blaetter r

/-- (a) Anzahl der inneren Knoten -/
def innen : Baum → ℕ :=
  --<<loesung>>
  fun
  | .blatt => 0
  | .knoten l r => innen l + innen r + 1
  --<<loesungEnd>>

/-- (a) Spiegeln: linken und rechten Teilbaum tauschen, auf allen Ebenen -/
def spiegel : Baum → Baum :=
  --<<loesung>>
  fun
  | .blatt => .blatt
  | .knoten l r => .knoten (spiegel r) (spiegel l)
  --<<loesungEnd>>

/-- Der Baum `t` aus A5 (a) -/
def beispiel : Baum :=
  .knoten (.knoten .blatt .blatt) (.knoten .blatt (.knoten .blatt .blatt))

/- (b) Zuerst die Werte aus A5 (a) -- das prueft auch Ihre Definitionen. -/
example : innen beispiel = 4 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : spiegel beispiel =
    .knoten (.knoten (.knoten .blatt .blatt) .blatt) (.knoten .blatt .blatt) :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

/- Dann A5 (c) und (d) am Beispiel -- der Kern rechnet beide Seiten aus. -/
example : blaetter beispiel = innen beispiel + 1 :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

example : spiegel (spiegel beispiel) = beispiel :=
  --<<loesung>>
  rfl
  --<<loesungEnd>>

#check @Baum.rec    -- zum Vergleich mit A5 (b)
--<<b4End>>

--<<b4c>>
/-- (c) A5 (d) fuer alle Baeume. Im Fall `.knoten l r` sind
`spiegel_spiegel l` und `spiegel_spiegel r` die Induktionsvoraussetzungen. -/
theorem spiegel_spiegel : (t : Baum) → spiegel (spiegel t) = t
  | .blatt =>
    --<<loesung>>
    rfl
    --<<loesungEnd>>
  | .knoten l r =>
    --<<loesung>>
    congrArg₂ knoten (spiegel_spiegel l) (spiegel_spiegel r)
    --<<loesungEnd>>
end Baum
--<<b4cEnd>>

end Uebung02
