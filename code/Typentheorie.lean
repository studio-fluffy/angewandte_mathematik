/-
  Angewandte Mathematik -- Begleitcode zum Skript
  Kapitel 3: Typentheorie

  Zuordnung zu den Vorlesungen. Jeder Teil beginnt mit einer Kopfzeile
  "VORLESUNG n" zwischen zwei Linien aus Gleichheitszeichen.
    Vorlesung 2  Teil 0 (Tutorial; 0.5 und 0.6 zum Nachlesen), Skript 3.1-3.5
                 und 3.7
                 (Folien: folien/vorlesung02.tex)
    Vorlesung 3  Strukturen und Typklassen (Skript 3.2, Ende), Skript 3.8
                 (Folien: folien/vorlesung03.tex); die Mengen und die
                 Taktiken am Beispiel der Mengen stehen in Mengenlehre.lean

  Teil 0 ist ein kurzes Tutorial: wie man Lean liest und schreibt. Danach
  folgen die Abschnitte 3.1-3.5, 3.7 und 3.8 des Skripts (3.6 steht in
  Mengenlehre.lean), in der Reihenfolge der Vorlesungen: 3.1, 3.2, 3.4, die
  Regeln aus 3.5, dann 3.3 (Curry-Howard braucht die Regeln) und danach
  Diaconescus Beweis aus 3.4 (er braucht ∨ und ∃ als Typen und die
  Beweisirrelevanz), der Rest von 3.5 und 3.7; die Strukturen und
  Typklassen vom Ende von 3.2 stehen erst vor 3.8. Jeder Baustein wird eingefuehrt, bevor er benutzt wird -- auch
  die Gleichheit: `=` und `rfl` erklaert Teil 0.5, die zwei
  Gleichheitsbegriffe folgen nach `MeinNat` in 3.2, Gleichheit in Lean und
  die Axiome in 3.4 direkt danach.

  Ergaenzt LeanMath/basics/lean_typetheory_curry_howard_intro.lean:
  dort die ausfuehrliche Taktik-Einfuehrung, hier die Typentheorie
  als *Grundlage* (Alternative zu ZFC).
-/
import Mathlib

namespace Skript.Typentheorie

-- =============================================================================
-- VORLESUNG 2: Lean und Typentheorie: Terme, Typen, Regeln
--   Teil 0 (Tutorial; 0.5 und 0.6 zum Nachlesen), Skript 3.1-3.5 und 3.7
-- =============================================================================

/-! ## Teil 0: Lean lesen und schreiben -- ein kurzes Tutorial

Dieser Teil setzt nichts voraus. Er zeigt die Bausteine, die spaeter ohne
weitere Erklaerung vorkommen. Setzen Sie den Cursor auf eine Zeile: Das
Infofenster (Infoview) zeigt, was Lean dazu meldet -- einen Typ, einen Wert
oder das offene Beweisziel. -/

/-! ### 0.1 Kommentare und die drei Fragebefehle

`--` beginnt einen Kommentar bis zum Zeilenende, `/- ... -/` einen Kommentar
ueber mehrere Zeilen. `/-- ... -/` vor einer Definition ist ein Docstring; er
erscheint, wenn die Maus ueber dem Namen steht.

`#check` zeigt den Typ eines Ausdrucks, `#eval` rechnet ihn aus, und `#print`
zeigt, wie ein Name definiert ist. Keiner der drei Befehle definiert etwas.
In diesem Projekt zeigt das Infofenster Zahlen mit ihrem Typ, etwa `(5 : ℕ)`
statt `5` (Option `pp.numericTypes` in lakefile.toml). -/

--<<leanBefehle>>
#check 2 + 3            -- ein Ausdruck vom Typ ℕ
#eval 2 + 3             -- 5
#check (2 : ℤ) - 5      -- Typangabe: dieselben Ziffern als ganze Zahlen
#eval (2 : ℤ) - 5       -- -3; in ℕ ergaebe 2 - 5 die 0
#print Nat              -- ℕ ist ein induktiver Typ mit zero und succ
--<<leanBefehleEnd>>

/-! ### 0.2 Definitionen und Funktionen

`def Name (Argumente) : Ergebnistyp := Rumpf` fuehrt einen neuen Namen ein.
Funktionen wendet man ohne Klammern an: `verdopple 21`, nicht `verdopple(21)`.
`#check (verdopple)` zeigt den Typ `ℕ → ℕ`; ohne Klammern zeigt `#check` bei
einem Namen die Signatur, `verdopple (n : ℕ) : ℕ`, mit vollem Namen.
Eine Funktion ohne Namen schreibt man mit `fun`: `fun x => x + 1` ist die
Abbildung x ↦ x + 1, im Lambda-Kalkuel (Skript 2.4) λx. x + 1; das
Infofenster schreibt den Pfeil als ↦.

Eine Funktion zweier Argumente hat den Typ `ℕ → ℕ → ℕ`, gelesen
`ℕ → (ℕ → ℕ)`: Gibt man nur das erste Argument, bleibt eine Funktion uebrig,
die auf das zweite wartet. -/

--<<leanFunktionen>>
def verdopple (n : ℕ) : ℕ := 2 * n

#check (verdopple)                -- verdopple : ℕ → ℕ
#eval verdopple 21                -- 42

#check fun x : ℕ => x + 1         -- eine Funktion ohne Namen
#eval (fun x : ℕ => x + 1) 41     -- 42

def addiere (m n : ℕ) : ℕ := m + n

#check (addiere)                  -- addiere : ℕ → ℕ → ℕ
#check addiere 2                  -- ℕ → ℕ: wartet auf das zweite Argument
#eval addiere 2 40                -- 42
--<<leanFunktionenEnd>>

/- Fallunterscheidung mit `if`, Zwischenergebnisse mit `let`. Paare: `α × β`
   ist der Typ der Paare, `p.1` und `p.2` sind die Komponenten. Argumente in
   geschweiften Klammern sind *implizit*: Lean ergaenzt sie aus den uebrigen
   Argumenten. Mit `@` gibt man sie selbst an. `Type` ist der Typ der
   Datentypen wie ℕ oder String (mehr in 3.1). -/

--<<leanMehrFunktionen>>
def betrag (z : ℤ) : ℤ := if z < 0 then -z else z

def quadratsumme (a b : ℕ) : ℕ :=
  let qa := a * a
  let qb := b * b
  qa + qb

#eval betrag (-7)                 -- 7
#eval quadratsumme 3 4            -- 25

def tausche {α β : Type} (p : α × β) : β × α := (p.2, p.1)

#eval tausche (1, "eins")         -- ("eins", 1): Lean findet α und β selbst
#check @tausche ℕ String          -- α und β ausdruecklich angegeben
--<<leanMehrFunktionenEnd>>

/-! ### 0.3 Aussagen und Beweise

Eine Aussage ist ein Ausdruck vom Typ `Prop`. Ein Beweis einer Aussage `P` ist
ein Term vom Typ `P`; Abschnitt 3.3 erklaert, warum das genau zur Logik passt.
`theorem` gibt einem Beweis einen Namen, `example` prueft ihn nur. `True` ist
die Aussage, die immer gilt, `trivial` ihr Beweis. Aus Beweisen von `p` und
`q` wird ein Beweis von `p ∧ q`: das Paar ⟨hp, hq⟩. Eine Implikation `p → q`
beweist eine Funktion, die aus einem Beweis von `p` einen von `q` macht; ein
Unterstrich `_` steht fuer ein Argument, das nicht gebraucht wird. In Beweisen
benennt `have` ein Zwischenergebnis, so wie `let` bei Daten. -/

--<<leanBeweise>>
#check 2 < 3                      -- eine Aussage, vom Typ Prop
#check ∀ n : ℕ, n < n + 1         -- auch mit Quantor

theorem stets_wahr : True := True.intro
theorem stets_wahr_trivial : True := trivial

example (p q : Prop) (hp : p) (hq : q) : p ∧ q := ⟨hp, hq⟩
example (p q : Prop) (hq : q) : p → q := fun _ => hq
--<<leanBeweiseEnd>>

/-! ### 0.4 Beweise mit Taktiken

Nach `by` baut man den Beweis Schritt fuer Schritt. Das Infofenster zeigt nach
jeder Zeile das offene Ziel hinter `⊢` und darueber die Annahmen. Abschnitt
3.7 vergleicht beide Wege. Die ersten Taktiken:

  intro x           nimmt eine Annahme in den Kontext (Ziel p → q oder ∀ x, ...)
  intros            nimmt alle Annahmen auf einmal
  exact t           schliesst das Ziel mit dem Term t
  apply h           fuehrt mit h : p → q das Ziel q auf das Ziel p zurueck
  constructor       zerlegt das Ziel p ∧ q in zwei Ziele
  induction         Induktion (Abschnitt 3.2)
  cases             Fallunterscheidung (Abschnitt 3.3)
  t <;> s           wendet s auf jedes Ziel an, das t erzeugt
  norm_num, omega   rechnen mit Zahlen (Abschnitt 3.8)
  gcongr            Ungleichungen wie c * a ≤ c * b aus a ≤ b (Abschnitt 3.8)

Ein Punkt · am Zeilenanfang bearbeitet das naechste von mehreren Zielen.
`sorry` steht fuer einen fehlenden Beweis; Lean warnt dann. Taktiken fuer
Gleichungen folgen in 0.5. -/
example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  sorry
--<<leanTaktiken>>
example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  · exact hp
  · exact hq

example (p q : Prop) (hq : q) : p → q := by
  intro _
  exact hq
--<<leanTaktikenEnd>>

-- 0.5 und 0.6: zum Nachlesen, in keiner Vorlesung gezeigt.
/-! ### 0.5 Gleichungen und `rfl`

`a = b` ist eine Aussage wie jede andere. Ihr einfachster Beweis ist `rfl`
(Reflexivitaet: a = a). `rfl` gelingt, wenn Lean beide Seiten zum selben
Ergebnis ausrechnen kann -- das prueft der Kern durch Rechnen.

Wo nichts zu rechnen ist, hilft `rfl` nicht: Fuer eine Variable `n` kann Lean
`0 + n` nicht zu `n` ausrechnen. Die Gleichung stimmt trotzdem, man braucht
aber einen Beweis, hier den Satz `Nat.zero_add` aus Leans Standardbibliothek.
Warum `n + 0` rechnet und `0 + n` nicht, zeigt `MeinNat` in Abschnitt 3.2;
die beiden Gleichheitsbegriffe dahinter behandelt Abschnitt 3.4.

Taktiken fuer Gleichungen:

  rfl      schliesst eine Gleichung, die Lean ausrechnen kann
  rw [h]   schreibt mit der Gleichung h um, von links nach rechts, und
           versucht danach rfl
  subst h  setzt eine Gleichung h ein, deren eine Seite eine Variable ist (3.7)
  simp     vereinfacht mit bekannten Gleichungen
  decide   rechnet eine entscheidbare Aussage aus, etwa (0 : ℕ) ≠ 1 -/

example : 2 + 3 = 5 := rfl
example : verdopple 21 = 42 := rfl
example : (fun x : ℕ => x + 1) 41 = 42 := rfl

example (n : ℕ) : 0 + n = n := Nat.zero_add n     -- hier genuegt rfl nicht

example (a b : ℕ) (h : a = b) : a + 1 = b + 1 := by
  rw [h]

example : (0 : ℕ) ≠ 1 := by decide      -- a ≠ b heisst ¬ (a = b)

/-! ### 0.6 Saetze finden

Lean und Mathlib benennen Saetze nach ihrer Aussage: `Nat.add_comm` ist die
Kommutativitaet der Addition auf ℕ. `#check` zeigt, was ein Satz genau sagt.
Im Editor sucht `exact?` selbst nach einem passenden Satz. -/

#check Nat.add_comm
example (a b : ℕ) : a + b = b + a := Nat.add_comm a b
#check @lt_of_le_of_lt       -- erst der Schluss, nach `of` die Voraussetzungen

/-! ## 3.1 Terme, Typen, Universen -/

--<<termeTypen>>
/- Jeder Ausdruck hat einen Typ; es gibt kein untypisiertes "Universum". -/
#check (3 : ℕ)
#check ℕ
#check Type
#check Prop

/- Typen leben selbst wieder in Typen, gestaffelt in Universen. -/
#check Type       -- Type : Type 1
#check Type 1     -- Type 1 : Type 2
example : Type = Type 0 := rfl

/- Eine Stufe ist besonders: `Prop`, die Stufe der Aussagen.
   Aussagen sind Typen wie andere auch -- man kann ueber sie quantifizieren. -/
#check Prop
example (p q : Prop) : Prop := p → q
--<<termeTypenEnd>>

/- Ein Term vom Typ ℕ: der Konstruktor `Nat.zero`, die Zahl 0. -/
def null : ℕ := Nat.zero

#check null                       -- null : ℕ
#eval null                        -- 0
example : null = 0 := rfl

--<<universeProblem>>
/- Ohne Universenvariablen braucht jede Stufe ihre eigene Definition. -/
def idNull {α : Type} (a : α) : α := a
example : ℕ := idNull 3          -- α ist ℕ, und ℕ : Type
-- example : Type := idNull ℕ    -- Fehler: α waere Type, aber Type : Type 1

def idEins {α : Type 1} (a : α) : α := a
example : Type := idEins ℕ       -- α ist Type, und Type : Type 1
-- example : ℕ := idEins 3       -- Fehler: α waere ℕ, aber ℕ : Type
--<<universeProblemEnd>>

--<<universePoly>>
universe u v

/-- Definitionen koennen ueber Universen polymorph sein: `u` ist eine
Universenvariable, kein Wert, den man ausrechnen koennte. -/
def meineId {α : Type u} (a : α) : α := a

#check @meineId.{u}
#check @meineId.{0}
#check @meineId.{1}

example : ℕ := meineId 3
example : Type := meineId ℕ
--<<universePolyEnd>>

--<<abhaengig>>
/-- Ein **abhaengiger** Funktionstyp: der Ergebnistyp haengt vom Argument ab. -/
def diagonale : (n : ℕ) → (Fin n → Fin n) := fun _ i => i

/-- Laenge im Typ: `Vektor α n` kennt seine eigene Laenge. -/
inductive Vektor (α : Type u) : ℕ → Type u where
  | nil : Vektor α 0
  | cons : α → {n : ℕ} → Vektor α n → Vektor α (n + 1)

/-- `head` ist nur fuer nichtleere Vektoren definiert -- garantiert vom Typsystem,
nicht durch eine Laufzeitpruefung. -/
def Vektor.head {α : Type u} {n : ℕ} : Vektor α (n + 1) → α
  | .cons a _ => a


#eval Vektor.head (Vektor.cons 7 Vektor.nil)
example : Vektor.head (Vektor.cons 7 Vektor.nil) = 7 := rfl
--<<abhaengigEnd>>

/-! ## 3.2 Induktive Typen, Strukturen, Typklassen

Hier nur die induktiven Typen; Strukturen und Typklassen folgen im Teil fuer
Vorlesung 3, vor 3.8. -/

--<<induktiv>>
/-- Ein induktiver Typ wird durch seine Konstruktoren erzeugt. -/
inductive MeinNat where
  | zero : MeinNat
  | succ : MeinNat → MeinNat
  deriving Repr

def MeinNat.add : MeinNat → MeinNat → MeinNat
  | n, .zero => n
  | n, .succ m => .succ (MeinNat.add n m)

def zro : MeinNat := MeinNat.zero
#check MeinNat.succ zro

#eval MeinNat.succ zro
/- Der automatisch erzeugte Rekursor ist zugleich das Induktionsprinzip. -/
#check @MeinNat.rec

theorem MeinNat.add_zero (n : MeinNat) : n.add .zero = n := rfl

theorem MeinNat.zero_add (n : MeinNat) : MeinNat.zero.add n = n := by
  induction n with
  | zero => rfl
  | succ k ih => simp [MeinNat.add, ih]
--<<induktivEnd>>

/- Zwei Gleichheitsbegriffe (in der Vorlesung direkt nach `MeinNat`):
   `MeinNat.add` rekursiert ueber das *zweite* Argument. Deshalb rechnet
   `n.add .zero` von selbst zu `n`, und `rfl` genuegt: Die Gleichheit ist
   *definitorisch*. In `MeinNat.zero.add n` steht dort die Variable `n`, es
   gibt nichts zu rechnen. Die Gleichung gilt trotzdem, aber *propositional*,
   mit einem Beweis per Induktion. Fuer `Nat` genauso, siehe Abschnitt 3.4. -/

/-! ## 3.4 Gleichheit und die Axiome -/

--<<gleichheit>>
/- Definitorische Gleichheit: der Kern rechnet selbst. -/
example : 2 + 2 = 4 := rfl
example (n : ℕ) : n + 0 = n := rfl

/- Propositionale Gleichheit: braucht einen Beweis per Induktion. -/
example (n : ℕ) : 0 + n = n := Nat.zero_add n

/- Funktionale und propositionale Extensionalitaet sind Saetze bzw. Axiome. -/
#check @funext
#check @propext
#check @Quot.sound
--<<gleichheitEnd>>

--<<axiome>>
/- Ganz Mathlib ruht auf genau drei Axiomen ueber der Typentheorie hinaus:
   `propext`, `Classical.choice`, `Quot.sound`. -/
#print axioms Nat.zero_add
#print axioms Classical.em
#print axioms Function.cantor_surjective

/- Aus der Auswahl folgt klassisch das Prinzip vom ausgeschlossenen Dritten
   (Satz von Diaconescu) -- konstruktiv ist es nicht beweisbar. -/
#check @Classical.em
example (P : Prop) : P ∨ ¬ P := Classical.em P
--<<axiomeEnd>>

-- Diaconescus Beweis in Lean (`tertium_non_datur`) steht unten nach 3.3: Er
-- braucht ∨ und ∃ als Typen und die Beweisirrelevanz.

/-! ## 3.5 Inferenzregeln und Typpruefung

In der Vorlesung zuerst die Regeln, dann Curry-Howard (3.3), dann der Rest
von 3.5: Rechnen, Pruefen, Vertrauen. -/

--<<urteile>>
/- Ein Urteil hat die Form `Γ ⊢ t : A`: Im Kontext `Γ` hat der Term `t`
   den Typ `A`. Genau danach fragt `#check`. -/
#check (3 : ℕ)
#check (fun x : ℕ => x + 1)
#check ℕ
#check Type

/- Der Kontext sind die Annahmen links vom Doppelpunkt. -/
example (x : ℕ) (y : ℕ) : ℕ := x + y
--<<urteileEnd>>

/- Die Regeln der induktiven Typen sind Konstanten: Die Konstruktoren fuehren
   Elemente ein; der Rekursor oder eine daraus gebaute Funktion wie
   `Prod.fst` oder `Sum.elim` beseitigt sie. -/
#check @Nat.zero          -- (zero)
#check @Nat.succ          -- (succ)
#check @Nat.rec           -- (rec): Rekursion und Induktion
#check @Prod.mk           -- (Paar)
#check @Prod.fst          -- (fst)
#check @Prod.snd          -- (snd)
#check @Sum.inl           -- (inl)
#check @Sum.inr           -- (inr)
#check @Sum.elim          -- (Fall)
#check @Or.elim           -- (Fall) fuer ∨: Disjunktion zuerst, Ziel c nur in Prop
#check @False.elim        -- (⊥E), ex falso: False hat keinen Konstruktor
#check @Eq.refl           -- (rfl)
#check @Eq.ndrec          -- (Transport) wie auf der Folie: motive a → a = b → motive b
#check @Eq.rec            -- dasselbe allgemeiner: das Motiv haengt auch vom Beweis ab

/- Die Berechnungsregeln gelten definitorisch: Der Kern rechnet sie selbst
   nach, also genuegt `rfl`. Die beiden letzten zeigen, was `Sum.elim` tut:
   links `f`, rechts `g`. -/
example {α β : Type} (a : α) (b : β) : (a, b).1 = a := rfl
example {α β : Type} (a : α) (b : β) : (a, b).2 = b := rfl
example {α β γ : Type} (f : α → γ) (g : β → γ) (a : α) :
    Sum.elim f g (.inl a) = f a := rfl
example {α β γ : Type} (f : α → γ) (g : β → γ) (b : β) :
    Sum.elim f g (.inr b) = g b := rfl

/- Mit (Transport) folgen Symmetrie, Transitivitaet und Kongruenz -- nicht
   aber `funext` und `propext`, dafuer braucht es die Axiome aus 3.4.
   `Eq.subst` ist (Transport) fuer eine Familie von Aussagen. Die Symmetrie:
   Mit dem Motiv P x := (x = a) wird aus rfl : P a ein Beweis von P b, also
   von b = a. -/
example {α : Type} (a b : α) (h : a = b) : b = a :=
  Eq.subst (motive := fun x => x = a) h rfl

/- Kurz mit dem Dreieck der Folie, `h ▸ t` (Eingabe \t); das Motiv findet
   Lean selbst. -/
example {α : Type} (a b : α) (h : a = b) : b = a := h ▸ rfl

--<<zweimal>>
/- Lean findet die Herleitung von ⊢ fun f x => f (f x) : (ℕ → ℕ) → ℕ → ℕ
   selbst. -/
example : (ℕ → ℕ) → ℕ → ℕ := fun f x => f (f x)
#check fun (f : ℕ → ℕ) (x : ℕ) => f (f x)
--<<zweimalEnd>>

/- Zwei einzelne `intro` statt `intro f x`: Jedes `intro` ist ein Schritt mit
   der Regel (λ). Die Option schaltet den Hinweis ab, der beide zusammenfasst. -/
set_option linter.tacticAnalysis.introMerge false in
--<<zweimalTaktik>>
example : (ℕ → ℕ) → ℕ → ℕ := by
  intro f      -- (λ):    f : ℕ → ℕ ⊢ ℕ → ℕ
  intro x      -- (λ):    f : ℕ → ℕ, x : ℕ ⊢ ℕ
  apply f      -- (App):  Funktion f; offen bleibt das Argument, ⊢ ℕ
  apply f      -- (App):  noch einmal f; offen bleibt ⊢ ℕ
  exact x      -- (Var):  x : ℕ steht im Kontext
--<<zweimalTaktikEnd>>

--<<konversion>>
/- Die **Konversionsregel** verbindet Rechnen und Typisierung: Sind `A` und
   `B` definitorisch gleich, so hat jeder Term vom Typ `A` auch den Typ `B`.
   Hier auf Typebene -- `2 + 1` und `3` rechnen zum selben Wert, also sind
   die Vektortypen derselbe Typ. -/
example (v : Vektor ℕ (2 + 1)) : Vektor ℕ 3 := v
--<<konversionEnd>>

/-! ## 3.3 Proof as Proposition: die Curry-Howard-Korrespondenz

In der Vorlesung erst nach den Regeln aus 3.5: Streicht man in einer
Typherleitung die Terme, bleibt ein Beweis im natuerlichen Schliessen. -/

--<<curryHoward>>
/- Derselbe Term, zwei Lesarten: einmal als Programm, einmal als Beweis. -/
def konstante {α β : Type} : α → β → α := fun a _ => a
theorem abschwaechung {A B : Prop} : A → B → A := fun a _ => a

/- Die Junktoren sind Typkonstruktionen: -/
example (P : Prop) : (¬ P) = (P → False) := rfl          -- Negation = Funktion nach False
example (A B : Prop) (h : A ∧ B) : A := h.1              -- Konjunktion = Produkt
example (A B : Prop) (h : A ∨ B) : B ∨ A :=              -- Disjunktion = Summe
  h.elim Or.inr Or.inl
example (p : ℕ → Prop) (h : ∃ n, p n) : ∃ m, p m := h    -- Existenz = abh. Paar

/- Ein Existenzbeweis *ist* ein Paar aus Zeuge und Nachweis. -/
example : ∃ n : ℕ, 3 < n := ⟨4, by norm_num⟩
--<<curryHowardEnd>>

/- Warum ein Term ein Beweis ist: Er schreibt den Herleitungsbaum in eine
   Zeile. Jeder Baustein ist eine Regel des natuerlichen Schliessens.
   `abtrennung` heisst nach der Abtrennungsregel, dem modus ponens. -/
theorem abtrennung {A B : Prop} : A ∧ (A → B) → B :=
  fun h =>        -- (→I): nimmt h : A ∧ (A → B) an und entlaesst die Annahme
    h.2 h.1       -- (→E) mit h.2 : A → B und h.1 : A, beide per (∧E) aus h

/- Derselbe Beweis mit Taktiken, von unten nach oben. Heraus kommt derselbe
   Term: Beide `#print` zeigen ihn gleich, mit `h.left` und `h.right` fuer
   `h.1` und `h.2`. -/
theorem abtrennung_taktik {A B : Prop} : A ∧ (A → B) → B := by
  intro h         -- (→I)
  exact h.2 h.1   -- (→E), darin zweimal (∧E)

#print abtrennung
#print abtrennung_taktik

--<<induktionTerm>>
/- Induktion als Term: der Rekursor mit einem Motiv in Prop. Derselbe Satz wie
   `MeinNat.zero_add` in 3.2, dort mit der Taktik `induction`. -/
theorem MeinNat.zero_add_term (n : MeinNat) : MeinNat.zero.add n = n :=
  MeinNat.rec (motive := fun n => MeinNat.zero.add n = n)
    rfl                                     -- Anfang: 0 + 0 = 0
    (fun _ ih => congrArg MeinNat.succ ih)  -- Schritt: aus 0 + k = k wird 0 + (k+1) = k+1
    n
--<<induktionTermEnd>>

--<<proofIrrelevance>>
/- **Beweisirrelevanz**: je zwei Beweise derselben Aussage sind
   *definitorisch* gleich. Beweise tragen keine Information. -/
example (P : Prop) (h₁ h₂ : P) : h₁ = h₂ := rfl

/- Fuer Daten gilt das natuerlich nicht: -/
example : (0 : ℕ) ≠ 1 := by decide

/- Deshalb trennt Lean `Prop` (beweisirrelevant, loeschbar)
   von `Type` (Daten, rechenrelevant). -/
--<<proofIrrelevanceEnd>>

/- Beweiseindeutigkeit und UIP: im Skript ein Absatz in Abschnitt 3.1, in der
   Vorlesung direkt nach der Beweisirrelevanz. -/

--<<beweiseindeutig>>
/- In `Prop` zaehlt nur, *dass* es einen Beweis gibt, nicht welchen.
   Je zwei Beweise derselben Aussage sind definitorisch gleich: -/
example (p : Prop) (h₁ h₂ : p) : h₁ = h₂ := rfl

/- Das gilt auch, wenn man sie sichtbar verschieden herstellt. -/
example : 0 < 2 := by norm_num
example : 0 < 2 := by omega
example (h₁ h₂ : 0 < 2) : h₁ = h₂ := rfl

/- Das greift weiter, als es zunaechst aussieht: `a = b` ist selbst eine
   `Prop`. Also sind auch je zwei *Gleichheitsbeweise* gleich -- das nennt
   man UIP, uniqueness of identity proofs. -/
example {α : Type} {a b : α} (p q : a = b) : p = q := rfl

/- Bei Daten ist es umgekehrt: verschiedene Werte bleiben verschieden. -/
example : (0 : ℕ) ≠ 1 := by decide
example (b : Bool) : b = true ∨ b = false := by cases b <;> simp
--<<beweiseindeutigEnd>>

/- Diaconescus Beweis (Skript 3.4). In der Vorlesung erst hier, am Ende von
   3.3: Er benutzt ∨ und ∃ als Typen und die Beweisirrelevanz. -/

--<<diaconescu>>
/- Diaconescus Argument, nachgebaut wie `Classical.em` in Leans
   Standardbibliothek. `Prop` spielt die Rolle der Menge {0, 1};
   statt 0 ≠ 1 genuegt `True ≠ False`. -/
theorem tertium_non_datur (p : Prop) : p ∨ ¬ p :=
  -- Aussonderung kostet nichts: ein Praedikat ist eine Funktion nach Prop.
  let U : Prop → Prop := fun x => x = True ∨ p
  let V : Prop → Prop := fun x => x = False ∨ p
  have hU : ∃ x, U x := ⟨True, Or.inl rfl⟩
  have hV : ∃ x, V x := ⟨False, Or.inl rfl⟩
  -- Auswahl (Classical.choice): aus U und aus V je ein Element waehlen.
  let u := Classical.choose hU
  let v := Classical.choose hV
  have hu : U u := Classical.choose_spec hU
  have hv : V v := Classical.choose_spec hV
  -- Extensionalitaet: gilt p, so ist U = V (erst propext, dann funext) ...
  have gleich : p → u = v := fun hp =>
    have hUV : U = V :=
      funext fun _ => propext ⟨fun _ => Or.inr hp, fun _ => Or.inr hp⟩
    -- ... und aus gleichen Praedikaten waehlt choose dasselbe Element;
    -- die beiden Existenzbeweise sind gleich (Beweisirrelevanz).
    have h : ∀ (e₁ : ∃ x, U x) (e₂ : ∃ x, V x),
        Classical.choose e₁ = Classical.choose e₂ := by
      rw [hUV]; intros; rfl
    h hU hV
  -- Fallunterscheidung wie im Beweis des Satzes von Diaconescu.
  match hu, hv with
  | Or.inr hp, _ => Or.inl hp
  | _, Or.inr hp => Or.inl hp
  | Or.inl hut, Or.inl hvf => Or.inr fun hp =>
      -- u = True und v = False; aus p folgte True = u = v = False.
      true_ne_false (hut.symm.trans ((gleich hp).trans hvf))

#print axioms tertium_non_datur
--<<diaconescuEnd>>

/- Der Preis der Beweisirrelevanz: Aus einem Existenzbeweis laesst sich der
   Zeuge nicht herausrechnen (Vorlesung 2, Folie nach "Was das praktisch
   bedeutet"). -/

--<<zeuge>>
/- Zwei Beweise mit verschiedenen Zeugen sind gleich (Beweisirrelevanz): -/
example : (⟨0, by decide⟩ : ∃ n : ℕ, n < 2) = ⟨1, by decide⟩ := rfl
/- Den Zeugen herausrechnen lehnt Lean ab, `Exists.casesOn` beseitigt nur in Prop:
def zeuge (h : ∃ n : ℕ, n < 2) : ℕ := match h with | ⟨n, _⟩ => n -/
/- In `Type` ist das Paar Daten, und `.1` rechnet den Zeugen aus: -/
example : (⟨0, by decide⟩ : {n : ℕ // n < 2}).1 = 0 := rfl
--<<zeugeEnd>>

/-! ## 3.5 (Fortsetzung) Rechnen, Pruefen, Vertrauen -/

--<<reduktionen>>
/- Definitorische Gleichheit entsteht aus Reduktionsregeln. -/

/- beta: eine Anwendung wird eingesetzt. -/
example : (fun x : ℕ => x + 1) 2 = 3 := rfl

/- delta: eine Definition wird aufgefaltet. -/
def zwei : ℕ := 2
example : zwei + zwei = 4 := rfl

/- iota: eine Fallunterscheidung trifft auf einen Konstruktor und rechnet. -/
def doppelt : ℕ → ℕ
  | 0 => 0
  | k + 1 => doppelt k + 2
example : doppelt 3 = 6 := rfl

/- eta: jede Funktion ist ihre eigene Abstraktion. -/
example (f : ℕ → ℕ) : f = fun x => f x := rfl

/- Dazu die Beweisirrelevanz in `Prop`. -/
example (p : Prop) (h₁ h₂ : p) : h₁ = h₂ := rfl

/- Die Konversionspruefung rechnet wirklich. -/
example : 2 ^ 12 = 4096 := rfl
--<<reduktionenEnd>>

-- NUR IM SKRIPT: bidirektional und elaboration. Vorlesung 2 fasst beides auf
-- den Folien "Warum der Typcheck entscheidbar ist" und "Elaborator und Kern"
-- zusammen und verweist fuer Einzelheiten auf "Type Checking in Lean 4" und
-- das Kapitel "Elaboration and Compilation" der Lean-Referenz.

--<<bidirektional>>
/- Die Pruefung laeuft in zwei Richtungen. Mit Typangabe am Binder wird der
   Typ *inferiert*, ohne sie gegen einen erwarteten Typ *geprueft*. -/
#check fun x : ℕ => x + 1      -- inferiert
#check (fun x => x : ℕ → ℕ)    -- geprueft: der Typ kommt von aussen
--<<bidirektionalEnd>>

--<<elaboration>>
/- Was man schreibt, ist nicht das, was der Kern sieht. Die Elaboration
   fuellt implizite Argumente, Instanzen und Koerzierungen ein. -/
#check (rfl : 2 + 2 = 4)

set_option pp.explicit true in
#check (rfl : 2 + 2 = 4)

/- Geprueft wird am Ende nur der fertige Term, und nur vom Kern. -/
#print axioms Nat.add_comm
--<<elaborationEnd>>

/-! ## 3.7 Lean: Term- und Taktikmodus -/

--<<termTaktik>>
/-- Termmodus: der Beweis wird direkt hingeschrieben. -/
theorem symm_term (a b : ℕ) (h : a = b) : b = a := h.symm

/-- Taktikmodus: der Beweisterm wird schrittweise konstruiert.
Auch hier entsteht ein Kern-Term, nur ein anderer. -/
theorem symm_taktik (a b : ℕ) (h : a = b) : b = a := by
  rw [h]

example (a b : ℕ) (h : a = b) : b = a := by
  subst h
  rfl
--<<termTaktikEnd>>

-- =============================================================================
-- VORLESUNG 3: Mathematik in Lean
--   Strukturen und Typklassen (Skript 3.2, Ende), Skript 3.8. Die Mengen
--   und die Taktiken am Beispiel der Mengen stehen in Mengenlehre.lean.
-- =============================================================================

/-! ## Strukturen und Typklassen (Skript 3.2, Ende) -/

--<<typklassen>>
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

/-- Ein Satz, der fuer *jedes* Monoid gilt -- Instanzsuche liefert die Struktur. -/
theorem op_e_op {α : Type} [M : MeinMonoid α] (a : α) :
    M.op (M.op a M.e) M.e = a := by
  rw [M.op_e, M.op_e]

/- So findet Lean die echten Mathlib-Strukturen: -/
#check (inferInstance : AddGroup ℤ)
#check (inferInstance : CommRing ℤ)
#check (inferInstance : Field ℝ)
--<<typklassenEnd>>

--<<strukturen>>
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
--<<strukturenEnd>>

--<<weitereMonoide>>
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
--<<weitereMonoideEnd>>

--<<erweiterung>>
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
--<<erweiterungEnd>>

/-! ## 3.8 Die Strukturen der Mathematik in Lean -/

--<<arithmetik>>
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

/- Gesetze fuer *alle* Zahlen sind kein Rechnen mehr, sondern Induktion.
   Man sieht das an der Asymmetrie: `n + 0` reduziert, `0 + n` nicht. -/
example (n : ℕ) : n + 0 = n := rfl
example (n : ℕ) : 0 + n = n := Nat.zero_add n

/- Drei Taktiken decken das meiste ab:
     ring     -- Gleichungen in kommutativen Ringen
     omega    -- lineare Arithmetik ueber ℕ und ℤ, mit Ungleichungen
     norm_num -- konkrete Zahlen, auch in ℚ und ℝ                      -/
example (a b : ℤ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by ring
example (n : ℕ) (h : 3 < n) : 2 < n := by omega
example : (1 : ℝ) / 3 + 1 / 6 = 1 / 2 := by norm_num

/-- Ein Induktionsbeweis: die Gaussche Summenformel. -/
theorem gauss (n : ℕ) : 2 * (∑ i ∈ Finset.range (n + 1), i) = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
      rw [Finset.sum_range_succ, Nat.mul_add, ih]
      ring

/- `ℝ` ist nicht berechenbar -- `#eval` gibt es dort nicht, und `decide`
   fuehrt zu nichts. Stattdessen `norm_num`. -/
#check (inferInstance : LinearOrder ℝ)
--<<arithmetikEnd>>

--<<hierarchie>>
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
--<<hierarchieEnd>>

--<<linalg>>
/- Lineare Algebra: ein Vektorraum ist ein Modul ueber einem Koerper.
   Einen eigenen Typ `VectorSpace` gibt es nicht -- er waere ueberfluessig. -/
#check @Module
#check @Module.Basis
#check @Module.finrank
#check @LinearMap

example {K V : Type*} [Field K] [AddCommGroup V] [Module K V] (a : K) (v : V) : V :=
  a • v
--<<linalgEnd>>

--<<ordnung>>
/- Ordnungen sind ebenso gestuft. -/
#check @Preorder
#check @PartialOrder
#check @LinearOrder
#check @Lattice
#check @CompleteLattice

/- Die Teilmengen eines Typs bilden einen vollstaendigen Verband. -/
example : CompleteLattice (Set ℕ) := inferInstance
--<<ordnungEnd>>

--<<mixin>>
/- **Mixins**: Struktur wird kombiniert, nicht vererbt. Ein geordnetes
   Monoid ist ein Monoid *und* eine Ordnung *und* die Vertraeglichkeit
   beider: aus a <= b folgt c*a <= c*b. Das sind drei getrennte
   Instanzargumente. `IsOrderedMonoid` traegt selbst keine Daten, es ist
   eine `Prop`. -/
#check @IsOrderedMonoid

example {M : Type*} [CommMonoid M] [PartialOrder M] [IsOrderedMonoid M]
    (a b c : M) (h : a ≤ b) : c * a ≤ c * b := by gcongr
--<<mixinEnd>>

--<<unterobjekte>>
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
--<<unterobjekteEnd>>

end Skript.Typentheorie
