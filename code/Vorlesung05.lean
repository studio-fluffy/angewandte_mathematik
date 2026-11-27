/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 5
  (Folien: folien/vorlesung05.tex)

  Fuer etwa eine Stunde, in der Reihenfolge der Folien. Fast alles wird von Hand
  aus den Definitionen bewiesen; Mathlibs fertige Saetze kommen erst danach, zum
  Vergleich. Mehr Beispiele und laengere Fassungen stehen in code/Kapitel4.lean.
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

/-! ## 4.2 Wiederholung: "schliesslich" und Konvergenz in metrischen Raeumen -/

-- Folie: "Rueckblick: Vorlesung 4" (Punkt "Schliesslich")
/- "P gilt schliesslich laengs F" heisst: Die Menge {x | P x} ist gross, sie liegt
   im Filter F. Mathlib schreibt dafuer `∀ᶠ x in F, P x` ("fuer schliesslich alle x
   laengs F"). Per Definition ist das genau die Mengenaussage: -/
example {α : Type*} (F : Filter α) (P : α → Prop) :
    (∀ᶠ x in F, P x) ↔ {x | P x} ∈ F := Iff.rfl

/- Vorweg `apply` im Kleinen. `apply` schliesst rueckwaerts: Passt die
   Folgerung eines Satzes zum Ziel, bleiben seine Annahmen zu zeigen. Ein
   bekannter Satz (in Mathlib fuer jede Ordnung, hier fuer ℝ):
     le_trans {a b c : ℝ} : a ≤ b → b ≤ c → a ≤ c
   Die Zahlen a, b, c in geschweiften Klammern sind implizit: Man schreibt sie
   beim Benutzen nicht hin, Lean liest sie ab. `apply le_trans hxy` gleicht
   die erste Annahme a ≤ b mit hxy : x ≤ y ab (also a = x, b = y) und die
   Folgerung a ≤ c mit dem Ziel x ≤ z (also c = z). Neues Ziel ist die zweite
   Annahme b ≤ c, also y ≤ z. -/
example (x y z : ℝ) (hxy : x ≤ y) (hyz : y ≤ z) : x ≤ z := by
  apply le_trans hxy                    -- Ziel vorher: x ≤ z, danach: y ≤ z
  exact hyz                             -- das ist die Annahme hyz

/- `∀ x, P x` verlangt mehr: Die Menge {x | P x} ist das Ganze. Was fuer alle x
   gilt, gilt also schliesslich, laengs jedes Filters: Das Ganze ist gross
   (Axiom (i)), und {x | P x} enthaelt es, ist also auch gross (Axiom (ii):
   Obermengen grosser Mengen sind gross). -/
example {α : Type*} (F : Filter α) (P : α → Prop) (h : ∀ x, P x) :
    ∀ᶠ x in F, P x := by
  change {x | P x} ∈ F                  -- die Definition von ∀ᶠ
  -- Wie eben bei `le_trans`, nur ist der Satz jetzt Axiom (ii), in Mathlib
  -- (mit F, U, V statt Mathlibs Namen f, x, y; alle drei implizit):
  --   mem_of_superset {F : Filter α} {U V : Set α} : U ∈ F → U ⊆ V → V ∈ F
  -- Die erste Annahme geben wir gleich mit, das Axiom (i):
  --   univ_mem : univ ∈ F            (univ ist das Ganze, in Lean `Set.univ`)
  -- Abgelesen: U = univ (aus U ∈ F und univ ∈ F), dann V = {x | P x} und der
  -- Filter F (aus V ∈ F und dem Ziel {x | P x} ∈ F). Es bleibt die zweite
  -- Annahme:
  --   Ziel vorher:   {x | P x} ∈ F
  --   Ziel danach:   univ ⊆ {x | P x}
  -- Mit allen Argumenten ausgeschrieben hiesse die Zeile
  --   apply mem_of_superset (f := F) (x := Set.univ) (y := {x | P x}) univ_mem
  apply mem_of_superset univ_mem
  -- univ ⊆ {x | P x} heisst: Aus x ∈ univ folgt x ∈ {x | P x}, fuer jedes x.
  intro x _                             -- x beliebig (x ∈ univ gilt ohnehin)
  exact h x                             -- x ∈ {x | P x} heisst P x; das ist h x

/- In Mathlib heisst das `Eventually.of_forall`. Die Umkehrung gilt nicht:
   Laengs atTop gilt 10 < x schliesslich, aber nicht fuer x = 0. -/
example {α : Type*} (F : Filter α) (P : α → Prop) (h : ∀ x, P x) :
    ∀ᶠ x in F, P x :=
  Eventually.of_forall h

-- Folie: "Konvergenz in metrischen Raeumen: ε-δ und ε-N" (vorweg: Tendsto an Beispielen)
/- Was `Tendsto f F G` heisst: Fuer jede Menge V, die gross bezueglich G ist, ist das
   Urbild f⁻¹(V) = {x | f x ∈ V} gross bezueglich F. Die beiden Filter haben
   verschiedene Aufgaben:
     G (am Ziel)   sagt, was "nah am Grenzwert" heisst. Bei 𝓝 b ist gross, was eine
                   Kugel um b enthaelt -- Zielscheiben um b, beliebig klein.
     F (am Start)  sagt, wohin x laeuft. Bei 𝓝 a: x nah genug an a (gross ist, was
                   eine Kugel um a enthaelt). Bei atTop: n gross genug (gross ist,
                   was ein Endstueck {n | n ≥ N} enthaelt).
   Tendsto heisst also: Zu jeder Zielscheibe V landet f x schliesslich (laengs F) in
   V. Denn "f⁻¹(V) ist gross" heisst per Definition "schliesslich f x ∈ V": -/
example {α β : Type*} (f : α → β) (F : Filter α) (G : Filter β) :
    Tendsto f F G ↔ ∀ V ∈ G, ∀ᶠ x in F, f x ∈ V :=
  tendsto_def

/- Beispiel 1: f(x) = 2x konvergiert bei 1 gegen 2. Eine grosse Menge V um 2 enthaelt
   eine Kugel B(2, ε), und deren Urbild ist
     f⁻¹(B(2, ε)) = {x | |2x - 2| < ε} = {x | |x - 1| < ε/2} = B(1, ε/2).
   Also enthaelt f⁻¹(V) die Kugel B(1, ε/2) und ist gross um 1. Mit Zahlen:
     Zielscheibe (1.9, 2.1),     ε = 0.1    Urbild (0.95, 1.05),     δ = 0.05
     Zielscheibe (1.99, 2.01),   ε = 0.01   Urbild (0.995, 1.005),   δ = 0.005
   Der Radius der Kugel im Urbild ist das δ der ε-δ-Definition: δ = ε/2. -/
example : Tendsto (fun x : ℝ => 2 * x) (𝓝 1) (𝓝 2) := by
  -- Schritt 1: die Definition (`tendsto_def`). Sei V gross um 2.
  rw [tendsto_def]
  intro V hV
  -- Schritt 2: V enthaelt eine Kugel B(2, ε) (`Metric.mem_nhds_iff`, Richtung →).
  obtain ⟨ε, hε, hV⟩ := Metric.mem_nhds_iff.1 hV
  -- Schritt 3: Das Urbild ist gross um 1, sobald es eine Kugel um 1 enthaelt
  -- (Richtung ←); wir nehmen B(1, ε/2). Fuer x darin bleibt zu zeigen: 2x ∈ B(2, ε),
  -- denn dann liegt 2x in V (hV), also x im Urbild. `?_` markiert diese Luecke.
  refine Metric.mem_nhds_iff.2 ⟨ε / 2, by positivity, fun x hx => hV ?_⟩
  -- Schritt 4: Kugeln als Abstaende lesen. hx : |x - 1| < ε/2, Ziel: |2x - 2| < ε.
  rw [Metric.mem_ball, Real.dist_eq] at hx ⊢
  -- Schritt 5: |2x - 2| = 2·|x - 1| < 2·(ε/2) = ε.
  calc |2 * x - 2| = 2 * |x - 1| := by
        rw [show 2 * x - 2 = 2 * (x - 1) by ring, abs_mul, abs_two]
    _ < ε := by linarith

/-- Die Sprungfunktion: 0 links von 0, 1 ab 0. -/
noncomputable def sprung (x : ℝ) : ℝ := if x < 0 then 0 else 1

/- Beispiel 2: Die Sprungfunktion konvergiert bei 0 nicht gegen 1. Zum Widerlegen
   genuegt eine einzige grosse Menge, deren Urbild nicht gross ist: V = B(1, 1/2) =
   (1/2, 3/2) ist gross um 1, aber das Urbild [0, ∞) enthaelt keine Kugel um 0 --
   jede Kugel B(0, δ) enthaelt den Punkt -δ/2, und dort ist die Funktion 0. -/
example : ¬ Tendsto sprung (𝓝 0) (𝓝 1) := by
  -- Schritt 1: Angenommen, sie konvergiert doch (`tendsto_def`).
  intro h
  rw [tendsto_def] at h
  -- Schritt 2: V = B(1, 1/2) ist gross um 1 (`Metric.ball_mem_nhds`).
  have hV : Metric.ball (1 : ℝ) (1 / 2) ∈ 𝓝 1 := Metric.ball_mem_nhds 1 (by norm_num)
  -- Schritt 3: Dann waere das Urbild gross um 0, enthielte also eine Kugel B(0, δ).
  obtain ⟨δ, hδ, hsub⟩ := Metric.mem_nhds_iff.1 (h _ hV)
  -- Schritt 4: Der Punkt -δ/2 liegt in B(0, δ), denn |-δ/2 - 0| = δ/2 < δ.
  have hx : -(δ / 2) ∈ Metric.ball (0 : ℝ) δ := by
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_neg, abs_of_pos (by positivity)]
    linarith
  -- Schritt 5: Also laege sprung(-δ/2) in B(1, 1/2). Aber -δ/2 < 0, also ist
  -- sprung(-δ/2) = 0, und |0 - 1| = 1 ist nicht < 1/2. Widerspruch.
  have h2 := hsub hx
  rw [Set.mem_preimage, Metric.mem_ball, Real.dist_eq, sprung, if_pos (by linarith)] at h2
  norm_num at h2

/- Beispiel 3: dieselbe Definition fuer eine Folge, nur mit atTop am Start. 1/(n + 1)
   konvergiert gegen 0: Eine grosse Menge V um 0 enthaelt ein Intervall (-ε, ε). Fuer
   N > 1/ε und n ≥ N ist 1/(n + 1) < 1/N < ε, also liegt 1/(n + 1) in V. Das Urbild
   enthaelt das Endstueck {n | n ≥ N}, ist also gross laengs atTop. -/
example : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (𝓝 0) := by
  -- Schritt 1: die Definition. Sei V gross um 0; V enthaelt eine Kugel B(0, ε).
  rw [tendsto_def]
  intro V hV
  obtain ⟨ε, hε, hV⟩ := Metric.mem_nhds_iff.1 hV
  -- Schritt 2: ein N > 1/ε (Archimedes, `exists_nat_gt`).
  obtain ⟨N, hN⟩ := exists_nat_gt (1 / ε)
  -- Schritt 3: Das Urbild enthaelt das Endstueck ab N (`mem_atTop_sets`): Fuer n ≥ N
  -- bleibt 1/(n + 1) ∈ B(0, ε) zu zeigen, dann liegt es in V (hV).
  refine mem_atTop_sets.2 ⟨N, fun n hn => hV ?_⟩
  -- Schritt 4: 1/(n + 1) ∈ B(0, ε) heisst |1/(n + 1) - 0| < ε, also 1/(n + 1) < ε,
  -- also 1 < ε·(n + 1). Das folgt aus 1 < N·ε (hN) und N ≤ n.
  rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos (by positivity)]
  have hn' : (N : ℝ) ≤ n := by exact_mod_cast hn
  rw [div_lt_iff₀ (by positivity)]
  rw [div_lt_iff₀ hε] at hN
  nlinarith

/- Ein Begriff fuer alle Grenzwerte, nur die Filter wechseln:
     Tendsto u atTop (𝓝 a)        die Folge u n konvergiert gegen a
     Tendsto f (𝓝 a) (𝓝 (f a))    f ist stetig in a (`ContinuousAt f a`)
     Tendsto f (𝓝[≠] a) (𝓝 b)     klassischer Grenzwert: f x → b fuer x → a, x ≠ a
     Tendsto f atTop atTop         f x → ∞ fuer x → ∞
   Vorsicht bei Tendsto f (𝓝 a) (𝓝 b): 𝓝 a enthaelt den Punkt a selbst. Mit x = a in
   der ε-δ-Bedingung unten folgt d(f a, b) < ε fuer jedes ε, also b = f a. -/

-- Folie: "Konvergenz in metrischen Raeumen: ε-δ und ε-N" (der Satz und sein Beweis)
/- Konvergenz heisst `Tendsto f F G`: Urbilder grosser Mengen sind gross. In einem
   metrischen Raum ist das genau die klassische Definition:

     Funktionen   Tendsto f (𝓝 a) (𝓝 b)  ↔  zu jedem ε > 0 ein δ > 0 mit
                                             d(f x, b) < ε, sobald d(x, a) < δ
     Folgen       Tendsto u atTop (𝓝 a)  ↔  zu jedem ε > 0 ein N mit
                                             d(u n, a) < ε fuer alle n ≥ N

   Die Beweise uebersetzen nur Definitionen:
     `tendsto_def`          Tendsto heisst: Urbilder grosser Mengen sind gross
     `Metric.mem_nhds_iff`  V ist gross um a, wenn V eine Kugel
                            B(a, ε) = {y | d(y, a) < ε} enthaelt
     `mem_atTop_sets`       V ist gross laengs atTop, wenn V ein Endstueck
                            {n | n ≥ N} enthaelt                                -/
/-- Konvergenz von Funktionen zwischen metrischen Raeumen ist die
ε-δ-Definition. Mit b = f a ist das die ε-δ-Stetigkeit: `ContinuousAt f a`
ist per Definition `Tendsto f (𝓝 a) (𝓝 (f a))`. -/
theorem konvergenz_eps_delta {X Y : Type*} [MetricSpace X] [MetricSpace Y]
    (f : X → Y) (a : X) (b : Y) :
    Tendsto f (𝓝 a) (𝓝 b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ x, dist x a < δ → dist (f x) b < ε := by
  -- Schreibweise: `V ∈ 𝓝 b` heisst "V ist gross um b". B(b, ε) ist die
  -- Kugel `Metric.ball b ε` = {y | d(y, b) < ε}, mit dem Abstand
  -- d(y, b) = `dist y b`. f⁻¹(V) ist das Urbild `f ⁻¹' V` = {x | f x ∈ V}.
  --
  -- Schritt 1: Tendsto aufklappen. Tendsto heisst: Jede grosse Menge um b
  -- hat ein grosses Urbild um a. Beispiel f(x) = 2x, a = 1, b = 2: Eine
  -- grosse Menge V um 2 enthaelt eine Kugel B(2, ε), etwa (1.9, 2.1) fuer
  -- ε = 0.1. Deren Urbild ist B(1, ε/2), hier (0.95, 1.05). Damit enthaelt
  -- f⁻¹(V) eine Kugel um 1 und ist gross um 1. In Mathlib ist das der Satz
  --   tendsto_def : Tendsto f l₁ l₂ ↔ ∀ s ∈ l₂, f ⁻¹' s ∈ l₁
  -- `rw` ersetzt im Ziel die linke Seite durch die rechte, mit l₁ = 𝓝 a und
  -- l₂ = 𝓝 b. Ziel danach:
  --   (∀ s ∈ 𝓝 b, f ⁻¹' s ∈ 𝓝 a) ↔
  --     ∀ ε > 0, ∃ δ > 0, ∀ x, dist x a < δ → dist (f x) b < ε
  rw [tendsto_def]
  -- Schritt 2: Ein ↔ zeigt man in zwei Richtungen. `constructor` macht
  -- daraus zwei Ziele, (→) und (←); die beiden mit `·` beginnenden Bloecke
  -- beweisen je eines.
  constructor
  · -- (→) Gegeben: Urbilder grosser Mengen sind gross.
    -- Gesucht: zu jedem ε ein δ.
    -- Schritt 3: Das Ziel lautet (∀ s ∈ 𝓝 b, ...) → (∀ ε > 0, ...).
    -- `intro` nimmt die linke Seite als Annahme h und dazu ein beliebiges
    -- ε. Dabei ist `∀ ε > 0, ...` kurz fuer `∀ ε, ε > 0 → ...`, darum
    -- zwei Namen: ε und hε.
    --   h  : ∀ s ∈ 𝓝 b, f ⁻¹' s ∈ 𝓝 a
    --   hε : ε > 0
    --   Ziel: ∃ δ > 0, ∀ x, dist x a < δ → dist (f x) b < ε
    intro h ε hε
    -- Schritt 4: B(b, ε) ist gross um b, denn sie enthaelt eine Kugel um
    -- b, naemlich sich selbst. "Gross um x" heisst in Mathlib (Radius r
    -- statt Mathlibs ε, damit er nicht mit unserem ε verwechselt wird):
    --   Metric.mem_nhds_iff : s ∈ 𝓝 x ↔ ∃ r > 0, Metric.ball x r ⊆ s
    -- Ein Beweis von A ↔ B ist ein Paar zweier Implikationen, so wie ein
    -- Beweis von A ∧ B ein Paar ist: `.1` ist A → B, `.2` ist B → A. Mit
    -- `.2` genuegt ein Beweis der rechten Seite, fuer s = B(b, ε) und x = b
    -- also einer von ∃ r > 0, B(b, r) ⊆ B(b, ε). Das ist das Tripel
    -- ⟨ε, hε, subset_rfl⟩, kurz fuer ⟨ε, ⟨hε, subset_rfl⟩⟩:
    --   Zeuge      r = ε
    --   Nachweis   hε : ε > 0
    --   Nachweis   subset_rfl : B(b, ε) ⊆ B(b, ε)
    -- (`subset_rfl`: Jede Menge ist Teilmenge von sich selbst.)
    have hB : Metric.ball b ε ∈ 𝓝 b := Metric.mem_nhds_iff.2 ⟨ε, hε, subset_rfl⟩
    -- Schritt 5: Also ist das Urbild f⁻¹(B(b, ε)) gross um a. Dazu h
    -- anwenden: `∀ s ∈ 𝓝 b, ...` ist kurz fuer `∀ s, s ∈ 𝓝 b → ...`; h
    -- nimmt also erst eine Menge s, dann den Nachweis s ∈ 𝓝 b. Fuer s
    -- steht `_`: Lean liest s = B(b, ε) an hB ab.
    --   h _ hB : f ⁻¹' Metric.ball b ε ∈ 𝓝 a
    -- Gross um a heisst: enthaelt eine Kugel um a. `.1` von
    -- `Metric.mem_nhds_iff` macht daraus
    --   ∃ r > 0, Metric.ball a r ⊆ f ⁻¹' Metric.ball b ε,
    -- und `obtain` packt das aus und nennt den Radius δ:
    --   δ    : ℝ
    --   hδ   : δ > 0
    --   hsub : Metric.ball a δ ⊆ f ⁻¹' Metric.ball b ε
    --          (also B(a, δ) ⊆ f⁻¹(B(b, ε)))
    obtain ⟨δ, hδ, hsub⟩ := Metric.mem_nhds_iff.1 (h _ hB)
    -- Schritt 6: Dieses δ leistet es. Das Ziel ∃ δ > 0, ... bekommt wieder
    -- ein Tripel: Zeuge δ, Nachweis hδ und einen Beweis von
    --   ∀ x, dist x a < δ → dist (f x) b < ε
    -- Ein Beweis von "fuer alle x: aus P x folgt Q x" ist eine Funktion:
    -- Sie nimmt x und hx : P x und liefert einen Beweis von Q x. Hier ist
    -- das hsub hx:
    --   hx      : dist x a < δ        per Definition: x ∈ B(a, δ)
    --   hsub hx : x ∈ f⁻¹(B(b, ε))    per Definition: f x ∈ B(b, ε),
    --                                 also dist (f x) b < ε
    -- Kugel und Urbild sind so definiert (`Metric.mem_ball`,
    -- `Set.mem_preimage`); Lean setzt die Definitionen selbst ein, ein `rw`
    -- ist nicht noetig.
    exact ⟨δ, hδ, fun x hx => hsub hx⟩
  · -- (←) Gegeben: zu jedem ε ein δ.
    -- Gesucht: Urbilder grosser Mengen sind gross.
    -- Schritt 7: Das Ziel lautet
    --   (∀ ε > 0, ...) → (∀ s ∈ 𝓝 b, f ⁻¹' s ∈ 𝓝 a).
    -- `intro` nimmt die linke Seite als Annahme h, dann eine Menge s, die
    -- wir V nennen, und den Nachweis hV, dass V gross um b ist:
    --   h  : ∀ ε > 0, ∃ δ > 0, ∀ x, dist x a < δ → dist (f x) b < ε
    --   hV : V ∈ 𝓝 b
    --   Ziel: f ⁻¹' V ∈ 𝓝 a
    intro h V hV
    -- Schritt 8: V ist gross um b, enthaelt also eine Kugel um b (`.1` von
    -- `Metric.mem_nhds_iff`). `obtain` packt aus und nennt den Radius ε:
    --   ε  : ℝ
    --   hε : ε > 0
    --   hV : Metric.ball b ε ⊆ V            (also B(b, ε) ⊆ V)
    -- Der Name hV ist damit neu vergeben: Das alte hV : V ∈ 𝓝 b wird nicht
    -- mehr gebraucht und ist ab hier verdeckt.
    obtain ⟨ε, hε, hV⟩ := Metric.mem_nhds_iff.1 hV
    -- Schritt 9: Zu diesem ε gibt es ein δ: h in ε und hε einsetzen,
    -- auspacken.
    --   δ  : ℝ
    --   hδ : δ > 0
    --   hd : ∀ x, dist x a < δ → dist (f x) b < ε
    obtain ⟨δ, hδ, hd⟩ := h ε hε
    -- Schritt 10: Die Kugel B(a, δ) liegt im Urbild f⁻¹(V), also ist f⁻¹(V)
    -- gross um a. Mit `.2` von `Metric.mem_nhds_iff` genuegt ein Tripel:
    -- Zeuge δ, Nachweis hδ und ein Beweis von B(a, δ) ⊆ f⁻¹(V). Eine
    -- Inklusion s ⊆ t heisst ∀ x, x ∈ s → x ∈ t; ihr Beweis ist also wieder
    -- eine Funktion von x und hx:
    --   hx           : x ∈ B(a, δ)        per Definition: dist x a < δ
    --   hd x hx      : dist (f x) b < ε   per Definition: f x ∈ B(b, ε)
    --   hV (hd x hx) : f x ∈ V            per Definition: x ∈ f⁻¹(V)
    exact Metric.mem_nhds_iff.2 ⟨δ, hδ, fun x hx => hV (hd x hx)⟩

/-- Konvergenz von Folgen in einem metrischen Raum ist die ε-N-Definition. -/
theorem konvergenz_eps_N {X : Type*} [MetricSpace X] (u : ℕ → X) (a : X) :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, dist (u n) a < ε := by
  -- Tendsto: Jede grosse Menge V um a hat ein grosses Urbild u⁻¹(V).
  rw [tendsto_def]
  constructor
  · -- (→) Sei ε > 0. Die Kugel B(a, ε) ist gross um a, also ist ihr Urbild gross
    -- laengs atTop: Es enthaelt ein Endstueck. Ab N liegen alle u n in B(a, ε).
    intro h ε hε
    have hB : Metric.ball a ε ∈ 𝓝 a := Metric.mem_nhds_iff.2 ⟨ε, hε, subset_rfl⟩
    exact mem_atTop_sets.1 (h _ hB)
  · -- (←) Sei V gross um a. Dann enthaelt V eine Kugel B(a, ε). Ab dem N zu ε
    -- liegen alle u n in B(a, ε) ⊆ V: Das Urbild u⁻¹(V) enthaelt das Endstueck.
    intro h V hV
    obtain ⟨ε, hε, hV⟩ := Metric.mem_nhds_iff.1 hV
    obtain ⟨N, hN⟩ := h ε hε
    exact mem_atTop_sets.2 ⟨N, fun n hn => hV (hN n hn)⟩

/- Mathlib hat beide Saetze: `Metric.tendsto_nhds_nhds` (ε-δ) und
   `Metric.tendsto_atTop` (ε-N). -/
example {X Y : Type*} [MetricSpace X] [MetricSpace Y] (f : X → Y) (a : X) (b : Y) :
    Tendsto f (𝓝 a) (𝓝 b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ x, dist x a < δ → dist (f x) b < ε :=
  Metric.tendsto_nhds_nhds

-- Folie: "Konvergenz in metrischen Raeumen: ε-δ und ε-N" (das Beispiel: die geometrische Reihe)
/- Eine bekannte Reihe: 1/2 + 1/4 + 1/8 + ... = 1. Die Partialsummen sind
   s_n = 1/2 + ... + 1/2^n = 1 - 1/2^n: Es fehlt immer genau das letzte Stueck
   1/2^n (wer einen Kuchen immer wieder halbiert, behaelt 1/2^n uebrig). Erst die
   Formel, durch Induktion: -/
/-- Die Partialsummen der geometrischen Reihe 1/2 + 1/4 + 1/8 + ... -/
theorem geometrische_partialsumme (n : ℕ) :
    ∑ k ∈ Finset.range n, (1 / 2 : ℝ) ^ (k + 1) = 1 - (1 / 2) ^ n := by
  induction n with
  | zero => simp                 -- leere Summe: 0 = 1 - 1
  | succ n ih =>
    -- s_(n+1) = s_n + 1/2^(n+1) (`Finset.sum_range_succ`), dann ih fuer s_n
    rw [Finset.sum_range_succ, ih]
    ring                         -- 1 - 1/2^n + 1/2^(n+1) = 1 - 1/2^(n+1)

/- Dann der Grenzwert, mit der ε-N-Definition. Der Abstand zum Grenzwert ist
   |s_n - 1| = 1/2^n. Zu ε > 0 waehlt man N > 1/ε (Archimedes, `exists_nat_gt`).
   Fuer n ≥ N ist dann 2^n > n ≥ N > 1/ε, also 1/2^n < ε. -/
/-- Die geometrische Reihe 1/2 + 1/4 + 1/8 + ... hat den Wert 1. -/
theorem geometrische_reihe :
    Tendsto (fun n => ∑ k ∈ Finset.range n, (1 / 2 : ℝ) ^ (k + 1)) atTop
      (𝓝 1) := by
  rw [konvergenz_eps_N]                    -- die ε-N-Definition
  intro ε hε                               -- ε > 0 beliebig
  obtain ⟨N, hN⟩ := exists_nat_gt (1 / ε)  -- Schritt 1: ein N > 1/ε
  use N
  intro n hn                               -- hn : n ≥ N
  -- Schritt 2: der Abstand. d(s_n, 1) = |(1 - 1/2^n) - 1| = |-(1/2^n)| = 1/2^n.
  rw [geometrische_partialsumme, Real.dist_eq, sub_sub_cancel_left, abs_neg,
    abs_of_pos (by positivity)]
  -- Schritt 3: 2^n > n ≥ N > 1/ε, mit `Nat.lt_two_pow_self` (n < 2^n);
  -- `exact_mod_cast` holt die Ungleichungen fuer natuerliche Zahlen nach ℝ.
  have h1 : (n : ℝ) < 2 ^ n := by exact_mod_cast Nat.lt_two_pow_self
  have h2 : (N : ℝ) ≤ n := by exact_mod_cast hn
  have h3 : 1 / ε < 2 ^ n := by linarith
  -- Schritt 4: Aus 1/ε < 2^n wird 1 < 2^n·ε, und das heisst 1/2^n < ε.
  rw [div_lt_iff₀ hε] at h3                -- h3 : 1 < 2^n * ε
  rw [one_div_pow, div_lt_iff₀ (by positivity)]  -- Ziel: 1 < ε * 2^n
  linarith

/- Mathlib hat die Reihe als `hasSum_geometric_two'` (Σ a/2/2^k = a); "die Reihe hat
   die Summe s" heisst dort `HasSum`. -/

/-! ## 4.4 Landau-Symbole: Vergleichen laengs eines Filters -/

-- Folie: "Schliesslich: die Menge, auf der P gilt"
/- Laengs `atTop` ist gross, was ein Endstueck [N, ∞) enthaelt. Zum Beispiel
   {x | 10 < x}: Die Menge enthaelt das Endstueck [11, ∞).

     --|----|----|----|----|----|------> x
       8    9   10   11   12   13
                 o=====================>    {x | 10 < x}
                      [================>    das Endstueck [11, ∞)          -/
example : {x : ℝ | 10 < x} ∈ atTop := by
  -- Axiom (ii), in Mathlib `mem_of_superset`: Aus U ∈ F und U ⊆ V folgt V ∈ F. Als U
  -- nehmen wir [11, ∞), in Mathlib `Set.Ici 11` (Intervall, links abgeschlossen, nach
  -- rechts unendlich); gross ist es nach `Ici_mem_atTop 11`.
  apply mem_of_superset (Ici_mem_atTop (11 : ℝ))
  -- Bleibt U ⊆ V: Jedes x mit 11 ≤ x erfuellt 10 < x.
  intro x hx                                       -- hx : x ∈ [11, ∞)
  have h : (11 : ℝ) ≤ x := hx                      -- also 11 ≤ x
  change 10 < x                                    -- x ∈ {x | 10 < x} heisst 10 < x
  linarith

-- Folie: "Klein-o laengs eines Filters" (die Definition, selbst gebaut)
/-- Klein-o laengs eines Filters: Zu jedem c > 0 gilt schliesslich
|f x| ≤ c·|g x|. Erst kommt c, dann die grosse Menge -- zu jedem c eine
eigene. `∀ᶠ x in l, P x` heisst: Die Menge, auf der P gilt, ist gross. -/
def KleinO {α : Type*} (f g : α → ℝ) (l : Filter α) : Prop :=
  ∀ c > 0, ∀ᶠ x in l, |f x| ≤ c * |g x|

-- Folie: "Was schliesslich heisst, sagt der Filter" (die Tabelle)
/-- Laengs 𝓝 a heisst "schliesslich" "auf einer Kugel um a": die ε-δ-Form,
mit c statt ε. Zu jedem c gehoert ein eigenes δ. -/
theorem klein_o_nhds_iff (f g : ℝ → ℝ) (a : ℝ) :
    KleinO f g (𝓝 a) ↔
      ∀ c > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x| ≤ c * |g x| := by
  -- Schritt 1: Die Definition von `KleinO` einsetzen. Links steht dann
  --   ∀ c > 0, ∀ᶠ x in 𝓝 a, |f x| ≤ c * |g x|      (zu jedem c "schliesslich")
  unfold KleinO
  -- Schritt 2: "schliesslich laengs 𝓝 a" als "auf einer Kugel um a" lesen. Das sagt
  -- `Metric.eventually_nhds_iff`: (∀ᶠ x in 𝓝 a, P x) ↔ ∃ ε > 0, ∀ x, dist x a < ε → P x.
  -- Auf ℝ ist dist x a = |x - a| (`Real.dist_eq`). Danach sind beide Seiten gleich.
  simp only [Metric.eventually_nhds_iff, Real.dist_eq]

-- Folie: "Warum ohne Teilen?" (das Beispiel (y - x)² = o(y - x), ganz ohne Teilen)
-- Folie: "Vorgerechnet: f(y) = y^2" ((y - x)² = o(y - x), zu jedem c eine eigene Kugel)
-- und "Landau-Symbole in Lean" (x² = o(x) bei 0)
/-- Bei x ist (y - x)² klein gegen y - x: Zu jedem c reicht die Kugel vom
Radius c um x. Wie nah "nahe genug" ist, haengt von c ab. -/
theorem quadrat_klein_o (x : ℝ) :
    KleinO (fun y : ℝ => (y - x) ^ 2) (fun y => y - x) (𝓝 x) := by
  -- Schritt 1 ("zu jedem c > 0" im Beweis auf Papier): c > 0 beliebig. Das Ziel ist
  --   ∀ᶠ y in 𝓝 x, |(y - x) ^ 2| ≤ c * |y - x|
  intro c hc
  -- Schritt 2 ("auf der Kugel |y - x| < c"): Die Kugel B(x, c) ist gross
  -- (`Metric.ball_mem_nhds`). `filter_upwards` nimmt sie und verlangt die Ungleichung
  -- nur noch fuer ein y darin: hy : y ∈ B(x, c).
  filter_upwards [Metric.ball_mem_nhds x hc] with y hy
  -- Schritt 3: y ∈ B(x, c) heisst dist y x < c, also |y - x| < c.
  rw [Metric.mem_ball, Real.dist_eq] at hy
  -- Schritt 4: |(y - x)²| = |y - x|² = |y - x|·|y - x|. Zu zeigen bleibt
  --   |y - x| * |y - x| ≤ c * |y - x|
  rw [abs_pow, sq]
  -- Schritt 5: Einen Faktor |y - x| durch das groessere c ersetzen (hy : |y - x| < c);
  -- das geht, weil der andere Faktor |y - x| nicht negativ ist.
  exact mul_le_mul_of_nonneg_right hy.le (abs_nonneg _)

/- Bei 0 ist x² klein gegen x -- der Fall x = 0 von `quadrat_klein_o`: -/
example : KleinO (fun x : ℝ => x ^ 2) (fun x => x) (𝓝 0) := by
  -- `quadrat_klein_o 0` sagt: (y - 0)² ist klein gegen y - 0. `sub_zero` macht aus
  -- y - 0 ein y, im Satz und im Ziel; dann stimmen beide ueberein.
  simpa only [sub_zero] using quadrat_klein_o 0

-- Folie: "Rechnen mit o: Die Filteraxiome arbeiten" und "Die Summenregel im Bild: zwei
-- grosse Mengen schneiden" (Schritt 2 unten ist der Schnitt U_1 ∩ U_2 im Bild)
/-- Summenregel fuer o, von Hand: Zu c nimmt man fuer beide Summanden c/2.
Jede Abschaetzung gilt auf einer grossen Menge, beide zusammen auf dem
Schnitt -- gross nach Axiom (iii); `filter_upwards [h₁ _, h₂ _]` bildet ihn.
In Mathlib heisst die Regel `Asymptotics.IsLittleO.add`. -/
theorem klein_o_add {α : Type*} {l : Filter α} {f₁ f₂ g : α → ℝ}
    (h₁ : KleinO f₁ g l) (h₂ : KleinO f₂ g l) :
    KleinO (fun x => f₁ x + f₂ x) g l := by
  -- Schritt 0: Sei c > 0. Zu zeigen: Die Menge {x | |f₁ x + f₂ x| ≤ c·|g x|} ist gross.
  intro c hc
  -- Schritt 1 (Papier 1.): Mit c/2 statt c sind U₁ = {|f₁ x| ≤ c/2·|g x|} und
  -- U₂ = {|f₂ x| ≤ c/2·|g x|} gross: `h₁ (c / 2) _` und `h₂ (c / 2) _`.
  -- Schritt 2 und 3 (Papier 2., 3.): `filter_upwards` bildet den Schnitt U₁ ∩ U₂
  -- (Axiom (iii)) und verlangt die Behauptung nur fuer ein x darin -- das genuegt nach
  -- Axiom (ii), denn die Menge der Behauptung enthaelt dann U₁ ∩ U₂.
  -- Es ist hx₁ : |f₁ x| ≤ c/2·|g x| und hx₂ : |f₂ x| ≤ c/2·|g x|.
  filter_upwards [h₁ (c / 2) (half_pos hc),
    h₂ (c / 2) (half_pos hc)] with x hx₁ hx₂
  calc |f₁ x + f₂ x|
      ≤ |f₁ x| + |f₂ x| := abs_add_le _ _           -- Dreiecksungleichung
    _ ≤ c / 2 * |g x| + c / 2 * |g x| := add_le_add hx₁ hx₂  -- beide mit hx₁, hx₂
    _ = c * |g x| := by ring                         -- c/2 + c/2 = c

-- Folie: "Landau-Symbole in Lean" (der Abgleich mit Mathlib)
/- Mathlibs `f =o[l] g` ist dieselbe Definition, fuer Funktionen in normierte
   Raeume. Kein `Iff.rfl`: Mathlib verpackt die Definition, und
   `Asymptotics.isLittleO_iff` klappt sie auf. Auf ℝ ist die Norm der Betrag. -/
theorem klein_o_iff_isLittleO {α : Type*} (f g : α → ℝ) (l : Filter α) :
    KleinO f g l ↔ f =o[l] g := by
  -- Schritt 1: Mathlibs `=o` aufklappen (`Asymptotics.isLittleO_iff`). Rechts steht dann
  --   ∀ c, 0 < c → ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖
  rw [Asymptotics.isLittleO_iff]
  -- Schritt 2: `KleinO` einsetzen. Auf ℝ ist die Norm der Betrag (`Real.norm_eq_abs`), und
  -- "c > 0" ist "0 < c" (`gt_iff_lt`). Danach sind beide Seiten gleich.
  simp only [KleinO, Real.norm_eq_abs, gt_iff_lt]

/-! ## 4.4 Ableitung in einer Variablen -/

-- Folie: "Die Definition" (der Ableitung)
/- `HasDerivAt f f' x` ist Mathlibs Ableitung. Sie heisst: Der Fehler der Tangente,
   f y - f x - (y - x)·f', ist o(y - x) laengs 𝓝 x. In unserer Schreibweise `KleinO`: -/
/-- Die Ableitung, wie auf der Folie: Der Fehler der Tangente ist klein gegen
y - x laengs 𝓝 x. Mathlib sagt das mit `=o` (`hasDerivAt_iff_isLittleO`);
`klein_o_iff_isLittleO` uebersetzt es in unsere Fassung. -/
theorem ableitung_klein_o (f : ℝ → ℝ) (f' x : ℝ) :
    HasDerivAt f f' x ↔
      KleinO (fun y => f y - f x - (y - x) * f') (fun y => y - x) (𝓝 x) := by
  -- Schritt 1: Mathlibs Definition der Ableitung (`hasDerivAt_iff_isLittleO`). Links steht
  --   (fun y => f y - f x - (y - x) • f') =o[𝓝 x] fun y => y - x
  rw [hasDerivAt_iff_isLittleO]
  -- Schritt 2: Unser `KleinO` ist Mathlibs `=o` (`klein_o_iff_isLittleO`); rechts wird
  -- `KleinO ...` zu `... =o[𝓝 x] ...`.
  rw [klein_o_iff_isLittleO]
  -- Schritt 3: In ℝ ist • die Multiplikation, (y - x) • f' = (y - x) * f'
  -- (`smul_eq_mul`). Danach sind beide Seiten gleich.
  simp only [smul_eq_mul]

-- Folie: "Schliesslich bei der Ableitung: zu jedem c eine Kugel" (die ε-δ-Form)
/-- Die Ableitung, ganz ausgeschrieben: Laengs 𝓝 x heisst "schliesslich"
"auf einer Kugel um x". Heraus kommt die vertraute ε-δ-Form, mit c statt ε. -/
theorem ableitung_eps_delta (f : ℝ → ℝ) (f' x : ℝ) :
    HasDerivAt f f' x ↔ ∀ c > 0, ∃ δ > 0, ∀ y, |y - x| < δ →
      |f y - f x - (y - x) * f'| ≤ c * |y - x| := by
  -- Schritt 1: Die Definition einsetzen (`ableitung_klein_o`): Fehler = o(y - x). Das Ziel
  --   KleinO (fun y => f y - f x - (y - x) * f') (fun y => y - x) (𝓝 x) ↔ ∀ c > 0, ...
  rw [ableitung_klein_o]
  -- Schritt 2: "schliesslich laengs 𝓝 x" heisst "auf einer Kugel um x"
  -- (`klein_o_nhds_iff`). Danach sind beide Seiten gleich.
  rw [klein_o_nhds_iff]

-- Folie: "Vorgerechnet: f(y) = y^2" (zu Fuss, mit der Definition)
/- y² hat in x die Ableitung 2x, aus der Definition: Der Fehler der Tangente
   ist (y - x)², und das ist o(y - x) nach `quadrat_klein_o`. -/
example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
  -- Die Definition (`ableitung_klein_o`): Der Fehler der Tangente soll o(y - x) sein.
  rw [ableitung_klein_o]
  -- Schritt 1 (Papier 1.): Den Fehler ausrechnen. y² - x² - (y - x)·2x = (y - x)²;
  -- `ext y` greift sich ein y, `ring` rechnet nach.
  have fehler : (fun y : ℝ => y ^ 2 - x ^ 2 - (y - x) * (2 * x)) =
      fun y => (y - x) ^ 2 := by
    ext y; ring
  rw [fehler]                            -- Ziel: (y - x)² ist o(y - x)
  -- Schritt 2 (Papier 2.): Das ist `quadrat_klein_o` -- ohne zu teilen, mit der Kugel B(x, c).
  exact quadrat_klein_o x

/- Mathlib hat die Rechenregeln fertig. Mit der Potenzregel `hasDerivAt_pow` ist
   dasselbe eine Zeile -- erst die Definition verstehen, dann die Regel benutzen. -/
example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
  simpa using hasDerivAt_pow 2 x

-- Folie: "Fermat: Am lokalen Minimum ist f' = 0"
/-- Fermat, zu Fuss aus der ε-δ-Form der Ableitung: Waere f' ≠ 0, so ginge man
einen kleinen Schritt s gegen die Steigung -- nach links, falls f' > 0, nach
rechts, falls f' < 0. Mit c = |f'|/2 ist dort f(y) - f(a) ≤ -s·|f'|/2 < 0, also
f(y) < f(a): kein Minimum. -/
theorem fermat_zu_fuss {f : ℝ → ℝ} {a f' : ℝ} (h : IsLocalMin f a)
    (hf : HasDerivAt f f' a) : f' = 0 := by
  -- Schritt 1: Annahme f' ≠ 0. Dann ist c = |f'|/2 > 0.
  by_contra hne
  have hc : 0 < |f'| / 2 := by positivity
  -- Schritt 2: die Ableitung in der ε-δ-Form, mit diesem c. Es gibt ein δ > 0 mit
  --   |f y - f a - (y - a)·f'| ≤ |f'|/2 · |y - a|   fuer |y - a| < δ.
  obtain ⟨δ, hδ, hab⟩ := (ableitung_eps_delta f f' a).1 hf (|f'| / 2) hc
  -- Schritt 3: das Minimum auf einer Kugel. `IsLocalMin f a` heisst "schliesslich laengs
  -- 𝓝 a ist f a ≤ f y", also gibt es ein ρ > 0 mit f a ≤ f y fuer d(y, a) < ρ.
  obtain ⟨ρ, hρ, hmin⟩ := Metric.eventually_nhds_iff.1 h
  -- Schritt 4: eine Schrittweite s > 0, kleiner als δ und als ρ.
  have hs : 0 < min δ ρ / 2 := by positivity
  have hsδ : min δ ρ / 2 < δ := by linarith [min_le_left δ ρ]
  have hsρ : min δ ρ / 2 < ρ := by linarith [min_le_right δ ρ]
  set s := min δ ρ / 2
  -- Schritt 5: ein Schritt s gegen die Steigung.
  rcases lt_or_gt_of_ne hne with hneg | hpos
  · -- f' < 0: nach rechts, y = a + s, also y - a = s.
    have h1 := hab (a + s) (by rw [add_sub_cancel_left, abs_of_pos hs]; exact hsδ)
    have h2 := hmin (show dist (a + s) a < ρ by
      rw [Real.dist_eq, add_sub_cancel_left, abs_of_pos hs]; exact hsρ)
    rw [add_sub_cancel_left, abs_of_pos hs, abs_of_neg hneg] at h1
    -- h1 : |f(a + s) - f a - s·f'| ≤ -f'/2 · s. Also f(a + s) - f a ≤ s·f'/2 < 0,
    -- aber h2 : f a ≤ f(a + s).
    have h3 := (abs_le.1 h1).2
    nlinarith [mul_neg_of_pos_of_neg hs hneg]
  · -- f' > 0: nach links, y = a - s, also y - a = -s.
    have h1 := hab (a - s) (by rw [sub_sub_cancel_left, abs_neg, abs_of_pos hs]; exact hsδ)
    have h2 := hmin (show dist (a - s) a < ρ by
      rw [Real.dist_eq, sub_sub_cancel_left, abs_neg, abs_of_pos hs]; exact hsρ)
    rw [sub_sub_cancel_left, abs_neg, abs_of_pos hs, abs_of_pos hpos] at h1
    -- h1 : |f(a - s) - f a - (-s)·f'| ≤ f'/2 · s. Also f(a - s) - f a ≤ -s·f'/2 < 0,
    -- aber h2 : f a ≤ f(a - s).
    have h3 := (abs_le.1 h1).2
    nlinarith [mul_pos hs hpos]

/- Mathlib hat Fermat als `IsLocalMin.hasDerivAt_eq_zero`. -/
example {f : ℝ → ℝ} {a f' : ℝ} (h : IsLocalMin f a) (hf : HasDerivAt f f' a) :
    f' = 0 :=
  h.hasDerivAt_eq_zero hf

/-! ## 4.4 Mehrere Veraenderliche: partielle Ableitungen, Jacobi-Matrix, Kettenregel -/

-- Folie: "Mehrere Veraenderliche: Die Ableitung ist eine Matrix" (die Matrix als lineare
-- Abbildung)
/-- Eine m×n-Matrix als lineare Abbildung ℝⁿ → ℝᵐ, h ↦ M h (stetig, denn die
Dimension ist endlich). So wird aus einer Jacobi-Matrix eine Ableitung. -/
noncomputable def matrixAbb {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) :
    (Fin n → ℝ) →L[ℝ] (Fin m → ℝ) :=
  LinearMap.toContinuousLinearMap (Matrix.toLin' M)

/-- Angewandt auf h ist sie Matrix mal Vektor. -/
@[simp] lemma matrixAbb_apply {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ)
    (h : Fin n → ℝ) : matrixAbb M h = M.mulVec h := rfl

/-- Hintereinander ausfuehren heisst Matrizen multiplizieren. -/
lemma matrixAbb_comp {m n k : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ) :
    (matrixAbb A).comp (matrixAbb B) = matrixAbb (A * B) := by
  -- Schritt 1: Zwei stetige lineare Abbildungen sind gleich, wenn sie an jedem Vektor h
  -- uebereinstimmen (`ext h : 1`). Ziel: A (B h) = (A * B) h.
  ext h : 1
  -- Schritt 2: Matrix mal (Matrix mal Vektor) ist (Matrix mal Matrix) mal Vektor
  -- (`Matrix.mulVec_mulVec`).
  simp [Matrix.mulVec_mulVec]

-- Folie: "Lineare Abbildungen und Bewegungen: Die Ableitung ist die Matrix"
/-- Satz: Eine affine Abbildung f(w) = M w + b hat in jedem Punkt v die Ableitung M, als
lineare Abbildung `matrixAbb M`: Ihre Jacobi-Matrix ist M selbst. Von Hand nach der
Definition, wie auf der Folie: Der Rest f(v + h) - f(v) - M h ist null, und null ist o(h)
-- fuer jedes h, nicht erst schliesslich. -/
theorem ableitung_affin {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ)
    (v : Fin n → ℝ) :
    HasFDerivAt (fun w : Fin n → ℝ => M.mulVec w + b) (matrixAbb M) v := by
  -- Schritt 1 (= 1. auf der Folie, der Rest ist null): M ist linear,
  -- M (v + h) = M v + M h (`Matrix.mulVec_add`). Dann hebt sich alles weg; `abel` ordnet
  -- die Summanden und kuerzt.
  have rest : ∀ h, M.mulVec (v + h) + b - (M.mulVec v + b) - matrixAbb M h = 0 := by
    intro h
    rw [matrixAbb_apply, Matrix.mulVec_add]
    abel
  -- Schritt 2: Die Definition auspacken: "Rest = o(h) laengs 𝓝 0"
  -- (`hasFDerivAt_iff_isLittleO_nhds_zero`), und o(h) heisst "zu jedem c > 0 schliesslich
  -- ‖Rest‖ ≤ c‖h‖" (`isLittleO_iff`). Sei also c > 0. Ziel:
  --   ∀ᶠ h in 𝓝 0, ‖M.mulVec (v + h) + b - (M.mulVec v + b) - matrixAbb M h‖ ≤ c * ‖h‖
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, Asymptotics.isLittleO_iff]
  intro c hc
  -- Schritt 3 (= 2. auf der Folie, null ist o(h)): Die Ungleichung gilt fuer jedes h, nicht
  -- erst auf einer Kugel; die Menge ist der ganze Raum. Was ueberall gilt, gilt schliesslich:
  -- Das Ganze ist gross (Axiom (i)), also auch jede Menge, die es enthaelt (Axiom (ii)). In
  -- Mathlib heisst das `Eventually.of_forall`. Sei h beliebig. Ziel:
  --   ‖M.mulVec (v + h) + b - (M.mulVec v + b) - matrixAbb M h‖ ≤ c * ‖h‖
  refine Eventually.of_forall fun h => ?_
  -- Schritt 4: Der Rest ist null (Schritt 1), seine Norm auch (`norm_zero`). Ziel:
  -- 0 ≤ c * ‖h‖, und das gilt, weil c > 0 und ‖h‖ ≥ 0 ist (`mul_nonneg`).
  rw [rest h, norm_zero]
  exact mul_nonneg hc.le (norm_nonneg h)

/- Abgleich mit Mathlib: Eine stetige lineare Abbildung ist ihre eigene Ableitung
   (`ContinuousLinearMap.hasFDerivAt`), und eine Konstante dazu aendert daran nichts
   (`HasFDerivAt.add_const`). -/
example {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (v : Fin n → ℝ) :
    HasFDerivAt (fun w : Fin n → ℝ => M.mulVec w + b) (matrixAbb M) v :=
  (matrixAbb M).hasFDerivAt.add_const b

-- Folie: "Partielle Ableitungen" (die Definition)
/- Punkte des ℝⁿ sind in Lean Funktionen `Fin n → ℝ`: Der Punkt v hat die Koordinaten
   v 0, v 1, ..., gezaehlt ab 0. `Pi.single j 1` ist der Einheitsvektor e_j (1 an der
   Stelle j, sonst 0); x + t • e_j bewegt also nur die Koordinate j. -/
/-- Partielle Ableitung, selbst definiert: f hat in x nach der j-ten Koordinate
die partielle Ableitung a, wenn die Schnittkurve t ↦ f(x + t·e_j) in 0 die
Ableitung a hat. Alle anderen Koordinaten bleiben dabei fest. -/
def HatPartielleAbleitung {n : ℕ} {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : (Fin n → ℝ) → F) (x : Fin n → ℝ) (j : Fin n) (a : F) : Prop :=
  HasDerivAt (fun t : ℝ => f (x + t • (Pi.single j 1 : Fin n → ℝ))) a 0

/- Mathlib hat denselben Begriff: `HasLineDerivAt` ist die Richtungsableitung, mit v = e_j
   steht dort wortgleich unsere Definition. -/
example {n : ℕ} {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : (Fin n → ℝ) → F) (x : Fin n → ℝ) (j : Fin n) (a : F) :
    HatPartielleAbleitung f x j a ↔ HasLineDerivAt ℝ f a x (Pi.single j 1) := Iff.rfl

-- Folie: "Partielle Ableitung: eine Ableitung in einer Variablen" (das Beispiel x²y nach x)
/- Das Beispiel der Folie, f(x, y) = x²y nach x, von Hand: Die Schnittkurve ist
   t ↦ (x + t)²·y, ihr Fehler der Tangente bei 0 ist t²·y, und der ist o(t): Zu c > 0
   reicht die Kugel |t| < c/(|y| + 1), denn dort ist |t²·y| = |t|·|t|·|y| ≤ c·|t|. -/
example (v : Fin 2 → ℝ) :
    HatPartielleAbleitung (fun w : Fin 2 → ℝ => w 0 ^ 2 * w 1) v 0 (2 * v 0 * v 1) := by
  -- Schritt 1: die Definition. Die Schnittkurve t ↦ f(v + t·e₀) soll in 0 die
  -- Ableitung 2·v₀·v₁ haben.
  unfold HatPartielleAbleitung
  -- Schritt 2: die Schnittkurve ausrechnen. v + t·e₀ = (v₀ + t, v₁), nur die
  -- Koordinate 0 bewegt sich (`simp` liest die Koordinaten ab): t ↦ (v₀ + t)²·v₁.
  have e : (fun t : ℝ => (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 0 ^ 2 *
      (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 1) = fun t => (v 0 + t) ^ 2 * v 1 := by
    funext t
    simp
  rw [e]
  -- Schritt 3: die Definition der Ableitung (`ableitung_klein_o`): Der Fehler der
  -- Tangente soll o(t - 0) sein. Zu c > 0 nehmen wir die Kugel |t| < c/(|v₁| + 1).
  rw [ableitung_klein_o]
  intro c hc
  filter_upwards [Metric.ball_mem_nhds (0 : ℝ) (by positivity : 0 < c / (|v 1| + 1))]
    with t ht
  rw [Metric.mem_ball, Real.dist_eq, sub_zero, lt_div_iff₀ (by positivity)] at ht
  -- Schritt 4: Der Fehler ist (v₀ + t)²v₁ - v₀²v₁ - t·2v₀v₁ = t·t·v₁, sein Betrag
  -- |t|·|t|·|v₁|. Das ist ≤ c·|t|, denn |t|·|v₁| ≤ |t|·(|v₁| + 1) < c (ht).
  calc |(v 0 + t) ^ 2 * v 1 - (v 0 + 0) ^ 2 * v 1 - (t - 0) * (2 * v 0 * v 1)|
      = |t| * |t| * |v 1| := by
        rw [← abs_mul, ← abs_mul]
        congr 1
        ring
    _ ≤ c * |t - 0| := by
        rw [sub_zero]
        nlinarith [abs_nonneg t, abs_nonneg (v 1)]

-- Folie: "Richtungsableitung: die Schnittkurve in Richtung v", "Beweis von (a): die Idee",
-- "Beweis von (a): die Details (1) -- h = tv einsetzen" und
-- "Beweis von (a): die Details (2) -- den Faktor waehlen"
/-- Satz, Teil (a), zu Fuss: Ist f in x differenzierbar mit der Ableitung f', so hat
die Schnittkurve t ↦ f(x + t·v) in 0 die Ableitung f' v. Der Beweis der Folie, mit
den Faktoren c/(‖v‖ + 1) und δ/(‖v‖ + 1) statt c/‖v‖ und δ/‖v‖ -- das + 1, damit
auch v = 0 erlaubt ist. -/
theorem richtungsableitung_zu_fuss {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F}
    {f' : E →L[ℝ] F} {x : E} (hf : HasFDerivAt f f' x) (v : E) :
    HasDerivAt (fun t : ℝ => f (x + t • v)) (f' v) 0 := by
  -- Gegeben: Der Rest r(h) = f(x + h) - f(x) - f' h ist o(h) laengs 𝓝 0, also zu jedem
  -- c' > 0 schliesslich ‖r(h)‖ ≤ c'·‖h‖ (`hasFDerivAt_iff_isLittleO_nhds_zero`,
  -- `Asymptotics.isLittleO_iff`).
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, Asymptotics.isLittleO_iff] at hf
  -- Gesucht: zu jedem c > 0 schliesslich ‖f(x + t·v) - f(x) - t·f' v‖ ≤ c·|t|.
  rw [hasDerivAt_iff_isLittleO_nhds_zero, Asymptotics.isLittleO_iff]
  -- 1. Das Ziel gibt c > 0 vor.
  intro c hc
  -- 2. Man waehlt c' = c/(‖v‖ + 1) > 0.
  have hc' : 0 < c / (‖v‖ + 1) := by positivity
  -- 3. Die Voraussetzung liefert eine Kugel: ein δ > 0 mit ‖r(h)‖ ≤ c'·‖h‖ fuer ‖h‖ < δ.
  obtain ⟨δ, hδ, hr⟩ := Metric.eventually_nhds_iff.1 (hf hc')
  -- 4. Man setzt fuer t den Radius δ/(‖v‖ + 1).
  rw [Metric.eventually_nhds_iff]
  refine ⟨δ / (‖v‖ + 1), by positivity, fun t ht => ?_⟩
  rw [Real.dist_eq, sub_zero, lt_div_iff₀ (by positivity)] at ht   -- |t|·(‖v‖ + 1) < δ
  -- 5. h = t·v liegt in der Kugel: ‖t·v‖ = |t|·‖v‖ ≤ |t|·(‖v‖ + 1) < δ.
  have htv : dist (t • v) 0 < δ := by
    rw [dist_zero_right, norm_smul, Real.norm_eq_abs]
    nlinarith [abs_nonneg t, norm_nonneg v]
  -- Also gilt die Voraussetzung mit h = t·v. f' ist linear: f'(t·v) = t·f' v (`map_smul`).
  have h := hr htv
  rw [map_smul] at h
  simp only [zero_add, zero_smul, add_zero]       -- im Ziel: x + 0·v = x und 0 + t = t
  -- Zum Schluss: c'·‖t·v‖ = c·|t|·‖v‖/(‖v‖ + 1) ≤ c·|t|.
  calc ‖f (x + t • v) - f x - t • f' v‖
      ≤ c / (‖v‖ + 1) * ‖t • v‖ := h
    _ = c * ‖t‖ * (‖v‖ / (‖v‖ + 1)) := by rw [norm_smul]; ring
    _ ≤ c * ‖t‖ * 1 := by
        gcongr
        rw [div_le_one (by positivity)]
        linarith
    _ = c * ‖t‖ := mul_one _

-- Folie: "Ableitung und partielle Ableitungen" (Teil (a): die Spalte j der Jacobi-Matrix)
/-- Teil (a) fuer v = e_j: Die partielle Ableitung nach x_j existiert und ist f' e_j,
die Spalte j der Jacobi-Matrix. -/
theorem partielle_ableitung_spalte {n : ℕ} {F : Type*} [NormedAddCommGroup F]
    [NormedSpace ℝ F] {f : (Fin n → ℝ) → F} {f' : (Fin n → ℝ) →L[ℝ] F}
    {x : Fin n → ℝ} (hf : HasFDerivAt f f' x) (j : Fin n) :
    HatPartielleAbleitung f x j (f' (Pi.single j 1)) :=
  richtungsableitung_zu_fuss hf _                   -- mit v = e_j

/- Mathlib hat Teil (a) als `HasFDerivAt.hasLineDerivAt`; die Richtungsableitung heisst
   dort `HasLineDerivAt`. -/
example {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F} {f' : E →L[ℝ] F} {x : E}
    (hf : HasFDerivAt f f' x) (v : E) : HasLineDerivAt ℝ f (f' v) x v :=
  hf.hasLineDerivAt v

-- Folie: "Partielle Ableitungen allein reichen nicht"
/-- f(x, y) = xy/(x² + y²). In Lean ist a/0 = 0, also f(0, 0) = 0 von selbst. -/
noncomputable def gegenbeispiel (w : Fin 2 → ℝ) : ℝ :=
  w 0 * w 1 / (w 0 ^ 2 + w 1 ^ 2)

/- Auf der x-Achse ist f null, die partielle Ableitung nach x in 0 ist also 0;
   ebenso nach y. -/
example : HatPartielleAbleitung gegenbeispiel 0 0 0 := by
  -- Schritt 1: Die Definition einsetzen: Die Schnittkurve t ↦ f(0 + t·e₀) = f(t, 0) soll in
  -- 0 die Ableitung 0 haben.
  unfold HatPartielleAbleitung
  -- Schritt 2: Die Schnittkurve ist die Nullfunktion, denn auf der x-Achse ist y = 0 und
  -- t·0/(t² + 0²) = 0 (in Lean sogar fuer t = 0, wegen a/0 = 0). `funext t` nimmt ein
  -- beliebiges t, `simp` rechnet f(t, 0) aus.
  have h : (fun t : ℝ => gegenbeispiel (0 + t • (Pi.single 0 1 : Fin 2 → ℝ))) =
      fun _ => 0 := by
    funext t
    simp [gegenbeispiel]
  rw [h]
  -- Schritt 3: Eine konstante Funktion hat die Ableitung 0 (`hasDerivAt_const`).
  exact hasDerivAt_const 0 0

/-- Auf der Diagonalen ist f = 1/2, ausser in 0. -/
lemma gegenbeispiel_diagonale {t : ℝ} (ht : t ≠ 0) :
    gegenbeispiel ![t, t] = 1 / 2 := by
  -- Schritt 1: f(t, t) ausschreiben, die Koordinaten von ![t, t] einsetzen. Ziel:
  --   t * t / (t ^ 2 + t ^ 2) = 1 / 2
  simp only [gegenbeispiel, Matrix.cons_val_zero, Matrix.cons_val_one]
  -- Schritt 2: `field_simp` multipliziert die Nenner weg (t ≠ 0, also auch t² + t² ≠ 0);
  -- uebrig bleibt eine Gleichung zwischen Polynomen in t, die `ring` nachrechnet.
  field_simp
  ring

/-- f ist in 0 nicht einmal stetig, mit der ε-δ-Definition (`konvergenz_eps_delta`
mit b = f 0): Zu ε = 1/2 gibt es kein δ. Der Punkt (δ/2, δ/2) liegt naeher als δ
bei 0, aber dort ist f = 1/2, waehrend f(0) = 0 ist. -/
theorem gegenbeispiel_unstetig : ¬ ContinuousAt gegenbeispiel 0 := by
  intro h
  -- Schritt 1: stetig in 0 heisst: Zu ε = 1/2 gibt es ein δ > 0 mit
  -- d(f w, f 0) < 1/2 fuer d(w, 0) < δ.
  obtain ⟨δ, hδ, hd⟩ :=
    (konvergenz_eps_delta gegenbeispiel 0 (gegenbeispiel 0)).1 h (1 / 2) (by norm_num)
  -- Schritt 2: Der Punkt w = (δ/2, δ/2) liegt naeher als δ bei 0. In Lean ist der Abstand
  -- in ℝ² der groesste Koordinatenabstand (`dist_pi_lt_iff`): zweimal |δ/2| < δ.
  have hw : dist (![δ / 2, δ / 2] : Fin 2 → ℝ) 0 < δ := by
    rw [dist_pi_lt_iff hδ]
    intro i
    fin_cases i <;> simp [abs_of_pos hδ] <;> linarith
  -- Schritt 3: Dort ist f = 1/2 (`gegenbeispiel_diagonale`), und f(0) = 0. Also waere
  -- |1/2 - 0| < 1/2 -- Widerspruch.
  have h1 := hd _ hw
  rw [gegenbeispiel_diagonale (by positivity)] at h1
  norm_num [gegenbeispiel, Real.dist_eq] at h1

/- Differenzierbar heisst erst recht stetig (in Mathlib `DifferentiableAt.continuousAt`).
   Also ist f in 0 nicht differenzierbar -- obwohl beide partiellen Ableitungen existieren. -/
example : ¬ DifferentiableAt ℝ gegenbeispiel 0 :=
  fun h => gegenbeispiel_unstetig h.continuousAt

-- Folie: "Vorgerechnet: f(x, y) = (xy, x + y)" (zu Fuss nach der Definition: Differenz,
-- linearer Teil und Rest, den Rest abschaetzen)
/-- (xy, x + y) zu Fuss, nach der Definition: Der Rest ist (h₀h₁, 0), und
|h₀h₁| ≤ ‖h‖·‖h‖, denn jede Koordinate ist hoechstens ‖h‖ (in Lean traegt ℝ²
die Maximumsnorm). Auf der Kugel vom Radius c ist das hoechstens c·‖h‖. -/
theorem jacobi_xy_zu_fuss (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 * w 1, w 0 + w 1])
      (matrixAbb !![v 1, v 0; 1, 1]) v := by
  -- Schritt 1: Die Definition auspacken: "Rest = o(h) laengs 𝓝 0"
  -- (`hasFDerivAt_iff_isLittleO_nhds_zero`), und o(h) heisst "zu jedem c > 0 schliesslich
  -- ‖Rest‖ ≤ c‖h‖" (`isLittleO_iff`). Sei also c > 0. Ziel:
  --   ∀ᶠ h in 𝓝 0, ‖f(v + h) - f(v) - J h‖ ≤ c * ‖h‖
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, Asymptotics.isLittleO_iff]
  intro c hc
  -- Schritt 2: "schliesslich" ist hier "auf der Kugel ‖h‖ < c" -- wie in `quadrat_klein_o`
  -- (`filter_upwards` mit `ball_mem_nhds`). hh : h ∈ B(0, c), also ‖h‖ < c.
  filter_upwards [Metric.ball_mem_nhds (0 : Fin 2 → ℝ) hc] with h hh
  rw [Metric.mem_ball, dist_zero_right] at hh
  -- Schritt 3 (= 1. und 2. auf der Folie, Differenz, linearer Teil und Rest): Der Rest
  -- f(v + h) - f(v) - J h ist (h₀h₁, 0). Komponentenweise nachrechnen: `ext i` greift sich
  -- die Koordinate i, `fin_cases i` macht daraus die Faelle 0 und 1, `simp` schreibt
  -- "Matrix mal Vektor" als Summe aus, `ring` rechnet aus.
  have rest : ![(v + h) 0 * (v + h) 1, (v + h) 0 + (v + h) 1] - ![v 0 * v 1, v 0 + v 1]
      - matrixAbb !![v 1, v 0; 1, 1] h = ![h 0 * h 1, 0] := by
    ext i
    fin_cases i <;> simp [dotProduct, Fin.sum_univ_two] <;> ring
  rw [rest]                                      -- Ziel: ‖(h₀h₁, 0)‖ ≤ c * ‖h‖
  -- Schritt 4 (= 3. auf der Folie, den Rest abschaetzen): Jede Koordinate ist hoechstens so
  -- gross wie die Norm von h (`norm_le_pi_norm`; in Lean traegt ℝ² die Maximumsnorm):
  have h0 : |h 0| ≤ ‖h‖ := by simpa using norm_le_pi_norm h 0
  have h1 : |h 1| ≤ ‖h‖ := by simpa using norm_le_pi_norm h 1
  -- Schritt 5: ‖(h₀h₁, 0)‖ ≤ ‖h‖·‖h‖. Eine Maximumsnorm ist ≤ s, wenn jede Koordinate
  -- betragsmaessig ≤ s ist (`pi_norm_le_iff_of_nonneg`, s = ‖h‖·‖h‖ ≥ 0), also zwei Faelle:
  have hr : ‖(![h 0 * h 1, 0] : Fin 2 → ℝ)‖ ≤ ‖h‖ * ‖h‖ := by
    refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun i => ?_
    fin_cases i
    -- Koordinate 0: |h₀h₁| = |h₀|·|h₁| ≤ ‖h‖·‖h‖ (beide Faktoren abschaetzen, `mul_le_mul`).
    · simpa [abs_mul] using mul_le_mul h0 h1 (abs_nonneg _) (norm_nonneg _)
    -- Koordinate 1: |0| = 0 ≤ ‖h‖·‖h‖.
    · simpa using mul_nonneg (norm_nonneg h) (norm_nonneg h)
  -- Schritt 6: Einen Faktor ‖h‖ durch das groessere c ersetzen (hh : ‖h‖ < c). Auf der Folie
  -- steht ‖h‖²/2 ≤ c‖h‖ fuer ‖h‖ ≤ 2c; hier genuegt die groebere Schranke ‖h‖² ≤ c‖h‖
  -- fuer ‖h‖ < c.
  calc ‖(![h 0 * h 1, 0] : Fin 2 → ℝ)‖ ≤ ‖h‖ * ‖h‖ := hr
    _ ≤ c * ‖h‖ := mul_le_mul_of_nonneg_right hh.le (norm_nonneg h)

-- Folie: "Die Kettenregel: Matrizen multiplizieren"
/-- Kettenregel als Matrixprodukt: Hat f in x die Jacobi-Matrix A und g in f(x) die
Jacobi-Matrix B, so hat g ∘ f in x die Jacobi-Matrix B·A. Die Kettenregel selbst
beweisen die Folien von Hand; Lean zitiert sie als `HasFDerivAt.comp`. Von Hand
bewiesen ist hier der zweite Teil: Hintereinanderausfuehren ist Matrizen
multiplizieren (`matrixAbb_comp`). -/
theorem kettenregel_matrix {k n m : ℕ} {f : (Fin k → ℝ) → (Fin n → ℝ)}
    {g : (Fin n → ℝ) → (Fin m → ℝ)} {x : Fin k → ℝ}
    {A : Matrix (Fin n) (Fin k) ℝ} {B : Matrix (Fin m) (Fin n) ℝ}
    (hf : HasFDerivAt f (matrixAbb A) x) (hg : HasFDerivAt g (matrixAbb B) (f x)) :
    HasFDerivAt (g ∘ f) (matrixAbb (B * A)) x := by
  rw [← matrixAbb_comp]              -- B·A ist "erst A, dann B"
  exact hg.comp x hf                 -- die Kettenregel: D(g ∘ f)(x) = Dg(f x) ∘ Df(x)

-- Folie: "Die Kettenregel: Matrizen multiplizieren" (ein Beispiel, ganz aus Stuecken von Hand)
/- f(x, y) = (xy, x + y) wie auf der Folie, J_f von Hand (`jacobi_xy_zu_fuss`), danach
   g(u, v) = u + v, affin mit J_g = (1, 1) (`ableitung_affin`). Die Kettenregel
   multipliziert: J_(g ∘ f)(x, y) = (1, 1) · [[y, x], [1, 1]] = (y + 1, x + 1). -/
example (v : Fin 2 → ℝ) :
    HasFDerivAt ((fun u : Fin 2 → ℝ => (!![1, 1] : Matrix (Fin 1) (Fin 2) ℝ).mulVec u + 0)
        ∘ (fun w : Fin 2 → ℝ => ![w 0 * w 1, w 0 + w 1]))
      (matrixAbb ((!![1, 1] : Matrix (Fin 1) (Fin 2) ℝ) * !![v 1, v 0; 1, 1])) v :=
  kettenregel_matrix (jacobi_xy_zu_fuss v) (ableitung_affin !![1, 1] 0 _)

-- Folie: "Zusammenfassung" (die Zahl f'(x) ist in Mathlib total)
/- Die Zahl `deriv f x` ist total: Wo f nicht differenzierbar ist, liefert Mathlib den
   Junk-Wert 0. Die inhaltliche Aussage ist `HasDerivAt f f' x`. -/
example (f : ℝ → ℝ) (x : ℝ) (h : ¬ DifferentiableAt ℝ f x) : deriv f x = 0 :=
  deriv_zero_of_not_differentiableAt h

/-! ### Mathlib-Wegweiser zu Vorlesung 5

Nur zum Nachschlagen, ohne Folie. -/
#check @Metric.tendsto_nhds_nhds                -- ε-δ
#check @Metric.tendsto_atTop                    -- ε-N
#check @hasSum_geometric_two'                   -- 1/2 + 1/4 + ... = 1 (mit a = 1)
#check @Asymptotics.isLittleO_iff               -- o(g) ausgeschrieben
#check @Asymptotics.isLittleO_iff_tendsto'      -- mit Teilen, g schliesslich ≠ 0
#check @Asymptotics.IsLittleO.add               -- Summenregel
#check @hasDerivAt_iff_isLittleO                -- Ableitung: Fehler = o(y - x)
#check @IsLocalMin.hasDerivAt_eq_zero           -- Fermat
#check @hasFDerivAt_iff_isLittleO_nhds_zero     -- Frechet: Rest = o(h)
#check @HasFDerivAt.comp                        -- Kettenregel
#check @HasFDerivAt.hasLineDerivAt              -- Richtungsableitung

end Skript.Analysis
