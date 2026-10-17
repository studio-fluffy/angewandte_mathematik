/-
  Angewandte Mathematik -- Begleitcode zum Skript
  Kapitel 4: Infinitesimalrechnung

  Aufbau wie bei `MeinNat` in Kapitel 3: Die tragenden Begriffe --
  Topologie, offene Menge, Filter, Konvergenz, Cauchy-Folge, Kontraktion --
  werden zuerst selbst gebaut und ihre entscheidenden Eigenschaften von Hand
  bewiesen. Danach zeigt ein Abgleich, dass es genau Mathlibs Begriffe sind;
  ab da wird Mathlib benutzt. Landau-Symbole und Ableitung kommen direkt aus
  Mathlib; die Kernsaetze darueber, auch Fermat, werden trotzdem von Hand
  bewiesen, Rechenregeln wie die Produktregel nur zitiert. Weitere
  Mathlib-Namen stehen im "Mathlib-Wegweiser" am Ende jeder Vorlesung --
  nur hier in der Datei.

  Zuordnung zu den Vorlesungen. Jeder Teil beginnt mit einer Kopfzeile
  "VORLESUNG n" zwischen zwei Linien aus Gleichheitszeichen.
    Vorlesung 4  Skript 4.1-4.2: Raeume, Konvergenz durch Filter
                 (Folien: folien/vorlesung04.tex)
    Vorlesung 5  Skript 4.3-4.6: Vollstaendigkeit, Ableitung,
                 Gradientenabstieg, Lagrange (Folien: folien/vorlesung05.tex)
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

-- =============================================================================
-- VORLESUNG 4: Raeume und Konvergenz
--   Skript 4.1-4.2
-- =============================================================================

/-! ## 4.1 Raeume -/

--<<beispieleTop>>
/-- Eine Topologie: ein System offener Mengen mit den Axiomen (i)-(iii).
`Set.sUnion S` ist die Vereinigung aller Mengen der Familie `S`. -/
structure Topologie (X : Type*) where
  offen : Set (Set X)
  leer_offen : ∅ ∈ offen                                            -- (i)
  univ_offen : Set.univ ∈ offen                                     -- (i)
  vereinigung {S : Set (Set X)} : S ⊆ offen → Set.sUnion S ∈ offen  -- (ii)
  schnitt {U V : Set X} : U ∈ offen → V ∈ offen → U ∩ V ∈ offen     -- (iii)

/-- Diskret: Alle Mengen sind offen, jedes Axiom gilt trivial. -/
def diskret (X : Type*) : Topologie X where
  offen := Set.univ                       -- alle Teilmengen von X
  leer_offen := trivial
  univ_offen := trivial
  vereinigung _ := trivial
  schnitt _ _ := trivial
--<<beispieleTopEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<indiskret>>
/-- Indiskret: Nur ∅ und X sind offen. -/
def indiskret (X : Type*) : Topologie X where
  offen := {U | U = ∅ ∨ U = Set.univ}
  leer_offen := Or.inl rfl
  univ_offen := Or.inr rfl
  vereinigung := by
    intro S hS                            -- jede Menge in S ist ∅ oder X
    rcases Classical.em (Set.univ ∈ S) with h | h  -- X in S oder nicht
    · -- X liegt in S: Jedes x liegt in X, also in der Vereinigung.
      right
      exact Set.eq_univ_of_forall fun x => ⟨Set.univ, h, trivial⟩
    · -- X fehlt: Dann ist jede Menge in S leer, also auch die Vereinigung.
      left
      rw [Set.sUnion_eq_empty]
      intro U hU
      rcases hS hU with h0 | rfl          -- U = ∅ oder U = X
      · exact h0
      · contradiction                     -- X liegt aber nicht in S
  schnitt := by                           -- U, V je ∅ oder X: vier Faelle
    rintro U V (rfl | rfl) (rfl | rfl) <;> simp

/- Die Menge {7} ist offen in `diskret ℕ`, aber nicht in `indiskret ℕ`:
   Sie ist weder leer noch ganz ℕ. -/
example : {7} ∈ (diskret ℕ).offen := trivial
example : {7} ∉ (indiskret ℕ).offen := by
  rintro (h | h)                          -- {7} = ∅ oder {7} = ℕ
  · exact Set.singleton_ne_empty 7 h      -- nein: 7 liegt in {7}
  · have h8 : 8 ∈ ({7} : Set ℕ) := by rw [h]; trivial  -- 8 ∈ ℕ = {7}
    simp at h8                            -- also 8 = 7: Widerspruch
--<<indiskretEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<topologieBruecke>>
/- Mathlibs `TopologicalSpace` hat die Felder `IsOpen`, `isOpen_univ`,
   `isOpen_inter` und `isOpen_sUnion`, aber keines fuer ∅. Das folgt aus
   (ii): ∅ ist die Vereinigung der leeren Familie. -/
example {X : Type*} (T : Topologie X) : ∅ ∈ T.offen := by
  have h := T.vereinigung (Set.empty_subset T.offen)  -- (ii), leere Familie
  rwa [Set.sUnion_empty] at h                         -- ihre Vereinigung: ∅

/-- Feld fuer Feld wird eine `Topologie` eine Mathlib-Topologie;
`leer_offen` wird dafuer nicht gebraucht. -/
def Topologie.toTopologicalSpace {X : Type*} (T : Topologie X) :
    TopologicalSpace X where
  IsOpen U := U ∈ T.offen
  isOpen_univ := T.univ_offen
  isOpen_inter _ _ := T.schnitt
  isOpen_sUnion _ := T.vereinigung

/- Mathlib ordnet Topologien so, dass mehr offene Mengen kleiner heisst. Die
   diskrete hat alle Mengen offen, ist also `⊥` -- sogar per Definition: -/
example (X : Type*) : (diskret X).toTopologicalSpace = ⊥ := rfl
--<<topologieBrueckeEnd>>

--<<beispieleMetrik>>
/- Auf ℝ ist der Abstand per Definition |x - y| (`Real.dist_eq`). -/
example (x y : ℝ) : dist x y = |x - y| := rfl
example : dist (3:ℝ) 7 = 4 := by rw [Real.dist_eq]; norm_num

/- In der Ebene ist es der Satz des Pythagoras. -/
example (x y : EuclideanSpace ℝ (Fin 2)) :
    dist x y = √((x 0 - y 0) ^ 2 + (x 1 - y 1) ^ 2) := by
  rw [EuclideanSpace.dist_eq]
  simp [Fin.sum_univ_two, Real.dist_eq, sq_abs]

/-- U heisst offen, wenn um jeden Punkt von U eine Kugel ganz in U passt. -/
def Offen {X : Type*} [MetricSpace X] (U : Set X) : Prop :=
  ∀ x ∈ U, ∃ ε > 0, Metric.ball x ε ⊆ U

/- (0,1) ist offen: Um x passt die Kugel vom Radius r = min(x, 1 - x),
   dem Abstand zum naeheren Rand. -/
example : Offen (Set.Ioo (0:ℝ) 1) := by
  rintro x ⟨h0, h1⟩                                 -- 0 < x < 1
  use min x (1 - x), lt_min h0 (by linarith)        -- r = min(x, 1 - x) > 0
  intro y hy
  rw [Metric.mem_ball, Real.dist_eq, abs_lt] at hy  -- -r < y - x < r
  constructor
  · linarith [hy.1, min_le_left x (1 - x)]          -- 0 ≤ x - r < y
  · linarith [hy.2, min_le_right x (1 - x)]         -- y < x + r ≤ 1

/- Erst mit der Metrik laesst sich sagen, warum [0,1] nicht offen ist:
   um 0 passt keine Kugel hinein. Diesmal mit Mathlibs `IsOpen`;
   `Metric.isOpen_iff` uebersetzt es in die Bedingung von `Offen`. -/
example : ¬ IsOpen (Set.Icc (0:ℝ) 1) := by
  intro h
  obtain ⟨ε, hε, hsub⟩ := Metric.isOpen_iff.1 h 0 (by norm_num)
  have h1 : (-(ε / 2) : ℝ) ∈ Metric.ball (0:ℝ) ε := by
    rw [Metric.mem_ball, Real.dist_eq, abs_of_nonpos (by linarith)]
    linarith
  have h2 := hsub h1
  rw [Set.mem_Icc] at h2
  linarith [h2.1]
--<<beispieleMetrikEnd>>

--<<beispieleNorm>>
/- Auf ℝ ist die Laenge per Definition der Betrag (`Real.norm_eq_abs`). -/
example (x : ℝ) : ‖x‖ = |x| := rfl

/- In der Ebene: Laenge ueber die Koordinaten, und ein Zahlenbeispiel. -/
example (x : EuclideanSpace ℝ (Fin 2)) : ‖x‖ = √(x 0 ^ 2 + x 1 ^ 2) := by
  rw [EuclideanSpace.norm_eq]
  simp [Fin.sum_univ_two]

example : ‖(!₂[(3:ℝ), 4] : EuclideanSpace ℝ (Fin 2))‖ = 5 := by
  rw [EuclideanSpace.norm_eq]
  simp [Fin.sum_univ_two]
  rw [show (3:ℝ) ^ 2 + 4 ^ 2 = 5 ^ 2 by norm_num]
  exact Real.sqrt_sq (by norm_num)

/- Auch Funktionen sind Vektoren: die stetigen Funktionen auf [0,1] mit der
   Supremumsnorm. Nur zitiert -- das Maximum braucht Kompaktheit. -/
#check (inferInstance : NormedAddCommGroup C(Set.Icc (0:ℝ) 1, ℝ))
example (f : C(Set.Icc (0:ℝ) 1, ℝ)) (x) : ‖f x‖ ≤ ‖f‖ := f.norm_coe_le_norm x
--<<beispieleNormEnd>>

section Innenprodukt

--<<beispieleInner>>
/- Das Skalarprodukt in Koordinaten. -/
example (x y : EuclideanSpace ℝ (Fin 2)) :
    inner ℝ x y = x 0 * y 0 + x 1 * y 1 := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, mul_comm]

/- Die beiden Einheitsvektoren stehen senkrecht aufeinander. -/
example : inner ℝ (EuclideanSpace.single 0 (1:ℝ) : EuclideanSpace ℝ (Fin 2))
    (EuclideanSpace.single 1 (1:ℝ)) = (0:ℝ) := by
  simp [EuclideanSpace.inner_single_left, EuclideanSpace.single_apply]

/- Ab hier ist V ein beliebiger reeller Innenproduktraum. -/
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/- Pythagoras fuer senkrechte Vektoren. `norm_add_sq_real` ist die Rechnung
   aus den Axiomen: ‖x + y‖² = ‖x‖² + 2⟨x,y⟩ + ‖y‖². -/
example (x y : V) (h : inner ℝ x y = 0) :
    ‖x + y‖ ^ 2 = ‖x‖ ^ 2 + ‖y‖ ^ 2 := by
  rw [norm_add_sq_real, h]; ring

/-- Cauchy-Schwarz, von Hand: ‖x + t • y‖² ist ein Polynom in t, das nie
negativ wird. Also ist seine Diskriminante ≤ 0 (`discrim_le_zero`). -/
theorem cauchy_schwarz (x y : V) : inner ℝ x y ^ 2 ≤ ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
  have h : ∀ t : ℝ, 0 ≤ ‖y‖ ^ 2 * (t * t) + 2 * inner ℝ x y * t + ‖x‖ ^ 2 := by
    intro t                                 -- a = ‖y‖², b = 2⟨x,y⟩, c = ‖x‖²
    have h0 := sq_nonneg ‖x + t • y‖        -- 0 ≤ ‖x + t • y‖²
    rw [norm_add_sq_real, real_inner_smul_right, norm_smul, mul_pow,
      Real.norm_eq_abs, sq_abs] at h0       -- 0 ≤ ‖x‖² + 2t⟨x,y⟩ + t²‖y‖²
    linarith                                -- nur umgeordnet
  have hd : (2 * inner ℝ x y) ^ 2 - 4 * ‖y‖ ^ 2 * ‖x‖ ^ 2 ≤ 0 :=
    discrim_le_zero h                       -- b² - 4ac ≤ 0
  linarith                                  -- durch 4 teilen

/- Die Fassung mit Betrag; in Mathlib heisst sie `abs_real_inner_le_norm`. -/
example (x y : V) : |inner ℝ x y| ≤ ‖x‖ * ‖y‖ := by
  have h := cauchy_schwarz x y
  rw [← mul_pow] at h                        -- ⟨x,y⟩² ≤ (‖x‖ ‖y‖)²
  exact abs_le_of_sq_le_sq h (by positivity) -- Wurzel ziehen, ‖x‖ ‖y‖ ≥ 0
--<<beispieleInnerEnd>>

end Innenprodukt

/-! ### Wie eine Stufe die naechste liefert -/

section Stufen
variable {X : Type*} [MetricSpace X] {U V : Set X}

--<<stufen>>
/- Norm → Abstand: d(x,y) = ‖x - y‖. Die Dreiecksungleichung fuer d folgt
   aus der fuer die Norm, denn x - z = (x - y) + (y - z). -/
example {E : Type*} [NormedAddCommGroup E] (x y z : E) :
    dist x z ≤ dist x y + dist y z := by
  simp only [dist_eq_norm]                      -- d(x,y) = ‖x - y‖
  calc ‖x - z‖
      = ‖(x - y) + (y - z)‖ := by congr 1; abel  -- Umweg ueber y
    _ ≤ ‖x - y‖ + ‖y - z‖ := norm_add_le _ _    -- Dreiecksungleichung der Norm

/-- Abstand → Topologie, Axiom (iii): Der kleinere Radius passt in beide
Mengen. Bei unendlich vielen koennen die Radien gegen 0 gehen (etwa 1/n). -/
theorem Offen.inter (hU : Offen U) (hV : Offen V) : Offen (U ∩ V) := by
  rintro x ⟨hxU, hxV⟩
  obtain ⟨ε₁, h₁, hU'⟩ := hU x hxU          -- B(x, ε₁) ⊆ U
  obtain ⟨ε₂, h₂, hV'⟩ := hV x hxV          -- B(x, ε₂) ⊆ V
  use min ε₁ ε₂, lt_min h₁ h₂               -- der kleinere Radius
  intro y hy
  exact ⟨hU' (Metric.ball_subset_ball (min_le_left ε₁ ε₂) hy),
    hV' (Metric.ball_subset_ball (min_le_right ε₁ ε₂) hy)⟩

/- Abgleich mit Mathlib. Kein `Iff.rfl`: Mathlib gewinnt die Topologie
   eines metrischen Raums ueber die uniforme Struktur. Der Weg geht ueber
   Umgebungen (`U ∈ 𝓝 x`: U ist Umgebung von x): -/
example : Offen U ↔ IsOpen U := by
  rw [isOpen_iff_mem_nhds]          -- offen: Umgebung jedes seiner Punkte
  simp only [Metric.mem_nhds_iff]   -- Umgebung: enthaelt eine Kugel um x
  rfl                               -- genau die Definition von `Offen`
--<<stufenEnd>>

end Stufen

/-! ## 4.2 Konvergenz durch Filter -/

--<<meinFilter>>
/-- Ein Filter: ein System "grosser" Mengen mit den Axiomen (i)-(iii). -/
structure MeinFilter (α : Type*) where
  gross : Set (Set α)
  univ_gross : Set.univ ∈ gross                                 -- (i)
  obermenge {U V : Set α} : U ∈ gross → U ⊆ V → V ∈ gross       -- (ii)
  schnitt {U V : Set α} : U ∈ gross → V ∈ gross → U ∩ V ∈ gross  -- (iii)

/-- Die Endstuecke: gross ist, was ein Endstueck {n | n ≥ N} enthaelt. -/
def endstuecke : MeinFilter ℕ where
  gross := {U | ∃ N, ∀ n ≥ N, n ∈ U}
  univ_gross := ⟨0, fun n _ => Set.mem_univ n⟩
  obermenge := by
    rintro U V ⟨N, hN⟩ hUV
    exact ⟨N, fun n hn => hUV (hN n hn)⟩
  schnitt := by
    rintro U V ⟨N₁, h₁⟩ ⟨N₂, h₂⟩            -- ab N₁ in U, ab N₂ in V
    use max N₁ N₂                           -- (iii): das groessere N
    intro n hn
    exact ⟨h₁ n (le_of_max_le_left hn), h₂ n (le_of_max_le_right hn)⟩
--<<meinFilterEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<umgebungen>>
/-- Die Umgebungen von a: gross ist, was eine Kugel um a enthaelt. -/
def umgebungen {X : Type*} [MetricSpace X] (a : X) : MeinFilter X where
  gross := {U | ∃ ε > 0, Metric.ball a ε ⊆ U}
  univ_gross := ⟨1, one_pos, Set.subset_univ _⟩
  obermenge := by
    rintro U V ⟨ε, hε, h⟩ hUV
    exact ⟨ε, hε, h.trans hUV⟩
  schnitt := by
    rintro U V ⟨ε₁, h₁, hU⟩ ⟨ε₂, h₂, hV⟩
    use min ε₁ ε₂, lt_min h₁ h₂             -- (iii): das kleinere ε
    intro y hy
    exact ⟨hU (Metric.ball_subset_ball (min_le_left ε₁ ε₂) hy),
      hV (Metric.ball_subset_ball (min_le_right ε₁ ε₂) hy)⟩
--<<umgebungenEnd>>

section Konvergenz
variable {α β γ : Type*}

--<<meinTendsto>>
/-- Konvergenz laengs Filtern: Urbilder grosser Mengen sind gross. -/
def MeinTendsto (f : α → β) (F : MeinFilter α) (G : MeinFilter β) : Prop :=
  ∀ V ∈ G.gross, f ⁻¹' V ∈ F.gross

/-- Verkettung. Wie in Mathlib steht die aeussere Funktion vorn:
`hg.comp hf` fuer `g ∘ f`. -/
theorem MeinTendsto.comp {f : α → β} {g : β → γ} {F : MeinFilter α}
    {G : MeinFilter β} {H : MeinFilter γ} (hg : MeinTendsto g G H)
    (hf : MeinTendsto f F G) : MeinTendsto (g ∘ f) F H := by
  intro W hW                     -- W gross in H
  exact hf _ (hg W hW)           -- g⁻¹(W) gross in G, also f⁻¹(g⁻¹(W)) in F
--<<meinTendstoEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<filterBruecke>>
/- Mathlibs `Filter` hat dieselben vier Felder; ein `MeinFilter` wird Feld
   fuer Feld ein Mathlib-Filter. `Tendsto` ist dort dieselbe Definition: -/
def MeinFilter.toFilter (F : MeinFilter α) : Filter α :=
  ⟨F.gross, F.univ_gross, F.obermenge, F.schnitt⟩

example (f : α → β) (F : MeinFilter α) (G : MeinFilter β) :
    Tendsto f F.toFilter G.toFilter ↔ MeinTendsto f F G := Iff.rfl

/- Die beiden Beispiele sind Mathlibs `atTop` und `𝓝 a`: -/
theorem endstuecke_eq : endstuecke.toFilter = atTop :=
  Filter.ext fun _ => mem_atTop_sets.symm

theorem umgebungen_eq {X : Type*} [MetricSpace X] (a : X) :
    (umgebungen a).toFilter = 𝓝 a :=
  Filter.ext fun _ => Metric.mem_nhds_iff.symm
--<<filterBrueckeEnd>>

end Konvergenz

--<<epsNHand>>
/-- Die ε-N-Form, von Hand. Hin: Die Kugel um a ist gross, ihr Urbild
enthaelt ein Endstueck. Rueck: In jede grosse Menge um a passt eine Kugel. -/
theorem meinTendsto_iff_eps_N (u : ℕ → ℝ) (a : ℝ) :
    MeinTendsto u endstuecke (umgebungen a) ↔
      ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  constructor
  · intro h ε hε
    obtain ⟨N, hN⟩ := h (Metric.ball a ε) ⟨ε, hε, subset_rfl⟩
    exact ⟨N, hN⟩                -- u n ∈ B(a, ε) heisst |u n - a| < ε
  · rintro h V ⟨ε, hε, hV⟩
    obtain ⟨N, hN⟩ := h ε hε
    exact ⟨N, fun n hn => hV (hN n hn)⟩
--<<epsNHandEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<epsNMathlib>>
/-- Dieselbe Aussage fuer Mathlibs `Tendsto`: zurueckuebersetzen, fertig. -/
theorem tendsto_iff_eps_N (u : ℕ → ℝ) (a : ℝ) :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  rw [← endstuecke_eq, ← umgebungen_eq]
  exact meinTendsto_iff_eps_N u a

/- Mathlib hat die ε-N-Form schon, fuer Folgen in jedem metrischen Raum: -/
example (u : ℕ → ℝ) (a : ℝ) :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  rw [Metric.tendsto_atTop]
  simp [Real.dist_eq]
--<<epsNMathlibEnd>>

--<<schliesslich>>
/- "P gilt schliesslich" (`∀ᶠ`) heisst: Die Menge, auf der P gilt, ist
   gross -- per Definition. -/
example {α : Type*} (F : Filter α) (P : α → Prop) :
    (∀ᶠ x in F, P x) ↔ {x | P x} ∈ F := Iff.rfl

/- Fuer `atTop` also: ab einem N. Im Beispiel reicht N = 11. -/
example : ∀ᶠ n : ℕ in atTop, 10 < n :=
  eventually_atTop.2 ⟨11, fun n hn => by omega⟩
--<<schliesslichEnd>>

--<<summenregel>>
--<<summenregelSatz>>
/-- Summenregel, einmal fuer jeden Filter: ε halbieren wie immer -- aber statt
N₁ und N₂ zu maximieren, schneidet Axiom (iii) die beiden grossen Mengen. -/
theorem summenregel {α : Type*} {F : Filter α} {u v : α → ℝ} {a b : ℝ}
    (hu : Tendsto u F (𝓝 a)) (hv : Tendsto v F (𝓝 b)) :
    Tendsto (fun x => u x + v x) F (𝓝 (a + b)) := by
  rw [Metric.tendsto_nhds] at *
  intro ε hε
  have h1 := hu (ε / 2) (by linarith)    -- schliesslich |u x - a| < ε/2
  have h2 := hv (ε / 2) (by linarith)    -- schliesslich |v x - b| < ε/2
  have h := inter_mem h1 h2              -- (iii): beides zugleich
  apply mem_of_superset h                -- (ii): eine Obermenge genuegt
  rintro x ⟨hx₁, hx₂⟩
  rw [Set.mem_setOf_eq, Real.dist_eq] at hx₁ hx₂ ⊢
  calc |u x + v x - (a + b)| = |(u x - a) + (v x - b)| := by congr 1; ring
    _ ≤ |u x - a| + |v x - b| := abs_add_le _ _
    _ < ε := by linarith
--<<summenregelSatzEnd>>

/- Derselbe Satz fuer Folgen (`atTop`) und fuer Funktionen (`𝓝 1`): -/
example : Tendsto (fun n : ℕ => 2 + 1 / (n : ℝ)) atTop (𝓝 (2 + 0)) :=
  summenregel tendsto_const_nhds tendsto_one_div_atTop_nhds_zero_nat
example : Tendsto (fun x : ℝ => x + 3) (𝓝 1) (𝓝 (1 + 3)) :=
  summenregel tendsto_id tendsto_const_nhds
--<<summenregelEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<eindeutig>>
/-- Grenzwerte sind eindeutig, fuer jeden Filter, in dem ∅ nicht gross ist
(`NeBot`): Die Kugeln vom Radius d(a,b)/2 um a und b sind disjunkt. -/
theorem grenzwert_eindeutig {α X : Type*} [MetricSpace X] {F : Filter α}
    [F.NeBot] {u : α → X} {a b : X}
    (ha : Tendsto u F (𝓝 a)) (hb : Tendsto u F (𝓝 b)) : a = b := by
  by_contra hne
  have hd : 0 < dist a b / 2 := half_pos (dist_pos.2 hne)
  rw [Metric.tendsto_nhds] at ha hb
  obtain ⟨x, hxa, hxb⟩ := ((ha _ hd).and (hb _ hd)).exists   -- (iii), NeBot
  have h := dist_triangle_left a b (u x)   -- d(a,b) ≤ d(u x,a) + d(u x,b)
  linarith
--<<eindeutigEnd>>

/-! ### Mathlib-Wegweiser zu Vorlesung 4

Nur hier in der Datei, nicht im Skript und nicht auf den Folien: die Begriffe
der Vorlesung unter ihren Mathlib-Namen. Auf ℝ sind Metrik und Norm eigens
definiert und passen zusammen; Lean findet alle Instanzen selbst. -/
#check @TopologicalSpace            -- wie `Topologie`, ohne Feld fuer ∅
#check @isOpen_discrete             -- diskret: jede Menge ist offen
#check @isOpen_Ioo                  -- (0,1) ist offen
#check @isClosed_Icc                -- [0,1] ist abgeschlossen
#check @MetricSpace
#check @NormedAddCommGroup          -- Laenge
#check @NormedSpace                 -- Laenge und Skalare
#check @InnerProductSpace           -- Skalarprodukt
#check @real_inner_self_eq_norm_sq  -- Skalarprodukt → Norm
#check @abs_real_inner_le_norm      -- Cauchy-Schwarz
#check @dist_eq_norm                -- Norm → Abstand
#check @Metric.isOpen_iff           -- offen: um jeden Punkt eine Kugel
#print Filter                       -- dieselben Felder wie `MeinFilter`
#check @Filter.Tendsto.comp         -- Verkettung: `hg.comp hf`
#check @Filter.tendsto_def          -- Tendsto: Urbilder grosser Mengen
#check @Filter.eventually_atTop     -- ∀ᶠ in atTop: ab einem N
#check @Metric.tendsto_atTop        -- ε-N in jedem metrischen Raum
#check @Filter.Tendsto.add          -- Summenregel
#check @tendsto_nhds_unique         -- Grenzwerte sind eindeutig

-- =============================================================================
-- VORLESUNG 5: Vollstaendigkeit, Ableitung, Optimierung
--   Skript 4.3-4.6
-- =============================================================================

/-! ## 4.3 Vollstaendigkeit -/

section Vollstaendig
variable {X : Type*} [MetricSpace X]

--<<istCauchy>>
/-- Cauchy-Folge: Die Glieder kommen sich *untereinander* beliebig nahe --
ein Grenzwert kommt nicht vor. -/
def IstCauchy (u : ℕ → X) : Prop :=
  ∀ ε > 0, ∃ N, ∀ m ≥ N, ∀ n ≥ N, dist (u m) (u n) < ε

/- Mathlib definiert `CauchySeq` ueber Filter; fuer Folgen ist es dasselbe: -/
example (u : ℕ → X) : IstCauchy u ↔ CauchySeq u := Metric.cauchySeq_iff.symm

/- Vollstaendig (`CompleteSpace`) heisst: Jede Cauchy-Folge hat einen
   Grenzwert. Mehr liefert die Vollstaendigkeit nicht: -/
example [CompleteSpace X] (u : ℕ → X) (h : CauchySeq u) :
    ∃ x, Tendsto u atTop (𝓝 x) :=
  cauchySeq_tendsto_of_complete h
--<<istCauchyEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<konvergentCauchy>>
/-- Konvergent ⇒ Cauchy: ε halbieren, dann Dreiecksungleichung ueber a. -/
theorem IstCauchy.of_tendsto {u : ℕ → X} {a : X}
    (h : Tendsto u atTop (𝓝 a)) : IstCauchy u := by
  intro ε hε
  obtain ⟨N, hN⟩ := Metric.tendsto_atTop.1 h (ε / 2) (by linarith)
  use N
  intro m hm n hn
  have h1 := dist_triangle_right (u m) (u n) a  -- ≤ d(u m, a) + d(u n, a)
  linarith [hN m hm, hN n hn]
--<<konvergentCauchyEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<kontraktion>>
/-- Kontraktion: Abstaende schrumpfen mindestens um den Faktor K < 1. -/
structure Kontraktion (K : ℝ) (T : X → X) : Prop where
  nichtneg : 0 ≤ K
  klein : K < 1
  schrumpft : ∀ x y, dist (T x) (T y) ≤ K * dist x y

/-- Die Iteration x₀, x₁ = T x₀, x₂ = T (T x₀), ... -- gebaut wie `gdIter`. -/
def iter {Y : Type*} (T : Y → Y) (x₀ : Y) : ℕ → Y
  | 0 => x₀
  | n + 1 => T (iter T x₀ n)
--<<kontraktionEnd>>

variable {K : ℝ} {T : X → X}

-- Nur im Skript, nicht auf den Folien.
--<<banachSchritte>>
/-- Schritt 1: d(x_n, x_(n+1)) ≤ d(x₀, x₁) · Kⁿ, wobei x_n = iter T x₀ n. -/
theorem Kontraktion.schritt1 (hT : Kontraktion K T) (x₀ : X) (n : ℕ) :
    dist (iter T x₀ n) (iter T x₀ (n + 1)) ≤ dist x₀ (T x₀) * K ^ n := by
  induction n with
  | zero => simp [iter]
  | succ k ih =>
      calc dist (iter T x₀ (k + 1)) (iter T x₀ (k + 2))
          ≤ K * dist (iter T x₀ k) (iter T x₀ (k + 1)) := hT.schrumpft _ _
        _ ≤ K * (dist x₀ (T x₀) * K ^ k) := by gcongr; exact hT.nichtneg
        _ = dist x₀ (T x₀) * K ^ (k + 1) := by ring

/-- Schritt 4: Der Grenzwert x ist ein Fixpunkt. Das Glied x_(N+1) = T x_N
liegt nahe bei x und nahe bei T x. -/
theorem Kontraktion.schritt4 (hT : Kontraktion K T) (x₀ : X) {x : X}
    (hx : Tendsto (iter T x₀) atTop (𝓝 x)) : T x = x := by
  apply eq_of_forall_dist_le
  intro ε hε
  obtain ⟨N, hN⟩ := Metric.tendsto_atTop.1 hx (ε / 2) (by linarith)
  have h1 := hN N le_rfl                   -- d(x_N, x) < ε/2
  have h2 := hN (N + 1) (by omega)         -- d(x_(N+1), x) < ε/2
  have h3 : dist (T x) (iter T x₀ (N + 1)) ≤ K * dist x (iter T x₀ N) :=
    hT.schrumpft x (iter T x₀ N)           -- x_(N+1) = T x_N
  have h4 := dist_triangle (T x) (iter T x₀ (N + 1)) x
  rw [dist_comm] at h1
  nlinarith [hT.klein, hT.nichtneg, dist_nonneg (x := x) (y := iter T x₀ N)]

/-- Schritt 5: hoechstens ein Fixpunkt -- aus d ≤ K·d und K < 1. -/
theorem Kontraktion.schritt5 (hT : Kontraktion K T) {x y : X}
    (hx : T x = x) (hy : T y = y) : x = y := by
  have h := hT.schrumpft x y
  rw [hx, hy] at h                         -- d(x,y) ≤ K·d(x,y)
  have h0 : dist x y ≤ 0 := by
    nlinarith [hT.klein, dist_nonneg (x := x) (y := y)]
  exact dist_le_zero.1 h0
--<<banachSchritteEnd>>

--<<banach>>
/-- Banachscher Fixpunktsatz: genau ein Fixpunkt, Konvergenz, Schranke. Der
Startwert x₀ ist "X nichtleer"; nur Schritt 3 braucht `[CompleteSpace X]`. -/
theorem banach [CompleteSpace X] (hT : Kontraktion K T) (x₀ : X) :
    ∃ x, T x = x ∧ (∀ y, T y = y → y = x) ∧ Tendsto (iter T x₀) atTop (𝓝 x) ∧
      ∀ n, dist (iter T x₀ n) x ≤ dist x₀ (T x₀) * K ^ n / (1 - K) := by
  have h2 : CauchySeq (iter T x₀) :=                       -- Schritt 2
    cauchySeq_of_le_geometric K _ hT.klein (hT.schritt1 x₀)
  have h3 : ∃ x, Tendsto (iter T x₀) atTop (𝓝 x) :=        -- Schritt 3
    cauchySeq_tendsto_of_complete h2
  obtain ⟨x, hx⟩ := h3
  have h4 : T x = x := hT.schritt4 x₀ hx                   -- Schritt 4
  refine ⟨x, h4, fun y hy => hT.schritt5 hy h4, hx, ?_⟩    -- Schritt 5
  intro n                                                  -- Schritt 6
  exact dist_le_of_le_geometric_of_tendsto K _ hT.klein (hT.schritt1 x₀) hx n
--<<banachEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<fixpunktMathlib>>
/- Mathlibs `ContractingWith K T` (K : NNReal) ist derselbe Begriff: -/
example (K : NNReal) (T : X → X) : Kontraktion K T ↔ ContractingWith K T := by
  rw [ContractingWith, lipschitzWith_iff_dist_le_mul]
  exact ⟨fun h => ⟨by exact_mod_cast h.klein, h.schrumpft⟩,
    fun h => ⟨K.2, by exact_mod_cast h.1, h.2⟩⟩

/- Mathlibs Satz, mit `f^[n] x` statt `iter T x₀ n`. Er gilt fuer erweiterte
   Metriken (`edist` darf ∞ sein, in Lean `⊤`) und setzt daher
   `edist x (f x) ≠ ⊤` voraus; im metrischen Raum ist das immer erfuellt: -/
#check @ContractingWith.exists_fixedPoint

/- Picard-Lindeloef sucht den Fixpunkt unter den stetigen Funktionen mit der
   Supremumsnorm -- ein vollstaendiger Raum, weil der Zielraum es ist: -/
#check (inferInstance : CompleteSpace C(Set.Icc (0:ℝ) 1, ℝ))
--<<fixpunktMathlibEnd>>

end Vollstaendig

-- Nur im Skript, nicht auf den Folien.
--<<projektion>>
/-- Projektionssatz, senkrecht ⇒ kuerzester Weg: Steht x - p senkrecht auf
p - u, so liegt p mindestens so nah an x wie u -- wieder Pythagoras. -/
theorem senkrecht_kuerzer {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (x p u : V)
    (h : inner ℝ (x - p) (p - u) = (0 : ℝ)) :
    ‖x - p‖ ^ 2 ≤ ‖x - u‖ ^ 2 := by
  have e : x - u = (x - p) + (p - u) := by abel
  rw [e, norm_add_sq_real, h]
  nlinarith [sq_nonneg ‖p - u‖]
--<<projektionEnd>>

/-! ## 4.4 Ableitung -/

--<<landau>>
/- o(g) laengs l, ausgeschrieben: fuer jedes c > 0 schliesslich |f| ≤ c·|g|. -/
example (f g : ℝ → ℝ) (l : Filter ℝ) :
    f =o[l] g ↔ ∀ c > 0, ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ :=
  Asymptotics.isLittleO_iff

/-- Bei x ist (y - x)² klein gegen y - x: Zu jedem c reicht die Kugel vom
Radius c um x. Wie nah "nahe genug" ist, haengt von c ab. -/
theorem quadrat_klein_o (x : ℝ) :
    (fun y : ℝ => (y - x) ^ 2) =o[𝓝 x] (fun y => y - x) := by
  rw [Asymptotics.isLittleO_iff]
  intro c hc
  filter_upwards [Metric.ball_mem_nhds x hc] with y hy  -- B(x, c) ist gross
  rw [Metric.mem_ball, Real.dist_eq] at hy              -- hy : |y - x| < c
  rw [norm_pow, Real.norm_eq_abs, sq]                   -- |y-x|·|y-x| ≤ c·|y-x|
  exact mul_le_mul_of_nonneg_right hy.le (abs_nonneg _)
--<<landauEnd>>

--<<ableitungZuFuss>>
/- y² hat in x die Ableitung 2x, aus der Definition: Der Fehler der Tangente
   ist (y - x)², und das ist o(y - x) nach `quadrat_klein_o`. -/
example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
  rw [hasDerivAt_iff_isLittleO]          -- Mathlibs Definition der Ableitung
  have fehler : (fun y : ℝ => y ^ 2 - x ^ 2 - (y - x) • (2 * x)) =
      fun y => (y - x) ^ 2 := by
    ext y; rw [smul_eq_mul]; ring
  rw [fehler]                            -- der Fehler ist (y - x)²
  exact quadrat_klein_o x

/- Mit den fertigen Ableitungsregeln ist es eine Zeile: -/
example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
  simpa using hasDerivAt_pow 2 x
--<<ableitungZuFussEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<ableitungEindeutig>>
/-- Nur eine Gerade passt am besten. Die Ableitung ist der Grenzwert der
Sekantensteigungen laengs 𝓝[≠] x (`hasDerivAt_iff_tendsto_slope`), und
Grenzwerte sind eindeutig: Neben x liegen beliebig nahe andere Punkte. -/
theorem ableitung_eindeutig {f : ℝ → ℝ} {x s t : ℝ}
    (hs : HasDerivAt f s x) (ht : HasDerivAt f t x) : s = t :=
  grenzwert_eindeutig (hasDerivAt_iff_tendsto_slope.1 hs)
    (hasDerivAt_iff_tendsto_slope.1 ht)
--<<ableitungEindeutigEnd>>

--<<fermat>>
/-- Fermat: An einem lokalen Minimum ist f' = 0. Nahe a ist f a ≤ f x, also
die Sekantensteigung `slope f a x` = (f x - f a)/(x - a) links von a ≤ 0,
rechts ≥ 0. Grenzwerte erhalten ≤, also f' ≤ 0 ≤ f'. -/
theorem fermat {f : ℝ → ℝ} {a f' : ℝ} (h : IsLocalMin f a)
    (hf : HasDerivAt f f' a) : f' = 0 := by
  have hs : Tendsto (slope f a) (𝓝[≠] a) (𝓝 f') :=   -- Steigungen → f'
    hasDerivAt_iff_tendsto_slope.1 hf
  apply le_antisymm                                 -- f' ≤ 0 und 0 ≤ f'
  · apply le_of_tendsto (hs.mono_left (nhdsLT_le_nhdsNE a))  -- nur links von a
    filter_upwards [nhdsWithin_le_nhds h, self_mem_nhdsWithin] with x hfx hxa
    rw [slope_comm]                                 -- Steigung von x nach a
    exact (slope_nonpos_iff_of_le hxa.le).2 hfx     -- f a ≤ f x, x < a: ≤ 0
  · apply ge_of_tendsto (hs.mono_left (nhdsGT_le_nhdsNE a))  -- nur rechts von a
    filter_upwards [nhdsWithin_le_nhds h, self_mem_nhdsWithin] with x hfx hax
    exact (slope_nonneg_iff_of_le hax.le).2 hfx     -- f a ≤ f x, a < x: ≥ 0
--<<fermatEnd>>

--<<ableitungRegeln>>
/- Produktregel: eine Rechenregel, aus Mathlib (`HasDerivAt.mul`). -/
example (f g : ℝ → ℝ) (x f' g' : ℝ)
    (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) :
    HasDerivAt (fun y => f y * g y) (f' * g x + f x * g') x := hf.mul hg

/- `deriv` ist total: Wo f nicht differenzierbar ist, liefert es den
   Junk-Wert 0 -- so legt Mathlib `deriv` fest, daher zitiert. -/
example (f : ℝ → ℝ) (x : ℝ) (h : ¬ DifferentiableAt ℝ f x) : deriv f x = 0 :=
  deriv_zero_of_not_differentiableAt h

/- Deshalb braucht Fermat fuer `deriv` keine Differenzierbarkeit: -/
example (f : ℝ → ℝ) (a : ℝ) (h : IsLocalMin f a) : deriv f a = 0 := by
  rcases Classical.em (DifferentiableAt ℝ f a) with hd | hd
  · exact fermat h hd.hasDerivAt                  -- differenzierbar
  · exact deriv_zero_of_not_differentiableAt hd   -- sonst Junk-Wert 0
--<<ableitungRegelnEnd>>

/-! ## 4.5 Gradientenabstieg -/

--<<gradientAbleitung>>
/- Der Gradient ist der Riesz-Repraesentant des Differentials in einem
   Hilbertraum (`[CompleteSpace F]`): Das Differential ist primaer. Ueber ℝ
   faellt er in einer Variablen mit der Ableitung zusammen: -/
example (f : ℝ → ℝ) (x : ℝ) : gradient f x = deriv f x := gradient_eq_deriv'
--<<gradientAbleitungEnd>>

--<<gdIteration>>
/-- Ein Schritt des Gradientenabstiegs in einer Variablen. -/
noncomputable def gdSchritt (f : ℝ → ℝ) (t x : ℝ) : ℝ := x - t * deriv f x

/-- Die Iteration `x₀, x₁, x₂, …` mit fester Schrittweite `t`. -/
noncomputable def gdIter (f : ℝ → ℝ) (t x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => gdSchritt f t (gdIter f t x₀ n)
--<<gdIterationEnd>>

section Quadratisch
variable (a m c : ℝ)

--<<quadratischeZielfunktion>>
/-- Modellproblem `q(x) = a·(x - m)² + c`; fuer `a > 0` Minimum bei `m`. -/
noncomputable def q (x : ℝ) : ℝ := a * (x - m) ^ 2 + c

lemma hasDerivAt_q (x : ℝ) : HasDerivAt (q a m c) (2 * a * (x - m)) x := by
  show HasDerivAt (fun x : ℝ => a * (x - m) ^ 2 + c) (2 * a * (x - m)) x
  have h1 : HasDerivAt (fun y : ℝ => y - m) 1 x := (hasDerivAt_id x).sub_const m
  have h2 := ((h1.pow 2).const_mul a).add_const c
  convert h2 using 1
  push_cast
  ring

lemma deriv_q (x : ℝ) : deriv (q a m c) x = 2 * a * (x - m) :=
  (hasDerivAt_q a m c x).deriv

/-- `m` ist globales Minimum, sobald `a ≥ 0`. -/
lemma q_min (ha : 0 ≤ a) (x : ℝ) : q a m c m ≤ q a m c x := by
  show a * (m - m) ^ 2 + c ≤ a * (x - m) ^ 2 + c
  nlinarith [mul_nonneg ha (sq_nonneg (x - m))]
--<<quadratischeZielfunktionEnd>>

--<<gdKonvergenz>>
/-- Ein Schritt multipliziert den Fehler mit dem Faktor `1 - 2at`. -/
lemma gd_q_schritt (t x : ℝ) :
    gdSchritt (q a m c) t x - m = (1 - 2 * a * t) * (x - m) := by
  show x - t * deriv (q a m c) x - m = (1 - 2 * a * t) * (x - m)
  rw [deriv_q]
  ring

/-- Exakte Fehlerformel nach `n` Schritten. -/
lemma gd_q_fehler (t x₀ : ℝ) (n : ℕ) :
    gdIter (q a m c) t x₀ n - m = (1 - 2 * a * t) ^ n * (x₀ - m) := by
  induction n with
  | zero => simp [gdIter]
  | succ k ih =>
      simp only [gdIter]
      rw [gd_q_schritt, ih]
      ring

/-- **Konvergenzsatz.** Ist `|1 - 2at| < 1`, so konvergiert der
Gradientenabstieg gegen `m` (fuer `a > 0` das Minimum). -/
theorem gd_q_tendsto (t : ℝ) (h : |1 - 2 * a * t| < 1) (x₀ : ℝ) :
    Tendsto (fun n => gdIter (q a m c) t x₀ n) atTop (𝓝 m) := by
  have key : ∀ n,
      gdIter (q a m c) t x₀ n = (1 - 2 * a * t) ^ n * (x₀ - m) + m := by
    intro n
    have := gd_q_fehler a m c t x₀ n
    linarith
  have hpow : Tendsto (fun n : ℕ => (1 - 2 * a * t) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_norm_lt_one (by rwa [Real.norm_eq_abs])
  simpa [key] using (hpow.mul_const (x₀ - m)).add_const m

/-- Die zulaessigen Schrittweiten: genau `0 < t < 1/a`. Zu grosse Schritte
divergieren, zu kleine konvergieren langsam. -/
theorem schrittweite (ha : 0 < a) (t : ℝ) :
    |1 - 2 * a * t| < 1 ↔ (0 < t ∧ t < 1 / a) := by
  rw [abs_lt, lt_div_iff₀ ha]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by nlinarith, by nlinarith⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by nlinarith, by nlinarith⟩
--<<gdKonvergenzEnd>>

-- Nur im Skript, nicht auf den Folien.
--<<gdKontraktion>>
/-- Fuer |1 - 2at| < 1 ist ein Schritt eine Kontraktion mit K = |1 - 2at|:
Der Konvergenzsatz ist ein Fall des Fixpunktsatzes. -/
lemma gd_q_kontraktion (t : ℝ) (h : |1 - 2 * a * t| < 1) :
    Kontraktion |1 - 2 * a * t| (gdSchritt (q a m c) t) where
  nichtneg := abs_nonneg _
  klein := h
  schrumpft x y := by
    have e : gdSchritt (q a m c) t x - gdSchritt (q a m c) t y =
        (1 - 2 * a * t) * (x - y) := by
      linarith [gd_q_schritt a m c t x, gd_q_schritt a m c t y]
    rw [Real.dist_eq, Real.dist_eq, e, abs_mul]

/- Ihr Fixpunkt ist m, fuer a > 0 das Minimum: -/
example (t : ℝ) : gdSchritt (q a m c) t m = m := by
  linarith [gd_q_schritt a m c t m]
--<<gdKontraktionEnd>>

end Quadratisch

--<<konvexMinimum>>
/-- Fuer konvexe f ist ein Punkt mit Ableitung 0 ein globales Minimum: Rechts
von c sind die Sekanten mindestens so steil wie f'(c) = 0, links hoechstens.
Mit `deriv f c = 0` allein waere es falsch (Junk-Wert). -/
theorem konvex_kritisch {f : ℝ → ℝ} {c : ℝ} (hf : ConvexOn ℝ Set.univ f)
    (hc : HasDerivAt f 0 c) (y : ℝ) : f c ≤ f y := by
  rcases lt_trichotomy y c with hy | rfl | hy
  · -- y < c: Die Sekante von y nach c steigt hoechstens mit 0.
    have h := hf.slope_le_of_hasDerivAt (Set.mem_univ y) (Set.mem_univ c) hy hc
    exact (slope_nonpos_iff_of_le hy.le).1 h
  · exact le_rfl
  · -- c < y: Die Sekante von c nach y steigt mindestens mit 0.
    have h := hf.le_slope_of_hasDerivAt (Set.mem_univ c) (Set.mem_univ y) hy hc
    exact (slope_nonneg_iff_of_le hy.le).1 h
--<<konvexMinimumEnd>>

/-! ## 4.6 Optima unter Nebenbedingungen: Lagrange-Multiplikatoren -/

-- Nur im Skript, nicht auf den Folien.
--<<lagrangeZusammenhang>>
/-- Projektiv → klassisch, sobald Df ≠ 0: Waere Λ₀ = 0, so waere Λ ≠ 0 und
Λ • Df = 0, also Df = 0. Sonst darf man durch Λ₀ teilen: μ = -Λ/Λ₀ ist das
λ der klassischen Fassung (`λ` ist in Lean fuer `fun` reserviert). -/
theorem klassisch_aus_projektiv {E : Type*} [AddCommGroup E] [Module ℝ E]
    {Df Dφ : E} {Λ Λ₀ : ℝ} (hne : (Λ, Λ₀) ≠ 0) (h : Λ • Df + Λ₀ • Dφ = 0)
    (hDf : Df ≠ 0) : ∃ μ : ℝ, Dφ = μ • Df := by
  have hΛ₀ : Λ₀ ≠ 0 := by
    rintro rfl                                   -- Fall Λ₀ = 0
    have hΛ : Λ ≠ 0 := fun h0 => hne (by simp [h0])
    rw [zero_smul, add_zero, smul_eq_zero] at h  -- Λ = 0 oder Df = 0
    exact h.elim hΛ hDf
  have h2 : Λ₀ • Dφ = -(Λ • Df) := eq_neg_of_add_eq_zero_right h
  use -Λ / Λ₀                                    -- Fall Λ₀ ≠ 0: teilen
  calc Dφ = Λ₀⁻¹ • (Λ₀ • Dφ) := by rw [smul_smul, inv_mul_cancel₀ hΛ₀, one_smul]
    _ = Λ₀⁻¹ • -(Λ • Df) := by rw [h2]
    _ = (-Λ / Λ₀) • Df := by rw [smul_neg, smul_smul, ← neg_smul]; congr 1; ring
--<<lagrangeZusammenhangEnd>>

--<<lagrangeKlassisch>>
/-- Lagrange, klassisch, mit allen Voraussetzungen: f und φ streng
differenzierbar und Df ≠ 0. Beweis: Mathlibs projektiver Satz, dann der
Zusammenhang von oben. -/
theorem lagrange_klassisch {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] {f φ : E → ℝ} {x₀ : E}
    {Df Dφ : E →L[ℝ] ℝ} (hext : IsLocalExtrOn φ {x | f x = f x₀} x₀)
    (hf : HasStrictFDerivAt f Df x₀) (hφ : HasStrictFDerivAt φ Dφ x₀)
    (hDf : Df ≠ 0) : ∃ μ : ℝ, Dφ = μ • Df := by
  obtain ⟨Λ, Λ₀, hne, h⟩ :=
    hext.exists_multipliers_of_hasStrictFDerivAt_1d hf hφ
  exact klassisch_aus_projektiv hne h hDf
--<<lagrangeKlassischEnd>>

--<<lagrangeBeispiel>>
/-- Beispiel: maximiere `x·y` unter `x + y = 1`.
Lagrange liefert den Kandidaten `x = y = 1/2`; hier die elementare
Bestaetigung, dass dieser Kandidat wirklich das Maximum ist. -/
theorem xy_le_quarter (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  nlinarith [sq_nonneg (x - y)]

/- Der Wert wird im Lagrange-Punkt angenommen. -/
example : (1 / 2 : ℝ) * (1 / 2) = 1 / 4 := by norm_num
--<<lagrangeBeispielEnd>>

/-! ### Mathlib-Wegweiser zu Vorlesung 5

Nur hier in der Datei, nicht im Skript und nicht auf den Folien. -/
#check @CauchySeq
#check @CompleteSpace
#check @ContractingWith                                -- K : NNReal
#check @ContractingWith.fixedPoint_isFixedPt           -- Schritte 2-4
#check @ContractingWith.fixedPoint_unique              -- Schritt 5
#check @ContractingWith.apriori_dist_iterate_fixedPoint_le  -- Schritt 6
#check @exists_norm_eq_iInf_of_complete_convex         -- Projektion: Existenz
#check @Submodule.norm_eq_iInf_iff_real_inner_eq_zero  -- ... und senkrecht
#check @Asymptotics.isBigO_iff                         -- O(g) ausgeschrieben
#check @Asymptotics.isLittleO_pow_pow_atTop_of_lt      -- x = o(x²) bei +∞
#check @Asymptotics.isLittleO_one_iff                  -- o(1) heisst → 0
#check @HasDerivAt.unique                              -- nur eine Steigung
#check @hasDerivAt_iff_hasFDerivAt                     -- Spezialfall Frechet
#check @IsLocalMin.hasDerivAt_eq_zero                  -- Fermat, HasDerivAt
#check @IsLocalMin.deriv_eq_zero                       -- Fermat, deriv
#check @exists_hasDerivAt_eq_slope                     -- Mittelwertsatz
#check @ConvexOn.slope_le_of_hasDerivAt                -- Sekante, Tangente
#check @IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt  -- mehrere NB

end Skript.Analysis
