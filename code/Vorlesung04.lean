/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 4
  (Folien: folien/vorlesung04.tex)
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

/-! ## 4.1 Raeume -/

-- Folie: "Topologie in Lean, selbst gebaut"
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
  leer_offen := trivial                   -- ∅ liegt in univ
  univ_offen := trivial                   -- X liegt in univ
  vereinigung _ := trivial                -- jede Vereinigung liegt in univ
  schnitt _ _ := trivial                  -- jeder Schnitt liegt in univ

-- Folie: "Abstand auf \RR und in der Ebene" (Kasten "In Lean")
/- Auf ℝ ist der Abstand per Definition |x - y| (`Real.dist_eq`). -/
example (x y : ℝ) : dist x y = |x - y| := rfl
example : dist (3:ℝ) 7 = 4 := by
  rw [Real.dist_eq]
  norm_num

/- In der Ebene ist es der Satz des Pythagoras: vier Gleichungen aus Mathlib
   einsetzen, eine pro Zeile. -/
example (x y : EuclideanSpace ℝ (Fin 2)) :
    dist x y = √((x 0 - y 0) ^ 2 + (x 1 - y 1) ^ 2) := by
  rw [EuclideanSpace.dist_eq]       -- dist x y = √(∑ i, dist (x i) (y i) ^ 2)
  rw [Fin.sum_univ_two]             -- ∑ i : Fin 2, f i = f 0 + f 1
  rw [Real.dist_eq, Real.dist_eq]   -- dist a b = |a - b|, je Koordinate
  rw [sq_abs, sq_abs]               -- |a| ^ 2 = a ^ 2, je Koordinate

-- Folie: "Offen in Lean: (0,1) ja, [0,1] nein"
/-- U heisst offen, wenn um jeden Punkt von U eine Kugel ganz in U passt. -/
def Offen {X : Type*} [MetricSpace X] (U : Set X) : Prop :=
  ∀ x ∈ U, ∃ ε > 0, Metric.ball x ε ⊆ U

/- (0,1) ist offen: Um x passt die Kugel vom Radius r = min(x, 1 - x),
   dem Abstand zum naeheren Rand. -/
example : Offen (Set.Ioo (0:ℝ) 1) := by
  change ∀ x ∈ Set.Ioo (0:ℝ) 1,
    ∃ ε > 0, Metric.ball x ε ⊆ Set.Ioo 0 1          -- Definition von Offen
  rintro x ⟨h0, h1⟩                                 -- 0 < x < 1
  use min x (1 - x), lt_min h0 (by linarith)        -- r = min(x, 1 - x) > 0
  intro y hy                                        -- hy : y ∈ B(x, r)
  rw [Metric.mem_ball, Real.dist_eq, abs_lt] at hy  -- -r < y - x < r
  constructor                                       -- 0 < y und y < 1
  · linarith [hy.1, min_le_left x (1 - x)]          -- 0 ≤ x - r < y
  · linarith [hy.2, min_le_right x (1 - x)]         -- y < x + r ≤ 1

/- Erst mit der Metrik laesst sich sagen, warum [0,1] nicht offen ist:
   um 0 passt keine Kugel hinein. Diesmal mit Mathlibs `IsOpen`;
   `Metric.isOpen_iff` uebersetzt es in die Bedingung von `Offen`. -/
example : ¬ IsOpen (Set.Icc (0:ℝ) 1) := by
  intro h                                       -- Annahme: [0,1] ist offen
  have h' : ∀ x ∈ Set.Icc (0:ℝ) 1,
      ∃ ε > 0, Metric.ball x ε ⊆ Set.Icc 0 1 :=
    Metric.isOpen_iff.1 h                       -- die Bedingung von `Offen`
  obtain ⟨ε, hε, hsub⟩ := h' 0 (by norm_num)    -- um 0: B(0, ε) ⊆ [0,1]
  have h1 : (-(ε / 2) : ℝ) ∈ Metric.ball (0:ℝ) ε := by
    rw [Metric.mem_ball, Real.dist_eq, abs_of_nonpos (by linarith)]
    linarith
  have h2 := hsub h1                            -- also -ε/2 ∈ [0,1]
  rw [Set.mem_Icc] at h2                        -- 0 ≤ -ε/2 und -ε/2 ≤ 1
  linarith [h2.1]                               -- 0 ≤ -ε/2 geht nicht: ε > 0

-- Folie: "Funktionen als Punkte" (Kasten "In Lean": Zahlen, Pfeile, Funktionen)
/- Auf ℝ ist die Laenge per Definition der Betrag (`Real.norm_eq_abs`). -/
example (x : ℝ) : ‖x‖ = |x| := rfl

/- In der Ebene: Laenge ueber die Koordinaten, und ein Zahlenbeispiel. -/
example (x : EuclideanSpace ℝ (Fin 2)) : ‖x‖ = √(x 0 ^ 2 + x 1 ^ 2) := by
  rw [EuclideanSpace.norm_eq]               -- ‖x‖ = √(∑ i, ‖x i‖ ^ 2)
  rw [Fin.sum_univ_two]                     -- ∑ i : Fin 2, f i = f 0 + f 1
  rw [Real.norm_eq_abs, Real.norm_eq_abs]   -- ‖a‖ = |a|, je Koordinate
  rw [sq_abs, sq_abs]                       -- |a| ^ 2 = a ^ 2, je Koordinate

example : ‖(!₂[(3:ℝ), 4] : EuclideanSpace ℝ (Fin 2))‖ = 5 := by
  rw [EuclideanSpace.norm_eq]               -- ‖x‖ = √(∑ i, ‖x i‖ ^ 2)
  rw [Fin.sum_univ_two]                     -- ∑ i : Fin 2, f i = f 0 + f 1
  change √(‖(3:ℝ)‖ ^ 2 + ‖(4:ℝ)‖ ^ 2) = 5   -- die Eintraege 3 und 4 ablesen
  norm_num                                  -- 9 + 16 = 25, √25 = 5

/- Auch Funktionen sind Vektoren: die stetigen Funktionen auf [0,1] mit der
   Supremumsnorm. Nur zitiert -- das Maximum braucht Kompaktheit. -/
#check (inferInstance : NormedAddCommGroup C(Set.Icc (0:ℝ) 1, ℝ))
example (f : C(Set.Icc (0:ℝ) 1, ℝ)) (x) : ‖f x‖ ≤ ‖f‖ := f.norm_coe_le_norm x

section Innenprodukt

-- Folie: "Das Skalarprodukt in der Ebene" / "Senkrecht -- und was daraus folgt" / "Cauchy--Schwarz ueber die Diskriminante"
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

end Innenprodukt

section Stufen
variable {X : Type*} [MetricSpace X] {U V : Set X}

/-! ### Wie eine Stufe die naechste liefert -/

-- Folie: "Wie eine Stufe die naechste liefert -- in Lean"
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
  change ∀ x ∈ U ∩ V,
    ∃ ε > 0, Metric.ball x ε ⊆ U ∩ V        -- Definition von Offen
  rintro x ⟨hxU, hxV⟩                       -- x ∈ U und x ∈ V
  obtain ⟨ε₁, h₁, hU'⟩ := hU x hxU          -- B(x, ε₁) ⊆ U
  obtain ⟨ε₂, h₂, hV'⟩ := hV x hxV          -- B(x, ε₂) ⊆ V
  use min ε₁ ε₂, lt_min h₁ h₂               -- der kleinere Radius
  intro y hy                                -- hy : y ∈ B(x, min ε₁ ε₂)
  constructor                               -- y ∈ U und y ∈ V
  · exact hU' (Metric.ball_subset_ball (min_le_left ε₁ ε₂) hy)
  · exact hV' (Metric.ball_subset_ball (min_le_right ε₁ ε₂) hy)

/- Abgleich mit Mathlib. Kein `Iff.rfl`: Mathlib gewinnt die Topologie
   eines metrischen Raums ueber die uniforme Struktur. Der Weg geht ueber
   Umgebungen (`U ∈ 𝓝 x`: U ist Umgebung von x): -/
example : Offen U ↔ IsOpen U := by
  rw [isOpen_iff_mem_nhds]          -- offen: Umgebung jedes seiner Punkte
  simp only [Metric.mem_nhds_iff]   -- Umgebung: enthaelt eine Kugel um x
  rfl                               -- genau die Definition von `Offen`

end Stufen

/-! ## 4.2 Konvergenz durch Filter -/

-- Folie: "Ein Filter in Lean: drei Axiome, drei Beweisfelder"
/-- Ein Filter: ein System "grosser" Mengen mit den Axiomen (i)-(iii). -/
structure MeinFilter (α : Type*) where
  gross : Set (Set α)
  univ_gross : Set.univ ∈ gross                                 -- (i)
  obermenge {U V : Set α} : U ∈ gross → U ⊆ V → V ∈ gross       -- (ii)
  schnitt {U V : Set α} : U ∈ gross → V ∈ gross → U ∩ V ∈ gross  -- (iii)

/-- Die Endstuecke: gross ist, was ein Endstueck {n | n ≥ N} enthaelt. -/
def endstuecke : MeinFilter ℕ where
  gross := {U | ∃ N, ∀ n ≥ N, n ∈ U}
  univ_gross := by
    change ∃ N, ∀ n ≥ N, n ∈ Set.univ       -- univ enthaelt ein Endstueck
    use 0                                   -- (i): Zeuge N = 0
    intro n _                               -- n ≥ 0 wird nicht gebraucht
    exact Set.mem_univ n                    -- jedes n liegt in univ
  obermenge := by
    rintro U V ⟨N, hN⟩ hUV                  -- ab N in U, und U ⊆ V
    change ∃ N, ∀ n ≥ N, n ∈ V              -- V enthaelt ein Endstueck
    use N                                   -- (ii): dasselbe N
    intro n hn                              -- hn : n ≥ N
    exact hUV (hN n hn)                     -- n ∈ U, also n ∈ V
  schnitt := by
    rintro U V ⟨N₁, h₁⟩ ⟨N₂, h₂⟩            -- ab N₁ in U, ab N₂ in V
    change ∃ N, ∀ n ≥ N, n ∈ U ∩ V          -- U ∩ V enthaelt ein Endstueck
    use max N₁ N₂                           -- (iii): das groessere N
    intro n hn                              -- hn : n ≥ max N₁ N₂
    constructor                             -- n ∈ U und n ∈ V
    · exact h₁ n (le_of_max_le_left hn)     -- n ≥ N₁, also n ∈ U
    · exact h₂ n (le_of_max_le_right hn)    -- n ≥ N₂, also n ∈ V

-- Folie: "Ein Filter in Lean: drei Axiome, drei Beweisfelder" (die Umgebungen)
/-- Die Umgebungen von a: gross ist, was eine Kugel um a enthaelt. -/
def umgebungen {X : Type*} [MetricSpace X] (a : X) : MeinFilter X where
  gross := {U | ∃ ε > 0, Metric.ball a ε ⊆ U}
  univ_gross := by
    change ∃ ε > 0, Metric.ball a ε ⊆ Set.univ  -- univ enthaelt eine Kugel
    use 1, one_pos                              -- (i): Radius ε = 1
    exact Set.subset_univ _                     -- jede Menge liegt in univ
  obermenge := by
    rintro U V ⟨ε, hε, h⟩ hUV                   -- B(a, ε) ⊆ U, und U ⊆ V
    change ∃ ε > 0, Metric.ball a ε ⊆ V         -- V enthaelt eine Kugel
    use ε, hε                                   -- (ii): dieselbe Kugel
    exact h.trans hUV                           -- B(a, ε) ⊆ U ⊆ V
  schnitt := by
    rintro U V ⟨ε₁, h₁, hU⟩ ⟨ε₂, h₂, hV⟩        -- Kugeln in U und in V
    change ∃ ε > 0, Metric.ball a ε ⊆ U ∩ V     -- U ∩ V enthaelt eine Kugel
    use min ε₁ ε₂, lt_min h₁ h₂                 -- (iii): das kleinere ε
    intro y hy                                  -- hy : y ∈ B(a, min ε₁ ε₂)
    constructor                                 -- y ∈ U und y ∈ V
    · exact hU (Metric.ball_subset_ball (min_le_left ε₁ ε₂) hy)
    · exact hV (Metric.ball_subset_ball (min_le_right ε₁ ε₂) hy)

section Konvergenz
variable {α β γ : Type*}

-- Folie: "Tendsto in Lean: selbst gebaut, dann abgeglichen"
/-- Konvergenz laengs Filtern: Urbilder grosser Mengen sind gross. -/
def MeinTendsto (f : α → β) (F : MeinFilter α) (G : MeinFilter β) : Prop :=
  ∀ V ∈ G.gross, f ⁻¹' V ∈ F.gross

/-- Verkettung. Wie in Mathlib steht die aeussere Funktion vorn:
`hg.comp hf` fuer `g ∘ f`. -/
theorem MeinTendsto.comp {f : α → β} {g : β → γ} {F : MeinFilter α}
    {G : MeinFilter β} {H : MeinFilter γ} (hg : MeinTendsto g G H)
    (hf : MeinTendsto f F G) : MeinTendsto (g ∘ f) F H := by
  change ∀ W ∈ H.gross, (g ∘ f) ⁻¹' W ∈ F.gross   -- Definition eingesetzt
  intro W hW                                      -- W gross in H
  have h1 : g ⁻¹' W ∈ G.gross := hg W hW          -- g⁻¹(W) gross in G
  have h2 : f ⁻¹' (g ⁻¹' W) ∈ F.gross := hf _ h1  -- f⁻¹(g⁻¹(W)) gross in F
  exact h2                                        -- das ist (g ∘ f)⁻¹(W)

-- Folie: "Tendsto in Lean: selbst gebaut, dann abgeglichen" (der Abgleich, Tabelle)
/- Mathlibs `Filter` hat dieselben vier Felder; ein `MeinFilter` wird Feld
   fuer Feld ein Mathlib-Filter. `Tendsto` ist dort dieselbe Definition: -/
def MeinFilter.toFilter (F : MeinFilter α) : Filter α where
  sets := F.gross                           -- die grossen Mengen
  univ_sets := F.univ_gross                 -- (i)
  sets_of_superset := F.obermenge           -- (ii)
  inter_sets := F.schnitt                   -- (iii)

example (f : α → β) (F : MeinFilter α) (G : MeinFilter β) :
    Tendsto f F.toFilter G.toFilter ↔ MeinTendsto f F G := Iff.rfl

/- Die beiden Beispiele sind Mathlibs `atTop` und `𝓝 a`: -/
theorem endstuecke_eq : endstuecke.toFilter = atTop := by
  ext U                                     -- U gross ↔ U gross
  change (∃ N, ∀ n ≥ N, n ∈ U) ↔ U ∈ atTop  -- Definition eingesetzt
  exact mem_atTop_sets.symm                 -- rechts: Mathlibs atTop

theorem umgebungen_eq {X : Type*} [MetricSpace X] (a : X) :
    (umgebungen a).toFilter = 𝓝 a := by
  ext U                                              -- U gross ↔ U gross
  change (∃ ε > 0, Metric.ball a ε ⊆ U) ↔ U ∈ 𝓝 a    -- Definition eingesetzt
  exact Metric.mem_nhds_iff.symm                     -- rechts: Mathlibs 𝓝 a

end Konvergenz

-- Folie: "Der Satz in Lean: derselbe Beweis"
/-- Die ε-N-Form, von Hand. Hin: Die Kugel um a ist gross, ihr Urbild
enthaelt ein Endstueck. Rueck: In jede grosse Menge um a passt eine Kugel. -/
theorem meinTendsto_iff_eps_N (u : ℕ → ℝ) (a : ℝ) :
    MeinTendsto u endstuecke (umgebungen a) ↔
      ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  -- links: MeinTendsto, endstuecke und umgebungen eingesetzt
  change (∀ V, (∃ ε > 0, Metric.ball a ε ⊆ V) → ∃ N, ∀ n ≥ N, u n ∈ V) ↔
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε
  constructor                           -- Hin und Rueck
  · intro h ε hε                        -- ε > 0 beliebig
    have hB : ∃ δ > 0, Metric.ball a δ ⊆ Metric.ball a ε :=
      ⟨ε, hε, subset_rfl⟩               -- B(a, ε) ist gross
    obtain ⟨N, hN⟩ := h _ hB            -- u liegt ab N in B(a, ε)
    exact ⟨N, hN⟩                       -- u n ∈ B(a, ε) heisst |u n - a| < ε
  · rintro h V ⟨ε, hε, hV⟩              -- V gross: B(a, ε) ⊆ V
    obtain ⟨N, hN⟩ := h ε hε            -- ab N: |u n - a| < ε
    use N                               -- dasselbe N
    intro n hn                          -- hn : n ≥ N
    exact hV (hN n hn)                  -- u n ∈ B(a, ε) ⊆ V

/- Dieselbe Aussage mit `unfold`: Lean setzt die Definitionen selbst ein, wir
   muessen die Aussage nicht hinschreiben. Danach steht dasselbe Ziel da wie
   beim `change` oben, und der Rest ist derselbe Beweis. -/
example (u : ℕ → ℝ) (a : ℝ) :
    MeinTendsto u endstuecke (umgebungen a) ↔
      ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  -- links: MeinTendsto, endstuecke und umgebungen eingesetzt
  unfold MeinTendsto endstuecke umgebungen
  -- Felder abgelesen, Mengenschreibweise aufgeloest
  simp only [Set.mem_setOf_eq, Set.mem_preimage]
  constructor                           -- Hin und Rueck
  · intro h ε hε                        -- ε > 0 beliebig
    have hB : ∃ δ > 0, Metric.ball a δ ⊆ Metric.ball a ε :=
      ⟨ε, hε, subset_rfl⟩               -- B(a, ε) ist gross
    obtain ⟨N, hN⟩ := h _ hB            -- u liegt ab N in B(a, ε)
    exact ⟨N, hN⟩                       -- u n ∈ B(a, ε) heisst |u n - a| < ε
  · rintro h V ⟨ε, hε, hV⟩              -- V gross: B(a, ε) ⊆ V
    obtain ⟨N, hN⟩ := h ε hε            -- ab N: |u n - a| < ε
    use N                               -- dasselbe N
    intro n hn                          -- hn : n ≥ N
    exact hV (hN n hn)                  -- u n ∈ B(a, ε) ⊆ V

-- Folie: "Der Satz in Lean: derselbe Beweis" (Mathlibs Tendsto, Live-Demo)
/-- Dieselbe Aussage fuer Mathlibs `Tendsto`: zurueckuebersetzen, fertig. -/
theorem tendsto_iff_eps_N (u : ℕ → ℝ) (a : ℝ) :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  rw [← endstuecke_eq, ← umgebungen_eq]                -- atTop, 𝓝 a ersetzen
  change MeinTendsto u endstuecke (umgebungen a) ↔ _   -- dieselbe Definition
  exact meinTendsto_iff_eps_N u a                      -- der Satz von Hand

/- Mathlib hat die ε-N-Form schon, fuer Folgen in jedem metrischen Raum: -/
example (u : ℕ → ℝ) (a : ℝ) :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - a| < ε := by
  rw [Metric.tendsto_atTop]
  simp [Real.dist_eq]

-- Folie: ""Schliesslich`` als Filterbegriff"
/- "P gilt schliesslich" (`∀ᶠ`) heisst: Die Menge, auf der P gilt, ist
   gross -- per Definition. -/
example {α : Type*} (F : Filter α) (P : α → Prop) :
    (∀ᶠ x in F, P x) ↔ {x | P x} ∈ F := Iff.rfl

/- Fuer `atTop` also: ab einem N. Im Beispiel reicht N = 11. -/
example : ∀ᶠ n : ℕ in atTop, 10 < n := by
  rw [eventually_atTop]                     -- ∃ N, ∀ n ≥ N, 10 < n
  use 11                                    -- Zeuge N = 11
  intro n hn                                -- hn : n ≥ 11
  omega                                     -- 10 < n

-- Folie: "Die Summenregel in Lean -- fuer jeden Filter"
/-- Summenregel, einmal fuer jeden Filter: ε halbieren wie immer -- aber statt
N₁ und N₂ zu maximieren, schneidet Axiom (iii) die beiden grossen Mengen. -/
theorem summenregel {α : Type*} {F : Filter α} {u v : α → ℝ} {a b : ℝ}
    (hu : Tendsto u F (𝓝 a)) (hv : Tendsto v F (𝓝 b)) :
    Tendsto (fun x => u x + v x) F (𝓝 (a + b)) := by
  rw [Metric.tendsto_nhds] at *          -- ∀ ε > 0: schliesslich dist < ε
  intro ε hε                             -- ε > 0 beliebig
  have h1 := hu (ε / 2) (by linarith)    -- schliesslich |u x - a| < ε/2
  have h2 := hv (ε / 2) (by linarith)    -- schliesslich |v x - b| < ε/2
  have h := inter_mem h1 h2              -- (iii): beides zugleich
  apply mem_of_superset h                -- (ii): eine Obermenge genuegt
  rintro x ⟨hx₁, hx₂⟩                    -- x liegt in beiden Mengen
  rw [Set.mem_setOf_eq, Real.dist_eq] at hx₁ hx₂ ⊢
  calc |u x + v x - (a + b)| = |(u x - a) + (v x - b)| := by congr 1; ring
    _ ≤ |u x - a| + |v x - b| := abs_add_le _ _
    _ < ε := by linarith

/- Derselbe Satz fuer Folgen (`atTop`) und fuer Funktionen (`𝓝 1`): -/
example : Tendsto (fun n : ℕ => 2 + 1 / (n : ℝ)) atTop (𝓝 (2 + 0)) :=
  summenregel tendsto_const_nhds tendsto_one_div_atTop_nhds_zero_nat
example : Tendsto (fun x : ℝ => x + 3) (𝓝 1) (𝓝 (1 + 3)) :=
  summenregel tendsto_id tendsto_const_nhds

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

end Skript.Analysis
