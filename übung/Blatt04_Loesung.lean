/-
Übungsblatt 4 -- Räume und Konvergenz
Ersetzen Sie jedes `sorry` durch einen Beweis. Die Datei ist fertig, wenn Lean
keine Warnung `declaration uses 'sorry'` mehr meldet.
-/
--<<nurLoesung>>
-- Loesungsfassung. Die Aufgabenfassung lean/Blatt04.lean erzeugt erzeuge.pl daraus.
--<<nurLoesungEnd>>
import Mathlib

namespace Uebung04

open Filter Topology

--<<b1>>
/-- Aufgabe B1 (Formalisierung von A1(a)): Jede Kugel ist offen. Ohne den
    fertigen Satz `Metric.isOpen_ball`, auch nicht versteckt ueber `simp`. -/
theorem kugel_offen {X : Type*} [MetricSpace X] (x : X) (r : ℝ) :
    IsOpen (Metric.ball x r) := by
  -- Nuetzlich: Metric.isOpen_iff (offen = um jeden Punkt passt eine Kugel),
  -- Metric.mem_ball (y ∈ Metric.ball x r ↔ dist y x < r),
  -- dist_triangle (dist x z ≤ dist x y + dist y z), use, constructor, linarith
  --<<loesung>>
  rw [Metric.isOpen_iff]
  intro y hy
  rw [Metric.mem_ball] at hy
  use r - dist y x
  constructor
  · linarith
  · intro z hz
    rw [Metric.mem_ball] at hz
    rw [Metric.mem_ball]
    linarith [dist_triangle z y x]
  --<<loesungEnd>>
--<<b1End>>

--<<b2>>
/-- Die drei Normaxiome der Vorlesung fuer eine Funktion `N : ℝ → ℝ`. -/
structure IstNorm (N : ℝ → ℝ) : Prop where
  null : ∀ x, N x = 0 ↔ x = 0
  homogen : ∀ c x, N (c * x) = |c| * N x
  dreieck : ∀ x y, N (x + y) ≤ N x + N y

/-- Aufgabe B2 (a) (Formalisierung von A2(d)): Die Metrik `min 1 |x - y|`
    kommt von keiner Norm. -/
theorem keine_norm (N : ℝ → ℝ) (hN : IstNorm N) :
    ¬ (∀ x y : ℝ, min 1 |x - y| = N (x - y)) := by
  -- Nuetzlich: intro h; have e := h x y mit passenden Zahlen x, y, ebenso mit
  -- hN.homogen c x; norm_num at ...; linarith
  --<<loesung>>
  intro h
  have h2 := h 2 0
  have h4 := h 4 0
  have h22 := hN.homogen 2 2
  norm_num at h2 h4 h22
  linarith
  --<<loesungEnd>>

/-- Aufgabe B2 (b) (Formalisierung von A3(a)): die Parallelogrammgleichung
    in jedem reellen Innenproduktraum. -/
theorem parallelogramm {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (x y : V) : ‖x + y‖ ^ 2 + ‖x - y‖ ^ 2 = 2 * ‖x‖ ^ 2 + 2 * ‖y‖ ^ 2 := by
  -- Nuetzlich: norm_add_sq_real, norm_sub_sq_real, ring
  --<<loesung>>
  rw [norm_add_sq_real, norm_sub_sq_real]
  ring
  --<<loesungEnd>>
--<<b2End>>

--<<b3>>
/-- Aufgabe B3 (a): der Abstand von n/(n+1) zu 1, wie in A4(a). -/
theorem abstand_bruch (n : ℕ) : dist ((n : ℝ) / (n + 1)) 1 = 1 / (n + 1) := by
  -- Nuetzlich: Real.dist_eq, div_sub_one; den Zaehler n - (n + 1) dann mit
  -- rw [show ... by ring] zu -1 machen; danach neg_div, abs_neg, abs_of_pos;
  -- (n : ℝ) + 1 ≠ 0 und 0 < 1 / ((n : ℝ) + 1) zeigt positivity
  --<<loesung>>
  have h : (n : ℝ) + 1 ≠ 0 := by positivity
  rw [Real.dist_eq, div_sub_one h]
  rw [show (n : ℝ) - (n + 1) = -1 by ring, neg_div, abs_neg,
    abs_of_pos (by positivity)]
  --<<loesungEnd>>

/-- Aufgabe B3 (b) (Formalisierung von A4(a)): n/(n+1) konvergiert gegen 1.
    Ohne den fertigen Satz `tendsto_natCast_div_add_atTop`. -/
theorem bruch_gegen_eins :
    Tendsto (fun n : ℕ => (n : ℝ) / (n + 1)) atTop (𝓝 1) := by
  -- Nuetzlich: Metric.tendsto_atTop, exists_nat_gt (Archimedes), abstand_bruch,
  -- one_div_lt (1 / a < b ↔ 1 / b < a fuer a, b > 0), exact_mod_cast, linarith
  --<<loesung>>
  rw [Metric.tendsto_atTop]
  intro ε hε
  obtain ⟨N, hN⟩ := exists_nat_gt (1 / ε)
  use N
  intro n hn
  rw [abstand_bruch n, one_div_lt (by positivity) hε]
  have hn' : (N : ℝ) ≤ n := by exact_mod_cast hn
  linarith
  --<<loesungEnd>>
--<<b3End>>

--<<b4>>
/-- Aufgabe B4 (a) (Formalisierung von A5(a), die Menge U₁):
    schliesslich gilt 100 * n < n ^ 2. -/
theorem schliesslich_quadrat : ∀ᶠ n : ℕ in atTop, 100 * n < n ^ 2 := by
  -- ∀ᶠ n in atTop, P n ist Filter.Eventually: P n gilt schliesslich.
  -- Nuetzlich: eventually_atTop, use, nlinarith
  --<<loesung>>
  rw [eventually_atTop]
  use 101
  intro n hn
  nlinarith
  --<<loesungEnd>>

/-- Aufgabe B4 (b) (Formalisierung von A5(c)): n ↦ n + k strebt gegen atTop.
    Ohne den fertigen Satz `tendsto_add_atTop_nat`. -/
theorem verschiebung (k : ℕ) : Tendsto (fun n : ℕ => n + k) atTop atTop := by
  -- Nuetzlich: tendsto_atTop_atTop, use, omega
  --<<loesung>>
  rw [tendsto_atTop_atTop]
  intro N
  use N
  intro n hn
  omega
  --<<loesungEnd>>

/-- Aufgabe B4 (c) (Folgerung in A5(c)): Die verschobene Folge hat denselben
    Grenzwert. Mit `verschiebung`, ohne `tendsto_add_atTop_iff_nat`. -/
theorem folge_verschoben (u : ℕ → ℝ) (a : ℝ) (k : ℕ)
    (h : Tendsto u atTop (𝓝 a)) :
    Tendsto (fun n => u (n + k)) atTop (𝓝 a) :=
  -- Nuetzlich: Tendsto.comp (die Verkettung aus der Vorlesung)
  --<<loesung>>
  h.comp (verschiebung k)
  --<<loesungEnd>>
--<<b4End>>

end Uebung04
