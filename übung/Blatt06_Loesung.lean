/-
Übungsblatt 6 -- Optimierung
Ersetzen Sie jedes `sorry` durch einen Beweis. Die Datei ist fertig, wenn Lean
keine Warnung `declaration uses 'sorry'` mehr meldet.
Beim Suchen nach Lemmas helfen `exact?` und `apply?`.
-/
--<<nurLoesung>>
-- Loesungsfassung. Die Aufgabenfassung lean/Blatt06.lean erzeugt erzeuge.pl daraus.
--<<nurLoesungEnd>>
import Mathlib

namespace Uebung06

open Filter Topology

/-! ## B1: Extrema in der Ebene (zu A2) -/

--<<b1>>
/-- Die Funktion aus A2, `f(x, y) = x³ - 3x + y³ - 3y`. Punkte der Ebene sind
    Paare `p` mit den Koordinaten `p.1` und `p.2`. -/
def landschaft (p : ℝ × ℝ) : ℝ := p.1 ^ 3 - 3 * p.1 + (p.2 ^ 3 - 3 * p.2)

-- `IsLocalMin f p` heisst: schliesslich laengs `𝓝 p` ist `f p ≤ f q`;
-- `IsLocalMax f p` ebenso mit `f q ≤ f p`. Auf Paaren rechnet Mathlib mit
-- der Maximumsnorm: Die Kugel vom Radius δ um p ist ein Quadrat.

/-- Aufgabe B1 (a) (A2 (b)): Fuer `x, y ≥ -2` ist `f(1, 1) ≤ f(x, y)`. -/
theorem landschaft_abschaetzung (p : ℝ × ℝ) (hx : -2 ≤ p.1)
    (hy : -2 ≤ p.2) : landschaft (1, 1) ≤ landschaft p := by
  -- Nuetzlich: die Rechnung von Papier als Gleichung `have e : ... := by
  -- simp only [landschaft]; ring`, dann mul_nonneg, sq_nonneg, linarith
  --<<loesung>>
  -- A2 (b) mit x = 1 + h und y = 1 + k:
  -- f(x, y) - f(1, 1) = (x - 1)²(x + 2) + (y - 1)²(y + 2)
  have e : landschaft p - landschaft (1, 1) =
      (p.1 - 1) ^ 2 * (p.1 + 2) + (p.2 - 1) ^ 2 * (p.2 + 2) := by
    simp only [landschaft]
    ring
  -- Quadrat mal Zahl ≥ 0 ist ≥ 0, fuer beide Summanden
  have h1 : 0 ≤ (p.1 - 1) ^ 2 * (p.1 + 2) :=
    mul_nonneg (sq_nonneg _) (by linarith)
  have h2 : 0 ≤ (p.2 - 1) ^ 2 * (p.2 + 2) :=
    mul_nonneg (sq_nonneg _) (by linarith)
  linarith
  --<<loesungEnd>>

/-- Aufgabe B1 (b) (A2 (b)): `f` hat in `(1, 1)` ein lokales Minimum. -/
theorem landschaft_tal : IsLocalMin landschaft (1, 1) := by
  -- Nuetzlich: filter_upwards mit Metric.ball_mem_nhds (die Kugel vom
  -- Radius 3 um (1, 1)), Metric.mem_ball, Prod.dist_eq, max_lt_iff,
  -- Real.dist_eq, abs_lt, linarith; dann (a)
  --<<loesung>>
  -- Die Kugel vom Radius 3 um (1, 1) ist gross.
  filter_upwards [Metric.ball_mem_nhds ((1, 1) : ℝ × ℝ) three_pos] with p hp
  -- In ihr ist |x - 1| < 3 und |y - 1| < 3, also x > -2 und y > -2.
  rw [Metric.mem_ball, Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq,
    abs_lt, abs_lt] at hp
  apply landschaft_abschaetzung p
  · linarith [hp.1.1]
  · linarith [hp.2.1]
  --<<loesungEnd>>

/-- Aufgabe B1 (c) (A2 (c)): In `(1, -1)` hat `f` weder ein lokales Minimum
    noch ein lokales Maximum: ein Sattelpunkt. -/
theorem landschaft_sattel :
    ¬ IsLocalMin landschaft (1, -1) ∧ ¬ IsLocalMax landschaft (1, -1) := by
  -- Nuetzlich: wie betrag_nicht_klein_o auf Blatt 5 (B3 (b)): Metric.eventually_nhds_iff
  -- liefert ein δ > 0. Die Punkte (1, -1 + t) und (1 + t, -1) mit
  -- t = min (δ / 2) 1 liegen in der Kugel vom Radius δ (lt_min, half_pos,
  -- min_le_left, min_le_right, half_lt_self, Prod.dist_eq, max_lt_iff,
  -- Real.dist_eq, abs_of_pos). Dann simp only [landschaft] und nlinarith mit
  -- dem Hinweis mul_pos (mul_pos ht ht) _ (mit ht : 0 < t) fuer
  -- t²(3 - t) > 0 bzw. t²(3 + t) > 0
  --<<loesung>>
  constructor
  · -- Kein Minimum: in y-Richtung faellt f, f(1, -1 + t) = t²(t - 3) < 0.
    intro h
    -- Auf einer Kugel vom Radius δ ist f(1, -1) ≤ f(q).
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    set t := min (δ / 2) 1
    have ht : 0 < t := lt_min (half_pos hδ) one_pos
    have ht1 : t ≤ 1 := min_le_right _ _               -- also t < 3
    have htδ : t < δ := lt_of_le_of_lt (min_le_left _ _) (half_lt_self hδ)
    -- Der Punkt (1, -1 + t) liegt in der Kugel.
    have h3 := hball (y := (1, -1 + t)) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      constructor
      · simpa using hδ
      · rw [show -1 + t - -1 = t by ring, abs_of_pos ht]
        exact htδ)
    simp only [landschaft] at h3          -- h3 : f(1, -1) ≤ f(1, -1 + t)
    nlinarith [mul_pos (mul_pos ht ht) (by linarith : (0 : ℝ) < 3 - t)]
  · -- Kein Maximum: in x-Richtung steigt f, f(1 + t, -1) = t²(3 + t) > 0.
    intro h
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    set t := min (δ / 2) 1
    have ht : 0 < t := lt_min (half_pos hδ) one_pos
    have htδ : t < δ := lt_of_le_of_lt (min_le_left _ _) (half_lt_self hδ)
    have h3 := hball (y := (1 + t, -1)) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      constructor
      · rw [show 1 + t - 1 = t by ring, abs_of_pos ht]
        exact htδ
      · simpa using hδ)
    simp only [landschaft] at h3          -- h3 : f(1 + t, -1) ≤ f(1, -1)
    nlinarith [mul_pos (mul_pos ht ht) (by linarith : (0 : ℝ) < 3 + t)]
  --<<loesungEnd>>
--<<b1End>>

/-! ## B2: Gradientenabstieg in der Ebene (zu A3) -/

--<<b2>>
/-- Der Gradient von `f(x, y) = (a x² + b y²)/2` -- fuer `a = 1`, `b = 4` die
    Funktion aus A3. Punkte der Ebene sind Paare `p` mit den Koordinaten `p.1`
    und `p.2`. -/
def gradF (a b : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := (a * p.1, b * p.2)

/-- Ein Schritt des Gradientenabstiegs, `p - t ∇f(p)` (wie gdSchritt). -/
def gdSchritt2 (a b t : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := p - t • gradF a b p

/-- Die Iteration `p₀, p₁, p₂, …` (wie gdIter in der Vorlesung). -/
def gdIter2 (a b t : ℝ) (p₀ : ℝ × ℝ) : ℕ → ℝ × ℝ
  | 0 => p₀
  | n + 1 => gdSchritt2 a b t (gdIter2 a b t p₀ n)

/-- Vorgegeben: Ein Schritt multipliziert die Koordinaten mit `1 - t a` bzw.
    `1 - t b`. `ext` zerlegt eine Gleichung zwischen Paaren in zwei
    Gleichungen, eine fuer jede Koordinate. -/
theorem gd2_schritt (a b t : ℝ) (p : ℝ × ℝ) :
    gdSchritt2 a b t p = ((1 - t * a) * p.1, (1 - t * b) * p.2) := by
  ext
  · simp [gdSchritt2, gradF]
    ring
  · simp [gdSchritt2, gradF]
    ring

/-- Aufgabe B2 (a) (A3 (c)): die geschlossene Form. -/
theorem gd2_formel (a b t : ℝ) (p₀ : ℝ × ℝ) (n : ℕ) :
    gdIter2 a b t p₀ n = ((1 - t * a) ^ n * p₀.1, (1 - t * b) ^ n * p₀.2) := by
  -- Nuetzlich: wie gd_q_fehler in der Vorlesung, mit gd2_schritt; fuer
  -- Gleichungen zwischen Paaren hilft ext (siehe gd2_schritt)
  --<<loesung>>
  induction n with
  | zero => simp [gdIter2]
  | succ k ih =>
      simp only [gdIter2]
      rw [gd2_schritt, ih]
      ext
      · simp only
        ring
      · simp only
        ring
  --<<loesungEnd>>

/-- Aufgabe B2 (b) (A3 (d), Rechnung): Beide Faktoren haben genau dann Betrag
    `< 1`, wenn `0 < t < 2/b`. -/
theorem schrittweite2 (a b t : ℝ) (ha : 0 < a) (hab : a ≤ b) :
    (|1 - t * a| < 1 ∧ |1 - t * b| < 1) ↔ (0 < t ∧ t < 2 / b) := by
  -- Nuetzlich: wie schrittweite in der Vorlesung: abs_lt, lt_div_iff₀
  -- (dafuer erst 0 < b zeigen), constructor, rintro, nlinarith
  --<<loesung>>
  have hb : 0 < b := by linarith
  rw [abs_lt, abs_lt, lt_div_iff₀ hb]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    exact ⟨by nlinarith, by nlinarith⟩
  · rintro ⟨h1, h2⟩
    exact ⟨⟨by nlinarith, by nlinarith⟩, by nlinarith, by nlinarith⟩
  --<<loesungEnd>>

/-- Aufgabe B2 (c) (A3 (d), eine Richtung): Fuer `0 < t < 2/b` konvergiert das
    Verfahren von jedem Startwert aus gegen das Minimum `(0, 0)`. -/
theorem gd2_konvergenz (a b t : ℝ) (ha : 0 < a) (hab : a ≤ b) (ht : 0 < t)
    (ht' : t < 2 / b) (p₀ : ℝ × ℝ) :
    Tendsto (gdIter2 a b t p₀) atTop (𝓝 (0, 0)) := by
  -- Nuetzlich: schrittweite2; aus gd_q_tendsto in der Vorlesung
  -- tendsto_pow_atTop_nhds_zero_of_norm_lt_one, Real.norm_eq_abs und
  -- Tendsto.mul_const; Tendsto.prodMk_nhds; funext (gd2_formel a b t p₀)
  -- ist die Formel als Gleichung zwischen Folgen
  --<<loesung>>
  obtain ⟨hA, hB⟩ := (schrittweite2 a b t ha hab).mpr ⟨ht, ht'⟩
  have hpa : Tendsto (fun n : ℕ => (1 - t * a) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_norm_lt_one (by rwa [Real.norm_eq_abs])
  have hpb : Tendsto (fun n : ℕ => (1 - t * b) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_norm_lt_one (by rwa [Real.norm_eq_abs])
  have hx : Tendsto (fun n : ℕ => (1 - t * a) ^ n * p₀.1) atTop (𝓝 0) := by
    simpa using hpa.mul_const p₀.1
  have hy : Tendsto (fun n : ℕ => (1 - t * b) ^ n * p₀.2) atTop (𝓝 0) := by
    simpa using hpb.mul_const p₀.2
  have hf : gdIter2 a b t p₀ =
      fun n => ((1 - t * a) ^ n * p₀.1, (1 - t * b) ^ n * p₀.2) :=
    funext (gd2_formel a b t p₀)
  rw [hf]
  exact hx.prodMk_nhds hy
  --<<loesungEnd>>
--<<b2End>>

/-! ## B3: Lagrange (zu A4) -/

--<<b3>>
/-- Aufgabe B3 (a) (A4 (a)): Auf der Geraden `x + 2y = 5` ist `x² + y² ≥ 5`;
    der Lagrange-Kandidat `(1, 2)` liefert den Wert `5`. -/
theorem abstand_gerade (x y : ℝ) (h : x + 2 * y = 5) : 5 ≤ x ^ 2 + y ^ 2 := by
  -- Nuetzlich: wie xy_le_quarter in der Vorlesung: nlinarith mit dem Hinweis
  -- sq_nonneg (2 * x - y); denn 5·(x² + y²) = (x + 2y)² + (2x - y)²
  --<<loesung>>
  nlinarith [sq_nonneg (2 * x - y)]
  --<<loesungEnd>>

/-- Aufgabe B3 (b) (A4 (a)): Der Wert `5` wird nur in `(1, 2)` angenommen. -/
theorem abstand_gerade_eindeutig (x y : ℝ) (h : x + 2 * y = 5)
    (hmin : x ^ 2 + y ^ 2 = 5) : x = 1 ∧ y = 2 := by
  -- Nuetzlich: Aus der Gleichung 5·(x² + y²) = (x + 2y)² + (2x - y)² folgt
  -- (2x - y)² = 0 (nlinarith), also 2x - y = 0 (pow_eq_zero_iff); dann linarith
  --<<loesung>>
  have h1 : (2 * x - y) ^ 2 = 0 := by nlinarith
  have h2 : 2 * x - y = 0 := pow_eq_zero_iff (by norm_num) |>.1 h1
  constructor <;> linarith
  --<<loesungEnd>>
--<<b3End>>

end Uebung06
