/-
Übungsblatt 5 -- Ableitung
Ersetzen Sie jedes `sorry` durch einen Beweis. Die Datei ist fertig, wenn Lean
keine Warnung `declaration uses 'sorry'` mehr meldet.
Beim Suchen nach Lemmas helfen `exact?` und `apply?`.
-/
--<<nurLoesung>>
-- Loesungsfassung. Die Aufgabenfassung lean/Blatt05.lean erzeugt erzeuge.pl daraus.
--<<nurLoesungEnd>>
import Mathlib

namespace Uebung05

open Filter Topology

/-! ## B1: "Schliesslich" (zu A1) -/

--<<b1>>
-- `∀ᶠ x in F, P x` heisst: Die Menge {x | P x} ist gross, sie liegt im Filter F.
-- Laengs `atTop` ist gross, was ein Endstueck [N, ∞) enthaelt, laengs `𝓝 a`,
-- was eine Kugel um a enthaelt.

/-- Aufgabe B1 (a) (A1 (a), ii): Schliesslich laengs `atTop` ist
    `3x + 7 < x²`. -/
theorem schliesslich_quadrat : ∀ᶠ x : ℝ in atTop, 3 * x + 7 < x ^ 2 := by
  -- Nuetzlich: eventually_ge_atTop (das Endstueck [N, ∞)), filter_upwards, nlinarith
  --<<loesung>>
  filter_upwards [eventually_ge_atTop (5 : ℝ)] with x hx   -- das Endstueck [5, ∞)
  nlinarith                                                 -- hx : 5 ≤ x
  --<<loesungEnd>>

/-- Aufgabe B1 (b) (A1 (a), iv): Schliesslich laengs `𝓝 2` ist
    `3 < x²`. -/
theorem schliesslich_nahe_zwei : ∀ᶠ x : ℝ in 𝓝 2, 3 < x ^ 2 := by
  -- Nuetzlich: Metric.ball_mem_nhds (2 : ℝ) (die Kugel vom Radius 1/4),
  -- Metric.mem_ball, Real.dist_eq, abs_lt, nlinarith
  --<<loesung>>
  filter_upwards [Metric.ball_mem_nhds (2 : ℝ) (show (0 : ℝ) < 1 / 4 by norm_num)] with x hx
  rw [Metric.mem_ball, Real.dist_eq, abs_lt] at hx          -- -1/4 < x - 2 < 1/4
  nlinarith [hx.1, hx.2]
  --<<loesungEnd>>

/-- Aufgabe B1 (c): Gilt `P` schliesslich und `Q` schliesslich, und folgt `R`
    aus `P` und `Q`, so gilt `R` schliesslich -- die Filteraxiome (ii) und
    (iii) in einem Satz. -/
theorem schliesslich_folgt {α : Type*} {F : Filter α} {P Q R : α → Prop}
    (hP : ∀ᶠ x in F, P x) (hQ : ∀ᶠ x in F, Q x) (h : ∀ x, P x → Q x → R x) :
    ∀ᶠ x in F, R x := by
  -- Nuetzlich: filter_upwards [hP, hQ] with x hp hq
  --<<loesung>>
  filter_upwards [hP, hQ] with x hp hq   -- der Schnitt {P} ∩ {Q}: Axiom (iii)
  exact h x hp hq                        -- darin gilt R: Axiom (ii)
  --<<loesungEnd>>
--<<b1End>>

/-! ## B2: Klein-o laengs eines Filters (zu A1) -/

--<<b2>>
/-- Klein-o laengs eines Filters, wie in der Vorlesung: Zu jedem `c > 0` gilt
    schliesslich `|f x| ≤ c·|g x|`. -/
def KleinO {α : Type*} (f g : α → ℝ) (l : Filter α) : Prop :=
  ∀ c > 0, ∀ᶠ x in l, |f x| ≤ c * |g x|

/-- Aufgabe B2 (a) (A1 (c), i): Bei `0` ist `x³` klein gegen `x²`. -/
theorem kubik_klein_o_null :
    KleinO (fun x : ℝ => x ^ 3) (fun x => x ^ 2) (𝓝 0) := by
  -- Nuetzlich: wie quadrat_klein_o in der Vorlesung: intro c hc,
  -- filter_upwards [Metric.ball_mem_nhds (0 : ℝ) hc] with x hx, Metric.mem_ball,
  -- Real.dist_eq, sub_zero, abs_pow, positivity, nlinarith
  --<<loesung>>
  intro c hc
  filter_upwards [Metric.ball_mem_nhds (0 : ℝ) hc] with x hx  -- (-c, c) ist gross
  rw [Metric.mem_ball, Real.dist_eq, sub_zero] at hx            -- hx : |x| < c
  rw [abs_pow, abs_pow]                                         -- |x|³ ≤ c·|x|²
  have h2 : 0 ≤ |x| ^ 2 := by positivity
  nlinarith [abs_nonneg x]
  --<<loesungEnd>>

/-- Aufgabe B2 (b) (A1 (c), ii): Bei `+∞` ist es umgekehrt: `x²` ist
    klein gegen `x³`. -/
theorem quadrat_klein_o_kubik :
    KleinO (fun x : ℝ => x ^ 2) (fun x => x ^ 3) atTop := by
  -- Nuetzlich: wie linear_klein_o_unendlich in der Vorlesung:
  -- filter_upwards [eventually_ge_atTop (1 / c)] with x hx, div_le_iff₀,
  -- positivity, abs_of_nonneg, sq_nonneg, pow_nonneg, nlinarith
  --<<loesung>>
  intro c hc
  filter_upwards [eventually_ge_atTop (1 / c)] with x hx      -- [1/c, ∞) ist gross
  have h1 : 1 ≤ c * x := by
    rwa [div_le_iff₀ hc, mul_comm] at hx
  have hx0 : 0 ≤ x := by
    have : 0 < 1 / c := by positivity
    linarith
  rw [abs_of_nonneg (sq_nonneg x), abs_of_nonneg (pow_nonneg hx0 3)]   -- x² ≤ c·x³
  nlinarith [sq_nonneg x]
  --<<loesungEnd>>

/-- Aufgabe B2 (c) (A1 (c), iii): Bei `+∞` ist `x³` nicht klein gegen
    `x²`. -/
theorem kubik_nicht_klein_o :
    ¬ KleinO (fun x : ℝ => x ^ 3) (fun x => x ^ 2) atTop := by
  -- Nuetzlich: wie quadrat_nicht_klein_o_unendlich in der Vorlesung: c = 1
  -- (one_pos); Filter.Eventually.and und .exists liefern einen Punkt x, an
  -- dem die Ungleichung und x ≥ 2 (eventually_ge_atTop) zugleich gelten
  --<<loesung>>
  intro h
  obtain ⟨x, hx, h2⟩ := ((h 1 one_pos).and (eventually_ge_atTop (2 : ℝ))).exists
  have hx0 : (0 : ℝ) ≤ x := by linarith
  rw [abs_of_nonneg (pow_nonneg hx0 3), abs_of_nonneg (sq_nonneg x), one_mul] at hx
  nlinarith [sq_nonneg x]                                       -- hx : x³ ≤ x²
  --<<loesungEnd>>

/-- Aufgabe B2 (d) (A1 (d)): Linearkombinationen von `o(g)` sind `o(g)` -- mit
    dem Schnitt zweier grosser Mengen. -/
theorem klein_o_linear {α : Type*} {l : Filter α} {f₁ f₂ g : α → ℝ} (a b : ℝ)
    (h₁ : KleinO f₁ g l) (h₂ : KleinO f₂ g l) :
    KleinO (fun x => a * f₁ x + b * f₂ x) g l := by
  -- Nuetzlich: wie klein_o_add in der Vorlesung, mit c / (|a| + |b| + 1)
  -- statt c / 2; positivity, div_pos, filter_upwards [h₁ _ _, h₂ _ _],
  -- abs_add_le, abs_mul, gcongr, div_le_one, calc
  --<<loesung>>
  intro c hc
  have hk : 0 < |a| + |b| + 1 := by positivity
  have hc' : 0 < c / (|a| + |b| + 1) := div_pos hc hk
  filter_upwards [h₁ _ hc', h₂ _ hc'] with x hx₁ hx₂             -- Schnitt: Axiom (iii)
  calc |a * f₁ x + b * f₂ x| ≤ |a| * |f₁ x| + |b| * |f₂ x| := by
        rw [← abs_mul, ← abs_mul]
        exact abs_add_le _ _
    _ ≤ |a| * (c / (|a| + |b| + 1) * |g x|) + |b| * (c / (|a| + |b| + 1) * |g x|) := by
        gcongr
    _ = (|a| + |b|) / (|a| + |b| + 1) * c * |g x| := by ring
    _ ≤ 1 * c * |g x| := by
        gcongr
        rw [div_le_one hk]
        linarith
    _ = c * |g x| := by ring
  --<<loesungEnd>>
--<<b2End>>

/-! ## B3: Die Ableitung (zu A2) -/

--<<b3>>
-- Vorgegeben, wie in der Vorlesung: Bei `x` ist `(y - x)²` klein gegen `y - x`,
-- und `f` hat in `x` die Ableitung `f'`, wenn der Fehler der Tangente klein
-- gegen `y - x` ist.

/-- Bei `x` ist `(y - x)²` klein gegen `y - x`: Zu jedem `c` reicht die Kugel
    vom Radius `c` um `x`. -/
theorem quadrat_klein_o (x : ℝ) :
    KleinO (fun y : ℝ => (y - x) ^ 2) (fun y => y - x) (𝓝 x) := by
  intro c hc
  filter_upwards [Metric.ball_mem_nhds x hc] with y hy
  rw [Metric.mem_ball, Real.dist_eq] at hy
  rw [abs_pow, sq]
  exact mul_le_mul_of_nonneg_right hy.le (abs_nonneg _)

/-- Die Ableitung, wie auf der Folie: Der Fehler der Tangente ist klein gegen
    `y - x` laengs `𝓝 x`. -/
theorem ableitung_klein_o (f : ℝ → ℝ) (f' x : ℝ) :
    HasDerivAt f f' x ↔
      KleinO (fun y => f y - f x - (y - x) * f') (fun y => y - x) (𝓝 x) := by
  rw [hasDerivAt_iff_isLittleO, Asymptotics.isLittleO_iff]
  simp only [KleinO, Real.norm_eq_abs, smul_eq_mul, gt_iff_lt]

/-- Aufgabe B3 (a) (A2 (b)): `y² + 3y` hat in `x` die Ableitung `2x + 3` --
    aus der Definition, wie `y²` in der Vorlesung. -/
theorem ableitung_quadrat_plus (x : ℝ) :
    HasDerivAt (fun y : ℝ => y ^ 2 + 3 * y) (2 * x + 3) x := by
  -- Nuetzlich: rw [ableitung_klein_o]; der Fehler der Tangente ist (y - x)^2
  -- (have fehler : (fun y => ...) = fun y => (y - x) ^ 2, ext y, ring);
  -- dann quadrat_klein_o
  --<<loesung>>
  rw [ableitung_klein_o]
  have fehler : (fun y : ℝ => (y ^ 2 + 3 * y) - (x ^ 2 + 3 * x) - (y - x) * (2 * x + 3)) =
      fun y => (y - x) ^ 2 := by
    funext y
    ring
  rw [fehler]                          -- Ziel: (y - x)² ist klein gegen y - x
  exact quadrat_klein_o x
  --<<loesungEnd>>

/-- Aufgabe B3 (b) (A2 (c)): Der Betrag ist in `0` nicht klein gegen `y`:
    Mit `c = 1/2` muesste schliesslich `|y| ≤ (1/2)·|y|` gelten. -/
theorem betrag_nicht_klein_o :
    ¬ KleinO (fun y : ℝ => |y|) (fun y => y) (𝓝 0) := by
  -- Nuetzlich: intro h; h (1 / 2) (by norm_num); Metric.eventually_nhds_iff
  -- liefert ein δ > 0; der Punkt δ / 2 liegt in der Kugel (Real.dist_eq,
  -- abs_of_pos, half_pos) und verletzt die Ungleichung; abs_abs, linarith
  --<<loesung>>
  intro h
  have h1 := h (1 / 2) (by norm_num)               -- schliesslich |(|y|)| ≤ (1/2)·|y|
  rw [Metric.eventually_nhds_iff] at h1
  obtain ⟨δ, hδ, h2⟩ := h1                         -- gilt auf der Kugel vom Radius δ
  have hd : δ / 2 > 0 := half_pos hδ
  have h3 := h2 (y := δ / 2) (by
    rw [Real.dist_eq, sub_zero, abs_of_pos hd]     -- δ/2 liegt in der Kugel
    linarith)
  rw [abs_abs, abs_of_pos hd] at h3                -- h3 : δ/2 ≤ (1/2)·(δ/2)
  linarith
  --<<loesungEnd>>

/-- Aufgabe B3 (c) (A2 (c)): Der Betrag hat in `0` keine Ableitung. -/
theorem betrag_keine_ableitung :
    ¬ ∃ f' : ℝ, HasDerivAt (fun y : ℝ => |y|) f' 0 := by
  -- Nuetzlich: In 0 liegt ein lokales Minimum (IsLocalMin, Eventually.of_forall),
  -- also ist f' = 0 nach Fermat: IsLocalMin.hasDerivAt_eq_zero. Dann
  -- ableitung_klein_o und (b); simpa
  --<<loesung>>
  rintro ⟨f', hf⟩
  have hmin : IsLocalMin (fun y : ℝ => |y|) 0 :=
    Eventually.of_forall fun y => by simp               -- |0| ≤ |y|
  have h0 : f' = 0 := hmin.hasDerivAt_eq_zero hf        -- Fermat
  subst h0
  rw [ableitung_klein_o] at hf                          -- der Fehler der Tangente 0
  exact betrag_nicht_klein_o (by simpa using hf)
  --<<loesungEnd>>
--<<b3End>>

/-! ## B4: Partielle Ableitungen (zu A3) -/

--<<b4>>
/-- Partielle Ableitung, wie in der Vorlesung: `f` hat in `x` nach der
    `j`-ten Koordinate die partielle Ableitung `a`, wenn die Schnittkurve
    `t ↦ f(x + t·e_j)` in `0` die Ableitung `a` hat. Punkte der Ebene sind
    Vektoren `v : Fin 2 → ℝ` mit den Koordinaten `v 0` und `v 1`. -/
def HatPartielleAbleitung {n : ℕ} {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : (Fin n → ℝ) → F) (x : Fin n → ℝ) (j : Fin n) (a : F) : Prop :=
  HasDerivAt (fun t : ℝ => f (x + t • (Pi.single j 1 : Fin n → ℝ))) a 0

/-- Vorgegeben: das Beispiel der Vorlesung, `f(x, y) = x²y` nach `x`. -/
example (v : Fin 2 → ℝ) :
    HatPartielleAbleitung (fun w : Fin 2 → ℝ => w 0 ^ 2 * w 1) v 0 (2 * v 0 * v 1) := by
  unfold HatPartielleAbleitung
  have e : (fun t : ℝ => (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 0 ^ 2 *
      (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 1) = fun t => (v 0 + t) ^ 2 * v 1 := by
    funext t
    simp
  rw [e]
  have h := (((hasDerivAt_id (0 : ℝ)).const_add (v 0)).pow 2).mul_const (v 1)
  convert h using 1
  simp

/-- Aufgabe B4 (a) (A3 (a)): `f(x, y) = x²y + 3y²` hat in `v` die partielle
    Ableitung `x² + 6y` nach `y`. -/
theorem partiell_nach_y (v : Fin 2 → ℝ) :
    HatPartielleAbleitung (fun w : Fin 2 → ℝ => w 0 ^ 2 * w 1 + 3 * w 1 ^ 2) v 1
      (v 0 ^ 2 + 6 * v 1) := by
  -- Nuetzlich: wie das Beispiel x²y oben: unfold HatPartielleAbleitung; die
  -- Schnittkurve ausrechnen (funext t, simp); hasDerivAt_id, .const_add, .pow,
  -- .const_mul, .add; convert ... using 1; simp; ring
  --<<loesung>>
  unfold HatPartielleAbleitung
  -- Die Schnittkurve: x bleibt, y wird v₁ + t.
  have e : (fun t : ℝ => (v + t • (Pi.single 1 1 : Fin 2 → ℝ)) 0 ^ 2 *
      (v + t • (Pi.single 1 1 : Fin 2 → ℝ)) 1 +
      3 * (v + t • (Pi.single 1 1 : Fin 2 → ℝ)) 1 ^ 2) =
      fun t => v 0 ^ 2 * (v 1 + t) + 3 * (v 1 + t) ^ 2 := by
    funext t
    simp
  rw [e]
  -- v₁ + t hat die Ableitung 1; dann Faktor, Quadrat und Summe.
  have h1 : HasDerivAt (fun t : ℝ => v 1 + t) 1 0 := (hasDerivAt_id (0 : ℝ)).const_add (v 1)
  have h := (h1.const_mul (v 0 ^ 2)).add ((h1.pow 2).const_mul 3)
  convert h using 1
  simp
  ring
  --<<loesungEnd>>

/-- In Lean ist `a / 0 = 0`, also `f(0, 0) = 0` von selbst. -/
noncomputable def gegenbeispiel (w : Fin 2 → ℝ) : ℝ :=
  w 0 * w 1 / (w 0 ^ 2 + w 1 ^ 2)

/-- Aufgabe B4 (b) (A3 (d)): Auf der `y`-Achse ist `f` null; die partielle
    Ableitung nach `y` in `0` ist `0`. -/
theorem gegenbeispiel_partiell_y : HatPartielleAbleitung gegenbeispiel 0 1 0 := by
  -- Nuetzlich: wie die partielle Ableitung nach x in der Vorlesung: Die
  -- Schnittkurve ist die Nullfunktion (funext t, simp [gegenbeispiel]);
  -- hasDerivAt_const
  --<<loesung>>
  unfold HatPartielleAbleitung
  have h : (fun t : ℝ => gegenbeispiel (0 + t • (Pi.single 1 1 : Fin 2 → ℝ))) =
      fun _ => 0 := by
    funext t
    simp [gegenbeispiel]
  rw [h]
  exact hasDerivAt_const 0 0
  --<<loesungEnd>>

/-- Aufgabe B4 (c) (A3 (d)): Auf der Diagonalen ist `f = 1/2`, ausser in `0`. -/
theorem gegenbeispiel_diagonale {t : ℝ} (ht : t ≠ 0) :
    gegenbeispiel ![t, t] = 1 / 2 := by
  -- Nuetzlich: simp only [gegenbeispiel, Matrix.cons_val_zero, Matrix.cons_val_one],
  -- field_simp, ring
  --<<loesung>>
  simp only [gegenbeispiel, Matrix.cons_val_zero, Matrix.cons_val_one]
  field_simp
  ring
  --<<loesungEnd>>
--<<b4End>>

/-! ## B5: Jacobi-Matrix und Kettenregel (zu A4) -/

--<<b5>>
-- Vorgegeben, wie in der Vorlesung: eine Matrix als lineare Abbildung, das
-- Hintereinanderausfuehren als Matrixprodukt und die Jacobi-Matrix von
-- u² + v². Punkte sind Vektoren `v : Fin 2 → ℝ` mit `v 0 = x`, `v 1 = y`.

/-- Eine m×n-Matrix als lineare Abbildung ℝⁿ → ℝᵐ, h ↦ M h. -/
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
  ext h : 1
  simp [Matrix.mulVec_mulVec]

/-- g(u, v) = u² + v², als Abbildung nach ℝ¹: Jacobi-Matrix (2u, 2v). -/
theorem jacobi_quadratsumme (u : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 ^ 2 + w 1 ^ 2])
      (matrixAbb !![2 * u 0, 2 * u 1]) u := by
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  have h := ((hasFDerivAt_apply (𝕜 := ℝ) 0 u).pow 2).add
    ((hasFDerivAt_apply (𝕜 := ℝ) 1 u).pow 2)
  convert h using 1
  ext k
  simp [dotProduct, Fin.sum_univ_two]

/-- Aufgabe B5 (a) (A4 (a)): `F(x, y) = (x² - y², 2xy)` hat die Jacobi-Matrix
    `[[2x, -2y], [2y, 2x]]`. -/
theorem jacobi_quadrat (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 ^ 2 - w 1 ^ 2, 2 * w 0 * w 1])
      (matrixAbb !![2 * v 0, -(2 * v 1); 2 * v 1, 2 * v 0]) v := by
  -- Nuetzlich: wie jacobi_xy in der Vorlesung: rw [hasFDerivAt_pi'], intro i,
  -- fin_cases i, hasFDerivAt_apply (𝕜 := ℝ) 0 v, HasFDerivAt.pow, .sub,
  -- .const_mul, .mul, convert ... using 1, ext k,
  -- simp [dotProduct, Fin.sum_univ_two], ring
  --<<loesung>>
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  · -- erste Zeile: x² - y²
    have h := ((hasFDerivAt_apply (𝕜 := ℝ) 0 v).pow 2).sub
      ((hasFDerivAt_apply (𝕜 := ℝ) 1 v).pow 2)
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]
    ring
  · -- zweite Zeile: 2xy
    have h := ((hasFDerivAt_apply (𝕜 := ℝ) 0 v).const_mul 2).mul
      (hasFDerivAt_apply (𝕜 := ℝ) 1 v)
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]
    ring
  --<<loesungEnd>>

/-- Aufgabe B5 (b) (A4 (c), Kettenregel): `g ∘ F` hat die Jacobi-Matrix
    `J_g(F(v)) · J_F(v)`. -/
theorem kette_quadrat (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![(w 0 ^ 2 - w 1 ^ 2) ^ 2 + (2 * w 0 * w 1) ^ 2])
      (matrixAbb (!![2 * (v 0 ^ 2 - v 1 ^ 2), 2 * (2 * v 0 * v 1)] *
        !![2 * v 0, -(2 * v 1); 2 * v 1, 2 * v 0])) v := by
  -- Nuetzlich: wie kettenregel_polar in der Vorlesung: HasFDerivAt.comp mit
  -- jacobi_quadratsumme an der Stelle ![v 0 ^ 2 - v 1 ^ 2, 2 * v 0 * v 1] und
  -- jacobi_quadrat v; matrixAbb_comp; simpa
  --<<loesung>>
  have h := (jacobi_quadratsumme ![v 0 ^ 2 - v 1 ^ 2, 2 * v 0 * v 1]).comp v
    (jacobi_quadrat v)
  rw [matrixAbb_comp] at h
  simpa using h
  --<<loesungEnd>>

/-- Aufgabe B5 (c) (A4 (c), Rechnung): Das Produkt der Jacobi-Matrizen ist
    `4(x² + y²)·(x, y)` -- der Gradient von `(x² + y²)²`. -/
theorem produkt_quadrat (x y : ℝ) :
    !![2 * (x ^ 2 - y ^ 2), 2 * (2 * x * y)] * !![2 * x, -(2 * y); 2 * y, 2 * x] =
      !![4 * x * (x ^ 2 + y ^ 2), 4 * y * (x ^ 2 + y ^ 2)] := by
  -- Nuetzlich: ext i j, fin_cases i, fin_cases j,
  -- simp [Matrix.mul_apply, Fin.sum_univ_two], ring
  --<<loesung>>
  ext i j
  fin_cases i
  fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two] <;> ring
  --<<loesungEnd>>
--<<b5End>>

end Uebung05
