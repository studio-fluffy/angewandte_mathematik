/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 6
  (Folien: folien/vorlesung06.tex)

  Fuer etwa eine Stunde, in der Reihenfolge der Folien: Jedes Beispiel geht den Weg der
  Folie, mit Mathlibs Begriffen und von Hand; Mathlibs fertige Saetze stehen danach zum
  Vergleich. Weitere Beispiele und laengere Beweise stehen in code/Kapitel4.lean (Skript).
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

/-! ## 4.4 Taylor in mehreren Veraenderlichen; Hoch-, Tief- und Sattelpunkte -/

-- Folie: "Beweis (2): die Details -- die Kettenregel, Schritt fuer Schritt" (Schritte 1
-- und 2); die Idee dazu auf "Beweis (2): die Idee -- jede Ableitung bringt ein h"
/- Die Aussage Stueck fuer Stueck:
     {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
         E ist ein normierter Vektorraum ueber ℝ, etwa ℝ². Norm und Vektorraum
         stehen in den eckigen Klammern; Lean sucht sie selbst.
     {f : E → ℝ}              eine Funktion, etwa f(x, y) = x²y
     {f' : E →L[ℝ] ℝ}         eine stetige lineare Abbildung E → ℝ (L wie linear),
                              die Ableitung von f
     {p h : E} {t : ℝ}        ein Punkt p, eine Richtung h, eine Zahl t
     (hf : HasFDerivAt f f' (p + t • h))
         Annahme hf: f ist im Punkt x = p + t·h differenzierbar mit der Ableitung
         f'. Das heisst: f(x + k) = f(x) + f' k + r(k) mit einem Rest r(k) = o(k)
         fuer k → 0 (`hasFDerivAt_iff_isLittleO_nhds_zero`). Auf ℝ² ist
         f' k = ∂₁f(x)·k₁ + ∂₂f(x)·k₂. F wie Frechet: die Ableitung in mehreren
         Veraenderlichen, eine lineare Abbildung.
     HasDerivAt (fun s : ℝ => f (p + s • h)) (f' h) t
         Behauptung: Die Schnittkurve g(s) = f(p + s·h) hat in s = t die Ableitung
         g'(t) = f' h, eine Zahl (`HasDerivAt`: Ableitung in einer Veraenderlichen).
   Die Argumente in geschweiften Klammern sind implizit: Lean liest sie beim
   Benutzen aus dem Ziel und aus hf ab. `s • h` ist Zahl mal Vektor.
   Beispiel: f(x, y) = x²y, p = (1, 1), t = 0. Dort ist ∂₁f = 2xy = 2 und
   ∂₂f = x² = 1, also f' k = 2k₁ + k₂ und g'(0) = f' h = 2h₁ + h₂. Probe:
   g(s) = (1 + s·h₁)²(1 + s·h₂) hat die Ableitung 2h₁ + h₂ bei s = 0. -/
/-- Die Kettenregel fuer die Schnittkurve: g = f ∘ γ mit der Geraden γ(s) = p + s·h hat
in jedem t die Ableitung Df(p + t·h) h. Das ist die Kettenregel
D(f ∘ γ)(t) = Df(γ(t)) Dγ(t) mit Dγ(t) = h; bei t = 0 ist das die Richtungsableitung
in Richtung h. Wiederholt angewandt gibt das g^(j)(t) = D^j f(p + t·h) h^j, den Kern des
Beweises der Taylorformel. -/
theorem schnittkurve_ableitung {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : E → ℝ} {f' : E →L[ℝ] ℝ} {p h : E} {t : ℝ} (hf : HasFDerivAt f f' (p + t • h)) :
    HasDerivAt (fun s : ℝ => f (p + s • h)) (f' h) t := by
  -- Schritt 1 (= 1. auf der Folie, die Gerade ableiten): γ(s) = p + s·h hat die Ableitung
  -- Dγ(t) = (s ↦ s·h), die Spalte h (`smulRight h`). Die Identitaet hat die Ableitung id
  -- (`hasFDerivAt_id`), mal den festen Vektor h (`smul_const`), plus p (`const_add`).
  have hγ : HasFDerivAt (fun s : ℝ => p + s • h)
      ((ContinuousLinearMap.id ℝ ℝ).smulRight h) t :=
    ((hasFDerivAt_id t).smul_const h).const_add p
  -- Schritt 2 (= 2., die Kettenregel, `HasFDerivAt.comp`): D(f ∘ γ)(t) = Df(γ(t)) ∘ Dγ(t),
  -- also
  --   hk : HasFDerivAt (f ∘ fun s => p + s • h)
  --          (f'.comp ((ContinuousLinearMap.id ℝ ℝ).smulRight h)) t
  have hk := hf.comp t hγ
  -- Schritt 3: Eine Ableitung von ℝ nach ℝ ist eine Zahl, der Wert der linearen Abbildung
  -- bei 1 (`HasFDerivAt.hasDerivAt`).
  -- Hier ist das (f' ∘ Dγ(t)) 1 = f' (1·h) = f' h; `simpa` rechnet es aus und schreibt das
  -- ∘ aus (`Function.comp_def`).
  simpa [Function.comp_def] using hk.hasDerivAt

-- Folie: "Vorgerechnet: Taylor in zwei Variablen" (das Polynom x³ + y³ - 3xy in (1, 1)),
-- mit dem Beweisweg von "Beweis (1): zurueck auf eine Variable" und "Beweis (3): einsetzen"
/-- Das Beispiel f(x, y) = x³ + y³ - 3xy. Kritisch sind (0, 0) und (1, 1). -/
def kubisch (p : ℝ × ℝ) : ℝ := p.1 ^ 3 + p.2 ^ 3 - 3 * p.1 * p.2

/-- Taylor in einer Variablen, aus Mathlib, fuer ein Polynom vom Grad 3: Ist g analytisch
(`ContDiff ℝ ⊤ g`, etwa ein Polynom) und g'''' = 0, so ist g gleich seinem
Taylorpolynom: g(1) = g(0) + g'(0) + g''(0)/2 + g'''(0)/6. -/
theorem kubisch_taylor_eine_variable {g : ℝ → ℝ} (hg : ContDiff ℝ ⊤ g)
    (h4 : iteratedDeriv 4 g = 0) :
    g 1 = g 0 + deriv g 0 + iteratedDeriv 2 g 0 / 2 + iteratedDeriv 3 g 0 / 6 := by
  -- Taylor mit Lagrange-Restglied auf [0, 1], Ordnung 3 (Mathlib): Es gibt τ ∈ (0, 1) mit
  -- g(1) - T₃(1) = g''''(τ) · 1⁴ / 4!, und das ist 0.
  obtain ⟨τ, -, hτ⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (n := 3) zero_lt_one
    (hg.of_le le_top).contDiffOn
  -- Das Taylorpolynom T₃(1) = Σ g^(k)(0) / k! (k ≤ 3); die Ableitungen auf [0, 1] in 0 sind
  -- die gewoehnlichen, denn g ist ueberall differenzierbar.
  have hk (k : ℕ) : iteratedDerivWithin k g (Set.Icc 0 1) 0 = iteratedDeriv k g 0 :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_Icc zero_lt_one)
      (hg.contDiffAt.of_le le_top) (by simp)
  rw [taylor_within_apply] at hτ
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, hk, h4] at hτ
  norm_num [Nat.factorial] at hτ
  linarith

/-- Taylor in (1, 1): Ein Polynom vom Grad 3 ist gleich seinem Taylorpolynom dritter
Ordnung. Es ist f(1, 1) = -1, der Gradient ist 0, die Hesse-Matrix [[6, -3], [-3, 6]]
gibt den quadratischen Teil, h₁³ + h₂³ ist das Glied dritter Ordnung. Beweis wie auf den
Folien: zurueck auf eine Variable, dort Taylor aus Mathlib, dann einsetzen. -/
theorem kubisch_taylorformel (h : ℝ × ℝ) :
    kubisch ((1, 1) + h) =
      -1 + (6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2) / 2 + (h.1 ^ 3 + h.2 ^ 3) := by
  -- Schritt 1 (zurueck auf eine Variable): die Schnittkurve g(t) = f((1, 1) + t·h), in
  -- Koordinaten ausgeschrieben.
  set g := fun t : ℝ => kubisch ((1, 1) + t • h) with hg
  have hg' : g = fun t => (1 + t * h.1) ^ 3 + (1 + t * h.2) ^ 3
      - 3 * (1 + t * h.1) * (1 + t * h.2) := by
    funext t; simp [g, kubisch]
  -- Schritt 2 (die Ableitungen von g, mit den Regeln): g', g'', g''' und g'''' = 0.
  -- `simp` wendet Summen-, Produkt- und Kettenregel an, `fun_prop` liefert dabei die
  -- Differenzierbarkeit; `ring` ordnet das Ergebnis.
  have d1 : deriv g = fun t => 3 * h.1 * (1 + t * h.1) ^ 2 + 3 * h.2 * (1 + t * h.2) ^ 2
      - 3 * h.1 * (1 + t * h.2) - 3 * h.2 * (1 + t * h.1) := by
    rw [hg']; funext t; simp (disch := fun_prop); ring
  have d2 : iteratedDeriv 2 g = fun t => 6 * h.1 ^ 2 * (1 + t * h.1)
      + 6 * h.2 ^ 2 * (1 + t * h.2) - 6 * h.1 * h.2 := by
    rw [iteratedDeriv_succ, iteratedDeriv_one, d1]; funext t; simp (disch := fun_prop); ring
  have d3 : iteratedDeriv 3 g = fun _ => 6 * h.1 ^ 3 + 6 * h.2 ^ 3 := by
    rw [iteratedDeriv_succ, d2]; funext t; simp (disch := fun_prop); ring
  have d4 : iteratedDeriv 4 g = 0 := by
    rw [iteratedDeriv_succ, d3]; funext t; simp
  -- Schritt 3 (Taylor in einer Variablen, aus Mathlib): g(1) = g(0) + g'(0) + g''(0)/2
  -- + g'''(0)/6, denn die vierte Ableitung ist 0.
  have ht := kubisch_taylor_eine_variable (by rw [hg']; fun_prop) d4
  -- Schritt 4 (einsetzen): g(1) = f((1, 1) + h), g(0) = f(1, 1) = -1, g'(0) = 0 (der
  -- Gradient ist 0), g''(0) = 6h₁² - 6h₁h₂ + 6h₂² = hᵀ H h, g'''(0) = 6(h₁³ + h₂³).
  rw [d1, d2, d3] at ht
  simp only [g, one_smul, zero_smul, add_zero] at ht
  rw [ht]
  norm_num [kubisch]
  ring

-- Folie: "Tief-, Hoch- und Sattelpunkte"
/-- Mathlibs `IsLocalMin f p` heisst: Schliesslich laengs 𝓝 p ist f p ≤ f x. Klassisch
ausgeschrieben, wie in der Definition der Folie: Auf einer Kugel um p liegt f nirgends
unter f p. (`IsLocalMax` ebenso mit f x ≤ f p.) -/
theorem lokales_minimum_iff {E : Type*} [MetricSpace E] (f : E → ℝ) (p : E) :
    IsLocalMin f p ↔ ∃ δ > 0, ∀ x, dist x p < δ → f p ≤ f x := by
  -- Schritt 1: die Definition. "Schliesslich" heisst: Die Menge {x | f p ≤ f x} ist gross
  -- laengs 𝓝 p, sie liegt im Filter.
  change {x | f p ≤ f x} ∈ 𝓝 p ↔ _
  -- Schritt 2: Gross laengs 𝓝 p heisst, eine Kugel B(p, δ) liegt darin
  -- (`Metric.mem_nhds_iff`), und x ∈ B(p, δ) heisst dist x p < δ (`Metric.mem_ball`).
  simp only [Metric.mem_nhds_iff, Set.subset_def, Metric.mem_ball, Set.mem_setOf_eq]

-- Folie: "Optimieren: Am lokalen Extremum ist Df(x) = 0"
/-- Fermat in mehreren Veraenderlichen, mit dem Beweis von der Folie, hier fuer ein
Minimum (ein Maximum von f ist ein Minimum von -f): Die Schnittkurve g(t) = f(x + t·v)
hat bei 0 ein lokales Minimum und die Ableitung Df(x) v (`schnittkurve_ableitung` bei
t = 0). Nach Fermat in einer Variablen ist g'(0) = 0 -- fuer jedes v, also ist
Df(x) = 0. Fermat in einer Variablen nehmen wir aus Mathlib; mit dem eigenen Beweis:
`fermat_mehrdim` in code/Kapitel4.lean (Skript). -/
theorem fermat_mehrdim_zu_fuss {n : ℕ} {f : (Fin n → ℝ) → ℝ} {x : Fin n → ℝ}
    {f' : (Fin n → ℝ) →L[ℝ] ℝ} (h : IsLocalMin f x) (hf : HasFDerivAt f f' x) :
    f' = 0 := by
  -- Schritt 1 ("Sei v ∈ ℝⁿ"): Lineare Abbildungen sind gleich, wenn sie an jedem
  -- Vektor v gleich sind (`ext v`). Ziel: f' v = 0 v.
  ext v
  -- Schritt 2 ("g'(0) = Df(x) v"): die Kettenregel fuer die Schnittkurve bei t = 0; dort
  -- ist x + 0·v = x, das erledigt `simpa`.
  have hg : HasDerivAt (fun t : ℝ => f (x + t • v)) (f' v) 0 :=
    schnittkurve_ableitung (by simpa using hf)
  -- Schritt 3 ("g hat bei 0 ein lokales Minimum"): Fuer t → 0 geht x + t·v → x, und
  -- nahe x ist f x ≤ f y (h). `hv.eventually h` zieht das auf t zurueck.
  have hv : Tendsto (fun t : ℝ => x + t • v) (𝓝 0) (𝓝 x) := by
    simpa using (tendsto_id (x := 𝓝 (0 : ℝ))).smul_const v |>.const_add x
  have hmin : IsLocalMin (fun t : ℝ => f (x + t • v)) 0 := by
    simpa [IsLocalMin, IsMinFilter] using hv.eventually h
  -- Schritt 4 ("Fermat in einer Variablen: g'(0) = 0"), in Mathlib
  -- `IsLocalMin.hasDerivAt_eq_zero`; `simpa` schreibt noch (0 : _ →L[ℝ] ℝ) v = 0.
  simpa using hmin.hasDerivAt_eq_zero hg

-- Mathlib hat den ganzen Satz, fuer jeden normierten Raum: `IsLocalMin.hasFDerivAt_eq_zero`.

-- Klein-o laengs eines Filters wie in Vorlesung 5, fuer den Rest o(‖h‖²) der Taylorformel.
/-- Klein-o laengs eines Filters: Zu jedem c > 0 gilt schliesslich
|f x| ≤ c·|g x|. Erst kommt c, dann die grosse Menge -- zu jedem c eine
eigene. `∀ᶠ x in l, P x` heisst: Die Menge, auf der P gilt, ist gross. -/
def KleinO {α : Type*} (f g : α → ℝ) (l : Filter α) : Prop :=
  ∀ c > 0, ∀ᶠ x in l, |f x| ≤ c * |g x|

-- Hier arbeitet die Taylorformel zweiter Ordnung: Im kritischen Punkt p ist
-- f(p + h) = f(p) + Q(h)/2 + o(‖h‖²) mit Q(h) = hᵀ H_f(p) h; das ist die Voraussetzung hf.
-- Lean zeigt f(p) ≤ f(p + h) nahe p, also ein lokales Minimum, aber nicht "strikt".
-- Folie: "Das Hesse-Kriterium" und "Hesse-Kriterium (a) im Bild: Schranke und Band"
-- (die Schranke Q(h) ≥ m·‖h‖² und das Band mit c = m/4)
/-- Hesse-Kriterium, Teil (a), in der Form des Beweises: Ist an einem kritischen Punkt
  f(p + h) = f(p) + Q(h)/2 + o(‖h‖²)  laengs 𝓝 0
und Q(h) ≥ m·‖h‖² mit m > 0 (Q positiv definit), so hat f in p ein lokales Minimum.
Q ist die quadratische Form der Hesse-Matrix H, Q(h) = ⟨h, H h⟩. -/
theorem min_aus_taylor {E : Type*} [NormedAddCommGroup E] {f : E → ℝ} {p : E}
    (Q : E → ℝ) {m : ℝ} (hm : 0 < m) (hQ : ∀ h, m * ‖h‖ ^ 2 ≤ Q h)
    (hf : KleinO (fun h => f (p + h) - f p - Q h / 2) (fun h => ‖h‖ ^ 2) (𝓝 0)) :
    IsLocalMin f p := by
  -- Schritt 1 (o mit c = m/4): Schliesslich laengs 𝓝 0 ist der Rest
  -- r(h) = f(p + h) - f(p) - Q(h)/2 betragsmaessig hoechstens (m/4)·‖h‖².
  have h1 := hf (m / 4) (by positivity)
  -- Schritt 2: Dort ist f(p + h) - f(p) = Q(h)/2 + r(h) ≥ (m/2)‖h‖² - (m/4)‖h‖² ≥ 0.
  have h2 : ∀ᶠ h in 𝓝 (0 : E), f p ≤ f (p + h) := by
    filter_upwards [h1] with h hh
    rw [abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖h‖ ^ 2)] at hh
    have hr := (abs_le.1 hh).1                -- -(m/4)‖h‖² ≤ r(h)
    have hq := hQ h                           -- m‖h‖² ≤ Q(h)
    nlinarith [mul_nonneg hm.le (sq_nonneg ‖h‖)]
  -- Schritt 3: von h zum Punkt x = p + h. Geht x gegen p, so geht h = x - p gegen 0;
  -- `ht.eventually` macht aus "schliesslich laengs 𝓝 0" "schliesslich laengs 𝓝 p".
  have ht : Tendsto (fun x => x - p) (𝓝 p) (𝓝 0) := by
    simpa using (tendsto_id (x := 𝓝 p)).sub_const p
  filter_upwards [ht.eventually h2] with x hx
  simpa using hx                              -- p + (x - p) = x

-- Folie: "In zwei Dimensionen: Determinante und Spur"
/-- Die Identitaet hinter dem Kriterium in zwei Dimensionen: Fuer die quadratische Form
Q(h, k) = a h² + 2b hk + c k² der Matrix [[a, b], [b, c]] ist
Spur · Q - Determinante · (h² + k²) eine Summe von zwei Quadraten. -/
theorem spur_det_identitaet (a b c h k : ℝ) :
    (a + c) * (a * h ^ 2 + 2 * b * h * k + c * k ^ 2)
      - (a * c - b ^ 2) * (h ^ 2 + k ^ 2) = (a * h + b * k) ^ 2 + (b * h + c * k) ^ 2 := by
  ring

/-- Ist a > 0 und die Determinante ac - b² > 0, so ist Q positiv definit, genauer
Q(h, k) ≥ m·(h² + k²) mit m = Determinante / Spur > 0. -/
theorem quadratische_form_schranke (a b c : ℝ) (ha : 0 < a) (hdet : 0 < a * c - b ^ 2)
    (h k : ℝ) :
    (a * c - b ^ 2) / (a + c) * (h ^ 2 + k ^ 2) ≤ a * h ^ 2 + 2 * b * h * k + c * k ^ 2 := by
  -- Schritt 1: Auch c > 0, sonst waere ac ≤ 0 und ac - b² ≤ 0. Also ist die Spur a + c > 0.
  have hc : 0 < c := by
    by_contra hc
    push_neg at hc
    nlinarith [sq_nonneg b]
  have hs : 0 < a + c := by linarith
  -- Schritt 2: mit der Spur durchmultiplizieren (`div_le_iff₀`). Der Unterschied ist nach
  -- der Identitaet eine Summe von Quadraten, also ≥ 0.
  rw [div_mul_eq_mul_div, div_le_iff₀ hs]
  nlinarith [spur_det_identitaet a b c h k, sq_nonneg (a * h + b * k),
    sq_nonneg (b * h + c * k)]

-- Folie: "Vorgerechnet: f(x, y) = x^3 + y^3 - 3xy -- die kritischen Punkte" und
-- "... -- was die Hesse-Matrix sagt": das Beispiel, ganz mit den Saetzen von oben
/-- Schritt 1: (1, 1) ist kritisch, Df(1, 1) = 0. Die Regeln geben
Df(x, y) = (3x² - 3y) dx + (3y² - 3x) dy, in (1, 1) die Nullabbildung. -/
theorem kubisch_kritisch : HasFDerivAt kubisch (0 : ℝ × ℝ →L[ℝ] ℝ) (1, 1) := by
  -- x = p.1 und y = p.2 haben die Ableitungen dx und dy (`hasFDerivAt_fst`, `_snd`);
  -- Potenz-, Summen- und Produktregel setzen x³ + y³ - 3xy daraus zusammen.
  have hx := hasFDerivAt_fst (𝕜 := ℝ) (p := ((1 : ℝ), (1 : ℝ)))
  have hy := hasFDerivAt_snd (𝕜 := ℝ) (p := ((1 : ℝ), (1 : ℝ)))
  have h := ((hx.pow 3).add (hy.pow 3)).sub ((hx.const_mul (3 : ℝ)).mul hy)
  -- Bleibt: Diese lineare Abbildung ist 0 -- an (1, 0) und an (0, 1) nachgerechnet.
  unfold kubisch
  convert h using 1
  ext <;> simp

/-- In der Maximumsnorm ist ‖h‖² = max(h₁², h₂²) ≤ h₁² + h₂². -/
theorem norm_sq_le_quadratsumme (h : ℝ × ℝ) : ‖h‖ ^ 2 ≤ h.1 ^ 2 + h.2 ^ 2 := by
  rw [Prod.norm_def, Real.norm_eq_abs, Real.norm_eq_abs]
  -- Das Maximum ist |h₁| oder |h₂|; beide Male ist sein Quadrat hoechstens die Summe.
  rcases max_cases |h.1| |h.2| with ⟨hm, _⟩ | ⟨hm, _⟩
  · rw [hm, sq_abs]; nlinarith [sq_nonneg h.2]
  · rw [hm, sq_abs]; nlinarith [sq_nonneg h.1]

/-- Schritt 2: (1, 1) ist ein Tiefpunkt, nach dem Hesse-Kriterium `min_aus_taylor`. Die
Hesse-Matrix in (1, 1) ist [[6, -3], [-3, 6]], also Q(h) = 6h₁² - 6h₁h₂ + 6h₂²; mit a = 6 > 0
und Determinante 27 > 0 ist Q(h) ≥ (27/12)(h₁² + h₂²) ≥ (9/4)‖h‖². Der Rest der
Taylorformel (`kubisch_taylorformel`) ist h₁³ + h₂³, und der ist o(‖h‖²). Einen linearen
Teil hat die Entwicklung nicht, denn (1, 1) ist kritisch (Schritt 1). (Ausfuehrlicher,
mit dem Sattel in (0, 0): `kubisch_tiefpunkt`, `kubisch_sattel` in code/Kapitel4.lean.) -/
theorem kubisch_tiefpunkt_zu_fuss : IsLocalMin kubisch (1, 1) := by
  apply min_aus_taylor (fun h => 6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2)
    (by norm_num : (0 : ℝ) < 9 / 4)
  · -- Q ist positiv definit: das Kriterium in zwei Dimensionen, dazu ‖h‖² ≤ h₁² + h₂².
    intro h
    have h1 := quadratische_form_schranke 6 (-3) 6 (by norm_num) (by norm_num) h.1 h.2
    have h2 := norm_sq_le_quadratsumme h
    norm_num at h1
    nlinarith
  · -- Der Rest f((1, 1) + h) - f(1, 1) - Q(h)/2 ist nach der Taylorformel
    -- (`kubisch_taylorformel`, mit f(1, 1) = -1) das Glied dritter Ordnung h₁³ + h₂³.
    have e : (fun h : ℝ × ℝ => kubisch ((1, 1) + h) - kubisch (1, 1)
        - (6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2) / 2) = fun h => h.1 ^ 3 + h.2 ^ 3 := by
      funext h
      have h11 : kubisch (1, 1) = -1 := by norm_num [kubisch]
      rw [kubisch_taylorformel, h11]
      ring
    rw [e]
    -- Der Rest ist o(‖h‖²): Zu c > 0 nimmt man die Kugel vom Radius c/2. Dort ist
    -- |h₁|, |h₂| ≤ ‖h‖ und |h₁³ + h₂³| ≤ 2‖h‖³ ≤ c‖h‖².
    intro c hc
    filter_upwards [Metric.ball_mem_nhds (0 : ℝ × ℝ) (half_pos hc)] with h hh
    rw [Metric.mem_ball, dist_zero_right] at hh          -- hh : ‖h‖ < c/2
    have h1 : |h.1| ≤ ‖h‖ := by simpa using norm_fst_le h
    have h2 : |h.2| ≤ ‖h‖ := by simpa using norm_snd_le h
    rw [abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖h‖ ^ 2)]
    calc |h.1 ^ 3 + h.2 ^ 3| ≤ |h.1| ^ 3 + |h.2| ^ 3 := by
          rw [← abs_pow, ← abs_pow]
          exact abs_add_le _ _
      _ ≤ ‖h‖ ^ 3 + ‖h‖ ^ 3 := by gcongr
      _ = 2 * ‖h‖ * ‖h‖ ^ 2 := by ring
      _ ≤ c * ‖h‖ ^ 2 := by gcongr; linarith

/-! ## 4.5 Gradientenabstieg -/

-- Folie: "Der negative Gradient zeigt bergab"
/-- Steilster Abstieg, mit g = ∇f(x): In jede Richtung v mit ‖v‖ = 1 ist die
Richtungsableitung ⟨g, v⟩ ≥ -‖g‖, und die Richtung -g/‖g‖ erreicht -‖g‖. -/
theorem steilster_abstieg {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (g : V) :
    (∀ v : V, ‖v‖ = 1 → -‖g‖ ≤ inner ℝ g v) ∧
      (g ≠ 0 → inner ℝ g ((-‖g‖⁻¹) • g) = -‖g‖) := by
  constructor
  · -- Teil 1, Cauchy-Schwarz: |⟨g, v⟩| ≤ ‖g‖·‖v‖ = ‖g‖ (`abs_real_inner_le_norm`), und
    -- |a| ≤ b heisst -b ≤ a ≤ b (`abs_le`); die untere Haelfte ist das Ziel.
    intro v hv
    have h := abs_real_inner_le_norm g v
    rw [hv, mul_one] at h
    exact (abs_le.1 h).1
  · -- Teil 2: das Skalar herausziehen (`inner_smul_right`), ⟨g, g⟩ = ‖g‖²
    -- (`real_inner_self_eq_norm_sq`), dann kuerzen; das darf man, denn ‖g‖ ≠ 0.
    intro hg
    have hn : ‖g‖ ≠ 0 := norm_ne_zero_iff.2 hg
    rw [inner_smul_right, real_inner_self_eq_norm_sq]
    field_simp

-- Folie: "Das Verfahren" (der Gradient in Mathlib)
/- Der Gradient ist der Riesz-Repraesentant des Differentials in einem
   Hilbertraum (`[CompleteSpace F]`): Das Differential ist primaer. Ueber ℝ
   faellt er in einer Variablen mit der Ableitung zusammen: -/
example (f : ℝ → ℝ) (x : ℝ) : gradient f x = deriv f x := gradient_eq_deriv'

-- Folie: "Das Verfahren"
/-- Ein Schritt des Gradientenabstiegs in einer Variablen. -/
noncomputable def gdSchritt (f : ℝ → ℝ) (t x : ℝ) : ℝ := x - t * deriv f x

/-- Die Iteration `x₀, x₁, x₂, …` mit fester Schrittweite `t`. -/
noncomputable def gdIter (f : ℝ → ℝ) (t x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => gdSchritt f t (gdIter f t x₀ n)

-- Folie: "Ein kleines Beispiel: f(x) = x^2" und "Ein Schritt, exakt ausgerechnet"
/-- Ein Schritt fuer f(x) = x²: Wegen f'(x) = 2x ist x - t·f'(x) = (1 - 2t)·x. -/
theorem gd_quadrat_schritt (t x : ℝ) :
    gdSchritt (fun x : ℝ => x ^ 2) t x = (1 - 2 * t) * x := by
  -- Schritt 1: f'(x) = 2x, die Potenzregel (`hasDerivAt_pow`), als Zahl `deriv`.
  have hd : deriv (fun x : ℝ => x ^ 2) x = 2 * x := by
    rw [(hasDerivAt_pow 2 x).deriv]          -- f'(x) = 2·x^(2-1)
    norm_num
  -- Schritt 2: den Schritt ausschreiben, f'(x) einsetzen, ausmultiplizieren.
  rw [gdSchritt, hd]
  ring

-- Folie: "Fehlerformel und Konvergenz"
/-- Fehlerformel: Nach n Schritten ist x_n = (1 - 2t)^n·x₀. -/
theorem gd_quadrat_fehler (t x₀ : ℝ) (n : ℕ) :
    gdIter (fun x : ℝ => x ^ 2) t x₀ n = (1 - 2 * t) ^ n * x₀ := by
  induction n with
  | zero =>
      -- Induktionsanfang n = 0: x₀ = (1 - 2t)^0·x₀; `simp` entfaltet gdIter und rechnet.
      simp [gdIter]
  | succ k ih =>
      -- Induktionsschritt: eine Iteration entfalten, dann erst ein Schritt
      -- (`gd_quadrat_schritt`), dann die Induktionsannahme ih.
      simp only [gdIter]
      rw [gd_quadrat_schritt, ih]
      ring                                   -- (1 - 2t)·(1 - 2t)^k = (1 - 2t)^(k+1)

/-- Konvergenz: Ist |1 - 2t| < 1, so geht x_n gegen 0, das Minimum. -/
theorem gd_quadrat_tendsto (t x₀ : ℝ) (h : |1 - 2 * t| < 1) :
    Tendsto (fun n => gdIter (fun x : ℝ => x ^ 2) t x₀ n) atTop (𝓝 0) := by
  -- Schritt 1: r^n → 0 fuer |r| < 1 steht in Mathlib
  -- (`tendsto_pow_atTop_nhds_zero_of_norm_lt_one`; in ℝ ist ‖r‖ = |r|).
  have hpow : Tendsto (fun n : ℕ => (1 - 2 * t) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_norm_lt_one (by rwa [Real.norm_eq_abs])
  -- Schritt 2: mal x₀ geht es gegen 0·x₀ = 0 (`mul_const`), kein ε und kein N; die
  -- Fehlerformel ersetzt x_n.
  simpa [gd_quadrat_fehler] using hpow.mul_const x₀

-- Folie: "Die Schrittweitenbedingung"
/-- |1 - 2t| < 1 genau dann, wenn 0 < t < 1. -/
theorem schrittweite_quadrat (t : ℝ) : |1 - 2 * t| < 1 ↔ (0 < t ∧ t < 1) := by
  -- Den Betrag aufloesen, -1 < 1 - 2t < 1 (`abs_lt`), also 0 < 2t < 2: linear.
  rw [abs_lt]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by linarith, by linarith⟩

-- Allgemeiner fuer q(x) = a(x - m)² + c, dazu zwei Variablen und "konvex und kritisch":
-- `gd_q_fehler`, `gd_q_tendsto`, `schrittweite`, `konvex_kritisch` in code/Kapitel4.lean
-- (Skript).

/-! ## 4.6 Optima unter Nebenbedingungen: Lagrange-Multiplikatoren -/

-- Folie: "Extremum unter einer Nebenbedingung"
/-- Mathlibs `IsLocalMaxOn φ M x₀` mit M = {x | f x = c} heisst: Schliesslich laengs
𝓝[M] x₀, also fuer die Punkte von M nahe x₀, ist φ x ≤ φ x₀. Klassisch ausgeschrieben
steht rechts die Definition der Folie. (`IsLocalMinOn` ebenso mit ≥; `IsLocalExtrOn`
heisst "Maximum oder Minimum".) -/
theorem max_unter_nb_iff {E : Type*} [MetricSpace E] (φ f : E → ℝ) (c : ℝ) (x₀ : E) :
    IsLocalMaxOn φ {x | f x = c} x₀ ↔
      ∃ ε > 0, ∀ x, dist x x₀ < ε → f x = c → φ x ≤ φ x₀ := by
  -- Schritt 1: die Definition, schliesslich laengs 𝓝[M] x₀ (`IsLocalMaxOn`, `IsMaxFilter`).
  -- Schritt 2: Laengs 𝓝[M] x₀ heisst laengs 𝓝 x₀ fuer die Punkte von M
  -- (`eventually_nhdsWithin_iff`), und das heisst: auf einer Kugel um x₀
  -- (`Metric.eventually_nhds_iff`). Dann stehen links und rechts dieselben Worte.
  rw [IsLocalMaxOn, IsMaxFilter, eventually_nhdsWithin_iff, Metric.eventually_nhds_iff]
  rfl

-- Folie: "Ein Beispiel: xy unter x + y = 1" (das Maximum bestaetigen)
/-- Beispiel: maximiere `x·y` unter `x + y = 1`.
Lagrange liefert den Kandidaten `x = y = 1/2`; hier die elementare
Bestaetigung, dass dieser Kandidat wirklich das Maximum ist. -/
theorem xy_le_quarter (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  -- Der Beweis von der Folie: (x + y)² - 4xy = (x - y)² ≥ 0, und x + y = 1 gibt
  -- 1 - 4xy ≥ 0, also xy ≤ 1/4. `nlinarith` bekommt den Hinweis (x - y)² ≥ 0 und
  -- multipliziert h mit sich selbst, um (x + y)² = 1 zu bekommen; der Rest ist linear.
  nlinarith [sq_nonneg (x - y)]

-- Folie: "Klassisch aus projektiv, in Lean" (die Fallunterscheidung von
-- Folie "Lagrange in der Mathlib-Fassung")
/-- Projektiv → klassisch, sobald Df ≠ 0: Waere Λ₀ = 0, so waere Λ ≠ 0 und
Λ • Df = 0, also Df = 0. Sonst darf man durch Λ₀ teilen: μ = -Λ/Λ₀ ist das
λ der klassischen Fassung (`λ` ist in Lean fuer `fun` reserviert). -/
theorem klassisch_aus_projektiv {E : Type*} [AddCommGroup E] [Module ℝ E]
    {Df Dφ : E} {Λ Λ₀ : ℝ} (hne : (Λ, Λ₀) ≠ 0) (h : Λ • Df + Λ₀ • Dφ = 0)
    (hDf : Df ≠ 0) : ∃ μ : ℝ, Dφ = μ • Df := by
  -- Schritt 1 ("Fall Λ₀ = 0" auf der Folie): Λ₀ = 0 ist unmoeglich. `rintro rfl` nimmt es an
  -- und setzt Λ₀ := 0 ueberall ein: hne : (Λ, 0) ≠ 0 und h : Λ • Df + 0 • Dφ = 0.
  have hΛ₀ : Λ₀ ≠ 0 := by
    rintro rfl
    -- Dann ist Λ ≠ 0: Waere Λ = 0, so waere (Λ, 0) = (0, 0), gegen hne.
    have hΛ : Λ ≠ 0 := fun h0 => hne (by simp [h0])
    -- 0 • Dφ = 0 (`zero_smul`), + 0 weglassen (`add_zero`), und ein Produkt Λ • Df = 0 hat
    -- einen verschwindenden Faktor (`smul_eq_zero`):  h : Λ = 0 ∨ Df = 0.
    rw [zero_smul, add_zero, smul_eq_zero] at h
    -- Beide Faelle widersprechen den Annahmen: Λ = 0 gegen hΛ, Df = 0 gegen hDf.
    exact h.elim hΛ hDf
  -- Schritt 2 ("Fall Λ₀ ≠ 0, man darf teilen"): Aus Λ • Df + Λ₀ • Dφ = 0 folgt
  -- Λ₀ • Dφ = -(Λ • Df) (`eq_neg_of_add_eq_zero_right`).
  have h2 : Λ₀ • Dφ = -(Λ • Df) := eq_neg_of_add_eq_zero_right h
  -- Schritt 3: Das gesuchte μ ist -Λ/Λ₀. Zu zeigen bleibt  Dφ = (-Λ / Λ₀) • Df.
  use -Λ / Λ₀
  -- Schritt 4: "Durch Λ₀ teilen" heisst, mit Λ₀⁻¹ zu multiplizieren. Erste Zeile: Λ₀⁻¹·Λ₀ = 1
  -- (`inv_mul_cancel₀`) und 1 • Dφ = Dφ. Zweite Zeile: h2 einsetzen. Dritte Zeile: die
  -- Skalare zusammenfassen, Λ₀⁻¹·(-Λ) = -Λ/Λ₀ (`smul_neg`, `smul_smul`, `neg_smul`, `ring`).
  calc Dφ = Λ₀⁻¹ • (Λ₀ • Dφ) := by rw [smul_smul, inv_mul_cancel₀ hΛ₀, one_smul]
    _ = Λ₀⁻¹ • -(Λ • Df) := by rw [h2]
    _ = (-Λ / Λ₀) • Df := by rw [smul_neg, smul_smul, ← neg_smul]; congr 1; ring

-- Folie: "Klassisch aus projektiv, in Lean"
/-- Lagrange, klassisch, mit allen Voraussetzungen: f und φ streng
differenzierbar und Df ≠ 0. Beweis: Mathlibs projektiver Satz, dann der
Zusammenhang von oben. -/
theorem lagrange_klassisch {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] {f φ : E → ℝ} {x₀ : E}
    {Df Dφ : E →L[ℝ] ℝ} (hext : IsLocalExtrOn φ {x | f x = f x₀} x₀)
    (hf : HasStrictFDerivAt f Df x₀) (hφ : HasStrictFDerivAt φ Dφ x₀)
    (hDf : Df ≠ 0) : ∃ μ : ℝ, Dφ = μ • Df := by
  -- Schritt 1 (Folie "Klassisch aus projektiv", 1.): Mathlibs projektiver Satz liefert
  -- Multiplikatoren Λ, Λ₀ mit  hne : (Λ, Λ₀) ≠ 0  und  h : Λ • Df + Λ₀ • Dφ = 0.
  obtain ⟨Λ, Λ₀, hne, h⟩ :=
    hext.exists_multipliers_of_hasStrictFDerivAt_1d hf hφ
  -- Schritt 2 (2.): Unser Lemma macht die Fallunterscheidung und liefert das μ.
  exact klassisch_aus_projektiv hne h hDf

-- Folie: "Klassisch aus projektiv, in Lean" (der Satz angewandt auf das Beispiel xy unter
-- x + y = 1: der Kandidat)
/- Das Beispiel mit dem Satz: Hat φ(x, y) = xy unter f(x, y) = x + y = 1 in p ein
   Extremum, so liefert Lagrange ∇φ(p) = μ ∇f(p), also (y, x) = μ (1, 1): x = y, und mit
   x + y = 1 ist p = (1/2, 1/2). -/
example (p : ℝ × ℝ) (hp : p.1 + p.2 = 1)
    (hext : IsLocalExtrOn (fun q : ℝ × ℝ => q.1 * q.2) {q | q.1 + q.2 = 1} p) :
    p = (1 / 2, 1 / 2) := by
  -- Schritt 1: die Ableitungen, mit den Regeln. f = x + y hat Df = dx + dy, φ = xy hat
  -- Dφ = x dy + y dx (Produktregel).
  have hf := (hasStrictFDerivAt_fst (𝕜 := ℝ) (p := p)).add hasStrictFDerivAt_snd
  have hφ := (hasStrictFDerivAt_fst (𝕜 := ℝ) (p := p)).mul hasStrictFDerivAt_snd
  -- Df ≠ 0, denn Df(1, 0) = 1.
  have hDf : (ContinuousLinearMap.fst ℝ ℝ ℝ + ContinuousLinearMap.snd ℝ ℝ ℝ) ≠ 0 := by
    intro h0
    have := congrArg (fun L => L (1, 0)) h0
    simp at this
  -- Schritt 2: der Satz. Die Nebenbedingung {q | f q = f p} ist {q | q.1 + q.2 = 1}.
  obtain ⟨μ, hμ⟩ := lagrange_klassisch (by simpa [hp] using hext) hf hφ hDf
  -- Schritt 3: Dφ = μ Df an (1, 0) und an (0, 1) auswerten: y = μ und x = μ.
  have h1 := congrArg (fun L => L (1, 0)) hμ
  have h2 := congrArg (fun L => L (0, 1)) hμ
  simp at h1 h2
  -- Schritt 4: x = y und x + y = 1, also x = y = 1/2.
  ext <;> simp <;> linarith

-- Mathlib hat Lagrange auch fuer mehrere Nebenbedingungen:
-- `IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt`.

/-! ### Mathlib-Wegweiser zu Vorlesung 6

Nur zum Nachschlagen, ohne Folie. -/
#check @taylor_mean_remainder_lagrange_iteratedDeriv   -- Taylor, eine Variable, Lagrange
#check @taylor_isLittleO                                -- Taylor, eine Variable, o-Fassung
#check @exists_hasDerivAt_eq_slope                      -- Mittelwertsatz
#check @IsLocalMin.hasFDerivAt_eq_zero                  -- Fermat, mehrere Veraenderliche
#check @Matrix.PosDef                                   -- positiv definit
#check @Matrix.IsHermitian.posDef_iff_eigenvalues_pos   -- ... genau wenn Eigenwerte > 0
#check @tendsto_pow_atTop_nhds_zero_of_norm_lt_one      -- r^n → 0 fuer |r| < 1
#check @IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt_1d  -- Lagrange, projektiv

end Skript.Analysis
