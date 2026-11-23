/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 6
  (Folien: folien/vorlesung06.tex)
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

/-! ## Wiederholung aus Vorlesung 5 -/

-- Wiederholung aus Vorlesung 5 (code/Vorlesung05.lean, Abschnitt 4.4): Klein-o fuer das
-- Hesse-Kriterium, Fermat in einer Variablen und die Richtungsableitung fuer Fermat in
-- mehreren Veraenderlichen.
-- Folie: "Rueckblick: Vorlesung 5"
/-- Klein-o laengs eines Filters: Zu jedem c > 0 gilt schliesslich
|f x| ≤ c·|g x|. Erst kommt c, dann die grosse Menge -- zu jedem c eine
eigene. `∀ᶠ x in l, P x` heisst: Die Menge, auf der P gilt, ist gross. -/
def KleinO {α : Type*} (f g : α → ℝ) (l : Filter α) : Prop :=
  ∀ c > 0, ∀ᶠ x in l, |f x| ≤ c * |g x|

/-- Fermat: An einem lokalen Minimum ist f' = 0. Nahe a ist f a ≤ f x, also
die Sekantensteigung `slope f a x` = (f x - f a)/(x - a) links von a ≤ 0,
rechts ≥ 0. Grenzwerte erhalten ≤, also f' ≤ 0 ≤ f'. -/
theorem fermat {f : ℝ → ℝ} {a f' : ℝ} (h : IsLocalMin f a)
    (hf : HasDerivAt f f' a) : f' = 0 := by
  -- Schritt 1 (= 1. auf der Folie, Steigungen): Die Sekantensteigung
  -- slope f a x = (f x - f a)/(x - a) strebt gegen f', wenn x → a mit x ≠ a. Das ist die
  -- Ableitung als Grenzwert (`hasDerivAt_iff_tendsto_slope`, ein Satz aus Mathlib).
  -- 𝓝[≠] a ist 𝓝 a ohne den Punkt a selbst: Bei x = a waere der Quotient 0/0.
  have hs : Tendsto (slope f a) (𝓝[≠] a) (𝓝 f') :=
    hasDerivAt_iff_tendsto_slope.1 hf
  -- Schritt 2 (= 2., Vorzeichen): h : IsLocalMin f a heisst "schliesslich laengs 𝓝 a ist
  -- f a ≤ f x". Daraus das Vorzeichen der Steigung, links und rechts von a getrennt.
  --
  -- Links von a (Filter 𝓝[<] a, nur x < a): `filter_upwards` nimmt zwei grosse Mengen,
  -- {x | f a ≤ f x} -- das h, auf die linke Seite eingeschraenkt (`nhdsWithin_le_nhds`) --
  -- und {x | x < a} -- die Seite selbst ist gross in ihrem eigenen Filter
  -- (`self_mem_nhdsWithin`). Fuer ein x darin gilt hfx : f a ≤ f x und hxa : x < a.
  have links : ∀ᶠ x in 𝓝[<] a, slope f a x ≤ 0 := by
    filter_upwards [nhdsWithin_le_nhds h, self_mem_nhdsWithin] with x hfx hxa
    -- Die Steigung von a nach x ist die von x nach a (`slope_comm`) ...
    rw [slope_comm]
    -- ... und die ist ≤ 0 genau dann, wenn f a ≤ f x: Der Zaehler f a - f x ist ≤ 0,
    -- der Nenner a - x ist > 0 (`slope_nonpos_iff_of_le`, mit x ≤ a).
    exact (slope_nonpos_iff_of_le hxa.le).2 hfx
  -- Rechts von a (Filter 𝓝[>] a, nur x > a): genauso, nur ohne `slope_comm`. Jetzt ist
  -- der Zaehler f x - f a ≥ 0 und der Nenner x - a > 0 (`slope_nonneg_iff_of_le`).
  have rechts : ∀ᶠ x in 𝓝[>] a, 0 ≤ slope f a x := by
    filter_upwards [nhdsWithin_le_nhds h, self_mem_nhdsWithin] with x hfx hax
    exact (slope_nonneg_iff_of_le hax.le).2 hfx
  -- Schritt 3 (= 3., Grenzuebergang): Grenzwerte erhalten ≤ (`le_of_tendsto`, `ge_of_tendsto`).
  -- Dazu schraenkt man den Filter in Schritt 1 auf eine Seite ein (`mono_left`: eine
  -- kleinere Filtermenge, der Grenzwert bleibt): von links ist f' ≤ 0, von rechts 0 ≤ f'.
  have h1 : f' ≤ 0 := le_of_tendsto (hs.mono_left (nhdsLT_le_nhdsNE a)) links
  have h2 : 0 ≤ f' := ge_of_tendsto (hs.mono_left (nhdsGT_le_nhdsNE a)) rechts
  exact le_antisymm h1 h2                              -- f' ≤ 0 und 0 ≤ f', also f' = 0

/-- Satz, Teil (a): Ist f in x differenzierbar mit der Ableitung f', so hat die
Schnittkurve g(t) = f(x + t·v) in 0 die Ableitung f' v. Der Beweis mit o, wie auf
Papier; die Kommentare im Beweis nennen den Papierschritt, dem sie entsprechen:
Gegeben:   f(x + h) - f(x) - f' h = o(h)  laengs 𝓝 0  (der Rest r(h) ist o(h)).
Gesucht:   f(x + t·v) - f(x) - t·f' v = o(t)  laengs 𝓝 0  (die Ableitung von g in 0).
Schritt 1: Der Fehler von g ist der Rest bei h = t·v, denn f'(t·v) = t·f' v
           (f' ist linear).
Schritt 2: h = t·v einsetzen. Fuer t → 0 geht t·v → 0, also ist r(t·v) = o(t·v).
Schritt 3: Wegen ‖t·v‖ = |t|·‖v‖ ist t·v = O(t), also ist o(t·v) ein o(t). Den
           Faktor c/‖v‖ des Papierbeweises waehlt Mathlibs o-Rechnung dabei selbst.
Das Teilen durch t entfaellt: Die Ableitung ist in Mathlib selbst durch
"Fehler = o" definiert. -/
theorem richtungsableitung {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F} {f' : E →L[ℝ] F} {x : E}
    (hf : HasFDerivAt f f' x) (v : E) :
    HasDerivAt (fun t : ℝ => f (x + t • v)) (f' v) 0 := by
  -- Schritt 1 (Papier: Gegeben): Die Voraussetzung auspacken. Mathlib sagt "f hat in x
  -- die Ableitung f'" als o-Aussage laengs 𝓝 0, mit h als Schritt von x nach x + h
  -- (`hasFDerivAt_iff_isLittleO_nhds_zero`). Danach ist
  --   hf : (fun h => f (x + h) - f x - f' h) =o[𝓝 0] fun h => h      (Fehler = o(h))
  rw [hasFDerivAt_iff_isLittleO_nhds_zero] at hf
  -- Schritt 2 (Papier: Gesucht): Das Ziel genauso auspacken, nun fuer die Schnittkurve
  -- in einer Variablen (`hasDerivAt_iff_isLittleO_nhds_zero`). Das Ziel wird
  --   (fun h => f (x + (0 + h) • v) - f (x + 0 • v) - h • f' v) =o[𝓝 0] fun h => h
  -- also f(x + t·v) - f(x) - t·f' v = o(t), mit t = h; die 0 + h ist nur h.
  rw [hasDerivAt_iff_isLittleO_nhds_zero]
  -- Schritt 3 (Papier: Schritt 2): t·v → 0, wenn t → 0 -- die Punkte t·v landen in jeder
  -- Kugel um 0. Die Identitaet strebt gegen 0 (`tendsto_id`); mal den festen Vektor v
  -- (`smul_const`) strebt t·v gegen 0 • v, und das ist 0.
  have hv : Tendsto (fun t : ℝ => t • v) (𝓝 0) (𝓝 0) := by
    simpa using (tendsto_id (x := 𝓝 (0 : ℝ))).smul_const v
  -- Schritt 4 (Papier: Schritt 3): t·v = O(t), mit dem Faktor ‖v‖. `isBigO_iff` verlangt
  -- ein c mit ‖t·v‖ ≤ c·‖t‖ fuer alle t; wir nehmen c = ‖v‖. Fuer jedes t bleibt
  --   ‖t • v‖ ≤ ‖v‖ * ‖t‖
  -- zu zeigen.
  -- `norm_smul` macht links ‖t‖ * ‖v‖ daraus, `mul_comm` vertauscht die Faktoren:
  -- beide Seiten sind gleich.
  have hO : (fun t : ℝ => t • v) =O[𝓝 0] (fun t : ℝ => t) :=
    Asymptotics.isBigO_iff.2 ⟨‖v‖, Eventually.of_forall fun t => by
      rw [norm_smul, mul_comm]⟩
  -- Schritt 5 (Papier: Schritte 2 und 3): h = t·v einsetzen. `comp_tendsto hv` setzt die
  -- Funktion t ↦ t·v in "Fehler = o(h)" ein: Der Fehler an der Stelle t·v ist o(t·v).
  -- Weil t·v = O(t) ist (hO), wird daraus o(t) (`trans_isBigO`):
  --   h : ((fun h => f (x + h) - f x - f' h) ∘ fun t => t • v) =o[𝓝 0] fun t => t
  have h := (hf.comp_tendsto hv).trans_isBigO hO
  -- Schritt 6 (Papier: Schritt 1): Angleichen. `Function.comp_def` schreibt das ∘ aus,
  -- `map_smul` liefert f' (t • v) = t • f' v, denn f' ist linear. Dann steht in h genau
  -- das Ziel aus Schritt 2.
  simpa [Function.comp_def, map_smul] using h

/-! ## 4.4 Taylor in mehreren Veraenderlichen; Hoch-, Tief- und Sattelpunkte -/

-- Folie: "Beweis (2): die Details -- die Kettenregel, Schritt fuer Schritt" (Schritte 1
-- und 2); die Idee dazu auf "Beweis (2): die Idee -- jede Ableitung bringt ein h"
/-- Die Kettenregel fuer die Schnittkurve: g = f ∘ γ mit der Geraden γ(s) = p + s·h hat
in jedem t die Ableitung Df(p + t·h) h. Das ist die Kettenregel aus Vorlesung 5,
D(f ∘ γ)(t) = Df(γ(t)) Dγ(t), mit Dγ(t) = h. Bei t = 0 ist das `richtungsableitung`.
Wiederholt angewandt gibt das g^(j)(t) = D^j f(p + t·h) h^j, den Kern des Beweises der
Taylorformel. -/
theorem schnittkurve_ableitung {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : E → ℝ} {f' : E →L[ℝ] ℝ} {p h : E} {t : ℝ} (hf : HasFDerivAt f f' (p + t • h)) :
    HasDerivAt (fun s : ℝ => f (p + s • h)) (f' h) t := by
  -- Schritt 1 (= 1. auf der Folie, die Gerade ableiten): γ(s) = p + s·h hat die Ableitung
  -- Dγ(t) = (s ↦ s·h), die Spalte h (`smulRight h`). Die Identitaet hat die Ableitung id
  -- (`hasFDerivAt_id`), mal den festen Vektor h (`smul_const`), plus p (`const_add`).
  have hγ : HasFDerivAt (fun s : ℝ => p + s • h)
      ((ContinuousLinearMap.id ℝ ℝ).smulRight h) t :=
    ((hasFDerivAt_id t).smul_const h).const_add p
  -- Schritt 2 (= 2., die Kettenregel aus Vorlesung 5, `HasFDerivAt.comp`, wie in
  -- `kettenregel_polar`): D(f ∘ γ)(t) = Df(γ(t)) ∘ Dγ(t), also
  --   hk : HasFDerivAt (f ∘ fun s => p + s • h)
  --          (f'.comp ((ContinuousLinearMap.id ℝ ℝ).smulRight h)) t
  have hk := hf.comp t hγ
  -- Schritt 3: Eine Ableitung von ℝ nach ℝ ist eine Zahl, der Wert der linearen Abbildung
  -- bei 1 (`HasFDerivAt.hasDerivAt`; so wie `deriv f x = fderiv ℝ f x 1` in Vorlesung 5).
  -- Hier ist das (f' ∘ Dγ(t)) 1 = f' (1·h) = f' h; `simpa` rechnet es aus und schreibt das
  -- ∘ aus (`Function.comp_def`).
  simpa [Function.comp_def] using hk.hasDerivAt

-- Folie: "Beweis (1): zurueck auf eine Variable", "Der Satz von Schwarz und Multiindizes"
-- und "Hoehere Ableitungen in Richtung h" (D^j f(x) h^j ist `iteratedFDeriv ℝ j f x`,
-- angewandt auf das konstante Tupel (h, ..., h))
/- Taylor in einer Variablen mit dem Restglied von Lagrange steht in Mathlib. Mit der
   Schnittkurve folgt daraus die Taylorformel in mehreren Veraenderlichen. Mathlib hat
   auch die Symmetrie der zweiten Ableitung (Schwarz) und die iterierten Ableitungen,
   aber keine Taylorformel mit Restglied in mehreren Veraenderlichen. -/
#check @taylor_mean_remainder_lagrange     -- eine Variable, Restglied nach Lagrange
#check @second_derivative_symmetric        -- Schwarz: die Hesse-Matrix ist symmetrisch
#check @iteratedFDeriv                     -- die j-te Ableitung als j-lineare Abbildung

-- Folie: "Vorgerechnet: Taylor in zwei Variablen"
/-- Das Beispiel f(x, y) = x³ + y³ - 3xy. Kritisch sind (0, 0) und (1, 1). -/
def kubisch (p : ℝ × ℝ) : ℝ := p.1 ^ 3 + p.2 ^ 3 - 3 * p.1 * p.2

/-- Taylor in (1, 1): Ein Polynom vom Grad 3 ist gleich seinem Taylorpolynom dritter
Ordnung. Es ist f(1, 1) = -1, der Gradient ist 0, die Hesse-Matrix [[6, -3], [-3, 6]]
gibt den quadratischen Teil, h₁³ + h₂³ ist das Glied dritter Ordnung. -/
theorem kubisch_taylor (h : ℝ × ℝ) :
    kubisch ((1, 1) + h) =
      -1 + (6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2) / 2 + (h.1 ^ 3 + h.2 ^ 3) := by
  -- Die Koordinaten von (1, 1) + h sind 1 + h₁ und 1 + h₂; einsetzen, ausmultiplizieren.
  simp only [kubisch, Prod.fst_add, Prod.snd_add]
  ring

-- Folie: "Tief-, Hoch- und Sattelpunkte"
/-- Lokales Minimum, selbst definiert: Schliesslich laengs 𝓝 p ist f p ≤ f x -- auf
einer Kugel um p liegt f nirgends unter f p. Das lokale Maximum ist dasselbe mit
f x ≤ f p. -/
def LokalesMinimum {E : Type*} [TopologicalSpace E] (f : E → ℝ) (p : E) : Prop :=
  ∀ᶠ x in 𝓝 p, f p ≤ f x

/- Mathlibs `IsLocalMin` ist wortgleich definiert, als Aussage "schliesslich laengs
   𝓝 p"; ebenso `IsLocalMax` mit f x ≤ f p. -/
example {E : Type*} [TopologicalSpace E] (f : E → ℝ) (p : E) :
    LokalesMinimum f p ↔ IsLocalMin f p := Iff.rfl

/-- Ein Sattelpunkt ist ein kritischer Punkt (die Ableitung ist die Nullabbildung), in
dem weder ein lokales Minimum noch ein lokales Maximum liegt. -/
def Sattelpunkt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (f : E → ℝ)
    (p : E) : Prop :=
  HasFDerivAt f (0 : E →L[ℝ] ℝ) p ∧ ¬ IsLocalMin f p ∧ ¬ IsLocalMax f p

-- Folie: "Tief-, Hoch- und Sattelpunkte"
/- Die drei Grundformen in der Ebene. Punkte sind Paare p = (x, y) mit p.1 = x und
   p.2 = y. Auf Paaren rechnet Mathlib mit der Maximumsnorm: Die Kugel vom Radius δ
   um 0 ist das Quadrat |x| < δ, |y| < δ. -/

/-- Tiefpunkt: x² + y² hat in 0 ein Minimum, sogar ein globales: f ≥ 0 = f(0). -/
theorem tiefpunkt_modell : IsLocalMin (fun p : ℝ × ℝ => p.1 ^ 2 + p.2 ^ 2) 0 :=
  -- Die Ungleichung gilt ueberall (`Eventually.of_forall`), erst recht auf einer Kugel.
  Eventually.of_forall fun p => by simp; positivity

/-- Hochpunkt: -x² - y² hat in 0 ein Maximum. -/
theorem hochpunkt_modell : IsLocalMax (fun p : ℝ × ℝ => -p.1 ^ 2 - p.2 ^ 2) 0 :=
  Eventually.of_forall fun p => by simp; nlinarith [sq_nonneg p.1, sq_nonneg p.2]

/-- Sattel: x² - y² hat in 0 die Ableitung 0, aber kein Extremum. Laengs der x-Achse
liegt in 0 ein Minimum, laengs der y-Achse ein Maximum. -/
theorem sattel_modell : Sattelpunkt (fun p : ℝ × ℝ => p.1 ^ 2 - p.2 ^ 2) 0 := by
  -- Drei Teile: kritisch, kein Minimum, kein Maximum.
  refine ⟨?_, ?_, ?_⟩
  · -- Kritisch: x² und y² haben nach den Regeln (`.pow`, `.sub`, die Koordinaten
    -- `hasFDerivAt_fst` und `hasFDerivAt_snd`) die Ableitungen 2x·dx und 2y·dy; in 0
    -- ist das die Nullabbildung.
    have h := ((hasFDerivAt_fst (𝕜 := ℝ) (p := (0 : ℝ × ℝ))).pow 2).sub
      ((hasFDerivAt_snd (𝕜 := ℝ) (p := (0 : ℝ × ℝ))).pow 2)
    convert h using 1
    ext <;> simp
  · -- Kein Minimum: Auf einer Kugel vom Radius δ waere 0 = f(0) ≤ f(q). Der Punkt
    -- (0, δ/2) liegt darin, aber dort ist f = -(δ/2)² < 0.
    intro h
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    have h3 := hball (y := (0, δ / 2)) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      constructor
      · simpa using hδ
      · rw [Prod.snd_zero, sub_zero, abs_of_pos (half_pos hδ)]
        exact half_lt_self hδ)
    simp at h3                                -- h3: 0 ≤ -(δ/2)², umgeformt
    nlinarith [mul_pos (half_pos hδ) (half_pos hδ)]
  · -- Kein Maximum: genauso mit dem Punkt (δ/2, 0), dort ist f = (δ/2)² > 0.
    intro h
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    have h3 := hball (y := (δ / 2, 0)) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      constructor
      · rw [Prod.fst_zero, sub_zero, abs_of_pos (half_pos hδ)]
        exact half_lt_self hδ
      · simpa using hδ)
    simp at h3                                -- h3: (δ/2)² ≤ 0, umgeformt
    nlinarith [mul_pos (half_pos hδ) (half_pos hδ)]

-- Folie: "Optimieren: Am lokalen Extremum ist Df(x) = 0"
/- Fermat in mehreren Veraenderlichen, wie auf der Folie: Die Schnittkurve
   t ↦ f(x + t·v) hat bei 0 ein lokales Minimum und die Ableitung Df(x) v
   (`richtungsableitung`). Nach Fermat in einer Variablen ist sie 0 -- fuer jedes v,
   also ist Df(x) = 0. -/
theorem fermat_mehrdim {n : ℕ} {f : (Fin n → ℝ) → ℝ} {x : Fin n → ℝ}
    {f' : (Fin n → ℝ) →L[ℝ] ℝ} (h : IsLocalMin f x) (hf : HasFDerivAt f f' x) :
    f' = 0 := by
  -- Schritt 1 ("Sei v ∈ ℝⁿ" auf der Folie): Zwei lineare Abbildungen sind gleich, wenn sie an
  -- jedem Vektor v gleich sind (`ext v`). Ziel:  f' v = 0 v  (rechts steht die Nullabbildung).
  ext v
  -- Schritt 2 ("g'(0) = Df(x) v (Teil (a))"): Die Schnittkurve g(t) = f(x + t·v) hat in 0 die
  -- Ableitung f' v -- das ist `richtungsableitung`.
  have hg := richtungsableitung hf v
  -- Schritt 3: Die Punkte x + t·v streben gegen x, wenn t → 0 (t strebt gegen 0, mal den festen
  -- Vektor v, plus den festen Punkt x).
  have hv : Tendsto (fun t : ℝ => x + t • v) (𝓝 0) (𝓝 x) := by
    simpa using (tendsto_id (x := 𝓝 (0 : ℝ))).smul_const v |>.const_add x
  -- Schritt 4 ("g hat bei t = 0 ein lokales Minimum"): h sagt "schliesslich laengs 𝓝 x ist
  -- f x ≤ f y". `hv.eventually h` zieht das zurueck auf t: "schliesslich laengs 𝓝 0 ist
  -- f x ≤ f (x + t·v)" -- und das ist ein lokales Minimum von g bei 0.
  have hmin : IsLocalMin (fun t : ℝ => f (x + t • v)) 0 := by
    simpa [IsLocalMin, IsMinFilter] using hv.eventually h
  -- Schritt 5 ("Fermat in einer Variablen: g'(0) = 0"): `fermat hmin hg` gibt f' v = 0;
  -- `simpa` bringt noch (0 : E →L ℝ) v = 0 in Ordnung.
  simpa using fermat hmin hg

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

/- Mathlib hat positiv definite Matrizen: `Matrix.PosDef M` heisst, M ist symmetrisch
   (hermitesch) und x·(M x) > 0 fuer x ≠ 0. Fuer die Hesse-Matrix des Beispiels
   x³ + y³ - 3xy im Punkt (1, 1) folgt das aus der Schranke: -/
example : Matrix.PosDef !![(6 : ℝ), -3; -3, 6] := by
  -- Mathlibs Definition, ausgeschrieben: symmetrisch, und x·(M x) > 0 fuer x ≠ 0.
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_, fun x hx => ?_⟩
  · -- symmetrisch: Eintrag fuer Eintrag
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · -- x ≠ 0 heisst: eine der beiden Koordinaten ist ≠ 0, also x₀² + x₁² > 0.
    have hx' : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra hc
      push_neg at hc
      apply hx
      ext i
      fin_cases i <;> simp [hc.1, hc.2]
    have hpos : 0 < x 0 ^ 2 + x 1 ^ 2 := by
      rcases hx' with h0 | h1
      · have := pow_pos (abs_pos.2 h0) 2
        rw [sq_abs] at this
        nlinarith [sq_nonneg (x 1)]
      · have := pow_pos (abs_pos.2 h1) 2
        rw [sq_abs] at this
        nlinarith [sq_nonneg (x 0)]
    -- Die Schranke mit a = 6, b = -3, c = 6: Q ≥ (27/12)·(x₀² + x₁²) > 0.
    have hq := quadratische_form_schranke 6 (-3) 6 (by norm_num) (by norm_num) (x 0) (x 1)
    simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two]
    norm_num at hq
    nlinarith

-- Folie: "Vorgerechnet: f(x, y) = x^3 + y^3 - 3xy -- was die Hesse-Matrix sagt"
/-- In der Maximumsnorm ist ‖h‖² = max(h₁², h₂²) ≤ h₁² + h₂². -/
theorem norm_sq_le_quadratsumme (h : ℝ × ℝ) : ‖h‖ ^ 2 ≤ h.1 ^ 2 + h.2 ^ 2 := by
  rw [Prod.norm_def, Real.norm_eq_abs, Real.norm_eq_abs]
  -- Das Maximum ist |h₁| oder |h₂|; beide Male ist sein Quadrat hoechstens die Summe.
  rcases max_cases |h.1| |h.2| with ⟨hm, _⟩ | ⟨hm, _⟩
  · rw [hm, sq_abs]; nlinarith [sq_nonneg h.2]
  · rw [hm, sq_abs]; nlinarith [sq_nonneg h.1]

/-- (1, 1) ist ein Tiefpunkt. Die Hesse-Matrix ist [[6, -3], [-3, 6]] mit Determinante
27 und Spur 12, also Q(h) ≥ (27/12)(h₁² + h₂²) = (9/4)(h₁² + h₂²). Der Rest der
Taylorentwicklung ist h₁³ + h₂³, und der ist o(‖h‖²). -/
theorem kubisch_tiefpunkt : IsLocalMin kubisch (1, 1) := by
  -- Das Hesse-Kriterium (`min_aus_taylor`) mit Q(h) = 6h₁² - 6h₁h₂ + 6h₂² und m = 9/4.
  apply min_aus_taylor (fun h => 6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2)
    (by norm_num : (0 : ℝ) < 9 / 4)
  · -- Die Schranke: Kriterium in zwei Dimensionen, dazu ‖h‖² ≤ h₁² + h₂².
    intro h
    have h1 := quadratische_form_schranke 6 (-3) 6 (by norm_num) (by norm_num) h.1 h.2
    have h2 := norm_sq_le_quadratsumme h
    norm_num at h1
    nlinarith
  · -- Der Rest: Nach der Taylorformel `kubisch_taylor` ist
    -- f((1, 1) + h) - f(1, 1) - Q(h)/2 = h₁³ + h₂³ ...
    have e : (fun h : ℝ × ℝ => kubisch ((1, 1) + h) - kubisch (1, 1)
        - (6 * h.1 ^ 2 - 6 * h.1 * h.2 + 6 * h.2 ^ 2) / 2) = fun h => h.1 ^ 3 + h.2 ^ 3 := by
      funext h
      rw [kubisch_taylor]
      simp only [kubisch]
      ring
    rw [e]
    -- ... und das ist o(‖h‖²): Zu c > 0 nimmt man die Kugel vom Radius c/2. Dort ist
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

/-- (0, 0) ist ein Sattelpunkt. Auf der Diagonalen ist f(t, t) = t²(2t - 3) < 0 fuer
kleine t ≠ 0, auf der Gegendiagonalen f(t, -t) = 3t² > 0: die Eigenrichtungen (1, 1)
und (1, -1) der Hesse-Matrix [[0, -3], [-3, 0]], mit den Eigenwerten -3 und 3. -/
theorem kubisch_sattel : Sattelpunkt kubisch 0 := by
  refine ⟨?_, ?_, ?_⟩
  · -- Kritisch: die Ableitung aus den Regeln, in 0 ist sie die Nullabbildung.
    have hx := hasFDerivAt_fst (𝕜 := ℝ) (p := (0 : ℝ × ℝ))
    have hy := hasFDerivAt_snd (𝕜 := ℝ) (p := (0 : ℝ × ℝ))
    have h := ((hx.pow 3).add (hy.pow 3)).sub ((hx.const_mul (3 : ℝ)).mul hy)
    unfold kubisch
    convert h using 1
    ext <;> simp
  · -- Kein Minimum: der Punkt (t, t) mit t = min(δ/2, 1) liegt in der Kugel vom
    -- Radius δ, und dort ist f = t²(2t - 3) < 0 = f(0).
    intro h
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    set t := min (δ / 2) 1
    have ht : 0 < t := lt_min (half_pos hδ) one_pos
    have ht1 : t ≤ 1 := min_le_right _ _           -- also 2t - 3 < 0
    have htδ : t < δ := lt_of_le_of_lt (min_le_left _ _) (half_lt_self hδ)
    have h3 := hball (y := (t, t)) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      simp only [Prod.fst_zero, Prod.snd_zero, sub_zero, abs_of_pos ht]
      exact ⟨htδ, htδ⟩)
    simp only [kubisch, Prod.fst_zero, Prod.snd_zero] at h3   -- f(0) ≤ f(t, t)
    nlinarith [mul_pos (mul_pos ht ht) (by linarith : (0 : ℝ) < 3 - 2 * t)]
  · -- Kein Maximum: der Punkt (δ/2, -δ/2), dort ist f = 3(δ/2)² > 0 = f(0).
    intro h
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 h
    have ht : 0 < δ / 2 := half_pos hδ
    have h3 := hball (y := (δ / 2, -(δ / 2))) (by
      rw [Prod.dist_eq, max_lt_iff, Real.dist_eq, Real.dist_eq]
      simp only [Prod.fst_zero, Prod.snd_zero, sub_zero, abs_neg, abs_of_pos ht]
      exact ⟨half_lt_self hδ, half_lt_self hδ⟩)
    simp only [kubisch, Prod.fst_zero, Prod.snd_zero] at h3   -- f(δ/2, -δ/2) ≤ f(0)
    nlinarith [mul_pos ht ht]

-- Folie: "Hoehere Dimensionen"
/- Hoehere Dimensionen in Mathlib: positiv definite Matrizen, die Eigenwerte einer
   symmetrischen (allgemeiner: hermiteschen) Matrix und der Spektralsatz. Positiv
   definit heisst: alle Eigenwerte sind positiv. Fermat steht dort fuer jede Dimension,
   den Test mit der zweiten Ableitung gibt es nur in einer Variablen. Ein
   Hesse-Kriterium in mehreren Veraenderlichen und das Kriterium von Sylvester fehlen. -/
#check @Matrix.PosDef                                   -- positiv definit
#check @Matrix.IsHermitian.posDef_iff_eigenvalues_pos   -- ... genau wenn alle Eigenwerte > 0
#check @Matrix.IsHermitian.spectral_theorem             -- Spektralsatz
#check @IsLocalMin.hasFDerivAt_eq_zero                  -- Fermat, jede Dimension
#check @isLocalMin_of_deriv_deriv_pos                   -- f'' > 0, eine Variable


/-! ## 4.5 Gradientenabstieg -/

-- Folie: "Der negative Gradient zeigt bergab"
/- Cauchy-Schwarz: In jede Richtung v mit ‖v‖ = 1 ist ⟨g, v⟩ ≥ -‖g‖; und in der
   Richtung -g/‖g‖ wird -‖g‖ erreicht. Also zeigt -g in die Richtung des steilsten
   Abstiegs -- mit g = ∇f(x) ist ⟨g, v⟩ die Richtungsableitung. -/
theorem steilster_abstieg {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (g : V) :
    (∀ v : V, ‖v‖ = 1 → -‖g‖ ≤ inner ℝ g v) ∧
      (g ≠ 0 → inner ℝ g ((-‖g‖⁻¹) • g) = -‖g‖) := by
  -- Der Satz hat zwei Teile (`constructor`): (1) in jede Einheitsrichtung ist ⟨g, v⟩ ≥ -‖g‖,
  -- (2) in der Richtung -g/‖g‖ wird -‖g‖ erreicht.
  constructor
  · -- Teil (1), Beweis auf der Folie: Cauchy-Schwarz.
    -- Sei v mit ‖v‖ = 1 (hv).
    intro v hv
    -- Schritt 1: Cauchy-Schwarz (`abs_real_inner_le_norm`): |⟨g, v⟩| ≤ ‖g‖·‖v‖.
    have h := abs_real_inner_le_norm g v
    -- Schritt 2: ‖v‖ = 1 einsetzen. h : |⟨g, v⟩| ≤ ‖g‖.
    rw [hv, mul_one] at h
    -- Schritt 3: |a| ≤ b heisst -b ≤ a ≤ b (`abs_le`); die untere Haelfte ist das Ziel.
    exact (abs_le.1 h).1
  · -- Teil (2), "Gleichheit gilt, wenn v entgegengesetzt zu g zeigt": Sei g ≠ 0 (hg).
    intro hg
    -- Schritt 1: Dann ist ‖g‖ ≠ 0 (hn), wir duerfen durch ‖g‖ teilen.
    have hn : ‖g‖ ≠ 0 := norm_ne_zero_iff.2 hg
    -- Schritt 2: Das Skalar -‖g‖⁻¹ aus der rechten Seite des Skalarprodukts ziehen
    -- (`inner_smul_right`), und ⟨g, g⟩ = ‖g‖² (`real_inner_self_eq_norm_sq`). Ziel:
    --   -‖g‖⁻¹ * ‖g‖ ^ 2 = -‖g‖
    rw [inner_smul_right, real_inner_self_eq_norm_sq]
    -- Schritt 3: Kuerzen. `field_simp` benutzt hn : ‖g‖ ≠ 0.
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

section Quadratisch
variable (a m c : ℝ)

-- Folie: "Ein kleines Beispiel: f(x) = x^2" (allgemeiner: q(x) = a(x - m)^2 + c)
/-- Modellproblem `q(x) = a·(x - m)² + c`; fuer `a > 0` Minimum bei `m`. -/
noncomputable def q (x : ℝ) : ℝ := a * (x - m) ^ 2 + c

lemma hasDerivAt_q (x : ℝ) : HasDerivAt (q a m c) (2 * a * (x - m)) x := by
  -- Schritt 0: `q a m c` ist ausgeschrieben die Funktion x ↦ a·(x - m)² + c (`show` zeigt es).
  show HasDerivAt (fun x : ℝ => a * (x - m) ^ 2 + c) (2 * a * (x - m)) x
  -- Schritt 1: Die Regeln bauen sie von innen nach aussen auf. x - m hat die Ableitung 1
  -- (`hasDerivAt_id`, `sub_const`).
  have h1 : HasDerivAt (fun y : ℝ => y - m) 1 x := (hasDerivAt_id x).sub_const m
  -- Schritt 2: Quadrat (`pow 2`: (u²)' = 2u·u'), Faktor a (`const_mul`), Konstante c
  -- (`add_const`). Heraus kommt die Ableitung in der Form  a * (2 * (x - m) ^ (2 - 1) * 1),
  -- mit der natuerlichen Zahl 2 als Exponent und Faktor.
  have h2 := ((h1.pow 2).const_mul a).add_const c
  -- Schritt 3: `convert` laesst nur die Gleichung der beiden Ableitungswerte uebrig
  -- (die Funktionen sind gleich); `push_cast` macht aus der natuerlichen Zahl 2 die reelle
  -- Zahl 2, `ring` rechnet nach: 2·a·(x - m) = a·(2·(x - m)·1).
  convert h2 using 1
  push_cast
  ring

lemma deriv_q (x : ℝ) : deriv (q a m c) x = 2 * a * (x - m) :=
  (hasDerivAt_q a m c x).deriv

/-- `m` ist globales Minimum, sobald `a ≥ 0`. -/
lemma q_min (ha : 0 ≤ a) (x : ℝ) : q a m c m ≤ q a m c x := by
  -- Schritt 0: Beide Seiten ausschreiben. Ziel:  a·(m - m)² + c ≤ a·(x - m)² + c
  show a * (m - m) ^ 2 + c ≤ a * (x - m) ^ 2 + c
  -- Schritt 1: Der Unterschied ist q x - q m = a·(x - m)² - a·0² = a·(x - m)², und das ist
  -- ≥ 0, denn a ≥ 0 und ein Quadrat ist ≥ 0 (`mul_nonneg`, `sq_nonneg`). `nlinarith`
  -- bekommt dieses Produkt als Hinweis; mit m - m = 0 ist der Rest linear.
  nlinarith [mul_nonneg ha (sq_nonneg (x - m))]

-- Folie: "Der Konvergenzsatz in Lean" (von Hand auf "Ein Schritt, exakt
-- ausgerechnet", "Fehlerformel und Konvergenz" und "Die Schrittweitenbedingung")
/-- Ein Schritt multipliziert den Fehler mit dem Faktor `1 - 2at`. -/
lemma gd_q_schritt (t x : ℝ) :
    gdSchritt (q a m c) t x - m = (1 - 2 * a * t) * (x - m) := by
  -- Schritt 1 (= 1. "Ein Schritt" auf der Folie): Den Schritt ausschreiben,
  --   x_{k+1} - m = x - t * q'(x) - m   (Definition von `gdSchritt`).
  show x - t * deriv (q a m c) x - m = (1 - 2 * a * t) * (x - m)
  -- Schritt 2: q'(x) = 2a(x - m) einsetzen (`deriv_q`). Ziel:
  --   x - t * (2 * a * (x - m)) - m = (1 - 2 * a * t) * (x - m)
  rw [deriv_q]
  -- Schritt 3: Ausmultiplizieren, beide Seiten sind gleich: `ring`.
  ring

/-- Exakte Fehlerformel nach `n` Schritten. -/
lemma gd_q_fehler (t x₀ : ℝ) (n : ℕ) :
    gdIter (q a m c) t x₀ n - m = (1 - 2 * a * t) ^ n * (x₀ - m) := by
  -- Induktion ueber n, wie auf der Folie "Fehlerformel und Konvergenz".
  induction n with
  | zero =>
      -- Induktionsanfang n = 0: gdIter ... 0 ist x₀ und (1 - 2at)^0 = 1, es bleibt
      -- x₀ - m = 1·(x₀ - m); `simp` entfaltet gdIter und rechnet das aus.
      simp [gdIter]
  | succ k ih =>
      -- Induktionsschritt k → k + 1, mit der Induktionsannahme (IA)
      --   ih : gdIter ... k - m = (1 - 2at)^k * (x₀ - m).
      -- Schritt 1: Eine Iteration entfalten (`gdIter` fuer k + 1). Ziel:
      --   gdSchritt (q a m c) t (gdIter ... k) - m = (1 - 2at)^(k+1) * (x₀ - m)
      simp only [gdIter]
      -- Schritt 2: "erst ein Schritt, dann die IA": `gd_q_schritt` macht aus dem Fehler nach
      -- dem Schritt den Faktor (1 - 2at) mal den alten Fehler, `ih` ersetzt den alten Fehler
      -- durch (1 - 2at)^k·(x₀ - m). Ziel:
      --   (1 - 2at) * ((1 - 2at)^k * (x₀ - m)) = (1 - 2at)^(k+1) * (x₀ - m)
      rw [gd_q_schritt, ih]
      -- Schritt 3: Potenzgesetz r · r^k = r^(k+1): `ring`.
      ring

/-- **Konvergenzsatz.** Ist `|1 - 2at| < 1`, so konvergiert der
Gradientenabstieg gegen `m` (fuer `a > 0` das Minimum). -/
theorem gd_q_tendsto (t : ℝ) (h : |1 - 2 * a * t| < 1) (x₀ : ℝ) :
    Tendsto (fun n => gdIter (q a m c) t x₀ n) atTop (𝓝 m) := by
  -- Schritt 1: Die Fehlerformel nach x_n aufgeloest: x_n = (1 - 2at)^n·(x₀ - m) + m.
  -- `linarith` genuegt: Das Produkt (1 - 2at)^n·(x₀ - m) ist fuer `linarith` ein Block, und
  -- gd_q_fehler sagt  x_n - m = <Block>.
  have key : ∀ n,
      gdIter (q a m c) t x₀ n = (1 - 2 * a * t) ^ n * (x₀ - m) + m := by
    intro n
    have := gd_q_fehler a m c t x₀ n
    linarith
  -- Schritt 2 ("r^n → 0 fuer |r| < 1" auf der Folie): Das steht in Mathlib
  -- (`tendsto_pow_atTop_nhds_zero_of_norm_lt_one`). Verlangt ist ‖r‖ < 1, wir haben |r| < 1
  -- (h); in ℝ ist die Norm der Betrag (`Real.norm_eq_abs`), `rwa` schliesst mit h.
  have hpow : Tendsto (fun n : ℕ => (1 - 2 * a * t) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_norm_lt_one (by rwa [Real.norm_eq_abs])
  -- Schritt 3: Grenzwertsaetze, kein ε und kein N. `mul_const (x₀ - m)`: mit der Konstanten
  -- multiplizieren, r^n·(x₀ - m) → 0·(x₀ - m); `add_const m`: plus m, → 0·(x₀ - m) + m.
  -- `simpa [key]` ersetzt x_n durch die Formel aus Schritt 1 und rechnet 0·(x₀ - m) + m = m.
  simpa [key] using (hpow.mul_const (x₀ - m)).add_const m

/-- Die zulaessigen Schrittweiten: genau `0 < t < 1/a`. Zu grosse Schritte
divergieren, zu kleine konvergieren langsam. -/
theorem schrittweite (ha : 0 < a) (t : ℝ) :
    |1 - 2 * a * t| < 1 ↔ (0 < t ∧ t < 1 / a) := by
  -- Schritt 1 (= erste Zeile der Folie "Die Schrittweitenbedingung"): Den Betrag aufloesen,
  -- |x| < 1 ↔ -1 < x ∧ x < 1 (`abs_lt`), und rechts t < 1/a mit a durchmultiplizieren,
  -- t < 1/a ↔ t·a < 1 (`lt_div_iff₀`, braucht a > 0). Ziel:
  --   -1 < 1 - 2at ∧ 1 - 2at < 1  ↔  0 < t ∧ t·a < 1
  rw [abs_lt, lt_div_iff₀ ha]
  constructor
  · -- (→) h1 : -1 < 1 - 2at und h2 : 1 - 2at < 1.
    -- Aus h1 folgt a·t < 1 (also t·a < 1), aus h2 folgt a·t > 0 und damit t > 0, denn a > 0.
    -- Die Produkte a·t sind nicht linear; das Teilen durch a erledigt `nlinarith` mit dem
    -- Produkt aus der Annahme und a > 0.
    rintro ⟨h1, h2⟩
    exact ⟨by nlinarith, by nlinarith⟩
  · -- (←) h1 : 0 < t und h2 : t·a < 1.
    -- -1 < 1 - 2at gilt, weil t·a < 1; 1 - 2at < 1 gilt, weil a·t > 0 (Produkt aus a > 0
    -- und t > 0, das findet `nlinarith`).
    rintro ⟨h1, h2⟩
    exact ⟨by nlinarith, by nlinarith⟩

end Quadratisch

-- Folie: "Ein kleines Beispiel: f(x) = x^2" (der Fall a = 1, m = c = 0)
/- Fuer f(x) = x² ist ein Schritt x ↦ (1 - 2t)·x, nach n Schritten steht
   (1 - 2t)^n·x₀ da -- und das geht gegen 0, sobald 0 < t < 1: -/
example (t x₀ : ℝ) (n : ℕ) : gdIter (fun x : ℝ => x ^ 2) t x₀ n = (1 - 2 * t) ^ n * x₀ := by
  -- Schritt 1: x² ist der Fall a = 1, m = c = 0 von q: q 1 0 0 x = 1·(x - 0)² + 0 = x²
  -- (`funext x` nimmt ein x, `simp [q]` entfaltet q und rechnet).
  have e : (fun x : ℝ => x ^ 2) = q 1 0 0 := by funext x; simp [q]
  -- Schritt 2: Die Fehlerformel fuer q mit diesen Werten: x_n - 0 = (1 - 2·1·t)^n·(x₀ - 0).
  have h := gd_q_fehler 1 0 0 t x₀ n
  rw [e]                                  -- x² durch q 1 0 0 ersetzen
  -- Schritt 3: `simpa` vereinfacht - 0 und 2·1 und stimmt dann mit dem Ziel ueberein.
  simpa using h

example (t x₀ : ℝ) (ht : 0 < t ∧ t < 1) :
    Tendsto (fun n => gdIter (fun x : ℝ => x ^ 2) t x₀ n) atTop (𝓝 0) := by
  -- Schritt 1: wie oben, x² ist q 1 0 0.
  have e : (fun x : ℝ => x ^ 2) = q 1 0 0 := by funext x; simp [q]
  rw [e]
  -- Schritt 2: Der Konvergenzsatz `gd_q_tendsto` verlangt |1 - 2·1·t| < 1. Das ist nach
  -- `schrittweite` (mit a = 1) dasselbe wie 0 < t ∧ t < 1/1; `.2` nimmt die Richtung "←",
  -- und `simpa using ht` macht aus der Annahme 0 < t ∧ t < 1 die Form 0 < t ∧ t < 1/1.
  exact gd_q_tendsto 1 0 0 t ((schrittweite 1 one_pos t).2 (by simpa using ht)) x₀

-- Folie: "Gradientenabstieg in zwei Variablen"
/- Gradientenabstieg fuer f(x, y) = x² + xy + y² - x - y: Der Fehler zum
   Minimum (1/3, 1/3) wird in jedem Schritt mit I - tA multipliziert,
   A = [[2, 1], [1, 2]] die Hesse-Matrix. Die Richtungen (1, 1) und (1, -1)
   streckt A um 3 und um 1 -- dort ist der Faktor 1 - 3t bzw. 1 - t. -/
example : (!![2, 1; 1, 2] : Matrix (Fin 2) (Fin 2) ℝ).mulVec ![1, 1] =
    (3 : ℝ) • ![1, 1] := by
  -- Gleichheit von Vektoren pruefen wir je Koordinate (`ext i`, die Faelle 0 und 1): `simp`
  -- schreibt "Matrix mal Vektor" als Summe aus, `norm_num` rechnet nach. Erste Koordinate:
  -- 2·1 + 1·1 = 3 = 3·1; zweite: 1·1 + 2·1 = 3 = 3·1.
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> norm_num

example : (!![2, 1; 1, 2] : Matrix (Fin 2) (Fin 2) ℝ).mulVec ![1, -1] =
    (1 : ℝ) • ![1, -1] := by
  -- Genauso: 2·1 + 1·(-1) = 1 = 1·1 und 1·1 + 2·(-1) = -1 = 1·(-1).
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> norm_num

/- Ein Schritt, ausgerechnet: Mit e = (x - 1/3, y - 1/3) ist der neue Fehler
   (I - tA) e. -/
example (t x y : ℝ) :
    (x - t * (2 * x + y - 1) - 1 / 3, y - t * (x + 2 * y - 1) - 1 / 3) =
      ((1 - 2 * t) * (x - 1 / 3) - t * (y - 1 / 3),
       -t * (x - 1 / 3) + (1 - 2 * t) * (y - 1 / 3)) := by
  -- Ein Paar ist gleich, wenn beide Komponenten es sind (`ext`); jede Komponente ist eine
  -- Gleichung zwischen Polynomen in x, y, t, die `ring` nachrechnet. Zum Beispiel erste
  -- Komponente: x - t(2x + y - 1) - 1/3 = (1 - 2t)(x - 1/3) - t(y - 1/3).
  ext <;> ring

-- Folie: "Konvex und kritisch: ein globales Minimum"
/-- Fuer konvexe f ist ein Punkt mit Ableitung 0 ein globales Minimum: Rechts
von c sind die Sekanten mindestens so steil wie f'(c) = 0, links hoechstens.
Mit `deriv f c = 0` allein waere es falsch (Junk-Wert). -/
theorem konvex_kritisch {f : ℝ → ℝ} {c : ℝ} (hf : ConvexOn ℝ Set.univ f)
    (hc : HasDerivAt f 0 c) (y : ℝ) : f c ≤ f y := by
  -- Drei Faelle (`lt_trichotomy`): y < c, y = c, c < y -- wie auf der Folie.
  rcases lt_trichotomy y c with hy | rfl | hy
  · -- Fall y < c. "Sekante gegen Tangente": Fuer konvexes f ist die Sekantensteigung von y nach c
    -- hoechstens die Tangentensteigung in c (`slope_le_of_hasDerivAt`; alle Punkte liegen in
    -- univ). Mit f'(c) = 0 ist  h : slope f y c ≤ 0.
    have h := hf.slope_le_of_hasDerivAt (Set.mem_univ y) (Set.mem_univ c) hy hc
    -- Eine Steigung ≤ 0 mit y ≤ c heisst: Der Zaehler f c - f y ist ≤ 0, also f c ≤ f y
    -- (`slope_nonpos_iff_of_le`, Richtung "→").
    exact (slope_nonpos_iff_of_le hy.le).1 h
  · -- Fall y = c: `rfl` setzt y := c ein, es bleibt f c ≤ f c.
    exact le_rfl
  · -- Fall c < y. Jetzt umgekehrt: Die Tangentensteigung in c ist hoechstens die Sekantensteigung
    -- von c nach y (`le_slope_of_hasDerivAt`), also  h : 0 ≤ slope f c y.
    have h := hf.le_slope_of_hasDerivAt (Set.mem_univ c) (Set.mem_univ y) hy hc
    -- Eine Steigung ≥ 0 mit c ≤ y heisst: Der Zaehler f y - f c ist ≥ 0, also f c ≤ f y
    -- (`slope_nonneg_iff_of_le`).
    exact (slope_nonneg_iff_of_le hy.le).1 h

/-! ## 4.6 Optima unter Nebenbedingungen: Lagrange-Multiplikatoren -/

-- Folie: "Extremum unter einer Nebenbedingung"
/-- Lokales Maximum unter einer Nebenbedingung, von Hand: `x₀` liegt in
`M = {x | f x = c}`, und auf einer Kugel um `x₀` ist `φ x ≤ φ x₀` fuer
alle `x` aus `M`. Fuer Punkte ausserhalb von `M` verlangt sie nichts. -/
def MaxUnterNB {E : Type*} [MetricSpace E]
    (φ f : E → ℝ) (c : ℝ) (x₀ : E) : Prop :=
  f x₀ = c ∧ ∃ ε > 0, ∀ x, dist x x₀ < ε → f x = c → φ x ≤ φ x₀

/-- Das Beispiel: `xy` hat auf der Geraden `x + y = 1` in `(1/2, 1/2)`
ein Maximum. -/
theorem xy_max_auf_gerade :
    MaxUnterNB (fun p : ℝ × ℝ => p.1 * p.2) (fun p => p.1 + p.2) 1
      (1 / 2, 1 / 2) := by
  -- Erstens liegt (1/2, 1/2) auf der Geraden. Zweitens tut es jede Kugel,
  -- etwa ε = 1: Die Ungleichung gilt sogar auf der ganzen Geraden.
  refine ⟨by norm_num, 1, one_pos, fun p _ (hp : p.1 + p.2 = 1) => ?_⟩
  -- Nach y aufloesen: p.2 = 1 - p.1, und einsetzen.
  have hy : p.2 = 1 - p.1 := by linarith
  show p.1 * p.2 ≤ 1 / 2 * (1 / 2)
  rw [hy]
  -- Zu zeigen ist x(1 - x) ≤ 1/4, und 1/4 - x(1 - x) = (x - 1/2)² ≥ 0.
  nlinarith [sq_nonneg (p.1 - 1 / 2)]

/-- Mathlib nennt dasselbe `IsLocalMaxOn φ M x₀`: schliesslich laengs
`𝓝[M] x₀`, also fuer die Punkte von `M` nahe `x₀`, gilt `φ x ≤ φ x₀`.
Mit `c = f x₀` liegt `x₀` von selbst in `M`, und `M = {x | f x = f x₀}`
ist die Form, in der Mathlib die Nebenbedingung schreibt. Fuer das
Minimum heisst es `IsLocalMinOn`, fuer "Maximum oder Minimum"
`IsLocalExtrOn`. -/
theorem maxUnterNB_iff {E : Type*} [MetricSpace E] {φ f : E → ℝ} {x₀ : E} :
    MaxUnterNB φ f (f x₀) x₀ ↔ IsLocalMaxOn φ {x | f x = f x₀} x₀ := by
  -- Mathlibs Seite ausschreiben: `IsLocalMaxOn` heisst "schliesslich
  -- laengs 𝓝[M] x₀ gilt φ x ≤ φ x₀" (`IsLocalMaxOn`, `IsMaxFilter`).
  -- Schliesslich laengs 𝓝[M] x₀ heisst: schliesslich laengs 𝓝 x₀ fuer
  -- die Punkte von M (`eventually_nhdsWithin_iff`), und das heisst: auf
  -- einer Kugel um x₀ (`Metric.eventually_nhds_iff`).
  rw [IsLocalMaxOn, IsMaxFilter, eventually_nhdsWithin_iff,
    Metric.eventually_nhds_iff]
  -- Jetzt steht rechts unsere Kugel-Bedingung. Links kommt f x₀ = f x₀
  -- dazu: Hin vergisst man es (`h.2`), zurueck gilt es mit `rfl`.
  exact ⟨fun h => h.2, fun h => ⟨rfl, h⟩⟩

-- Folie: "Ein Beispiel: xy unter x + y = 1" (das Maximum bestaetigen)
/-- Beispiel: maximiere `x·y` unter `x + y = 1`.
Lagrange liefert den Kandidaten `x = y = 1/2`; hier die elementare
Bestaetigung, dass dieser Kandidat wirklich das Maximum ist. -/
theorem xy_le_quarter (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  -- Der Beweis von der Folie: (x + y)² - 4xy = (x - y)² ≥ 0, und x + y = 1 gibt
  -- 1 - 4xy ≥ 0, also xy ≤ 1/4. `nlinarith` bekommt den Hinweis (x - y)² ≥ 0 und
  -- multipliziert h mit sich selbst, um (x + y)² = 1 zu bekommen; der Rest ist linear.
  nlinarith [sq_nonneg (x - y)]

/- Der Wert wird im Lagrange-Punkt angenommen. -/
example : (1 / 2 : ℝ) * (1 / 2) = 1 / 4 := by norm_num

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

/-! ### Mathlib-Wegweiser zu Vorlesung 6

Nur hier in der Datei, nicht im Skript und nicht auf den Folien. -/
#check @IsLocalMin.hasFDerivAt_eq_zero                 -- Fermat, jede Dimension
#check @ConvexOn.slope_le_of_hasDerivAt                -- Sekante, Tangente
#check @IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt  -- mehrere NB
#check @HasStrictFDerivAt.implicitFunction             -- implizite Funktionen

end Skript.Analysis
