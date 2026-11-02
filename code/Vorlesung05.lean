/-
  Angewandte Mathematik -- Begleitcode zur Vorlesung 5
  (Folien: folien/vorlesung05.tex)
-/
import Mathlib

namespace Skript.Analysis

open Filter Topology

/-! ## 4.2 Wiederholung: "schliesslich" laengs eines Filters -/

-- Wiederholung aus Vorlesung 4 (code/Vorlesung04.lean, Abschnitt 4.2).
-- Folie: "Rueckblick: Vorlesung 4" (Punkt "Schliesslich")
/- "P gilt schliesslich laengs F" heisst: Die Menge der Punkte, an denen P
   gilt, ist gross -- sie liegt im Filter F. Mathlib hat dafuer einen eigenen
   Quantor, `∀ᶠ x in F, P x`, gelesen "fuer schliesslich alle x laengs F gilt
   P x". Er ist gebaut wie `∀ x, P x`, nur steht hinter `in` der Filter.
   Ausgeschrieben heisst er `Filter.Eventually`, in Lemmanamen steht deshalb
   `eventually`. Per Definition ist er genau die Mengenaussage: -/
example {α : Type*} (F : Filter α) (P : α → Prop) :
    (∀ᶠ x in F, P x) ↔ {x | P x} ∈ F := Iff.rfl

/- Man kann die Menge zeichnen. Bei `atTop` und P x = "10 < x":

     --|----|----|----|----|----|------> x
       8    9   10   11   12   13
                 o=====================>    {x | 10 < x}
                      [================>    das Endstueck [11, ∞)

   Das Endstueck [11, ∞) liegt ganz in der Menge {x | 10 < x}, also ist sie gross
   (Axiom (ii)): `10 < x` gilt schliesslich. Was links von 10 geschieht, zaehlt nicht. -/
example : {x : ℝ | 10 < x} ∈ atTop := by
  apply mem_of_superset (Ici_mem_atTop (11 : ℝ))   -- das Endstueck [11, ∞) ist gross
  intro x hx                                       -- hx : 11 ≤ x
  have h : (11 : ℝ) ≤ x := hx
  show 10 < x                                      -- x liegt in {x | 10 < x}
  linarith

/- `∀ x, P x` verlangt mehr: P gilt ueberall, die Menge {x | P x} ist das
   Ganze. Was fuer alle x gilt, gilt also schliesslich -- das Ganze ist gross
   (Axiom (i)), und {x | P x} enthaelt es (Axiom (ii)). In Mathlib heisst das
   `Eventually.of_forall`.

   Die Zeile `apply mem_of_superset univ_mem` im Einzelnen. Axiom (ii) heisst in
   Mathlib `mem_of_superset`: Aus `U ∈ F` und `U ⊆ V` folgt `V ∈ F` -- eine
   Obermenge V einer grossen Menge U ist gross. Hier ist V = {x | P x}, und fuer U
   nehmen wir das Ganze `univ`; das ist gross nach Axiom (i), `univ_mem`. Also:

     Ziel vorher:   {x | P x} ∈ F
     Beweisstueck:  univ_mem : univ ∈ F        (U = univ ist gross)
     Ziel danach:   univ ⊆ {x | P x}           (der zweite Teil, U ⊆ V)

   `apply` setzt das Beweisstueck fuer `U ∈ F` ein und laesst den Rest als neues
   Ziel stehen. Das Ziel `univ ⊆ {x | P x}` heisst: Jedes x liegt in {x | P x},
   also gilt P x -- und das ist die Annahme `h`. -/
example {α : Type*} (F : Filter α) (P : α → Prop) (h : ∀ x, P x) :
    ∀ᶠ x in F, P x := by
  change {x | P x} ∈ F                  -- die Definition von ∀ᶠ
  apply mem_of_superset univ_mem        -- (i), (ii): Obermenge des Ganzen
  intro x _                             -- Ziel: x ∈ {x | P x}, also P x
  exact h x                             -- das ist die Annahme h

/- Was gross ist, sagt der Filter. Laengs `atTop` heisst schliesslich "ab
   einem N" (`eventually_atTop`). Schliesslich ist 10 < n, ab N = 11: -/
example : ∀ᶠ n : ℕ in atTop, 10 < n := by
  rw [eventually_atTop]                 -- ∃ N, ∀ n ≥ N, 10 < n
  use 11                                -- Zeuge N = 11
  intro n hn                            -- hn : n ≥ 11
  omega                                 -- 10 < n

/- Fuer alle n gilt es aber nicht: Aus `∀ᶠ` folgt nicht `∀`. -/
example : ¬ ∀ n : ℕ, 10 < n := by
  intro h                               -- Annahme: 10 < n fuer alle n
  have h0 := h 0                        -- also 10 < 0
  omega                                 -- Widerspruch

/- Laengs `𝓝 a` heisst schliesslich "auf einer Kugel um a"
   (`Metric.eventually_nhds_iff`). Nahe 1 ist x positiv, die Kugel vom
   Radius 1 reicht: -/
example : ∀ᶠ x : ℝ in 𝓝 1, 0 < x := by
  rw [Metric.eventually_nhds_iff]       -- ∃ δ > 0, ∀ x, dist x 1 < δ → 0 < x
  use 1, one_pos                        -- Radius δ = 1
  intro x hx                            -- hx : dist x 1 < 1
  rw [Real.dist_eq, abs_lt] at hx       -- -1 < x - 1 < 1
  linarith [hx.1]                       -- also 0 < x

/- In den Beweisen unten nennt `filter_upwards [...] with x hx` eine grosse
   Menge -- oder eine Aussage, die schliesslich gilt; per Definition dasselbe.
   Zu zeigen bleibt P x fuer die x darin, dann ist {x | P x} gross nach
   Axiom (ii). Hier mit der Kugel B(1, 1): -/
example : ∀ᶠ x : ℝ in 𝓝 1, 0 < x := by
  filter_upwards [Metric.ball_mem_nhds (1 : ℝ) one_pos] with x hx  -- B(1, 1) ist gross
  rw [Metric.mem_ball, Real.dist_eq, abs_lt] at hx                -- -1 < x - 1 < 1
  linarith [hx.1]                                                  -- also 0 < x

/- Und die klassische Konvergenz? Am einfachsten Beispiel, u n = 1/n gegen 0, ist
   es dieselbe Aussage in drei Sprachen:

     klassisch    zu jedem ε > 0 ein N, so dass fuer alle n ≥ N gilt |1/n - 0| < ε
     mit ∀ᶠ       zu jedem ε > 0 gilt schliesslich laengs atTop |1/n - 0| < ε
     mit Filtern  das Urbild jeder Kugel B(0, ε) ist laengs atTop gross (`Tendsto`)

   "Es gibt ein N, so dass fuer alle n ≥ N" ist genau `∀ᶠ n in atTop`, und
   "schliesslich" heisst "die Menge ist gross". Zu ε = 1/10 heisst |1/n - 0| < 1/10
   dasselbe wie 10 < n, also ab N = 11 -- das Beispiel von oben: -/
example : ∀ᶠ n : ℕ in atTop, |1 / (n : ℝ) - 0| < 1 / 10 := by
  rw [eventually_atTop]                 -- ∃ N, ∀ n ≥ N, |1/n - 0| < 1/10
  use 11                                -- Zeuge N = 11
  intro n hn                            -- hn : n ≥ 11
  -- Schritt 1: n ist eine natuerliche Zahl; fuer die Rechnung braucht man 11 ≤ n in ℝ
  -- (`exact_mod_cast`: aus der Aussage in ℕ wird die in ℝ).
  have h : (11 : ℝ) ≤ n := by exact_mod_cast hn
  -- Schritt 2: 1/n - 0 = 1/n (`sub_zero`), und 1/n > 0 ist positiv (n ≥ 11 > 0), also ist
  -- der Betrag |1/n| = 1/n (`abs_of_pos`). Ziel: 1 / n < 1 / 10.
  rw [sub_zero, abs_of_pos (by positivity)]
  -- Schritt 3: Brueche mit positiven Nennern vergleichen: 1/n < 1/10 ↔ 1·10 < 1·n
  -- (`div_lt_div_iff₀`; die Nenner n und 10 sind positiv). Ziel: 1 * 10 < 1 * n.
  rw [div_lt_div_iff₀ (by linarith) (by norm_num)]
  linarith                              -- 10 < n, denn n ≥ 11

/-! ## 4.4 Ableitung -/

/-! ### Landau-Symbole: Vergleichen laengs eines Filters

Erst selbst gebaut, dann mit Mathlib abgeglichen -- wie in Vorlesung 4:
`KleinO` ist die Definition der Folie, hier fuer reelle Funktionen (mit |·|
statt ‖·‖). Mathlibs `=o[l]` ist derselbe Begriff fuer normierte Raeume; der
Abgleich steht am Ende des Abschnitts. -/

-- Folie: "Vernachlaessigbar klein" (Filter, "schliesslich", o und O)
/- Die Bausteine, in einer Zeile gelesen:

     Filter l            ein System "grosser" Mengen: Er sagt, wohin x laeuft
       atTop             x → +∞,  gross = enthaelt ein Endstueck [N, ∞)
       𝓝 a               x → a,   gross = enthaelt eine Kugel um a
       𝓝[≠] a            x → a, x ≠ a,  gross = eine Kugel ohne a
     ∀ᶠ x in l, P x      "P x gilt schliesslich laengs l": {x | P x} ist gross
     f = o(g) laengs l   zu JEDEM c > 0 gilt schliesslich |f x| ≤ c·|g x|
                         (f ist neben g vernachlaessigbar)
     f = O(g) laengs l   fuer EIN c gilt schliesslich |f x| ≤ c·|g x|
                         (f ist hoechstens so gross wie g)
     HasDerivAt g a t₀   der Fehler g t - g t₀ - a·(t - t₀) ist o(t - t₀)
                         laengs 𝓝 t₀ (die Tangente ist die beste Gerade)

   Zum Beispiel gilt `10 < x` schliesslich laengs atTop: ab N = 11. Wie es bei
   x = 3 aussieht, ist egal. -/

-- Folie: "Klein-o laengs eines Filters" (die Definition, selbst gebaut)
/-- Klein-o laengs eines Filters: Zu jedem c > 0 gilt schliesslich
|f x| ≤ c·|g x|. Erst kommt c, dann die grosse Menge -- zu jedem c eine
eigene. `∀ᶠ x in l, P x` heisst: Die Menge, auf der P gilt, ist gross. -/
def KleinO {α : Type*} (f g : α → ℝ) (l : Filter α) : Prop :=
  ∀ c > 0, ∀ᶠ x in l, |f x| ≤ c * |g x|

/- Gross-O, f = O(g), heisst: Es gibt *ein* c, mit dem schliesslich
   |f x| ≤ c·|g x| gilt, nicht jedes. Hier brauchen wir nur o; in Mathlib ist
   O das Symbol `f =O[l] g`. -/

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

/-- Bei +∞ ist es umgekehrt: x ist klein gegen x². Zu c > 0 reicht das
Endstueck ab 1/c, denn dort ist 1 ≤ c·x, also x ≤ c·x². -/
theorem linear_klein_o_unendlich :
    KleinO (fun x : ℝ => x) (fun x => x ^ 2) atTop := by
  -- Schritt 1: c > 0 beliebig. Ziel: ∀ᶠ x in atTop, |x| ≤ c * |x ^ 2|
  intro c hc
  -- Schritt 2: Das Endstueck [1/c, ∞) ist gross (`eventually_ge_atTop`). `filter_upwards`
  -- verlangt die Ungleichung nur fuer ein x darin: hx : 1 / c ≤ x.
  filter_upwards [eventually_ge_atTop (1 / c)] with x hx
  -- Schritt 3: Aus 1/c ≤ x folgt 1 ≤ c·x. `div_le_iff₀ hc` macht aus 1 / c ≤ x die
  -- Aussage 1 ≤ x * c, `mul_comm` vertauscht die Faktoren.
  have h1 : 1 ≤ c * x := by
    rwa [div_le_iff₀ hc, mul_comm] at hx
  -- Schritt 4: x ist nicht negativ, denn 0 < 1/c ≤ x.
  have hx0 : 0 ≤ x := by
    have : 0 < 1 / c := by positivity
    linarith
  -- Schritt 5: Die Betraege weglassen (x ≥ 0 und x² ≥ 0). Zu zeigen: x ≤ c * x ^ 2.
  rw [abs_of_nonneg hx0, abs_of_nonneg (sq_nonneg x)]
  -- Schritt 6: x = 1·x ≤ (c·x)·x = c·x². `nlinarith` multipliziert h1 mit x ≥ 0.
  nlinarith

/-- Und x² ist bei +∞ nicht klein gegen x: Mit c = 1 muesste schliesslich
x² ≤ x gelten. Schliesslich gilt aber auch x ≥ 2, also gibt es ein x mit
beidem (Axiom (iii), und die leere Menge ist nicht gross) -- und dort ist
x² ≥ 2x > x. -/
theorem quadrat_nicht_klein_o_unendlich :
    ¬ KleinO (fun x : ℝ => x ^ 2) (fun x => x) atTop := by
  -- Schritt 1: Annahme x² = o(x), also h : ∀ c > 0, ∀ᶠ x in atTop, |x ^ 2| ≤ c * |x|.
  intro h
  -- Schritt 2: Zwei grosse Mengen zugleich. `h 1 one_pos` mit c = 1: schliesslich
  -- |x²| ≤ 1·|x|. `eventually_ge_atTop 2`: schliesslich 2 ≤ x. `.and` bildet den Schnitt
  -- (Axiom (iii)), `.exists` holt einen Punkt x daraus -- moeglich, denn eine grosse
  -- Menge ist nicht leer. Dann ist hx : |x ^ 2| ≤ 1 * |x| und h2 : 2 ≤ x.
  obtain ⟨x, hx, h2⟩ := ((h 1 one_pos).and (eventually_ge_atTop (2 : ℝ))).exists
  -- Schritt 3: x ist nicht negativ (x ≥ 2), also darf man die Betraege weglassen.
  have hx0 : (0 : ℝ) ≤ x := by linarith
  rw [abs_of_nonneg (sq_nonneg x), abs_of_nonneg hx0, one_mul] at hx   -- hx : x² ≤ x
  -- Schritt 4: Widerspruch: x² ≥ 2x > x fuer x ≥ 2. `nlinarith` multipliziert h2 mit x ≥ 0.
  nlinarith

-- Folie: "Warum ohne Teilen?"
/- Ist g schliesslich ungleich 0, heisst f = o(g) einfach f/g → 0; in Mathlib ist
   das der Satz `Asymptotics.isLittleO_iff_tendsto'`. Die Definition mit c
   braucht diese Voraussetzung nicht -- sie teilt nie. -/

-- Folie: "Rechnen mit o: Die Filteraxiome arbeiten"
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

/-! ### Mehrere Veraenderliche: partielle Ableitungen, Jacobi-Matrix, Kettenregel -/

-- Zuerst die Ableitung in einer Variablen, die Grundlage fuer alles Folgende. Die Regeln und
-- die Junk-Werte stehen im Anhang am Ende der Datei.

-- Folie: "Die Definition" (der Ableitung)
/- `HasDerivAt g a t₀` heisst: Der Fehler der Tangente, g t - g t₀ - a·(t - t₀), ist
   o(t - t₀) laengs 𝓝 t₀. Ausgeschrieben: Zu jedem c > 0 gilt schliesslich
   |Fehler| ≤ c·|t - t₀|, und laengs 𝓝 t₀ heisst schliesslich "auf einer Kugel um
   t₀" -- das ε-δ (`ableitung_eps_delta`, gleich darunter). Das δ ist das
   "schliesslich", das c das "zu jedem c" des o. In Mathlib steckt der
   Inhalt hinter zwei Umbenennungen (`HasDerivAtFilter`, die Frechet-Ableitung);
   benutzbar ist er als Satz `hasDerivAt_iff_isLittleO`, hier in unserer
   Schreibweise `KleinO`: -/
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

-- Folie: "Die Definition" (ausgeschrieben, die ε-δ-Form)
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

-- Folie: "Fermat: Am lokalen Minimum ist f' = 0"
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

-- Folie: "Vernachlaessigbar klein" (Filter: ein System grosser Mengen)
/- Was ist `𝓝 t₀` genau? Ein System "grosser" Mengen: gross ist, was eine Kugel um t₀
   enthaelt. Es ist ein Filter, weil das Ganze gross ist, Obermengen grosser Mengen
   gross sind und der Schnitt zweier grosser Mengen gross ist (die kleinere Kugel).
   Zum Beispiel ist das Intervall (-1, 1) gross fuer `𝓝 0`, denn es ist selbst eine
   Kugel:

       -2     -1      0      1      2
     ---|------o======|======o------|---  x      {x | |x| < 1}

   -/
example : Set.Ioo (-1 : ℝ) 1 ∈ 𝓝 0 := by
  -- Schritt 1: gross heisst "enthaelt eine Kugel um 0" (`Metric.mem_nhds_iff`). Ziel:
  --   ∃ ε > 0, B(0, ε) ⊆ (-1, 1)
  rw [Metric.mem_nhds_iff]
  -- Schritt 2: Wir waehlen die Kugel vom Radius 1. Ziel: B(0, 1) ⊆ (-1, 1)
  use 1, one_pos
  -- Schritt 3: Sei x in der Kugel, hx : x ∈ B(0, 1). Zu zeigen: -1 < x und x < 1.
  intro x hx
  -- Schritt 4: x ∈ B(0, 1) heisst dist x 0 < 1, also |x - 0| < 1, also |x| < 1
  -- (`sub_zero`), und das heisst -1 < x < 1 (`abs_lt`). Das ist das Ziel.
  rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_lt] at hx
  exact hx

-- Folie: "Mehrere Veraenderliche: Die Ableitung ist eine Matrix" (ℝⁿ in Lean)
/- Was ist `Fin n → ℝ`? Der Vektorraum ℝⁿ. `Fin n` sind die n Nummern
   0, 1, …, n-1 (gezaehlt wird ab 0), und `Fin n → ℝ` ist eine Funktion, die
   jeder Nummer eine reelle Zahl zuordnet: ihre Koordinate. Ein Vektor ist hier
   also keine Liste, sondern eine Funktion; `w i` ist sein i-ter Eintrag.

     w = ![4, 5, 6] : Fin 3 → ℝ       w 0 = 4, w 1 = 5, w 2 = 6

   Gerechnet wird eintragsweise: (v + w) i = v i + w i und (t • v) i = t * v i.
   `Pi.single j 1` ist der Einheitsvektor e_j: Eintrag 1 an der Stelle j, sonst
   0. Also bewegt `x + t • e_j` nur die Koordinate j, alle anderen bleiben:

     x            = (4,   5,    6)
     t • e_1      = (0,   t,    0)
     x + t • e_1  = (4, 5 + t,  6)

   Genau diese Gerade durch x in Richtung e_j ist die Schnittkurve unten. -/
example (t : ℝ) :
    (![4, 5, 6] : Fin 3 → ℝ) + t • (Pi.single 1 1 : Fin 3 → ℝ) = ![4, 5 + t, 6] := by
  ext i                                 -- eintragsweise: i = 0, 1, 2
  fin_cases i <;> simp                  -- nur Eintrag 1 aendert sich

-- Folie: "Partielle Ableitung: die Steigung einer Schnittkurve"
/- Eine partielle Ableitung ist eine Ableitung in einer Variablen: Man haelt alle
   Koordinaten bis auf eine fest und leitet die Schnittkurve ab. Fuer f(x, y) = x²y
   im Punkt (1, 2) ist sie t ↦ f(1 + t, 2) = (1 + t)²·2, mit der Steigung 4 bei
   t = 0: ∂f/∂x (1, 2) = 2xy = 4. Die Definition unten, Zeile fuer Zeile gelesen:

     {n : ℕ}                    n Variablen (oben n = 2)
     {F : Type*} [...] [...]    die Werte liegen in F: ℝ oder ℝᵐ; die eckigen
                                Klammern sagen nur, dass F eine Laenge und
                                Skalare hat, damit "Ableitung" Sinn ergibt
     (f : (Fin n → ℝ) → F)      die Funktion, ein Punkt von ℝⁿ hinein (oben x²y)
     (x : Fin n → ℝ)            der Punkt (oben (1, 2))
     (j : Fin n)                die Koordinate, die sich bewegt (oben 0)
     (a : F) : Prop             die behauptete Ableitung (oben 4)
     HasDerivAt (...) a 0       die Schnittkurve t ↦ f(x + t·e_j) hat bei t = 0
                                die Ableitung a

   Das Beispiel x²y gleich darunter gilt fuer jeden Punkt v. -/
-- Folie: "Partielle Ableitungen" (die Definition, das Beispiel x²y) und
-- "Ableitung und partielle Ableitungen" (Teil (a), mit o bewiesen; Mathlibs Fassung)
/-- Partielle Ableitung, selbst definiert: f hat in x nach der j-ten Koordinate
die partielle Ableitung a, wenn die Schnittkurve t ↦ f(x + t·e_j) in 0 die
Ableitung a hat. Alle anderen Koordinaten bleiben dabei fest. -/
def HatPartielleAbleitung {n : ℕ} {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : (Fin n → ℝ) → F) (x : Fin n → ℝ) (j : Fin n) (a : F) : Prop :=
  HasDerivAt (fun t : ℝ => f (x + t • (Pi.single j 1 : Fin n → ℝ))) a 0

/- f(x, y) = x²y nach x: y bleibt fest, die Schnittkurve ist t ↦ (x + t)²·y,
   ihre Ableitung in 0 ist 2xy -- eine Ableitung in einer Variablen. -/
example (v : Fin 2 → ℝ) :
    HatPartielleAbleitung (fun w : Fin 2 → ℝ => w 0 ^ 2 * w 1) v 0 (2 * v 0 * v 1) := by
  -- Schritt 1: Die Definition einsetzen: Die Schnittkurve t ↦ f(v + t·e₀) soll in 0 die
  -- Ableitung 2·v₀·v₁ haben.
  unfold HatPartielleAbleitung
  -- Schritt 2: Die Schnittkurve ausrechnen. Der Vektor v + t·e₀ hat die Koordinaten
  -- (v₀ + t, v₁); `funext t` nimmt ein t, `simp` liest die Koordinaten ab. Also ist
  -- die Kurve t ↦ (v₀ + t)²·v₁.
  have e : (fun t : ℝ => (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 0 ^ 2 *
      (v + t • (Pi.single 0 1 : Fin 2 → ℝ)) 1) = fun t => (v 0 + t) ^ 2 * v 1 := by
    funext t
    simp
  rw [e]
  -- Schritt 3: Ableitung in einer Variablen mit den Regeln, von innen nach aussen:
  -- t hat die Ableitung 1 (`hasDerivAt_id`), v₀ + t auch (`const_add`), das Quadrat
  -- (v₀ + t)² hat 2(v₀ + t) (`pow 2`), der Faktor v₁ bleibt stehen (`mul_const`).
  have h := (((hasDerivAt_id (0 : ℝ)).const_add (v 0)).pow 2).mul_const (v 1)
  -- Schritt 4: Die Regeln liefern die Ableitung in der Form 2·(v₀ + 0)·1·v₁; `convert` laesst
  -- die Gleichung mit der Behauptung 2·v₀·v₁ uebrig, `simp` rechnet sie nach.
  convert h using 1
  simp

-- Folie: "Ableitung und partielle Ableitungen" (Beweis von (a), mit o)
/-- Satz, Teil (a): Ist f in x differenzierbar mit der Ableitung f', so hat die
Schnittkurve t ↦ f(x + t·v) in 0 die Ableitung f' v. Der Beweis mit o, wie auf
Papier. Die Schritte in Lean stehen als Kommentare im Beweis und nennen den
Papierschritt, dem sie entsprechen:
1. Nach Definition ist  f(x + h) - f(x) - f' h = o(h)  laengs 𝓝 0.
2. Zu zeigen ist  f(x + t·v) - f(x) - t·f' v = o(t)  laengs 𝓝 0 -- das ist die
   Ableitung der Schnittkurve in 0.
3. Man setzt h = t·v ein; fuer t → 0 geht t·v → 0. Aus 1. wird
   f(x + t·v) - f(x) - f'(t·v) = o(t·v), und f'(t·v) = t·f' v, denn f' ist linear.
4. Wegen ‖t·v‖ = |t|·‖v‖ ist t·v = O(t); also ist o(t·v) ein o(t).
Das Teilen durch t am Ende des Papierbeweises entfaellt: Die Ableitung ist in
Mathlib selbst durch "Fehler = o" definiert. -/
theorem richtungsableitung {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F} {f' : E →L[ℝ] F} {x : E}
    (hf : HasFDerivAt f f' x) (v : E) :
    HasDerivAt (fun t : ℝ => f (x + t • v)) (f' v) 0 := by
  -- Schritt 1 (= 1. auf Papier): Die Voraussetzung auspacken. Mathlib sagt "f hat in x
  -- die Ableitung f'" als o-Aussage laengs 𝓝 0, mit h als Schritt von x nach x + h
  -- (`hasFDerivAt_iff_isLittleO_nhds_zero`). Danach ist
  --   hf : (fun h => f (x + h) - f x - f' h) =o[𝓝 0] fun h => h      (Fehler = o(h))
  rw [hasFDerivAt_iff_isLittleO_nhds_zero] at hf
  -- Schritt 2 (= 2.): Das Ziel genauso auspacken, nun fuer die Schnittkurve in einer
  -- Variablen (`hasDerivAt_iff_isLittleO_nhds_zero`). Das Ziel wird
  --   (fun h => f (x + (0 + h) • v) - f (x + 0 • v) - h • f' v) =o[𝓝 0] fun h => h
  -- also f(x + t·v) - f(x) - t·f' v = o(t), mit t = h; die 0 + h ist nur h.
  rw [hasDerivAt_iff_isLittleO_nhds_zero]
  -- Schritt 3 (= 3.): t·v → 0, wenn t → 0. Die Identitaet strebt gegen 0 (`tendsto_id`);
  -- mal den festen Vektor v (`smul_const`) strebt t·v gegen 0 • v, und das ist 0.
  have hv : Tendsto (fun t : ℝ => t • v) (𝓝 0) (𝓝 0) := by
    simpa using (tendsto_id (x := 𝓝 (0 : ℝ))).smul_const v
  -- Schritt 4 (= 4.): t·v = O(t). `isBigO_iff` verlangt ein c mit ‖t·v‖ ≤ c·‖t‖ fuer alle
  -- t; wir nehmen c = ‖v‖. Fuer jedes t bleibt  ‖t • v‖ ≤ ‖v‖ * ‖t‖  zu zeigen.
  -- `norm_smul` macht links ‖t‖ * ‖v‖ daraus, `mul_comm` vertauscht die Faktoren:
  -- beide Seiten sind gleich.
  have hO : (fun t : ℝ => t • v) =O[𝓝 0] (fun t : ℝ => t) :=
    Asymptotics.isBigO_iff.2 ⟨‖v‖, Eventually.of_forall fun t => by
      rw [norm_smul, mul_comm]⟩
  -- Schritt 5 (= 3. und 4.): h = t·v einsetzen. `comp_tendsto hv` setzt die Funktion
  -- t ↦ t·v in "Fehler = o(h)" ein: Der Fehler an der Stelle t·v ist o(t·v). Weil
  -- t·v = O(t) ist (hO), wird daraus o(t) (`trans_isBigO`):
  --   h : ((fun h => f (x + h) - f x - f' h) ∘ fun t => t • v) =o[𝓝 0] fun t => t
  have h := (hf.comp_tendsto hv).trans_isBigO hO
  -- Schritt 6: Angleichen. `Function.comp_def` schreibt das ∘ aus, `map_smul` liefert
  -- f' (t • v) = t • f' v, denn f' ist linear. Dann steht in h genau das Ziel aus
  -- Schritt 2.
  simpa [Function.comp_def, map_smul] using h

-- Folie: "Ableitung und partielle Ableitungen" (Teil (a): die Spalte j der Jacobi-Matrix)
/-- Satz, Teil (a), die partielle Ableitung: Ist f in x differenzierbar, so existiert
die partielle Ableitung nach x_j, und sie ist die j-te Spalte der Ableitung, also
der Jacobi-Matrix: f' e_j.
Beweis: Eine partielle Ableitung ist die Richtungsableitung in Richtung des
Einheitsvektors e_j (`Pi.single j 1`). `HatPartielleAbleitung f x j a` heisst nach
Definition: Die Schnittkurve t ↦ f(x + t·e_j) hat in 0 die Ableitung a. Das ist
`richtungsableitung` fuer v = e_j, mit a = f' e_j -- derselbe Beweis, ein v
eingesetzt. -/
theorem partiell_aus_ableitung {n : ℕ} {F : Type*} [NormedAddCommGroup F]
    [NormedSpace ℝ F] {f : (Fin n → ℝ) → F} {f' : (Fin n → ℝ) →L[ℝ] F}
    {x : Fin n → ℝ} (hf : HasFDerivAt f f' x) (j : Fin n) :
    HatPartielleAbleitung f x j (f' (Pi.single j 1)) :=
  richtungsableitung hf _                           -- mit v = e_j, von Lean erschlossen

/- Mathlib hat denselben Satz, mit dem Begriff `HasLineDerivAt` -- dort ist
   die Schnittkurve wie oben gebildet: -/
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

/-- Also ist f in 0 nicht stetig -- und damit nicht differenzierbar, obwohl
beide partiellen Ableitungen existieren. -/
theorem gegenbeispiel_nicht_differenzierbar :
    ¬ DifferentiableAt ℝ gegenbeispiel 0 := by
  -- Plan: Waere f in 0 differenzierbar, so waere f dort stetig. Laeuft man auf der Diagonalen
  -- (t, t) gegen 0, muesste f(t, t) gegen f(0) gehen. Aber f(t, t) = 1/2 fuer t ≠ 0, und
  -- f(0) = 0. Grenzwerte sind eindeutig -- Widerspruch.
  -- Schritt 0: Annahme h : DifferentiableAt ℝ gegenbeispiel 0. Ziel: False.
  intro h
  -- Schritt 1: Die Kurve t ↦ (t, t) strebt gegen 0, wenn t → 0 mit t ≠ 0.
  have hγ : Tendsto (fun t : ℝ => (![t, t] : Fin 2 → ℝ)) (𝓝[≠] 0) (𝓝 0) := by
    -- Sie ist stetig: Ein Vektor ist stetig, wenn jede Koordinate es ist
    -- (`continuous_pi`); beide Koordinaten sind t ↦ t (`continuous_id`).
    have hc : Continuous (fun t : ℝ => (![t, t] : Fin 2 → ℝ)) :=
      continuous_pi fun i => by fin_cases i <;> simpa using continuous_id
    -- Bei t = 0 ist der Wert (0, 0) -- der Nullvektor.
    have h0 : (![0, 0] : Fin 2 → ℝ) = 0 := by
      ext i
      fin_cases i <;> rfl
    -- Stetig heisst: Fuer t → 0 strebt die Kurve gegen ihren Wert bei 0; und 𝓝[≠] 0 ist
    -- kleiner als 𝓝 0 (`nhdsWithin_le_nhds`).
    simpa [h0] using (hc.tendsto 0).mono_left nhdsWithin_le_nhds
  -- Schritt 2: Differenzierbar heisst stetig (`h.continuousAt`), also strebt
  -- f(t, t) gegen f(0). Mit hγ eingesetzt (`comp`):
  --   h1 : Tendsto (f ∘ fun t => (t, t)) (𝓝[≠] 0) (𝓝 (f 0))
  have h1 := h.continuousAt.tendsto.comp hγ
  -- Schritt 3: Auf der Diagonalen ist f = 1/2 (`gegenbeispiel_diagonale`), fuer t ≠ 0. Die
  -- konstante Funktion 1/2 strebt gegen 1/2 (`tendsto_const_nhds`); `congr'` erlaubt es, sie
  -- nur "schliesslich" mit f(t, t) zu vergleichen. {t | t ≠ 0} ist gross fuer 𝓝[≠] 0
  -- (`self_mem_nhdsWithin`), und dort ist die Gleichung die Diagonalen-Formel.
  have h2 : Tendsto (fun t : ℝ => gegenbeispiel ![t, t]) (𝓝[≠] 0) (𝓝 (1 / 2)) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [self_mem_nhdsWithin] with t ht
    exact (gegenbeispiel_diagonale ht).symm
  -- Schritt 4: Ein Grenzwert ist eindeutig (`tendsto_nhds_unique`): h1 und h2 haben denselben
  -- Filter, also f(0) = 1/2.   h3 : gegenbeispiel 0 = 1 / 2
  have h3 := tendsto_nhds_unique h1 h2
  -- Schritt 5: Aber f(0) = 0 * 0 / (0 ^ 2 + 0 ^ 2) = 0 (in Lean ist a/0 = 0). Also 0 = 1/2.
  simp [gegenbeispiel] at h3

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

-- Folie: "Vorgerechnet: f(x, y) = (xy, x + y)" (zu Fuss nach der Definition,
-- dann mit den Regeln, dann die erste Spalte als partielle Ableitung)
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

/-- Dasselbe mit den Regeln: f(x, y) = (x·y, x + y) hat in v = (x, y) die
Jacobi-Matrix [[y, x], [1, 1]]. `hasFDerivAt_pi'` zerlegt die Aussage in eine
je Komponente, also je Zeile; jede Zeile setzen die Regeln zusammen
(`hasFDerivAt_apply i`: die i-te Koordinate, hier x = w 0 und y = w 1). -/
theorem jacobi_xy (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 * w 1, w 0 + w 1])
      (matrixAbb !![v 1, v 0; 1, 1]) v := by
  -- Schritt 1: Eine Abbildung nach ℝ² ist differenzierbar, wenn jede Koordinate es ist
  -- (`hasFDerivAt_pi'`): Zu zeigen ist eine Aussage je Zeile i, mit der i-ten Zeile der
  -- Matrix (`ContinuousLinearMap.proj i` waehlt die Zeile).
  rw [hasFDerivAt_pi']
  intro i
  -- Schritt 2: Die zwei Zeilen einzeln, `fin_cases i` macht die Faelle i = 0 und i = 1.
  fin_cases i
  · -- Zeile 0, x·y. Die Koordinatenfunktionen x = w 0 und y = w 1 haben die Ableitungen
    -- `proj 0` und `proj 1` (`hasFDerivAt_apply`); die Produktregel (`.mul`) setzt
    -- sie zusammen:  h : (x·y)' = x • proj 1 + y • proj 0, also  v 0 • proj 1 + v 1 • proj 0.
    have h := (hasFDerivAt_apply (𝕜 := ℝ) 0 v).mul (hasFDerivAt_apply (𝕜 := ℝ) 1 v)
    -- Die Regel liefert eine lineare Abbildung. Ist sie die Zeile (y, x) der Matrix?
    -- `convert h using 1` laesst nur die Gleichung der beiden Abbildungen uebrig
    -- (die Funktionen sind schon gleich). Ziel:
    --   (proj 0) ∘ (matrixAbb J) = v 0 • proj 1 + v 1 • proj 0
    convert h using 1
    -- Zwei lineare Abbildungen sind gleich, wenn sie an jedem Vektor k gleich sind
    -- (`ext k`). Links steht die erste Zeile von J mal k, also v 1 * k 0 + v 0 * k 1.
    ext k
    -- `simp` schreibt "Matrix mal Vektor" als Summe aus; `ring` ordnet die Summanden um.
    simp [dotProduct, Fin.sum_univ_two]
    ring
  · -- Zeile 1, x + y: die Summenregel (`.add`) gibt  proj 0 + proj 1, und das ist die Zeile
    -- (1, 1) der Matrix.
    have h := (hasFDerivAt_apply (𝕜 := ℝ) 0 v).add (hasFDerivAt_apply (𝕜 := ℝ) 1 v)
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]

/- Fuer f(x, y) = (xy, x + y) ist die partielle Ableitung nach x die erste
   Spalte (y, 1) der Jacobi-Matrix: -/
example (v : Fin 2 → ℝ) :
    HatPartielleAbleitung (fun w : Fin 2 → ℝ => ![w 0 * w 1, w 0 + w 1]) v 0 ![v 1, 1] := by
  -- Schritt 1: Teil (a) (`partiell_aus_ableitung`) auf die Matrix aus `jacobi_xy` angewendet,
  -- mit j = 0: Die partielle Ableitung nach x ist J e₀, die erste Spalte von J.
  have h := partiell_aus_ableitung (jacobi_xy v) 0
  -- Schritt 2: Das ist unsere Behauptung, wenn J e₀ = (y, 1) ist. `convert ... using 1` laesst
  -- diese Gleichung als Ziel uebrig; sie wird je Koordinate nachgerechnet (`ext i`, die
  -- Faelle 0 und 1) -- `simp` liest die erste Spalte der Matrix ab.
  convert h using 1
  ext i
  fin_cases i <;> simp

-- Folie: "Polarkoordinaten"
/-- Polarkoordinaten P(r, φ) = (r cos φ, r sin φ), mit v = (r, φ): Die Spalten
der Jacobi-Matrix sind die Ableitungen nach r und nach φ. -/
theorem jacobi_polar (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 * Real.cos (w 1), w 0 * Real.sin (w 1)])
      (matrixAbb !![Real.cos (v 1), -(v 0 * Real.sin (v 1));
                    Real.sin (v 1), v 0 * Real.cos (v 1)]) v := by
  -- Wie bei `jacobi_xy`: erst zeilenweise (`hasFDerivAt_pi'`, `fin_cases`), dann je Zeile die
  -- Regeln zusammensetzen, `convert` auf die Gleichung der linearen Abbildungen, und die
  -- Gleichung an jedem Vektor k nachrechnen (`ext k`, `simp`, `ring`).
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  · -- Zeile 0, r·cos φ. r = w 0 und φ = w 1 sind Koordinatenfunktionen (`hasFDerivAt_apply`);
    -- `.cos` ist die Kettenregel fuer cos(φ), die Produktregel `.mul` setzt r·cos φ zusammen:
    -- (r cos φ)' = cos φ dr - r sin φ dφ.
    have h := (hasFDerivAt_apply (𝕜 := ℝ) 0 v).mul (hasFDerivAt_apply (𝕜 := ℝ) 1 v).cos
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]
    ring
  · -- Zeile 1, r·sin φ: genauso mit `.sin`. (r sin φ)' = sin φ dr + r cos φ dφ.
    have h := (hasFDerivAt_apply (𝕜 := ℝ) 0 v).mul (hasFDerivAt_apply (𝕜 := ℝ) 1 v).sin
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]
    ring

/- Ihre Determinante ist r -- der Faktor, um den P kleine Flaechen streckt: -/
example (r φ : ℝ) :
    (!![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ]).det = r := by
  -- Schritt 1: Die Determinante einer 2×2-Matrix ist a·d - b·c (`det_fin_two_of`). Ziel:
  --   cos φ * (r * cos φ) - (-(r * sin φ)) * sin φ = r
  rw [Matrix.det_fin_two_of]
  -- Schritt 2: Links steht r·(cos² φ + sin² φ); mit sin² + cos² = 1 ist das r.
  -- `linear_combination` rechnet nach, dass linke minus rechte Seite gleich r mal
  -- (sin² φ + cos² φ - 1) ist.
  linear_combination r * Real.sin_sq_add_cos_sq φ

-- Folie: "Die Kettenregel: Matrizen multiplizieren"
/-- g(u, v) = u² + v², als Abbildung nach ℝ¹ geschrieben: Die Jacobi-Matrix ist
die Zeile (2u, 2v). -/
theorem jacobi_quadratsumme (u : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 ^ 2 + w 1 ^ 2])
      (matrixAbb !![2 * u 0, 2 * u 1]) u := by
  -- Es gibt nur eine Zeile (die Abbildung geht nach ℝ¹); `fin_cases i` laesst genau einen
  -- Fall uebrig.
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  -- u² + v²: Koordinatenfunktionen (`hasFDerivAt_apply`), Potenzregel (`.pow 2`: (u²)' = 2u du)
  -- und Summenregel (`.add`) -- die Zeile (2u, 2v).
  have h := ((hasFDerivAt_apply (𝕜 := ℝ) 0 u).pow 2).add
    ((hasFDerivAt_apply (𝕜 := ℝ) 1 u).pow 2)
  convert h using 1
  ext k
  simp [dotProduct, Fin.sum_univ_two]

/-- Kettenregel: Die Jacobi-Matrix von g ∘ P ist J_g(P(v)) · J_P(v).
`HasFDerivAt.comp` setzt die Ableitungen zusammen, `matrixAbb_comp` macht
daraus das Matrixprodukt. -/
theorem kettenregel_polar (v : Fin 2 → ℝ) :
    HasFDerivAt
      (fun w : Fin 2 → ℝ => ![(w 0 * Real.cos (w 1)) ^ 2 + (w 0 * Real.sin (w 1)) ^ 2])
      (matrixAbb (!![2 * (v 0 * Real.cos (v 1)), 2 * (v 0 * Real.sin (v 1))] *
        !![Real.cos (v 1), -(v 0 * Real.sin (v 1)); Real.sin (v 1), v 0 * Real.cos (v 1)]))
      v := by
  -- Schritt 1: Die Kettenregel (`HasFDerivAt.comp`): g ∘ P hat in v die Ableitung
  -- (Dg an der Stelle P(v)) ∘ (DP an der Stelle v). g ist `jacobi_quadratsumme`, an der Stelle
  -- P(v) = (r cos φ, r sin φ); P ist `jacobi_polar`.
  have h := (jacobi_quadratsumme ![v 0 * Real.cos (v 1), v 0 * Real.sin (v 1)]).comp v
    (jacobi_polar v)
  -- Schritt 2: Hintereinanderausfuehren ist Matrizen multiplizieren (`matrixAbb_comp`):
  -- aus (matrixAbb A) ∘ (matrixAbb B) wird matrixAbb (A * B).
  rw [matrixAbb_comp] at h
  -- Schritt 3: Angleichen. In h steht noch `![a, b] 0` fuer die Koordinaten und das Zeichen
  -- ∘ fuer g ∘ P; `simp` liest die Koordinaten ab und schreibt ∘ aus. Dann ist h das Ziel.
  simpa using h

/- Das Produkt ausgerechnet ist (2r, 0) -- die Ableitung von r² nach (r, φ),
   denn g(P(r, φ)) = r². -/
example (r φ : ℝ) :
    !![2 * (r * Real.cos φ), 2 * (r * Real.sin φ)] *
      !![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ] = !![2 * r, 0] := by
  -- Zwei Matrizen sind gleich, wenn jeder Eintrag es ist (`ext i j`). Es gibt eine Zeile
  -- (`fin_cases i`) und zwei Spalten (`fin_cases j`).
  ext i j
  fin_cases i
  fin_cases j
  · -- Eintrag (0,0): 2r cos φ · cos φ + 2r sin φ · sin φ = 2r. `simp` schreibt das Produkt
    -- als Summe aus; sin² + cos² = 1 liefert den Rest (wie bei der Determinante oben).
    simp [Matrix.mul_apply, Fin.sum_univ_two]
    linear_combination 2 * r * Real.sin_sq_add_cos_sq φ
  · -- Eintrag (0,1): -2r cos φ · r sin φ + 2r sin φ · r cos φ = 0, die Terme heben sich auf.
    simp [Matrix.mul_apply, Fin.sum_univ_two]
    ring

-- Folie: "Gradient und Hesse-Matrix"
/-- f(x, y) = x² + xy + y² - x - y hat die Jacobi-Matrix (2x + y - 1, x + 2y - 1),
eine Zeile: der Gradient, als Zeile geschrieben. -/
theorem gradient_beispiel (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![w 0 ^ 2 + w 0 * w 1 + w 1 ^ 2 - w 0 - w 1])
      (matrixAbb !![2 * v 0 + v 1 - 1, v 0 + 2 * v 1 - 1]) v := by
  -- Wie oben: eine Zeile (`hasFDerivAt_pi'`, `fin_cases`), die Regeln zusammensetzen,
  -- `convert` auf die Gleichung der linearen Abbildungen, an jedem Vektor k nachrechnen.
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  -- h0 und h1 sind die Koordinatenfunktionen x und y mit ihren Ableitungen.
  have h0 := hasFDerivAt_apply (𝕜 := ℝ) 0 v
  have h1 := hasFDerivAt_apply (𝕜 := ℝ) 1 v
  -- x² + x·y + y² - x - y, Term fuer Term: `.pow 2`, `.mul`, `.add`, `.sub`.
  have h := ((((h0.pow 2).add (h0.mul h1)).add (h1.pow 2)).sub h0).sub h1
  convert h using 1
  ext k
  simp [dotProduct, Fin.sum_univ_two]
  ring

/-- Der Gradient selbst hat die Jacobi-Matrix [[2, 1], [1, 2]]: die
Hesse-Matrix von f, hier konstant. -/
theorem hesse_beispiel (v : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => ![2 * w 0 + w 1 - 1, w 0 + 2 * w 1 - 1])
      (matrixAbb !![2, 1; 1, 2]) v := by
  -- Zwei Zeilen: die beiden Komponenten des Gradienten, jede wieder aus den Regeln.
  rw [hasFDerivAt_pi']
  intro i
  fin_cases i
  · -- Zeile 0, 2x + y - 1: `.const_mul 2` (Faktor 2), `.add`, `.sub_const 1` (Konstante weg).
    -- Die Zeile ist (2, 1).
    have h := (((hasFDerivAt_apply (𝕜 := ℝ) 0 v).const_mul 2).add
      (hasFDerivAt_apply (𝕜 := ℝ) 1 v)).sub_const 1
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]
  · -- Zeile 1, x + 2y - 1: dieselben Regeln. Die Zeile ist (1, 2).
    have h := ((hasFDerivAt_apply (𝕜 := ℝ) 0 v).add
      ((hasFDerivAt_apply (𝕜 := ℝ) 1 v).const_mul 2)).sub_const 1
    convert h using 1
    ext k
    simp [dotProduct, Fin.sum_univ_two]

/- Der Gradient verschwindet nur bei (1/3, 1/3) -- dem einzigen Kandidaten
   fuer ein Minimum: -/
example (x y : ℝ) :
    (2 * x + y - 1 = 0 ∧ x + 2 * y - 1 = 0) ↔ (x = 1 / 3 ∧ y = 1 / 3) := by
  constructor
  · -- (→) h1 : 2x + y - 1 = 0 und h2 : x + 2y - 1 = 0. Zwei lineare Gleichungen fuer x und y:
    -- 2·h1 - h2 gibt 3x = 1, 2·h2 - h1 gibt 3y = 1. `linarith` findet diese Kombination.
    rintro ⟨h1, h2⟩
    constructor <;> linarith
  · -- (←) x = 1/3 und y = 1/3 einsetzen (`rfl`): 2/3 + 1/3 - 1 = 0, beide Male.
    rintro ⟨rfl, rfl⟩
    constructor <;> norm_num

/-! ## 4.5 Gradientenabstieg -/

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

/-! ## Anhang: die Ableitung in einer Variablen

Nicht im Fokus der Vorlesung, zum Nachschlagen: die Regeln und die
Junk-Werte. Die Definition von `HasDerivAt` samt Filter steht am Anfang des
Kapitels ueber mehrere Veraenderliche; dieser Abschnitt wird dafuer nicht
gebraucht. -/

/-! ### Ableitung in einer Variablen -/

-- Regeln und Junk-Werte (ohne Folie)
/- Mit den fertigen Ableitungsregeln ist y² eine Zeile: -/
example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
  simpa using hasDerivAt_pow 2 x

/- Produktregel: eine Rechenregel, aus Mathlib (`HasDerivAt.mul`). -/
example (f g : ℝ → ℝ) (x f' g' : ℝ)
    (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) :
    HasDerivAt (fun y => f y * g y) (f' * g x + f x * g') x := hf.mul hg

/- Die Zahl `deriv f x` ist die Frechet-Ableitung `fderiv`, an der Stelle 1
   ausgewertet. Wo es keine Ableitung gibt, waehlt `fderiv` die 0: der Junk-Wert. -/
example (f : ℝ → ℝ) (x : ℝ) : deriv f x = fderiv ℝ f x 1 := rfl

/- `deriv` ist total: Wo f nicht differenzierbar ist, liefert es den
   Junk-Wert 0 -- so legt Mathlib `deriv` fest, daher zitiert. -/
example (f : ℝ → ℝ) (x : ℝ) (h : ¬ DifferentiableAt ℝ f x) : deriv f x = 0 :=
  deriv_zero_of_not_differentiableAt h

/- Deshalb braucht Fermat fuer `deriv` keine Differenzierbarkeit: -/
example (f : ℝ → ℝ) (a : ℝ) (h : IsLocalMin f a) : deriv f a = 0 := by
  rcases Classical.em (DifferentiableAt ℝ f a) with hd | hd
  · exact fermat h hd.hasDerivAt                  -- differenzierbar
  · exact deriv_zero_of_not_differentiableAt hd   -- sonst Junk-Wert 0

/-! ### Mathlib-Wegweiser zu Vorlesung 5

Nur hier in der Datei, nicht im Skript und nicht auf den Folien. -/
#check @Asymptotics.isLittleO_iff                      -- o(g) ausgeschrieben
#check @Asymptotics.isBigO_iff                         -- O(g) ausgeschrieben
#check @Asymptotics.IsLittleO.add                      -- Summenregel
#check @Asymptotics.IsLittleO.const_mul_left           -- mal Konstante
#check @Asymptotics.IsLittleO.trans_isBigO             -- o(g), g = O(h): o(h)
#check @Asymptotics.isLittleO_iff_tendsto'             -- mit Teilen, g schliesslich ≠ 0
#check @Asymptotics.isLittleO_pow_pow_atTop_of_lt      -- x = o(x²) bei +∞
#check @Asymptotics.isLittleO_one_iff                  -- o(1) heisst → 0
#check @hasDerivAt_iff_isLittleO                       -- Ableitung: Fehler = o(y - x)
#check @HasDerivAt.unique                              -- nur eine Steigung
#check @hasFDerivAt_iff_isLittleO_nhds_zero           -- Frechet: o(h)
#check @hasFDerivAt_pi'                                -- je Komponente
#check @HasFDerivAt.hasLineDerivAt                     -- Richtungsableitung
#check @lineDeriv                                      -- ... als Zahl, total
#check @HasFDerivAt.comp                               -- Kettenregel
#check @Matrix.toLin'_mul                              -- ... als Matrixprodukt
#check @LinearMap.toMatrix'                            -- Abbildung -> Matrix
#check @hasDerivAt_iff_hasFDerivAt                     -- Spezialfall Frechet
#check @IsLocalMin.hasDerivAt_eq_zero                  -- Fermat, HasDerivAt
#check @IsLocalMin.deriv_eq_zero                       -- Fermat, deriv
#check @exists_hasDerivAt_eq_slope                     -- Mittelwertsatz
#check @ConvexOn.slope_le_of_hasDerivAt                -- Sekante, Tangente
#check @IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt  -- mehrere NB

end Skript.Analysis
