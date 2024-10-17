import Mathlib.Analysis.InnerProductSpace.Dual
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic

open Topology InnerProductSpace Set Filter Real

noncomputable section

variable {𝕜 F : Type*} [RCLike 𝕜]


--variable [NormedAddCommGroup F] [InnerProductSpace  ℝ F] [CompleteSpace F]
variables {n : ℕ} -- Dimension des Raums

-- Definiere F als endlichdimensionalen Vektorraum der Dimension n
variables {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [FiniteDimensional ℝ F]
variables [FiniteDimensional ℝ F] [CompleteSpace F]

variable {f : F →  ℝ} {f' x : F}
local notation "⟪" x ", " y "⟫" => @inner ℝ _ _ x y
scoped[Gradient] notation "∇" => gradient

-- Beweis, dass der Gradient in die Richtung des steilsten Anstiegs zeigt
theorem gradient_csu (hf : DifferentiableAt  ℝ f x) (v : F) :
⟪gradient f x, v⟫ * ⟪gradient f x, v⟫  ≤ ⟪gradient f x, gradient f x⟫ *  ⟪v, v⟫ :=
by
  exact real_inner_mul_inner_self_le (gradient f x) v

-- Definition der Richtungsableitung
def directional_deriv (f : F → ℝ) (x : F) (v : F) : ℝ :=
  (fderiv ℝ f x) v

-- Beweis, dass die Richtungsableitung gleich dem inneren Produkt des Gradienten mit dem Vektor ist
theorem directional_deriv_eq_inner_product:
  directional_deriv f x v = ⟪gradient f x, v⟫ :=
by
  rw [directional_deriv, gradient]
  rw [toDual_symm_apply]

-- Beweis, dass der Gradient in die Richtung des steilsten Anstiegs zeigt
theorem gradient_max_directional_deriv (hf : DifferentiableAt ℝ f x) (v : F) :
directional_deriv f x v ≤ ‖gradient f x‖ * ‖v‖ :=
by
  rw [directional_deriv_eq_inner_product]
  exact real_inner_le_norm (gradient f x) v



-- Beweis, dass die Richtungsableitung in Richtung des Gradienten maximal ist
theorem gradient_steepest_ascent (hf : DifferentiableAt ℝ f x) :
  ∀ v : F, ‖v‖ = 1 → directional_deriv f x v ≤ ‖gradient f x‖ :=
by
  intros v hv
  have h : ‖directional_deriv f x v‖ ≤ ‖gradient f x‖ * ‖v‖ :=
--    gradient_max_directional_deriv hf v
--  rw [hv, mul_one] at h
--  exact h
    sorry
  sorry


--def directional_deriv_std (f : F → ℝ) (x : F) (n : ℕ) : ℝ :=
--  (fderiv ℝ f x)


end
