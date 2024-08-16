open Nat
example : succ zero + succ zero = succ (succ zero) := rfl
example : 1 + 1 = 2 := rfl

example {a b : Prop}: a ∧ b → b ∧ a :=
by
  intros hab
  exact And.intro hab.right hab.left

#check 911
