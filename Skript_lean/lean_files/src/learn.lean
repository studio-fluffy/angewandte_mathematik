example : 1 + 1 = 2 := rfl

example {a b : Prop}: a ∧ b → b ∧ a :=
by
  intros hab
  exact And.intro hab.right hab.left
