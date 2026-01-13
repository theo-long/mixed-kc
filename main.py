from math import exp
from kc.prob import (
    Const,
    Flip,
    Gaussian,
    IfThenElse,
    Let,
    ObserveReal,
    ObserveRealInequality,
    Var,
    gaussian_cdf,
    gaussian_pdf,
    run_kc,
)

# Flip a coin, choose between two different Gaussians, observe the result
p1 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            Gaussian(0, 1),
            Gaussian(0, 2),
        ),
        Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
    ),
)
# Expected probability of b is the ratio of the density of x=1.0 under N(0, 1) to the sum of the densities of x=1.0 under N(0, 1) and N(0, 2)
expected_p1 = gaussian_pdf(0.0, 1.0, 1.0) / (
    gaussian_pdf(0.0, 1.0, 1.0) + gaussian_pdf(0.0, 2.0, 1.0)
)

# Flip a coin, choose between two different Gaussians, observe the result twice
# We should see that the probability of b is the same as above
p2 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
        ),
    ),
)
expected_p2 = expected_p1

# Flip a coin, choose between two different Gaussians, observe the result twice but with *different* values
# We should get an error
p3 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", ObserveReal(Var("x"), 0.1), Var("b")),
        ),
    ),
)
expected_p3 = None

# Observe a Gaussian, then flip a coin, then choose between existing or new Gaussian
# then observe the *same* value again
p4 = Let(
    "x",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("x"), 0.5),
        Let(
            "b",
            Flip(0.5),
            Let(
                "y",
                IfThenElse(Var("b"), Var("x"), Gaussian(0, 1)),
                Let("_", ObserveReal(Var("y"), 0.5), Var("b")),
            ),
        ),
    ),
)
# We should see that the probability of b is 1.0 because once we've observed the value of x as 1.0,
# we know that also observing the other gaussian having the same value is measure 0.
# This works because for ObserveReal, for each variable in a GaussianUnion, the clause states that *this* score node is true and all the other score nodes are false.
expected_p4 = 1.0

# Observe a Gaussian union, then observe another Gaussian union that contains the original union
b1_and_b2 = IfThenElse(Var("b1"), Var("b2"), Var("b1"))
p5 = Let(
    "b1",
    Flip(0.5),
    Let(
        "b2",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(Var("b1"), Gaussian(0, 1), Gaussian(0, 2)),
            Let(
                "_",
                ObserveReal(Var("x"), 1.0),
                Let(
                    "y",
                    IfThenElse(Var("b2"), Var("x"), Gaussian(0, 10)),
                    Let("_", ObserveReal(Var("y"), 1.0), b1_and_b2),
                ),
            ),
        ),
    ),
)
# Should be the same as p1 because the first observe is same as in p1, and the second observe has the same value as the first observe so it is measure 0.
expected_p5 = expected_p1

# Same as p5, but now the second observe has a different value than the first observe so it is not measure 0.
p6 = Let(
    "b1",
    Flip(0.5),
    Let(
        "b2",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(Var("b1"), Gaussian(0, 1), Gaussian(0, 2)),
            Let(
                "_",
                ObserveReal(Var("x"), 1.0),
                Let(
                    "y",
                    IfThenElse(Var("b2"), Var("x"), Gaussian(0, 10)),
                    Let("_", ObserveReal(Var("y"), 2.0), b1_and_b2),
                ),
            ),
        ),
    ),
)
# Impossible to observe b1 and b2, because the second observe implies that y != x so b2 must be false.
expected_p6 = 0.0

# Observe that a Gaussian variable is between two values
# (Note that the expression has to evaluate to a boolean, we use Flip(0.5) as a dummy boolean)
p7 = Let(
    "x",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveRealInequality(Var("x"), "<=", 1.0),
        Let("_", ObserveRealInequality(Var("x"), ">", 0.0), Flip(0.5)),
    ),
)
expected_p7 = 0.5
expected_Z_p7 = gaussian_cdf(0.0, 1.0, 1.0) - gaussian_cdf(0.0, 1.0, 0.0)


# Observe that a Gaussian union is > 0. and < 1.0
p8 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 10)),
        Let(
            "_",
            ObserveRealInequality(Var("x"), "<=", 1.0),  # Observe that x <= 1.0
            Let(
                "_", ObserveRealInequality(Var("x"), ">", 0.0), Var("b")
            ),  # Observe that x > 0.0
        ),
    ),
)
expected_p8 = (gaussian_cdf(0.0, 1.0, 1.0) - gaussian_cdf(0.0, 1.0, 0.0)) / (
    gaussian_cdf(0.0, 1.0, 1.0)
    - gaussian_cdf(0.0, 1.0, 0.0)
    + gaussian_cdf(0.0, 10.0, 1.0)
    - gaussian_cdf(0.0, 10.0, 0.0)
)

# Observe that a Gaussian union is <= 0. and > 1.0
p9 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 10)),
        Let(
            "_",
            ObserveRealInequality(Var("x"), ">", 1.0),  # Observe that x <= 1.0
            Let(
                "_", ObserveRealInequality(Var("x"), "<=", 0.0), Var("b")
            ),  # Observe that x <= 0.0
        ),
    ),
)
expected_p9 = None  # Impossible observation

# Same as p1, except we also observe an inequality that agrees with the equality
p10 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            Gaussian(0, 1),
            Gaussian(0, 2),
        ),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", ObserveRealInequality(Var("x"), ">", 0.42), Var("b")),
        ),
    ),
)
expected_p10 = expected_p1  # Should be the same as p1 since the inequality is redundant given the equality

# Observe that both Gaussians in a Gaussian union are equal to 1.0, then observe the union again
# This is to test that observing equalities for all components of a Gaussian union works correctly
p11 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 2),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 1.0),
                    Let(
                        "_",
                        ObserveReal(Var("g1"), 1.0),
                        Let("_", ObserveReal(Var("g2"), 1.0), Var("b")),
                    ),
                ),
            ),
        ),
    ),
)
# Since we observe both g1 and g2 to be 1.0, observing x=1.0 is redundant - observing b is 50/50
expected_p11 = 0.5

# We have IID Gaussians in a nested if-then-else
# We observe that the first is > 0, that the combination of the first two is > 0, and that the whole thing is > 0
flip_1_and_flip_2 = IfThenElse(
    Var("flip_1"),
    Var("flip_2"),
    Const(False),
)
p12 = Let(
    "flip_1",
    Flip(0.5),
    Let(
        "flip_2",
        Flip(0.5),
        Let(
            "g1",
            Gaussian(0, 1),
            Let(
                "_",
                ObserveRealInequality(Var("g1"), ">", 0.0),
                Let(
                    "g1 or g2",
                    IfThenElse(
                        Var("flip_1"),
                        Var("g1"),
                        Gaussian(0, 1),
                    ),
                    Let(
                        "_",
                        ObserveRealInequality(Var("g1 or g2"), ">", 0.0),
                        Let(
                            "g1 or g2 or g3",
                            IfThenElse(
                                Var("flip_2"),
                                Var("g1 or g2"),
                                Gaussian(0, 1),
                            ),
                            Let(
                                "_",
                                ObserveRealInequality(Var("g1 or g2 or g3"), ">", 1.0),
                                flip_1_and_flip_2,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
# We know g1 or g2 is > 0, whereas g3 is 50/50 positive/negative
# Therefore 2/3 probability of flip_2
# Similarly for g2 vs g2
# So expected prob is (2/3) ** 2
expected_p12 = (2 / 3) ** 2

# Observe that one Gaussian in a Gaussian union is equal to 1.0, then observe the union is equal to 2.0
p13 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 2.0),
                    Let(
                        "_",
                        ObserveReal(Var("g2"), 1.0),
                        Var("b"),
                    ),
                ),
            ),
        ),
    ),
)
expected_p13 = 1.0

# Same as above but we also observe that the other Gaussian is equal to 2.0, which shouldn't change anything
p14 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 2.0),
                    Let(
                        "_",
                        ObserveReal(Var("g2"), 1.0),
                        Let("_", ObserveReal(Var("g1"), 2.0), Var("b")),
                    ),
                ),
            ),
        ),
    ),
)
expected_p14 = expected_p13


# We have 4 gaussians in a nested if-then-else
# We observe that pairs (1,2) and (2,3) and (3,4) and (4,1) are all equal to 1.0
def quad_gaussian(y_obs, output):
    return Let(
        "b0",
        Flip(0.5),
        Let(
            "b1",
            Flip(0.5),
            Let(
                "b2",
                Flip(0.5),
                Let(
                    "b3",
                    Flip(0.5),
                    Let(
                        "b4",
                        Flip(0.5),
                        Let(
                            "g1",
                            Gaussian(0, 1),
                            Let(
                                "g2",
                                Gaussian(0, 1),
                                Let(
                                    "g3",
                                    Gaussian(0, 1),
                                    Let(
                                        "g4",
                                        Gaussian(0, 1),
                                        Let(
                                            "x1",
                                            IfThenElse(
                                                Var("b1"),
                                                Var("g1"),
                                                Var("g2"),
                                            ),
                                            Let(
                                                "x2",
                                                IfThenElse(
                                                    Var("b2"),
                                                    Var("g2"),
                                                    Var("g3"),
                                                ),
                                                Let(
                                                    "x3",
                                                    IfThenElse(
                                                        Var("b3"),
                                                        Var("g3"),
                                                        Var("g4"),
                                                    ),
                                                    Let(
                                                        "x4",
                                                        IfThenElse(
                                                            Var("b4"),
                                                            Var("g4"),
                                                            Var("g1"),
                                                        ),
                                                        Let(
                                                            "y",
                                                            IfThenElse(
                                                                Var("b0"),
                                                                Var("x1"),
                                                                Var("x3"),
                                                            ),
                                                            Let(
                                                                "_",
                                                                ObserveReal(
                                                                    Var("x1"), 1.0
                                                                ),
                                                                Let(
                                                                    "_",
                                                                    ObserveReal(
                                                                        Var("x2"), 1.0
                                                                    ),
                                                                    Let(
                                                                        "_",
                                                                        ObserveReal(
                                                                            Var("x3"),
                                                                            1.0,
                                                                        ),
                                                                        Let(
                                                                            "_",
                                                                            ObserveReal(
                                                                                Var(
                                                                                    "x4"
                                                                                ),
                                                                                1.0,
                                                                            ),
                                                                            Let(
                                                                                "_",
                                                                                ObserveReal(
                                                                                    Var(
                                                                                        "y"
                                                                                    ),
                                                                                    y_obs,
                                                                                ),
                                                                                output,
                                                                            ),
                                                                        ),
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

# Observe that y = 1.0, x1 = (g1 or g2) = 1.0, x2 = (g2 or g3) = 1.0, x3 = (g3 or g4) = 1.0, x4 = (g4 or g1) = 1.0
# In this case, we must have g1 = 1, g3 = 1, g2 != 1, g4 != 1 or g2 = 1, g4 = 1, g1 != 1, g3 != 1
# i.e. either only (g1, g3) are 1.0 or only (g2, g4) are 1.0
# since any of the other gaussians also being 1.0 would have measure 0.
# The case (g1, g3) are 1.0 happens when b1 = True, b3 = True
# The case (g2, g4) are 1.0 happens when b1 = False, b3 = False
b1_and_b3 = IfThenElse(
    Var("b1"),
    Var("b3"),
    Const(False),
)
p15 = quad_gaussian(1.0, b1_and_b3)
expected_p15 = 0.5

# b2 and b4 can be anything in both cases
p16 = quad_gaussian(1.0, Var("b2"))
p17 = quad_gaussian(1.0, Var("b4"))
expected_p16 = 0.5
expected_p17 = 0.5


def main():
    print("Hello from mixed-kc!")
    errors = 0
    count = 0
    for name, program, expected_prob in [
        ("p1", p1, expected_p1),
        ("p2", p2, expected_p2),
        ("p3", p3, expected_p3),
        ("p4", p4, expected_p4),
        ("p5", p5, expected_p5),
        ("p6", p6, expected_p6),
        ("p7", p7, expected_p7),
        ("p8", p8, expected_p8),
        ("p9", p9, expected_p9),
        ("p10", p10, expected_p10),
        ("p11", p11, expected_p11),
        ("p12", p12, expected_p12),
        ("p13", p13, expected_p13),
        ("p14", p14, expected_p14),
        ("p15", p15, expected_p15),
        ("p16", p16, expected_p16),
        ("p17", p17, expected_p17),
    ]:
        count += 1
        print(f"--- {name} ---")
        prob, normalizing_constant = run_kc(program)
        if prob != expected_prob:
            print("### ERROR ###")
            errors += 1
        print(
            f"Probability of b: {prob: .3%}"
            if prob is not None
            else "Probability of b: None"
        )
        print(
            f"Expected probability of b: {expected_prob: .3%}"
            if expected_prob is not None
            else "Expected probability of b: None"
        )
        if name == "p7":
            print(f"Expected normalizing constant: {expected_Z_p7: .6f}")
        print(f"Normalizing constant: {normalizing_constant: .6f}")
        print()

    print(f"{count - errors} / {count} tests passed.")


if __name__ == "__main__":
    main()
