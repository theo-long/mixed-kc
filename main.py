from kc.prob import (
    Const,
    Flip,
    Gaussian,
    IfThenElse,
    Let,
    Observe,
    ObserveReal,
    Var,
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
expected_p1 = gaussian_pdf(1.0, 1.0, 1.0) / (
    gaussian_pdf(1.0, 1.0, 1.0) + gaussian_pdf(1.0, 2.0, 1.0)
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
                ObserveReal(Var("x"), 0.5),
                Let(
                    "y",
                    IfThenElse(Flip(0.5), Var("x"), Gaussian(0, 10)),
                    Let("_", ObserveReal(Var("y"), 0.5), b1_and_b2),
                ),
            ),
        ),
    ),
)
expected_p5 = None


def main():
    print("Hello from mixed-kc!")

    for name, program, expected_prob in [
        ("p1", p1, expected_p1),
        ("p2", p2, expected_p2),
        ("p3", p3, expected_p3),
        ("p4", p4, expected_p4),
        ("p5", p5, expected_p5),
    ]:
        print(f"--- {name} ---")
        prob, normalizing_constant = run_kc(program)
        print(f"Probability of b: {prob}")
        print(f"Expected probability of b: {expected_prob}")
        print(f"Normalizing constant: {normalizing_constant}")
        print()


if __name__ == "__main__":
    main()
    from IPython import embed

    embed()
