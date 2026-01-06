from kc.prob import (
    Let,
    ObserveReal,
    Var,
    Const,
    Flip,
    IfThenElse,
    Observe,
    Gaussian,
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
        Let("_", ObserveReal(0.5, Var("x")), Var("b")),
    ),
)

# Flip a coin, choose between two different Gaussians, observe the result twice
p2 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)),
        Let(
            "_",
            ObserveReal(0.5, Var("x")),
            Let("_", ObserveReal(0.5, Var("x")), Var("b")),
        ),
    ),
)

# Observe a Gaussian, then flip a coin, then choose between existing or new Gaussian
# then observe the *same* value again
p3 = Let(
    "x",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(0.5, Var("x")),
        Let(
            "b",
            Flip(0.5),
            Let(
                "y",
                IfThenElse(Var("b"), Var("x"), Gaussian(0, 2)),
                Let("_", ObserveReal(0.5, Var("y")), Var("b")),
            ),
        ),
    ),
)


def main():
    print("Hello from mixed-kc!")

    for name, program in [("p1", p1), ("p2", p2), ("p3", p3)]:
        print(f"--- {name} ---")
        prob, normalizing_constant = run_kc(program)
        print(f"Probability of b: {prob}")
        print(f"Normalizing constant: {normalizing_constant}")
        print()


if __name__ == "__main__":
    main()
