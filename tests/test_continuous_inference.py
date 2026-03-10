from kc import (
    Beta,
    Const,
    Flip,
    Gaussian,
    IfThenElse,
    Let,
    Observe,
    ObserveReal,
    Sum,
    Var,
    run_kc,
)


def test_beta_posterior():
    p = Let(
        "b1",
        Beta(1, 5),
        Let(
            "b2",
            Beta(5, 1),
            Let(
                "f1",
                Flip(
                    Var("b1"),
                ),
                Let(
                    "x",
                    IfThenElse(
                        Var("f1"),
                        IfThenElse(
                            Var("b2"),
                            Flip(
                                Var("b1"),
                            ),
                            Flip(Var("b2")),
                        ),
                        IfThenElse(
                            Var("b1"),
                            Flip(
                                Var("b1"),
                            ),
                            Const(True),
                        ),
                    ),
                    Let("_", Observe(Var("x")), Var("b1")),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)


def test_gaussian_posterior():
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let("_", ObserveReal(Sum(Var("g1"), Var("g2")), 1.0), Var("g1")),
        ),
    )
    posterior, Z = run_kc(p)
