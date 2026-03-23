from kc import (
    Sum,
    Affine,
    DirichletProcess,
    Draw,
    Gaussian,
    Let,
    ObserveReal,
    Var,
    run_kc,
)


def test_dirichlet_process():
    expr = Let(
        "DP",
        DirichletProcess(1.0, Gaussian(0, 1)),
        Let(
            "x1",
            Draw(
                Var("DP"),
            ),
            Let(
                "x2",
                Draw(
                    Var("DP"),
                ),
                Let(
                    "x3",
                    Draw(
                        Var("DP"),
                    ),
                    Let(
                        "_",
                        ObserveReal(Sum(Var("x1"), Affine(Var("x2"), -1)), 0.0),
                        Var("x3"),
                    ),
                ),
            ),
        ),
    )

    posterior, Z = run_kc(expr)

test_dirichlet_process()