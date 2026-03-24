from kc import (
    Affine,
    DirichletProcess,
    Draw,
    Gaussian,
    Let,
    ObserveReal,
    Sum,
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

    from kc.inference import preprocess, kc, model_count, get_normalizing_constant, compute_spn_likelihood

    preprocess_state = preprocess(expr)
    val, state = kc(expr, preprocess_state)
    spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    Z = compute_spn_likelihood(spn, state)

    from IPython import embed

    embed()

test_dirichlet_process()