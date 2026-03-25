import pytest

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


@pytest.mark.parametrize("alpha", [i / 10 for i in range(1, 21)])
def test_dirichlet_process(alpha: float):
    expr = Let(
        "DP",
        DirichletProcess(alpha, Gaussian(0, 1)),
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
                        Sum(Var("x1"), Var("x3")),
                    ),
                ),
            ),
        ),
    )

    posterior, Z = run_kc(expr)
    assert isinstance(posterior, list), "Expected mixture posterior density"

    # from kc.inference import (
    #     preprocess,
    #     kc,
    #     model_count,
    #     get_normalizing_constant,
    #     compute_spn_likelihood,
    # )

    # preprocess_state = preprocess(expr)
    # val, state = kc(expr, preprocess_state)
    # spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    # Z = compute_spn_likelihood(spn, state)

    # from IPython import embed

    # embed()


if __name__ == "__main__":
    test_dirichlet_process(1.0)
