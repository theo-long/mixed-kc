import numpy as np
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
                        # x1 and x2 are the same
                        ObserveReal(Sum(Var("x1"), Affine(Var("x2"), -1)), 0.0),
                        Let(
                            "_",
                            # x2 is equal to 1
                            ObserveReal(Var("x2"), 1.0),
                            Sum(Var("x1"), Var("x3")),
                        ),
                    ),
                ),
            ),
        ),
    )

    posterior, Z = run_kc(expr)
    assert isinstance(posterior, list), "Expected mixture posterior density"

    p_new_draw = alpha / (alpha + 2)
    p_existing = 2 / (alpha + 2)

    assert np.allclose(posterior[0].likelihood.log_likelihood, np.log(p_new_draw))
    assert np.allclose(posterior[1].likelihood.log_likelihood, np.log(p_existing))

    assert np.allclose(posterior[0].gaussian.mu, 1.0)
    assert np.allclose(posterior[1].gaussian.mu, 2.0)

    assert np.allclose(posterior[0].gaussian.cov, 1.0)
    assert np.allclose(posterior[1].gaussian.cov, 0.0)


@pytest.mark.parametrize("alpha", [0.1, 1, 4])
def test_dirichlet_process_two_draw(alpha: float):
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
                    "_",
                    # x1 and x2 are the same
                    ObserveReal(Sum(Var("x1"), Affine(Var("x2"), -1)), 0.0),
                    Let(
                        "_",
                        # x2 is equal to 1
                        ObserveReal(Var("x2"), 1.0),
                        Sum(Var("x1"), Var("x2")),
                    ),
                ),
            ),
        ),
    )

    posterior, Z = run_kc(expr)
    assert isinstance(posterior, list), "Expected mixture posterior density"

    # We observe x1 == x2 == 1 so x1 + x2 should equal exactly 2
    assert len(posterior) == 1, "Should only be a single element"
    assert np.allclose(posterior[0].likelihood.log_likelihood, 0.0)
    assert np.allclose(posterior[0].gaussian.mu, 2.0)
    assert np.allclose(posterior[0].gaussian.cov, 0.0)


if __name__ == "__main__":
    test_dirichlet_process(1.0)
