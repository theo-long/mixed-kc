import numpy as np
import pytest

from kc import (
    DirichletProcess,
    Draw,
    Equality,
    Let,
    Observe,
    Var,
    run_kc,
)


@pytest.mark.parametrize("alpha", [i / 10 for i in range(1, 21)])
def test_dirichlet_process(alpha: float):
    expr = Let(
        "DP",
        DirichletProcess(alpha),
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
                        Observe(Equality(Var("x1"), Var("x2"))),
                        Equality(Var("x3"), Var("x1")),
                    ),
                ),
            ),
        ),
    )

    posterior, Z = run_kc(expr)
    p_existing = 2 / (alpha + 2)

    assert isinstance(posterior, float)
    assert np.allclose(posterior, p_existing)


if __name__ == "__main__":
    test_dirichlet_process(1.0)
