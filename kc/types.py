from typing import TYPE_CHECKING, Literal

import sympy
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kc.real_values import DistributionWithMoments

epsilon = sympy.Symbol("epsilon")

LikelihoodType = int | float | sympy.Expr
PosteriorUpdateType = NDArray
WeightType = PosteriorUpdateType | LikelihoodType

InequalityLiteral = Literal["<", "<=", ">", ">="]
inequality_flip_mapping: dict[InequalityLiteral, InequalityLiteral] = {
    ">": "<",
    "<": ">",
    "<=": ">=",
    ">=": "<=",
}


def get_float_value(
    v: LikelihoodType, priors: dict[sympy.Symbol, "DistributionWithMoments"]
) -> float:
    if isinstance(v, (int, float)):
        return float(v)

    v_poly = v.as_poly(epsilon)
    assert v_poly is not None, "Should get a polynomial"
    v_poly.simplify()

    coeff = v_poly.coeffs()[-1].simplify()
    for symbol, prior in priors.items():
        max_degree = sympy.degree(coeff, symbol)
        coeff = coeff.xreplace(
            {symbol**i: prior.moment(i) for i in range(1, max_degree + 1)}
        )
    return float(coeff)
