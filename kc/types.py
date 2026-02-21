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


def get_degree(v: LikelihoodType):
    if isinstance(v, (int, float)):
        return 0

    v_poly = sympy.expand(v).as_poly(epsilon)
    assert v_poly is not None, "Should get a polynomial"
    return sympy.degree(v_poly, epsilon)


def get_float_value(
    v: LikelihoodType, priors: dict[sympy.Symbol, "DistributionWithMoments"]
) -> float:
    if isinstance(v, (int, float)):
        return float(v)

    v_poly = sympy.expand(v).as_poly(epsilon)
    assert v_poly is not None, "Should get a polynomial"

    coeff = v_poly.coeffs()[-1]
    for symbol, prior in priors.items():
        max_degree = sympy.degree(coeff, symbol)
        coeff = coeff.xreplace(
            {symbol**i: prior.moment(i) for i in range(1, max_degree + 1)}
        )
    return float(coeff)
