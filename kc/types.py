import sympy
from typing import TYPE_CHECKING

epsilon = sympy.Symbol("epsilon")
WeightType = int | float | sympy.Expr


if TYPE_CHECKING:
    from kc.prob import DistributionWithMoments


def get_float_value(v: WeightType, priors: dict[sympy.Symbol, "DistributionWithMoments"]) -> float:
    if isinstance(v, (int, float)):
        return float(v)

    weight_poly = v.as_poly(epsilon)
    assert weight_poly is not None, "Should get a polynomial"

    coeff = weight_poly.coeffs()[-1]
    for symbol, prior in priors.items():
        max_degree = sympy.degree(coeff, symbol)
        coeff = coeff.xreplace(
            {symbol**i: prior.moment(i) for i in range(1, max_degree + 1)}
        )
    return float(coeff)
