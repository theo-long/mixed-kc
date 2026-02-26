from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import sympy
from scipy.special import logsumexp
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


@dataclass
class GradedLikelihoodType:
    log_likelihood: LikelihoodType
    n_obs: int

    def __add__(self, other: "GradedLikelihoodType"):
        if self.n_obs < other.n_obs:
            return self
        elif other.n_obs < self.n_obs:
            return other
        else:
            log_likelihood = logsumexp(
                [self.log_likelihood, other.log_likelihood]
            ).item()  # type: ignore
            return GradedLikelihoodType(log_likelihood, self.n_obs)

    def __mul__(self, other: "GradedLikelihoodType"):
        log_likelihood = self.log_likelihood + other.log_likelihood  # type: ignore
        return GradedLikelihoodType(log_likelihood, self.n_obs + other.n_obs)
