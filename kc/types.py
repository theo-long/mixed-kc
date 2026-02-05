import itertools
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import sympy
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kc.real_values import DistributionWithMoments

epsilon = sympy.Symbol("epsilon")

LikelihoodType = int | float | sympy.Expr
PosteriorUpdateType = NDArray


class WeightType(list[tuple[LikelihoodType, PosteriorUpdateType]]):
    @classmethod
    def from_likelihood(cls, likelihood: LikelihoodType, latent_dim: int):
        return cls([(likelihood, np.zeros((1, latent_dim)))])

    def __mul__(self, other: Self):  # type: ignore
        new_weight_update = WeightType()
        for left, right in itertools.product(self, other):
            likelihood = left[0] * right[0]  # type: ignore

            # Sort by columns (from last to first for lexsort logic)
            # This sorts by Column 0, then Column 1, etc.
            # TODO: resorting each time is slow (nlogn), just do a merge (n)
            posterior_update = np.vstack((left[1], right[1]))
            indices = np.lexsort(
                [
                    posterior_update[:, i]
                    for i in reversed(range(posterior_update.shape[1]))
                ]
            )
            posterior_update = posterior_update[indices]

            # TODO: add check to see if all posterior updates are mutually compatible
            # if not, we can set the likelihood to 0 (and even add some kind of termination symbol for post. update)
            # maybe we can just remove this element from the list?

            # TODO: see if there are any linearly *dependent* observes, in which case we can reduce
            # the number of rows in the posterior update matrix

            # Check if there is existing matching posterior update
            # TODO: this loop makes it N^2! Use something faster where we sort the posteriors
            # and merge
            added = False
            for i in range(len(new_weight_update)):
                if np.allclose(new_weight_update[i][1], posterior_update):
                    new_weight_update[i] = (
                        new_weight_update[i][0] + likelihood,  # type:ignore
                        new_weight_update[i][1],
                    )
                    added = True

            if not added:
                new_weight_update.append((likelihood, posterior_update))

        return new_weight_update

    def __add__(self, other: Self):  # type: ignore
        # TODO: don't loop through the list each time and check, that's N^2
        new_weight_update = WeightType()
        for likelihood, posterior_update in itertools.chain(self, other):
            added = False
            for i in range(len(new_weight_update)):
                if np.allclose(new_weight_update[i][1], posterior_update):
                    new_weight_update[i] = (
                        new_weight_update[i][0] + likelihood,  # type:ignore
                        new_weight_update[i][1],
                    )
                    added = True

            if not added:
                new_weight_update.append((likelihood, posterior_update))
        return new_weight_update


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
