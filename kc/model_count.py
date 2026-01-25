from typing import TYPE_CHECKING

import dd.autoref as _bdd
import sympy

from kc.types import WeightType, get_float_value

if TYPE_CHECKING:
    from kc.real_values import DistributionWithMoments


def is_negated(u: _bdd.Function) -> bool:
    return u.node < 0


def _model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function,
    count: dict[int, WeightType],
    weights: dict[int, tuple[WeightType, WeightType]],
):
    if u in count:
        return count[u]
    if is_negated(u):
        low, high = (~u.low, ~u.high)
    else:
        low, high = (u.low, u.high)
    left_count = _model_count(bdd, low, count, weights)
    right_count = _model_count(bdd, high, count, weights)
    (wpos, wneg) = weights[u.var]
    count[u] = wpos * right_count + wneg * left_count

    return count[u]


def model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function,
    weights: dict[int, tuple[WeightType, WeightType]],
    priors: dict[sympy.Symbol, "DistributionWithMoments"],
):
    count = dict()
    count[bdd.true] = 1.0
    count[bdd.false] = 0.0
    weight = _model_count(bdd, u, count, weights)
    return get_float_value(weight, priors)
