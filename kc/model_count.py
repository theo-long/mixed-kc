import dd.autoref as _bdd
from numpy.polynomial import Polynomial
import numpy as np


def is_negated(u: _bdd.Function) -> bool:
    return u.node < 0


def _model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function,
    count: dict[int, Polynomial],
    weights: dict[int, tuple[Polynomial, Polynomial]],
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
    bdd: _bdd.BDD, u: _bdd.Function, weights: dict[int, tuple[Polynomial, Polynomial]]
):
    count = dict()
    count[bdd.true] = Polynomial([1])
    count[bdd.false] = Polynomial([0])
    count_polynomial = _model_count(bdd, u, count, weights).coef
    for deg in range(len(count_polynomial)):
        val = count_polynomial[deg]
        if val > 0:
            return float(val)
    return 0.0
